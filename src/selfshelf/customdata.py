"""Custom retail data import: schema, column mapping, validation,
persistence, and historically-estimated demand parameters.

This module lets Self-Shelf run on a user's own inventory and transaction
history instead of the synthetic simulator. It deliberately contains **no
economic formulas**: elasticity estimation is delegated to the frozen
``demand.estimate_elasticities`` (the same log-log regression the synthetic
pipeline uses), and price adjustment of historical demand observations uses
the frozen ``economics.price_effect``. What lives here is data plumbing —
mapping arbitrary CSV headers onto Self-Shelf's schema, rejecting bad rows
with human-readable reasons, being explicit about how much information the
data actually contains, and persisting the imported dataset.

Two files are supported:

    products:     one row per product (current state of the shelf)
    transactions: one row per product per day of sales history

Transactions are required for optimization — without observed sales there
is nothing to estimate demand from, and Self-Shelf refuses to invent it.
Missing dates in the history are treated as days with zero sales, so the
file should contain daily aggregates.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PricingConfig
from .demand import estimate_elasticities
from .economics import price_effect

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

PRODUCT_REQUIRED = [
    "product_id", "product_name", "current_price", "cost",
    "inventory", "days_to_expiry",
]
PRODUCT_OPTIONAL = ["category", "retail_price", "promotion"]

TRANSACTION_REQUIRED = ["date", "product_id", "price", "units_sold"]
TRANSACTION_OPTIONAL = ["promotion"]

FIELD_SYNONYMS: Dict[str, List[str]] = {
    "product_id": [
        "product_id", "sku", "item_id", "product", "id", "upc",
        "item_code", "product_code", "item", "article", "article_id",
    ],
    "product_name": [
        "product_name", "name", "item_name", "description", "product_desc",
        "title", "product_description",
    ],
    "category": [
        "category", "department", "dept", "product_category", "class",
        "group", "product_group", "section",
    ],
    "current_price": [
        "current_price", "price", "sell_price", "selling_price",
        "unit_price", "price_current", "shelf_price",
    ],
    "cost": [
        "cost", "unit_cost", "cogs", "procurement_cost", "cost_price",
        "purchase_price", "supplier_cost",
    ],
    "inventory": [
        "inventory", "stock", "on_hand", "quantity", "qty", "units",
        "inventory_units", "stock_on_hand", "soh", "stock_units",
        "stock_available",
    ],
    "days_to_expiry": [
        "days_to_expiry", "expiry_days", "shelf_life", "days_left",
        "expiration_days", "days_until_expiry", "remaining_days",
        "days_to_expiration",
    ],
    "retail_price": [
        "retail_price", "list_price", "msrp", "full_price", "price_retail",
        "rrp", "regular_price",
    ],
    "promotion": [
        "promotion", "promo", "on_promo", "is_promo", "discount_flag",
        "on_promotion",
    ],
    "date": [
        "date", "day", "transaction_date", "sale_date", "sales_date",
        "period", "datetime", "timestamp", "business_date",
    ],
    "price": [
        "price", "sell_price", "unit_price", "sale_price",
        "transaction_price", "selling_price", "avg_price",
    ],
    "units_sold": [
        "units_sold", "units", "quantity", "qty", "sales", "volume",
        "qty_sold", "sales_units", "demand", "sold",
    ],
}

# Fields whose synonyms overlap (e.g. "price"); resolved in listed order so
# the more specific field wins the exact match.
_PRODUCT_FIELDS = PRODUCT_REQUIRED + PRODUCT_OPTIONAL
_TRANSACTION_FIELDS = TRANSACTION_REQUIRED + TRANSACTION_OPTIONAL


def _normalize(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def fields_for(kind: str) -> Tuple[List[str], List[str]]:
    if kind == "products":
        return PRODUCT_REQUIRED, PRODUCT_OPTIONAL
    if kind == "transactions":
        return TRANSACTION_REQUIRED, TRANSACTION_OPTIONAL
    raise ValueError(f"unknown data kind: {kind}")


def suggest_mapping(columns: Sequence[str], kind: str) -> Dict[str, Dict]:
    """Suggest a CSV-column -> Self-Shelf-field mapping.

    Exact synonym matches win over substring matches; each CSV column is
    used at most once. Returns {field: {column, confidence}} where
    confidence is "exact" or "fuzzy".
    """
    required, optional = fields_for(kind)
    normalized = {_normalize(c): c for c in columns}
    taken = set()
    suggestions: Dict[str, Dict] = {}

    for confidence in ("exact", "fuzzy"):
        for fld in required + optional:
            if fld in suggestions:
                continue
            for syn in FIELD_SYNONYMS[fld]:
                match = None
                if confidence == "exact":
                    match = normalized.get(syn)
                else:
                    for norm, original in normalized.items():
                        if syn in norm and original not in taken:
                            match = original
                            break
                if match and match not in taken:
                    suggestions[fld] = {
                        "column": match, "confidence": confidence,
                    }
                    taken.add(match)
                    break
    return suggestions


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    kind: str
    valid: pd.DataFrame
    rejected: pd.DataFrame          # original rows + a "reject_reason" column
    issues: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)  # file-level, fatal

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "rows_total": int(len(self.valid) + len(self.rejected)),
            "rows_valid": int(len(self.valid)),
            "rows_rejected": int(len(self.rejected)),
            "issues": self.issues,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _apply_mapping(
    df: pd.DataFrame, mapping: Dict[str, str], kind: str
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    required, optional = fields_for(kind)
    errors = []
    for fld in required:
        col = mapping.get(fld)
        if not col:
            errors.append(f"required field '{fld}' is not mapped")
        elif col not in df.columns:
            errors.append(
                f"mapped column '{col}' for '{fld}' does not exist in the file"
            )
    used = [c for c in mapping.values() if c]
    dupes = {c for c in used if used.count(c) > 1}
    if dupes:
        errors.append(
            "each CSV column may map to only one field; duplicated: "
            + ", ".join(sorted(dupes))
        )
    if errors:
        return None, errors
    out = pd.DataFrame(index=df.index)
    for fld in required + optional:
        col = mapping.get(fld)
        if col and col in df.columns:
            out[fld] = df[col]
    return out, []


def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.replace(r"[$,€£\s]", "", regex=True)
            .replace({"": None, "nan": None, "None": None})
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _reject(reasons: pd.Series, mask: pd.Series, message: str):
    new = mask & reasons.isna()
    reasons.loc[new] = message
    return reasons


def _collect_issues(
    reasons: pd.Series, issues: List[Dict[str, object]]
) -> None:
    counts = reasons.dropna().value_counts()
    for message, count in counts.items():
        issues.append({
            "count": int(count),
            "message": f"{int(count)} row(s): {message}",
        })


def validate_products(
    raw: pd.DataFrame, mapping: Dict[str, str]
) -> ValidationResult:
    mapped, errors = _apply_mapping(raw, mapping, "products")
    if errors:
        return ValidationResult(
            "products", pd.DataFrame(), pd.DataFrame(), errors=errors
        )

    df = mapped.copy()
    reasons = pd.Series(pd.NA, index=df.index, dtype=object)

    df["product_id"] = df["product_id"].astype(str).str.strip()
    ids_missing = df["product_id"].isin(["", "nan", "None", "<NA>"])
    reasons = _reject(reasons, ids_missing, "missing product_id")

    df["product_name"] = df["product_name"].astype(str).str.strip()
    name_missing = df["product_name"].isin(["", "nan", "None", "<NA>"])
    reasons = _reject(reasons, name_missing, "missing product_name")

    for fld, rule, message in (
        ("current_price", lambda s: s.isna() | (s <= 0),
         "current_price is missing, non-numeric, or not positive"),
        ("cost", lambda s: s.isna() | (s < 0),
         "cost is missing, non-numeric, or negative"),
        ("inventory", lambda s: s.isna() | (s < 0),
         "inventory is missing, non-numeric, or negative"),
        ("days_to_expiry", lambda s: s.isna() | (s < 0),
         "days_to_expiry is missing, non-numeric, or negative"),
    ):
        df[fld] = _numeric(df[fld])
        reasons = _reject(reasons, rule(df[fld]), message)

    dupes = df["product_id"].duplicated(keep="first") & ~ids_missing
    reasons = _reject(
        reasons, dupes, "duplicate product_id (first occurrence kept)"
    )

    warnings: List[str] = []
    issues: List[Dict[str, object]] = []
    _collect_issues(reasons, issues)

    valid = df[reasons.isna()].copy()
    rejected = raw.loc[reasons.notna()].copy()
    if len(rejected):
        rejected["reject_reason"] = reasons.dropna()

    # Optional fields with defaults.
    if "category" in valid.columns:
        valid["category"] = (
            valid["category"].astype(str).str.strip()
            .replace({"": "General", "nan": "General", "None": "General"})
        )
    else:
        valid["category"] = "General"

    if "retail_price" in valid.columns:
        valid["retail_price"] = _numeric(valid["retail_price"])
        bad_retail = (
            valid["retail_price"].isna()
            | (valid["retail_price"] <= 0)
        )
        if bad_retail.any():
            warnings.append(
                f"{int(bad_retail.sum())} row(s) have a missing/invalid "
                f"retail_price; the current price is used as the list price"
            )
        valid.loc[bad_retail, "retail_price"] = valid.loc[
            bad_retail, "current_price"
        ]
    else:
        valid["retail_price"] = valid["current_price"]
    # The engine requires current <= retail.
    below = valid["retail_price"] < valid["current_price"]
    if below.any():
        warnings.append(
            f"{int(below.sum())} row(s) have retail_price below "
            f"current_price; retail_price was raised to the current price"
        )
        valid.loc[below, "retail_price"] = valid.loc[below, "current_price"]

    if "promotion" in valid.columns:
        valid["promotion"] = (
            _numeric(valid["promotion"]).fillna(0).ne(0).astype(int)
        )
    else:
        valid["promotion"] = 0

    margin_neg = valid["cost"] > valid["current_price"]
    if margin_neg.any():
        warnings.append(
            f"{int(margin_neg.sum())} product(s) currently sell below cost "
            f"(cost > current_price) — imported as-is"
        )

    valid["inventory"] = valid["inventory"].round().astype(int)
    valid = valid[
        ["product_id", "product_name", "category", "current_price",
         "retail_price", "cost", "inventory", "days_to_expiry", "promotion"]
    ].reset_index(drop=True)

    return ValidationResult(
        "products", valid, rejected, issues=issues, warnings=warnings
    )


def validate_transactions(
    raw: pd.DataFrame,
    mapping: Dict[str, str],
    known_ids: Optional[Sequence[str]] = None,
) -> ValidationResult:
    mapped, errors = _apply_mapping(raw, mapping, "transactions")
    if errors:
        return ValidationResult(
            "transactions", pd.DataFrame(), pd.DataFrame(), errors=errors
        )

    df = mapped.copy()
    reasons = pd.Series(pd.NA, index=df.index, dtype=object)

    df["product_id"] = df["product_id"].astype(str).str.strip()
    reasons = _reject(
        reasons,
        df["product_id"].isin(["", "nan", "None", "<NA>"]),
        "missing product_id",
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    reasons = _reject(
        reasons, df["date"].isna(), "date is missing or unparseable"
    )

    df["price"] = _numeric(df["price"])
    reasons = _reject(
        reasons, df["price"].isna() | (df["price"] <= 0),
        "price is missing, non-numeric, or not positive",
    )

    df["units_sold"] = _numeric(df["units_sold"])
    reasons = _reject(
        reasons, df["units_sold"].isna() | (df["units_sold"] < 0),
        "units_sold is missing, non-numeric, or negative",
    )

    if known_ids is not None:
        known = set(map(str, known_ids))
        unknown = ~df["product_id"].isin(known)
        reasons = _reject(
            reasons, unknown,
            "product_id does not appear in the products file",
        )

    issues: List[Dict[str, object]] = []
    _collect_issues(reasons, issues)

    valid = df[reasons.isna()].copy()
    rejected = raw.loc[reasons.notna()].copy()
    if len(rejected):
        rejected["reject_reason"] = reasons.dropna()

    if "promotion" in valid.columns:
        valid["promotion"] = (
            _numeric(valid["promotion"]).fillna(0).ne(0).astype(int)
        )
    else:
        valid["promotion"] = 0

    valid = valid[
        ["date", "product_id", "price", "units_sold", "promotion"]
    ].sort_values("date").reset_index(drop=True)

    return ValidationResult("transactions", valid, rejected, issues=issues)


# ---------------------------------------------------------------------------
# Dataset + persistence
# ---------------------------------------------------------------------------

@dataclass
class CustomDataset:
    products: pd.DataFrame
    transactions: pd.DataFrame
    meta: Dict[str, object] = field(default_factory=dict)

    def quality_summary(self) -> Dict[str, object]:
        txn = self.transactions
        by_product = txn.groupby("product_id").size() if len(txn) else (
            pd.Series(dtype=int)
        )
        return {
            "products": int(len(self.products)),
            "categories": int(self.products["category"].nunique()),
            "transactions": int(len(txn)),
            "date_range": (
                {
                    "start": str(txn["date"].min().date()),
                    "end": str(txn["date"].max().date()),
                }
                if len(txn) else None
            ),
            "products_with_history": int(
                self.products["product_id"].isin(by_product.index).sum()
            ),
            "products_without_history": int(
                (~self.products["product_id"].isin(by_product.index)).sum()
            ),
            "median_observations_per_product": (
                float(by_product.median()) if len(by_product) else 0.0
            ),
        }


def save_dataset(dataset: CustomDataset, directory: str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    dataset.products.to_csv(path / "products.csv", index=False)
    txn = dataset.transactions.copy()
    txn["date"] = txn["date"].dt.strftime("%Y-%m-%d")
    txn.to_csv(path / "transactions.csv", index=False)
    with open(path / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(dataset.meta, fh, indent=2)


def load_dataset(directory: str) -> Optional[CustomDataset]:
    path = Path(directory)
    products_file = path / "products.csv"
    transactions_file = path / "transactions.csv"
    if not products_file.exists() or not transactions_file.exists():
        return None
    products = pd.read_csv(products_file, dtype={"product_id": str})
    transactions = pd.read_csv(
        transactions_file, dtype={"product_id": str},
        parse_dates=["date"],
    )
    meta_file = path / "meta.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as fh:
            meta = json.load(fh)
    return CustomDataset(products, transactions, meta)


# ---------------------------------------------------------------------------
# Demand parameters from history
# ---------------------------------------------------------------------------

@dataclass
class ProductDemandParams:
    product_id: str
    baseline_daily_demand: float
    baseline_source: str        # "history" | "category" | "global"
    baseline_n_days: int
    elasticity: float
    elasticity_source: str      # "estimated" | "fallback"
    elasticity_n_obs: int
    elasticity_reason: str


def estimate_demand_params(
    dataset: CustomDataset,
    config: PricingConfig,
    min_observations: int = 30,
    min_price_variation: float = 0.02,
    baseline_window_days: int = 28,
) -> Tuple[Dict[str, ProductDemandParams], Dict[str, Dict[str, object]]]:
    """Per-product demand parameters from the imported history.

    Elasticities per category come from the frozen
    ``demand.estimate_elasticities`` log-log regression, fed daily
    product-level aggregates with PRICE_RATIO relative to each product's
    list price. All history is usable (it all precedes "now", so nothing
    leaks from the future into the estimate); a stability check re-runs
    the same frozen estimator on only the earliest 80% of observations
    and flags categories whose estimate moves materially.

    Baseline daily demand per product is the recent observed sales rate
    translated to the *current* price with the frozen
    ``economics.price_effect`` — days absent from the history count as
    zero-sale days. Products without history fall back to their category's
    median rate (then the global median), and are labelled as such.
    """
    products = dataset.products
    txn = dataset.transactions
    fallback = config.elasticity.default

    # Daily product-level aggregation.
    if len(txn):
        daily = (
            txn.groupby(["product_id", txn["date"].dt.normalize()])
            .agg(
                units_sold=("units_sold", "sum"),
                price=("price", "mean"),
                promotion=("promotion", "max"),
            )
            .reset_index()
        )
    else:
        daily = pd.DataFrame(
            columns=["product_id", "date", "units_sold", "price", "promotion"]
        )

    ref = products.set_index("product_id")
    daily = daily[daily["product_id"].isin(ref.index)]
    categories: Dict[str, Dict[str, object]] = {}

    # -- category elasticities (frozen estimator on mapped frame) ----------
    if len(daily):
        frame = pd.DataFrame({
            "DEPARTMENT": daily["product_id"].map(ref["category"]),
            "DEMAND": daily["units_sold"].astype(float),
            "PRICE_RATIO": (
                daily["price"]
                / daily["product_id"].map(ref["retail_price"]).astype(float)
            ),
            "PROMOTION": daily["promotion"].astype(int),
            "date": daily["date"],
        })
        frame = frame[np.isfinite(frame["PRICE_RATIO"])]
        estimates = estimate_elasticities(frame, config, min_observations)

        frame_sorted = frame.sort_values("date")
        cutoff = max(1, int(len(frame_sorted) * 0.8))
        early = estimate_elasticities(
            frame_sorted.head(cutoff), config, min_observations
        )

        for cat, group in frame.groupby("DEPARTMENT"):
            est = estimates.get(cat)
            variation = float(group["PRICE_RATIO"].std() or 0.0)
            n_obs = int(est.n_observations) if est else 0
            if est is None or est.source == "default":
                if n_obs < min_observations:
                    reason = (
                        f"only {n_obs} usable observations "
                        f"(needs {min_observations})"
                    )
                else:
                    reason = "regression did not yield a credible elasticity"
                categories[cat] = {
                    "elasticity": fallback, "source": "fallback",
                    "n_observations": n_obs, "price_variation": variation,
                    "reason": reason, "stability": None,
                }
                continue
            if variation < min_price_variation:
                categories[cat] = {
                    "elasticity": fallback, "source": "fallback",
                    "n_observations": n_obs, "price_variation": variation,
                    "reason": (
                        "insufficient price variation in the history "
                        f"({variation:.3f} < {min_price_variation})"
                    ),
                    "stability": None,
                }
                continue
            early_est = early.get(cat)
            stability = (
                abs(est.elasticity - early_est.elasticity)
                if early_est and early_est.source == "estimated" else None
            )
            categories[cat] = {
                "elasticity": float(est.elasticity), "source": "estimated",
                "n_observations": n_obs, "price_variation": variation,
                "reason": "estimated from observed price variation",
                "stability": stability,
            }
    # Categories that exist in products but have no transactions at all.
    for cat in products["category"].unique():
        if cat not in categories:
            categories[cat] = {
                "elasticity": fallback, "source": "fallback",
                "n_observations": 0, "price_variation": 0.0,
                "reason": "no transaction history for this category",
                "stability": None,
            }

    # -- per-product baseline demand ----------------------------------------
    params: Dict[str, ProductDemandParams] = {}
    rates: Dict[str, Tuple[float, int]] = {}

    for pid, group in daily.groupby("product_id"):
        row = ref.loc[pid]
        cat = row["category"]
        elasticity = float(categories[cat]["elasticity"])
        end = group["date"].max()
        window = group[
            group["date"] >= end - pd.Timedelta(days=baseline_window_days - 1)
        ]
        span_days = max(
            1, int((end - window["date"].min()).days) + 1
        )
        current_price = float(row["current_price"])
        adjusted = [
            float(u) / price_effect(float(p), current_price, elasticity)
            if p > 0 else float(u)
            for u, p in zip(window["units_sold"], window["price"])
        ]
        rate = sum(adjusted) / span_days
        rates[pid] = (rate, int(len(window)))

    category_median: Dict[str, float] = {}
    for cat in products["category"].unique():
        cat_ids = products.loc[
            products["category"] == cat, "product_id"
        ]
        values = [rates[p][0] for p in cat_ids if p in rates]
        if values:
            category_median[cat] = float(np.median(values))
    global_median = (
        float(np.median([r for r, _ in rates.values()])) if rates else 0.0
    )

    for _, row in products.iterrows():
        pid = row["product_id"]
        cat = row["category"]
        cat_info = categories[cat]
        if pid in rates:
            rate, n_days = rates[pid]
            baseline_source = "history"
        elif cat in category_median:
            rate, n_days = category_median[cat], 0
            baseline_source = "category"
        else:
            rate, n_days = global_median, 0
            baseline_source = "global"
        params[pid] = ProductDemandParams(
            product_id=pid,
            baseline_daily_demand=max(0.0, float(rate)),
            baseline_source=baseline_source,
            baseline_n_days=n_days,
            elasticity=float(cat_info["elasticity"]),
            elasticity_source=str(cat_info["source"]),
            elasticity_n_obs=int(cat_info["n_observations"]),
            elasticity_reason=str(cat_info["reason"]),
        )

    return params, categories


def build_meta(
    products_result: ValidationResult,
    transactions_result: ValidationResult,
    source_files: Dict[str, str],
    mappings: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    return {
        "imported_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source_files": source_files,
        "mappings": mappings,
        "validation": {
            "products": products_result.summary(),
            "transactions": transactions_result.summary(),
        },
    }
