"""Read-side service layer for the dashboard.

Runs the frozen pricing engine once and holds the rich per-product objects
(`ProductContext`, `OptimizationResult`) in memory so the web API can serve
normalized recommendation data, high-resolution price sweeps, and on-demand
scenario evaluations — all computed by the engine itself. No economic
formula lives outside `selfshelf.economics` / `selfshelf.optimizer`.

The run mirrors `pipeline.run_pipeline` step for step (same seed streams,
same per-product RNG derivation), so the recommendations served here are
identical to the CLI output. `tests/test_webdata.py` locks that parity down.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .backtest import backtest_recommendations
from .config import PricingConfig
from .demand import DemandModel, estimate_elasticities
from .economics import EconomicObjective, days_of_supply
from .evaluation import evaluate_model, split_data
from .optimizer import OptimizationResult, optimize_product, price_sweep
from .pipeline import _product_context, _recommendation_row, prepare_data

import pandas as pd

# Presentation-only risk taxonomy. Bands are thresholds on the engine's own
# inventory-pressure output (the fraction of stock expected to expire unsold
# at the current price); they do not feed back into any pricing decision.
RISK_BANDS = (
    ("clearance", 0.50),
    ("at_risk", 0.25),
    ("watch", 0.05),
    ("healthy", 0.0),
)

SWEEP_POINTS = 41


def risk_band(pressure: float) -> str:
    for name, threshold in RISK_BANDS:
        if pressure >= threshold:
            return name
    return "healthy"


def _f(value, digits: Optional[int] = None):
    """JSON-safe float: numpy scalars to Python, inf/nan to None."""
    v = float(value)
    if not math.isfinite(v):
        return None
    return round(v, digits) if digits is not None else v


@dataclass
class ProductRecord:
    """One product's engine outputs, kept for detail/sweep/scenario views."""

    sku: str
    name: str
    department: str
    result: OptimizationResult
    csv_row: Dict[str, object]


@dataclass
class PricingService:
    """Computes and caches everything the dashboard API serves."""

    data_path: str
    num_items: int = 50
    config: PricingConfig = field(default_factory=PricingConfig)

    def __post_init__(self):
        self._products: Dict[str, ProductRecord] = {}
        self._order: List[str] = []
        self._recommendations: Optional[pd.DataFrame] = None
        self._backtest: Optional[Dict[str, object]] = None
        self._model_report: Dict[str, Dict[str, float]] = {}
        self._elasticities: Dict[str, object] = {}
        self._sweep_cache: Dict[str, List[Dict[str, float]]] = {}
        self.generated_at: Optional[str] = None

    # -- computation ---------------------------------------------------------

    def compute(self) -> "PricingService":
        """Run the engine end to end, exactly as the CLI pipeline does."""
        config = self.config
        rng = np.random.default_rng(config.seed)

        df = prepare_data(self.data_path, config, rng)
        split = split_data(df, seed=config.seed)

        model = DemandModel(seed=config.seed).fit(split.train)
        self._model_report = evaluate_model(model, split)
        estimates = estimate_elasticities(split.train, config)

        items = split.test.head(self.num_items)
        baselines = model.predict(items)

        rows: List[Dict[str, object]] = []
        for pos, (_, row) in enumerate(items.iterrows()):
            estimate = estimates.get(row["DEPARTMENT"])
            elasticity = (
                estimate.elasticity if estimate else config.elasticity.default
            )
            product = _product_context(row, float(baselines[pos]), elasticity)
            product_rng = np.random.default_rng([config.seed, pos])
            result = optimize_product(product, config, product_rng)

            csv_row = _recommendation_row(row, result)
            rows.append(csv_row)
            sku = str(csv_row["SKU"])
            self._products[sku] = ProductRecord(
                sku=sku,
                name=str(row["PRODUCT_NAME"]),
                department=str(row["DEPARTMENT"]),
                result=result,
                csv_row=csv_row,
            )
            self._order.append(sku)

        self._recommendations = pd.DataFrame(rows)
        self._backtest = backtest_recommendations(
            items, self._recommendations, config
        )
        self._elasticities = {
            dept: {
                "elasticity": round(est.elasticity, 3),
                "n_observations": est.n_observations,
                "source": est.source,
            }
            for dept, est in estimates.items()
        }
        self.generated_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        return self

    @property
    def ready(self) -> bool:
        return self._recommendations is not None

    @property
    def recommendations(self) -> pd.DataFrame:
        return self._recommendations

    # -- normalized views ----------------------------------------------------

    def _summary(self, record: ProductRecord) -> Dict[str, object]:
        r = record.result
        p = r.product
        cur, opt = r.current, r.optimized
        dos = days_of_supply(p.inventory_units, cur["daily_demand"])
        pressure = record.csv_row["Inventory_Pressure"]
        return {
            "id": record.sku,
            "name": record.name,
            "department": record.department,
            "action": "markdown" if r.action == "Markdown" else "hold",
            "status": risk_band(float(pressure)),
            "pricing": {
                "current": _f(p.current_price, 2),
                "recommended": _f(r.optimized_price, 2),
                "unit_cost": _f(p.unit_cost, 2),
                "markdown_pct": _f(r.markdown_pct, 1),
                "min_allowed": _f(r.bounds[0], 2),
                "max_allowed": _f(r.bounds[1], 2),
            },
            "inventory": {
                "units": _f(p.inventory_units),
                "days_to_expiry": _f(p.days_to_expiry),
                "days_of_supply": _f(dos, 1),
                "inventory_pressure": _f(pressure, 3),
                "expiry_pressure": _f(cur["expiry_pressure"], 3),
            },
            "demand": {
                "elasticity": _f(p.elasticity, 2),
                "current_daily": _f(cur["daily_demand"], 2),
                "recommended_daily": _f(opt["daily_demand"], 2),
            },
            "economics": {
                "current": self._econ(cur),
                "recommended": self._econ(opt),
                "improvement": _f(r.value_improvement, 2),
                "gross_profit_delta": _f(
                    opt["gross_profit"] - cur["gross_profit"], 2
                ),
                "waste_avoided_units": _f(
                    cur["expected_waste_units"] - opt["expected_waste_units"],
                    2,
                ),
                "holding_cost_saved": _f(
                    cur["holding_cost"] - opt["holding_cost"], 2
                ),
            },
        }

    @staticmethod
    def _econ(breakdown: Dict[str, float]) -> Dict[str, object]:
        return {
            "price": _f(breakdown["price"], 2),
            "daily_demand": _f(breakdown["daily_demand"], 2),
            "units_sold": _f(breakdown["expected_sales_units"], 2),
            "sell_through": _f(breakdown["expected_sell_through"], 3),
            "revenue": _f(breakdown["expected_revenue"], 2),
            "gross_profit": _f(breakdown["gross_profit"], 2),
            "waste_units": _f(breakdown["expected_waste_units"], 2),
            "terminal_inventory": _f(breakdown["terminal_inventory"], 2),
            "holding_cost": _f(breakdown["holding_cost"], 2),
            "economic_value": _f(breakdown["score"], 2),
        }

    def products(self) -> List[Dict[str, object]]:
        return [self._summary(self._products[sku]) for sku in self._order]

    def product_detail(self, sku: str) -> Optional[Dict[str, object]]:
        record = self._products.get(sku)
        if record is None:
            return None
        r = record.result
        detail = self._summary(record)
        detail["reasons"] = list(r.reasons)
        detail["break_even"] = {
            "required_uplift": _f(r.break_even_uplift, 2),
            "predicted_uplift": _f(r.predicted_uplift, 2),
            "meets_margin_hurdle": bool(
                math.isfinite(r.break_even_uplift)
                and math.isfinite(r.predicted_uplift)
                and r.predicted_uplift >= r.break_even_uplift
            ),
        }
        detail["timing"] = (
            {
                "wait_days": _f(r.timing["wait_days"], 0),
                "value_now": _f(r.timing["value_now"], 2),
                "value_wait": _f(r.timing["value_wait"], 2),
                "advantage_now": _f(r.timing["advantage_now"], 2),
            }
            if r.timing
            else None
        )
        return detail

    def sweep(self, sku: str) -> Optional[List[Dict[str, float]]]:
        record = self._products.get(sku)
        if record is None:
            return None
        if sku not in self._sweep_cache:
            self._sweep_cache[sku] = price_sweep(
                record.result.product, self.config, num_points=SWEEP_POINTS
            )
        return self._sweep_cache[sku]

    def scenario(self, sku: str, price: float) -> Optional[Dict[str, object]]:
        """Engine-computed breakdown at an arbitrary candidate price."""
        record = self._products.get(sku)
        if record is None:
            return None
        lower, upper = record.result.bounds
        clamped = min(max(float(price), lower), upper)
        was_clamped = abs(clamped - float(price)) > 1e-9
        # Display prices are rounded to whole cents; the exact current price
        # may not be. Mirror the optimizer's "maintained = the exact current
        # price" semantics so the slider reproduces the hold baseline.
        exact_current = record.result.product.current_price
        if abs(clamped - exact_current) <= 0.005:
            clamped = exact_current
        objective = EconomicObjective(record.result.product, self.config)
        return {
            "price": _f(clamped, 4),
            "clamped": was_clamped,
            "breakdown": self._econ(objective.breakdown(clamped)),
            "baseline": self._econ(record.result.current),
        }

    # -- aggregates ----------------------------------------------------------

    def dashboard(self) -> Dict[str, object]:
        products = self.products()
        markdowns = [p for p in products if p["action"] == "markdown"]
        risk_counts = {name: 0 for name, _ in RISK_BANDS}
        for p in products:
            risk_counts[p["status"]] += 1
        queue = sorted(
            markdowns,
            key=lambda p: p["economics"]["improvement"],
            reverse=True,
        )
        return {
            "kpis": {
                "products": len(products),
                "markdown_recommendations": len(markdowns),
                "avg_markdown_pct": _f(
                    np.mean([p["pricing"]["markdown_pct"] for p in markdowns])
                    if markdowns else 0.0,
                    1,
                ),
                "products_at_risk": (
                    risk_counts["at_risk"] + risk_counts["clearance"]
                ),
                "expected_waste_current": _f(
                    sum(p["economics"]["current"]["waste_units"]
                        for p in products), 0
                ),
                "expected_waste_recommended": _f(
                    sum(p["economics"]["recommended"]["waste_units"]
                        for p in products), 0
                ),
                "value_improvement": _f(
                    sum(p["economics"]["improvement"] for p in products), 2
                ),
                "sell_through_current": _f(
                    self._weighted_sell_through("current"), 3
                ),
                "sell_through_recommended": _f(
                    self._weighted_sell_through("recommended"), 3
                ),
            },
            "queue": queue,
            "risk_counts": risk_counts,
            "backtest": self._backtest_summary(),
            "meta": self.meta(),
        }

    def _weighted_sell_through(self, which: str) -> float:
        total_units = sold = 0.0
        for record in self._products.values():
            b = (record.result.current if which == "current"
                 else record.result.optimized)
            total_units += record.result.product.inventory_units
            sold += b["expected_sales_units"]
        return sold / total_units if total_units > 0 else 0.0

    def _backtest_summary(self) -> Optional[Dict[str, object]]:
        if not self._backtest:
            return None
        out = {}
        for strategy in ("hold", "recommended"):
            s = self._backtest[strategy]
            out[strategy] = {k: _f(v, 2) for k, v in s.items()}
        return out

    def meta(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "seed": self.config.seed,
            "num_items": self.num_items,
            "synthetic": True,
            "data_source": "Synthetic retail simulation",
            "model_report": {
                split: {k: _f(v, 3) for k, v in metrics.items()}
                for split, metrics in self._model_report.items()
            },
            "elasticities": self._elasticities,
        }

    def analytics(self) -> Dict[str, object]:
        products = self.products()
        markdowns = [p for p in products if p["action"] == "markdown"]

        by_department: Dict[str, Dict[str, float]] = {}
        for p in products:
            dept = by_department.setdefault(p["department"], {
                "products": 0, "markdowns": 0,
                "waste_current": 0.0, "waste_recommended": 0.0,
                "value_improvement": 0.0,
            })
            dept["products"] += 1
            dept["markdowns"] += 1 if p["action"] == "markdown" else 0
            dept["waste_current"] += p["economics"]["current"]["waste_units"]
            dept["waste_recommended"] += (
                p["economics"]["recommended"]["waste_units"]
            )
            dept["value_improvement"] += p["economics"]["improvement"]
        for dept in by_department.values():
            for key in ("waste_current", "waste_recommended",
                        "value_improvement"):
                dept[key] = _f(dept[key], 2)

        depth_bins = {"0-10%": 0, "10-20%": 0, "20-30%": 0, "30-40%": 0}
        for p in markdowns:
            pct = p["pricing"]["markdown_pct"]
            if pct < 10:
                depth_bins["0-10%"] += 1
            elif pct < 20:
                depth_bins["10-20%"] += 1
            elif pct < 30:
                depth_bins["20-30%"] += 1
            else:
                depth_bins["30-40%"] += 1

        supply_bins = {"<3d": 0, "3-7d": 0, "7-14d": 0, "14d+": 0, "no demand": 0}
        for p in products:
            dos = p["inventory"]["days_of_supply"]
            if dos is None:
                supply_bins["no demand"] += 1
            elif dos < 3:
                supply_bins["<3d"] += 1
            elif dos < 7:
                supply_bins["3-7d"] += 1
            elif dos < 14:
                supply_bins["7-14d"] += 1
            else:
                supply_bins["14d+"] += 1

        price_changes = [
            {
                "id": p["id"],
                "name": p["name"],
                "current": p["pricing"]["current"],
                "recommended": p["pricing"]["recommended"],
                "markdown_pct": p["pricing"]["markdown_pct"],
            }
            for p in products
        ]

        return {
            "markdowns": {
                "count": len(markdowns),
                "avg_depth_pct": _f(
                    np.mean([p["pricing"]["markdown_pct"] for p in markdowns])
                    if markdowns else 0.0, 1),
                "depth_distribution": depth_bins,
            },
            "risk_counts": {
                name: sum(1 for p in products if p["status"] == name)
                for name, _ in RISK_BANDS
            },
            "days_of_supply_distribution": supply_bins,
            "waste_by_department": by_department,
            "price_changes": price_changes,
            "totals": {
                "waste_current": _f(
                    sum(p["economics"]["current"]["waste_units"]
                        for p in products), 0),
                "waste_recommended": _f(
                    sum(p["economics"]["recommended"]["waste_units"]
                        for p in products), 0),
                "value_improvement": _f(
                    sum(p["economics"]["improvement"] for p in products), 2),
            },
            "meta": self.meta(),
        }
