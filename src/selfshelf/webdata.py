"""Read-side service layer for the dashboard.

Runs the frozen pricing engine once per data source and holds the rich
per-product objects (`ProductContext`, `OptimizationResult`) in memory so
the web API can serve normalized recommendation data, high-resolution price
sweeps, multi-period price paths, and on-demand scenario evaluations — all
computed by the engine itself. No economic formula lives outside
`selfshelf.economics` / `selfshelf.optimizer` / `selfshelf.pathopt`.

Two concrete services share one presentation layer:

- ``PricingService``: the synthetic demo source. The run mirrors
  ``pipeline.run_pipeline`` step for step (same seed streams, same
  per-product RNG derivation), so the recommendations served here are
  identical to the CLI output. `tests/test_webdata.py` locks that parity
  down.
- ``CustomPricingService``: a user-imported dataset (see
  ``selfshelf.customdata``). Elasticities and baseline demand come from
  the imported transaction history with transparent fallbacks; there is no
  synthetic backtest because no ground-truth simulator exists for real
  data.
"""

import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .backtest import backtest_recommendations
from .config import PricingConfig
from .customdata import CustomDataset, estimate_demand_params, load_dataset
from .demand import DemandModel, estimate_elasticities
from .economics import (
    EconomicObjective,
    ProductContext,
    days_of_supply,
    evaluate_price_path,
    inventory_pressure,
)
from .evaluation import evaluate_model, split_data
from .optimizer import OptimizationResult, optimize_product, price_sweep
from .pathbacktest import backtest_price_paths
from .pathopt import (
    PathResult,
    optimize_path,
    path_price_floor,
    path_trajectory,
    validate_daily_prices,
)
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
    # Custom-data provenance: where elasticity/baseline demand came from.
    provenance: Optional[Dict[str, object]] = None


class BasePricingService:
    """Shared presentation layer over a computed set of ProductRecords.

    Subclasses implement ``compute()`` (filling ``_products``/``_order``)
    and ``meta()``; every view below only reads engine outputs.
    """

    source = "unknown"

    def _init_state(self):
        self._products: Dict[str, ProductRecord] = {}
        self._order: List[str] = []
        self._recommendations: Optional[pd.DataFrame] = None
        self._backtest: Optional[Dict[str, object]] = None
        self._sweep_cache: Dict[str, List[Dict[str, float]]] = {}
        self._path_cache: Dict[str, PathResult] = {}
        self._path_backtest: Optional[Dict[str, object]] = None
        self.generated_at: Optional[str] = None

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
        summary = {
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
        if record.provenance:
            summary["provenance"] = record.provenance
        return summary

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

    @staticmethod
    def _path_econ(evaluation: Dict[str, float]) -> Dict[str, object]:
        """Normalize an ``evaluate_price_path`` result for the API."""
        return {
            "revenue": _f(evaluation["expected_revenue"], 2),
            "gross_profit": _f(evaluation["gross_profit"], 2),
            "waste_units": _f(evaluation["expected_waste_units"], 2),
            "waste_cost": _f(evaluation["expected_waste_cost"], 2),
            "holding_cost": _f(evaluation["holding_cost"], 2),
            "markdown_cost": _f(evaluation["markdown_cost"], 2),
            "terminal_inventory": _f(evaluation["terminal_inventory"], 2),
            "economic_value": _f(evaluation["score"], 2),
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

    # -- multi-period paths --------------------------------------------------

    def _path_result(self, sku: str) -> Optional[PathResult]:
        record = self._products.get(sku)
        if record is None:
            return None
        if sku not in self._path_cache:
            self._path_cache[sku] = optimize_path(
                record.result.product,
                self.config,
                single_price=record.result.optimized_price,
            )
        return self._path_cache[sku]

    def path(self, sku: str) -> Optional[Dict[str, object]]:
        """Recommended multi-period schedule with hold/immediate baselines."""
        result = self._path_result(sku)
        if result is None:
            return None
        record = self._products[sku]
        product = record.result.product
        horizon = result.horizon_days
        cur = product.current_price

        trajectory = [
            {k: _f(v, 3) if k != "day" else int(v) for k, v in row.items()}
            for row in path_trajectory(product, self.config, result.schedule)
        ]

        strategies = {
            "recommended": {
                "daily_prices": [_f(p, 2) for p in result.daily_prices],
                "econ": self._path_econ(result.evaluation),
            },
            "hold": {
                "daily_prices": [_f(cur, 2)] * horizon,
                "econ": self._path_econ(result.hold),
            },
        }
        if result.single is not None:
            strategies["immediate"] = {
                "daily_prices": [_f(result.single_price, 2)] * horizon,
                "price": _f(result.single_price, 2),
                "econ": self._path_econ(result.single),
            }

        return {
            "id": record.sku,
            "name": record.name,
            "action": result.action,
            "horizon_days": horizon,
            "daily_prices": [_f(p, 2) for p in result.daily_prices],
            "schedule": [
                {"days": int(round(days)), "price": _f(price, 2)}
                for days, price in result.schedule
            ],
            "strategies": strategies,
            "improvement_vs_hold": _f(result.improvement_vs_hold, 2),
            "improvement_vs_single": (
                _f(result.improvement_vs_single, 2)
                if result.improvement_vs_single is not None else None
            ),
            "trajectory": trajectory,
            "reasons": list(result.reasons),
            "constraints": {
                "floor": _f(path_price_floor(product, self.config), 2),
                "ceiling": _f(cur, 2),
                "max_daily_drop_pct": _f(
                    100 * self.config.constraints.max_daily_price_drop, 0
                ),
            },
        }

    def path_scenario(
        self, sku: str, daily_prices: List[float]
    ) -> Optional[Dict[str, object]]:
        """Engine valuation of a user-proposed daily price path."""
        record = self._products.get(sku)
        if record is None:
            return None
        product = record.result.product
        prices, errors = validate_daily_prices(
            product, self.config, daily_prices
        )
        if errors:
            return {"errors": errors}
        from .pathopt import daily_to_schedule

        schedule = daily_to_schedule(prices)
        evaluation = evaluate_price_path(product, self.config, schedule)
        recommended = self._path_result(sku)
        return {
            "errors": [],
            "daily_prices": [_f(p, 2) for p in prices],
            "econ": self._path_econ(evaluation),
            "trajectory": [
                {
                    k: _f(v, 3) if k != "day" else int(v)
                    for k, v in row.items()
                }
                for row in path_trajectory(product, self.config, schedule)
            ],
            "compare": {
                "hold": self._path_econ(recommended.hold),
                "recommended": self._path_econ(recommended.evaluation),
            },
        }

    def path_backtest(self) -> Optional[Dict[str, object]]:
        """Synthetic-only; overridden by the synthetic service."""
        return None

    # -- exports -------------------------------------------------------------

    def export_recommendations_csv(self) -> str:
        rows = []
        for summary in self.products():
            cur = summary["economics"]["current"]
            rec = summary["economics"]["recommended"]
            row = {
                "Product_ID": summary["id"],
                "Product_Name": summary["name"],
                "Department": summary["department"],
                "Action": summary["action"],
                "Current_Price": summary["pricing"]["current"],
                "Recommended_Price": summary["pricing"]["recommended"],
                "Markdown_Pct": summary["pricing"]["markdown_pct"],
                "Inventory_Units": summary["inventory"]["units"],
                "Days_To_Expiry": summary["inventory"]["days_to_expiry"],
                "Elasticity": summary["demand"]["elasticity"],
                "Economic_Value_Current": cur["economic_value"],
                "Economic_Value_Recommended": rec["economic_value"],
                "Economic_Value_Improvement":
                    summary["economics"]["improvement"],
                "Expected_Waste_Current": cur["waste_units"],
                "Expected_Waste_Recommended": rec["waste_units"],
                "Sell_Through_Current": cur["sell_through"],
                "Sell_Through_Recommended": rec["sell_through"],
            }
            if "provenance" in summary:
                row["Elasticity_Source"] = summary["provenance"].get(
                    "elasticity_source"
                )
                row["Baseline_Demand_Source"] = summary["provenance"].get(
                    "baseline_source"
                )
            rows.append(row)
        return self._to_csv(pd.DataFrame(rows))

    def export_paths_csv(self) -> str:
        rows = []
        for sku in self._order:
            payload = self.path(sku)
            if payload is None:
                continue
            for day_row in payload["trajectory"]:
                rows.append({
                    "Product_ID": payload["id"],
                    "Product_Name": payload["name"],
                    "Day": day_row["day"],
                    "Recommended_Price": day_row["price"],
                    "Expected_Daily_Demand": day_row["daily_demand"],
                    "Expected_Sales_Units":
                        day_row["expected_sales_units"],
                    "Start_Inventory": day_row["start_inventory"],
                    "End_Inventory": day_row["end_inventory"],
                    "Expected_Revenue": day_row["expected_revenue"],
                    "Projected_Waste_Units":
                        day_row.get("projected_waste_units"),
                    "Path_Economic_Value":
                        payload["strategies"]["recommended"]["econ"][
                            "economic_value"
                        ],
                })
        return self._to_csv(pd.DataFrame(rows))

    @staticmethod
    def _to_csv(frame: pd.DataFrame) -> str:
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        return buffer.getvalue()

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

    def meta(self) -> Dict[str, object]:  # pragma: no cover - abstract
        raise NotImplementedError

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
            "path_backtest": self.path_backtest(),
            "meta": self.meta(),
        }


@dataclass
class PricingService(BasePricingService):
    """Synthetic demo source: computes and caches everything the dashboard
    API serves, mirroring the CLI pipeline exactly."""

    data_path: str
    num_items: int = 50
    config: PricingConfig = field(default_factory=PricingConfig)

    source = "synthetic"

    def __post_init__(self):
        self._init_state()
        self._items: Optional[pd.DataFrame] = None
        self._model_report: Dict[str, Dict[str, float]] = {}
        self._elasticities: Dict[str, object] = {}

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

        self._items = items
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

    def path_backtest(self) -> Optional[Dict[str, object]]:
        """Hold vs immediate markdown vs optimized path, replayed under the
        synthetic simulator with common random numbers. Cached."""
        if not self.ready or self._items is None:
            return None
        if self._path_backtest is None:
            immediate = [
                float(self._recommendations.iloc[pos]["Recommended_Price"])
                for pos in range(len(self._order))
            ]
            daily_paths = []
            for sku in self._order:
                result = self._path_result(sku)
                daily_paths.append([float(p) for p in result.daily_prices])
            self._path_backtest = backtest_price_paths(
                self._items, immediate, daily_paths, self.config
            )
        return self._path_backtest

    def meta(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "seed": self.config.seed,
            "num_items": self.num_items,
            "source": "synthetic",
            "synthetic": True,
            "data_source": "Synthetic retail simulation",
            "model_report": {
                split: {k: _f(v, 3) for k, v in metrics.items()}
                for split, metrics in self._model_report.items()
            },
            "elasticities": self._elasticities,
        }


class CustomPricingService(BasePricingService):
    """User-imported data source.

    Same frozen engine, same presentation layer — but demand parameters
    come from the imported history (with labelled fallbacks) and there is
    deliberately no synthetic backtest: no ground-truth simulator exists
    for real data, so Self-Shelf does not fabricate one.
    """

    source = "custom"

    def __init__(
        self,
        directory: str,
        config: Optional[PricingConfig] = None,
        dataset: Optional[CustomDataset] = None,
    ):
        self.directory = directory
        self.config = config or PricingConfig()
        self._dataset = dataset
        self._categories: Dict[str, Dict[str, object]] = {}
        self._quality: Dict[str, object] = {}
        self._imported_at: Optional[str] = None
        self._init_state()

    def compute(self) -> "CustomPricingService":
        dataset = self._dataset or load_dataset(self.directory)
        if dataset is None:
            raise ValueError(
                "no imported dataset found — upload data on the Data page"
            )
        if not len(dataset.products):
            raise ValueError("the imported dataset contains no products")
        if not len(dataset.transactions):
            raise ValueError(
                "the imported dataset has no transaction history; "
                "Self-Shelf cannot estimate demand without observed sales"
            )
        self._dataset = dataset
        config = self.config

        params, categories = estimate_demand_params(dataset, config)
        self._categories = categories
        self._quality = dataset.quality_summary()
        self._imported_at = dataset.meta.get("imported_at")

        rows: List[Dict[str, object]] = []
        for pos, (_, row) in enumerate(dataset.products.iterrows()):
            pid = str(row["product_id"])
            prm = params[pid]
            product = ProductContext(
                current_price=float(row["current_price"]),
                retail_price=float(row["retail_price"]),
                unit_cost=float(row["cost"]),
                inventory_units=float(row["inventory"]),
                days_to_expiry=float(row["days_to_expiry"]),
                baseline_daily_demand=prm.baseline_daily_demand,
                elasticity=prm.elasticity,
            )
            product_rng = np.random.default_rng([config.seed, pos])
            result = optimize_product(product, config, product_rng)
            pseudo_row = pd.Series({
                "SKU": pid,
                "PRODUCT_NAME": str(row["product_name"]),
                "DEPARTMENT": str(row["category"]),
            })
            csv_row = _recommendation_row(pseudo_row, result)
            rows.append(csv_row)
            self._products[pid] = ProductRecord(
                sku=pid,
                name=str(row["product_name"]),
                department=str(row["category"]),
                result=result,
                csv_row=csv_row,
                provenance={
                    "elasticity_source": prm.elasticity_source,
                    "elasticity_n_obs": prm.elasticity_n_obs,
                    "elasticity_reason": prm.elasticity_reason,
                    "baseline_source": prm.baseline_source,
                    "baseline_n_days": prm.baseline_n_days,
                },
            )
            self._order.append(pid)

        self._recommendations = pd.DataFrame(rows)
        self._backtest = None  # no ground-truth simulator for real data
        self.generated_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        return self

    def meta(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "seed": self.config.seed,
            "num_items": len(self._order),
            "source": "custom",
            "synthetic": False,
            "data_source": "Custom data import",
            "imported_at": self._imported_at,
            "quality": self._quality,
            "model_report": {},
            "elasticities": {
                cat: {
                    "elasticity": _f(info["elasticity"], 3),
                    "n_observations": info["n_observations"],
                    "source": (
                        "estimated" if info["source"] == "estimated"
                        else "default"
                    ),
                    "reason": info["reason"],
                    "price_variation": _f(info["price_variation"], 4),
                }
                for cat, info in self._categories.items()
            },
        }
