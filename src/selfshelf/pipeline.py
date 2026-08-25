"""End-to-end pricing pipeline orchestration."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PricingConfig
from .demand import (
    DemandModel,
    apply_historical_discounts,
    estimate_elasticities,
    simulate_demand,
)
from .economics import ProductContext, days_of_supply
from .evaluation import SplitData, evaluate_model, split_data
from .features import clean_data, engineer_features, load_data
from .optimizer import OptimizationResult, optimize_product, price_sweep


@dataclass
class PipelineResult:
    recommendations: pd.DataFrame
    sweeps: Optional[pd.DataFrame]
    model_report: Dict[str, Dict[str, float]]
    elasticities: Dict[str, object]
    config_summary: Dict[str, object]


def prepare_data(
    data_path: str, config: PricingConfig, rng: np.random.Generator
) -> pd.DataFrame:
    df = load_data(data_path)
    if config.sample_size and len(df) > config.sample_size:
        df = df.sample(config.sample_size, random_state=config.seed)
        df = df.reset_index(drop=True)
    df = clean_data(df)
    df = engineer_features(df, config, rng)
    df = apply_historical_discounts(df, config, rng)
    df = simulate_demand(df, config, rng)
    return df


def _product_context(
    row: pd.Series, baseline_demand: float, elasticity: float
) -> ProductContext:
    return ProductContext(
        current_price=float(row["PRICE_CURRENT"]),
        retail_price=float(row["PRICE_RETAIL"]),
        unit_cost=float(row["COST"]),
        inventory_units=float(row["INVENTORY_UNITS"]),
        days_to_expiry=float(row["DAYS_TO_EXPIRY"]),
        baseline_daily_demand=baseline_demand,
        elasticity=elasticity,
    )


def _recommendation_row(
    row: pd.Series, result: OptimizationResult
) -> Dict[str, object]:
    p = result.product
    cur, opt = result.current, result.optimized
    dos = days_of_supply(p.inventory_units, cur["daily_demand"])
    return {
        "SKU": row.get("SKU", ""),
        "Product_Name": row["PRODUCT_NAME"],
        "Department": row["DEPARTMENT"],
        "Unit_Cost": round(p.unit_cost, 2),
        "Current_Price": round(p.current_price, 2),
        "Optimized_Price": round(result.optimized_price, 2),
        "Markdown_Percentage": round(result.markdown_pct, 1),
        "Action": result.action,
        "Days_To_Expiry": int(p.days_to_expiry),
        "Inventory_Units": int(p.inventory_units),
        "Days_Of_Supply": (
            round(dos, 1) if np.isfinite(dos) else "inf"
        ),
        "Elasticity": round(p.elasticity, 2),
        "Predicted_Demand_Current": round(cur["daily_demand"], 2),
        "Predicted_Demand_Optimized": round(opt["daily_demand"], 2),
        "Expected_Revenue": round(opt["expected_revenue"], 2),
        "Expected_Profit": round(
            opt["expected_revenue"] - opt["cogs"], 2
        ),
        "Expected_Waste_Units": round(opt["expected_waste_units"], 2),
        "Expected_Waste_Units_At_Current_Price": round(
            cur["expected_waste_units"], 2
        ),
        "Expiry_Pressure": round(cur["expiry_pressure"], 3),
        "Economic_Reason": "; ".join(result.reasons),
    }


def run_pipeline(
    data_path: str,
    config: PricingConfig,
    num_items: int,
    collect_sweeps: bool = False,
    progress: bool = False,
) -> PipelineResult:
    """Load data, train and evaluate the demand model, estimate
    elasticities, and optimize prices for ``num_items`` test-set products.
    """
    rng = np.random.default_rng(config.seed)

    df = prepare_data(data_path, config, rng)
    split: SplitData = split_data(df, seed=config.seed)

    model = DemandModel(seed=config.seed).fit(split.train)
    model_report = evaluate_model(model, split)

    # Elasticities come from training data only — never from the simulator's
    # config directly, and never from validation/test rows.
    estimates = estimate_elasticities(split.train, config)

    items = split.test.head(num_items)
    baselines = model.predict(items)

    recommendations: List[Dict[str, object]] = []
    sweep_rows: List[Dict[str, object]] = []

    for pos, (_, row) in enumerate(items.iterrows()):
        estimate = estimates.get(row["DEPARTMENT"])
        elasticity = (
            estimate.elasticity if estimate else config.elasticity.default
        )
        product = _product_context(row, float(baselines[pos]), elasticity)

        # Independent, deterministic RNG stream per product: results do not
        # depend on how many other products were optimized before this one.
        product_rng = np.random.default_rng([config.seed, pos])
        result = optimize_product(product, config, product_rng)
        recommendations.append(_recommendation_row(row, result))

        if collect_sweeps:
            for point in price_sweep(product, config):
                point = {"SKU": row.get("SKU", ""), **point}
                sweep_rows.append(point)

        if progress and (pos + 1) % 10 == 0:
            print(f"Optimized {pos + 1}/{len(items)} items...")

    return PipelineResult(
        recommendations=pd.DataFrame(recommendations),
        sweeps=pd.DataFrame(sweep_rows) if collect_sweeps else None,
        model_report=model_report,
        elasticities={
            dept: {
                "elasticity": round(est.elasticity, 3),
                "n_observations": est.n_observations,
                "source": est.source,
            }
            for dept, est in estimates.items()
        },
        config_summary=config.describe(),
    )
