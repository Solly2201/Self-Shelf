"""End-to-end pricing pipeline orchestration."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import backtest_recommendations
from .config import PricingConfig
from .demand import (
    DemandModel,
    apply_historical_discounts,
    estimate_elasticities,
    simulate_demand,
)
from .economics import ProductContext, days_of_supply, inventory_pressure
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
    backtest: Optional[Dict[str, object]] = None


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
        "Recommended_Price": round(result.optimized_price, 2),
        "Markdown_Percentage": round(result.markdown_pct, 1),
        "Action": result.action,
        "Days_To_Expiry": int(p.days_to_expiry),
        "Inventory_Units": int(p.inventory_units),
        "Days_Of_Supply": (
            round(dos, 1) if np.isfinite(dos) else "inf"
        ),
        "Elasticity": round(p.elasticity, 2),
        "Expiry_Pressure": round(cur["expiry_pressure"], 3),
        "Inventory_Pressure": round(
            inventory_pressure(
                p.inventory_units, cur["daily_demand"], p.days_to_expiry
            ),
            3,
        ),
        "Predicted_Demand_Current": round(cur["daily_demand"], 2),
        "Predicted_Demand_Optimized": round(opt["daily_demand"], 2),
        "Expected_Units_Sold_Current": round(cur["expected_sales_units"], 2),
        "Expected_Units_Sold_Optimized": round(opt["expected_sales_units"], 2),
        "Sell_Through_Current": round(cur["expected_sell_through"], 3),
        "Sell_Through_Optimized": round(opt["expected_sell_through"], 3),
        "Gross_Revenue_Current": round(cur["expected_revenue"], 2),
        "Gross_Revenue_Optimized": round(opt["expected_revenue"], 2),
        "Gross_Profit_Current": round(cur["gross_profit"], 2),
        "Gross_Profit_Optimized": round(opt["gross_profit"], 2),
        "Expected_Waste_Current": round(cur["expected_waste_units"], 2),
        "Expected_Waste_Optimized": round(opt["expected_waste_units"], 2),
        "Holding_Cost_Current": round(cur["holding_cost"], 2),
        "Holding_Cost_Optimized": round(opt["holding_cost"], 2),
        "Terminal_Inventory_Current": round(cur["terminal_inventory"], 2),
        "Terminal_Inventory_Optimized": round(opt["terminal_inventory"], 2),
        "Economic_Value_Current": round(cur["score"], 2),
        "Economic_Value_Optimized": round(opt["score"], 2),
        "Economic_Value_Improvement": round(result.value_improvement, 2),
        "Break_Even_Unit_Uplift": (
            round(result.break_even_uplift, 2)
            if np.isfinite(result.break_even_uplift) else "inf"
        ),
        "Predicted_Unit_Uplift": round(result.predicted_uplift, 2),
        "Economic_Reason": "; ".join(result.reasons),
    }


def run_pipeline(
    data_path: str,
    config: PricingConfig,
    num_items: int,
    collect_sweeps: bool = False,
    collect_backtest: bool = False,
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

    recommendations_df = pd.DataFrame(recommendations)
    backtest = (
        backtest_recommendations(items, recommendations_df, config)
        if collect_backtest and len(recommendations_df)
        else None
    )

    return PipelineResult(
        recommendations=recommendations_df,
        sweeps=pd.DataFrame(sweep_rows) if collect_sweeps else None,
        backtest=backtest,
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
