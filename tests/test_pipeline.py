"""End-to-end pipeline test on a small sample of the real dataset."""

from pathlib import Path

import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.pipeline import run_pipeline

DATA = (
    Path(__file__).resolve().parent.parent
    / "data" / "walmart_large_sample_data_with_categories.csv"
)


def small_config() -> PricingConfig:
    cfg = PricingConfig()
    cfg.sample_size = 600
    return cfg


@pytest.fixture(scope="module")
def result():
    return run_pipeline(
        str(DATA), small_config(), num_items=12, collect_sweeps=True
    )


class TestPipeline:
    def test_produces_requested_recommendations(self, result):
        assert len(result.recommendations) == 12

    def test_output_columns(self, result):
        expected = {
            "SKU", "Product_Name", "Department", "Unit_Cost",
            "Current_Price", "Recommended_Price", "Markdown_Percentage",
            "Action", "Days_To_Expiry", "Inventory_Units", "Days_Of_Supply",
            "Elasticity", "Expiry_Pressure", "Inventory_Pressure",
            "Predicted_Demand_Current", "Predicted_Demand_Optimized",
            "Expected_Units_Sold_Current", "Expected_Units_Sold_Optimized",
            "Sell_Through_Current", "Sell_Through_Optimized",
            "Gross_Revenue_Current", "Gross_Revenue_Optimized",
            "Gross_Profit_Current", "Gross_Profit_Optimized",
            "Expected_Waste_Current", "Expected_Waste_Optimized",
            "Holding_Cost_Current", "Holding_Cost_Optimized",
            "Terminal_Inventory_Current", "Terminal_Inventory_Optimized",
            "Economic_Value_Current", "Economic_Value_Optimized",
            "Economic_Value_Improvement", "Break_Even_Unit_Uplift",
            "Predicted_Unit_Uplift", "Economic_Reason",
        }
        assert expected <= set(result.recommendations.columns)

    def test_prices_respect_business_rules(self, result):
        df = result.recommendations
        assert (df["Recommended_Price"] > 0).all()
        assert (df["Recommended_Price"] <= df["Current_Price"] + 1e-9).all()
        assert (df["Expected_Waste_Optimized"] >= 0).all()
        assert (df["Sell_Through_Optimized"] <= 1.0 + 1e-9).all()
        assert (df["Terminal_Inventory_Optimized"] >= -1e-9).all()
        # A recommended markdown must never lower the expected economic
        # value versus keeping the current price.
        assert (df["Economic_Value_Improvement"] >= -1e-9).all()
        # Sell-through can never fall when the price is reduced.
        marked = df[df["Action"] == "Markdown"]
        assert (
            marked["Sell_Through_Optimized"]
            >= marked["Sell_Through_Current"] - 1e-9
        ).all()

    def test_model_report_has_leak_free_splits(self, result):
        assert set(result.model_report) == {"validation", "test"}
        for metrics in result.model_report.values():
            assert set(metrics) == {"mae", "rmse", "r2"}

    def test_elasticities_are_negative(self, result):
        for info in result.elasticities.values():
            assert info["elasticity"] < 0

    def test_sweeps_generated_per_product(self, result):
        assert result.sweeps is not None
        assert set(result.sweeps["SKU"]) <= set(
            result.recommendations["SKU"]
        )

    def test_reproducible_with_same_seed(self, result):
        again = run_pipeline(
            str(DATA), small_config(), num_items=12, collect_sweeps=False
        )
        pd.testing.assert_frame_equal(
            result.recommendations, again.recommendations
        )
