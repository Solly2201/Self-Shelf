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
            "Current_Price", "Optimized_Price", "Markdown_Percentage",
            "Action", "Days_To_Expiry", "Inventory_Units", "Days_Of_Supply",
            "Elasticity", "Predicted_Demand_Current",
            "Predicted_Demand_Optimized", "Expected_Revenue",
            "Expected_Profit", "Expected_Waste_Units",
            "Expected_Waste_Units_At_Current_Price", "Expiry_Pressure",
            "Economic_Reason",
        }
        assert expected <= set(result.recommendations.columns)

    def test_prices_respect_business_rules(self, result):
        df = result.recommendations
        assert (df["Optimized_Price"] > 0).all()
        assert (df["Optimized_Price"] <= df["Current_Price"] + 1e-9).all()
        assert (df["Expected_Waste_Units"] >= 0).all()

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
