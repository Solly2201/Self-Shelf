"""Service-layer tests: the dashboard adapter must serve exactly the
frozen engine's numbers — never its own."""

import math

import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.pipeline import run_pipeline
from selfshelf.webdata import PricingService, risk_band

DATA = "data/walmart_large_sample_data_with_categories.csv"
N_ITEMS = 8


@pytest.fixture(scope="module")
def service():
    return PricingService(data_path=DATA, num_items=N_ITEMS).compute()


@pytest.fixture(scope="module")
def pipeline_result():
    return run_pipeline(DATA, PricingConfig(), num_items=N_ITEMS)


class TestPipelineParity:
    def test_recommendations_match_the_frozen_pipeline_exactly(
        self, service, pipeline_result
    ):
        ours = service.recommendations.reset_index(drop=True)
        theirs = pipeline_result.recommendations.reset_index(drop=True)
        pd.testing.assert_frame_equal(ours, theirs)

    def test_summary_prices_match_csv_rows(self, service):
        by_sku = {
            str(row["SKU"]): row
            for _, row in service.recommendations.iterrows()
        }
        for product in service.products():
            row = by_sku[product["id"]]
            assert product["pricing"]["current"] == row["Current_Price"]
            assert product["pricing"]["recommended"] == (
                row["Recommended_Price"]
            )
            assert product["economics"]["improvement"] == (
                row["Economic_Value_Improvement"]
            )
            assert product["action"] == (
                "markdown" if row["Action"] == "Markdown" else "hold"
            )


class TestRiskBands:
    def test_thresholds(self):
        assert risk_band(0.0) == "healthy"
        assert risk_band(0.049) == "healthy"
        assert risk_band(0.05) == "watch"
        assert risk_band(0.25) == "at_risk"
        assert risk_band(0.5) == "clearance"
        assert risk_band(1.0) == "clearance"

    def test_every_product_gets_a_band(self, service):
        bands = {"healthy", "watch", "at_risk", "clearance"}
        for product in service.products():
            assert product["status"] in bands


class TestDetailAndSweep:
    def test_unknown_sku_returns_none(self, service):
        assert service.product_detail("nope") is None
        assert service.sweep("nope") is None
        assert service.scenario("nope", 1.0) is None

    def test_detail_contains_reasons_and_break_even(self, service):
        sku = service.products()[0]["id"]
        detail = service.product_detail(sku)
        assert detail["reasons"]
        assert "required_uplift" in detail["break_even"]
        assert isinstance(detail["break_even"]["meets_margin_hurdle"], bool)

    def test_sweep_covers_the_feasible_range_and_demand_is_monotone(
        self, service
    ):
        for product in service.products():
            sweep = service.sweep(product["id"])
            prices = [pt["price"] for pt in sweep]
            demands = [pt["daily_demand"] for pt in sweep]
            assert prices == sorted(prices)
            assert prices[0] == pytest.approx(
                product["pricing"]["min_allowed"], abs=0.01
            )
            assert prices[-1] == pytest.approx(
                product["pricing"]["max_allowed"], abs=0.01
            )
            # Constant-elasticity model: price up -> demand weakly down.
            assert all(
                a >= b - 1e-9 for a, b in zip(demands, demands[1:])
            )

    def test_scenario_matches_engine_breakdown_at_current_price(
        self, service
    ):
        for product in service.products()[:3]:
            current = product["pricing"]["current"]
            scenario = service.scenario(product["id"], current)
            assert scenario["clamped"] is False
            assert scenario["breakdown"] == product["economics"]["current"]

    def test_scenario_clamps_out_of_bounds_prices(self, service):
        product = service.products()[0]
        low = service.scenario(product["id"], 0.01)
        assert low["clamped"] is True
        assert low["price"] == pytest.approx(
            product["pricing"]["min_allowed"], abs=0.01
        )
        high = service.scenario(product["id"], 10_000.0)
        assert high["clamped"] is True
        assert high["price"] == pytest.approx(
            product["pricing"]["max_allowed"], abs=0.01
        )


class TestAggregates:
    def test_dashboard_kpis_are_sums_over_products(self, service):
        products = service.products()
        dashboard = service.dashboard()
        kpis = dashboard["kpis"]
        assert kpis["products"] == len(products)
        assert kpis["markdown_recommendations"] == sum(
            1 for p in products if p["action"] == "markdown"
        )
        assert kpis["expected_waste_current"] == pytest.approx(
            sum(p["economics"]["current"]["waste_units"] for p in products),
            abs=0.5,
        )
        assert kpis["value_improvement"] == pytest.approx(
            sum(p["economics"]["improvement"] for p in products), abs=0.05
        )
        assert sum(dashboard["risk_counts"].values()) == len(products)

    def test_queue_is_markdowns_sorted_by_improvement(self, service):
        queue = service.dashboard()["queue"]
        improvements = [p["economics"]["improvement"] for p in queue]
        assert improvements == sorted(improvements, reverse=True)
        assert all(p["action"] == "markdown" for p in queue)

    def test_backtest_summary_reconciles_units(self, service):
        backtest = service.dashboard()["backtest"]
        starting_hold = (
            backtest["hold"]["units_sold"]
            + backtest["hold"]["terminal_inventory"]
        )
        starting_rec = (
            backtest["recommended"]["units_sold"]
            + backtest["recommended"]["terminal_inventory"]
        )
        assert starting_hold == pytest.approx(starting_rec, abs=0.5)

    def test_analytics_totals_and_distributions_are_consistent(
        self, service
    ):
        products = service.products()
        analytics = service.analytics()
        markdowns = [p for p in products if p["action"] == "markdown"]
        assert analytics["markdowns"]["count"] == len(markdowns)
        assert sum(
            analytics["markdowns"]["depth_distribution"].values()
        ) == len(markdowns)
        assert sum(
            analytics["days_of_supply_distribution"].values()
        ) == len(products)
        dept_products = sum(
            d["products"] for d in analytics["waste_by_department"].values()
        )
        assert dept_products == len(products)
        assert len(analytics["price_changes"]) == len(products)

    def test_meta_is_labeled_synthetic(self, service):
        meta = service.meta()
        assert meta["synthetic"] is True
        assert "simulation" in meta["data_source"].lower()
        assert meta["generated_at"] is not None
