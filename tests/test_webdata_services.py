"""Service-layer tests for the completion features: multi-period paths,
path scenarios, the three-strategy path backtest, CSV exports, and the
custom-data service — all of which must serve engine numbers only."""

import io

import numpy as np
import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.customdata import (
    CustomDataset,
    validate_products,
    validate_transactions,
)
from selfshelf.economics import evaluate_price_path
from selfshelf.pathopt import daily_to_schedule
from selfshelf.webdata import CustomPricingService, PricingService

DATA = "data/walmart_large_sample_data_with_categories.csv"
N_ITEMS = 8

PRODUCT_MAPPING = {
    "product_id": "product_id", "product_name": "product_name",
    "category": "category", "current_price": "current_price",
    "retail_price": "retail_price", "cost": "cost",
    "inventory": "inventory", "days_to_expiry": "days_to_expiry",
}
TXN_MAPPING = {
    "date": "date", "product_id": "product_id",
    "price": "price", "units_sold": "units_sold",
}


@pytest.fixture(scope="module")
def service():
    return PricingService(data_path=DATA, num_items=N_ITEMS).compute()


def make_custom_dataset():
    rng = np.random.default_rng(11)
    products = pd.DataFrame({
        "product_id": ["P1", "P2", "P3", "P4"],
        "product_name": [
            "Ciabatta Roll", "Chicken Salad", "Iced Tea", "Fruit Cup",
        ],
        "category": ["Bakery", "Deli", "Beverages", "Deli"],
        "current_price": [3.5, 6.0, 2.5, 4.0],
        "retail_price": [4.0, 6.5, 2.5, 4.5],
        "cost": [1.5, 3.0, 1.0, 2.0],
        "inventory": [80, 45, 30, 120],
        "days_to_expiry": [3, 4, 60, 2],
    })
    rows = []
    for date in pd.date_range("2026-06-01", periods=75, freq="D"):
        for pid, retail in (("P1", 4.0), ("P2", 6.5), ("P4", 4.5)):
            price = retail * rng.choice([1.0, 0.9, 0.8, 0.7])
            units = 15.0 * (price / retail) ** -1.6 * rng.lognormal(0, 0.12)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": pid,
                "price": round(price, 2),
                "units_sold": round(units, 1),
            })
    transactions = pd.DataFrame(rows)
    return CustomDataset(
        validate_products(products, PRODUCT_MAPPING).valid,
        validate_transactions(transactions, TXN_MAPPING).valid,
        {"imported_at": "2026-08-25T00:00:00+00:00"},
    )


@pytest.fixture(scope="module")
def custom_service(tmp_path_factory):
    directory = tmp_path_factory.mktemp("custom")
    return CustomPricingService(
        str(directory), dataset=make_custom_dataset()
    ).compute()


# ---------------------------------------------------------------------------
# Multi-period paths through the service
# ---------------------------------------------------------------------------

class TestPathView:
    def test_payload_shape(self, service):
        sku = service.products()[0]["id"]
        payload = service.path(sku)
        assert payload["id"] == sku
        assert payload["horizon_days"] == len(payload["daily_prices"])
        assert set(payload["strategies"]) >= {"recommended", "hold"}
        assert payload["action"] in (
            "Hold", "Immediate markdown", "Staged markdown"
        )
        assert payload["constraints"]["floor"] is not None

    def test_unknown_sku_returns_none(self, service):
        assert service.path("NOPE") is None
        assert service.path_scenario("NOPE", [1.0]) is None

    def test_path_never_worse_than_hold(self, service):
        for summary in service.products():
            payload = service.path(summary["id"])
            rec = payload["strategies"]["recommended"]["econ"]
            hold = payload["strategies"]["hold"]["econ"]
            assert rec["economic_value"] >= hold["economic_value"] - 0.01

    def test_markdown_product_has_immediate_strategy(self, service):
        marked = [
            p for p in service.products() if p["action"] == "markdown"
        ]
        assert marked, "expected at least one markdown in the demo data"
        payload = service.path(marked[0]["id"])
        assert "immediate" in payload["strategies"]
        assert payload["strategies"]["immediate"]["price"] == (
            marked[0]["pricing"]["recommended"]
        )

    def test_trajectory_days_match_horizon(self, service):
        sku = service.products()[0]["id"]
        payload = service.path(sku)
        assert len(payload["trajectory"]) <= payload["horizon_days"]
        assert [r["day"] for r in payload["trajectory"]] == list(
            range(len(payload["trajectory"]))
        )

    def test_path_is_cached_and_deterministic(self, service):
        sku = service.products()[0]["id"]
        assert service.path(sku) == service.path(sku)


class TestPathScenario:
    def test_valid_path_is_evaluated_by_the_engine(self, service):
        summary = service.products()[0]
        sku = summary["id"]
        record = service._products[sku]
        cur = record.result.product.current_price
        result = service.path_scenario(sku, [cur, cur, cur])
        assert result["errors"] == []
        # A constant-current path must reproduce the hold strategy.
        hold = service.path(sku)["strategies"]["hold"]["econ"]
        assert result["econ"] == hold

    def test_engine_valuation_matches_direct_call(self, service):
        summary = next(
            p for p in service.products() if p["action"] == "markdown"
        )
        sku = summary["id"]
        record = service._products[sku]
        cur = record.result.product.current_price
        low = summary["pricing"]["recommended"]
        result = service.path_scenario(sku, [cur, cur, low])
        assert result["errors"] == []
        prices = [float(p) for p in result["daily_prices"]]
        direct = evaluate_price_path(
            record.result.product, service.config, daily_to_schedule(prices)
        )
        assert result["econ"]["economic_value"] == pytest.approx(
            round(direct["score"], 2)
        )

    def test_invalid_path_returns_errors_not_numbers(self, service):
        sku = service.products()[0]["id"]
        result = service.path_scenario(sku, [1000.0])
        assert result["errors"]
        assert "econ" not in result


class TestPathBacktest:
    def test_three_strategies_reported(self, service):
        result = service.path_backtest()
        assert result is not None
        for strategy in ("hold", "immediate", "path"):
            assert "economic_value" in result[strategy]
        assert result["n_products"] == N_ITEMS
        assert "SYNTHETIC SIMULATION" in result["label"]

    def test_backtest_is_cached(self, service):
        assert service.path_backtest() is service.path_backtest()

    def test_analytics_carries_path_backtest(self, service):
        analytics = service.analytics()
        assert analytics["path_backtest"] is not None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestExports:
    def test_recommendations_csv(self, service):
        csv_text = service.export_recommendations_csv()
        frame = pd.read_csv(io.StringIO(csv_text))
        assert len(frame) == N_ITEMS
        for column in (
            "Product_ID", "Product_Name", "Action", "Current_Price",
            "Recommended_Price", "Economic_Value_Improvement",
            "Expected_Waste_Current", "Sell_Through_Recommended",
        ):
            assert column in frame.columns
        products = service.products()
        assert frame.iloc[0]["Recommended_Price"] == (
            products[0]["pricing"]["recommended"]
        )

    def test_paths_csv_has_one_row_per_product_day(self, service):
        csv_text = service.export_paths_csv()
        frame = pd.read_csv(io.StringIO(csv_text))
        assert set(frame["Product_ID"].astype(str)) == set(
            p["id"] for p in service.products()
        )
        for sku, group in frame.groupby("Product_ID"):
            payload = service.path(str(sku))
            assert len(group) == len(payload["trajectory"])
            assert list(group["Day"]) == list(
                range(len(group))
            )

    def test_custom_export_includes_provenance_columns(self, custom_service):
        frame = pd.read_csv(
            io.StringIO(custom_service.export_recommendations_csv())
        )
        assert "Elasticity_Source" in frame.columns
        assert "Baseline_Demand_Source" in frame.columns


# ---------------------------------------------------------------------------
# Custom data service
# ---------------------------------------------------------------------------

class TestCustomService:
    def test_serves_all_products(self, custom_service):
        products = custom_service.products()
        assert len(products) == 4
        assert {p["id"] for p in products} == {"P1", "P2", "P3", "P4"}

    def test_meta_is_clearly_custom(self, custom_service):
        meta = custom_service.meta()
        assert meta["synthetic"] is False
        assert meta["source"] == "custom"
        assert meta["data_source"] == "Custom data import"
        assert meta["quality"]["products"] == 4
        assert meta["quality"]["transactions"] == 225

    def test_provenance_reported_per_product(self, custom_service):
        products = {p["id"]: p for p in custom_service.products()}
        # P1 has 75 days of history with price variation.
        assert products["P1"]["provenance"]["elasticity_source"] == (
            "estimated"
        )
        assert products["P1"]["provenance"]["baseline_source"] == "history"
        # P3 (Beverages) has no history at all.
        assert products["P3"]["provenance"]["elasticity_source"] == (
            "fallback"
        )
        assert products["P3"]["provenance"]["baseline_source"] in (
            "category", "global"
        )

    def test_no_synthetic_backtest_for_custom_data(self, custom_service):
        assert custom_service.dashboard()["backtest"] is None
        assert custom_service.path_backtest() is None
        assert custom_service.analytics()["path_backtest"] is None

    def test_detail_sweep_scenario_and_path_work(self, custom_service):
        detail = custom_service.product_detail("P1")
        assert detail["reasons"]
        sweep = custom_service.sweep("P1")
        assert len(sweep) == 41
        scenario = custom_service.scenario("P1", 3.0)
        assert scenario["breakdown"]["economic_value"] is not None
        path = custom_service.path("P1")
        assert path["horizon_days"] >= 1

    def test_elasticities_expose_reasons(self, custom_service):
        elasticities = custom_service.meta()["elasticities"]
        assert elasticities["Beverages"]["source"] == "default"
        assert "reason" in elasticities["Beverages"]

    def test_missing_transactions_is_a_clear_error(self, tmp_path):
        dataset = make_custom_dataset()
        empty = CustomDataset(
            dataset.products, dataset.transactions.iloc[0:0], {}
        )
        with pytest.raises(ValueError, match="transaction history"):
            CustomPricingService(str(tmp_path), dataset=empty).compute()

    def test_missing_dataset_is_a_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="no imported dataset"):
            CustomPricingService(str(tmp_path / "empty")).compute()

    def test_custom_is_deterministic(self):
        a = CustomPricingService("x", dataset=make_custom_dataset()).compute()
        b = CustomPricingService("x", dataset=make_custom_dataset()).compute()
        pd.testing.assert_frame_equal(a.recommendations, b.recommendations)
