"""API + service tests for the V1.1 adaptive layer: elasticity confidence,
replenishment endpoints, closed-loop simulation and backtest, enriched
exports — and the full integration path (custom data + replenishment file
+ confidence + closed loop)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from selfshelf.webapp import create_app

DATA = "data/walmart_large_sample_data_with_categories.csv"
N_ITEMS = 8


@pytest.fixture(scope="module")
def custom_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("adaptive_custom"))


@pytest.fixture(scope="module")
def client(custom_dir):
    app = create_app(
        data_path=DATA, num_items=N_ITEMS,
        compute_async=False, custom_dir=custom_dir,
    )
    with TestClient(app) as client:
        yield client


def products_csv() -> bytes:
    frame = pd.DataFrame({
        "sku": ["P1", "P2", "P3"],
        "item_name": ["Ciabatta Roll", "Chicken Salad", "Iced Tea"],
        "dept": ["Bakery", "Deli", "Beverages"],
        "sell_price": [3.5, 6.0, 2.5],
        "list_price": [4.0, 6.5, 2.5],
        "unit_cost": [1.5, 3.0, 1.0],
        "qty": [80, 45, 30],
        "shelf_life": [3, 4, 60],
    })
    return frame.to_csv(index=False).encode()


def transactions_csv() -> bytes:
    rng = np.random.default_rng(5)
    rows = []
    for date in pd.date_range("2026-06-01", periods=70, freq="D"):
        for pid, retail in (("P1", 4.0), ("P2", 6.5)):
            price = retail * rng.choice([1.0, 0.9, 0.8, 0.7])
            units = 15.0 * (price / retail) ** -1.6 * rng.lognormal(0, 0.1)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "sku": pid,
                "sale_price": round(price, 2),
                "quantity": round(units, 1),
            })
    return pd.DataFrame(rows).to_csv(index=False).encode()


def replenishment_csv() -> bytes:
    # History ends 2026-08-09 -> "today" is 2026-08-10 in the dataset's
    # timeline. One future delivery for P1, one past delivery (excluded),
    # one bad row (rejected).
    frame = pd.DataFrame({
        "delivery_date": [
            "2026-08-11", "2026-07-01", "2026-08-12", "not-a-date",
        ],
        "sku": ["P1", "P1", "P2", "P1"],
        "units_received": [200, 50, 60, 10],
    })
    return frame.to_csv(index=False).encode()


PRODUCTS_MAPPING = {
    "product_id": "sku", "product_name": "item_name", "category": "dept",
    "current_price": "sell_price", "retail_price": "list_price",
    "cost": "unit_cost", "inventory": "qty", "days_to_expiry": "shelf_life",
}
TXN_MAPPING = {
    "date": "date", "product_id": "sku",
    "price": "sale_price", "units_sold": "quantity",
}
REP_MAPPING = {
    "date": "delivery_date", "product_id": "sku",
    "quantity": "units_received",
}


def upload(client, kind, name, payload):
    response = client.post(
        f"/api/data/upload?kind={kind}",
        files={"file": (name, payload, "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Synthetic source
# ---------------------------------------------------------------------------

class TestElasticityEndpointSynthetic:
    def test_confidence_payload(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        payload = client.get(f"/api/products/{sku}/elasticity").json()
        assert payload["id"] == sku
        conf = payload["confidence"]
        assert conf is not None
        assert conf["source"] in ("estimated", "fallback")
        if conf["source"] == "estimated":
            assert conf["lower_ci"] < conf["elasticity"] < conf["upper_ci"]
            assert conf["standard_error"] > 0
            assert conf["n_observations"] > 0
            # The interval brackets the engine's own pricing elasticity.
            assert conf["elasticity"] == pytest.approx(
                payload["elasticity"], abs=1e-3
            )
        else:
            assert conf["lower_ci"] is None

    def test_unknown_sku_404(self, client):
        assert client.get("/api/products/nope/elasticity").status_code == 404

    def test_detail_carries_confidence_and_replenishment(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        detail = client.get(f"/api/products/{sku}").json()
        assert "elasticity_confidence" in detail
        assert detail["replenishment"]["status"] == "not_modeled"


class TestReplenishmentEndpointSynthetic:
    def test_synthetic_source_reports_not_modeled(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        payload = client.get(f"/api/products/{sku}/replenishment").json()
        assert payload["status"] == "not_modeled"
        assert "Not modeled" in payload["message"]


class TestAdaptiveSimulationEndpoint:
    def test_shortfall_simulation_shows_planned_vs_actual(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        payload = client.get(
            f"/api/products/{sku}/adaptive?demand=0.4"
        ).json()
        assert "SYNTHETIC SIMULATION" in payload["label"]
        assert payload["demand_factor"] == 0.4
        assert len(payload["days"]) >= 1
        day0 = payload["days"][0]
        # Planned vs actual is explicit on every day.
        assert "forecast_sales" in day0 and "observed_sales" in day0
        assert day0["observed_sales"] <= day0["forecast_sales"] + 1e-6
        assert "outcomes" in payload
        assert "closed_loop" in payload["outcomes"]
        assert "open_loop" in payload["outcomes"]
        assert payload["value_of_feedback"] is not None

    def test_deterministic_and_cached(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        a = client.get(f"/api/products/{sku}/adaptive?demand=0.6").json()
        b = client.get(f"/api/products/{sku}/adaptive?demand=0.6").json()
        assert a == b

    def test_demand_factor_validation(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        assert client.get(
            f"/api/products/{sku}/adaptive?demand=0"
        ).status_code == 422
        assert client.get(
            f"/api/products/{sku}/adaptive?demand=99"
        ).status_code == 422


class TestAdaptiveBacktestEndpoint:
    def test_synthetic_backtest_compares_three_strategies(self, client):
        payload = client.get("/api/backtest/adaptive").json()
        assert payload["available"] is True
        assert "SYNTHETIC SIMULATION" in payload["label"]
        for strategy in ("hold", "open_loop", "closed_loop"):
            assert payload[strategy]["revenue"] >= 0
            assert payload[strategy]["waste_units"] >= 0
        assert payload["n_products"] == N_ITEMS


class TestEnrichedExports:
    def test_recommendations_csv_includes_confidence_columns(self, client):
        text = client.get("/api/export/recommendations.csv").text
        header = text.splitlines()[0]
        for column in (
            "Elasticity_Confidence", "Elasticity_CI_Low",
            "Elasticity_CI_High", "Elasticity_N_Obs",
            "Replenishment_Status",
        ):
            assert column in header

    def test_elasticity_csv_export(self, client):
        response = client.get("/api/export/elasticity.csv")
        assert response.status_code == 200
        header = response.text.splitlines()[0]
        for column in ("Department", "Elasticity", "Standard_Error",
                       "CI_Low", "CI_High", "Confidence", "Reason"):
            assert column in header

    def test_paths_csv_includes_replenishment_column(self, client):
        text = client.get("/api/export/paths.csv").text
        assert "Replenishment_Received" in text.splitlines()[0]


# ---------------------------------------------------------------------------
# Full integration: custom data + replenishment + confidence + closed loop
# ---------------------------------------------------------------------------

class TestCustomDataIntegration:
    @pytest.fixture(scope="class", autouse=True)
    def imported(self, client):
        upload(client, "products", "products.csv", products_csv())
        upload(client, "transactions", "txn.csv", transactions_csv())
        rep_info = upload(
            client, "replenishment", "deliveries.csv", replenishment_csv()
        )
        assert rep_info["suggested_mapping"]["quantity"]["column"] == (
            "units_received"
        )
        response = client.post("/api/data/import", json={
            "products_mapping": PRODUCTS_MAPPING,
            "transactions_mapping": TXN_MAPPING,
            "replenishment_mapping": REP_MAPPING,
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["replenishment"]["has_data"] is True
        yield payload
        client.post("/api/data/source", json={"source": "synthetic"})

    def test_replenishment_rows_validated(self, imported):
        rows = imported["replenishment_rows"]
        assert rows["rows_valid"] == 3   # bad date rejected
        assert rows["rows_rejected"] == 1

    def test_product_with_future_delivery_reports_known(self, client):
        payload = client.get("/api/products/P1/replenishment").json()
        assert payload["status"] == "known"
        assert payload["next_arrival"]["day"] == 1
        assert payload["next_arrival"]["units"] == 200.0
        # The past delivery (2026-07-01) is already in inventory: excluded.
        assert payload["total_units"] == 200.0

    def test_product_without_deliveries_is_explicit(self, client):
        payload = client.get("/api/products/P3/replenishment").json()
        assert payload["status"] == "none_scheduled"
        assert "no future" in payload["message"].lower()

    def test_path_incorporates_the_delivery(self, client):
        payload = client.get("/api/products/P1/path").json()
        assert payload["replenishment"]["status"] == "known"
        rows = payload["trajectory"]
        assert rows[1]["replenishment_received"] == 200.0
        # Inventory jumps when the delivery lands, and day 0 never sees it.
        assert rows[0]["start_inventory"] == 80.0
        assert rows[1]["available_inventory"] > rows[0]["end_inventory"]

    def test_custom_elasticity_confidence_from_history(self, client):
        payload = client.get("/api/products/P1/elasticity").json()
        conf = payload["confidence"]
        assert conf["source"] == "estimated"
        assert conf["lower_ci"] < conf["elasticity"] < conf["upper_ci"]
        assert conf["n_observations"] >= 50
        # Beverages (P3) has no history at all -> fallback, no fake CI.
        p3 = client.get("/api/products/P3/elasticity").json()
        assert p3["confidence"]["confidence"] == "fallback"
        assert p3["confidence"]["standard_error"] is None

    def test_custom_adaptive_simulation_with_replenishment(self, client):
        payload = client.get(
            "/api/products/P1/adaptive?demand=0.5"
        ).json()
        assert "SYNTHETIC SIMULATION" in payload["label"]
        assert payload["outcomes"]["closed_loop"][
            "replenishment_received"
        ] == 200.0
        # The state machine saw the delivery on the correct day.
        assert payload["days"][0]["arrivals_applied"] == 200.0

    def test_custom_source_has_no_ground_truth_backtest(self, client):
        payload = client.get("/api/backtest/adaptive").json()
        assert payload["available"] is False
        assert "ground-truth" in payload["message"]

    def test_meta_reports_replenishment_coverage(self, client):
        meta = client.get("/api/products").json()["meta"]
        assert meta["replenishment"]["has_data"] is True
        assert meta["replenishment"]["products_with_future_deliveries"] >= 1
        assert meta["elasticity_confidence"]["Bakery"] is not None

    def test_persistence_roundtrip_keeps_replenishment(
        self, client, custom_dir
    ):
        # A new app instance over the same directory reloads deliveries.
        app = create_app(
            data_path=DATA, num_items=N_ITEMS,
            compute_async=False, custom_dir=custom_dir,
        )
        with TestClient(app) as fresh:
            payload = fresh.get("/api/products/P1/replenishment").json()
            assert payload["status"] == "known"
            assert payload["total_units"] == 200.0
