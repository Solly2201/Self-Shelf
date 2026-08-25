"""API tests: the FastAPI layer must faithfully expose the service data
and behave well before the optimization run finishes."""

import pytest
from fastapi.testclient import TestClient

from selfshelf.webapp import create_app

DATA = "data/walmart_large_sample_data_with_categories.csv"
N_ITEMS = 8


@pytest.fixture(scope="module")
def client():
    app = create_app(data_path=DATA, num_items=N_ITEMS, compute_async=False)
    with TestClient(app) as client:
        yield client


class TestNotReady:
    def test_endpoints_report_computing_before_startup(self):
        app = create_app(
            data_path=DATA, num_items=N_ITEMS, compute_async=False
        )
        # No lifespan context: the compute step has not run.
        cold = TestClient(app)
        assert cold.get("/api/status").json()["ready"] is False
        assert cold.get("/api/dashboard").status_code == 503
        assert cold.get("/api/products").status_code == 503


class TestEndpoints:
    def test_status_ready(self, client):
        payload = client.get("/api/status").json()
        assert payload["ready"] is True
        assert payload["error"] is None

    def test_dashboard_shape(self, client):
        payload = client.get("/api/dashboard").json()
        assert payload["kpis"]["products"] == N_ITEMS
        assert payload["meta"]["synthetic"] is True
        assert "queue" in payload and "risk_counts" in payload
        assert payload["backtest"]["hold"]["gross_profit"] is not None

    def test_products_list(self, client):
        payload = client.get("/api/products").json()
        assert len(payload["products"]) == N_ITEMS
        first = payload["products"][0]
        for key in ("id", "name", "department", "action", "status",
                    "pricing", "inventory", "demand", "economics"):
            assert key in first

    def test_product_detail_roundtrip(self, client):
        listing = client.get("/api/products").json()["products"]
        sku = listing[0]["id"]
        detail = client.get(f"/api/products/{sku}").json()
        assert detail["id"] == sku
        assert detail["reasons"]
        # The detail view serves the same numbers as the listing.
        assert detail["pricing"] == listing[0]["pricing"]
        assert detail["economics"] == listing[0]["economics"]

    def test_unknown_product_is_404(self, client):
        assert client.get("/api/products/does-not-exist").status_code == 404
        assert (
            client.get("/api/products/does-not-exist/sweep").status_code
            == 404
        )
        assert (
            client.get(
                "/api/products/does-not-exist/scenario", params={"price": 1}
            ).status_code
            == 404
        )

    def test_sweep_has_markers_and_points(self, client):
        listing = client.get("/api/products").json()["products"]
        sku = listing[0]["id"]
        payload = client.get(f"/api/products/{sku}/sweep").json()
        assert len(payload["points"]) == 41
        assert payload["current_price"] == listing[0]["pricing"]["current"]
        assert payload["recommended_price"] == (
            listing[0]["pricing"]["recommended"]
        )

    def test_scenario_requires_positive_price(self, client):
        listing = client.get("/api/products").json()["products"]
        sku = listing[0]["id"]
        assert (
            client.get(
                f"/api/products/{sku}/scenario", params={"price": -1}
            ).status_code
            == 422
        )
        ok = client.get(
            f"/api/products/{sku}/scenario",
            params={"price": listing[0]["pricing"]["current"]},
        ).json()
        assert ok["breakdown"]["economic_value"] is not None

    def test_analytics_shape(self, client):
        payload = client.get("/api/analytics").json()
        assert payload["meta"]["synthetic"] is True
        assert sum(
            payload["days_of_supply_distribution"].values()
        ) == N_ITEMS

    def test_index_serves_the_dashboard_shell(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Self-Shelf" in response.text
