"""API tests for the completion features: multi-period path endpoints,
CSV exports, and the full custom-data import workflow (upload -> mapping ->
validate -> import -> custom mode -> persistence)."""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from selfshelf.webapp import create_app

DATA = "data/walmart_large_sample_data_with_categories.csv"
N_ITEMS = 8


@pytest.fixture(scope="module")
def custom_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("custom_data"))


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
        "sku": ["P1", "P2", "P3", "", "P5"],
        "item_name": [
            "Ciabatta Roll", "Chicken Salad", "Iced Tea", "Ghost Row",
            "Fruit Cup",
        ],
        "dept": ["Bakery", "Deli", "Beverages", "Deli", "Deli"],
        "sell_price": [3.5, 6.0, 2.5, 1.0, -4.0],
        "list_price": [4.0, 6.5, 2.5, 1.0, 4.5],
        "unit_cost": [1.5, 3.0, 1.0, 0.5, 2.0],
        "qty": [80, 45, 30, 10, 120],
        "shelf_life": [3, 4, 60, 5, 2],
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
    rows.append({
        "date": "bad-date", "sku": "P1", "sale_price": 4.0, "quantity": 3,
    })
    return pd.DataFrame(rows).to_csv(index=False).encode()


PRODUCTS_MAPPING = {
    "product_id": "sku", "product_name": "item_name", "category": "dept",
    "current_price": "sell_price", "retail_price": "list_price",
    "cost": "unit_cost", "inventory": "qty", "days_to_expiry": "shelf_life",
}
TXN_MAPPING = {
    "date": "date", "product_id": "sku",
    "price": "sale_price", "units_sold": "quantity",
}


# ---------------------------------------------------------------------------
# Path endpoints (synthetic source)
# ---------------------------------------------------------------------------

class TestPathEndpoints:
    def test_path_payload(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        payload = client.get(f"/api/products/{sku}/path").json()
        assert payload["id"] == sku
        assert payload["horizon_days"] == len(payload["daily_prices"])
        assert "hold" in payload["strategies"]
        assert "recommended" in payload["strategies"]

    def test_unknown_sku_404(self, client):
        assert client.get("/api/products/nope/path").status_code == 404
        assert client.post(
            "/api/products/nope/path/scenario",
            json={"daily_prices": [1.0]},
        ).status_code == 404

    def test_path_scenario_roundtrip(self, client):
        listing = client.get("/api/products").json()["products"]
        summary = listing[0]
        cur = summary["pricing"]["current"]
        result = client.post(
            f"/api/products/{summary['id']}/path/scenario",
            json={"daily_prices": [cur, cur]},
        ).json()
        assert result["errors"] == []
        assert result["econ"]["economic_value"] is not None
        assert "compare" in result

    def test_path_scenario_invalid_path_returns_errors(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        result = client.post(
            f"/api/products/{sku}/path/scenario",
            json={"daily_prices": [9999.0]},
        ).json()
        assert result["errors"]

    def test_path_scenario_rejects_over_60_days(self, client):
        sku = client.get("/api/products").json()["products"][0]["id"]
        response = client.post(
            f"/api/products/{sku}/path/scenario",
            json={"daily_prices": [1.0] * 61},
        )
        assert response.status_code == 422

    def test_analytics_includes_path_backtest(self, client):
        payload = client.get("/api/analytics").json()
        backtest = payload["path_backtest"]
        assert backtest is not None
        for strategy in ("hold", "immediate", "path"):
            assert strategy in backtest
        assert "SYNTHETIC" in backtest["label"]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestExports:
    def test_recommendations_csv(self, client):
        response = client.get("/api/export/recommendations.csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        frame = pd.read_csv(io.StringIO(response.text))
        assert len(frame) == N_ITEMS
        assert "Recommended_Price" in frame.columns

    def test_paths_csv(self, client):
        response = client.get("/api/export/paths.csv")
        assert response.status_code == 200
        frame = pd.read_csv(io.StringIO(response.text))
        assert frame["Product_ID"].nunique() == N_ITEMS
        assert "Recommended_Price" in frame.columns
        assert "Day" in frame.columns


# ---------------------------------------------------------------------------
# Custom data workflow
# ---------------------------------------------------------------------------

class TestCustomDataWorkflow:
    """Ordered end-to-end flow; later tests depend on earlier ones, which
    pytest preserves within a class."""

    def test_initial_status_is_synthetic(self, client):
        payload = client.get("/api/status").json()
        assert payload["source"] == "synthetic"
        assert payload["custom"]["available"] is False

    def test_switch_to_custom_without_data_fails(self, client):
        response = client.post(
            "/api/data/source", json={"source": "custom"}
        )
        assert response.status_code == 400

    def test_upload_rejects_bad_kind_and_bad_file(self, client):
        assert client.post(
            "/api/data/upload?kind=nonsense",
            files={"file": ("x.csv", b"a,b\n1,2", "text/csv")},
        ).status_code == 422
        assert client.post(
            "/api/data/upload?kind=products",
            files={"file": ("x.csv", b"", "text/csv")},
        ).status_code == 400

    def test_upload_products_suggests_mapping(self, client):
        response = client.post(
            "/api/data/upload?kind=products",
            files={"file": ("products.csv", products_csv(), "text/csv")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["row_count"] == 5
        suggested = payload["suggested_mapping"]
        assert suggested["product_id"]["column"] == "sku"
        assert suggested["inventory"]["column"] == "qty"
        assert len(payload["sample_rows"]) == 5

    def test_upload_transactions(self, client):
        response = client.post(
            "/api/data/upload?kind=transactions",
            files={
                "file": (
                    "transactions.csv", transactions_csv(), "text/csv"
                ),
            },
        )
        assert response.status_code == 200
        suggested = response.json()["suggested_mapping"]
        assert suggested["date"]["column"] == "date"
        assert suggested["units_sold"]["column"] == "quantity"

    def test_validate_reports_bad_rows(self, client):
        response = client.post("/api/data/validate", json={
            "products_mapping": PRODUCTS_MAPPING,
            "transactions_mapping": TXN_MAPPING,
        })
        assert response.status_code == 200
        payload = response.json()
        # 2 bad product rows (missing id, negative price); 1 bad txn date.
        assert payload["products"]["rows_valid"] == 3
        assert payload["products"]["rows_rejected"] == 2
        assert payload["transactions"]["rows_rejected"] == 1
        assert payload["can_import"] is True
        assert payload["products"]["preview"]

    def test_validate_with_missing_mapping_cannot_import(self, client):
        response = client.post("/api/data/validate", json={
            "products_mapping": {"product_id": "sku"},
            "transactions_mapping": TXN_MAPPING,
        })
        payload = response.json()
        assert payload["can_import"] is False
        assert payload["products"]["errors"]

    def test_import_switches_to_custom_mode(self, client):
        response = client.post("/api/data/import", json={
            "products_mapping": PRODUCTS_MAPPING,
            "transactions_mapping": TXN_MAPPING,
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["imported"] is True
        assert payload["quality"]["products"] == 3
        assert payload["quality"]["transactions"] == 140

        status = client.get("/api/status").json()
        assert status["source"] == "custom"
        assert status["ready"] is True

    def test_custom_mode_serves_imported_products(self, client):
        payload = client.get("/api/products").json()
        assert {p["id"] for p in payload["products"]} == {"P1", "P2", "P3"}
        assert payload["meta"]["synthetic"] is False
        assert payload["meta"]["data_source"] == "Custom data import"
        p1 = next(p for p in payload["products"] if p["id"] == "P1")
        assert p1["provenance"]["elasticity_source"] == "estimated"

    def test_custom_dashboard_has_no_synthetic_backtest(self, client):
        payload = client.get("/api/dashboard").json()
        assert payload["backtest"] is None
        assert payload["kpis"]["products"] == 3

    def test_custom_paths_and_exports_work(self, client):
        path = client.get("/api/products/P1/path").json()
        assert path["horizon_days"] >= 1
        response = client.get("/api/export/recommendations.csv")
        frame = pd.read_csv(io.StringIO(response.text))
        assert "Elasticity_Source" in frame.columns
        assert len(frame) == 3

    def test_rejected_rows_are_downloadable(self, client):
        response = client.get("/api/data/rejected?kind=products")
        assert response.status_code == 200
        frame = pd.read_csv(io.StringIO(response.text))
        assert len(frame) == 2
        assert "reject_reason" in frame.columns

    def test_data_status_reflects_import(self, client):
        payload = client.get("/api/data/status").json()
        assert payload["active_source"] == "custom"
        assert payload["custom"]["available"] is True
        assert payload["custom"]["meta"]["quality"]["products"] == 3
        assert payload["uploads"]["products"] is True

    def test_switch_back_to_synthetic(self, client):
        response = client.post(
            "/api/data/source", json={"source": "synthetic"}
        )
        assert response.status_code == 200
        payload = client.get("/api/products").json()
        assert payload["meta"]["synthetic"] is True
        assert len(payload["products"]) == N_ITEMS

    def test_switch_to_custom_again(self, client):
        client.post("/api/data/source", json={"source": "custom"})
        assert client.get("/api/status").json()["source"] == "custom"


class TestPersistence:
    def test_custom_dataset_survives_restart(self, client, custom_dir):
        # A brand-new app instance pointed at the same directory must load
        # the imported dataset and honor the persisted active source.
        app = create_app(
            data_path=DATA, num_items=N_ITEMS,
            compute_async=False, custom_dir=custom_dir,
        )
        with TestClient(app) as fresh:
            status = fresh.get("/api/status").json()
            assert status["source"] == "custom"
            assert status["custom"]["available"] is True
            products = fresh.get("/api/products").json()["products"]
            assert {p["id"] for p in products} == {"P1", "P2", "P3"}
