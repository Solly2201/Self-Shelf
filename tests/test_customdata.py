"""Custom data ingestion tests: mapping suggestions, validation, quality
reporting, persistence, and historically-estimated demand parameters."""

import numpy as np
import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.customdata import (
    CustomDataset,
    estimate_demand_params,
    load_dataset,
    save_dataset,
    suggest_mapping,
    validate_products,
    validate_transactions,
)

PRODUCT_MAPPING = {
    "product_id": "product_id",
    "product_name": "product_name",
    "category": "category",
    "current_price": "current_price",
    "retail_price": "retail_price",
    "cost": "cost",
    "inventory": "inventory",
    "days_to_expiry": "days_to_expiry",
}

TXN_MAPPING = {
    "date": "date",
    "product_id": "product_id",
    "price": "price",
    "units_sold": "units_sold",
}


def products_frame(**overrides):
    base = {
        "product_id": ["A1", "B2", "C3"],
        "product_name": ["Sourdough Loaf", "Greek Yogurt", "Orange Juice"],
        "category": ["Bakery", "Dairy", "Beverages"],
        "current_price": [4.5, 1.2, 3.99],
        "retail_price": [5.0, 1.5, 3.99],
        "cost": [2.0, 0.6, 2.5],
        "inventory": [40, 120, 60],
        "days_to_expiry": [3, 10, 30],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def transactions_frame(n_days=60, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2026-05-01", periods=n_days, freq="D")
    for date in dates:
        for pid, retail, cur in (("A1", 5.0, 4.5), ("B2", 1.5, 1.2)):
            price = retail * rng.choice([1.0, 0.9, 0.8, 0.7])
            units = 20.0 * (price / retail) ** -1.8 * rng.lognormal(0, 0.1)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": pid,
                "price": round(price, 2),
                "units_sold": round(units, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mapping suggestions
# ---------------------------------------------------------------------------

class TestSuggestMapping:
    def test_exact_synonyms(self):
        cols = ["sku", "item_name", "dept", "sell_price", "unit_cost",
                "qty", "shelf_life"]
        s = suggest_mapping(cols, "products")
        assert s["product_id"]["column"] == "sku"
        assert s["product_name"]["column"] == "item_name"
        assert s["category"]["column"] == "dept"
        assert s["current_price"]["column"] == "sell_price"
        assert s["cost"]["column"] == "unit_cost"
        assert s["inventory"]["column"] == "qty"
        assert s["days_to_expiry"]["column"] == "shelf_life"
        assert all(v["confidence"] == "exact" for v in s.values())

    def test_transaction_synonyms(self):
        s = suggest_mapping(
            ["transaction_date", "sku", "sale_price", "quantity"],
            "transactions",
        )
        assert s["date"]["column"] == "transaction_date"
        assert s["product_id"]["column"] == "sku"
        assert s["price"]["column"] == "sale_price"
        assert s["units_sold"]["column"] == "quantity"

    def test_case_and_spaces_normalized(self):
        s = suggest_mapping(["Product ID", "Item Name"], "products")
        assert s["product_id"]["column"] == "Product ID"
        assert s["product_name"]["column"] == "Item Name"

    def test_each_column_used_once(self):
        s = suggest_mapping(["price"], "transactions")
        used = [v["column"] for v in s.values()]
        assert len(used) == len(set(used))

    def test_fuzzy_fallback(self):
        s = suggest_mapping(["my_store_inventory_level"], "products")
        assert s.get("inventory", {}).get("column") == (
            "my_store_inventory_level"
        )
        assert s["inventory"]["confidence"] == "fuzzy"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            suggest_mapping(["a"], "nonsense")


# ---------------------------------------------------------------------------
# Product validation
# ---------------------------------------------------------------------------

class TestValidateProducts:
    def test_clean_file_passes(self):
        result = validate_products(products_frame(), PRODUCT_MAPPING)
        assert result.ok
        assert len(result.valid) == 3
        assert len(result.rejected) == 0

    def test_missing_required_mapping_is_fatal(self):
        mapping = dict(PRODUCT_MAPPING)
        del mapping["cost"]
        result = validate_products(products_frame(), mapping)
        assert not result.ok
        assert any("cost" in e for e in result.errors)

    def test_nonexistent_mapped_column_is_fatal(self):
        mapping = dict(PRODUCT_MAPPING, cost="no_such_column")
        result = validate_products(products_frame(), mapping)
        assert not result.ok

    def test_bad_rows_rejected_with_reasons(self):
        df = products_frame(
            product_id=["A1", "", "C3"],
            current_price=[4.5, 1.2, -2.0],
        )
        result = validate_products(df, PRODUCT_MAPPING)
        assert len(result.valid) == 1
        assert len(result.rejected) == 2
        assert "reject_reason" in result.rejected.columns
        messages = " ".join(i["message"] for i in result.issues)
        assert "missing product_id" in messages
        assert "current_price" in messages

    def test_negative_cost_and_inventory_rejected(self):
        df = products_frame(cost=[2.0, -1.0, 2.5], inventory=[40, 120, -5])
        result = validate_products(df, PRODUCT_MAPPING)
        assert len(result.valid) == 1
        assert len(result.rejected) == 2

    def test_duplicate_product_id_keeps_first(self):
        df = products_frame(product_id=["A1", "A1", "C3"])
        result = validate_products(df, PRODUCT_MAPPING)
        assert len(result.valid) == 2
        assert any("duplicate" in i["message"] for i in result.issues)

    def test_currency_symbols_parsed(self):
        df = products_frame(current_price=["$4.50", "$1.20", "$3.99"])
        result = validate_products(df, PRODUCT_MAPPING)
        assert len(result.valid) == 3
        assert result.valid["current_price"].tolist() == [4.5, 1.2, 3.99]

    def test_retail_below_current_is_raised_with_warning(self):
        df = products_frame(retail_price=[4.0, 1.5, 3.99])  # A1: 4.0 < 4.5
        result = validate_products(df, PRODUCT_MAPPING)
        assert result.valid.iloc[0]["retail_price"] == 4.5
        assert any("retail_price" in w for w in result.warnings)

    def test_missing_optional_fields_defaulted(self):
        df = products_frame().drop(columns=["category", "retail_price"])
        mapping = {
            k: v for k, v in PRODUCT_MAPPING.items()
            if k not in ("category", "retail_price")
        }
        result = validate_products(df, mapping)
        assert result.ok
        assert (result.valid["category"] == "General").all()
        assert (
            result.valid["retail_price"] == result.valid["current_price"]
        ).all()

    def test_selling_below_cost_warns_but_imports(self):
        df = products_frame(cost=[5.5, 0.6, 2.5])  # A1 sells below cost
        result = validate_products(df, PRODUCT_MAPPING)
        assert len(result.valid) == 3
        assert any("below cost" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Transaction validation
# ---------------------------------------------------------------------------

class TestValidateTransactions:
    def test_clean_file_passes(self):
        result = validate_transactions(transactions_frame(), TXN_MAPPING)
        assert result.ok
        assert len(result.rejected) == 0
        assert result.valid["date"].is_monotonic_increasing

    def test_bad_dates_and_negative_units_rejected(self):
        df = pd.DataFrame({
            "date": ["2026-05-01", "not-a-date", "2026-05-03"],
            "product_id": ["A1", "A1", "A1"],
            "price": [4.5, 4.5, 4.5],
            "units_sold": [10, 12, -3],
        })
        result = validate_transactions(df, TXN_MAPPING)
        assert len(result.valid) == 1
        assert len(result.rejected) == 2
        messages = " ".join(i["message"] for i in result.issues)
        assert "date" in messages
        assert "units_sold" in messages

    def test_unknown_product_ids_rejected_when_known_ids_given(self):
        df = pd.DataFrame({
            "date": ["2026-05-01", "2026-05-01"],
            "product_id": ["A1", "GHOST"],
            "price": [4.5, 2.0],
            "units_sold": [10, 5],
        })
        result = validate_transactions(df, TXN_MAPPING, known_ids=["A1"])
        assert len(result.valid) == 1
        assert any(
            "does not appear" in i["message"] for i in result.issues
        )

    def test_zero_units_are_valid(self):
        df = pd.DataFrame({
            "date": ["2026-05-01"],
            "product_id": ["A1"],
            "price": [4.5],
            "units_sold": [0],
        })
        result = validate_transactions(df, TXN_MAPPING)
        assert len(result.valid) == 1


# ---------------------------------------------------------------------------
# Dataset quality + persistence
# ---------------------------------------------------------------------------

class TestDatasetAndPersistence:
    def make_dataset(self):
        products = validate_products(products_frame(), PRODUCT_MAPPING).valid
        txn = validate_transactions(transactions_frame(), TXN_MAPPING).valid
        return CustomDataset(products, txn, {"note": "test"})

    def test_quality_summary(self):
        summary = self.make_dataset().quality_summary()
        assert summary["products"] == 3
        assert summary["transactions"] == 120
        assert summary["products_with_history"] == 2
        assert summary["products_without_history"] == 1
        assert summary["date_range"]["start"] == "2026-05-01"

    def test_save_load_round_trip(self, tmp_path):
        dataset = self.make_dataset()
        save_dataset(dataset, str(tmp_path / "custom"))
        loaded = load_dataset(str(tmp_path / "custom"))
        assert loaded is not None
        assert len(loaded.products) == 3
        assert len(loaded.transactions) == 120
        assert loaded.meta["note"] == "test"
        assert str(loaded.transactions["date"].dtype).startswith("datetime")
        # product_id stays a string even when it looks numeric.
        assert loaded.products["product_id"].dtype == object

    def test_load_missing_directory_returns_none(self, tmp_path):
        assert load_dataset(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# Demand parameter estimation
# ---------------------------------------------------------------------------

class TestEstimateDemandParams:
    def make_dataset(self, txn=None):
        products = validate_products(products_frame(), PRODUCT_MAPPING).valid
        if txn is None:
            txn = transactions_frame()
        txn_valid = validate_transactions(txn, TXN_MAPPING).valid
        return CustomDataset(products, txn_valid)

    def test_elasticity_recovered_from_history(self):
        config = PricingConfig()
        params, categories = estimate_demand_params(
            self.make_dataset(), config
        )
        # A1 (Bakery) and B2 (Dairy) histories were generated with a true
        # elasticity of -1.8 and plenty of price variation.
        for cat in ("Bakery", "Dairy"):
            assert categories[cat]["source"] == "estimated"
            assert categories[cat]["elasticity"] == pytest.approx(
                -1.8, abs=0.35
            )
            assert categories[cat]["n_observations"] >= 50
        assert params["A1"].elasticity_source == "estimated"

    def test_category_without_history_uses_fallback(self):
        config = PricingConfig()
        params, categories = estimate_demand_params(
            self.make_dataset(), config
        )
        # C3 (Beverages) has no transactions at all.
        assert categories["Beverages"]["source"] == "fallback"
        assert "no transaction history" in categories["Beverages"]["reason"]
        assert params["C3"].elasticity == config.elasticity.default

    def test_insufficient_observations_fall_back(self):
        config = PricingConfig()
        txn = transactions_frame(n_days=5)  # 10 observations only
        _, categories = estimate_demand_params(
            self.make_dataset(txn), config, min_observations=30
        )
        assert categories["Bakery"]["source"] == "fallback"
        assert "observations" in categories["Bakery"]["reason"]

    def test_no_price_variation_falls_back_transparently(self):
        config = PricingConfig()
        rows = []
        for date in pd.date_range("2026-05-01", periods=80, freq="D"):
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": "A1",
                "price": 4.5,          # never changes
                "units_sold": 12.0,
            })
        _, categories = estimate_demand_params(
            self.make_dataset(pd.DataFrame(rows)), config
        )
        assert categories["Bakery"]["source"] == "fallback"
        assert "price variation" in categories["Bakery"]["reason"]

    def test_baseline_rate_matches_recent_history(self):
        config = PricingConfig()
        rows = []
        # 30 days of exactly 12 units/day at the current price.
        for date in pd.date_range("2026-05-01", periods=30, freq="D"):
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": "A1",
                "price": 4.5,
                "units_sold": 12.0,
            })
        params, _ = estimate_demand_params(
            self.make_dataset(pd.DataFrame(rows)), config
        )
        assert params["A1"].baseline_daily_demand == pytest.approx(
            12.0, rel=0.01
        )
        assert params["A1"].baseline_source == "history"
        assert params["A1"].baseline_n_days >= 28

    def test_missing_days_count_as_zero_sales(self):
        config = PricingConfig()
        rows = []
        # Sales recorded only every other day: rate should be ~6/day.
        for i, date in enumerate(
            pd.date_range("2026-05-01", periods=28, freq="D")
        ):
            if i % 2 == 0:
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "product_id": "A1",
                    "price": 4.5,
                    "units_sold": 12.0,
                })
        params, _ = estimate_demand_params(
            self.make_dataset(pd.DataFrame(rows)), config
        )
        assert params["A1"].baseline_daily_demand == pytest.approx(
            12.0 * 14 / 27, rel=0.05
        )

    def test_product_without_history_uses_category_median(self):
        config = PricingConfig()
        params, _ = estimate_demand_params(self.make_dataset(), config)
        # C3 has no history and no Beverages peers -> global median.
        assert params["C3"].baseline_source in ("category", "global")
        assert params["C3"].baseline_daily_demand > 0

    def test_deterministic(self):
        config = PricingConfig()
        dataset = self.make_dataset()
        a, cat_a = estimate_demand_params(dataset, config)
        b, cat_b = estimate_demand_params(dataset, config)
        assert {k: vars(v) for k, v in a.items()} == {
            k: vars(v) for k, v in b.items()
        }
        assert cat_a == cat_b

    def test_estimated_categories_carry_confidence_intervals(self):
        config = PricingConfig()
        _, categories = estimate_demand_params(self.make_dataset(), config)
        for cat in ("Bakery", "Dairy"):
            conf = categories[cat]["confidence"]
            assert conf is not None
            assert conf["source"] == "estimated"
            # Interval brackets the same point estimate the engine prices
            # with — the two can never disagree.
            assert conf["elasticity"] == pytest.approx(
                categories[cat]["elasticity"], abs=1e-4
            )
            assert conf["lower_ci"] < conf["elasticity"] < conf["upper_ci"]
            assert conf["standard_error"] > 0
            assert conf["n_observations"] == categories[cat]["n_observations"]

    def test_fallback_categories_have_no_fake_interval(self):
        config = PricingConfig()
        _, categories = estimate_demand_params(self.make_dataset(), config)
        conf = categories["Beverages"]["confidence"]
        assert conf["confidence"] == "fallback"
        assert conf["standard_error"] is None
        assert conf["lower_ci"] is None and conf["upper_ci"] is None

    def test_price_variation_fallback_confidence_reason_matches(self):
        config = PricingConfig()
        rows = []
        for date in pd.date_range("2026-05-01", periods=80, freq="D"):
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": "A1",
                "price": 4.5,
                "units_sold": 12.0,
            })
        _, categories = estimate_demand_params(
            self.make_dataset(pd.DataFrame(rows)), config
        )
        conf = categories["Bakery"]["confidence"]
        assert conf["confidence"] == "fallback"
        assert conf["reason"] == categories["Bakery"]["reason"]
