import numpy as np
import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.features import clean_data, engineer_features


def raw_frame():
    return pd.DataFrame({
        "PRICE_RETAIL": [5.0, 3.0, -1.0, 4.0, np.nan],
        "PRICE_CURRENT": [4.5, 3.5, 2.0, 4.0, 3.0],
        "DEPARTMENT": [" Bakery ", "Deli", "Snacks", "Mystery", "Deli"],
        "PRODUCT_NAME": ["Bread ", "Ham", "Chips", "Thing", "Cheese"],
        "PROMOTION": [np.nan, 2, 0, 1, 0],
        "SKU": ["a", "b", "c", "d", "e"],
    })


class TestCleanData:
    def test_drops_invalid_rows(self):
        df = clean_data(raw_frame())
        # Negative retail price and NaN retail price rows removed.
        assert len(df) == 3
        assert (df["PRICE_RETAIL"] > 0).all()

    def test_strips_strings(self):
        df = clean_data(raw_frame())
        assert "Bakery" in df["DEPARTMENT"].values
        assert "Bread" in df["PRODUCT_NAME"].values

    def test_promotion_binarized(self):
        df = clean_data(raw_frame())
        assert set(df["PROMOTION"].unique()) <= {0, 1}

    def test_current_price_capped_at_retail(self):
        frame = raw_frame()
        frame.loc[1, "PRICE_CURRENT"] = 99.0
        df = clean_data(frame)
        assert (df["PRICE_CURRENT"] <= df["PRICE_RETAIL"]).all()

    def test_missing_promotion_column_defaults_to_zero(self):
        frame = raw_frame().drop(columns=["PROMOTION"])
        df = clean_data(frame)
        assert (df["PROMOTION"] == 0).all()


class TestEngineerFeatures:
    def setup_method(self):
        self.cfg = PricingConfig()
        self.df = engineer_features(
            clean_data(raw_frame()), self.cfg, np.random.default_rng(0)
        )

    def test_adds_expected_columns(self):
        for col in (
            "COST", "MAX_SHELF_LIFE", "DAYS_TO_EXPIRY", "URGENCY_RATIO",
            "INVENTORY_UNITS", "PRICE_RATIO",
        ):
            assert col in self.df.columns

    def test_cost_from_configured_ratio(self):
        expected = self.df["PRICE_RETAIL"] * self.cfg.cost_ratio_of_retail
        assert np.allclose(self.df["COST"], expected)

    def test_shelf_life_uses_department_map_with_default(self):
        by_dept = dict(zip(self.df["DEPARTMENT"], self.df["MAX_SHELF_LIFE"]))
        assert by_dept["Bakery"] == 7
        assert by_dept["Mystery"] == self.cfg.expiry.default_shelf_life_days

    def test_expiry_within_shelf_life(self):
        assert (self.df["DAYS_TO_EXPIRY"] >= 1).all()
        assert (self.df["DAYS_TO_EXPIRY"] <= self.df["MAX_SHELF_LIFE"]).all()

    def test_inventory_positive(self):
        assert (self.df["INVENTORY_UNITS"] >= self.cfg.inventory.min_units).all()

    def test_deterministic_given_seed(self):
        again = engineer_features(
            clean_data(raw_frame()), self.cfg, np.random.default_rng(0)
        )
        pd.testing.assert_frame_equal(self.df, again)
