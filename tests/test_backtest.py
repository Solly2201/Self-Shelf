"""Tests for the counterfactual backtest against the synthetic simulator."""

import numpy as np
import pandas as pd
import pytest

from selfshelf.backtest import _simulate_sell_down, backtest_recommendations
from selfshelf.config import PricingConfig


def make_item(**overrides) -> pd.Series:
    defaults = {
        "DEPARTMENT": "Bakery",
        "PRICE_RETAIL": 5.00,
        "PRICE_CURRENT": 5.00,
        "PROMOTION": 0,
        "Season": "Spring",
        "DAYS_TO_EXPIRY": 5,
        "INVENTORY_UNITS": 200,
        "COST": 3.50,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def flat_noise(days: int = 200) -> np.ndarray:
    return np.ones(days)


class TestSimulateSellDown:
    def test_units_sold_never_exceed_inventory(self):
        row = make_item(INVENTORY_UNITS=100, DAYS_TO_EXPIRY=30)
        out = _simulate_sell_down(4.00, row, PricingConfig(), flat_noise())
        assert out["units_sold"] <= 100 + 1e-9
        assert out["units_sold"] + out["waste_units"] == pytest.approx(100)

    def test_lower_price_sells_more_and_wastes_less(self):
        cfg = PricingConfig()
        row = make_item(INVENTORY_UNITS=500, DAYS_TO_EXPIRY=4)
        hold = _simulate_sell_down(5.00, row, cfg, flat_noise())
        cut = _simulate_sell_down(3.50, row, cfg, flat_noise())
        assert cut["units_sold"] > hold["units_sold"]
        assert cut["waste_units"] < hold["waste_units"]
        assert cut["sell_through"] > hold["sell_through"]

    def test_outputs_never_negative(self):
        cfg = PricingConfig()
        for price in (0.5, 2.0, 5.0):
            out = _simulate_sell_down(
                price, make_item(INVENTORY_UNITS=50), cfg, flat_noise()
            )
            assert out["revenue"] >= 0
            assert out["waste_units"] >= 0
            assert out["units_sold"] >= 0
            assert 0.0 <= out["sell_through"] <= 1.0 + 1e-9

    def test_zero_inventory_is_safe(self):
        out = _simulate_sell_down(
            4.0, make_item(INVENTORY_UNITS=0), PricingConfig(), flat_noise()
        )
        assert out["units_sold"] == 0.0
        assert out["sell_through"] == pytest.approx(1.0)

    def test_gross_profit_accounting(self):
        # Everything sells: GP is exactly margin x units.
        cfg = PricingConfig()
        row = make_item(INVENTORY_UNITS=10, DAYS_TO_EXPIRY=30)
        out = _simulate_sell_down(5.00, row, cfg, flat_noise())
        assert out["units_sold"] == pytest.approx(10)
        assert out["gross_profit"] == pytest.approx((5.00 - 3.50) * 10)


class TestBacktestRecommendations:
    def _frames(self):
        items = pd.DataFrame([
            make_item(DAYS_TO_EXPIRY=3, INVENTORY_UNITS=400),
            make_item(DAYS_TO_EXPIRY=30, INVENTORY_UNITS=60),
        ])
        recs = pd.DataFrame([
            {"Recommended_Price": 3.60, "Action": "Markdown",
             "Markdown_Percentage": 28.0},
            {"Recommended_Price": 5.00, "Action": "Price Maintained",
             "Markdown_Percentage": 0.0},
        ])
        return items, recs

    def test_summary_shape_and_labeling(self):
        items, recs = self._frames()
        summary = backtest_recommendations(items, recs, PricingConfig())
        assert "SYNTHETIC" in summary["label"]
        assert summary["n_products"] == 2
        assert summary["n_markdowns"] == 1
        for strategy in ("hold", "recommended"):
            for key in ("revenue", "gross_profit", "units_sold",
                        "waste_units", "sell_through", "cash_recovered"):
                assert key in summary[strategy]

    def test_deterministic_given_config_seed(self):
        items, recs = self._frames()
        a = backtest_recommendations(items, recs, PricingConfig())
        b = backtest_recommendations(items, recs, PricingConfig())
        assert a == b

    def test_markdown_on_overstocked_stock_reduces_waste(self):
        items, recs = self._frames()
        summary = backtest_recommendations(items, recs, PricingConfig())
        assert (
            summary["recommended"]["waste_units"]
            < summary["hold"]["waste_units"]
        )
