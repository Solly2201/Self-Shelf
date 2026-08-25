"""Multi-period backtest tests: parity with the frozen single-price replay,
common-random-numbers fairness, and accounting identities."""

import numpy as np
import pandas as pd
import pytest

from selfshelf.backtest import _simulate_sell_down, backtest_recommendations
from selfshelf.config import PricingConfig
from selfshelf.pathbacktest import (
    _simulate_path_sell_down,
    backtest_price_paths,
)


def make_row(
    price_current=5.0, price_retail=5.0, cost=3.5, inventory=80,
    days=6, department="Bakery", promotion=0,
):
    return pd.Series({
        "PRICE_CURRENT": price_current,
        "PRICE_RETAIL": price_retail,
        "COST": cost,
        "INVENTORY_UNITS": inventory,
        "DAYS_TO_EXPIRY": days,
        "DEPARTMENT": department,
        "PROMOTION": promotion,
    })


@pytest.fixture
def config():
    return PricingConfig()


def make_noise(config, days, seed=123):
    rng = np.random.default_rng(seed)
    return rng.lognormal(0.0, config.simulator.noise_sigma, size=days)


class TestParityWithFrozenBacktest:
    """A constant path must reproduce the audited single-price replay
    EXACTLY — this pins the path replay to the frozen accounting."""

    @pytest.mark.parametrize("price", [5.0, 4.2, 3.0])
    @pytest.mark.parametrize("days, inventory", [(6, 80), (3, 25), (12, 150)])
    def test_constant_path_matches_single_price(
        self, config, price, days, inventory
    ):
        row = make_row(days=days, inventory=inventory)
        noise = make_noise(config, days)
        single = _simulate_sell_down(price, row, config, noise)
        path = _simulate_path_sell_down([price], row, config, noise)
        for key, value in single.items():
            assert path[key] == pytest.approx(value, abs=1e-12), key

    def test_hold_totals_match_frozen_backtest(self, config):
        # backtest_price_paths' hold strategy must equal the frozen
        # backtest_recommendations' hold strategy on identical inputs.
        items = pd.DataFrame([
            make_row(days=5, inventory=60),
            make_row(days=8, inventory=100, department="Beverages"),
            make_row(days=3, inventory=40, department="Deli"),
        ])
        recs = pd.DataFrame({
            "Recommended_Price": [4.5, 4.0, 3.8],
            "Action": ["Markdown"] * 3,
            "Markdown_Percentage": [10.0, 20.0, 24.0],
        })
        frozen = backtest_recommendations(items, recs, config)
        paths = backtest_price_paths(
            items,
            immediate_prices=[4.5, 4.0, 3.8],
            daily_paths=[[4.5], [4.0], [3.8]],
            config=config,
        )
        for key, value in frozen["hold"].items():
            assert paths["hold"][key] == pytest.approx(value), key
        # And a constant "path" at the recommended price equals the frozen
        # "recommended" strategy.
        for key, value in frozen["recommended"].items():
            assert paths["path"][key] == pytest.approx(value), key
            assert paths["immediate"][key] == pytest.approx(value), key


class TestPathReplayBehavior:
    def test_staged_path_prices_take_effect_by_day(self, config):
        row = make_row(days=4, inventory=1000)  # never sells out
        noise = np.ones(4)
        staged = _simulate_path_sell_down(
            [5.0, 5.0, 4.0, 4.0], row, config, noise
        )
        hold = _simulate_path_sell_down([5.0], row, config, noise)
        low = _simulate_path_sell_down([4.0], row, config, noise)
        # Staged sells more units than holding, fewer than the deep cut.
        assert hold["units_sold"] < staged["units_sold"] < low["units_sold"]

    def test_short_path_extends_final_price(self, config):
        row = make_row(days=6, inventory=1000)
        noise = np.ones(6)
        explicit = _simulate_path_sell_down(
            [5.0, 4.0, 4.0, 4.0, 4.0, 4.0], row, config, noise
        )
        implicit = _simulate_path_sell_down([5.0, 4.0], row, config, noise)
        for key, value in explicit.items():
            assert implicit[key] == pytest.approx(value), key

    def test_units_accounting_identity(self, config):
        row = make_row(days=5, inventory=70)
        noise = make_noise(config, 5)
        out = _simulate_path_sell_down([5.0, 4.5, 4.0], row, config, noise)
        assert out["units_sold"] + out["terminal_inventory"] == (
            pytest.approx(float(row["INVENTORY_UNITS"]))
        )
        assert out["waste_units"] == pytest.approx(
            out["terminal_inventory"]
        )

    def test_common_random_numbers_are_deterministic(self, config):
        items = pd.DataFrame([make_row(), make_row(inventory=120)])
        args = dict(
            immediate_prices=[4.5, 4.0],
            daily_paths=[[5.0, 4.5], [5.0, 5.0, 4.0]],
            config=config,
        )
        a = backtest_price_paths(items, **args)
        b = backtest_price_paths(items, **args)
        assert a == b

    def test_empty_items(self, config):
        items = pd.DataFrame(
            columns=[
                "PRICE_CURRENT", "PRICE_RETAIL", "COST", "INVENTORY_UNITS",
                "DAYS_TO_EXPIRY", "DEPARTMENT", "PROMOTION",
            ]
        )
        result = backtest_price_paths(items, [], [], config)
        assert result["n_products"] == 0
        assert result["hold"]["revenue"] == 0.0

    def test_synthetic_label_present(self, config):
        items = pd.DataFrame([make_row()])
        result = backtest_price_paths(items, [4.5], [[4.5]], config)
        assert "SYNTHETIC SIMULATION" in result["label"]
