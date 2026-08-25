"""Adversarial matrix over the adaptive system.

Crosses elasticity x inventory x expiry x replenishment x observed-sales
scenarios and checks INVARIANTS and economically sensible directionality —
never hard-coded answers:

- prices always respect the constraint envelope (no increases, bounded
  daily drops, positive);
- inventory can never go negative, and the units accounting identity
  holds every day and in aggregate;
- future deliveries are never sold before they arrive;
- observed sales (not forecasts) drive the state;
- the system is neither "always markdown" nor "never markdown" across the
  matrix, and reacts in the economically sensible direction to demand
  surprises.
"""

import itertools

import pytest

from selfshelf.closedloop import (
    forecast_deviation_environment,
    run_closed_loop,
)
from selfshelf.config import PricingConfig
from selfshelf.economics import ProductContext
from selfshelf.replenishment import ReplenishmentSchedule

CONFIG = PricingConfig()

ELASTICITIES = {
    "very_inelastic": -0.5,
    "moderate": -1.5,
    "highly_elastic": -3.0,
}
INVENTORIES = {"low": 15.0, "medium": 80.0, "high": 300.0}
EXPIRIES = {"near": 2.0, "medium": 6.0, "long": 12.0}
REPLENISHMENTS = {
    "none": None,
    "delayed": {4: 100.0},
    "imminent_large": {1: 400.0},
}
SALES_FACTORS = {"below": 0.4, "near": 1.0, "above": 1.5}

MATRIX = list(itertools.product(
    ELASTICITIES.items(), INVENTORIES.items(), EXPIRIES.items(),
    REPLENISHMENTS.items(), SALES_FACTORS.items(),
))

_EPS = 1e-6


def make_product(elasticity, inventory, days):
    return ProductContext(
        current_price=5.0, retail_price=5.0, unit_cost=2.5,
        inventory_units=inventory, days_to_expiry=days,
        baseline_daily_demand=15.0, elasticity=elasticity,
    )


def run_cell(elasticity, inventory, days, arrivals, factor):
    product = make_product(elasticity, inventory, days)
    schedule = (
        ReplenishmentSchedule(arrivals) if arrivals else None
    )
    env = forecast_deviation_environment(product, CONFIG, factor)
    return product, run_closed_loop(
        product, CONFIG, env, replenishment=schedule
    )


@pytest.fixture(scope="module")
def matrix_results():
    results = {}
    for (e_name, e), (i_name, inv), (x_name, days), (r_name, rep), (
        s_name, factor,
    ) in MATRIX:
        key = (e_name, i_name, x_name, r_name, s_name)
        results[key] = run_cell(e, inv, days, rep, factor)
    return results


class TestMatrixInvariants:
    def test_every_cell_respects_hard_invariants(self, matrix_results):
        max_drop = CONFIG.constraints.max_daily_price_drop
        for key, (product, result) in matrix_results.items():
            prices = result.acted_prices
            # Price constraints: positive, never above the starting price,
            # never rising, per-day drop bounded.
            assert all(p > 0 for p in prices), key
            assert all(
                p <= product.current_price + 0.005 for p in prices
            ), key
            for prev, nxt in zip(prices, prices[1:]):
                assert nxt <= prev + 0.005, (key, "price rose")
                assert nxt >= prev * (1.0 - max_drop) - 0.005, (
                    key, "daily drop limit broken"
                )
            # Inventory: never negative, daily accounting exact. On the
            # final shelf-life night the old lot expires: the difference
            # between flow and end inventory is that waste, and it may
            # appear on no other day.
            for r in result.records:
                assert r.start_inventory >= -_EPS, key
                assert r.observed_sales <= r.start_inventory + _EPS, (
                    key, "sold more than was on the shelf"
                )
                flow_end = (
                    r.start_inventory - r.observed_sales
                    + r.arrivals_applied
                )
                expired = flow_end - r.end_inventory
                assert expired >= -_EPS, (key, "inventory appeared")
                if r.days_to_expiry > 1.0 + _EPS:
                    assert expired == pytest.approx(0.0, abs=1e-6), (
                        key, "waste booked before expiry"
                    )
            o = result.outcome
            assert o["revenue"] >= 0 and o["waste_units"] >= -_EPS, key
            assert o["units_sold"] >= -_EPS, key
            # Units identity: supplied = sold + terminal (waste subset of
            # terminal); waste can only come from the original lot.
            supplied = product.inventory_units + o["replenishment_received"]
            assert o["units_sold"] + o["terminal_inventory"] == (
                pytest.approx(supplied, abs=1e-6)
            ), key
            assert o["waste_units"] <= product.inventory_units + _EPS, key

    def test_future_deliveries_never_sold_early(self, matrix_results):
        for key, (product, result) in matrix_results.items():
            r_name = key[3]
            if r_name == "none":
                assert result.outcome["replenishment_received"] == 0.0
                continue
            arrival_day = min(REPLENISHMENTS[r_name])
            sold_before = sum(
                r.observed_sales
                for r in result.records[:arrival_day]
            )
            assert sold_before <= product.inventory_units + _EPS, key

    def test_replenishment_is_actually_received_in_time(
        self, matrix_results
    ):
        for key, (product, result) in matrix_results.items():
            r_name, x_name = key[3], key[2]
            if r_name == "none":
                continue
            episode_days = len(result.records)
            expected = sum(
                qty for day, qty in REPLENISHMENTS[r_name].items()
                if day <= episode_days
            )
            assert result.outcome["replenishment_received"] == (
                pytest.approx(expected)
            ), key


class TestMatrixDirectionality:
    def test_not_always_markdown(self, matrix_results):
        """Scarce stock + healthy demand must include cells that never
        touch the price."""
        holds = [
            key for key, (product, result) in matrix_results.items()
            if key[1] == "low" and key[4] in ("near", "above")
            and all(
                p >= product.current_price - 0.005
                for p in result.acted_prices
            )
        ]
        assert holds, "the system marked down even scarce healthy stock"

    def test_not_never_markdown(self, matrix_results):
        """Excess elastic stock near expiry with weak demand must include
        cells that do mark down."""
        markdowns = [
            key for key, (product, result) in matrix_results.items()
            if key[0] == "highly_elastic" and key[1] == "high"
            and key[4] == "below"
            and any(
                p < product.current_price - 0.005
                for p in result.acted_prices
            )
        ]
        assert markdowns, "the system never marked down distressed stock"

    def test_observed_sales_move_prices_in_the_sensible_direction(
        self, matrix_results
    ):
        """For the same product, weaker observed demand must never lead to
        HIGHER eventual prices than stronger observed demand."""
        compared = 0
        for e_name in ELASTICITIES:
            for i_name in ("medium", "high"):
                for x_name in ("medium", "long"):
                    below = matrix_results[
                        (e_name, i_name, x_name, "none", "below")
                    ][1]
                    above = matrix_results[
                        (e_name, i_name, x_name, "none", "above")
                    ][1]
                    assert min(below.acted_prices) <= (
                        min(above.acted_prices) + 0.005
                    ), (e_name, i_name, x_name)
                    compared += 1
        assert compared == 12

    def test_shortfalls_trigger_replanning_where_it_matters(
        self, matrix_results
    ):
        """Somewhere in the below-forecast, ample-stock cells the plan
        must actually change — feedback is not decorative."""
        replans = sum(
            result.replans
            for key, (_, result) in matrix_results.items()
            if key[4] == "below" and key[1] in ("medium", "high")
        )
        assert replans > 0

    def test_matched_forecasts_do_not_thrash_hold_plans(
        self, matrix_results
    ):
        """When reality matches beliefs exactly and the plan was to hold,
        re-optimizing daily must not invent price moves. (Markdown-active
        plans may legitimately refine by cents as the expiry-blended floor
        deepens day by day — that is adaptation, not thrash.)"""
        checked = 0
        for key, (product, result) in matrix_results.items():
            if key[4] != "near" or key[3] != "none":
                continue
            plan_was_hold = all(
                p >= product.current_price - 0.005
                for p in result.initial_plan
            )
            if plan_was_hold:
                assert result.replans == 0, key
                checked += 1
        assert checked > 0

    def test_surprises_cause_more_replanning_than_no_surprises(
        self, matrix_results
    ):
        totals = {name: 0 for name in SALES_FACTORS}
        for key, (_, result) in matrix_results.items():
            if key[3] == "none":
                totals[key[4]] += result.replans
        assert totals["below"] >= totals["near"]
