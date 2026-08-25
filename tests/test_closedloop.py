"""Closed-loop re-optimization: state transitions, observed-vs-expected
feedback, daily replanning, determinism, leakage protection, and the
hold / open-loop / closed-loop backtest."""

import numpy as np
import pandas as pd
import pytest

from selfshelf.adaptivebacktest import (
    backtest_closed_loop,
    backtest_replenishment_awareness,
)
from selfshelf.closedloop import (
    RetailState,
    forecast_deviation_environment,
    initial_state,
    observe_and_advance,
    run_closed_loop,
    simulator_environment,
)
from selfshelf.config import PricingConfig
from selfshelf.economics import EconomicObjective, ProductContext
from selfshelf.pathbacktest import _simulate_path_sell_down
from selfshelf.replenishment import ReplenishmentSchedule


def make_product(
    inventory=100.0,
    days_to_expiry=5.0,
    current_price=5.0,
    baseline_daily_demand=20.0,
    elasticity=-1.8,
    unit_cost=2.5,
) -> ProductContext:
    return ProductContext(
        current_price=current_price,
        retail_price=current_price,
        unit_cost=unit_cost,
        inventory_units=inventory,
        days_to_expiry=days_to_expiry,
        baseline_daily_demand=baseline_daily_demand,
        elasticity=elasticity,
    )


CONFIG = PricingConfig()


class TestStateTransition:
    def test_inventory_follows_observed_not_forecast_sales(self):
        """THE closed-loop property: expected 20, actually sold 5 -> the
        next state holds inventory - 5, never inventory - 20."""
        state = initial_state(make_product(inventory=100))
        nxt = observe_and_advance(
            state, acted_price=5.0, observed_sales=5.0, forecast_sales=20.0,
            arrivals_tomorrow=0.0, elasticity=-1.8, config=CONFIG,
        )
        assert nxt.inventory == pytest.approx(95.0)
        assert nxt.last_observed_sales == 5.0
        assert nxt.last_forecast_sales == 20.0
        assert nxt.days_to_expiry == pytest.approx(4.0)
        assert nxt.day == 1

    def test_arrivals_land_in_the_fresh_lot(self):
        state = initial_state(make_product(inventory=100))
        nxt = observe_and_advance(
            state, 5.0, 10.0, 12.0, arrivals_tomorrow=500.0,
            elasticity=-1.8, config=CONFIG,
        )
        assert nxt.old_inventory == pytest.approx(90.0)
        assert nxt.fresh_inventory == pytest.approx(500.0)
        assert nxt.replenishment_received == pytest.approx(500.0)

    def test_expiring_lot_sells_before_fresh_lot(self):
        state = RetailState(
            day=3, old_inventory=10.0, fresh_inventory=50.0,
            current_price=5.0, days_to_expiry=2.0,
            baseline_daily_demand=20.0,
        )
        nxt = observe_and_advance(
            state, 5.0, 25.0, 20.0, 0.0, -1.8, CONFIG
        )
        assert nxt.old_inventory == pytest.approx(0.0)   # FIFO
        assert nxt.fresh_inventory == pytest.approx(35.0)

    def test_expiry_converts_old_lot_to_waste_not_fresh_stock(self):
        state = RetailState(
            day=4, old_inventory=30.0, fresh_inventory=40.0,
            current_price=5.0, days_to_expiry=1.0,
            baseline_daily_demand=20.0,
        )
        nxt = observe_and_advance(
            state, 5.0, 10.0, 10.0, 0.0, -1.8, CONFIG
        )
        # 20 old units left, shelf life ends tonight -> waste, removed.
        assert nxt.old_inventory == 0.0
        assert nxt.cumulative_waste == pytest.approx(20.0)
        assert nxt.fresh_inventory == pytest.approx(40.0)  # unharmed

    def test_belief_update_smooths_toward_observation(self):
        cfg = PricingConfig()
        cfg.adaptive.demand_learning_rate = 0.5
        state = initial_state(make_product(baseline_daily_demand=20.0))
        nxt = observe_and_advance(state, 5.0, 8.0, 20.0, 0.0, -1.8, cfg)
        # Same price acted -> prior 20, observation 8, rate 0.5 -> 14.
        assert nxt.baseline_daily_demand == pytest.approx(14.0)

    def test_stockout_censored_day_does_not_update_belief(self):
        state = initial_state(
            make_product(inventory=10, baseline_daily_demand=20.0)
        )
        nxt = observe_and_advance(state, 5.0, 10.0, 10.0, 0.0, -1.8, CONFIG)
        # Sold every unit on the shelf: demand >= 10 is all we learned;
        # the level belief must not be dragged down to 10.
        assert nxt.baseline_daily_demand == pytest.approx(20.0)

    def test_belief_reanchors_to_the_acted_price(self):
        cfg = PricingConfig()
        cfg.adaptive.update_demand_beliefs = False
        state = initial_state(
            make_product(baseline_daily_demand=20.0, elasticity=-2.0)
        )
        nxt = observe_and_advance(state, 4.0, 25.0, 31.0, 0.0, -2.0, cfg)
        # Anchor moves 5.00 -> 4.00: 20 * (4/5)^-2 = 31.25 at the new price.
        assert nxt.baseline_daily_demand == pytest.approx(31.25)
        assert nxt.current_price == 4.0

    def test_negative_observed_sales_rejected(self):
        state = initial_state(make_product())
        with pytest.raises(ValueError):
            observe_and_advance(state, 5.0, -1.0, 5.0, 0.0, -1.8, CONFIG)


class TestClosedLoopController:
    def test_surprise_shortfall_deepens_the_plan(self):
        """Scenario 1: forecast is high, reality is 25% of it. After
        observing the shortfall the controller must re-optimize onto a
        more aggressive (lower-priced) path than the day-0 plan."""
        product = make_product(
            inventory=120, days_to_expiry=5, baseline_daily_demand=25.0
        )
        env = forecast_deviation_environment(product, CONFIG, 0.25)
        result = run_closed_loop(product, CONFIG, env)

        assert result.replans >= 1
        # By the final day the acted price sits below what the day-0 plan
        # scheduled for that day (deeper markdown after bad news).
        day = len(result.records) - 1
        planned_then = result.initial_plan[
            min(day, len(result.initial_plan) - 1)
        ]
        assert result.records[day].acted_price < planned_then - 0.005
        # And each day's state fed the next plan: surprises are negative.
        assert all(r.surprise <= 1e-9 for r in result.records)

    def test_no_surprise_means_no_replanning(self):
        """When reality matches the forecast exactly, re-optimizing daily
        must not thrash: the plan stays the initial plan."""
        product = make_product(
            inventory=60, days_to_expiry=5, baseline_daily_demand=12.0
        )
        env = forecast_deviation_environment(product, CONFIG, 1.0)
        result = run_closed_loop(product, CONFIG, env)
        assert result.replans == 0
        assert result.acted_prices == result.initial_plan[
            :len(result.acted_prices)
        ]

    def test_determinism_same_inputs_identical_episode(self):
        product = make_product(inventory=90, days_to_expiry=4)
        rng = np.random.default_rng(11)
        noise = rng.lognormal(0.0, 0.15, size=6)
        env = forecast_deviation_environment(product, CONFIG, 0.6, noise)
        a = run_closed_loop(product, CONFIG, env)
        b = run_closed_loop(product, CONFIG, env)
        assert a.acted_prices == b.acted_prices
        assert a.outcome == b.outcome
        assert [r.as_dict() for r in a.records] == [
            r.as_dict() for r in b.records
        ]

    def test_different_noise_can_change_the_trajectory(self):
        product = make_product(inventory=90, days_to_expiry=4)
        noisy = np.random.default_rng(1).lognormal(0.0, 0.6, size=6)
        other = np.random.default_rng(2).lognormal(0.0, 0.6, size=6)
        a = run_closed_loop(
            product, CONFIG,
            forecast_deviation_environment(product, CONFIG, 1.0, noisy),
        )
        b = run_closed_loop(
            product, CONFIG,
            forecast_deviation_environment(product, CONFIG, 1.0, other),
        )
        assert a.outcome != b.outcome

    def test_future_demand_cannot_leak_into_todays_price(self):
        """Changing what happens from day 3 onward must not change any
        action taken on days 0-2."""
        product = make_product(
            inventory=120, days_to_expiry=6, baseline_daily_demand=25.0
        )

        def env_collapse_late(day, price, days_left):
            base = forecast_deviation_environment(product, CONFIG, 1.0)
            return base(day, price, days_left) * (0.05 if day >= 3 else 1.0)

        def env_normal(day, price, days_left):
            base = forecast_deviation_environment(product, CONFIG, 1.0)
            return base(day, price, days_left)

        a = run_closed_loop(product, CONFIG, env_collapse_late)
        b = run_closed_loop(product, CONFIG, env_normal)
        assert a.acted_prices[:3] == b.acted_prices[:3]
        # After the collapse is OBSERVED, trajectories may diverge.

    def test_accounting_identity_with_replenishment(self):
        product = make_product(
            inventory=80, days_to_expiry=5, baseline_daily_demand=15.0
        )
        env = forecast_deviation_environment(product, CONFIG, 0.8)
        schedule = ReplenishmentSchedule({2: 60.0})
        result = run_closed_loop(
            product, CONFIG, env, replenishment=schedule
        )
        o = result.outcome
        supplied = product.inventory_units + o["replenishment_received"]
        # Frozen convention: terminal stock includes what expired, so
        # units_sold + terminal = supplied, and waste <= terminal.
        assert o["units_sold"] + o["terminal_inventory"] == pytest.approx(
            supplied, abs=1e-6
        )
        assert o["waste_units"] <= o["terminal_inventory"] + 1e-9
        assert o["replenishment_received"] == 60.0
        assert o["revenue"] >= 0 and o["waste_units"] >= 0

    def test_arrivals_are_sellable_only_after_they_arrive(self):
        product = make_product(
            inventory=10, days_to_expiry=5, baseline_daily_demand=40.0
        )
        env = forecast_deviation_environment(product, CONFIG, 1.0)
        result = run_closed_loop(
            product, CONFIG, env,
            replenishment=ReplenishmentSchedule({3: 200.0}),
        )
        sold_before = sum(
            r.observed_sales for r in result.records[:3]
        )
        assert sold_before <= 10.0 + 1e-9
        assert result.records[2].arrivals_applied == 200.0  # lands day 3
        assert result.records[3].start_inventory >= 200.0

    def test_fixed_hold_path_matches_frozen_backtest_replay(self):
        """Bit-parity: the closed-loop machinery with a fixed hold path
        must reproduce the frozen pathbacktest replay exactly."""
        config = PricingConfig()
        row = pd.Series({
            "PRICE_CURRENT": 5.0, "PRICE_RETAIL": 5.0, "COST": 3.5,
            "INVENTORY_UNITS": 80, "DAYS_TO_EXPIRY": 6,
            "DEPARTMENT": "Bakery", "PROMOTION": 0,
        })
        noise = np.random.default_rng(123).lognormal(
            0.0, config.simulator.noise_sigma, size=6
        )
        frozen = _simulate_path_sell_down([5.0], row, config, noise)
        env = simulator_environment(
            base_demand=config.simulator.base_daily_demand,
            retail_price=5.0,
            true_elasticity=config.elasticity.for_department("Bakery"),
            config=config,
            noise=noise,
        )
        product = make_product(
            inventory=80, days_to_expiry=6, unit_cost=3.5,
            baseline_daily_demand=20.0,
        )
        mine = run_closed_loop(
            product, config, env, fixed_daily_prices=[5.0] * 6
        )
        for key in ("revenue", "units_sold", "waste_units",
                    "terminal_inventory", "holding_cost", "gross_profit",
                    "economic_value", "cash_recovered", "sell_through"):
            assert mine.outcome[key] == pytest.approx(
                frozen[key], abs=1e-9
            ), key


def small_items():
    rows = []
    for days, inventory, dept in (
        (4, 90, "Bakery"), (5, 40, "Deli"), (3, 150, "Bakery"),
    ):
        rows.append({
            "PRICE_CURRENT": 5.0, "PRICE_RETAIL": 5.0, "COST": 3.0,
            "INVENTORY_UNITS": inventory, "DAYS_TO_EXPIRY": days,
            "DEPARTMENT": dept, "PROMOTION": 0,
        })
    return pd.DataFrame(rows)


def contexts_for(items, baseline=18.0, elasticity=-1.6):
    return [
        ProductContext(
            current_price=float(r["PRICE_CURRENT"]),
            retail_price=float(r["PRICE_RETAIL"]),
            unit_cost=float(r["COST"]),
            inventory_units=float(r["INVENTORY_UNITS"]),
            days_to_expiry=float(r["DAYS_TO_EXPIRY"]),
            baseline_daily_demand=baseline,
            elasticity=elasticity,
        )
        for _, r in items.iterrows()
    ]


class TestClosedLoopBacktest:
    def test_strategies_present_and_deterministic(self):
        items = small_items()
        contexts = contexts_for(items)
        a = backtest_closed_loop(items, contexts, CONFIG)
        b = backtest_closed_loop(items, contexts, CONFIG)
        assert a == b
        assert a["n_products"] == 3
        assert "SYNTHETIC SIMULATION" in a["label"]
        for strategy in ("hold", "open_loop", "closed_loop"):
            assert a[strategy]["revenue"] >= 0.0
            assert a[strategy]["waste_units"] >= 0.0

    def test_hold_matches_pathbacktest_hold(self):
        from selfshelf.pathbacktest import backtest_price_paths

        items = small_items()
        contexts = contexts_for(items)
        mine = backtest_closed_loop(items, contexts, CONFIG)
        frozen = backtest_price_paths(
            items,
            immediate_prices=[5.0] * len(items),
            daily_paths=[[5.0]] * len(items),
            config=CONFIG,
        )
        for key, value in frozen["hold"].items():
            assert mine["hold"][key] == pytest.approx(value, abs=0.01), key

    def test_feedback_pays_when_demand_disappoints(self):
        """Directional, not hard-coded: when reality delivers far less
        demand than believed, daily re-optimization must waste less and
        recover more value than executing the day-0 plan blind."""
        product = ProductContext(
            current_price=5.0, retail_price=5.0, unit_cost=2.5,
            inventory_units=120, days_to_expiry=5,
            baseline_daily_demand=25.0, elasticity=-1.8,
        )
        env = forecast_deviation_environment(product, CONFIG, 0.4)
        open_loop = run_closed_loop(product, CONFIG, env, reoptimize=False)
        closed = run_closed_loop(product, CONFIG, env, reoptimize=True)
        assert closed.outcome["waste_units"] < (
            open_loop.outcome["waste_units"]
        )
        assert closed.outcome["economic_value"] > (
            open_loop.outcome["economic_value"]
        )

    def test_max_products_caps_the_run(self):
        items = small_items()
        contexts = contexts_for(items)
        capped = backtest_closed_loop(
            items, contexts, CONFIG, max_products=2
        )
        assert capped["n_products"] == 2


class TestReplenishmentBacktest:
    def test_naive_vs_aware_comparison_structure(self):
        product = make_product(
            inventory=140, days_to_expiry=4, baseline_daily_demand=15.0
        )
        env = forecast_deviation_environment(product, CONFIG, 1.0)
        schedule = ReplenishmentSchedule({1: 300.0})
        result = backtest_replenishment_awareness(
            product, CONFIG, env, schedule
        )
        assert "SYNTHETIC SIMULATION" in result["label"]
        for strategy in ("naive", "aware"):
            s = result[strategy]
            assert s["revenue"] >= 0
            supplied = product.inventory_units + 300.0
            assert s["units_sold"] + s["terminal_inventory"] == (
                pytest.approx(supplied, abs=0.01)
            )
            assert s["waste_units"] <= s["terminal_inventory"] + 1e-9

    def test_deterministic(self):
        product = make_product(inventory=100, days_to_expiry=4)
        env = forecast_deviation_environment(product, CONFIG, 0.9)
        schedule = ReplenishmentSchedule({2: 100.0})
        a = backtest_replenishment_awareness(product, CONFIG, env, schedule)
        b = backtest_replenishment_awareness(product, CONFIG, env, schedule)
        assert a == b
