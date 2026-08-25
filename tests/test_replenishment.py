"""Replenishment modeling: schedules, time-respecting inventory, parity
with the frozen path evaluator, and replenishment-aware optimization.

The load-bearing guarantees:

- with NO replenishment, every number is bit-identical to the frozen
  ``economics.evaluate_price_path`` / ``pathopt.optimize_path`` (Case C);
- future deliveries can never be sold before they arrive;
- expired stock is never resurrected as fresh stock;
- the accounting identity holds:
      initial inventory + arrivals = sales + terminal inventory
  over the planning window (projected waste is the at-expiry share of the
  old lot's terminal stock, same convention as the frozen engine).
"""

import math

import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.economics import ProductContext, evaluate_price_path
from selfshelf.pathopt import optimize_path, schedule_to_daily
from selfshelf.replenishment import (
    ReplenishmentSchedule,
    evaluate_price_path_with_replenishment,
    optimize_path_with_replenishment,
    replenishment_trajectory,
    schedules_from_events,
)


def make_product(
    inventory=100.0,
    days_to_expiry=5.0,
    current_price=5.0,
    baseline_daily_demand=10.0,
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


class TestReplenishmentSchedule:
    def test_day_zero_or_past_is_rejected(self):
        with pytest.raises(ValueError, match="already-arrived"):
            ReplenishmentSchedule({0: 50.0})
        with pytest.raises(ValueError, match="already-arrived"):
            ReplenishmentSchedule({-2: 50.0})

    def test_negative_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            ReplenishmentSchedule({2: -5.0})

    def test_zero_quantities_are_dropped(self):
        s = ReplenishmentSchedule({2: 0.0, 3: 10.0})
        assert s.arrivals == {3: 10.0}

    def test_totals_and_next_arrival(self):
        s = ReplenishmentSchedule({2: 500.0, 5: 100.0})
        assert s.total_units() == 600.0
        assert s.total_units(within_days=3) == 500.0
        assert s.next_arrival() == (2, 500.0)
        assert not s.is_empty
        assert ReplenishmentSchedule.empty().is_empty

    def test_advanced_shifts_and_drops_arrived_days(self):
        s = ReplenishmentSchedule({1: 40.0, 4: 60.0})
        after_one_day = s.advanced(1)
        # Day-1 delivery has now arrived: it is the caller's job to add it
        # to inventory; the remaining schedule only holds future events.
        assert after_one_day.arrivals == {3: 60.0}


class TestFrozenParity:
    """Case C: no replenishment data -> the frozen engine's numbers."""

    SCENARIOS = [
        (make_product(), [(5, 5.0)]),
        (make_product(), [(2, 5.0), (3, 4.0)]),
        (make_product(inventory=30, days_to_expiry=3), [(1, 5.0), (2, 3.5)]),
        (make_product(days_to_expiry=20, baseline_daily_demand=2.0),
         [(4, 5.0), (10, 4.5)]),
        (make_product(baseline_daily_demand=0.0), [(5, 4.0)]),
        (make_product(inventory=0.0), [(5, 5.0)]),
    ]

    def test_empty_schedule_reproduces_frozen_evaluator_exactly(self):
        for product, schedule in self.SCENARIOS:
            frozen = evaluate_price_path(product, CONFIG, schedule)
            aware = evaluate_price_path_with_replenishment(
                product, CONFIG, schedule, ReplenishmentSchedule.empty()
            )
            for key, value in frozen.items():
                assert aware[key] == pytest.approx(value, abs=1e-9), (
                    key, schedule
                )
            assert aware["replenishment_units"] == 0.0

    def test_empty_schedule_optimizer_matches_frozen_optimizer(self):
        for product in (
            make_product(),
            make_product(inventory=200, days_to_expiry=4),
            make_product(days_to_expiry=12, baseline_daily_demand=3.0),
        ):
            frozen = optimize_path(product, CONFIG, single_price=4.25)
            aware = optimize_path_with_replenishment(
                product, CONFIG, ReplenishmentSchedule.empty(),
                single_price=4.25,
            )
            assert aware.schedule == frozen.schedule
            assert aware.daily_prices == frozen.daily_prices
            assert aware.n_candidates == frozen.n_candidates
            assert aware.evaluation["score"] == pytest.approx(
                frozen.evaluation["score"], abs=1e-9
            )
            assert aware.hold["score"] == pytest.approx(
                frozen.hold["score"], abs=1e-9
            )


class TestTimeRespectingInventory:
    def test_future_stock_is_not_available_today(self):
        product = make_product(inventory=10, baseline_daily_demand=50.0)
        rows = replenishment_trajectory(
            product, CONFIG, [(5, 5.0)], ReplenishmentSchedule({3: 500.0})
        )
        # Days 0-2: only the 10 on-hand units exist; demand is 50/day but
        # sales are capped by what has actually arrived.
        assert rows[0]["start_inventory"] == 10.0
        assert rows[0]["available_inventory"] == 10.0
        assert sum(r["expected_sales_units"] for r in rows[:3]) <= 10.0 + 1e-9
        assert rows[3]["replenishment_received"] == 500.0
        assert rows[3]["available_inventory"] > rows[2]["end_inventory"]

    def test_sales_before_arrival_are_identical_regardless_of_arrival(self):
        product = make_product(inventory=40, baseline_daily_demand=30.0)
        early = replenishment_trajectory(
            product, CONFIG, [(5, 5.0)], ReplenishmentSchedule({4: 300.0})
        )
        never = replenishment_trajectory(
            product, CONFIG, [(5, 5.0)], ReplenishmentSchedule.empty()
        )
        for day in range(4):
            assert early[day]["expected_sales_units"] == pytest.approx(
                never[day]["expected_sales_units"]
            )

    def test_accounting_identity(self):
        product = make_product(inventory=60, baseline_daily_demand=25.0)
        schedule = ReplenishmentSchedule({1: 80.0, 3: 40.0})
        totals = evaluate_price_path_with_replenishment(
            product, CONFIG, [(2, 5.0), (3, 4.0)], schedule
        )
        rows = replenishment_trajectory(
            product, CONFIG, [(2, 5.0), (3, 4.0)], schedule
        )
        sold = sum(r["expected_sales_units"] for r in rows)
        assert (
            product.inventory_units + totals["replenishment_units"]
        ) == pytest.approx(sold + totals["terminal_inventory"], abs=1e-9)
        # Lot split is internally consistent too.
        assert totals["terminal_inventory"] == pytest.approx(
            totals["old_terminal_inventory"]
            + totals["fresh_terminal_inventory"],
            abs=1e-9,
        )

    def test_arrivals_beyond_the_window_are_ignored(self):
        product = make_product(days_to_expiry=3)  # window = 3 days
        inside = evaluate_price_path_with_replenishment(
            product, CONFIG, [(3, 5.0)], ReplenishmentSchedule({10: 999.0})
        )
        frozen = evaluate_price_path(product, CONFIG, [(3, 5.0)])
        assert inside["score"] == pytest.approx(frozen["score"], abs=1e-9)
        assert inside["replenishment_units"] == 0.0


class TestExpiryAndWaste:
    def test_fresh_stock_never_becomes_waste(self):
        # Old stock cannot sell through, big delivery tomorrow: projected
        # waste must be bounded by the OLD lot, never touching the fresh
        # delivery.
        product = make_product(
            inventory=100, days_to_expiry=2, baseline_daily_demand=5.0
        )
        totals = evaluate_price_path_with_replenishment(
            product, CONFIG, [(2, 5.0)], ReplenishmentSchedule({1: 500.0})
        )
        assert totals["expected_waste_units"] <= product.inventory_units
        assert totals["expected_waste_units"] == pytest.approx(
            totals["old_terminal_inventory"], abs=1e-9
        )  # expiry at the window end -> old remainder is the waste
        assert totals["fresh_terminal_inventory"] > 0

    def test_expiring_stock_sells_before_fresh_stock(self):
        # FIFO rotation: with demand below total stock, the old lot
        # depletes first.
        product = make_product(
            inventory=20, days_to_expiry=4, baseline_daily_demand=10.0
        )
        totals = evaluate_price_path_with_replenishment(
            product, CONFIG, [(4, 5.0)], ReplenishmentSchedule({1: 100.0})
        )
        # 4 days x 10/day = 40 sold; old lot (20) must be gone entirely.
        assert totals["old_terminal_inventory"] == pytest.approx(0.0, abs=1e-9)
        assert totals["expected_waste_units"] == pytest.approx(0.0, abs=1e-9)


class TestReplenishmentAwareOptimization:
    def test_aware_plan_dominates_naive_plan_under_replenishment(self):
        """The replenishment-aware optimizer's plan can never score worse
        than the ignore-replenishment plan when both are valued against
        the same replenishment-aware economics."""
        config = PricingConfig()
        config.inventory.holding_cost_per_unit_day = 0.02
        for schedule in (
            ReplenishmentSchedule({1: 400.0}),
            ReplenishmentSchedule({2: 150.0, 4: 150.0}),
        ):
            for product in (
                make_product(inventory=120, days_to_expiry=3,
                             baseline_daily_demand=15.0),
                make_product(inventory=80, days_to_expiry=6,
                             baseline_daily_demand=8.0),
            ):
                naive = optimize_path(product, config)
                aware = optimize_path_with_replenishment(
                    product, config, schedule
                )
                naive_under_truth = evaluate_price_path_with_replenishment(
                    product, config, naive.schedule, schedule
                )
                assert aware.evaluation["score"] >= (
                    naive_under_truth["score"] - 1e-9
                )

    def test_case_a_expiring_stock_with_big_shipment_tomorrow(self):
        """Excess expiring stock, large delivery tomorrow. The aware
        optimizer must still act on the expiring stock (its waste risk is
        unchanged), and its valuation must reflect the delivery."""
        product = make_product(
            inventory=150, days_to_expiry=3, baseline_daily_demand=12.0
        )
        schedule = ReplenishmentSchedule({1: 500.0})
        aware = optimize_path_with_replenishment(product, CONFIG, schedule)
        naive = optimize_path(product, CONFIG)
        # Waste pressure on the current lot exists either way: the aware
        # plan may not simply hold while stock expires.
        assert naive.action != "Hold"
        assert aware.action != "Hold"
        assert aware.evaluation["replenishment_units"] == 500.0
        # Prices stay within the same constraint envelope.
        floor_prices = [p for p in aware.daily_prices]
        assert all(p <= product.current_price + 1e-9 for p in floor_prices)

    def test_case_b_scarce_stock_delayed_replenishment_holds(self):
        """Little stock, strong demand, delivery far away: scarcity means
        no markdown — emergent from the economics, not a rule."""
        product = make_product(
            inventory=15, days_to_expiry=6, baseline_daily_demand=10.0
        )
        aware = optimize_path_with_replenishment(
            product, CONFIG, ReplenishmentSchedule({5: 300.0})
        )
        assert aware.action == "Hold"
        assert aware.daily_prices[0] == product.current_price

    def test_deterministic(self):
        product = make_product(inventory=90, days_to_expiry=4)
        schedule = ReplenishmentSchedule({2: 120.0})
        a = optimize_path_with_replenishment(product, CONFIG, schedule)
        b = optimize_path_with_replenishment(product, CONFIG, schedule)
        assert a.schedule == b.schedule
        assert a.evaluation == b.evaluation


class TestSchedulesFromEvents:
    def test_dates_convert_to_future_offsets(self):
        events = pd.DataFrame({
            "date": pd.to_datetime(
                ["2026-08-20", "2026-08-26", "2026-08-27", "2026-08-27"]
            ),
            "product_id": ["A1", "A1", "A1", "B2"],
            "quantity": [50.0, 100.0, 25.0, 75.0],
        })
        schedules = schedules_from_events(events, "2026-08-25")
        # 2026-08-20 is in the past -> excluded (already in inventory).
        assert schedules["A1"].arrivals == {1: 100.0, 2: 25.0}
        assert schedules["B2"].arrivals == {2: 75.0}

    def test_same_day_deliveries_are_not_future_events(self):
        events = pd.DataFrame({
            "date": pd.to_datetime(["2026-08-25"]),
            "product_id": ["A1"],
            "quantity": [10.0],
        })
        assert schedules_from_events(events, "2026-08-25") == {}

    def test_empty_events(self):
        assert schedules_from_events(None, "2026-08-25") == {}
        assert schedules_from_events(pd.DataFrame(), "2026-08-25") == {}
