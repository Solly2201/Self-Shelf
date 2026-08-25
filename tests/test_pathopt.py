"""Multi-period path optimizer tests.

The optimizer is pure enumeration over the frozen path evaluator, so these
tests focus on: agreement with an independent brute-force reference,
constraint feasibility, economic behavior (wait-vs-markdown, expiry,
inventory, elasticity), accounting decomposition, and determinism.
"""

import itertools
import math

import pytest

from selfshelf.config import PricingConfig
from selfshelf.economics import (
    EconomicObjective,
    ProductContext,
    evaluate_price_path,
    price_bounds,
)
from selfshelf.pathopt import (
    candidate_price_levels,
    daily_to_schedule,
    optimize_path,
    path_horizon_days,
    path_price_floor,
    path_trajectory,
    schedule_to_daily,
    validate_daily_prices,
)


def make_product(
    current_price=5.0,
    retail_price=5.0,
    unit_cost=2.5,
    inventory_units=60,
    days_to_expiry=6,
    baseline_daily_demand=8.0,
    elasticity=-2.5,
):
    return ProductContext(
        current_price=current_price,
        retail_price=retail_price,
        unit_cost=unit_cost,
        inventory_units=inventory_units,
        days_to_expiry=days_to_expiry,
        baseline_daily_demand=baseline_daily_demand,
        elasticity=elasticity,
    )


@pytest.fixture
def config():
    return PricingConfig()


# ---------------------------------------------------------------------------
# Consistency with the frozen one-shot objective
# ---------------------------------------------------------------------------

class TestObjectiveConsistency:
    def test_hold_baseline_reproduces_one_shot_breakdown(self, config):
        product = make_product()
        result = optimize_path(product, config)
        one_shot = EconomicObjective(product, config).breakdown(
            product.current_price
        )
        assert result.hold["score"] == pytest.approx(one_shot["score"])
        assert result.hold["gross_profit"] == pytest.approx(
            one_shot["gross_profit"]
        )
        assert result.hold["expected_waste_units"] == pytest.approx(
            one_shot["expected_waste_units"]
        )

    def test_single_baseline_reproduces_one_shot_breakdown(self, config):
        product = make_product()
        result = optimize_path(product, config, single_price=4.0)
        one_shot = EconomicObjective(product, config).breakdown(4.0)
        assert result.single["score"] == pytest.approx(one_shot["score"])

    def test_one_day_horizon_only_constant_paths(self, config):
        product = make_product(days_to_expiry=1)
        result = optimize_path(product, config)
        assert result.horizon_days == 1
        assert len(result.daily_prices) == 1
        # Best constant price must match an explicit scan of the levels.
        levels = candidate_price_levels(product, config)
        objective = EconomicObjective(product, config)
        feasible = [
            p for p in levels
            if p >= product.current_price
            * (1 - config.constraints.max_daily_price_drop) - 1e-9
        ]
        best = max(feasible, key=objective.score)
        best_score = objective.score(best)
        assert result.evaluation["score"] == pytest.approx(best_score)


# ---------------------------------------------------------------------------
# Reference (brute force) validation
# ---------------------------------------------------------------------------

def brute_force_best(product, config, horizon, levels, max_moves):
    """Independent reference: enumerate ALL daily price sequences over the
    level grid that are non-increasing, respect the per-day drop limit,
    and use at most ``max_moves`` distinct below-current levels."""
    cur = product.current_price
    max_drop = config.constraints.max_daily_price_drop
    best_score = -math.inf
    best_seq = None
    for seq in itertools.product(levels, repeat=horizon):
        ok = True
        prev = cur
        for price in seq:
            if price > prev + 1e-9:
                ok = False
                break
            if price < prev * (1 - max_drop) - 1e-9:
                ok = False
                break
            prev = price
        if not ok:
            continue
        below = {p for p in seq if p < cur - 1e-9}
        if len(below) > max_moves:
            continue
        schedule = daily_to_schedule(seq)
        score = evaluate_price_path(product, config, schedule)["score"]
        if score > best_score + 1e-9:
            best_score = score
            best_seq = seq
    return best_seq, best_score


class TestReferenceValidation:
    @pytest.mark.parametrize("days_to_expiry, inventory, elasticity", [
        (4, 60, -2.5),
        (3, 90, -1.8),
        (4, 25, -1.2),
        (2, 45, -3.0),
    ])
    def test_matches_brute_force_enumeration(
        self, config, days_to_expiry, inventory, elasticity
    ):
        product = make_product(
            days_to_expiry=days_to_expiry,
            inventory_units=inventory,
            elasticity=elasticity,
        )
        horizon = path_horizon_days(product, config)
        levels = candidate_price_levels(product, config, num_levels=5)
        _, ref_score = brute_force_best(
            product, config, horizon, levels, max_moves=2
        )
        result = optimize_path(product, config, num_levels=5)
        assert result.evaluation["score"] == pytest.approx(
            ref_score, abs=1e-6
        )

    def test_never_worse_than_hold(self, config):
        for days in (1, 2, 5, 9, 14, 30):
            for inv in (5, 40, 120):
                product = make_product(
                    days_to_expiry=days, inventory_units=inv
                )
                result = optimize_path(product, config)
                assert (
                    result.evaluation["score"]
                    >= result.hold["score"] - 1e-9
                )


# ---------------------------------------------------------------------------
# Economic behavior
# ---------------------------------------------------------------------------

class TestEconomicBehavior:
    def test_healthy_product_holds(self, config):
        # Small stock, long shelf life: sells through comfortably.
        product = make_product(
            inventory_units=10, days_to_expiry=30, baseline_daily_demand=8.0
        )
        result = optimize_path(product, config)
        assert result.action == "Hold"
        assert all(
            p == product.current_price for p in result.daily_prices
        )

    def test_wait_then_markdown_beats_immediate_markdown(self, config):
        # Overstocked and elastic, but demand at the current price is real:
        # early full-price sales are worth capturing before discounting.
        product = make_product(
            current_price=5.0, unit_cost=2.5, inventory_units=55,
            days_to_expiry=6, baseline_daily_demand=8.0, elasticity=-2.5,
        )
        horizon = path_horizon_days(product, config)
        levels = candidate_price_levels(product, config)
        objective = EconomicObjective(product, config)
        feasible = [
            p for p in levels
            if p >= product.current_price
            * (1 - config.constraints.max_daily_price_drop) - 1e-9
        ]
        best_single = max(feasible, key=objective.score)

        result = optimize_path(product, config, single_price=best_single)
        assert result.action == "Staged markdown"
        # Starts by holding, then marks down later.
        assert result.daily_prices[0] == product.current_price
        assert result.daily_prices[-1] < product.current_price
        assert result.improvement_vs_single > 0
        assert result.improvement_vs_hold > 0

    def test_nearer_expiry_lowers_the_path(self, config):
        far = make_product(days_to_expiry=10, inventory_units=80)
        near = make_product(days_to_expiry=3, inventory_units=80)
        far_path = optimize_path(far, config)
        near_path = optimize_path(near, config)
        far_avg = sum(far_path.daily_prices) / len(far_path.daily_prices)
        near_avg = sum(near_path.daily_prices) / len(near_path.daily_prices)
        assert near_avg < far_avg

    def test_larger_inventory_lowers_the_path(self, config):
        small = make_product(inventory_units=20)
        large = make_product(inventory_units=120)
        small_avg = _avg_price(optimize_path(small, config))
        large_avg = _avg_price(optimize_path(large, config))
        assert large_avg < small_avg

    def test_elastic_product_marks_down_deeper_than_inelastic(self, config):
        elastic = make_product(elasticity=-3.0, inventory_units=80)
        inelastic = make_product(elasticity=-0.4, inventory_units=80)
        elastic_path = optimize_path(elastic, config)
        inelastic_path = optimize_path(inelastic, config)
        assert _avg_price(elastic_path) < _avg_price(inelastic_path)

    def test_inelastic_near_expiry_can_hold(self, config):
        # Markdown cannot recover enough demand: the engine should not
        # discount just because expiry is near.
        product = make_product(
            elasticity=-0.2, inventory_units=200,
            baseline_daily_demand=2.0, days_to_expiry=3,
        )
        result = optimize_path(product, config)
        # Whatever the decision, it must strictly beat holding to act.
        if result.action != "Hold":
            assert result.improvement_vs_hold > 0

    def test_reasons_are_populated(self, config):
        product = make_product(inventory_units=90, days_to_expiry=5)
        result = optimize_path(product, config)
        assert result.reasons
        if result.action != "Hold":
            assert any("schedule" in r for r in result.reasons)


def _avg_price(result):
    return sum(result.daily_prices) / len(result.daily_prices)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    @pytest.mark.parametrize("days, inv, elasticity, cost", [
        (14, 150, -2.8, 2.5),
        (5, 90, -1.6, 4.0),
        (2, 30, -1.0, 1.0),
        (30, 200, -3.0, 3.2),
    ])
    def test_path_respects_all_constraints(
        self, config, days, inv, elasticity, cost
    ):
        product = make_product(
            days_to_expiry=days, inventory_units=inv,
            elasticity=elasticity, unit_cost=cost,
        )
        result = optimize_path(product, config)
        floor = path_price_floor(product, config)
        max_drop = config.constraints.max_daily_price_drop
        prev = product.current_price
        for price in result.daily_prices:
            assert price <= product.current_price + 1e-9
            assert price >= floor - 0.01  # whole-cent ceiling above floor
            assert price <= prev + 1e-9  # never moves back up
            assert price >= prev * (1 - max_drop) - 1e-9
            prev = price

    def test_floor_matches_price_bounds_when_drop_not_binding(self, config):
        # When the one-day drop limit is not the binding constraint,
        # price_bounds' lower bound IS the blended floor — the two
        # formulations must agree exactly.
        product = make_product(
            current_price=3.0, retail_price=3.0, unit_cost=2.5,
            days_to_expiry=20,
        )
        lower, _ = price_bounds(product, config)
        assert path_price_floor(product, config) == pytest.approx(lower)

    def test_degenerate_floor_above_current_collapses_to_hold(self, config):
        # Current price already below the margin floor: only holding is
        # feasible, mirroring price_bounds' collapse rule.
        product = make_product(
            current_price=2.0, retail_price=5.0, unit_cost=2.5,
            days_to_expiry=25,
        )
        result = optimize_path(product, config)
        assert result.action == "Hold"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_identical_runs_identical_paths(self, config):
        product = make_product(inventory_units=85, days_to_expiry=7)
        a = optimize_path(product, config)
        b = optimize_path(product, config)
        assert a.daily_prices == b.daily_prices
        assert a.evaluation == b.evaluation
        assert a.n_candidates == b.n_candidates


# ---------------------------------------------------------------------------
# Trajectory decomposition
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_daily_rows_reproduce_evaluator_totals(self):
        config = PricingConfig()
        config.inventory.holding_cost_per_unit_day = 0.02
        product = make_product(inventory_units=70, days_to_expiry=8)
        result = optimize_path(product, config)
        rows = path_trajectory(product, config, result.schedule)
        ev = evaluate_price_path(product, config, result.schedule)

        revenue = sum(r["expected_revenue"] for r in rows)
        holding = sum(r["holding_cost"] for r in rows)
        friction = sum(r["markdown_cost"] for r in rows)
        assert revenue == pytest.approx(ev["expected_revenue"], abs=1e-6)
        assert holding == pytest.approx(ev["holding_cost"], abs=1e-6)
        assert friction == pytest.approx(ev["markdown_cost"], abs=1e-6)
        assert rows[-1]["end_inventory"] == pytest.approx(
            ev["terminal_inventory"], abs=1e-6
        )
        assert rows[-1]["projected_waste_units"] == pytest.approx(
            ev["expected_waste_units"], abs=1e-6
        )

    def test_inventory_is_conserved_day_to_day(self, config):
        product = make_product(inventory_units=55, days_to_expiry=6)
        result = optimize_path(product, config)
        rows = path_trajectory(product, config, result.schedule)
        for prev, cur in zip(rows, rows[1:]):
            assert cur["start_inventory"] == pytest.approx(
                prev["end_inventory"]
            )
        total_sales = sum(r["expected_sales_units"] for r in rows)
        assert total_sales + rows[-1]["end_inventory"] == pytest.approx(
            product.inventory_units
        )


# ---------------------------------------------------------------------------
# Daily <-> schedule conversion and custom-path validation
# ---------------------------------------------------------------------------

class TestConversionAndValidation:
    def test_round_trip(self):
        daily = [5.0, 5.0, 4.5, 4.5, 4.5, 3.99]
        schedule = daily_to_schedule(daily)
        assert schedule == [(2, 5.0), (3, 4.5), (1, 3.99)]
        assert schedule_to_daily(schedule, 6) == daily

    def test_short_daily_list_is_extended(self):
        assert schedule_to_daily([(2, 4.0)], 5) == [4.0] * 5

    def test_valid_path_passes(self, config):
        product = make_product()
        horizon = path_horizon_days(product, config)
        prices, errors = validate_daily_prices(
            product, config, [5.0, 4.5, 4.0]
        )
        assert errors == []
        assert len(prices) == horizon
        assert prices[-1] == 4.0

    def test_price_increase_rejected(self, config):
        product = make_product()
        _, errors = validate_daily_prices(product, config, [4.0, 4.5])
        assert any("move back up" in e for e in errors)

    def test_below_floor_rejected(self, config):
        product = make_product()
        _, errors = validate_daily_prices(product, config, [5.0, 0.5])
        assert any("floor" in e or "daily price move" in e for e in errors)

    def test_too_deep_daily_drop_rejected(self, config):
        product = make_product(
            current_price=10.0, retail_price=10.0, unit_cost=1.0
        )
        _, errors = validate_daily_prices(product, config, [10.0, 4.0])
        assert any("maximum daily price move" in e for e in errors)

    def test_empty_path_rejected(self, config):
        product = make_product()
        _, errors = validate_daily_prices(product, config, [])
        assert errors
