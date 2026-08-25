"""Markdown-economics validation tests.

These encode the core retail markdown principles: a markdown must create
more total economic value than keeping the current price; volume uplift
must be weighed against sacrificed unit margin (break-even uplift); waste,
holding cost and terminal stock — not expiry proximity by itself — drive
clearance; and the deepest discount is not automatically the best one.
"""

import math

import numpy as np
import pytest

from selfshelf.config import PricingConfig
from selfshelf.economics import (
    EconomicObjective,
    ProductContext,
    break_even_unit_uplift,
    compare_markdown_timing,
    evaluate_price_path,
    price_bounds,
    unit_days_held,
    waste_risk_weight,
)
from selfshelf.optimizer import optimize_product


def optimize(product, config=None, seed=0):
    config = config or PricingConfig()
    return optimize_product(product, config, np.random.default_rng(seed))


def make_product(**overrides) -> ProductContext:
    defaults = dict(
        current_price=5.00,
        retail_price=5.00,
        unit_cost=3.00,
        inventory_units=300,
        days_to_expiry=10,
        baseline_daily_demand=20.0,
        elasticity=-1.5,
    )
    defaults.update(overrides)
    return ProductContext(**defaults)


# ---------------------------------------------------------------------------
# Break-even unit uplift (gross-profit trade-off)
# ---------------------------------------------------------------------------

class TestBreakEvenUplift:
    def test_textbook_example(self):
        # Price 100 -> 80 with cost 60: margin halves, so volume must double.
        assert break_even_unit_uplift(100.0, 80.0, 60.0) == pytest.approx(2.0)

    def test_no_markdown_needs_no_uplift(self):
        assert break_even_unit_uplift(5.0, 5.0, 3.0) == pytest.approx(1.0)

    def test_markdown_to_or_below_cost_is_infinite(self):
        assert math.isinf(break_even_unit_uplift(100.0, 60.0, 60.0))
        assert math.isinf(break_even_unit_uplift(100.0, 50.0, 60.0))

    def test_already_below_cost_has_no_hurdle(self):
        assert break_even_unit_uplift(2.5, 2.0, 3.0) == pytest.approx(1.0)

    def test_rejects_price_increase(self):
        with pytest.raises(ValueError):
            break_even_unit_uplift(80.0, 100.0, 60.0)

    def test_gross_profit_crossover_matches_formula(self):
        """The objective's gross-profit component must agree with the
        break-even math: uplift above the hurdle -> more GP dollars,
        uplift below it -> fewer.

        With constant elasticity the predicted uplift at price ratio r is
        r**e, and the hurdle for 5.00 -> 4.00 at cost 3.00 is 2.0.
        r**e = 2 at e = ln(2)/ln(0.8) = -3.106...
        """
        hurdle = break_even_unit_uplift(5.00, 4.00, 3.00)
        assert hurdle == pytest.approx(2.0)

        def gp(elasticity, price):
            # Long-dated, deep stock: waste weight ~0, nothing capped.
            product = make_product(
                elasticity=elasticity, days_to_expiry=200,
                inventory_units=1_000_000,
            )
            return EconomicObjective(product, PricingConfig()).breakdown(
                price
            )["gross_profit"]

        weak, strong = -2.8, -3.4  # uplift 1.87x vs 2.14x at 20% off
        assert gp(weak, 4.00) < gp(weak, 5.00)
        assert gp(strong, 4.00) > gp(strong, 5.00)


# ---------------------------------------------------------------------------
# Waste risk weight and unit-days held
# ---------------------------------------------------------------------------

class TestWasteRiskWeight:
    def test_full_charge_inside_the_decision_window(self):
        for days in (1, 5, 10, 14):
            assert waste_risk_weight(days, 14, 5.0) == pytest.approx(1.0)

    def test_continuous_at_the_horizon_boundary(self):
        just_in = waste_risk_weight(14.0, 14, 5.0)
        just_out = waste_risk_weight(14.1, 14, 5.0)
        assert abs(just_in - just_out) < 0.03

    def test_discounted_beyond_the_window(self):
        weights = [waste_risk_weight(d, 14, 5.0) for d in (15, 21, 30, 90)]
        assert all(0.0 <= w < 1.0 for w in weights)
        assert weights == sorted(weights, reverse=True)

    def test_rejects_bad_tau(self):
        with pytest.raises(ValueError):
            waste_risk_weight(5, 14, 0.0)


class TestUnitDaysHeld:
    def test_zero_demand_holds_everything_all_window(self):
        assert unit_days_held(100, 0.0, 14) == pytest.approx(1400.0)

    def test_stock_that_never_sells_out_in_window(self):
        # 100 units selling 5/day over 10 days: integral of (100 - 5t).
        assert unit_days_held(100, 5.0, 10) == pytest.approx(750.0)

    def test_stock_selling_out_mid_window(self):
        # 100 units at 20/day sells out in 5 days: 100^2 / (2*20).
        assert unit_days_held(100, 20.0, 14) == pytest.approx(250.0)

    def test_faster_sell_down_holds_fewer_unit_days(self):
        slow = unit_days_held(300, 10.0, 14)
        fast = unit_days_held(300, 25.0, 14)
        assert fast < slow

    def test_edge_cases(self):
        assert unit_days_held(0, 10.0, 14) == 0.0
        assert unit_days_held(100, 10.0, 0) == 0.0
        with pytest.raises(ValueError):
            unit_days_held(-1, 10.0, 14)


# ---------------------------------------------------------------------------
# The no-markdown baseline is always the benchmark
# ---------------------------------------------------------------------------

class TestNoMarkdownBaseline:
    def test_recommendation_never_scores_below_current_price(self):
        rng = np.random.default_rng(77)
        config = PricingConfig()
        for _ in range(30):
            price = float(rng.uniform(1.0, 15.0))
            product = make_product(
                current_price=price, retail_price=price,
                unit_cost=price * float(rng.uniform(0.4, 0.9)),
                inventory_units=float(rng.integers(1, 900)),
                days_to_expiry=float(rng.integers(1, 90)),
                baseline_daily_demand=float(rng.uniform(0.0, 40.0)),
                elasticity=float(rng.uniform(-3.0, -0.3)),
            )
            result = optimize(product, config)
            assert result.value_improvement >= -1e-9
            if result.action == "Price Maintained":
                assert result.optimized_price == pytest.approx(price)


# ---------------------------------------------------------------------------
# Deep markdown trap: deepest discount is not automatically chosen
# ---------------------------------------------------------------------------

class TestDeepMarkdownTrap:
    def test_interior_optimum_beats_the_floor(self):
        # Elastic product with healthy margin: GP-optimal price (~3.4)
        # sits above the clearance floor (3.0). The optimizer must find
        # the interior optimum, not slam to the deepest allowed discount.
        product = make_product(
            current_price=5.00, retail_price=5.00, unit_cost=2.00,
            inventory_units=100_000, days_to_expiry=60,
            baseline_daily_demand=20.0, elasticity=-2.5,
        )
        config = PricingConfig()
        result = optimize(product, config)
        lower, _ = price_bounds(product, config)
        assert result.action == "Markdown"
        assert result.optimized_price > lower + 0.05
        objective = EconomicObjective(product, config)
        assert objective.score(result.optimized_price) > objective.score(lower)

    def test_recommendation_is_argmax_of_a_candidate_grid(self):
        # Sweep candidate prices explicitly and confirm the optimizer's
        # choice is at least as good as every grid candidate.
        product = make_product(
            inventory_units=400, days_to_expiry=4, elasticity=-2.0,
            unit_cost=3.00,
        )
        config = PricingConfig()
        result = optimize(product, config)
        objective = EconomicObjective(product, config)
        lower, upper = price_bounds(product, config)
        best_score = objective.score(result.optimized_price)
        for price in np.linspace(lower, upper, 41):
            assert best_score >= objective.score(float(price)) - 1e-6

    def test_huge_demand_at_tiny_margin_does_not_win(self):
        # Very elastic, but margin at the floor is a few cents: volume
        # cannot rescue near-zero unit margin without waste pressure.
        product = make_product(
            current_price=5.00, retail_price=5.00, unit_cost=3.00,
            inventory_units=1_000_000, days_to_expiry=200,
            baseline_daily_demand=20.0, elasticity=-4.0,
        )
        result = optimize(product)
        floor = 5.00 * (1 - PricingConfig().constraints.max_daily_price_drop)
        assert result.optimized_price > floor + 0.05


# ---------------------------------------------------------------------------
# Scenario matrix (remaining cases; A/C/D/E live in test_pricing.py)
# ---------------------------------------------------------------------------

class TestScenarioMatrix:
    def test_b_overstock_moderate_time_marks_down_when_value_positive(self):
        product = make_product(
            inventory_units=800, days_to_expiry=10,
            baseline_daily_demand=20.0, elasticity=-1.8, unit_cost=3.49,
            current_price=4.99, retail_price=4.99,
        )
        result = optimize(product)
        assert result.action == "Markdown"
        assert result.value_improvement > 0
        assert (
            result.optimized["expected_waste_units"]
            < result.current["expected_waste_units"]
        )

    def test_near_expiry_inelastic_is_not_forced_into_markdown(self):
        # Expiry alone must not trigger a markdown: with elasticity -0.4
        # a price cut barely moves demand, so the waste saved never pays
        # for the margin given up.
        product = make_product(
            inventory_units=100, days_to_expiry=2,
            baseline_daily_demand=10.0, elasticity=-0.4, unit_cost=3.49,
            current_price=4.99, retail_price=4.99,
        )
        result = optimize(product)
        assert result.action == "Price Maintained"

    def test_g_tiny_inventory_strong_demand_keeps_price(self):
        product = make_product(
            inventory_units=5, days_to_expiry=5,
            baseline_daily_demand=30.0, elasticity=-2.0,
        )
        result = optimize(product)
        assert result.action == "Price Maintained"

    def test_h_large_stock_weak_demand_feels_markdown_pressure(self):
        product = make_product(
            inventory_units=1000, days_to_expiry=8,
            baseline_daily_demand=5.0, elasticity=-1.8, unit_cost=3.49,
            current_price=4.99, retail_price=4.99,
        )
        result = optimize(product)
        assert result.action == "Markdown"
        assert result.value_improvement > 0

    def test_g_vs_h_same_expiry_opposite_decisions(self):
        # Inventory position, not expiry date, separates these two.
        tiny = make_product(
            inventory_units=10, days_to_expiry=8,
            baseline_daily_demand=20.0, elasticity=-1.8,
        )
        huge = make_product(
            inventory_units=1000, days_to_expiry=8,
            baseline_daily_demand=5.0, elasticity=-1.8, unit_cost=3.49,
            current_price=4.99, retail_price=4.99,
        )
        assert optimize(tiny).action == "Price Maintained"
        assert optimize(huge).action == "Markdown"


# ---------------------------------------------------------------------------
# Holding cost (scenarios I and J)
# ---------------------------------------------------------------------------

class TestHoldingCost:
    def _slow_mover(self) -> ProductContext:
        return make_product(
            current_price=5.00, retail_price=5.00, unit_cost=3.00,
            inventory_units=600, days_to_expiry=30,
            baseline_daily_demand=20.0, elasticity=-2.2,
        )

    def test_j_zero_holding_cost_creates_no_markdown_pressure(self):
        config = PricingConfig()
        assert config.inventory.holding_cost_per_unit_day == 0.0
        result = optimize(self._slow_mover(), config)
        assert result.action == "Price Maintained"

    def test_i_positive_holding_cost_makes_clearing_attractive(self):
        config = PricingConfig()
        config.inventory.holding_cost_per_unit_day = 0.10
        result = optimize(self._slow_mover(), config)
        assert result.action == "Markdown"
        # The saving comes from carrying fewer unit-days of stock.
        assert (
            result.optimized["holding_cost"]
            < result.current["holding_cost"]
        )

    def test_holding_cost_responds_to_price(self):
        config = PricingConfig()
        config.inventory.holding_cost_per_unit_day = 0.05
        objective = EconomicObjective(self._slow_mover(), config)
        assert (
            objective.breakdown(4.00)["holding_cost"]
            < objective.breakdown(5.00)["holding_cost"]
        )


# ---------------------------------------------------------------------------
# Price paths and markdown timing
# ---------------------------------------------------------------------------

class TestPricePath:
    def test_single_segment_reproduces_the_objective(self):
        config = PricingConfig()
        for product in (
            make_product(),
            make_product(inventory_units=800, days_to_expiry=4),
            make_product(baseline_daily_demand=0.0),
        ):
            objective = EconomicObjective(product, config)
            window = min(
                product.days_to_expiry,
                config.objective.planning_horizon_days,
            )
            for price in (5.00, 4.20, 3.40):
                path = evaluate_price_path(product, config, [(window, price)])
                assert path["score"] == pytest.approx(
                    objective.score(price), abs=1e-9
                )

    def test_split_segments_at_one_price_equal_a_single_segment(self):
        config = PricingConfig()
        product = make_product(inventory_units=500, days_to_expiry=12)
        whole = evaluate_price_path(product, config, [(12, 4.00)])
        split = evaluate_price_path(product, config, [(5, 4.00), (7, 4.00)])
        assert split["score"] == pytest.approx(whole["score"], abs=1e-6)

    def test_sales_never_exceed_inventory_across_segments(self):
        config = PricingConfig()
        product = make_product(inventory_units=50, days_to_expiry=12)
        path = evaluate_price_path(
            product, config, [(6, 4.50), (6, 3.50)]
        )
        assert path["terminal_inventory"] >= -1e-9
        sold = product.inventory_units - path["terminal_inventory"]
        assert sold <= product.inventory_units + 1e-9


class TestMarkdownTiming:
    def test_acting_now_beats_waiting_for_overstocked_short_dated_stock(self):
        product = make_product(
            inventory_units=800, days_to_expiry=6,
            baseline_daily_demand=20.0, elasticity=-1.8, unit_cost=3.49,
            current_price=4.99, retail_price=4.99,
        )
        config = PricingConfig()
        result = optimize(product, config)
        assert result.action == "Markdown"
        timing = compare_markdown_timing(
            product, config, result.optimized_price, wait_days=3
        )
        assert timing is not None
        assert timing["advantage_now"] > 0

    def test_one_day_window_has_no_waiting_option(self):
        product = make_product(days_to_expiry=1)
        assert compare_markdown_timing(product, PricingConfig(), 4.0) is None

    def test_markdown_results_carry_timing_information(self):
        product = make_product(
            inventory_units=800, days_to_expiry=6,
            baseline_daily_demand=20.0, elasticity=-1.8, unit_cost=3.49,
            current_price=4.99, retail_price=4.99,
        )
        result = optimize(product)
        assert result.timing is not None
        assert "advantage_now" in result.timing


# ---------------------------------------------------------------------------
# Edge cases from the verification checklist
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_cost_at_or_above_price_is_safe(self):
        product = make_product(
            current_price=3.00, retail_price=5.00, unit_cost=3.49,
            days_to_expiry=30,
        )
        result = optimize(product)
        assert result.optimized_price == pytest.approx(3.00)
        assert result.action == "Price Maintained"

    def test_zero_inventory_is_safe(self):
        product = make_product(inventory_units=0)
        result = optimize(product)
        assert result.optimized_price > 0
        assert result.current["expected_sell_through"] == pytest.approx(1.0)
        assert result.current["expected_waste_units"] == 0.0

    def test_break_even_fields_default_for_maintained_prices(self):
        result = optimize(make_product(
            inventory_units=30, days_to_expiry=60,
        ))
        assert result.action == "Price Maintained"
        assert result.break_even_uplift == 1.0
        assert result.predicted_uplift == 1.0
