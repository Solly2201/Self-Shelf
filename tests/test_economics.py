import math

import pytest

from selfshelf.config import PricingConfig
from selfshelf.economics import (
    EconomicObjective,
    ProductContext,
    days_of_supply,
    expected_waste_units,
    expiry_pressure,
    inventory_pressure,
    price_bounds,
    price_effect,
)


def make_product(**overrides) -> ProductContext:
    defaults = dict(
        current_price=4.99,
        retail_price=4.99,
        unit_cost=3.49,
        inventory_units=100,
        days_to_expiry=10,
        baseline_daily_demand=20.0,
        elasticity=-1.5,
    )
    defaults.update(overrides)
    return ProductContext(**defaults)


class TestPriceEffect:
    def test_reference_price_gives_unit_multiplier(self):
        assert price_effect(5.0, 5.0, -1.5) == pytest.approx(1.0)

    def test_lower_price_raises_demand(self):
        assert price_effect(4.0, 5.0, -1.5) > 1.0

    def test_higher_price_lowers_demand(self):
        assert price_effect(6.0, 5.0, -1.5) < 1.0

    def test_monotone_decreasing_in_price(self):
        prices = [2.0, 3.0, 4.0, 5.0, 6.0]
        effects = [price_effect(p, 5.0, -1.5) for p in prices]
        assert effects == sorted(effects, reverse=True)

    def test_stronger_elasticity_reacts_more(self):
        mild = price_effect(4.0, 5.0, -0.5)
        strong = price_effect(4.0, 5.0, -2.5)
        assert strong > mild > 1.0

    def test_rejects_positive_elasticity(self):
        with pytest.raises(ValueError):
            price_effect(4.0, 5.0, 1.5)

    def test_rejects_non_positive_prices(self):
        with pytest.raises(ValueError):
            price_effect(0.0, 5.0, -1.5)
        with pytest.raises(ValueError):
            price_effect(4.0, 0.0, -1.5)


class TestExpiryPressure:
    def test_one_day_left_is_full_pressure(self):
        assert expiry_pressure(1, tau_days=5.0) == pytest.approx(1.0)

    def test_long_shelf_life_is_near_zero(self):
        assert expiry_pressure(120, tau_days=5.0) < 0.001

    def test_monotone_increasing_as_expiry_approaches(self):
        days = [30, 14, 7, 3, 1]
        pressures = [expiry_pressure(d, 5.0) for d in days]
        assert pressures == sorted(pressures)

    def test_smooth_no_cliff_around_three_days(self):
        # No arbitrary discontinuity at the old <=3-day threshold.
        p4, p3 = expiry_pressure(4, 5.0), expiry_pressure(3, 5.0)
        assert (p3 - p4) < 0.2

    def test_bounded_in_unit_interval(self):
        for d in (0, 1, 2, 10, 1000):
            assert 0.0 <= expiry_pressure(d, 5.0) <= 1.0

    def test_rejects_bad_tau(self):
        with pytest.raises(ValueError):
            expiry_pressure(5, tau_days=0)


class TestDaysOfSupply:
    def test_basic(self):
        assert days_of_supply(100, 10) == pytest.approx(10.0)

    def test_zero_demand_is_infinite(self):
        assert math.isinf(days_of_supply(100, 0.0))

    def test_zero_inventory_is_zero(self):
        assert days_of_supply(0, 10) == 0.0

    def test_negative_inventory_rejected(self):
        with pytest.raises(ValueError):
            days_of_supply(-1, 10)


class TestExpectedWaste:
    def test_no_waste_when_demand_clears_stock(self):
        assert expected_waste_units(50, 10, 10) == 0.0

    def test_waste_when_overstocked(self):
        assert expected_waste_units(500, 10, 10) == pytest.approx(400.0)

    def test_never_negative(self):
        assert expected_waste_units(5, 100, 100) == 0.0

    def test_negative_inventory_rejected(self):
        with pytest.raises(ValueError):
            expected_waste_units(-5, 10, 10)


class TestInventoryPressure:
    def test_healthy_stock_zero_pressure(self):
        assert inventory_pressure(50, 10, 10) == 0.0

    def test_hopeless_stock_full_pressure(self):
        assert inventory_pressure(1000, 0.0, 2) == pytest.approx(1.0)

    def test_partial_risk(self):
        # 500 units, can sell 100 -> 80% at risk.
        assert inventory_pressure(500, 10, 10) == pytest.approx(0.8)

    def test_distinguishes_same_expiry_different_stock(self):
        # The scenario from the design brief: 10 days to expiry, 10/day
        # demand -- 5 units vs 500 units must not look alike.
        low = inventory_pressure(5, 10, 10)
        high = inventory_pressure(500, 10, 10)
        assert low == 0.0
        assert high > 0.7


class TestEconomicObjective:
    def test_demand_monotone_decreasing_in_price(self):
        obj = EconomicObjective(make_product(), PricingConfig())
        demands = [obj.demand_at(p) for p in (3.0, 4.0, 5.0, 6.0)]
        assert demands == sorted(demands, reverse=True)

    def test_breakdown_components_are_sane(self):
        obj = EconomicObjective(make_product(), PricingConfig())
        b = obj.breakdown(4.0)
        assert b["expected_waste_units"] >= 0
        assert b["expected_waste_cost"] >= 0
        assert b["expected_sales_units"] >= 0
        assert b["expected_revenue"] == pytest.approx(
            4.0 * b["expected_sales_units"]
        )
        assert b["expected_sales_units"] <= make_product().inventory_units

    def test_sales_capped_by_inventory(self):
        product = make_product(inventory_units=5, days_to_expiry=30)
        obj = EconomicObjective(product, PricingConfig())
        assert obj.breakdown(4.99)["expected_sales_units"] == pytest.approx(5)

    def test_waste_shrinks_when_price_drops(self):
        product = make_product(inventory_units=400, days_to_expiry=5)
        obj = EconomicObjective(product, PricingConfig())
        high = obj.breakdown(4.99)["expected_waste_units"]
        low = obj.breakdown(3.49)["expected_waste_units"]
        assert low < high

    def test_zero_baseline_demand_is_safe(self):
        product = make_product(baseline_daily_demand=0.0)
        obj = EconomicObjective(product, PricingConfig())
        b = obj.breakdown(4.0)
        assert b["expected_sales_units"] == 0.0
        assert b["expected_waste_units"] == pytest.approx(100.0)

    def test_context_rejects_negative_inputs(self):
        with pytest.raises(ValueError):
            make_product(inventory_units=-1)
        with pytest.raises(ValueError):
            make_product(current_price=0)
        with pytest.raises(ValueError):
            make_product(baseline_daily_demand=-2)


class TestPriceBounds:
    def test_ceiling_never_exceeds_current_or_retail(self):
        cfg = PricingConfig()
        product = make_product(current_price=4.5, retail_price=4.99)
        lower, upper = price_bounds(product, cfg)
        assert upper <= 4.5
        assert lower <= upper

    def test_healthy_floor_protects_margin(self):
        cfg = PricingConfig()
        product = make_product(days_to_expiry=120)
        lower, _ = price_bounds(product, cfg)
        assert lower >= product.unit_cost * (1 + cfg.constraints.min_margin) - 1e-9

    def test_urgent_floor_relaxes_toward_clearance(self):
        cfg = PricingConfig()
        healthy_lower, _ = price_bounds(make_product(days_to_expiry=120), cfg)
        urgent_lower, _ = price_bounds(make_product(days_to_expiry=1), cfg)
        assert urgent_lower < healthy_lower
        # Fully urgent floor is bounded by the daily drop limit and never
        # below the configured clearance floor.
        floor_limit = max(
            cfg.constraints.clearance_floor_cost_fraction * 3.49,
            4.99 * (1 - cfg.constraints.max_daily_price_drop),
        )
        assert urgent_lower == pytest.approx(floor_limit)

    def test_daily_drop_limit_respected(self):
        cfg = PricingConfig()
        product = make_product(days_to_expiry=1)
        lower, _ = price_bounds(product, cfg)
        assert lower >= product.current_price * (
            1 - cfg.constraints.max_daily_price_drop
        ) - 1e-9

    def test_bounds_always_positive_and_ordered(self):
        cfg = PricingConfig()
        for days in (1, 3, 10, 60):
            for price in (0.5, 2.0, 20.0):
                product = make_product(
                    current_price=price, retail_price=price,
                    unit_cost=price * 0.7, days_to_expiry=days,
                )
                lower, upper = price_bounds(product, cfg)
                assert 0 < lower <= upper

    def test_degenerate_case_never_forces_increase(self):
        # Current price already below the margin floor: range collapses to
        # the current price instead of pushing the price up.
        cfg = PricingConfig()
        product = make_product(
            current_price=3.0, retail_price=4.99, unit_cost=3.49,
            days_to_expiry=60,
        )
        lower, upper = price_bounds(product, cfg)
        assert upper == pytest.approx(3.0)
        assert lower <= upper
