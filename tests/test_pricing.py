"""Behavioral tests for the pricing engine.

These verify business behavior — when the optimizer should and should not
mark down — rather than implementation details.
"""

import numpy as np
import pytest

from selfshelf.config import PricingConfig
from selfshelf.economics import ProductContext, price_bounds
from selfshelf.optimizer import optimize_product, price_sweep


def optimize(product: ProductContext, config: PricingConfig = None, seed=0):
    config = config or PricingConfig()
    return optimize_product(product, config, np.random.default_rng(seed))


class TestScenario1HealthyInventory:
    """Long expiry, low inventory pressure, normal demand -> no markdown."""

    def test_price_maintained(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=60, days_to_expiry=60,
            baseline_daily_demand=20.0, elasticity=-1.2,
        )
        result = optimize(product)
        assert result.action == "Price Maintained"
        assert result.optimized_price == pytest.approx(4.99)

    def test_no_markdown_even_when_slightly_discounted_already(self):
        product = ProductContext(
            current_price=4.49, retail_price=4.99, unit_cost=3.49,
            inventory_units=80, days_to_expiry=45,
            baseline_daily_demand=15.0, elasticity=-1.2,
        )
        result = optimize(product)
        assert result.action == "Price Maintained"


class TestScenario2OverstockedNearExpiry:
    """High inventory, modest demand, expiry approaching -> markdown."""

    def test_meaningful_downward_pressure(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=400, days_to_expiry=3,
            baseline_daily_demand=20.0, elasticity=-2.0,
        )
        result = optimize(product)
        assert result.action == "Markdown"
        assert result.markdown_pct >= 10.0
        # The markdown must actually reduce expected waste...
        assert (
            result.optimized["expected_waste_units"]
            < result.current["expected_waste_units"]
        )
        # ...and improve the overall economic outcome.
        assert result.optimized["score"] > result.current["score"]

    def test_same_expiry_low_stock_is_left_alone(self):
        # Identical product but only 30 units on hand: demand clears it
        # comfortably, so no markdown is warranted.
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=30, days_to_expiry=3,
            baseline_daily_demand=20.0, elasticity=-2.0,
        )
        result = optimize(product)
        assert result.action == "Price Maintained"


class TestScenario3ExtremelyUrgent:
    """Very little time left, stock far beyond sell-through -> clearance."""

    def test_strong_clearance_behavior(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=500, days_to_expiry=1,
            baseline_daily_demand=10.0, elasticity=-1.5,
        )
        config = PricingConfig()
        result = optimize(product, config)
        assert result.action == "Markdown"
        assert result.markdown_pct >= 30.0
        # Clearance is still bounded by the configured constraints.
        lower, upper = price_bounds(product, config)
        assert result.optimized_price >= lower - 1e-9
        assert result.optimized_price <= upper + 1e-9

    def test_urgent_item_never_gets_price_increase(self):
        product = ProductContext(
            current_price=4.99, retail_price=6.99, unit_cost=3.49,
            inventory_units=500, days_to_expiry=1,
            baseline_daily_demand=10.0, elasticity=-1.5,
        )
        result = optimize(product)
        assert result.optimized_price <= product.current_price


class TestScenario4HighlyPriceSensitive:
    """Elastic demand + healthy margin -> markdown that raises profit."""

    def test_optimizer_discovers_profitable_markdown(self):
        product = ProductContext(
            current_price=5.00, retail_price=5.00, unit_cost=2.00,
            inventory_units=2000, days_to_expiry=60,
            baseline_daily_demand=20.0, elasticity=-3.0,
        )
        result = optimize(product)
        assert result.action == "Markdown"
        assert result.markdown_pct >= 20.0
        profit_current = (
            result.current["expected_revenue"] - result.current["cogs"]
        )
        profit_optimized = (
            result.optimized["expected_revenue"] - result.optimized["cogs"]
        )
        assert profit_optimized > profit_current


class TestScenario5PriceInsensitive:
    """Inelastic demand -> markdown buys almost nothing, keep the price."""

    def test_price_maintained(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=1000, days_to_expiry=60,
            baseline_daily_demand=20.0, elasticity=-0.5,
        )
        result = optimize(product)
        assert result.action == "Price Maintained"
        assert result.optimized_price == pytest.approx(4.99)


class TestSanityChecks:
    """Automated validation of impossible/unreasonable outcomes."""

    def _random_product(self, rng) -> ProductContext:
        price = float(rng.uniform(1.0, 20.0))
        return ProductContext(
            current_price=price,
            retail_price=price * float(rng.uniform(1.0, 1.3)),
            unit_cost=price * float(rng.uniform(0.4, 0.9)),
            inventory_units=float(rng.integers(1, 800)),
            days_to_expiry=float(rng.integers(1, 120)),
            baseline_daily_demand=float(rng.uniform(0.0, 50.0)),
            elasticity=float(rng.uniform(-3.0, -0.3)),
        )

    def test_prices_always_positive_and_within_bounds(self):
        config = PricingConfig()
        rng = np.random.default_rng(123)
        for _ in range(40):
            product = self._random_product(rng)
            result = optimize(product, config)
            lower, upper = price_bounds(product, config)
            assert result.optimized_price > 0
            assert lower - 1e-9 <= result.optimized_price <= upper + 1e-9
            assert result.optimized_price <= product.current_price + 1e-9

    def test_waste_and_sales_never_negative(self):
        config = PricingConfig()
        rng = np.random.default_rng(321)
        for _ in range(40):
            result = optimize(self._random_product(rng), config)
            for b in (result.current, result.optimized):
                assert b["expected_waste_units"] >= 0
                assert b["expected_waste_cost"] >= 0
                assert b["expected_sales_units"] >= 0

    def test_markdown_only_when_it_improves_the_score(self):
        config = PricingConfig()
        rng = np.random.default_rng(555)
        for _ in range(40):
            result = optimize(self._random_product(rng), config)
            if result.action == "Markdown":
                assert result.optimized["score"] >= result.current["score"]

    def test_deterministic_given_seed(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=400, days_to_expiry=3,
            baseline_daily_demand=20.0, elasticity=-2.0,
        )
        a = optimize(product, seed=9)
        b = optimize(product, seed=9)
        assert a.optimized_price == b.optimized_price
        assert a.action == b.action

    def test_zero_demand_product_is_handled(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=200, days_to_expiry=2,
            baseline_daily_demand=0.0, elasticity=-1.5,
        )
        result = optimize(product)
        assert result.optimized_price > 0
        assert result.optimized["expected_sales_units"] == 0.0


class TestExplanations:
    def test_markdown_explanations_reference_real_numbers(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=400, days_to_expiry=3,
            baseline_daily_demand=20.0, elasticity=-2.0,
        )
        result = optimize(product)
        text = " ".join(result.reasons)
        assert "expire unsold" in text
        assert "waste" in text

    def test_healthy_explanation_says_why_no_markdown(self):
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=60, days_to_expiry=60,
            baseline_daily_demand=20.0, elasticity=-1.2,
        )
        result = optimize(product)
        assert any("sell through" in r for r in result.reasons)


class TestPriceSweep:
    def test_sweep_covers_bounds_and_demand_is_monotone(self):
        config = PricingConfig()
        product = ProductContext(
            current_price=4.99, retail_price=4.99, unit_cost=3.49,
            inventory_units=400, days_to_expiry=3,
            baseline_daily_demand=20.0, elasticity=-2.0,
        )
        sweep = price_sweep(product, config, num_points=12)
        assert len(sweep) == 12
        lower, upper = price_bounds(product, config)
        assert sweep[0]["price"] == pytest.approx(lower, abs=0.01)
        assert sweep[-1]["price"] == pytest.approx(upper, abs=0.01)
        demands = [pt["daily_demand"] for pt in sweep]
        assert demands == sorted(demands, reverse=True)
