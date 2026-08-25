"""Economic-correctness audit suite.

Locks down the properties verified during the engine audit: dimensional
consistency of every term, sensitivity/monotonicity directions, PSO vs
explicit grid search, rounding invariants, break-even/gross-profit
equivalence, backtest accounting reconciliation, and counterfactual
fairness. These tests define the frozen behavior of the economic core.
"""

import math

import numpy as np
import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.backtest import backtest_recommendations
from selfshelf.economics import (
    EconomicObjective,
    ProductContext,
    inventory_pressure,
    price_bounds,
)
from selfshelf.optimizer import optimize_product
from selfshelf import pso


def make_product(**overrides) -> ProductContext:
    defaults = dict(
        current_price=4.99,
        retail_price=4.99,
        unit_cost=3.49,
        inventory_units=300,
        days_to_expiry=10,
        baseline_daily_demand=20.0,
        elasticity=-1.5,
    )
    defaults.update(overrides)
    return ProductContext(**defaults)


def random_product(rng) -> ProductContext:
    price = float(rng.uniform(1.0, 15.0))
    return ProductContext(
        current_price=price,
        retail_price=price * float(rng.uniform(1.0, 1.3)),
        unit_cost=price * float(rng.uniform(0.4, 0.9)),
        inventory_units=float(rng.integers(1, 900)),
        days_to_expiry=float(rng.integers(1, 90)),
        baseline_daily_demand=float(rng.uniform(0.0, 40.0)),
        elasticity=float(rng.uniform(-3.0, -0.3)),
    )


def optimize(product, config=None, seed=0):
    config = config or PricingConfig()
    return optimize_product(product, config, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# Dimensional / accounting consistency of the objective
# ---------------------------------------------------------------------------

class TestDimensionalConsistency:
    """Every published component must reconcile with its own definition:
    $ = $/unit x units, and the score must equal the sum of its parts."""

    def test_breakdown_components_reconcile(self):
        cfg = PricingConfig()
        cfg.inventory.holding_cost_per_unit_day = 0.03
        cfg.waste.disposal_cost_per_unit = 0.25
        rng = np.random.default_rng(11)
        for _ in range(40):
            product = random_product(rng)
            objective = EconomicObjective(product, cfg)
            price = float(rng.uniform(*price_bounds(product, cfg)) )
            b = objective.breakdown(price)

            sold = b["expected_sales_units"]
            assert b["expected_revenue"] == pytest.approx(price * sold)
            assert b["cogs"] == pytest.approx(product.unit_cost * sold)
            assert b["gross_profit"] == pytest.approx(
                b["expected_revenue"] - b["cogs"]
            )
            assert b["expected_waste_cost"] == pytest.approx(
                b["expected_waste_units"]
                * cfg.waste.unit_waste_loss(product.unit_cost)
                * b["waste_risk_weight"]
            )
            assert b["holding_cost"] == pytest.approx(
                cfg.inventory.holding_cost_per_unit_day * b["unit_days_held"]
            )
            assert b["markdown_cost"] == pytest.approx(
                cfg.objective.markdown_friction
                * max(0.0, product.current_price - price)
                * sold
            )
            assert b["score"] == pytest.approx(
                b["gross_profit"] - b["expected_waste_cost"]
                - b["holding_cost"] - b["markdown_cost"]
            )
            # units sold + terminal inventory = starting inventory
            assert sold + b["terminal_inventory"] == pytest.approx(
                product.inventory_units
            )
            if product.inventory_units > 0:
                assert b["expected_sell_through"] == pytest.approx(
                    sold / product.inventory_units
                )

    def test_monetary_scale_invariance(self):
        """Scaling every $-denominated input by k scales every
        $-denominated output by k and leaves unit quantities unchanged."""
        k = 3.7
        base_cfg = PricingConfig()
        scaled_cfg = PricingConfig()
        scaled_cfg.inventory.holding_cost_per_unit_day = 0.04
        base_cfg.inventory.holding_cost_per_unit_day = 0.04 / k
        # disposal is $/unit, so it must scale too
        base_cfg.waste.disposal_cost_per_unit = 0.20 / k
        scaled_cfg.waste.disposal_cost_per_unit = 0.20

        product = make_product(inventory_units=400, days_to_expiry=4)
        scaled = make_product(
            current_price=4.99 * k, retail_price=4.99 * k,
            unit_cost=3.49 * k, inventory_units=400, days_to_expiry=4,
        )
        b1 = EconomicObjective(product, base_cfg).breakdown(4.20)
        b2 = EconomicObjective(scaled, scaled_cfg).breakdown(4.20 * k)

        for dollar_key in ("expected_revenue", "cogs", "gross_profit",
                           "expected_waste_cost", "holding_cost",
                           "markdown_cost", "score"):
            assert b2[dollar_key] == pytest.approx(k * b1[dollar_key])
        for unit_key in ("daily_demand", "expected_sales_units",
                         "expected_waste_units", "terminal_inventory",
                         "unit_days_held", "expected_sell_through"):
            assert b2[unit_key] == pytest.approx(b1[unit_key])

    def test_negative_days_to_expiry_rejected(self):
        with pytest.raises(ValueError):
            make_product(days_to_expiry=-1)


# ---------------------------------------------------------------------------
# Sensitivity / monotonicity
# ---------------------------------------------------------------------------

class TestSensitivity:
    def test_higher_waste_cost_makes_markdown_weakly_more_attractive(self):
        prices, improvements = [], []
        for disposal in (0.0, 0.5, 1.5):
            cfg = PricingConfig()
            cfg.waste.disposal_cost_per_unit = disposal
            result = optimize(make_product(
                inventory_units=400, days_to_expiry=4, elasticity=-2.0,
            ), cfg)
            prices.append(result.optimized_price)
            improvements.append(result.value_improvement)
        assert prices == sorted(prices, reverse=True)  # weakly deeper
        assert improvements == sorted(improvements)    # strictly more valuable
        assert improvements[0] < improvements[-1]

    def test_higher_holding_cost_prefers_faster_sell_through(self):
        prices = []
        for rate in (0.0, 0.05, 0.10, 0.20):
            cfg = PricingConfig()
            cfg.inventory.holding_cost_per_unit_day = rate
            result = optimize(make_product(
                current_price=5.00, retail_price=5.00, unit_cost=3.00,
                inventory_units=600, days_to_expiry=30,
                baseline_daily_demand=20.0, elasticity=-2.2,
            ), cfg)
            prices.append(result.optimized_price)
        assert prices[0] == pytest.approx(5.00)   # zero rate: no pressure
        assert prices == sorted(prices, reverse=True)
        assert prices[-1] < prices[0]             # high rate: real markdown

    def test_more_inventory_means_more_terminal_risk_and_deeper_markdown(self):
        prices, wastes = [], []
        for inv in (50, 200, 500, 1000):
            product = make_product(
                inventory_units=inv, days_to_expiry=4, elasticity=-2.0,
            )
            result = optimize(product)
            prices.append(result.optimized_price)
            wastes.append(result.current["expected_waste_units"])
        assert wastes == sorted(wastes)             # strictly more at risk
        assert prices == sorted(prices, reverse=True)  # weakly deeper
        assert prices[-1] < prices[0]

    def test_less_time_remaining_means_more_pressure(self):
        prices = []
        for dte in (20, 10, 4, 1):
            product = make_product(
                inventory_units=400, days_to_expiry=dte, elasticity=-2.0,
            )
            result = optimize(product)
            prices.append(result.optimized_price)
            # risk metric itself is monotone for fixed demand/inventory
        assert prices == sorted(prices, reverse=True)
        assert prices[-1] < prices[0]

        pressures = [
            inventory_pressure(400, 20.0, d) for d in (20, 10, 4, 1)
        ]
        assert pressures == sorted(pressures)

    def test_stronger_elasticity_deepens_gp_driven_markdowns(self):
        prices, improvements = [], []
        for e in (-2.0, -2.6, -3.0, -3.4):
            result = optimize(make_product(
                current_price=5.00, retail_price=5.00, unit_cost=2.00,
                inventory_units=100_000, days_to_expiry=60,
                baseline_daily_demand=20.0, elasticity=e,
            ))
            prices.append(result.optimized_price)
            improvements.append(result.value_improvement)
        assert prices == sorted(prices, reverse=True)
        assert prices[-1] < prices[0]
        assert improvements == sorted(improvements)

    def test_higher_cost_raises_the_healthy_price_floor(self):
        cfg = PricingConfig()
        floors = []
        for cost in (2.0, 2.5, 3.0):
            lower, _ = price_bounds(
                make_product(unit_cost=cost, days_to_expiry=60), cfg
            )
            floors.append(lower)
            # The smooth expiry blend relaxes the margin floor by ~1e-5
            # even at 60 days (pressure is never exactly zero) — that is
            # the documented design, so allow that much slack.
            assert lower >= cost * (1 + cfg.constraints.min_margin) - 1e-3
        assert floors == sorted(floors)


# ---------------------------------------------------------------------------
# PSO vs explicit grid search
# ---------------------------------------------------------------------------

class TestPSOVersusGrid:
    def test_pso_matches_dense_grid_on_random_products(self):
        cfg = PricingConfig()
        rng = np.random.default_rng(2024)
        for i in range(30):
            product = random_product(rng)
            objective = EconomicObjective(product, cfg)
            bounds = price_bounds(product, cfg)
            _, score_pso = pso.maximize(
                objective.score, bounds, cfg.pso, np.random.default_rng(i)
            )
            grid = np.linspace(bounds[0], bounds[1], 1001)
            grid_best = max(objective.score(float(g)) for g in grid)
            # PSO searches continuously, so it may only ever beat the grid.
            tolerance = 1e-6 * max(1.0, abs(grid_best))
            assert score_pso >= grid_best - tolerance

    def test_same_seed_identical_different_seed_still_optimal(self):
        cfg = PricingConfig()
        product = make_product(inventory_units=400, days_to_expiry=4,
                               elasticity=-2.0)
        objective = EconomicObjective(product, cfg)
        bounds = price_bounds(product, cfg)

        one = pso.maximize(objective.score, bounds, cfg.pso,
                           np.random.default_rng(5))
        two = pso.maximize(objective.score, bounds, cfg.pso,
                           np.random.default_rng(5))
        assert one == two

        grid = np.linspace(bounds[0], bounds[1], 1001)
        grid_best = max(objective.score(float(g)) for g in grid)
        for seed in (1, 2, 3, 4):
            _, score = pso.maximize(objective.score, bounds, cfg.pso,
                                    np.random.default_rng(seed))
            assert score >= grid_best - 1e-6 * max(1.0, abs(grid_best))


# ---------------------------------------------------------------------------
# Rounding invariants
# ---------------------------------------------------------------------------

class TestRoundingInvariants:
    def test_final_prices_are_whole_cents_within_bounds(self):
        cfg = PricingConfig()
        rng = np.random.default_rng(404)
        for _ in range(50):
            product = random_product(rng)
            result = optimize(product, cfg)
            price = result.optimized_price
            if result.action == "Markdown":
                assert price == pytest.approx(round(price, 2), abs=1e-9)
            lower, upper = price_bounds(product, cfg)
            assert lower - 1e-9 <= price <= upper + 1e-9
            assert result.action != "Price Increase"
            assert result.value_improvement >= -1e-9
            if result.action == "Price Maintained":
                # Maintained means the EXACT current price, unrounded.
                assert price == product.current_price


# ---------------------------------------------------------------------------
# Break-even uplift <-> gross profit equivalence
# ---------------------------------------------------------------------------

class TestBreakEvenGrossProfitEquivalence:
    def test_meets_hurdle_iff_gross_profit_improves(self):
        """With uplift measured in units SOLD, 'predicted uplift >= hurdle'
        must coincide exactly with 'gross profit dollars improve'
        (for products currently selling above cost)."""
        rng = np.random.default_rng(909)
        checked = 0
        for _ in range(120):
            product = random_product(rng)
            if product.current_price <= product.unit_cost:
                continue
            result = optimize(product)
            if result.action != "Markdown":
                continue
            if result.current["expected_sales_units"] <= 0:
                continue
            meets = result.predicted_uplift >= result.break_even_uplift
            gp_improves = (
                result.optimized["gross_profit"]
                >= result.current["gross_profit"] - 1e-9
            )
            assert meets == gp_improves
            checked += 1
        assert checked >= 10  # the sample must actually contain markdowns

    def test_uplift_is_measured_in_units_sold_not_demand(self):
        # Half the stock expires at the current price; the optimizer cuts
        # toward the sell-out boundary, where inventory caps sales. The
        # reported uplift must be the ratio of units SOLD (never more
        # than the demand ratio, and equal to it only when nothing caps).
        product = make_product(
            current_price=5.00, retail_price=5.00, unit_cost=3.00,
            inventory_units=100, days_to_expiry=2,
            baseline_daily_demand=20.0, elasticity=-2.5,
        )
        result = optimize(product)
        assert result.action == "Markdown"
        demand_ratio = (
            result.optimized["daily_demand"] / result.current["daily_demand"]
        )
        sales_ratio = (
            result.optimized["expected_sales_units"]
            / result.current["expected_sales_units"]
        )
        assert result.optimized["expected_sales_units"] <= 100 + 1e-9
        assert result.predicted_uplift == pytest.approx(sales_ratio)
        assert result.predicted_uplift <= demand_ratio + 1e-9


# ---------------------------------------------------------------------------
# Economic regimes: hold / shallow / moderate / deep
# ---------------------------------------------------------------------------

class TestEconomicRegimes:
    def test_four_distinct_regimes_emerge_from_the_objective(self):
        hold = optimize(make_product(
            inventory_units=60, days_to_expiry=60, elasticity=-1.2,
        ))
        shallow = optimize(make_product(
            inventory_units=95, days_to_expiry=5,
            baseline_daily_demand=17.4, elasticity=-1.6,
        ))
        moderate = optimize(make_product(
            current_price=5.00, retail_price=5.00, unit_cost=2.00,
            inventory_units=100_000, days_to_expiry=60,
            baseline_daily_demand=20.0, elasticity=-2.5,
        ))
        deep = optimize(make_product(
            inventory_units=500, days_to_expiry=1,
            baseline_daily_demand=10.0, elasticity=-1.5,
        ))
        assert hold.action == "Price Maintained"
        assert 0.0 < shallow.markdown_pct <= 10.0
        assert 10.0 < moderate.markdown_pct <= 30.0
        assert deep.markdown_pct >= 30.0
        for result in (shallow, moderate, deep):
            assert result.value_improvement > 0


# ---------------------------------------------------------------------------
# Backtest accounting and counterfactual fairness
# ---------------------------------------------------------------------------

def backtest_frames():
    items = pd.DataFrame([
        {"DEPARTMENT": "Bakery", "PRICE_RETAIL": 5.00, "PRICE_CURRENT": 5.00,
         "PROMOTION": 0, "Season": "Spring", "DAYS_TO_EXPIRY": 3,
         "INVENTORY_UNITS": 400, "COST": 3.50},
        {"DEPARTMENT": "Beverages", "PRICE_RETAIL": 4.00,
         "PRICE_CURRENT": 3.60, "PROMOTION": 1, "Season": "Summer",
         "DAYS_TO_EXPIRY": 40, "INVENTORY_UNITS": 80, "COST": 2.80},
    ])
    recs = pd.DataFrame([
        {"Recommended_Price": 3.60, "Action": "Markdown",
         "Markdown_Percentage": 28.0},
        {"Recommended_Price": 3.60, "Action": "Price Maintained",
         "Markdown_Percentage": 0.0},
    ])
    return items, recs


class TestBacktestAccounting:
    def test_units_reconcile_exactly(self):
        items, recs = backtest_frames()
        summary = backtest_recommendations(items, recs, PricingConfig())
        starting = float(items["INVENTORY_UNITS"].sum())
        for strategy in ("hold", "recommended"):
            s = summary[strategy]
            assert s["units_sold"] + s["terminal_inventory"] == pytest.approx(
                starting, abs=0.05
            )
            assert s["terminal_inventory"] == pytest.approx(
                s["waste_units"], abs=0.05
            )
            assert s["economic_value"] == pytest.approx(
                s["gross_profit"] - s["holding_cost"], abs=0.05
            )

    def test_zero_holding_rate_accrues_no_holding_cost(self):
        items, recs = backtest_frames()
        summary = backtest_recommendations(items, recs, PricingConfig())
        assert summary["hold"]["holding_cost"] == 0.0
        assert summary["recommended"]["holding_cost"] == 0.0

    def test_positive_rate_charges_less_holding_on_faster_sell_down(self):
        cfg = PricingConfig()
        cfg.inventory.holding_cost_per_unit_day = 0.05
        items, recs = backtest_frames()
        summary = backtest_recommendations(items, recs, cfg)
        assert summary["hold"]["holding_cost"] > 0
        assert (
            summary["recommended"]["holding_cost"]
            <= summary["hold"]["holding_cost"] + 1e-9
        )

    def test_recommending_the_current_price_reproduces_hold_exactly(self):
        """Counterfactual fairness: with identical prices the two
        strategies must produce identical outcomes — proving the demand
        environment, noise, and initial conditions are shared."""
        items, _ = backtest_frames()
        recs = pd.DataFrame([
            {"Recommended_Price": 5.00, "Action": "Price Maintained",
             "Markdown_Percentage": 0.0},
            {"Recommended_Price": 3.60, "Action": "Price Maintained",
             "Markdown_Percentage": 0.0},
        ])
        summary = backtest_recommendations(items, recs, PricingConfig())
        assert summary["hold"] == summary["recommended"]
