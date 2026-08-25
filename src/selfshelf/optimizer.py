"""Per-product price optimization and explanation generation."""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import PricingConfig
from .economics import (
    EconomicObjective,
    ProductContext,
    break_even_unit_uplift,
    compare_markdown_timing,
    days_of_supply,
    price_bounds,
)
from . import pso


@dataclass
class OptimizationResult:
    product: ProductContext
    optimized_price: float
    action: str
    current: Dict[str, float]    # objective breakdown at the current price
    optimized: Dict[str, float]  # objective breakdown at the optimized price
    bounds: tuple
    reasons: List[str] = field(default_factory=list)
    # Gross-profit break-even sanity check: unit-sales multiplier the
    # markdown needs vs the multiplier the demand model predicts.
    break_even_uplift: float = 1.0
    predicted_uplift: float = 1.0
    # Simplified "act now vs wait" comparison (None when not applicable).
    timing: Dict[str, float] = None

    @property
    def markdown_pct(self) -> float:
        return 100.0 * (
            1.0 - self.optimized_price / self.product.current_price
        )

    @property
    def value_improvement(self) -> float:
        """Economic score gained by moving off the current price."""
        return self.optimized["score"] - self.current["score"]


def optimize_product(
    product: ProductContext,
    config: PricingConfig,
    rng: np.random.Generator,
) -> OptimizationResult:
    """Search the constrained price range for the best economic score."""
    objective = EconomicObjective(product, config)
    bounds = price_bounds(product, config)

    best_price, _ = pso.maximize(objective.score, bounds, config.pso, rng)

    current_price = product.current_price
    # Guard against float drift outside the feasible range.
    best_price = float(np.clip(best_price, bounds[0], bounds[1]))

    # Ignore markdowns too small to matter operationally, and keep the
    # exact current price in that case — rounding a maintained price can
    # otherwise nudge it a cent above the current price and masquerade as
    # an increase.
    min_step = config.constraints.min_meaningful_markdown * current_price
    if abs(best_price - current_price) <= min_step:
        best_price = current_price
    else:
        # Round to a whole cent, then nudge back inside the feasible range
        # (rounding 2.994 down to 2.99 must not fall through the floor).
        best_price = round(best_price, 2)
        if best_price < bounds[0]:
            best_price = math.ceil(bounds[0] * 100) / 100
        if best_price > bounds[1]:
            best_price = math.floor(bounds[1] * 100) / 100
        if best_price < bounds[0] - 1e-9:
            # No whole-cent price exists inside the range; the current
            # price is always feasible, fall back to it.
            best_price = current_price
        elif abs(best_price - current_price) <= min_step:
            best_price = current_price

    current_breakdown = objective.breakdown(current_price)
    optimized_breakdown = objective.breakdown(best_price)

    # A markdown must pay for itself: if the score at the "optimized" price
    # is not an improvement, keep the current price.
    if (
        best_price != current_price
        and optimized_breakdown["score"] <= current_breakdown["score"]
    ):
        best_price = current_price
        optimized_breakdown = current_breakdown

    if best_price < current_price:
        action = "Markdown"
    elif best_price > current_price:
        action = "Price Increase"
    else:
        action = "Price Maintained"

    result = OptimizationResult(
        product=product,
        optimized_price=best_price,
        action=action,
        current=current_breakdown,
        optimized=optimized_breakdown,
        bounds=bounds,
    )

    if action == "Markdown":
        result.break_even_uplift = break_even_unit_uplift(
            current_price, best_price, product.unit_cost
        )
        # The hurdle is about units SOLD, not demand: when inventory caps
        # sales at the lower price, the demand ratio would overstate the
        # achievable uplift. With the sales ratio,
        # uplift >= hurdle <=> gross profit improves, exactly.
        sold_current = current_breakdown["expected_sales_units"]
        sold_optimized = optimized_breakdown["expected_sales_units"]
        if sold_current > 0:
            result.predicted_uplift = sold_optimized / sold_current
        elif sold_optimized > 0:
            result.predicted_uplift = math.inf
        result.timing = compare_markdown_timing(product, config, best_price)

    result.reasons = _build_reasons(result, config)
    return result


def _build_reasons(result: OptimizationResult, config: PricingConfig) -> List[str]:
    """Structured explanation derived from the actual economic numbers."""
    p = result.product
    cur, opt = result.current, result.optimized
    reasons: List[str] = []

    dos = days_of_supply(p.inventory_units, cur["daily_demand"])
    if np.isinf(dos):
        reasons.append(
            "predicted demand at the current price is ~0, so stock will not "
            "sell through on its own"
        )
    elif p.inventory_units > 0:
        reasons.append(
            f"{dos:.1f} days of supply vs {p.days_to_expiry:.0f} days of "
            f"shelf life remaining"
        )

    if cur["expiry_pressure"] >= 0.5:
        reasons.append(
            f"expiry pressure is high ({cur['expiry_pressure']:.2f}) with "
            f"{p.days_to_expiry:.0f} day(s) left"
        )

    if cur["expected_waste_units"] > 0.5:
        reasons.append(
            f"~{cur['expected_waste_units']:.0f} of "
            f"{p.inventory_units:.0f} units are expected to expire unsold "
            f"at the current price"
        )

    if result.action == "Markdown":
        sell_through_delta = (
            opt["expected_sell_through"] - cur["expected_sell_through"]
        )
        if sell_through_delta > 0.01:
            reasons.append(
                f"expected sell-through rises from "
                f"{100 * cur['expected_sell_through']:.0f}% to "
                f"{100 * opt['expected_sell_through']:.0f}%"
            )
        waste_delta = cur["expected_waste_units"] - opt["expected_waste_units"]
        if waste_delta > 0.5:
            reasons.append(
                f"marking down to ${result.optimized_price:.2f} cuts expected "
                f"waste by ~{waste_delta:.0f} units"
            )
        demand_delta = opt["daily_demand"] - cur["daily_demand"]
        if demand_delta > 0.05 * max(cur["daily_demand"], 1e-9):
            reasons.append(
                f"demand is price-sensitive (elasticity {p.elasticity:.2f}): "
                f"predicted daily demand rises from {cur['daily_demand']:.1f} "
                f"to {opt['daily_demand']:.1f}"
            )
        if math.isfinite(result.break_even_uplift):
            met = "meets" if (
                result.predicted_uplift >= result.break_even_uplift
            ) else "does not meet"
            uplift_text = (
                f"{result.predicted_uplift:.2f}x"
                if math.isfinite(result.predicted_uplift) else "unbounded"
            )
            reasons.append(
                f"gross-profit break-even needs {result.break_even_uplift:.2f}x "
                f"unit volume; predicted uplift is {uplift_text} "
                f"({met} the hurdle on margin alone)"
            )
        else:
            reasons.append(
                "price is at or below cost: justified by avoiding waste/"
                "terminal-stock losses, not by demand stimulation"
            )
        if result.timing and result.timing["advantage_now"] > 0.01:
            reasons.append(
                f"acting now rather than waiting "
                f"{result.timing['wait_days']:.0f} day(s) preserves "
                f"~${result.timing['advantage_now']:.2f} of expected value"
            )
        reasons.append(
            f"expected economic outcome improves by "
            f"${result.value_improvement:.2f} over the evaluation window"
        )
    elif result.action == "Price Maintained":
        if cur["expected_waste_units"] <= 0.5:
            reasons.append(
                "stock is expected to sell through before expiry, so a "
                "markdown would only give away margin"
            )
        else:
            reasons.append(
                "no price in the allowed range improves the expected "
                "economic outcome"
            )

    return reasons


def price_sweep(
    product: ProductContext,
    config: PricingConfig,
    num_points: int = 15,
) -> List[Dict[str, float]]:
    """Evaluate the economic breakdown across the feasible price range.

    Powers price-demand curves, profit curves and explainability views.
    """
    objective = EconomicObjective(product, config)
    lower, upper = price_bounds(product, config)
    prices = np.linspace(lower, upper, num_points)
    sweep = []
    for price in prices:
        b = objective.breakdown(float(price))
        sweep.append({
            "price": round(b["price"], 2),
            "daily_demand": round(b["daily_demand"], 2),
            "expected_units_sold": round(b["expected_sales_units"], 2),
            "sell_through": round(b["expected_sell_through"], 3),
            "expected_revenue": round(b["expected_revenue"], 2),
            "gross_profit": round(b["gross_profit"], 2),
            "expected_waste_units": round(b["expected_waste_units"], 2),
            "terminal_inventory": round(b["terminal_inventory"], 2),
            "holding_cost": round(b["holding_cost"], 2),
            "economic_score": round(b["score"], 2),
        })
    return sweep
