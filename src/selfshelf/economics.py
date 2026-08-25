"""Economic reasoning layer.

The ML model only supplies a baseline demand forecast at the current price.
Everything economic — how demand responds to price, how expiry and inventory
create markdown pressure, what waste costs — is encoded explicitly here, so
the optimizer's behavior is driven by auditable economics rather than by
whatever shape a tree ensemble happened to memorize.
"""

import math
from dataclasses import dataclass
from typing import Dict

from .config import PricingConfig


# ---------------------------------------------------------------------------
# Primitive economic relationships
# ---------------------------------------------------------------------------

def price_effect(price: float, reference_price: float, elasticity: float) -> float:
    """Constant-elasticity demand multiplier: (P / P_ref) ** e.

    Equals 1.0 at the reference price; with e < 0 it rises as price falls
    and falls as price rises, guaranteeing the monotonic price-demand
    relationship the optimizer depends on.
    """
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    if price <= 0:
        raise ValueError("price must be positive")
    if elasticity > 0:
        raise ValueError("elasticity must be <= 0 (price up -> demand down)")
    return (price / reference_price) ** elasticity


def expiry_pressure(days_to_expiry: float, tau_days: float) -> float:
    """Smooth clearance pressure in [0, 1].

    exp(-(d - 1) / tau): ~1.0 with one day of shelf life left, decaying
    smoothly toward 0 as remaining life grows. No hard cliffs at arbitrary
    thresholds.
    """
    if tau_days <= 0:
        raise ValueError("tau_days must be positive")
    d = max(1.0, float(days_to_expiry))
    return math.exp(-(d - 1.0) / tau_days)


def days_of_supply(inventory_units: float, daily_demand: float) -> float:
    """How many days current stock lasts at the given demand rate.

    Zero demand yields infinity (stock never sells), which callers must
    treat as maximal inventory pressure rather than an error.
    """
    if inventory_units < 0:
        raise ValueError("inventory cannot be negative")
    if inventory_units == 0:
        return 0.0
    if daily_demand <= 0:
        return math.inf
    return inventory_units / daily_demand


def expected_waste_units(
    inventory_units: float, daily_demand: float, days_to_expiry: float
) -> float:
    """Units expected to remain unsold when the stock expires.

    max(0, inventory - demand_rate * remaining_days). Never negative.
    """
    if inventory_units < 0:
        raise ValueError("inventory cannot be negative")
    sellable = max(0.0, daily_demand) * max(0.0, days_to_expiry)
    return max(0.0, inventory_units - sellable)


def inventory_pressure(
    inventory_units: float, daily_demand: float, days_to_expiry: float
) -> float:
    """Fraction of current stock at risk of expiring unsold, in [0, 1].

    0.0 -> demand clears the stock comfortably before expiry;
    1.0 -> essentially none of it will sell in time.
    """
    if inventory_units <= 0:
        return 0.0
    waste = expected_waste_units(inventory_units, daily_demand, days_to_expiry)
    return min(1.0, waste / inventory_units)


# ---------------------------------------------------------------------------
# Product context + objective
# ---------------------------------------------------------------------------

@dataclass
class ProductContext:
    """Everything the economic layer needs to know about one product."""

    current_price: float
    retail_price: float
    unit_cost: float
    inventory_units: float
    days_to_expiry: float
    # ML forecast of daily demand at the current price.
    baseline_daily_demand: float
    # Own-price elasticity for this product's department.
    elasticity: float

    def __post_init__(self):
        if self.current_price <= 0 or self.retail_price <= 0:
            raise ValueError("prices must be positive")
        if self.unit_cost < 0:
            raise ValueError("cost cannot be negative")
        if self.inventory_units < 0:
            raise ValueError("inventory cannot be negative")
        if self.baseline_daily_demand < 0:
            raise ValueError("baseline demand cannot be negative")


class EconomicObjective:
    """Modular economic score for a candidate price.

        score(P) = revenue(P) - COGS(P) - expected waste loss(P)
                   - holding cost(P) - markdown friction(P)

    Expected sales are evaluated over min(days_to_expiry, planning horizon).
    Waste risk is evaluated over the full remaining shelf life and weighted
    by expiry pressure: a chronic overstock that expires in 90 days can
    still be fixed by future decisions, so its waste cost is discounted;
    stock expiring tomorrow bears the full loss.
    """

    def __init__(self, product: ProductContext, config: PricingConfig):
        self.product = product
        self.config = config
        self._pressure = expiry_pressure(
            product.days_to_expiry, config.expiry.tau_days
        )

    # -- demand ------------------------------------------------------------
    def demand_at(self, price: float) -> float:
        """Daily demand at a candidate price: ML baseline x price effect."""
        p = self.product
        return p.baseline_daily_demand * price_effect(
            price, p.current_price, p.elasticity
        )

    # -- components --------------------------------------------------------
    def breakdown(self, price: float) -> Dict[str, float]:
        """All objective components at a candidate price."""
        p = self.product
        cfg = self.config
        daily_demand = self.demand_at(price)

        sales_days = min(p.days_to_expiry, cfg.objective.planning_horizon_days)
        expected_sales = min(p.inventory_units, daily_demand * sales_days)

        revenue = price * expected_sales
        cogs = p.unit_cost * expected_sales

        waste_units = expected_waste_units(
            p.inventory_units, daily_demand, p.days_to_expiry
        )
        unit_loss = cfg.waste.unit_waste_loss(p.unit_cost)
        waste_cost = waste_units * unit_loss * self._pressure

        holding_cost = (
            cfg.inventory.holding_cost_per_unit_day
            * p.inventory_units
            * sales_days
        )

        markdown_depth = max(0.0, p.current_price - price)
        markdown_cost = (
            cfg.objective.markdown_friction * markdown_depth * expected_sales
        )

        return {
            "price": price,
            "daily_demand": daily_demand,
            "expected_sales_units": expected_sales,
            "expected_revenue": revenue,
            "cogs": cogs,
            "expected_waste_units": waste_units,
            "expected_waste_cost": waste_cost,
            "holding_cost": holding_cost,
            "markdown_cost": markdown_cost,
            "expiry_pressure": self._pressure,
            "score": (
                revenue - cogs - waste_cost - holding_cost - markdown_cost
            ),
        }

    def score(self, price: float) -> float:
        return self.breakdown(price)["score"]


# ---------------------------------------------------------------------------
# Price bounds
# ---------------------------------------------------------------------------

def price_bounds(product: ProductContext, config: PricingConfig):
    """Constrained (lower, upper) price range for one product.

    The floor blends smoothly from the healthy margin floor toward the
    clearance floor as expiry pressure rises, instead of switching on a
    hard days-to-expiry threshold. The ceiling never exceeds the retail
    (list) price, and never exceeds the current price unless price
    increases are explicitly enabled.
    """
    p = product
    c = config.constraints
    pressure = expiry_pressure(p.days_to_expiry, config.expiry.tau_days)

    margin_floor = p.unit_cost * (1.0 + c.min_margin)
    clearance_floor = p.unit_cost * c.clearance_floor_cost_fraction
    blended_floor = (
        margin_floor + (clearance_floor - margin_floor) * pressure
    )

    daily_drop_floor = p.current_price * (1.0 - c.max_daily_price_drop)
    lower = max(blended_floor, daily_drop_floor)

    if c.allow_price_increase:
        upper = min(
            p.retail_price,
            p.current_price * (1.0 + c.max_daily_price_increase),
        )
    else:
        upper = min(p.retail_price, p.current_price)

    # Degenerate cases (e.g. current price already below the margin floor):
    # never force a price increase; collapse the range to the current price.
    if lower > upper:
        lower = upper
    return lower, upper
