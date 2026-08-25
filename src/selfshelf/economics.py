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


def waste_risk_weight(
    days_to_expiry: float, planning_horizon_days: float, tau_days: float
) -> float:
    """How much of the projected at-expiry waste loss to charge today.

    Waste that falls inside the current decision window is certain to
    happen under the model's demand forecast, so it is charged in full.
    Waste projected beyond the window can still be averted by future
    repricing decisions, so it is discounted exponentially with the time
    remaining after the window closes:

        weight = 1.0                              if expiry <= horizon
        weight = exp(-(expiry - horizon) / tau)   otherwise

    Continuous at the horizon boundary; approaches 0 for long-dated stock.
    """
    if tau_days <= 0:
        raise ValueError("tau_days must be positive")
    overhang = max(0.0, float(days_to_expiry) - float(planning_horizon_days))
    return math.exp(-overhang / tau_days)


def unit_days_held(
    inventory_units: float, daily_demand: float, window_days: float
) -> float:
    """Unit-days of stock carried over the window as it sells down.

    Inventory declines linearly at the demand rate until it sells out or
    the window ends: the integral of remaining stock over time. This is
    what carrying cost should apply to — a faster sell-down (e.g. after a
    markdown) genuinely reduces units held.
    """
    if inventory_units < 0:
        raise ValueError("inventory cannot be negative")
    if window_days <= 0 or inventory_units == 0:
        return 0.0
    if daily_demand <= 0:
        return inventory_units * window_days
    sellout_days = inventory_units / daily_demand
    if sellout_days >= window_days:
        return inventory_units * window_days - daily_demand * window_days**2 / 2
    return inventory_units**2 / (2.0 * daily_demand)


def break_even_unit_uplift(
    current_price: float, markdown_price: float, unit_cost: float
) -> float:
    """Unit-sales multiplier needed for a markdown to hold gross profit.

        Q1 / Q0 = (P0 - C) / (P1 - C)

    Returns inf when the markdown price is at or below cost (no volume can
    recover gross profit; such a price is only justifiable by waste/terminal
    -stock economics, never by demand stimulation alone), and 1.0 when
    there is no markdown.
    """
    if markdown_price > current_price:
        raise ValueError("markdown_price must not exceed current_price")
    current_margin = current_price - unit_cost
    markdown_margin = markdown_price - unit_cost
    if current_margin <= 0:
        # Already selling at/below cost: any volume at a non-negative
        # margin is an improvement; no uplift hurdle applies.
        return 1.0
    if markdown_margin <= 0:
        return math.inf
    return current_margin / markdown_margin


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

        score(P) = gross profit(P)              [revenue - COGS, dollars]
                   - expected waste loss(P)
                   - holding cost(P)
                   - markdown friction(P)

    Accounting semantics (each dollar is counted exactly once):

    - Expected sales are evaluated over min(days_to_expiry, horizon) and
      can never exceed inventory on hand.
    - COGS is charged only on units expected to sell; unsold units keep
      their book value unless they expire.
    - A unit that expires unsold loses (cost - salvage) plus disposal.
      That loss is charged in full when expiry falls inside the decision
      window, and discounted by ``waste_risk_weight`` when it lies beyond
      it (future repricing can still avert it).
    - Holding cost applies to unit-days actually carried as the stock
      sells down, so faster sell-through genuinely reduces it.
    - Markdown friction is a small operational cost per discounted dollar
      actually given away; it never justifies a markdown, only tempers
      marginal ones.
    """

    def __init__(self, product: ProductContext, config: PricingConfig):
        self.product = product
        self.config = config
        self._pressure = expiry_pressure(
            product.days_to_expiry, config.expiry.tau_days
        )
        self._waste_weight = waste_risk_weight(
            product.days_to_expiry,
            config.objective.planning_horizon_days,
            config.expiry.tau_days,
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
        terminal_inventory = p.inventory_units - expected_sales
        sell_through = (
            expected_sales / p.inventory_units if p.inventory_units > 0 else 1.0
        )

        revenue = price * expected_sales
        cogs = p.unit_cost * expected_sales
        gross_profit = revenue - cogs

        waste_units = expected_waste_units(
            p.inventory_units, daily_demand, p.days_to_expiry
        )
        unit_loss = cfg.waste.unit_waste_loss(p.unit_cost)
        waste_cost = waste_units * unit_loss * self._waste_weight

        units_days = unit_days_held(
            p.inventory_units, daily_demand, sales_days
        )
        holding_cost = cfg.inventory.holding_cost_per_unit_day * units_days

        markdown_depth = max(0.0, p.current_price - price)
        markdown_cost = (
            cfg.objective.markdown_friction * markdown_depth * expected_sales
        )

        return {
            "price": price,
            "daily_demand": daily_demand,
            "expected_sales_units": expected_sales,
            "expected_sell_through": sell_through,
            "terminal_inventory": terminal_inventory,
            "expected_revenue": revenue,
            "cogs": cogs,
            "gross_profit": gross_profit,
            "expected_waste_units": waste_units,
            "waste_risk_weight": self._waste_weight,
            "expected_waste_cost": waste_cost,
            "unit_days_held": units_days,
            "holding_cost": holding_cost,
            "markdown_cost": markdown_cost,
            "expiry_pressure": self._pressure,
            "score": (
                gross_profit - waste_cost - holding_cost - markdown_cost
            ),
        }

    def score(self, price: float) -> float:
        return self.breakdown(price)["score"]


# ---------------------------------------------------------------------------
# Price paths (foundation for staged markdowns / timing decisions)
# ---------------------------------------------------------------------------

def evaluate_price_path(
    product: ProductContext,
    config: PricingConfig,
    schedule,
) -> Dict[str, float]:
    """Economic value of a piecewise-constant price path.

    ``schedule`` is a list of ``(days, price)`` segments applied in order,
    truncated to the decision window min(days_to_expiry, horizon). Uses the
    same accounting as ``EconomicObjective`` — a single-segment path at
    price P reproduces ``breakdown(P)['score']`` exactly.

    This is deliberately a *one-shot evaluation of a fixed path*, not a
    dynamic program: it supports "markdown now vs wait N days" comparisons
    and staged-markdown analysis, while full multi-period optimization
    (re-deciding each day as sales are observed) remains future work.
    """
    p = product
    cfg = config
    window = min(p.days_to_expiry, cfg.objective.planning_horizon_days)

    remaining = p.inventory_units
    elapsed = 0.0
    revenue = cogs = holding = friction = 0.0
    last_daily_demand = 0.0

    for days, price in schedule:
        if elapsed >= window:
            break
        segment_days = min(float(days), window - elapsed)
        daily_demand = p.baseline_daily_demand * price_effect(
            price, p.current_price, p.elasticity
        )
        sales = min(remaining, daily_demand * segment_days)
        revenue += price * sales
        cogs += p.unit_cost * sales
        holding += (
            cfg.inventory.holding_cost_per_unit_day
            * unit_days_held(remaining, daily_demand, segment_days)
        )
        depth = max(0.0, p.current_price - price)
        friction += cfg.objective.markdown_friction * depth * sales
        remaining -= sales
        elapsed += segment_days
        last_daily_demand = daily_demand

    # Stock left at the window's end keeps selling at the final segment's
    # rate until expiry; whatever still cannot sell is projected waste.
    tail_days = max(0.0, p.days_to_expiry - elapsed)
    projected_waste = max(0.0, remaining - last_daily_demand * tail_days)
    unit_loss = cfg.waste.unit_waste_loss(p.unit_cost)
    weight = waste_risk_weight(
        p.days_to_expiry, cfg.objective.planning_horizon_days,
        cfg.expiry.tau_days,
    )
    waste_cost = projected_waste * unit_loss * weight

    gross_profit = revenue - cogs
    return {
        "expected_revenue": revenue,
        "gross_profit": gross_profit,
        "expected_waste_units": projected_waste,
        "expected_waste_cost": waste_cost,
        "holding_cost": holding,
        "markdown_cost": friction,
        "terminal_inventory": remaining,
        "score": gross_profit - waste_cost - holding - friction,
    }


def compare_markdown_timing(
    product: ProductContext,
    config: PricingConfig,
    markdown_price: float,
    wait_days: float = 3.0,
):
    """Simplified two-path timing comparison.

        Path now:  markdown immediately for the whole window
        Path wait: hold the current price for ``wait_days``, then apply
                   the same markdown for the rest of the window

    Returns a dict with both path values and ``advantage_now`` (positive
    when acting immediately is worth more), or None when the window is too
    short for waiting to be a distinct option.
    """
    window = min(
        product.days_to_expiry, config.objective.planning_horizon_days
    )
    wait = min(float(wait_days), window - 1.0)
    if wait < 1.0:
        return None
    now = evaluate_price_path(
        product, config, [(window, markdown_price)]
    )
    later = evaluate_price_path(
        product, config,
        [(wait, product.current_price), (window - wait, markdown_price)],
    )
    return {
        "wait_days": wait,
        "value_now": now["score"],
        "value_wait": later["score"],
        "advantage_now": now["score"] - later["score"],
    }


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
