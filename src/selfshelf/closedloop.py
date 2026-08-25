"""Closed-loop daily re-optimization.

The multi-period path optimizer is open-loop: it plans day 0..N and assumes
the plan is followed while reality complies. This module closes the loop:

    plan  ->  act today's price  ->  observe actual sales
          ->  update inventory (sales, deliveries, expiry)
          ->  update the demand belief from the observation
          ->  RE-OPTIMIZE from the new state  ->  repeat

The controller explicitly distinguishes the *planned* path from the
*realized* state: every day records what was expected, what actually
happened, and what the new plan is. That distinction — not merely running
the optimizer repeatedly — is the point.

Division of responsibility (nothing economic is re-derived here):

- Each day's plan comes from the replenishment-aware path optimizer, which
  itself delegates all valuation to the frozen engine's primitives.
- Realized-outcome accounting (revenue, gross profit net of waste losses,
  midpoint holding cost) uses the exact conventions of the frozen
  ``backtest._simulate_sell_down`` so closed-loop results are directly
  comparable with the existing hold/immediate/path backtests.

Belief updates are deliberately modest and defensible: only the demand
LEVEL is smoothed toward observations (exponential smoothing, rate in
``config.adaptive``), and only on days where sales were not censored by a
stockout (a sold-out day proves demand >= sales, not demand == sales).
Elasticity is never re-estimated from a single day at a single price —
that would be fake learning; its uncertainty is reported separately by
``selfshelf.uncertainty``.

Data leakage rules enforced by construction: the planner sees only the
current state and *future* deliveries (offset >= 1); observed sales enter
the state only after the day's price has been acted; future sales are
never consulted.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

from .config import PricingConfig
from .economics import EconomicObjective, ProductContext, price_effect
from .replenishment import (
    ReplenishmentSchedule,
    optimize_path_with_replenishment,
)

_EPS = 1e-9

# environment(day, price, days_to_expiry_remaining) -> uncapped daily demand
Environment = Callable[[int, float, float], float]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetailState:
    """Everything the controller knows at the start of one day.

    ``old_inventory`` is the original expiring lot; ``fresh_inventory`` is
    stock from deliveries that have already arrived (assumed fresher than
    the old lot, FIFO rotation). ``baseline_daily_demand`` is the current
    demand-level belief, anchored at ``current_price``.
    """

    day: int
    old_inventory: float
    fresh_inventory: float
    current_price: float
    days_to_expiry: float
    baseline_daily_demand: float
    cumulative_sales: float = 0.0
    cumulative_revenue: float = 0.0
    cumulative_waste: float = 0.0
    replenishment_received: float = 0.0
    last_observed_sales: Optional[float] = None
    last_forecast_sales: Optional[float] = None

    @property
    def inventory(self) -> float:
        return self.old_inventory + self.fresh_inventory

    def as_dict(self) -> Dict[str, object]:
        return {
            "day": self.day,
            "inventory": round(self.inventory, 3),
            "old_inventory": round(self.old_inventory, 3),
            "fresh_inventory": round(self.fresh_inventory, 3),
            "current_price": round(self.current_price, 2),
            "days_to_expiry": round(self.days_to_expiry, 3),
            "baseline_daily_demand": round(self.baseline_daily_demand, 3),
            "cumulative_sales": round(self.cumulative_sales, 3),
            "cumulative_revenue": round(self.cumulative_revenue, 2),
            "cumulative_waste": round(self.cumulative_waste, 3),
            "replenishment_received": round(self.replenishment_received, 3),
            "last_observed_sales": (
                round(self.last_observed_sales, 3)
                if self.last_observed_sales is not None else None
            ),
            "last_forecast_sales": (
                round(self.last_forecast_sales, 3)
                if self.last_forecast_sales is not None else None
            ),
        }


def initial_state(product: ProductContext) -> RetailState:
    return RetailState(
        day=0,
        old_inventory=float(product.inventory_units),
        fresh_inventory=0.0,
        current_price=float(product.current_price),
        days_to_expiry=float(product.days_to_expiry),
        baseline_daily_demand=float(product.baseline_daily_demand),
    )


def observe_and_advance(
    state: RetailState,
    acted_price: float,
    observed_sales: float,
    forecast_sales: float,
    arrivals_tomorrow: float,
    elasticity: float,
    config: PricingConfig,
) -> RetailState:
    """Pure state transition for one day of the loop.

    Applies the observation to inventory (FIFO: the expiring lot sells
    first), books tomorrow's delivery, decrements shelf life (expiring the
    old lot when it reaches zero, using the plain waste identity — expired
    units are removed and counted, never resold), and updates the demand
    belief from the observation when it is not stockout-censored.
    """
    if observed_sales < -_EPS:
        raise ValueError("observed sales cannot be negative")
    available = state.inventory
    sales = min(float(observed_sales), available)
    sold_old = min(state.old_inventory, sales)
    sold_fresh = sales - sold_old

    old = max(0.0, state.old_inventory - sold_old)
    fresh = max(0.0, state.fresh_inventory - sold_fresh)

    # Belief update (demand level only). A sold-out day is censored: it
    # proves demand >= sales, so smoothing the level DOWN toward the
    # observed number would be wrong — the update is skipped.
    censored = sales >= available - _EPS
    baseline = state.baseline_daily_demand
    prior_at_acted = baseline * price_effect(
        max(acted_price, _EPS), state.current_price, elasticity
    )
    if config.adaptive.update_demand_beliefs and not censored:
        rate = config.adaptive.demand_learning_rate
        posterior = (1.0 - rate) * prior_at_acted + rate * sales
    else:
        posterior = prior_at_acted

    # Shelf life advances; the old lot expires when it hits zero.
    days_left = state.days_to_expiry - 1.0
    waste = 0.0
    if days_left <= _EPS:
        waste = old
        old = 0.0
        days_left = 0.0

    return replace(
        state,
        day=state.day + 1,
        old_inventory=old,
        fresh_inventory=fresh + max(0.0, float(arrivals_tomorrow)),
        current_price=float(acted_price),
        days_to_expiry=days_left,
        baseline_daily_demand=max(0.0, posterior),
        cumulative_sales=state.cumulative_sales + sales,
        cumulative_revenue=state.cumulative_revenue + acted_price * sales,
        cumulative_waste=state.cumulative_waste + waste,
        replenishment_received=(
            state.replenishment_received + max(0.0, float(arrivals_tomorrow))
        ),
        last_observed_sales=sales,
        last_forecast_sales=float(forecast_sales),
    )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

@dataclass
class DayRecord:
    """One day of the loop: the plan, the action, and what actually
    happened."""

    day: int
    days_to_expiry: float
    start_inventory: float
    planned_prices: List[float]      # today's re-optimized path (day 0 = act)
    acted_price: float
    forecast_sales: float            # planner's expectation for today
    observed_sales: float
    surprise: float                  # observed - forecast
    censored: bool                   # sales capped by a stockout
    arrivals_applied: float          # delivery landing tomorrow morning
    end_inventory: float
    baseline_before: float
    baseline_after: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "day": self.day,
            "days_to_expiry": round(self.days_to_expiry, 2),
            "start_inventory": round(self.start_inventory, 2),
            "planned_prices": [round(p, 2) for p in self.planned_prices],
            "acted_price": round(self.acted_price, 2),
            "forecast_sales": round(self.forecast_sales, 2),
            "observed_sales": round(self.observed_sales, 2),
            "surprise": round(self.surprise, 2),
            "censored": self.censored,
            "arrivals_applied": round(self.arrivals_applied, 2),
            "end_inventory": round(self.end_inventory, 2),
            "baseline_before": round(self.baseline_before, 3),
            "baseline_after": round(self.baseline_after, 3),
        }


@dataclass
class ClosedLoopResult:
    """Full episode: initial open-loop plan, day records, outcomes."""

    initial_plan: List[float]        # day-0 plan (the open-loop path)
    records: List[DayRecord]
    final_state: RetailState
    outcome: Dict[str, float]        # realized totals (backtest accounting)
    replans: int = 0                 # days where the new plan differed

    @property
    def acted_prices(self) -> List[float]:
        return [r.acted_price for r in self.records]


def run_closed_loop(
    product: ProductContext,
    config: PricingConfig,
    environment: Environment,
    replenishment: Optional[ReplenishmentSchedule] = None,
    reoptimize: bool = True,
    fixed_daily_prices: Optional[Sequence[float]] = None,
) -> ClosedLoopResult:
    """Deterministic closed-loop episode over the remaining shelf life.

    ``environment`` supplies each day's *actual* demand at the acted price
    — the controller never looks at it before acting, and never sees
    beyond today (no future leakage by construction). Determinism: the
    same product, config, environment, and delivery schedule produce an
    identical episode.

    ``reoptimize=False`` executes the day-0 plan open-loop under the same
    environment — the paired baseline for measuring what feedback is
    worth. ``fixed_daily_prices`` bypasses planning entirely and executes
    the given path (e.g. a hold path), so every strategy in a comparison
    shares this one accounting."""
    schedule = replenishment or ReplenishmentSchedule.empty()
    state = initial_state(product)
    days = max(1, int(product.days_to_expiry))
    fixed = (
        [float(p) for p in fixed_daily_prices]
        if fixed_daily_prices is not None else None
    )

    initial_plan: List[float] = []
    records: List[DayRecord] = []
    replans = 0

    # Realized-outcome accounting, frozen backtest conventions.
    revenue = 0.0
    sold = 0.0
    unit_days = 0.0

    for day in range(days):
        # -- plan from the current state (today only sees today) ----------
        context = ProductContext(
            current_price=state.current_price,
            retail_price=product.retail_price,
            unit_cost=product.unit_cost,
            inventory_units=state.old_inventory,
            days_to_expiry=state.days_to_expiry,
            baseline_daily_demand=state.baseline_daily_demand,
            elasticity=product.elasticity,
        )
        if fixed is not None:
            planned_prices = fixed[day:] or [fixed[-1]]
        elif day == 0 or reoptimize:
            plan = optimize_path_with_replenishment(
                context, config, schedule,
                fresh_inventory_units=state.fresh_inventory,
            )
            planned_prices = list(plan.daily_prices)
        else:
            # Open-loop: keep following the initial plan.
            planned_prices = initial_plan[day:] or [initial_plan[-1]]
        if day == 0:
            initial_plan = list(planned_prices)
        elif reoptimize:
            remaining_old = initial_plan[day:] or [initial_plan[-1]]
            horizon = min(len(planned_prices), len(remaining_old))
            if any(
                abs(planned_prices[i] - remaining_old[i]) > 0.005
                for i in range(horizon)
            ):
                replans += 1

        acted_price = planned_prices[0]

        # -- forecast, then observe ---------------------------------------
        objective = EconomicObjective(context, config)
        forecast = min(
            state.inventory, objective.demand_at(acted_price)
        )
        actual_demand = max(
            0.0, float(environment(day, acted_price, state.days_to_expiry))
        )
        observed = min(state.inventory, actual_demand)
        censored = observed >= state.inventory - _EPS and actual_demand > 0

        # Realized accounting (frozen backtest conventions: midpoint
        # holding approximation; waste booked at expiry).
        revenue += acted_price * observed
        sold += observed
        unit_days += state.inventory - observed / 2.0

        arrivals_tomorrow = schedule.arrivals_on(1)
        schedule = schedule.advanced(1)

        next_state = observe_and_advance(
            state, acted_price, observed, forecast, arrivals_tomorrow,
            product.elasticity, config,
        )
        records.append(DayRecord(
            day=day,
            days_to_expiry=state.days_to_expiry,
            start_inventory=state.inventory,
            planned_prices=planned_prices,
            acted_price=acted_price,
            forecast_sales=forecast,
            observed_sales=observed,
            surprise=observed - forecast,
            censored=censored,
            arrivals_applied=arrivals_tomorrow,
            end_inventory=next_state.inventory,
            baseline_before=state.baseline_daily_demand,
            baseline_after=next_state.baseline_daily_demand,
        ))
        state = next_state

    cost = product.unit_cost
    waste = state.cumulative_waste
    initial_inventory = float(product.inventory_units)
    supplied = initial_inventory + state.replenishment_received
    salvage = waste * config.waste.salvage_rate * cost
    holding_cost = config.inventory.holding_cost_per_unit_day * unit_days
    gross_profit = (
        revenue - cost * sold - waste * config.waste.unit_waste_loss(cost)
    )
    outcome = {
        "revenue": revenue,
        "units_sold": sold,
        "waste_units": waste,
        # Frozen backtest convention: terminal stock at the episode's end
        # includes the units that expired (waste is a subset of terminal),
        # plus any fresh deliveries still on the shelf. The identity is
        #     units_sold + terminal_inventory = initial + replenishment.
        "terminal_inventory": state.inventory + waste,
        "sell_through": sold / supplied if supplied > 0 else 1.0,
        "cash_recovered": revenue + salvage,
        "holding_cost": holding_cost,
        "gross_profit": gross_profit,
        "economic_value": gross_profit - holding_cost,
        "replenishment_received": state.replenishment_received,
    }
    return ClosedLoopResult(
        initial_plan=initial_plan,
        records=records,
        final_state=state,
        outcome=outcome,
        replans=replans,
    )


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def forecast_deviation_environment(
    product: ProductContext,
    config: PricingConfig,
    demand_factor: float = 1.0,
    noise: Optional[Sequence[float]] = None,
) -> Environment:
    """Actual demand = the engine's own demand curve scaled by a factor.

    A deterministic what-if: ``demand_factor=0.4`` means real customers
    buy 40% of what the model expects at every price. Useful for the
    "expected 20, sold 8" demos and for testing the feedback loop without
    a separate ground-truth simulator. Clearly synthetic.
    """
    objective = EconomicObjective(product, config)

    def env(day: int, price: float, days_left: float) -> float:
        demand = objective.demand_at(price) * demand_factor
        if noise is not None and day < len(noise):
            demand *= float(noise[day])
        return demand

    return env


def simulator_environment(
    base_demand: float,
    retail_price: float,
    true_elasticity: float,
    config: PricingConfig,
    noise: Sequence[float],
) -> Environment:
    """The synthetic data-generating process, matching the frozen
    ``backtest._simulate_sell_down`` formula exactly: base demand x true
    price effect (vs the LIST price) x freshness decay x lognormal noise.
    """
    from .economics import expiry_pressure

    def env(day: int, price: float, days_left: float) -> float:
        freshness = 1.0 - (
            config.expiry.freshness_sensitivity
            * expiry_pressure(days_left, config.expiry.tau_days)
        )
        price_eff = (price / retail_price) ** true_elasticity
        n = float(noise[day]) if day < len(noise) else 1.0
        return base_demand * price_eff * freshness * n

    return env
