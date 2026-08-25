"""Replenishment-aware inventory and price-path evaluation.

The frozen engine models one pool of stock that only ever shrinks. This
module adds *known future deliveries*:

    inventory tomorrow = inventory today - sales + arrivals

without introducing a second economic model. Every dollar is computed by
the frozen primitives (``EconomicObjective.demand_at``, ``unit_days_held``,
``waste_risk_weight``, ``unit_waste_loss``) using the exact accounting
conventions of ``economics.evaluate_price_path`` — with an empty
replenishment schedule the replay below reproduces that frozen evaluator's
totals exactly (proven in tests), so replenishment awareness can never
drift into different baseline economics.

Inventory model (two lots, time-respecting):

    old lot    the stock currently on the shelf; expires at
               ``days_to_expiry`` exactly as the frozen engine assumes
    fresh lot  future deliveries; a delivery on day d becomes sellable at
               the START of day d and never earlier

Assumptions stated openly rather than hidden:

- Deliveries are assumed fresh: they do not expire within the planning
  window (which is bounded by the old lot's expiry anyway). Expired stock
  is never rotated into the fresh lot.
- Sales rotate FIFO: the expiring lot sells first, matching standard shelf
  rotation. Only the old lot can therefore generate waste, priced with the
  frozen waste economics.
- A schedule contains only FUTURE deliveries (day offset >= 1). Stock that
  has already arrived belongs in current inventory; letting a "day 0"
  delivery into the planner would double-count it, so it is a hard error.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .config import PricingConfig
from .economics import (
    EconomicObjective,
    ProductContext,
    unit_days_held,
    waste_risk_weight,
)
from .pathopt import (
    DEFAULT_MAX_MOVES,
    DEFAULT_NUM_LEVELS,
    PathResult,
    Schedule,
    _iter_schedules,
    candidate_price_levels,
    daily_to_schedule,
    path_horizon_days,
    path_price_floor,
    schedule_to_daily,
)

_EPS = 1e-9


@dataclass(frozen=True)
class ReplenishmentSchedule:
    """Known future deliveries for one product, keyed by day offset.

    Offset 1 = tomorrow. Offset 0 (today) or negative offsets are invalid:
    anything that has already arrived must be counted in current inventory,
    never fed to the planner as a future event.
    """

    arrivals: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self):
        cleaned: Dict[int, float] = {}
        for day, qty in dict(self.arrivals).items():
            day = int(day)
            qty = float(qty)
            if day < 1:
                raise ValueError(
                    "replenishment offsets must be >= 1 (future days); "
                    "already-arrived stock belongs in current inventory"
                )
            if qty < 0:
                raise ValueError("replenishment quantities cannot be negative")
            if qty > 0:
                cleaned[day] = cleaned.get(day, 0.0) + qty
        object.__setattr__(self, "arrivals", cleaned)

    @classmethod
    def empty(cls) -> "ReplenishmentSchedule":
        return cls({})

    @property
    def is_empty(self) -> bool:
        return not self.arrivals

    def arrivals_on(self, day: int) -> float:
        return self.arrivals.get(int(day), 0.0)

    def total_units(self, within_days: Optional[int] = None) -> float:
        return sum(
            qty for day, qty in self.arrivals.items()
            if within_days is None or day <= within_days
        )

    def next_arrival(self) -> Optional[Tuple[int, float]]:
        if not self.arrivals:
            return None
        day = min(self.arrivals)
        return day, self.arrivals[day]

    def advanced(self, days: int = 1) -> "ReplenishmentSchedule":
        """The schedule as seen ``days`` later. Deliveries whose offset
        would drop below 1 are no longer *future* events — the caller is
        responsible for having applied them to inventory."""
        if days < 0:
            raise ValueError("cannot advance a schedule backwards")
        return ReplenishmentSchedule({
            day - days: qty
            for day, qty in self.arrivals.items()
            if day - days >= 1
        })

    def as_dict(self) -> Dict[str, object]:
        nxt = self.next_arrival()
        return {
            "arrivals": [
                {"day": int(day), "units": float(qty)}
                for day, qty in sorted(self.arrivals.items())
            ],
            "total_units": float(self.total_units()),
            "next_arrival": (
                {"day": int(nxt[0]), "units": float(nxt[1])}
                if nxt else None
            ),
        }


# ---------------------------------------------------------------------------
# Replenishment-aware path replay (frozen accounting, per-day)
# ---------------------------------------------------------------------------

def _replay(
    product: ProductContext,
    config: PricingConfig,
    daily_prices: Sequence[float],
    schedule: ReplenishmentSchedule,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Day-by-day replay of a price path with future deliveries.

    Identical per-day accounting to ``pathopt.path_trajectory`` (which is
    proven to reproduce the frozen ``evaluate_price_path`` totals); the
    only additions are the arrival events and the old/fresh lot split.
    """
    p = product
    cfg = config
    objective = EconomicObjective(p, cfg)
    window = min(p.days_to_expiry, cfg.objective.planning_horizon_days)

    old = p.inventory_units       # expires at p.days_to_expiry
    fresh = 0.0                   # delivered during the window; no expiry
    received = 0.0
    elapsed = 0.0
    revenue = cogs = holding = friction = 0.0
    last_demand = 0.0
    rows: List[Dict[str, float]] = []

    for day, price in enumerate(daily_prices):
        if elapsed >= window:
            break
        day_len = min(1.0, window - elapsed)
        arrived = schedule.arrivals_on(day)
        start_inventory = old + fresh
        fresh += arrived
        received += arrived
        available = old + fresh

        demand = objective.demand_at(price)
        sales = min(available, demand * day_len)
        sold_old = min(old, sales)      # FIFO: expiring stock sells first
        sold_fresh = sales - sold_old

        unit_days = unit_days_held(available, demand, day_len)
        depth = max(0.0, p.current_price - price)
        day_holding = cfg.inventory.holding_cost_per_unit_day * unit_days
        day_friction = cfg.objective.markdown_friction * depth * sales

        revenue += price * sales
        cogs += p.unit_cost * sales
        holding += day_holding
        friction += day_friction

        rows.append({
            "day": day,
            "price": price,
            "daily_demand": demand,
            "expected_sales_units": sales,
            "start_inventory": start_inventory,
            "replenishment_received": arrived,
            "available_inventory": available,
            "end_inventory": available - sales,
            "expected_revenue": price * sales,
            "gross_profit": (price - p.unit_cost) * sales,
            "holding_cost": day_holding,
            "markdown_cost": day_friction,
        })

        old -= sold_old
        fresh -= sold_fresh
        elapsed += day_len
        last_demand = demand

    # Tail (same convention as the frozen evaluator): stock keeps selling
    # at the final day's rate until the old lot expires. Rotation means
    # that demand serves the old lot first, so only its unsold remainder
    # becomes projected waste; fresh deliveries do not expire within the
    # model's knowledge and stay as terminal inventory.
    tail_days = max(0.0, p.days_to_expiry - elapsed)
    projected_waste = max(0.0, old - last_demand * tail_days)
    unit_loss = cfg.waste.unit_waste_loss(p.unit_cost)
    weight = waste_risk_weight(
        p.days_to_expiry, cfg.objective.planning_horizon_days,
        cfg.expiry.tau_days,
    )
    waste_cost = projected_waste * unit_loss * weight
    if rows:
        rows[-1]["projected_waste_units"] = projected_waste
        rows[-1]["projected_waste_cost"] = waste_cost

    gross_profit = revenue - cogs
    totals = {
        "expected_revenue": revenue,
        "gross_profit": gross_profit,
        "expected_waste_units": projected_waste,
        "expected_waste_cost": waste_cost,
        "holding_cost": holding,
        "markdown_cost": friction,
        "terminal_inventory": old + fresh,
        "score": gross_profit - waste_cost - holding - friction,
        "replenishment_units": received,
        "old_terminal_inventory": old,
        "fresh_terminal_inventory": fresh,
    }
    return totals, rows


def evaluate_price_path_with_replenishment(
    product: ProductContext,
    config: PricingConfig,
    schedule: Schedule,
    replenishment: ReplenishmentSchedule,
) -> Dict[str, float]:
    """Economic value of a price path given known future deliveries.

    With an empty replenishment schedule this equals the frozen
    ``economics.evaluate_price_path`` exactly (see tests).
    """
    horizon = path_horizon_days(product, config)
    daily = schedule_to_daily(schedule, horizon)
    totals, _ = _replay(product, config, daily, replenishment)
    return totals


def replenishment_trajectory(
    product: ProductContext,
    config: PricingConfig,
    schedule: Schedule,
    replenishment: ReplenishmentSchedule,
) -> List[Dict[str, float]]:
    """Day-by-day expected trajectory including delivery events."""
    horizon = path_horizon_days(product, config)
    daily = schedule_to_daily(schedule, horizon)
    _, rows = _replay(product, config, daily, replenishment)
    return rows


# ---------------------------------------------------------------------------
# Replenishment-aware path optimization
# ---------------------------------------------------------------------------

def optimize_path_with_replenishment(
    product: ProductContext,
    config: PricingConfig,
    replenishment: ReplenishmentSchedule,
    single_price: Optional[float] = None,
    max_moves: int = DEFAULT_MAX_MOVES,
    num_levels: int = DEFAULT_NUM_LEVELS,
) -> PathResult:
    """Exhaustive staged-markdown search that sees future deliveries.

    Identical candidate class, constraints, enumeration order, and
    tie-breaking to the frozen ``pathopt.optimize_path`` — only the
    *valuation* differs, and only by incorporating known arrivals. With an
    empty schedule the result is identical to ``optimize_path`` (tests).
    """
    horizon = path_horizon_days(product, config)
    levels = candidate_price_levels(product, config, num_levels)
    cur = product.current_price

    if single_price is not None:
        sp = float(single_price)
        floor = path_price_floor(product, config)
        if floor - _EPS <= sp < cur - _EPS and all(
            abs(sp - level) > _EPS for level in levels
        ):
            levels = sorted([*levels, sp], reverse=True)

    best_schedule: Schedule = [(horizon, cur)]
    best_eval = evaluate_price_path_with_replenishment(
        product, config, best_schedule, replenishment
    )
    best_score = best_eval["score"]
    hold_eval = best_eval
    n_candidates = 0

    for schedule in _iter_schedules(
        levels, cur, horizon, max_moves,
        config.constraints.max_daily_price_drop,
    ):
        n_candidates += 1
        if n_candidates == 1:
            continue  # hold path already evaluated above
        evaluation = evaluate_price_path_with_replenishment(
            product, config, schedule, replenishment
        )
        if evaluation["score"] > best_score + _EPS:
            best_schedule = schedule
            best_eval = evaluation
            best_score = evaluation["score"]

    single_eval = None
    if single_price is not None and single_price < cur - _EPS:
        single_eval = evaluate_price_path_with_replenishment(
            product, config, [(horizon, float(single_price))], replenishment
        )

    result = PathResult(
        product=product,
        horizon_days=horizon,
        schedule=best_schedule,
        daily_prices=schedule_to_daily(best_schedule, horizon),
        evaluation=best_eval,
        hold=hold_eval,
        single=single_eval,
        single_price=(
            float(single_price) if single_eval is not None else None
        ),
        n_candidates=n_candidates,
    )
    result.reasons = _build_replenishment_reasons(result, replenishment)
    return result


def _build_replenishment_reasons(
    result: PathResult, replenishment: ReplenishmentSchedule
) -> List[str]:
    """Explanation rows derived from the actual schedule economics."""
    reasons: List[str] = []
    ev, hold = result.evaluation, result.hold

    if not replenishment.is_empty:
        nxt = replenishment.next_arrival()
        reasons.append(
            f"known future deliveries: {replenishment.total_units():.0f} "
            f"unit(s), next in {nxt[0]} day(s) (+{nxt[1]:.0f} units)"
        )
    if result.action == "Hold":
        reasons.append(
            "no staged markdown schedule beats holding the current price, "
            "accounting for known future deliveries"
            if not replenishment.is_empty else
            "no staged markdown schedule beats holding the current price "
            "over the remaining shelf life"
        )
        return reasons

    moves = daily_to_schedule(result.daily_prices)
    steps = ", ".join(
        f"${price:.2f} from day {sum(int(d) for d, _ in moves[:i])}"
        for i, (_, price) in enumerate(moves)
    )
    reasons.append(f"recommended schedule: {steps}")
    reasons.append(
        f"expected economic value improves by "
        f"${result.improvement_vs_hold:.2f} vs holding "
        f"(${ev['score']:.2f} vs ${hold['score']:.2f})"
    )
    waste_delta = hold["expected_waste_units"] - ev["expected_waste_units"]
    if waste_delta > 0.5:
        reasons.append(
            f"the schedule avoids ~{waste_delta:.0f} units of expected "
            f"waste versus holding"
        )
    if not replenishment.is_empty and ev["expected_waste_units"] > 0.5:
        reasons.append(
            f"~{ev['expected_waste_units']:.0f} unit(s) of the current "
            f"(expiring) stock are still expected to expire before incoming "
            f"deliveries change the picture"
        )
    return reasons


# ---------------------------------------------------------------------------
# Building schedules from replenishment data
# ---------------------------------------------------------------------------

def schedules_from_events(
    events,
    reference_date,
) -> Dict[str, ReplenishmentSchedule]:
    """Per-product schedules from (date, product_id, quantity) records.

    ``events`` is a DataFrame with ``date`` (datetime), ``product_id`` and
    ``quantity`` columns. ``reference_date`` is "today" in the dataset's
    timeline; deliveries dated on/before it are assumed to already be part
    of current inventory and are excluded (time-respecting: the planner
    only ever sees genuinely future stock).
    """
    import pandas as pd

    reference = pd.Timestamp(reference_date).normalize()
    schedules: Dict[str, Dict[int, float]] = {}
    if events is None or not len(events):
        return {}
    for _, row in events.iterrows():
        day = int((pd.Timestamp(row["date"]).normalize() - reference).days)
        if day < 1:
            continue  # already arrived (or arrives today): in inventory
        qty = float(row["quantity"])
        if qty <= 0:
            continue
        pid = str(row["product_id"])
        per = schedules.setdefault(pid, {})
        per[day] = per.get(day, 0.0) + qty
    return {
        pid: ReplenishmentSchedule(arrivals)
        for pid, arrivals in schedules.items()
    }
