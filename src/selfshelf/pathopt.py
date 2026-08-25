"""Multi-period markdown path optimization.

Answers "what pricing path should this product follow over its remaining
shelf life?" rather than only "what price should it be today?".

This module is an adapter over the frozen economic engine: it *generates and
constrains* candidate price schedules, but every schedule is *valued*
exclusively by ``economics.evaluate_price_path`` — the audited one-shot path
evaluator. No second economic model is introduced here.

Method: exhaustive enumeration over an operationally realistic policy class
(piecewise-constant schedules with at most ``max_moves`` downward price
moves, at daily granularity). Retailers do not re-sticker shelves every few
hours, so staged markdowns — hold, then mark down, then clear — cover the
paths that matter, and exhaustive search over them is deterministic,
exactly optimal within the class, and trivially auditable against a
brute-force reference (see tests/test_pathopt.py).

Because the *hold-forever* schedule is always in the candidate set and ties
are broken toward holding and later/shallower markdowns, the optimizer can
never recommend a markdown path that does not strictly beat holding — a
markdown today is never treated as free when marking down tomorrow instead
would preserve more value.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .config import PricingConfig
from .economics import (
    EconomicObjective,
    ProductContext,
    evaluate_price_path,
    expiry_pressure,
    unit_days_held,
    waste_risk_weight,
)

# At most this many downward price moves per path (a move on day 0 counts).
DEFAULT_MAX_MOVES = 2
# Candidate price levels spanning floor..current.
DEFAULT_NUM_LEVELS = 9

_EPS = 1e-9

Schedule = List[Tuple[float, float]]  # [(days, price), ...]


def path_price_floor(product: ProductContext, config: PricingConfig) -> float:
    """Lowest price any day of a path may reach.

    This is the same expiry-blended margin/clearance floor that
    ``economics.price_bounds`` applies, *without* the single-move
    ``max_daily_price_drop`` term: a path may walk below the one-day drop
    limit through successive moves, but each individual move is still
    limited (enforced during enumeration). ``tests/test_pathopt.py``
    cross-checks this against ``price_bounds`` so the two can never drift.
    """
    c = config.constraints
    pressure = expiry_pressure(product.days_to_expiry, config.expiry.tau_days)
    margin_floor = product.unit_cost * (1.0 + c.min_margin)
    clearance_floor = product.unit_cost * c.clearance_floor_cost_fraction
    floor = margin_floor + (clearance_floor - margin_floor) * pressure
    # Never force a price increase: the current price is always feasible.
    return min(floor, product.current_price)


def path_horizon_days(product: ProductContext, config: PricingConfig) -> int:
    """Whole days in the path decision window (at least 1)."""
    window = min(
        product.days_to_expiry, config.objective.planning_horizon_days
    )
    return max(1, int(math.floor(window)))


def candidate_price_levels(
    product: ProductContext,
    config: PricingConfig,
    num_levels: int = DEFAULT_NUM_LEVELS,
) -> List[float]:
    """Whole-cent candidate prices from the path floor up to the current
    price (which is kept exact), descending. Markdown levels closer to the
    current price than ``min_meaningful_markdown`` are dropped as noise,
    matching the one-shot optimizer's convention."""
    cur = product.current_price
    floor = path_price_floor(product, config)
    if floor >= cur - _EPS or num_levels <= 1:
        return [cur]

    min_step = config.constraints.min_meaningful_markdown * cur
    levels = {cur}
    span = cur - floor
    for i in range(1, num_levels):
        raw = cur - span * i / (num_levels - 1)
        cents = round(raw, 2)
        if cents < floor - _EPS:
            cents = math.ceil(floor * 100.0) / 100.0
        if cents > cur - min_step or cents < floor - _EPS:
            continue
        levels.add(cents)
    return sorted(levels, reverse=True)


def _iter_schedules(
    levels: Sequence[float],
    current_price: float,
    horizon: int,
    max_moves: int,
    max_daily_drop: float,
) -> Iterator[Schedule]:
    """Yield candidate schedules, most conservative first.

    Ordering matters: the optimizer keeps the first schedule that achieves
    the best score (strict improvement to replace), so yielding hold first,
    then later/shallower moves before earlier/deeper ones, breaks ties
    toward not marking down and toward waiting.
    """
    cur = current_price

    def step_ok(from_price: float, to_price: float) -> bool:
        return to_price >= from_price * (1.0 - max_daily_drop) - _EPS

    yield [(horizon, cur)]
    if max_moves < 1:
        return

    below = [p for p in levels if p < cur - _EPS]

    # One move: hold for t days (0 = act immediately), then one markdown.
    for p in below:
        if not step_ok(cur, p):
            continue
        for t in range(horizon - 1, -1, -1):
            if t == 0:
                yield [(horizon, p)]
            else:
                yield [(t, cur), (horizon - t, p)]

    if max_moves < 2 or horizon < 2:
        return

    # Two moves: hold for t1 days, mark to p1 until t2, then deeper to p2.
    for i, p1 in enumerate(below):
        if not step_ok(cur, p1):
            continue
        for p2 in below[i + 1:]:
            if not step_ok(p1, p2):
                continue
            for t2 in range(horizon - 1, 0, -1):
                for t1 in range(t2 - 1, -1, -1):
                    if t1 == 0:
                        yield [(t2, p1), (horizon - t2, p2)]
                    else:
                        yield [
                            (t1, cur), (t2 - t1, p1), (horizon - t2, p2)
                        ]


def schedule_to_daily(schedule: Schedule, horizon: int) -> List[float]:
    """Expand segment form to one price per whole day of the horizon."""
    daily: List[float] = []
    for days, price in schedule:
        daily.extend([price] * int(round(days)))
    if daily:
        while len(daily) < horizon:
            daily.append(daily[-1])
    return daily[:horizon]


def daily_to_schedule(daily_prices: Sequence[float]) -> Schedule:
    """Compress a per-day price list back to [(days, price)] segments."""
    schedule: Schedule = []
    for price in daily_prices:
        if schedule and abs(schedule[-1][1] - price) <= _EPS:
            days, p = schedule[-1]
            schedule[-1] = (days + 1, p)
        else:
            schedule.append((1, float(price)))
    return schedule


@dataclass
class PathResult:
    """Optimized multi-period schedule plus its comparison baselines."""

    product: ProductContext
    horizon_days: int
    schedule: Schedule
    daily_prices: List[float]
    evaluation: Dict[str, float]          # engine valuation of the schedule
    hold: Dict[str, float]                # hold current price throughout
    single: Optional[Dict[str, float]]    # immediate one-shot markdown
    single_price: Optional[float]
    n_candidates: int = 0
    reasons: List[str] = field(default_factory=list)

    @property
    def n_moves(self) -> int:
        cur = self.product.current_price
        distinct = [p for _, p in self.schedule if p < cur - _EPS]
        return len(distinct)

    @property
    def action(self) -> str:
        if self.n_moves == 0:
            return "Hold"
        if self.n_moves == 1 and self.daily_prices[0] < (
            self.product.current_price - _EPS
        ):
            return "Immediate markdown"
        return "Staged markdown"

    @property
    def improvement_vs_hold(self) -> float:
        return self.evaluation["score"] - self.hold["score"]

    @property
    def improvement_vs_single(self) -> Optional[float]:
        if self.single is None:
            return None
        return self.evaluation["score"] - self.single["score"]


def optimize_path(
    product: ProductContext,
    config: PricingConfig,
    single_price: Optional[float] = None,
    max_moves: int = DEFAULT_MAX_MOVES,
    num_levels: int = DEFAULT_NUM_LEVELS,
) -> PathResult:
    """Exhaustively search the staged-markdown policy class.

    ``single_price`` is the existing one-shot recommendation (from
    ``optimizer.optimize_product``); when provided, the "immediate
    markdown" baseline replays that exact price for the whole window so
    the path can be compared against what the single-period engine would
    have done today.

    Fully deterministic: no randomness is involved anywhere.
    """
    horizon = path_horizon_days(product, config)
    levels = candidate_price_levels(product, config, num_levels)
    cur = product.current_price

    best_schedule: Schedule = [(horizon, cur)]
    best_eval = evaluate_price_path(product, config, best_schedule)
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
        evaluation = evaluate_price_path(product, config, schedule)
        if evaluation["score"] > best_score + _EPS:
            best_schedule = schedule
            best_eval = evaluation
            best_score = evaluation["score"]

    single_eval = None
    if single_price is not None and single_price < cur - _EPS:
        single_eval = evaluate_price_path(
            product, config, [(horizon, float(single_price))]
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
    result.reasons = _build_path_reasons(result)
    return result


def _build_path_reasons(result: PathResult) -> List[str]:
    """Deterministic explanation derived from the actual path economics."""
    reasons: List[str] = []
    p = result.product
    ev, hold = result.evaluation, result.hold

    if result.action == "Hold":
        reasons.append(
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
    if result.single is not None:
        adv = result.improvement_vs_single
        if adv is not None and adv > 0.01:
            reasons.append(
                f"waiting before the markdown preserves ${adv:.2f} of "
                f"margin versus marking down to "
                f"${result.single_price:.2f} immediately"
            )
        elif adv is not None and abs(adv) <= 0.01:
            reasons.append(
                "acting immediately and the staged schedule are "
                "economically equivalent"
            )
    if result.daily_prices[0] >= p.current_price - _EPS:
        first_move = next(
            (i for i, price in enumerate(result.daily_prices)
             if price < p.current_price - _EPS), None,
        )
        if first_move:
            reasons.append(
                f"holding the current price for {first_move} day(s) first "
                f"captures full-margin sales from demand that exists at "
                f"today's price"
            )
    return reasons


def path_trajectory(
    product: ProductContext,
    config: PricingConfig,
    schedule: Schedule,
) -> List[Dict[str, float]]:
    """Day-by-day expected trajectory of a schedule.

    Purely a *decomposition* of the frozen evaluator's accounting into
    daily rows — demand comes from ``EconomicObjective.demand_at``, and the
    per-day sales/holding/friction sums are proven (in tests) to reproduce
    ``evaluate_price_path``'s totals exactly.
    """
    p = product
    cfg = config
    objective = EconomicObjective(p, cfg)
    window = min(p.days_to_expiry, cfg.objective.planning_horizon_days)
    horizon = path_horizon_days(p, cfg)
    daily_prices = schedule_to_daily(schedule, horizon)

    rows: List[Dict[str, float]] = []
    remaining = p.inventory_units
    elapsed = 0.0
    for day, price in enumerate(daily_prices):
        if elapsed >= window:
            break
        day_len = min(1.0, window - elapsed)
        demand = objective.demand_at(price)
        sales = min(remaining, demand * day_len)
        unit_days = unit_days_held(remaining, demand, day_len)
        depth = max(0.0, p.current_price - price)
        rows.append({
            "day": day,
            "price": price,
            "daily_demand": demand,
            "expected_sales_units": sales,
            "start_inventory": remaining,
            "end_inventory": remaining - sales,
            "expected_revenue": price * sales,
            "gross_profit": (price - p.unit_cost) * sales,
            "holding_cost":
                cfg.inventory.holding_cost_per_unit_day * unit_days,
            "markdown_cost":
                cfg.objective.markdown_friction * depth * sales,
        })
        remaining -= sales
        elapsed += day_len

    # Same tail treatment as the evaluator: leftover stock keeps selling at
    # the final day's rate until expiry; the rest is projected waste.
    if rows:
        last_demand = rows[-1]["daily_demand"]
    else:
        last_demand = 0.0
    tail_days = max(0.0, p.days_to_expiry - elapsed)
    projected_waste = max(0.0, remaining - last_demand * tail_days)
    unit_loss = cfg.waste.unit_waste_loss(p.unit_cost)
    weight = waste_risk_weight(
        p.days_to_expiry, cfg.objective.planning_horizon_days,
        cfg.expiry.tau_days,
    )
    if rows:
        rows[-1]["projected_waste_units"] = projected_waste
        rows[-1]["projected_waste_cost"] = (
            projected_waste * unit_loss * weight
        )
    return rows


def validate_daily_prices(
    product: ProductContext,
    config: PricingConfig,
    daily_prices: Sequence[float],
) -> Tuple[List[float], List[str]]:
    """Validate a user-proposed daily price path against the same
    constraints the optimizer enumerates under. Returns the normalized
    path (extended to the horizon) and a list of human-readable errors —
    an empty list means the path is feasible."""
    errors: List[str] = []
    horizon = path_horizon_days(product, config)
    cur = product.current_price
    floor = path_price_floor(product, config)
    max_drop = config.constraints.max_daily_price_drop

    if not daily_prices:
        return [], ["the price path is empty"]
    prices = [float(p) for p in daily_prices][:horizon]

    prev = cur
    for day, price in enumerate(prices):
        if not math.isfinite(price) or price <= 0:
            errors.append(f"day {day}: price must be a positive number")
            continue
        if price > cur + 0.005:
            errors.append(
                f"day {day}: ${price:.2f} is above the current price "
                f"${cur:.2f} (price increases are not allowed)"
            )
        if price < floor - 0.005:
            errors.append(
                f"day {day}: ${price:.2f} is below the allowed floor "
                f"${floor:.2f}"
            )
        if price > prev + 0.005:
            errors.append(
                f"day {day}: the path may not move back up "
                f"(${prev:.2f} -> ${price:.2f})"
            )
        elif price < prev * (1.0 - max_drop) - 0.005:
            errors.append(
                f"day {day}: dropping ${prev:.2f} -> ${price:.2f} exceeds "
                f"the {max_drop:.0%} maximum daily price move"
            )
        prev = price

    while len(prices) < horizon:
        prices.append(prices[-1])
    return prices, errors
