"""Counterfactual backtests for the adaptive (closed-loop) layer.

Two comparisons, both replayed under the synthetic data-generating demand
model with common random numbers, so differences are attributable purely
to the strategy:

    closed-loop backtest
        hold            keep the current price until expiry
        open_loop       execute the day-0 optimized path, never look back
        closed_loop     re-optimize every day from the observed state

    replenishment backtest
        naive           plan ignoring known future deliveries
        aware           plan seeing the delivery schedule
        (both replayed against a reality where the deliveries DO arrive)

Every strategy runs through the same ``closedloop.run_closed_loop``
machinery (with planning disabled for the fixed-path strategies), which
uses the frozen backtest accounting conventions — the hold strategy here
reproduces ``pathbacktest``'s hold replay exactly (locked by tests).

IMPORTANT: results are a SYNTHETIC SIMULATION. They measure adaptive
behavior against the simulated economy, not real-world performance.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .closedloop import run_closed_loop, simulator_environment
from .config import PricingConfig
from .economics import ProductContext
from .pathopt import optimize_path, path_horizon_days, schedule_to_daily
from .replenishment import (
    ReplenishmentSchedule,
    optimize_path_with_replenishment,
)

CLOSED_LOOP_STRATEGIES = ("hold", "open_loop", "closed_loop")

_OUTCOME_KEYS = (
    "revenue", "gross_profit", "units_sold", "waste_units",
    "terminal_inventory", "holding_cost", "economic_value",
    "cash_recovered", "sell_through",
)


def _environment_for_row(row: pd.Series, config: PricingConfig, noise):
    """The data-generating process for one synthetic item — identical
    formula and noise-stream convention to the frozen backtests."""
    sim = config.simulator
    base = (
        sim.base_daily_demand
        * (1.0 + sim.promotion_uplift * row["PROMOTION"])
        * sim.season_multipliers.get(row.get("Season", ""), 1.0)
    )
    return simulator_environment(
        base_demand=base,
        retail_price=float(row["PRICE_RETAIL"]),
        true_elasticity=config.elasticity.for_department(row["DEPARTMENT"]),
        config=config,
        noise=noise,
    )


def backtest_closed_loop(
    items: pd.DataFrame,
    contexts: Sequence[ProductContext],
    config: PricingConfig,
    max_products: Optional[int] = None,
) -> Dict[str, object]:
    """Hold vs open-loop path vs daily closed-loop re-optimization.

    ``items`` rows align positionally with ``contexts`` (the engine's
    belief about each product). Noise streams are ([seed, 7919, pos]) —
    the same per-product streams as the frozen backtests, shared by all
    three strategies (common random numbers).
    """
    totals = {
        s: {k: 0.0 for k in _OUTCOME_KEYS}
        for s in CLOSED_LOOP_STRATEGIES
    }
    n = 0
    replans_total = 0
    max_days = int(items["DAYS_TO_EXPIRY"].max()) if len(items) else 0

    for pos, (_, row) in enumerate(items.iterrows()):
        if max_products is not None and pos >= max_products:
            break
        context = contexts[pos]
        rng = np.random.default_rng([config.seed, 7919, pos])
        noise = rng.lognormal(
            mean=0.0, sigma=config.simulator.noise_sigma,
            size=max(max_days, 1),
        )
        env = _environment_for_row(row, config, noise)
        days = max(1, int(context.days_to_expiry))

        hold = run_closed_loop(
            context, config, env,
            fixed_daily_prices=[context.current_price] * days,
        )
        open_loop = run_closed_loop(context, config, env, reoptimize=False)
        closed = run_closed_loop(context, config, env, reoptimize=True)

        for strategy, result in (
            ("hold", hold), ("open_loop", open_loop), ("closed_loop", closed),
        ):
            for key in _OUTCOME_KEYS:
                totals[strategy][key] += result.outcome[key]
        replans_total += closed.replans
        n += 1

    for strategy in totals:
        totals[strategy]["sell_through"] = (
            totals[strategy]["sell_through"] / n if n else 0.0
        )

    return {
        "label": "SYNTHETIC SIMULATION (not real-world performance)",
        "n_products": n,
        "closed_loop_replans": replans_total,
        **{
            s: {k: round(v, 2) for k, v in totals[s].items()}
            for s in CLOSED_LOOP_STRATEGIES
        },
    }


def backtest_replenishment_awareness(
    context: ProductContext,
    config: PricingConfig,
    environment,
    schedule: ReplenishmentSchedule,
) -> Dict[str, object]:
    """Ignore-deliveries planning vs delivery-aware planning for one
    product, both executed open-loop against the SAME reality in which the
    deliveries actually arrive (common random numbers by construction)."""
    horizon = path_horizon_days(context, config)
    days = max(1, int(context.days_to_expiry))

    naive_plan = optimize_path(context, config)
    aware_plan = optimize_path_with_replenishment(context, config, schedule)

    def full_path(daily: List[float]) -> List[float]:
        daily = list(daily) or [context.current_price]
        while len(daily) < days:
            daily.append(daily[-1])
        return daily

    naive = run_closed_loop(
        context, config, environment, replenishment=schedule,
        fixed_daily_prices=full_path(naive_plan.daily_prices),
    )
    aware = run_closed_loop(
        context, config, environment, replenishment=schedule,
        fixed_daily_prices=full_path(aware_plan.daily_prices),
    )
    return {
        "label": "SYNTHETIC SIMULATION (not real-world performance)",
        "horizon_days": horizon,
        "schedule": schedule.as_dict(),
        "naive": {
            "daily_prices": full_path(naive_plan.daily_prices),
            **{k: round(naive.outcome[k], 2) for k in _OUTCOME_KEYS},
        },
        "aware": {
            "daily_prices": full_path(aware_plan.daily_prices),
            **{k: round(aware.outcome[k], 2) for k in _OUTCOME_KEYS},
        },
    }
