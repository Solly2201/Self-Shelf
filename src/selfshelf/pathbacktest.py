"""Multi-period counterfactual backtest under the synthetic simulator.

Extends the frozen single-price backtest to *price paths*: three strategies
are replayed day by day under the data-generating demand model —

    hold:      keep the current price until expiry
    immediate: switch to the one-shot recommended price today
    path:      follow the multi-period optimized daily schedule

All three see identical noise draws (common random numbers, the same
deterministic per-product streams as ``backtest.backtest_recommendations``),
so differences are attributable purely to the pricing strategy. A
constant-price path replay is proven in tests to reproduce the frozen
``backtest._simulate_sell_down`` outputs exactly, so this module cannot
drift into a second accounting of the simulated economy.

IMPORTANT: results are a SYNTHETIC SIMULATION. They measure how well each
strategy performs against the simulated economy the engine was trained in —
they are not evidence about real-world retail performance. The backtest is
only defined for the synthetic data source: user-imported data has no
ground-truth simulator to replay against.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import PricingConfig
from .economics import expiry_pressure

STRATEGIES = ("hold", "immediate", "path")


def _simulate_path_sell_down(
    daily_prices: Sequence[float],
    row: pd.Series,
    config: PricingConfig,
    noise: np.ndarray,
) -> Dict[str, float]:
    """Replay one product's remaining shelf life at per-day prices.

    Identical accounting to the frozen ``backtest._simulate_sell_down``;
    the only generalization is that the price (and hence the simulator's
    price effect) may differ per day. Days beyond the provided path keep
    its final price.
    """
    sim = config.simulator
    elasticity = config.elasticity.for_department(row["DEPARTMENT"])

    base = (
        sim.base_daily_demand
        * (1.0 + sim.promotion_uplift * row["PROMOTION"])
        * sim.season_multipliers.get(row.get("Season", ""), 1.0)
    )

    days = int(row["DAYS_TO_EXPIRY"])
    remaining = float(row["INVENTORY_UNITS"])
    revenue = 0.0
    sold = 0.0
    unit_days = 0.0

    prices = list(daily_prices) if len(daily_prices) else [
        float(row["PRICE_CURRENT"])
    ]
    for t in range(days):
        if remaining <= 0:
            break
        price = float(prices[t] if t < len(prices) else prices[-1])
        price_eff = (price / row["PRICE_RETAIL"]) ** elasticity
        freshness = 1.0 - config.expiry.freshness_sensitivity * expiry_pressure(
            days - t, config.expiry.tau_days
        )
        demand = base * price_eff * freshness * noise[t]
        sales = min(remaining, max(0.0, demand))
        revenue += price * sales
        sold += sales
        # Midpoint approximation of stock carried through the day.
        unit_days += remaining - sales / 2.0
        remaining -= sales

    cost = float(row["COST"])
    waste = remaining
    salvage = waste * config.waste.salvage_rate * cost
    inventory = float(row["INVENTORY_UNITS"])
    holding_cost = config.inventory.holding_cost_per_unit_day * unit_days
    gross_profit = (
        revenue - cost * sold - waste * config.waste.unit_waste_loss(cost)
    )
    return {
        "revenue": revenue,
        "units_sold": sold,
        "waste_units": waste,
        "terminal_inventory": remaining,
        "sell_through": sold / inventory if inventory > 0 else 1.0,
        "cash_recovered": revenue + salvage,
        "holding_cost": holding_cost,
        "gross_profit": gross_profit,
        "economic_value": gross_profit - holding_cost,
    }


def backtest_price_paths(
    items: pd.DataFrame,
    immediate_prices: Sequence[float],
    daily_paths: Sequence[Sequence[float]],
    config: PricingConfig,
) -> Dict[str, object]:
    """Aggregate hold / immediate-markdown / optimized-path outcomes.

    ``items`` rows align positionally with ``immediate_prices`` (the
    one-shot recommendations) and ``daily_paths`` (per-day price lists from
    the multi-period optimizer). Noise streams match the frozen backtest's
    ([seed, 7919, pos]) so the hold strategy here is the same replay as in
    ``backtest_recommendations``.
    """
    totals = {
        strategy: {
            "revenue": 0.0, "gross_profit": 0.0, "units_sold": 0.0,
            "waste_units": 0.0, "terminal_inventory": 0.0,
            "holding_cost": 0.0, "economic_value": 0.0,
            "cash_recovered": 0.0, "sell_through": 0.0,
        }
        for strategy in STRATEGIES
    }

    max_days = int(items["DAYS_TO_EXPIRY"].max()) if len(items) else 0
    n = 0
    n_staged = 0
    for pos, (_, row) in enumerate(items.iterrows()):
        rng = np.random.default_rng([config.seed, 7919, pos])
        noise = rng.lognormal(
            mean=0.0, sigma=config.simulator.noise_sigma,
            size=max(max_days, 1),
        )
        current = float(row["PRICE_CURRENT"])
        path: List[float] = [float(p) for p in daily_paths[pos]]
        if len({round(p, 4) for p in path if p < current - 1e-9}) > 0 and (
            path[0] >= current - 1e-9
        ):
            n_staged += 1
        outcomes = {
            "hold": _simulate_path_sell_down([current], row, config, noise),
            "immediate": _simulate_path_sell_down(
                [float(immediate_prices[pos])], row, config, noise
            ),
            "path": _simulate_path_sell_down(path, row, config, noise),
        }
        for strategy, outcome in outcomes.items():
            for key in totals[strategy]:
                totals[strategy][key] += outcome[key]
        n += 1

    for strategy in totals:
        totals[strategy]["sell_through"] = (
            totals[strategy]["sell_through"] / n if n else 0.0
        )

    return {
        "label": "SYNTHETIC SIMULATION (not real-world performance)",
        "n_products": n,
        "n_staged_paths": n_staged,
        **{
            strategy: {k: round(v, 2) for k, v in totals[strategy].items()}
            for strategy in STRATEGIES
        },
    }
