"""Counterfactual backtest of recommendations against the simulator.

For every optimized product, two strategies are replayed day by day under
the *data-generating* demand model (the synthetic simulator's ground-truth
elasticities, freshness decay and noise — not the engine's own estimates):

    Strategy A (hold):        keep the current price until expiry
    Strategy B (recommended): switch to the recommended price

Both strategies see identical noise draws (common random numbers), so any
difference is attributable to the pricing decision.

IMPORTANT: results are a SYNTHETIC SIMULATION. They measure how well the
engine optimizes against the simulated economy it was trained in — they
are not evidence about real-world retail performance.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .config import PricingConfig
from .economics import expiry_pressure


def _simulate_sell_down(
    price: float,
    row: pd.Series,
    config: PricingConfig,
    noise: np.ndarray,
) -> Dict[str, float]:
    """Replay one product's remaining shelf life at a fixed price."""
    sim = config.simulator
    elasticity = config.elasticity.for_department(row["DEPARTMENT"])

    base = (
        sim.base_daily_demand
        * (1.0 + sim.promotion_uplift * row["PROMOTION"])
        * sim.season_multipliers.get(row.get("Season", ""), 1.0)
    )
    price_eff = (price / row["PRICE_RETAIL"]) ** elasticity

    days = int(row["DAYS_TO_EXPIRY"])
    remaining = float(row["INVENTORY_UNITS"])
    revenue = 0.0
    sold = 0.0

    for t in range(days):
        if remaining <= 0:
            break
        freshness = 1.0 - config.expiry.freshness_sensitivity * expiry_pressure(
            days - t, config.expiry.tau_days
        )
        demand = base * price_eff * freshness * noise[t]
        sales = min(remaining, max(0.0, demand))
        revenue += price * sales
        sold += sales
        remaining -= sales

    cost = float(row["COST"])
    waste = remaining
    salvage = waste * config.waste.salvage_rate * cost
    inventory = float(row["INVENTORY_UNITS"])
    return {
        "revenue": revenue,
        "units_sold": sold,
        "waste_units": waste,
        "sell_through": sold / inventory if inventory > 0 else 1.0,
        "cash_recovered": revenue + salvage,
        # Gross profit net of losses on expired stock: sold units earn
        # price - cost, expired units lose cost - salvage (plus disposal).
        "gross_profit": (
            revenue
            - cost * sold
            - waste * config.waste.unit_waste_loss(cost)
        ),
    }


def backtest_recommendations(
    items: pd.DataFrame,
    recommendations: pd.DataFrame,
    config: PricingConfig,
) -> Dict[str, object]:
    """Aggregate hold-vs-recommended outcomes across all optimized items.

    ``items`` rows must align positionally with ``recommendations`` rows
    (as produced by the pipeline).
    """
    totals = {
        strategy: {
            "revenue": 0.0, "gross_profit": 0.0, "units_sold": 0.0,
            "waste_units": 0.0, "cash_recovered": 0.0, "sell_through": 0.0,
        }
        for strategy in ("hold", "recommended")
    }

    max_days = int(items["DAYS_TO_EXPIRY"].max()) if len(items) else 0
    n = 0
    for pos, (_, row) in enumerate(items.iterrows()):
        rec_price = float(recommendations.iloc[pos]["Recommended_Price"])
        # Common random numbers: one deterministic noise stream per
        # product, shared by both strategies.
        rng = np.random.default_rng([config.seed, 7919, pos])
        noise = rng.lognormal(
            mean=0.0, sigma=config.simulator.noise_sigma, size=max(max_days, 1)
        )
        outcomes = {
            "hold": _simulate_sell_down(
                float(row["PRICE_CURRENT"]), row, config, noise
            ),
            "recommended": _simulate_sell_down(rec_price, row, config, noise),
        }
        for strategy, outcome in outcomes.items():
            for key in totals[strategy]:
                totals[strategy][key] += outcome[key]
        n += 1

    for strategy in totals:
        totals[strategy]["sell_through"] = (
            totals[strategy]["sell_through"] / n if n else 0.0
        )

    marked = recommendations[recommendations["Action"] == "Markdown"]
    return {
        "label": "SYNTHETIC SIMULATION (not real-world performance)",
        "n_products": n,
        "n_markdowns": int(len(marked)),
        "avg_markdown_pct": (
            float(marked["Markdown_Percentage"].mean()) if len(marked) else 0.0
        ),
        "hold": {k: round(v, 2) for k, v in totals["hold"].items()},
        "recommended": {
            k: round(v, 2) for k, v in totals["recommended"].items()
        },
    }
