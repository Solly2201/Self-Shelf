"""Demand simulation, the ML demand model, and elasticity estimation.

The dataset has no historical sales volume, so demand is produced by an
explicit synthetic economic simulator (documented below) instead of ad-hoc
business rules. The ML model is then trained on that simulated demand, and
department elasticities are re-estimated from the simulated sales data —
mirroring what the pipeline would do with real transaction history.

This means every economic claim downstream is conditional on the simulator's
assumptions. That circularity is inherent to synthetic data and is stated
openly rather than hidden.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import PricingConfig
from .economics import expiry_pressure

MODEL_FEATURES: List[str] = [
    "PRICE_CURRENT",
    "PRICE_RETAIL",
    "PRICE_RATIO",
    "PROMOTION",
    "DAYS_TO_EXPIRY",
    "URGENCY_RATIO",
    "MAX_SHELF_LIFE",
]


# ---------------------------------------------------------------------------
# Synthetic demand simulator
# ---------------------------------------------------------------------------

def apply_historical_discounts(
    df: pd.DataFrame, config: PricingConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Give a share of rows a simulated past markdown.

    The raw dataset prices are almost all at list price; without price
    variation there is nothing for the elasticity estimator or the ML model
    to learn a price response from.
    """
    df = df.copy()
    sim = config.simulator
    lo, hi = sim.historical_discount_range
    discounted = rng.random(len(df)) < sim.historical_discount_share
    discounts = rng.uniform(lo, hi, size=len(df))
    df["PRICE_CURRENT"] = np.where(
        discounted, df["PRICE_RETAIL"] * (1 - discounts), df["PRICE_CURRENT"]
    )
    df["PRICE_RATIO"] = df["PRICE_CURRENT"] / df["PRICE_RETAIL"]
    return df


def simulate_demand(
    df: pd.DataFrame, config: PricingConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate daily demand from an explicit multiplicative model:

        demand = base
                 x (price / list price) ** elasticity      (price effect)
                 x (1 + uplift * promotion)                (promotion)
                 x (1 - sensitivity * expiry_pressure)     (freshness)
                 x season multiplier                       (seasonality)
                 x lognormal noise
    """
    df = df.copy()
    sim = config.simulator

    elasticities = (
        df["DEPARTMENT"]
        .map(config.elasticity.by_department)
        .fillna(config.elasticity.default)
        .to_numpy()
    )
    price_eff = df["PRICE_RATIO"].to_numpy() ** elasticities

    promo_eff = 1.0 + sim.promotion_uplift * df["PROMOTION"].to_numpy()

    pressure = np.array([
        expiry_pressure(d, config.expiry.tau_days)
        for d in df["DAYS_TO_EXPIRY"].to_numpy()
    ])
    freshness_eff = 1.0 - config.expiry.freshness_sensitivity * pressure

    if "Season" in df.columns:
        season_eff = (
            df["Season"].map(sim.season_multipliers).fillna(1.0).to_numpy()
        )
    else:
        season_eff = 1.0

    noise = rng.lognormal(mean=0.0, sigma=sim.noise_sigma, size=len(df))

    df["DEMAND"] = (
        sim.base_daily_demand
        * price_eff
        * promo_eff
        * freshness_eff
        * season_eff
        * noise
    ).clip(min=0.0)
    return df


# ---------------------------------------------------------------------------
# ML demand model
# ---------------------------------------------------------------------------

class DemandModel:
    """Random Forest wrapper predicting daily demand at observed prices.

    The forest's job is *forecasting the demand level*, not encoding the
    price response — the optimizer moves demand along the price axis with
    the estimated elasticity (see ``economics.EconomicObjective``).
    """

    def __init__(self, seed: int, n_estimators: int = 100, max_depth: int = 12):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame) -> "DemandModel":
        self.model.fit(df[MODEL_FEATURES], df["DEMAND"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[MODEL_FEATURES]), 0.0, None)


# ---------------------------------------------------------------------------
# Elasticity estimation
# ---------------------------------------------------------------------------

@dataclass
class ElasticityEstimate:
    elasticity: float
    n_observations: int
    source: str  # "estimated" or "default"


def estimate_elasticities(
    df: pd.DataFrame,
    config: PricingConfig,
    min_observations: int = 50,
) -> Dict[str, ElasticityEstimate]:
    """Estimate own-price elasticity per department by log-log regression.

    log(demand) ~ const + e * log(price / list price) + b * promotion

    Only training data should be passed in. Departments with too few
    observations, or with a non-negative estimate (economically invalid
    for own-price elasticity under this demand model), fall back to the
    configured default.
    """
    results: Dict[str, ElasticityEstimate] = {}
    fallback = config.elasticity.default

    for dept, group in df.groupby("DEPARTMENT"):
        valid = group[(group["DEMAND"] > 0) & (group["PRICE_RATIO"] > 0)]
        if len(valid) < min_observations:
            results[dept] = ElasticityEstimate(fallback, len(valid), "default")
            continue

        y = np.log(valid["DEMAND"].to_numpy())
        x = np.column_stack([
            np.ones(len(valid)),
            np.log(valid["PRICE_RATIO"].to_numpy()),
            valid["PROMOTION"].to_numpy(dtype=float),
        ])
        coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
        estimate = float(coefs[1])

        if estimate >= -0.05:
            # Not a credible own-price elasticity; use the default rather
            # than letting the optimizer believe demand ignores (or loves)
            # higher prices.
            results[dept] = ElasticityEstimate(fallback, len(valid), "default")
        else:
            results[dept] = ElasticityEstimate(estimate, len(valid), "estimated")

    return results
