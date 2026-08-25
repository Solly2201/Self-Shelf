"""Data loading, cleaning and feature engineering.

The dataset provides prices, departments, promotions and dates, but no cost,
expiry, or inventory data. Those are simulated here with clearly labelled,
seeded, configurable assumptions (see ``config.py``) — the pipeline never
pretends the dataset contains them.
"""

import numpy as np
import pandas as pd

from .config import PricingConfig

REQUIRED_COLUMNS = ["PRICE_RETAIL", "PRICE_CURRENT", "DEPARTMENT", "PRODUCT_NAME"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[(df["PRICE_RETAIL"] > 0) & (df["PRICE_CURRENT"] > 0)]
    df = df.copy()

    df["DEPARTMENT"] = df["DEPARTMENT"].str.strip()
    df["PRODUCT_NAME"] = df["PRODUCT_NAME"].str.strip()

    if "PROMOTION" in df.columns:
        df["PROMOTION"] = df["PROMOTION"].fillna(0)
        df["PROMOTION"] = np.where(df["PROMOTION"] == 0, 0, 1)
    else:
        df["PROMOTION"] = 0

    # A current price above list price is a data error for this pipeline.
    df["PRICE_CURRENT"] = np.minimum(df["PRICE_CURRENT"], df["PRICE_RETAIL"])
    return df.reset_index(drop=True)


def engineer_features(
    df: pd.DataFrame, config: PricingConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Add cost, shelf-life, expiry and inventory features.

    COST, DAYS_TO_EXPIRY and INVENTORY_UNITS are *simulated* from the
    configured assumptions because the dataset does not contain them.
    """
    df = df.copy()

    df["COST"] = df["PRICE_RETAIL"] * config.cost_ratio_of_retail

    df["MAX_SHELF_LIFE"] = (
        df["DEPARTMENT"]
        .map(config.expiry.shelf_life_by_department)
        .fillna(config.expiry.default_shelf_life_days)
        .astype(int)
    )

    df["DAYS_TO_EXPIRY"] = np.ceil(
        rng.uniform(1, df["MAX_SHELF_LIFE"].to_numpy())
    ).astype(int)
    df["URGENCY_RATIO"] = df["DAYS_TO_EXPIRY"] / df["MAX_SHELF_LIFE"]

    # Synthetic on-hand stock: lognormal days-of-supply around the target,
    # scaled by the simulator's base demand rate.
    inv_cfg = config.inventory
    days_supply = rng.lognormal(
        mean=np.log(inv_cfg.target_days_of_supply),
        sigma=inv_cfg.days_of_supply_sigma,
        size=len(df),
    )
    df["INVENTORY_UNITS"] = np.maximum(
        inv_cfg.min_units,
        np.round(days_supply * config.simulator.base_daily_demand),
    ).astype(int)

    df["PRICE_RATIO"] = df["PRICE_CURRENT"] / df["PRICE_RETAIL"]
    return df
