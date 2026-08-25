"""Demand-model evaluation with a leak-free train/validation/test split."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_data(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
    validation_size: float = 0.2,
) -> SplitData:
    """Split rows 60/20/20 into train / validation / test.

    The model is fit on train only; validation guides modelling choices;
    test is touched once for the final report and to supply products for
    optimization. Elasticity estimation also uses train only.
    """
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    # validation_size is relative to the full dataset.
    val_fraction = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val, test_size=val_fraction, random_state=seed
    )
    return SplitData(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model, split: SplitData) -> Dict[str, Dict[str, float]]:
    """Metrics on validation and test sets for a fitted DemandModel."""
    report = {}
    for name, frame in (("validation", split.validation), ("test", split.test)):
        preds = model.predict(frame)
        report[name] = regression_metrics(frame["DEMAND"].to_numpy(), preds)
    return report
