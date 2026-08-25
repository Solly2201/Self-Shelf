"""Elasticity uncertainty quantification.

The frozen ``demand.estimate_elasticities`` returns a point estimate per
department from a log-log OLS regression:

    log(demand) ~ const + e * log(price / list price) + b * promotion

This module answers the follow-up question: *how much should that point
estimate be trusted?* It runs the **identical** regression (same row
filtering, same design matrix, same ``numpy.linalg.lstsq`` call) and adds
standard OLS inference on top:

    standard error   SE(e) = sqrt( sigma^2 * [ (X'X)^-1 ]_ee ),
                     sigma^2 = RSS / (n - k)
    confidence       e +/- t_{alpha/2, n-k} * SE(e)
    interval

Point estimates produced here are proven (in tests) to equal the frozen
estimator's exactly — this module can never disagree with the engine about
*what* the elasticity is, only report how well-identified it is.

Fallback elasticities get NO confidence interval: a configured constant has
no sampling distribution, and inventing one would be fake precision. They
are labelled ``source="fallback"`` with a reason instead.

Confidence labels are deliberately coarse and rule-based (documented so
they can be audited, not tuned to look good):

    fallback  the elasticity is a configured default, not learned
    low       SE unavailable (degenerate regression), OR relative CI
              half-width > 50%, OR fewer than 2 distinct observed price
              levels, OR an implausibly large magnitude (|e| > 10)
    medium    relative CI half-width <= 50% with >= 2 distinct price
              levels and >= ``min_observations`` observations
    high      relative CI half-width <= 25% with >= 3 distinct price
              levels and >= 100 observations

"Relative CI half-width" is (t * SE) / |e| — how wide the interval is
compared to the size of the estimate itself.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import PricingConfig

# Distinct price levels are counted after rounding the price ratio to 3
# decimals — sub-0.1% price differences are not separate "levels".
_PRICE_LEVEL_DECIMALS = 3

# |elasticity| beyond this is treated as implausible for retail demand and
# capped at "low" confidence regardless of the regression statistics.
_IMPLAUSIBLE_MAGNITUDE = 10.0

HIGH_MIN_OBSERVATIONS = 100
HIGH_MIN_PRICE_LEVELS = 3
HIGH_MAX_RELATIVE_HALF_WIDTH = 0.25
MEDIUM_MAX_RELATIVE_HALF_WIDTH = 0.50


@dataclass
class ElasticityConfidence:
    """Statistical context for one department/category elasticity."""

    elasticity: float
    source: str                       # "estimated" | "fallback"
    confidence: str                   # "high" | "medium" | "low" | "fallback"
    reason: str
    estimation_method: str            # "log-log OLS" | "configured fallback"
    n_observations: int
    n_distinct_prices: int
    price_ratio_min: Optional[float] = None
    price_ratio_max: Optional[float] = None
    price_ratio_std: Optional[float] = None
    standard_error: Optional[float] = None
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    confidence_level: float = 0.95

    def as_dict(self) -> Dict[str, object]:
        """JSON-safe dict (numpy scalars converted, non-finite to None)."""
        def f(v, digits):
            if v is None:
                return None
            v = float(v)
            if not math.isfinite(v):
                return None
            return round(v, digits)

        return {
            "elasticity": f(self.elasticity, 4),
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
            "estimation_method": self.estimation_method,
            "n_observations": int(self.n_observations),
            "n_distinct_prices": int(self.n_distinct_prices),
            "price_ratio_min": f(self.price_ratio_min, 4),
            "price_ratio_max": f(self.price_ratio_max, 4),
            "price_ratio_std": f(self.price_ratio_std, 4),
            "standard_error": f(self.standard_error, 4),
            "lower_ci": f(self.lower_ci, 4),
            "upper_ci": f(self.upper_ci, 4),
            "confidence_level": self.confidence_level,
        }


def fallback_confidence(
    fallback_value: float,
    reason: str,
    n_observations: int = 0,
    n_distinct_prices: int = 0,
    confidence_level: float = 0.95,
) -> ElasticityConfidence:
    return ElasticityConfidence(
        elasticity=float(fallback_value),
        source="fallback",
        confidence="fallback",
        reason=reason,
        estimation_method="configured fallback",
        n_observations=n_observations,
        n_distinct_prices=n_distinct_prices,
        confidence_level=confidence_level,
    )


def _confidence_label(
    estimate: float,
    standard_error: Optional[float],
    half_width: Optional[float],
    n_observations: int,
    n_distinct_prices: int,
    min_observations: int,
) -> str:
    if standard_error is None or half_width is None:
        return "low"
    if abs(estimate) > _IMPLAUSIBLE_MAGNITUDE:
        return "low"
    if n_distinct_prices < 2:
        return "low"
    relative = half_width / abs(estimate) if abs(estimate) > 0 else math.inf
    if (
        n_observations >= HIGH_MIN_OBSERVATIONS
        and n_distinct_prices >= HIGH_MIN_PRICE_LEVELS
        and relative <= HIGH_MAX_RELATIVE_HALF_WIDTH
    ):
        return "high"
    if (
        n_observations >= min_observations
        and relative <= MEDIUM_MAX_RELATIVE_HALF_WIDTH
    ):
        return "medium"
    return "low"


def estimate_elasticities_with_confidence(
    df: pd.DataFrame,
    config: PricingConfig,
    min_observations: int = 50,
    confidence_level: float = 0.95,
) -> Dict[str, ElasticityConfidence]:
    """Per-department elasticity with OLS-based uncertainty.

    Runs the same regression as the frozen ``demand.estimate_elasticities``
    on the same filtered rows, so the point estimate and the
    estimated/fallback decision are identical — the only addition is the
    inference (SE, CI, sufficiency diagnostics).
    """
    results: Dict[str, ElasticityConfidence] = {}
    fallback = config.elasticity.default

    for dept, group in df.groupby("DEPARTMENT"):
        # Identical filtering to the frozen estimator.
        valid = group[(group["DEMAND"] > 0) & (group["PRICE_RATIO"] > 0)]
        n = int(len(valid))
        ratios = valid["PRICE_RATIO"].to_numpy(dtype=float)
        n_levels = (
            int(len(np.unique(np.round(ratios, _PRICE_LEVEL_DECIMALS))))
            if n else 0
        )

        if n < min_observations:
            results[dept] = fallback_confidence(
                fallback,
                f"only {n} usable observations "
                f"(needs {min_observations})",
                n_observations=n,
                n_distinct_prices=n_levels,
                confidence_level=confidence_level,
            )
            continue

        # Identical design matrix and solver to the frozen estimator.
        y = np.log(valid["DEMAND"].to_numpy())
        x = np.column_stack([
            np.ones(n),
            np.log(ratios),
            valid["PROMOTION"].to_numpy(dtype=float),
        ])
        coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
        estimate = float(coefs[1])

        if estimate >= -0.05:
            results[dept] = fallback_confidence(
                fallback,
                "regression did not yield a credible own-price elasticity "
                f"(estimate {estimate:.3f} is not sufficiently negative)",
                n_observations=n,
                n_distinct_prices=n_levels,
                confidence_level=confidence_level,
            )
            continue

        # OLS inference on the price coefficient. A constant promotion
        # column is collinear with the intercept (or all-zero); the price
        # coefficient is still identified, so inference uses the reduced
        # full-rank design — same column space, same residuals, same price
        # coefficient — rather than a misleading pseudo-inverse variance.
        promo = valid["PROMOTION"].to_numpy(dtype=float)
        cols = [np.ones(n), np.log(ratios)]
        if np.ptp(promo) > 0:
            cols.append(promo)
        xr = np.column_stack(cols)
        k = xr.shape[1]
        dof = n - k
        standard_error = None
        lower = upper = half_width = None
        if dof > 0 and n_levels >= 2:
            xtx = xr.T @ xr
            if np.linalg.matrix_rank(xtx) == k:
                beta, *_ = np.linalg.lstsq(xr, y, rcond=None)
                residuals = y - xr @ beta
                sigma2 = float(residuals @ residuals) / dof
                var = sigma2 * np.linalg.inv(xtx)[1, 1]
                if math.isfinite(var) and var >= 0:
                    standard_error = math.sqrt(var)
                    t_crit = float(
                        stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, dof)
                    )
                    half_width = t_crit * standard_error
                    lower = estimate - half_width
                    upper = estimate + half_width

        label = _confidence_label(
            estimate, standard_error, half_width, n, n_levels,
            min_observations,
        )
        if standard_error is None:
            reason = (
                "regression is degenerate (insufficient independent price "
                "variation to identify a standard error)"
            )
        elif label == "low":
            reason = (
                "estimate is weakly identified (wide interval, few price "
                "levels, or implausible magnitude)"
            )
        else:
            reason = (
                f"estimated from {n} observations across "
                f"{n_levels} distinct price levels"
            )

        results[dept] = ElasticityConfidence(
            elasticity=estimate,
            source="estimated",
            confidence=label,
            reason=reason,
            estimation_method="log-log OLS",
            n_observations=n,
            n_distinct_prices=n_levels,
            price_ratio_min=float(ratios.min()),
            price_ratio_max=float(ratios.max()),
            price_ratio_std=float(ratios.std()),
            standard_error=standard_error,
            lower_ci=lower,
            upper_ci=upper,
            confidence_level=confidence_level,
        )

    return results
