"""Elasticity uncertainty: standard errors, confidence intervals, and
data-sufficiency behavior.

The uncertainty layer must (a) never disagree with the frozen estimator
about the point estimate, (b) derive every reported number from actual
regression statistics, and (c) refuse to invent confidence for fallback
values.
"""

import math

import numpy as np
import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.demand import estimate_elasticities
from selfshelf.uncertainty import (
    ElasticityConfidence,
    estimate_elasticities_with_confidence,
    fallback_confidence,
)

CONFIG = PricingConfig()


def synthetic_frame(
    true_elasticity: float,
    n: int,
    seed: int = 0,
    noise: float = 0.15,
    ratio_low: float = 0.6,
    ratio_high: float = 1.0,
    n_levels: int = 8,
    department: str = "Bakery",
) -> pd.DataFrame:
    """Log-linear demand data with a known elasticity."""
    rng = np.random.default_rng(seed)
    levels = np.linspace(ratio_low, ratio_high, n_levels)
    ratios = rng.choice(levels, size=n)
    demand = 20.0 * ratios ** true_elasticity * rng.lognormal(
        0.0, noise, size=n
    )
    return pd.DataFrame({
        "DEPARTMENT": [department] * n,
        "DEMAND": demand,
        "PRICE_RATIO": ratios,
        "PROMOTION": rng.integers(0, 2, size=n),
    })


class TestPointEstimateParity:
    """The uncertainty layer runs the SAME regression as the frozen
    estimator — the point estimates must be bit-identical."""

    def test_estimates_match_frozen_estimator_exactly(self):
        df = pd.concat([
            synthetic_frame(-1.7, 300, seed=1, department="Bakery"),
            synthetic_frame(-0.9, 200, seed=2, department="Beverages"),
        ])
        frozen = estimate_elasticities(df, CONFIG, min_observations=50)
        detailed = estimate_elasticities_with_confidence(
            df, CONFIG, min_observations=50
        )
        for dept in frozen:
            assert detailed[dept].elasticity == frozen[dept].elasticity
            assert detailed[dept].n_observations == (
                frozen[dept].n_observations
            )

    def test_fallback_decision_matches_frozen_estimator(self):
        # Too few observations -> both must fall back.
        df = synthetic_frame(-1.5, 20, seed=3)
        frozen = estimate_elasticities(df, CONFIG, min_observations=50)
        detailed = estimate_elasticities_with_confidence(
            df, CONFIG, min_observations=50
        )
        assert frozen["Bakery"].source == "default"
        assert detailed["Bakery"].source == "fallback"
        assert detailed["Bakery"].elasticity == CONFIG.elasticity.default


class TestConfidenceInterval:
    def test_interval_contains_known_elasticity_in_typical_simulation(self):
        # Not guaranteed for every finite sample — checked across seeds:
        # a 95% CI should cover the truth in the vast majority of runs.
        true_e = -1.6
        covered = 0
        runs = 40
        for seed in range(runs):
            df = synthetic_frame(true_e, 400, seed=seed)
            est = estimate_elasticities_with_confidence(df, CONFIG)["Bakery"]
            assert est.source == "estimated"
            if est.lower_ci <= true_e <= est.upper_ci:
                covered += 1
        assert covered >= int(0.85 * runs)

    def test_interval_brackets_the_estimate(self):
        est = estimate_elasticities_with_confidence(
            synthetic_frame(-1.4, 250, seed=7), CONFIG
        )["Bakery"]
        assert est.lower_ci < est.elasticity < est.upper_ci
        assert est.standard_error > 0

    def test_more_observations_generally_improve_precision(self):
        small = estimate_elasticities_with_confidence(
            synthetic_frame(-1.5, 60, seed=11), CONFIG
        )["Bakery"]
        large = estimate_elasticities_with_confidence(
            synthetic_frame(-1.5, 2000, seed=11), CONFIG
        )["Bakery"]
        assert large.standard_error < small.standard_error

    def test_more_price_variation_improves_identification(self):
        narrow = estimate_elasticities_with_confidence(
            synthetic_frame(-1.5, 400, seed=13, ratio_low=0.97), CONFIG
        )["Bakery"]
        wide = estimate_elasticities_with_confidence(
            synthetic_frame(-1.5, 400, seed=13, ratio_low=0.55), CONFIG
        )["Bakery"]
        assert wide.standard_error < narrow.standard_error

    def test_deterministic_output(self):
        df = synthetic_frame(-1.3, 300, seed=17)
        a = estimate_elasticities_with_confidence(df, CONFIG)["Bakery"]
        b = estimate_elasticities_with_confidence(df.copy(), CONFIG)["Bakery"]
        assert a == b


class TestFallbackHonesty:
    def test_fallback_has_no_confidence_interval(self):
        df = synthetic_frame(-1.5, 10, seed=19)
        est = estimate_elasticities_with_confidence(df, CONFIG)["Bakery"]
        assert est.source == "fallback"
        assert est.confidence == "fallback"
        assert est.standard_error is None
        assert est.lower_ci is None and est.upper_ci is None
        assert est.estimation_method == "configured fallback"
        assert "observations" in est.reason

    def test_non_negative_estimate_falls_back_without_fake_ci(self):
        # Demand that INCREASES with price -> not a credible own-price
        # elasticity; the frozen estimator falls back, so must we.
        rng = np.random.default_rng(23)
        ratios = rng.choice(np.linspace(0.6, 1.0, 6), size=200)
        df = pd.DataFrame({
            "DEPARTMENT": ["Bakery"] * 200,
            "DEMAND": 20.0 * ratios ** 1.5,  # positive "elasticity"
            "PRICE_RATIO": ratios,
            "PROMOTION": [0] * 200,
        })
        est = estimate_elasticities_with_confidence(df, CONFIG)["Bakery"]
        assert est.source == "fallback"
        assert est.lower_ci is None

    def test_fallback_constructor_shape(self):
        est = fallback_confidence(-1.2, "insufficient data")
        assert isinstance(est, ElasticityConfidence)
        d = est.as_dict()
        assert d["confidence"] == "fallback"
        assert d["standard_error"] is None


class TestDegenerateData:
    def test_single_price_level_yields_no_standard_error(self):
        # Constant price ratio: the price column is collinear with the
        # intercept, so the price coefficient is unidentified. The frozen
        # estimator's minimum-norm lstsq still returns SOME number (and may
        # pass the sign check) — the uncertainty layer must mirror that
        # point estimate but refuse to attach a standard error or interval
        # to an unidentified coefficient.
        rng = np.random.default_rng(29)
        n = 200
        df = pd.DataFrame({
            "DEPARTMENT": ["Bakery"] * n,
            "DEMAND": 20.0 * rng.lognormal(0.0, 0.1, size=n),
            "PRICE_RATIO": [0.9] * n,
            "PROMOTION": rng.integers(0, 2, size=n),
        })
        frozen = estimate_elasticities(df, CONFIG)["Bakery"]
        est = estimate_elasticities_with_confidence(df, CONFIG)["Bakery"]
        if est.source == "estimated":
            assert est.elasticity == frozen.elasticity
            assert est.standard_error is None
            assert est.lower_ci is None and est.upper_ci is None
            assert est.confidence == "low"
        else:
            assert frozen.source == "default"
            assert est.standard_error is None

    def test_zero_demand_rows_are_excluded_like_frozen_estimator(self):
        df = synthetic_frame(-1.5, 300, seed=31)
        df.loc[df.index[:100], "DEMAND"] = 0.0
        frozen = estimate_elasticities(df, CONFIG)["Bakery"]
        detailed = estimate_elasticities_with_confidence(df, CONFIG)["Bakery"]
        assert detailed.n_observations == frozen.n_observations == 200

    def test_empty_frame_returns_no_results(self):
        df = pd.DataFrame(
            columns=["DEPARTMENT", "DEMAND", "PRICE_RATIO", "PROMOTION"]
        )
        assert estimate_elasticities_with_confidence(df, CONFIG) == {}


class TestConfidenceLabels:
    def test_rich_data_earns_high_confidence(self):
        est = estimate_elasticities_with_confidence(
            synthetic_frame(-1.6, 1500, seed=37, noise=0.10), CONFIG
        )["Bakery"]
        assert est.confidence == "high"
        assert est.n_distinct_prices >= 3

    def test_noisy_sparse_data_is_not_high_confidence(self):
        est = estimate_elasticities_with_confidence(
            synthetic_frame(
                -1.6, 55, seed=41, noise=0.9, ratio_low=0.93, n_levels=2
            ),
            CONFIG,
        )["Bakery"]
        assert est.confidence in ("medium", "low", "fallback")

    def test_as_dict_is_json_safe(self):
        import json

        df = synthetic_frame(-1.5, 300, seed=43)
        est = estimate_elasticities_with_confidence(df, CONFIG)["Bakery"]
        payload = est.as_dict()
        json.dumps(payload)  # must not raise
        assert payload["confidence_level"] == 0.95
        assert payload["lower_ci"] < payload["elasticity"] < (
            payload["upper_ci"]
        )
