import numpy as np
import pandas as pd
import pytest

from selfshelf.config import PricingConfig
from selfshelf.demand import (
    DemandModel,
    apply_historical_discounts,
    estimate_elasticities,
    simulate_demand,
)


def make_frame(price_ratios, department="Bakery", promotion=0, days=30):
    n = len(price_ratios)
    ratios = np.asarray(price_ratios, dtype=float)
    return pd.DataFrame({
        "DEPARTMENT": [department] * n,
        "PRICE_RETAIL": [5.0] * n,
        "PRICE_CURRENT": 5.0 * ratios,
        "PRICE_RATIO": ratios,
        "PROMOTION": [promotion] * n,
        "DAYS_TO_EXPIRY": [days] * n,
        "URGENCY_RATIO": [days / 30] * n,
        "MAX_SHELF_LIFE": [30] * n,
        "Season": ["Spring"] * n,
    })


def noiseless_config() -> PricingConfig:
    cfg = PricingConfig()
    cfg.simulator.noise_sigma = 0.0
    return cfg


class TestSimulator:
    def test_demand_decreases_as_price_increases(self):
        cfg = noiseless_config()
        df = simulate_demand(
            make_frame([0.6, 0.8, 1.0]), cfg, np.random.default_rng(0)
        )
        demand = df["DEMAND"].tolist()
        assert demand[0] > demand[1] > demand[2]

    def test_promotion_lifts_demand(self):
        cfg = noiseless_config()
        rng = np.random.default_rng(0)
        base = simulate_demand(make_frame([1.0]), cfg, rng)["DEMAND"].iloc[0]
        promo = simulate_demand(
            make_frame([1.0], promotion=1), cfg, np.random.default_rng(0)
        )["DEMAND"].iloc[0]
        assert promo == pytest.approx(
            base * (1 + cfg.simulator.promotion_uplift)
        )

    def test_short_dated_stock_sells_less_at_same_price(self):
        cfg = noiseless_config()
        fresh = simulate_demand(
            make_frame([1.0], days=30), cfg, np.random.default_rng(0)
        )["DEMAND"].iloc[0]
        stale = simulate_demand(
            make_frame([1.0], days=1), cfg, np.random.default_rng(0)
        )["DEMAND"].iloc[0]
        assert stale < fresh

    def test_demand_never_negative(self):
        cfg = PricingConfig()
        cfg.simulator.noise_sigma = 1.5
        df = simulate_demand(
            make_frame(np.linspace(0.3, 1.0, 200)),
            cfg,
            np.random.default_rng(0),
        )
        assert (df["DEMAND"] >= 0).all()

    def test_deterministic_given_seed(self):
        cfg = PricingConfig()
        frame = make_frame(np.linspace(0.5, 1.0, 50))
        a = simulate_demand(frame, cfg, np.random.default_rng(7))["DEMAND"]
        b = simulate_demand(frame, cfg, np.random.default_rng(7))["DEMAND"]
        assert (a == b).all()


class TestHistoricalDiscounts:
    def test_creates_price_variation_and_stays_below_retail(self):
        cfg = PricingConfig()
        df = make_frame([1.0] * 500)
        out = apply_historical_discounts(df, cfg, np.random.default_rng(0))
        assert out["PRICE_RATIO"].nunique() > 100
        assert (out["PRICE_CURRENT"] <= out["PRICE_RETAIL"] + 1e-9).all()
        assert (out["PRICE_CURRENT"] > 0).all()


class TestElasticityEstimation:
    def test_recovers_known_elasticity(self):
        cfg = PricingConfig()
        cfg.elasticity.by_department = {"Bakery": -1.5}
        cfg.simulator.noise_sigma = 0.10
        rng = np.random.default_rng(3)
        ratios = rng.uniform(0.5, 1.0, size=400)
        df = simulate_demand(make_frame(ratios), cfg, rng)
        estimates = estimate_elasticities(df, cfg)
        est = estimates["Bakery"]
        assert est.source == "estimated"
        assert est.elasticity == pytest.approx(-1.5, abs=0.15)

    def test_falls_back_on_small_sample(self):
        cfg = PricingConfig()
        df = simulate_demand(
            make_frame([0.7, 0.9, 1.0]), cfg, np.random.default_rng(0)
        )
        estimates = estimate_elasticities(df, cfg)
        est = estimates["Bakery"]
        assert est.source == "default"
        assert est.elasticity == cfg.elasticity.default

    def test_falls_back_on_non_negative_estimate(self):
        # Demand that ignores price entirely -> estimate ~0 -> rejected.
        cfg = PricingConfig()
        rng = np.random.default_rng(1)
        df = make_frame(rng.uniform(0.5, 1.0, size=200))
        df["DEMAND"] = rng.uniform(15, 25, size=200)
        estimates = estimate_elasticities(df, cfg)
        assert estimates["Bakery"].source == "default"

    def test_estimated_elasticities_are_negative(self):
        cfg = PricingConfig()
        rng = np.random.default_rng(5)
        df = simulate_demand(
            make_frame(rng.uniform(0.5, 1.0, size=300)), cfg, rng
        )
        for est in estimate_elasticities(df, cfg).values():
            assert est.elasticity < 0


class TestDemandModel:
    def test_fit_predict_shapes_and_bounds(self):
        cfg = PricingConfig()
        rng = np.random.default_rng(0)
        df = simulate_demand(
            make_frame(rng.uniform(0.5, 1.0, size=300)), cfg, rng
        )
        model = DemandModel(seed=0, n_estimators=20).fit(df)
        preds = model.predict(df)
        assert len(preds) == len(df)
        assert (preds >= 0).all()
