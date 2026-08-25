import numpy as np
import pytest

from selfshelf.config import PSOConfig
from selfshelf import pso


CFG = PSOConfig()


class TestMaximize:
    def test_finds_maximum_of_concave_function(self):
        best, value = pso.maximize(
            lambda p: -(p - 3.7) ** 2,
            (1.0, 6.0),
            CFG,
            np.random.default_rng(0),
        )
        assert best == pytest.approx(3.7, abs=0.01)
        assert value == pytest.approx(0.0, abs=0.001)

    def test_boundary_optimum_is_found_exactly(self):
        # Monotonically increasing objective: optimum sits on the upper
        # bound, which finite swarms often only hover near.
        best, _ = pso.maximize(
            lambda p: p, (2.0, 5.0), CFG, np.random.default_rng(0)
        )
        assert best == pytest.approx(5.0)

    def test_result_respects_bounds(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            lo = float(rng.uniform(0.5, 3.0))
            hi = lo + float(rng.uniform(0.1, 5.0))
            best, _ = pso.maximize(
                lambda p: np.sin(3 * p), (lo, hi), CFG,
                np.random.default_rng(2),
            )
            assert lo <= best <= hi

    def test_deterministic_given_seed(self):
        runs = [
            pso.maximize(
                lambda p: -(p - 2.2) ** 2 + 0.3 * np.sin(9 * p),
                (0.5, 4.0),
                CFG,
                np.random.default_rng(42),
            )
            for _ in range(2)
        ]
        assert runs[0] == runs[1]

    def test_degenerate_bounds_return_the_single_price(self):
        best, value = pso.maximize(
            lambda p: p * 2, (3.0, 3.0), CFG, np.random.default_rng(0)
        )
        assert best == 3.0
        assert value == 6.0

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            pso.maximize(lambda p: p, (5.0, 1.0), CFG, np.random.default_rng(0))
