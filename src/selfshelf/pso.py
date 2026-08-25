"""Particle swarm optimization over a 1-D bounded price range.

The swarm is fully deterministic given a numpy Generator: no module-level
random state is touched, so two runs with the same seed and configuration
produce identical recommendations.
"""

from typing import Callable, Tuple

import numpy as np

from .config import PSOConfig


def maximize(
    objective: Callable[[float], float],
    bounds: Tuple[float, float],
    config: PSOConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Return (best_price, best_score) maximizing ``objective`` on bounds.

    ``objective`` is any callable price -> score; the optimizer knows
    nothing about economics, keeping the search and the business logic
    decoupled.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"invalid bounds: {bounds}")
    if lower == upper:
        return lower, objective(lower)

    span = upper - lower
    n = config.num_particles

    positions = rng.uniform(lower, upper, size=n)
    # Evaluate the bound endpoints too: price ceilings/floors are often the
    # true optimum and a finite swarm can otherwise hover just inside them.
    positions[0] = lower
    positions[-1] = upper
    velocities = rng.uniform(-0.1 * span, 0.1 * span, size=n)

    pbest_pos = positions.copy()
    pbest_val = np.array([objective(p) for p in positions])

    g_idx = int(np.argmax(pbest_val))
    gbest_pos = pbest_pos[g_idx]
    gbest_val = pbest_val[g_idx]

    for _ in range(config.iterations):
        r1 = rng.random(n)
        r2 = rng.random(n)
        velocities = (
            config.inertia * velocities
            + config.cognitive * r1 * (pbest_pos - positions)
            + config.social * r2 * (gbest_pos - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)

        values = np.array([objective(p) for p in positions])

        improved = values > pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = values[improved]

        g_idx = int(np.argmax(pbest_val))
        if pbest_val[g_idx] > gbest_val:
            gbest_val = pbest_val[g_idx]
            gbest_pos = pbest_pos[g_idx]

    return float(gbest_pos), float(gbest_val)
