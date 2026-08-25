"""Central configuration for all business assumptions.

Every economic constant the engine relies on lives here rather than being
hard-coded inside the optimizer or the demand model. Values that cannot be
derived from the dataset (inventory levels, cost ratios, salvage values,
elasticities) are explicit, documented assumptions that a deployment would
replace with real business data.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ElasticityConfig:
    """Own-price elasticity of demand per department.

    Elasticity e means: demand scales with (price / reference_price) ** e.
    Values must be negative (price up -> demand down). The defaults below are
    assumptions used by the synthetic demand simulator; the pipeline
    re-estimates elasticities from the (synthetic) sales data before
    optimizing, so the optimizer never reads these values directly.
    """

    default: float = -1.2
    by_department: Dict[str, float] = field(default_factory=lambda: {
        # Perishables shoppers can substitute easily -> more elastic.
        "Bakery": -1.8,
        "Deli": -1.6,
        "Frozen Foods": -1.3,
        # Habitual / branded purchases -> less elastic.
        "Snacks": -1.1,
        "Beverages": -0.9,
    })

    def for_department(self, department: str) -> float:
        return self.by_department.get(department, self.default)


@dataclass
class ExpiryConfig:
    """Shelf-life assumptions and the expiry-pressure curve.

    Expiry pressure follows exp(-(days_to_expiry - 1) / tau): ~1.0 with one
    day left, decaying smoothly toward 0 as shelf life remaining grows.
    """

    tau_days: float = 5.0
    default_shelf_life_days: int = 30
    shelf_life_by_department: Dict[str, int] = field(default_factory=lambda: {
        "Bakery": 7,
        "Deli": 5,
        "Snacks": 90,
        "Beverages": 120,
        "Frozen Foods": 180,
    })
    # How strongly shoppers avoid short-dated stock at an unchanged price
    # (0 = no aversion, 0.3 = demand drops up to 30% right before expiry).
    freshness_sensitivity: float = 0.3

    def shelf_life_for(self, department: str) -> int:
        return self.shelf_life_by_department.get(
            department, self.default_shelf_life_days
        )


@dataclass
class InventoryConfig:
    """Synthetic inventory generation and holding-cost assumptions.

    The dataset contains no stock quantities, so inventory is simulated as a
    lognormal number of days of supply around ``target_days_of_supply``. A
    deployment would replace this with real on-hand quantities.
    """

    target_days_of_supply: float = 6.0
    days_of_supply_sigma: float = 0.6
    min_units: int = 1
    # Cost of holding one unit for one day (storage, capital, shrink).
    # Default 0: we do not pretend to know it. Set > 0 when data exists.
    holding_cost_per_unit_day: float = 0.0


@dataclass
class WasteConfig:
    """Economic consequence of a unit expiring unsold."""

    # Fraction of unit cost recovered when an expired/clearance unit is
    # salvaged (donation tax credit, secondary channel, staff sale...).
    salvage_rate: float = 0.0
    # Extra disposal/handling cost per wasted unit.
    disposal_cost_per_unit: float = 0.0

    def unit_waste_loss(self, unit_cost: float) -> float:
        """Net loss when one unit expires unsold. Never negative."""
        recovered = self.salvage_rate * unit_cost
        return max(0.0, unit_cost - recovered) + self.disposal_cost_per_unit


@dataclass
class ConstraintConfig:
    """Hard pricing constraints applied to every recommendation."""

    # Minimum margin over cost for healthy (non-clearance) stock:
    # floor = cost * (1 + min_margin).
    min_margin: float = 0.05
    # Price floor for full clearance, as a fraction of cost. The effective
    # floor blends smoothly from the margin floor to this clearance floor as
    # expiry pressure rises from 0 to 1.
    clearance_floor_cost_fraction: float = 0.50
    # Largest allowed single-run price decrease, as a fraction of the
    # current price (0.40 -> price may not drop below 60% of current).
    max_daily_price_drop: float = 0.40
    # Whether the engine may recommend a price above the current price
    # (never above PRICE_RETAIL either way). Default matches the business
    # rule that Self-Shelf only maintains or marks down.
    allow_price_increase: bool = False
    max_daily_price_increase: float = 0.10
    # Markdowns smaller than this fraction of current price are treated as
    # noise and the price is maintained instead.
    min_meaningful_markdown: float = 0.01


@dataclass
class ObjectiveConfig:
    """Weights/frictions in the economic objective."""

    # Revenue horizon: expected sales are evaluated over
    # min(days_to_expiry, planning_horizon_days).
    planning_horizon_days: int = 14
    # Operational/brand friction per discounted dollar actually given away
    # (label changes, margin dilution ripple). Keeps the optimizer from
    # recommending tiny pointless markdowns.
    markdown_friction: float = 0.05


@dataclass
class PSOConfig:
    """Particle swarm parameters."""

    num_particles: int = 24
    iterations: int = 40
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5


@dataclass
class SimulatorConfig:
    """Parameters of the synthetic demand generator.

    Demand = base * price_effect * promotion_effect * freshness_effect
             * seasonality * noise
    """

    base_daily_demand: float = 20.0
    promotion_uplift: float = 0.35
    noise_sigma: float = 0.15
    season_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "Spring": 1.00,
        "Summer": 1.05,
        "Fall": 1.00,
        "Winter": 0.95,
    })
    # Share of rows given a simulated historical discount so the training
    # data contains price variation to learn from.
    historical_discount_share: float = 0.35
    historical_discount_range: tuple = (0.05, 0.40)


@dataclass
class PricingConfig:
    """Top-level configuration for a full pricing run."""

    seed: int = 42
    cost_ratio_of_retail: float = 0.70  # COST = PRICE_RETAIL * ratio
    sample_size: int = 5000
    elasticity: ElasticityConfig = field(default_factory=ElasticityConfig)
    expiry: ExpiryConfig = field(default_factory=ExpiryConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    waste: WasteConfig = field(default_factory=WasteConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)

    def describe(self) -> Dict[str, object]:
        """Flat summary of the assumptions behind a run, for logging."""
        return {
            "seed": self.seed,
            "cost_ratio_of_retail": self.cost_ratio_of_retail,
            "default_elasticity": self.elasticity.default,
            "expiry_tau_days": self.expiry.tau_days,
            "target_days_of_supply": self.inventory.target_days_of_supply,
            "holding_cost_per_unit_day":
                self.inventory.holding_cost_per_unit_day,
            "salvage_rate": self.waste.salvage_rate,
            "disposal_cost_per_unit": self.waste.disposal_cost_per_unit,
            "min_margin": self.constraints.min_margin,
            "clearance_floor_cost_fraction":
                self.constraints.clearance_floor_cost_fraction,
            "max_daily_price_drop": self.constraints.max_daily_price_drop,
            "allow_price_increase": self.constraints.allow_price_increase,
            "planning_horizon_days": self.objective.planning_horizon_days,
            "markdown_friction": self.objective.markdown_friction,
            "pso_particles": self.pso.num_particles,
            "pso_iterations": self.pso.iterations,
        }
