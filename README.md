# Self-Shelf

### Expiry-Aware Dynamic Pricing for Retail Inventory

Self-Shelf is a dynamic-pricing engine for perishable retail inventory. It combines a machine-learning demand forecast with an **explicit economic layer** — price elasticity, expiry pressure, inventory pressure, and expected waste — and uses **Particle Swarm Optimization (PSO)** to pick the price with the best expected economic outcome.

The design principle:

> **The ML model forecasts demand. The economics are encoded explicitly. The optimizer just searches.**

---

## Architecture

```text
Demand Model (Random Forest)
        │  baseline daily demand at the current price
        ▼
Economic Layer
        │  elasticity · expiry pressure · inventory pressure
        │  expected waste · constraints
        ▼
PSO
        │  searches the constrained price range
        ▼
Recommendation
           price · action · structured economic explanation
```

### Why not let the ML model discover the economics?

A tree ensemble trained on observed sales does not reliably encode the price–demand relationship: it cannot extrapolate outside the training distribution, and its prediction surface can be flat or even non-monotonic in price. An optimizer driven by such a surface either never marks anything down or does so for spurious reasons.

Self-Shelf therefore splits responsibilities:

- The **Random Forest** answers *"how much of this sells per day at today's price?"* — a forecasting problem trees are good at.
- The **economic layer** answers *"how does demand move when the price moves?"* using an explicit constant-elasticity relationship, which guarantees the monotonic behavior (price ↓ → demand ↑) the optimizer depends on.

```text
demand(P) = baseline_forecast × (P / P_current) ^ elasticity
```

---

## The Economic Layer

### Price elasticity

Each department has an own-price elasticity (e.g. `-1.8` = highly price-sensitive). Elasticities are **estimated from sales data** by department-level log-log regression, with a configurable default as fallback when a department has too few observations or the estimate is not economically credible (non-negative). Nothing in the optimizer reads hard-coded elasticities.

### Expiry pressure

Instead of a binary `days_to_expiry ≤ 3` switch, clearance urgency follows a smooth exponential curve:

```text
pressure = exp(-(days_to_expiry - 1) / τ)      τ = 5 days by default
```

~1.0 with one day left, decaying toward 0 as shelf life grows — no arbitrary cliffs.

### Inventory pressure

The engine compares **days of supply** (`inventory / predicted daily demand`) with remaining shelf life. 5 units with 10 days left is a very different problem from 500 units with 10 days left, even at identical expiry dates. The fraction of stock that cannot sell before expiry at the current demand rate is the product's inventory pressure.

### Expected waste

```text
expected waste units = max(0, inventory − daily demand × days_to_expiry)
waste cost           = waste units × (unit cost − salvage value) × expiry pressure
```

The expiry-pressure weighting discounts waste that is still far in the future (there will be later chances to correct course) while charging the full loss for stock expiring now.

### The objective

PSO maximizes a modular economic score:

```text
score(P) = expected revenue(P)
         − COGS(P)
         − expected waste cost(P)
         − holding cost(P)
         − markdown friction(P)
```

subject to constraints:

- minimum margin over cost for healthy stock
- a clearance floor (fraction of cost) that the margin floor **blends toward smoothly** as expiry pressure rises
- maximum single-run price drop
- ceiling at the current price (price increases are disabled by default, configurable)
- never above the retail/list price

A markdown is only recommended when it **improves** the economic score; otherwise the price is maintained.

### Explanations

Every recommendation carries a structured reason built from the actual numbers, e.g.:

```text
8.0 days of supply vs 4 days of shelf life remaining;
expiry pressure is high (0.55) with 4 day(s) left;
~77 of 154 units are expected to expire unsold at the current price;
marking down to $2.32 cuts expected waste by ~34 units;
demand is price-sensitive (elasticity -1.81): predicted daily demand rises from 19.3 to 27.8;
expected economic outcome improves by $3.76 over the evaluation window
```

---

## Important Limitation: Synthetic Demand

**The dataset contains no historical sales volume, cost, expiry dates, or inventory levels.** Those are simulated:

- **Demand** comes from an explicit multiplicative simulator:
  `base × price effect (constant elasticity) × promotion × freshness × seasonality × noise`
- **Cost** is assumed to be 70% of retail price
- **Days to expiry** are drawn uniformly within department shelf-life assumptions
- **Inventory** is drawn as lognormal days-of-supply around a configurable target

This creates an unavoidable circularity: the elasticities the pipeline estimates are the ones the simulator put in. The estimation step exists to demonstrate the *methodology* (recovering price response from sales data), not to prove the model learned real-world economics — **it did not, because no real demand data is available.** With real transaction history, the simulator would be dropped and the same estimation/optimization machinery would run on observed sales.

Every simulated quantity is a named, documented parameter in `src/selfshelf/config.py`, not a magic number.

---

## Project Structure

```text
Self-Shelf/
├── src/
│   ├── main.py                  CLI entry point
│   └── selfshelf/
│       ├── config.py            all business assumptions (dataclasses)
│       ├── economics.py         elasticity, expiry/inventory pressure,
│       │                        waste, objective, price bounds
│       ├── demand.py            demand simulator, RF model, elasticity
│       │                        estimation
│       ├── features.py          loading, cleaning, feature engineering
│       ├── evaluation.py        train/validation/test split + metrics
│       ├── pso.py               deterministic particle swarm search
│       ├── optimizer.py         per-product optimization + explanations
│       └── pipeline.py          end-to-end orchestration
├── tests/                       unit + behavioral test suite
├── data/                        retail dataset (prices, departments,
│                                promotions)
└── output/                      generated recommendations
```

---

## Model Evaluation

The demand model is evaluated on a leak-free 60/20/20 **train / validation / test** split. Elasticity estimation uses training data only; products to optimize are drawn from the test set. A typical run reports:

```text
validation  MAE=2.94  RMSE=3.92  R²=0.611
test        MAE=2.75  RMSE=3.53  R²=0.655
```

(R² is capped by the simulator's irreducible noise — the model cannot, and should not, explain the lognormal noise term.)

---

## Reproducibility

A single seed in the configuration drives everything: sampling, simulation, model training, and the PSO search. Each product gets its own deterministic RNG stream derived from the master seed, so results do not depend on how many items are optimized or in what order. Two runs with the same seed produce byte-identical output, and the CLI prints the full configuration used for each run.

---

## Running the Project

```bash
git clone https://github.com/Solly2201/Self-Shelf.git
cd Self-Shelf
pip install -r requirements.txt

python src/main.py                 # optimize 25 test-set products
python src/main.py -n 100          # optimize 100
python src/main.py --sweep         # also write per-product price sweeps
python src/main.py --seed 7        # different reproducible run
```

Output goes to `output/final_optimized_prices.csv` with columns including:

| Field | Description |
|---|---|
| `Optimized_Price`, `Markdown_Percentage`, `Action` | the recommendation |
| `Days_To_Expiry`, `Inventory_Units`, `Days_Of_Supply` | inventory situation |
| `Elasticity`, `Expiry_Pressure` | economic inputs |
| `Predicted_Demand_Current` / `_Optimized` | demand at both prices |
| `Expected_Revenue`, `Expected_Profit` | over the evaluation window |
| `Expected_Waste_Units` (+ at current price) | waste impact of the markdown |
| `Economic_Reason` | structured explanation |

`--sweep` additionally writes `final_optimized_prices_sweep.csv`: demand, revenue, profit, waste, and economic score across the feasible price range for each product — the data behind price–demand and profit curves.

---

## Testing

```bash
python -m pytest
```

The suite (≈90 tests) covers the economic primitives and, more importantly, **business behavior**:

- healthy inventory is not marked down
- overstocked near-expiry stock receives meaningful downward pressure that measurably reduces expected waste
- extremely urgent stock gets strong clearance behavior within configured bounds
- highly price-sensitive products get markdowns that *increase* expected profit
- price-insensitive products keep their price
- prices always respect bounds, waste/sales are never negative, zero-demand edge cases are safe
- the whole pipeline is reproducible end-to-end given a seed

---

## Assumptions That Need Real Business Data

Before production use, these configured assumptions must be replaced with real data:

| Assumption | Config | Default |
|---|---|---|
| Unit cost | `cost_ratio_of_retail` | 70% of retail |
| Department elasticities (fallback) | `ElasticityConfig` | −0.9 … −1.8 |
| Shelf life per department | `ExpiryConfig` | 5–180 days |
| Expiry-pressure time scale | `tau_days` | 5 days |
| On-hand inventory | `InventoryConfig` | ~6 days of supply |
| Salvage value / disposal cost | `WasteConfig` | 0 |
| Holding cost | `holding_cost_per_unit_day` | 0 |
| Minimum margin / clearance floor | `ConstraintConfig` | 5% / 50% of cost |
| Markdown friction | `ObjectiveConfig` | 5% of discounted dollars |

---

## Author

**Shreshtha Bindal**

Computer Engineering
Mukesh Patel School of Technology Management & Engineering (MPSTME), NMIMS, Mumbai

GitHub: https://github.com/Solly2201

---

## License

This project is intended for academic and educational purposes.
