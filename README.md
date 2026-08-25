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
waste cost           = waste units × (unit cost − salvage value) × risk weight
```

Waste projected to occur **inside the decision window** is certain under the demand forecast, so it is charged in full. Waste projected beyond the window can still be averted by future repricing, so it is discounted exponentially with the time remaining after the window closes.

### Holding cost

Carrying cost applies to **unit-days actually held** as stock sells down (the integral of remaining inventory over time), so a markdown that accelerates sell-through genuinely reduces it. It defaults to 0 — no fabricated carrying rate — and setting it to a positive value makes earlier clearance measurably more attractive.

### The objective

PSO maximizes a modular economic score:

```text
score(P) = gross profit(P)            [revenue − COGS, in dollars]
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

Each dollar is counted once: COGS is charged only on units expected to sell, expired units lose `(cost − salvage)`, and unsold-but-unexpired units keep their book value.

### The markdown decision

The current price is always an explicit candidate — the optimizer's winner is compared against the **no-markdown baseline**, and a markdown is recommended only when

```text
Economic Value(markdown) > Economic Value(current price)
```

Nothing marks a product down merely because expiry is near: a near-expiry product with an inelastic demand response keeps its price when the waste avoided cannot pay for the margin given up.

Two classic sanity metrics are computed for every markdown:

- **Break-even unit uplift** — `(P₀ − C) / (P₁ − C)`, the volume multiplier needed to hold gross profit. A 20% markdown on a 40%-margin product needs ~2× the volume. Recommendations report this hurdle next to the predicted uplift, so it is explicit when a markdown is justified by waste/terminal-stock avoidance rather than by demand stimulation alone.
- **Timing advantage** — a simplified two-path comparison (markdown now for the whole window vs hold for a few days, then apply the same markdown) quantifying the value preserved by acting early. This is a fixed-path evaluation, not a full multi-period optimization; staged markdown scheduling remains future work.

Markdown **depth** is discovered by the optimizer from elasticity, inventory pressure, remaining time, margin and waste economics — there are no "urgent = 30% off" rules, and the deepest allowed discount is chosen only when it actually scores best.

### Explanations

Every recommendation carries a structured reason built from the actual numbers, e.g.:

```text
5.5 days of supply vs 5 days of shelf life remaining;
~8 of 95 units are expected to expire unsold at the current price;
expected sell-through rises from 91% to 100%;
marking down to $4.72 cuts expected waste by ~8 units;
demand is price-sensitive (elasticity -1.60): predicted daily demand rises from 17.4 to 19.0;
gross-profit break-even needs 1.22x unit volume; predicted uplift is 1.09x (does not meet the hurdle on margin alone);
acting now rather than waiting 3 day(s) preserves ~$7.99 of expected value;
expected economic outcome improves by $13.31 over the evaluation window
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
│       ├── backtest.py          hold-vs-recommended counterfactual replay
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
python src/main.py --backtest      # replay hold vs recommended (synthetic)
python src/main.py --seed 7        # different reproducible run
```

Output goes to `output/final_optimized_prices.csv` as a full economic audit of each decision:

| Field | Description |
|---|---|
| `Recommended_Price`, `Markdown_Percentage`, `Action` | the recommendation |
| `Days_To_Expiry`, `Inventory_Units`, `Days_Of_Supply` | inventory situation |
| `Elasticity`, `Expiry_Pressure`, `Inventory_Pressure` | economic inputs |
| `Predicted_Demand_Current` / `_Optimized` | daily demand at both prices |
| `Expected_Units_Sold_*`, `Sell_Through_*` | sell-down at both prices |
| `Gross_Revenue_*`, `Gross_Profit_*` | dollars over the evaluation window |
| `Expected_Waste_*`, `Terminal_Inventory_*`, `Holding_Cost_*` | stock outcomes |
| `Economic_Value_Current` / `_Optimized` / `_Improvement` | the actual decision criterion |
| `Break_Even_Unit_Uplift`, `Predicted_Unit_Uplift` | gross-profit hurdle vs forecast |
| `Economic_Reason` | structured explanation |

`--backtest` replays every recommendation day by day under the synthetic simulator's ground truth (identical noise for both strategies) and prints hold-vs-recommended revenue, gross profit, holding cost, economic value, units sold, waste, terminal inventory, sell-through and cash recovered. The accounting reconciles by construction (`units sold + terminal inventory = starting inventory`). The output is labeled a **synthetic simulation** — it measures optimization quality inside the simulated economy, not real-world performance.

`--sweep` additionally writes `final_optimized_prices_sweep.csv`: demand, revenue, profit, waste, and economic score across the feasible price range for each product — the data behind price–demand and profit curves.

---

## Testing

```bash
python -m pytest
```

The suite (≈150 tests) covers the economic primitives and, more importantly, **markdown-economics behavior**:

- healthy inventory is not marked down
- overstocked near-expiry stock receives meaningful downward pressure that measurably reduces expected waste
- extremely urgent stock gets strong clearance behavior within configured bounds
- near-expiry but *inelastic* products are **not** forced into markdowns — expiry alone never triggers a discount
- highly price-sensitive products get markdowns that *increase* expected profit; price-insensitive products keep their price
- the break-even uplift formula agrees with the objective's gross-profit crossover
- the deepest allowed discount is not automatically chosen (interior optima beat the floor; the recommendation is the argmax of an explicit candidate grid)
- tiny stock with strong demand keeps its price while large stock with weak demand at the same expiry is marked down
- zero holding cost creates no artificial markdown pressure; positive holding cost makes clearing measurably more attractive
- price-path evaluation reproduces the one-shot objective exactly, and acting now beats waiting for overstocked short-dated stock
- prices always respect bounds, waste/sales are never negative, zero-demand/zero-inventory/cost-above-price edge cases are safe
- the whole pipeline and the backtest are reproducible end-to-end given a seed

An audit suite additionally locks down: dimensional reconciliation of every objective component ($ = $/unit × units; the score equals the sum of its parts; units sold + terminal inventory = starting inventory), monetary-scale invariance, sensitivity monotonicity (waste cost, holding cost, inventory, remaining days, elasticity), PSO agreement with a 1001-point explicit grid on random products, whole-cent rounding invariants, the break-even/gross-profit equivalence, four distinct economic regimes (hold / shallow / moderate / deep markdown), and backtest counterfactual fairness (recommending the current price reproduces the hold strategy exactly).

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

Beyond data, two modeling limitations are known and deliberate:

- **One-shot pricing.** The engine picks one price per run. The price-path evaluator provides the foundation for staged markdowns (small cut → observe → deeper cut), but full multi-period optimization with daily re-decisions is future work — the current timing comparison evaluates fixed paths only.
- **Myopic customers.** Demand depends only on today's price. The markdown literature shows strategic customers who anticipate future discounts can change optimal markdown policies; nothing here models that, and no claim is made either way.

---

## Author

**Shreshtha Bindal**

Computer Engineering
Mukesh Patel School of Technology Management & Engineering (MPSTME), NMIMS, Mumbai

GitHub: https://github.com/Solly2201

---

## License

This project is intended for academic and educational purposes.
