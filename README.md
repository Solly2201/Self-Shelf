# Self-Shelf

### Expiry-Aware Dynamic Pricing for Retail Inventory

Self-Shelf is a dynamic-pricing system for perishable retail inventory. It combines a machine-learning demand forecast with an **explicit economic layer** — price elasticity, expiry pressure, inventory pressure, and expected waste — and determines not only **whether** a product should be marked down, but **when, by how much, and why**: a one-shot optimal price today, and a **multi-period price path** over the remaining shelf life. It runs on either a synthetic demonstration simulation or **your own inventory and transaction history**, imported through the dashboard.

The design principle:

> **The ML model forecasts demand. The economics are encoded explicitly. The optimizer just searches.**

---

## Architecture

```text
Data (synthetic simulator  OR  imported products + sales history)
        │
        ▼
Demand layer
        │  baseline daily demand · price elasticity
        │  (estimated from observed sales, with labelled fallbacks)
        ▼
Economic layer
        │  expiry pressure · inventory pressure · expected waste
        │  salvage · holding cost · break-even · constraints
        ▼
Optimization
        │  PSO over the price range (today's price)
        │  exhaustive staged-schedule search (multi-period path)
        ▼
Explainable recommendations
        │  price · path · action · structured economic explanation
        ▼
Dashboard + API + CSV export
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
- **Timing advantage** — a simplified two-path comparison (markdown now for the whole window vs hold for a few days, then apply the same markdown) quantifying the value preserved by acting early. The full multi-period optimizer (below) generalizes this to arbitrary staged schedules.

Markdown **depth** is discovered by the optimizer from elasticity, inventory pressure, remaining time, margin and waste economics — there are no "urgent = 30% off" rules, and the deepest allowed discount is chosen only when it actually scores best.

### Multi-period markdown optimization

The engine also answers *"what pricing path should this product follow over its remaining shelf life?"*:

```text
Day 0   $4.99      hold
Day 1   $4.99      hold
Day 2   $4.49      first markdown
Day 3   $4.49
Day 4   $3.99      clearance step
```

The path optimizer (`pathopt.py`) is an adapter over the audited path evaluator `economics.evaluate_price_path` — the same accounting as the one-shot objective (a one-segment path reproduces `score(P)` exactly). It **exhaustively enumerates** an operationally realistic policy class: piecewise-constant schedules with at most two downward price moves at daily granularity (retailers do not re-sticker shelves hourly), subject to

- the expiry-blended price floor (identical to the one-shot bounds, cross-checked by test)
- no price increases, monotonically non-increasing paths
- a maximum per-day price move
- the minimum-meaningful-markdown convention

Because the **hold-forever schedule is always a candidate** and ties break toward holding and later/shallower markdowns, a markdown path is only ever recommended when it strictly beats holding — the optimizer explicitly understands that *a markdown today is not free when repricing tomorrow is possible*, which is what prevents unnecessarily early markdowns. This is deliberately **not** "run the daily optimizer repeatedly" (a greedy approach that ignores the value of waiting).

The method is deterministic (no randomness anywhere), exactly optimal within its candidate class, and validated against an independent brute-force enumeration over all feasible daily price sequences in the tests. Each product's result reports the recommended schedule, a day-by-day expected trajectory (demand, sales, inventory, revenue — proven to decompose the evaluator's totals exactly), and comparisons against two baselines: **hold throughout** and **immediate one-shot markdown**.

A three-strategy counterfactual backtest (hold vs immediate markdown vs optimized path) replays all three under the synthetic simulator with identical demand noise — synthetic simulation only, and only available in demo mode, because imported data has no ground-truth simulator to replay against.

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

## V1.1 — Adaptive Retail Optimization

V1.1 turns the open-loop optimizer into an adaptive system built from three pieces that share one loop:

```text
current state (inventory, actual sales, deliveries, days to expiry)
      → demand belief + elasticity (with confidence)
      → frozen economic engine
      → multi-period optimizer (replenishment-aware)
      → act today's price
      → observe actual sales
      → update state and beliefs
      → RE-OPTIMIZE, daily, until expiry
```

The frozen economic engine is untouched: every new layer *delegates* its valuation to the audited primitives, and parity tests prove that with the new features switched off (no deliveries, no feedback) each layer reproduces the frozen engine's numbers exactly.

### Closed-loop re-optimization (open-loop vs closed-loop)

The V1.0 path optimizer is **open-loop**: it plans day 0…N and assumes reality complies. If it expects to sell 20 units on day 0 and only 8 sell, the day-1 plan is already wrong — inventory is 12 units higher than the plan believed.

The V1.1 controller (`closedloop.py`) is **closed-loop**. `RetailState` tracks realized state (inventory by lot, price in effect, shelf life, cumulative sales/waste, deliveries received, last forecast vs last observation), and every day the controller:

1. re-optimizes the remaining path from the *actual* state,
2. acts today's planned price,
3. observes actual (not forecast) sales,
4. applies sales FIFO (expiring stock first), books tonight's expiry waste, receives tomorrow's delivery,
5. smooths its demand-level belief toward the observation — only on days *not* censored by a stockout (selling every unit proves demand ≥ sales, so such days must not drag the forecast down), and never touching elasticity (one day at one price carries no information about the price response),
6. repeats until expiry.

Planned vs realized is explicit: every day records the planned path, the forecast, the observation, and the surprise. Determinism: the same product, config, environment, and delivery schedule reproduce the identical episode.

A three-strategy backtest (hold / execute the day-0 plan blind / re-optimize daily) replays all strategies under the synthetic simulator with **common random numbers**; on the default 50-product demo run the closed loop recovers more economic value than the open-loop plan, which beats holding. Synthetic simulation only — it measures *adaptive behavior*, not real-world performance.

### Elasticity uncertainty

The engine has always priced with a point elasticity (say −1.67). V1.1 answers *how much that number should be trusted* (`uncertainty.py`): the identical log-log regression is run with standard OLS inference on top —

```text
elasticity   −1.67
SE            0.18
95% CI       [−2.02, −1.32]
270 observations across 5 distinct price levels
```

Point estimates are bit-identical to the frozen estimator by construction (and by test), so the confidence layer can never disagree with the engine about *what* the elasticity is — only report how well-identified it is. Confidence labels are coarse and rule-based (High / Medium / Low, documented thresholds on the relative CI half-width, observation count, and distinct price levels). **Fallback elasticities get no interval**: a configured constant has no sampling distribution, and the UI says "Fallback estimate" instead of inventing one.

Confidence is deliberately **information, not a pricing penalty** — the recommendation still comes from the frozen economics. No hidden uncertainty adjustment contaminates the baseline.

### Replenishment

Inventory can now rise: `replenishment.py` models **known future deliveries** (date, product id, quantity) as time-respecting events. Stock is held in two lots — the expiring on-shelf lot and fresh arrivals — rotated FIFO, so:

- a delivery on day *d* becomes sellable at the start of day *d* and *never earlier* (a "day 0" delivery is a hard error: already-arrived stock belongs in inventory, and letting it into the planner would double-count it);
- only the expiring lot can become waste, priced by the frozen waste economics; expired stock is never resurrected as fresh stock;
- the accounting identity `initial inventory + arrivals = sales + terminal inventory` holds throughout.

The multi-period optimizer enumerates the identical candidate class with the identical tie-breaking — only the valuation sees the arrivals — and with an empty schedule it returns exactly the frozen optimizer's result (tested). When no replenishment file is supplied the dashboard states **"No replenishment data supplied; inventory is modeled as non-replenished"** rather than silently assuming anything; in demo mode it says the simulator does not model deliveries at all.

### What V1.1 explicitly does not claim

- The closed loop runs against *simulated* observations (a labeled what-if demand environment, or the synthetic simulator in backtests) — this is a closed-loop **simulator/controller**, not a production event-streaming deployment.
- Confidence intervals are finite-sample statistical estimates under the regression's assumptions, not guarantees.
- Replenishment awareness is only as good as the supplied schedule; delivery shelf life defaults to "fresh" (no expiry inside the planning window) when the data does not say otherwise.
- Strategic customers and competitor behavior remain out of scope, as before.

---

## Two Data Modes

Self-Shelf runs in one of two explicitly separated modes; the dashboard always shows which one is active, and the two are never mixed.

### Demo mode (synthetic simulation)

The default. Demand, costs, expiry dates, and inventory come from the documented synthetic simulator below. The badge reads **Simulation mode** and no figure is presented as real-world performance.

### Custom data mode

Import your own data on the dashboard's **Data** page (or via the API). Two CSV files:

**Products** — one row per product, the current state of the shelf:

| Field | Required | Notes |
|---|---|---|
| `product_id` | ✔ | any unique identifier (SKU, UPC, …) |
| `product_name` | ✔ | |
| `current_price` | ✔ | today's shelf price |
| `cost` | ✔ | unit procurement cost |
| `inventory` | ✔ | units on hand |
| `days_to_expiry` | ✔ | remaining shelf life in days |
| `category` | | used to pool elasticity estimation (default "General") |
| `retail_price` | | list price; defaults to `current_price` |
| `promotion` | | 0/1 flag |

**Transactions** — one row per product per day of sales history (**required** — demand is estimated from observed sales, and Self-Shelf refuses to invent it):

| Field | Required | Notes |
|---|---|---|
| `date` | ✔ | daily granularity; missing days count as zero-sale days |
| `product_id` | ✔ | must match the products file |
| `price` | ✔ | price charged that day |
| `units_sold` | ✔ | units sold that day |
| `promotion` | | 0/1 flag, used as a regression control |

**Replenishment** — optional, one row per future delivery (without it, inventory is modeled as non-replenished and the dashboard says so):

| Field | Required | Notes |
|---|---|---|
| `date` | ✔ | delivery date; dates on/before the data's "today" (the day after the last recorded sale) are treated as already in inventory and excluded |
| `product_id` | ✔ | must match the products file |
| `quantity` | ✔ | units received (non-negative) |

The import flow: **upload → map columns → validate → preview → import**. Column names don't need to match — the importer suggests a mapping from common synonyms (`sku` → `product_id`, `qty_sold` → `units_sold`, …) and every suggestion can be corrected manually. Validation rejects malformed rows (missing ids, negative prices/costs/inventory, unparseable dates, duplicate product ids, transactions for unknown products) with **human-readable per-issue counts, and the rejected rows are downloadable** — nothing is silently dropped. A quality summary reports how much information the data actually contains: products with/without history, transaction counts, date range.

**Elasticity is estimated from your history**, per category, with the same log-log regression the synthetic pipeline uses (`log Q ~ α + e·log(P/P_list) + promo`), now with a standard error and 95% confidence interval alongside every estimated value (see the V1.1 section). All history precedes "now", so nothing leaks from the future into the estimate; a stability check re-runs the estimator on only the earliest 80% of observations and flags estimates that move materially. Categories with too few observations, no price variation, or an implausible estimate fall back to the configured default elasticity **and are labelled as fallbacks everywhere they appear** — the dashboard never pretends an assumption was learned from your data. Baseline daily demand per product is its recent observed sales rate translated to the current price via the elasticity; products with no history fall back to category/global medians, again labelled.

Imported datasets persist under `data/custom/` (gitignored — private retail data can never be committed) and survive restarts, including which source was active. There is **no synthetic backtest in custom mode**: no ground-truth simulator exists for real data, and the dashboard says so instead of fabricating one.

Sample files to try the flow: `data/sample_custom_products.csv`, `data/sample_custom_transactions.csv`, and `data/sample_custom_replenishment.csv` (12 products, 90 days of history, two products deliberately without history to demonstrate the labelled fallbacks, and a small delivery schedule — including one past delivery that the importer correctly treats as already on the shelf).

---

## Important Limitation: Synthetic Demand (demo mode)

**The demo dataset contains no historical sales volume, cost, expiry dates, or inventory levels.** Those are simulated:

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
│   ├── serve.py                 dashboard server entry point
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
│       ├── pipeline.py          end-to-end orchestration
│       ├── pathopt.py           multi-period path optimizer (adapter over
│       │                        the frozen path evaluator)
│       ├── pathbacktest.py      hold vs immediate vs path replay
│       ├── customdata.py        custom CSV import: mapping, validation,
│       │                        persistence, historical demand params,
│       │                        replenishment ingestion
│       ├── uncertainty.py       elasticity standard errors / confidence
│       │                        intervals (same regression, OLS inference)
│       ├── replenishment.py     known future deliveries: schedules,
│       │                        lot-based replay, aware path optimizer
│       ├── closedloop.py        RetailState + daily observe/re-optimize
│       │                        controller and demand environments
│       ├── adaptivebacktest.py  hold vs open-loop vs closed-loop and
│       │                        naive-vs-aware replenishment backtests
│       ├── webdata.py           dashboard service layer over the engine
│       │                        (synthetic + custom sources)
│       └── webapp.py            FastAPI app serving data + dashboard
├── web/                         dashboard frontend (no build step)
├── tests/                       unit + behavioral test suite
├── data/                        demo dataset + sample custom CSVs
│   └── custom/                  imported user data (gitignored)
├── Dockerfile, compose.yaml     containerized run
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

## Dashboard

```bash
python src/serve.py                # http://127.0.0.1:8765
python src/serve.py -n 100         # optimize 100 products instead of 50
```

Or with Docker:

```bash
docker compose up --build          # same app on http://127.0.0.1:8765
```

The compose file mounts `data/custom/` so imported datasets survive container restarts.

A pricing-intelligence interface on top of the engine. The optimization runs once at startup (the UI shows progress until it finishes); every number displayed is computed by the engine — the frontend contains no economic formulas.

- **Overview** — portfolio KPIs, the markdown queue ranked by expected economic improvement, inventory risk bands, the hold-vs-recommended synthetic backtest (demo mode), and an economic-value curve explaining the top recommendation.
- **Products** — searchable, sortable table of every recommendation with filters (markdown / hold / at risk / near expiry / high inventory pressure) and CSV export.
- **Product detail** — the full story for one decision: economic value and demand curves across the allowed price range, keep-vs-recommended breakdown, break-even economics, structured reasons, the **recommended multi-day price path** (vs hold and immediate markdown, with incoming deliveries marked on the timeline), an **elasticity confidence** card (estimate, 95% CI, SE, observation counts, High/Medium/Low/Fallback), a **replenishment** card (next delivery and quantities, or an explicit no-data statement), a **closed-loop simulation** card (planned vs acted prices, forecast vs observed sales per day, and the measured value of daily feedback under a labeled synthetic what-if), demand-parameter provenance in custom mode, and a scenario slider that evaluates any allowed price live in the backend engine.
- **Pricing** — a scenario lab with two modes: **single price** (slider across the allowed range) and **price path** (edit a price per day; the whole path is validated and valued by the backend, and compared against hold and the optimized schedule).
- **Analytics** — markdown depth distribution, risk bands, days-of-supply distribution, expected waste by department, current→recommended price changes, the **three-strategy multi-period backtest**, and the **closed-loop strategy backtest** (hold vs open-loop vs daily re-optimization; demo mode).
- **Data** — the custom-data workflow: upload (including the optional replenishment file), column mapping, validation with downloadable rejected rows, quality summary, per-category elasticity provenance with confidence intervals, source switching, and CSV exports (recommendations with confidence/replenishment columns, per-day price paths with delivery events, and a per-category elasticity/confidence file).

Architecture: the frozen pricing engine → an in-process service layer (`webdata.py`, one presentation layer over two data sources) → a thin FastAPI adapter (`webapp.py`) → a dependency-free static frontend (`web/`). A parity test asserts the synthetic service serves recommendations identical to the CLI pipeline's output. The frontend contains **no economic formulas** in either mode.

The UI is explicitly labeled **Simulation mode** or **Custom data** at all times.

---

## Testing

```bash
python -m pytest
```

The suite (400+ tests) covers the economic primitives and, more importantly, **markdown-economics behavior**:

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

The dashboard layer has its own tests: the service adapter must serve recommendations byte-identical to the CLI pipeline (parity), aggregates must be exact sums over the served products, scenario evaluations must reproduce the engine's breakdown at the current price and clamp to the allowed range, and the API endpoints are exercised end to end (including 404s, validation, and the not-ready state during startup).

The multi-period suite locks down: agreement with an independent brute-force enumeration of all feasible daily price sequences (small horizons), a one-segment path reproducing the one-shot objective exactly, hold/immediate baselines, wait-then-markdown beating immediate markdown where the economics favor waiting, expiry/inventory/elasticity moving the path in the economically correct direction, every constraint (floor, ceiling, monotonicity, max daily move) on every day of every path, exact decomposition of the evaluator's totals into the daily trajectory, determinism, and constant-path parity between the path backtest and the frozen single-price backtest.

The custom-data suite covers mapping suggestions, row-level validation and reject reasons, currency parsing, duplicate/unknown-id handling, persistence round trips, elasticity recovery from generated history (and every fallback reason: too few observations, no price variation, no history), the zero-sale-day convention, provenance labelling, mode isolation, restart persistence, and the full upload → map → validate → import API flow.

The V1.1 adaptive suite adds: point-estimate parity between the confidence layer and the frozen estimator, CI coverage of a known synthetic elasticity across seeds, precision improving with more observations and more price variation, no fake intervals on fallbacks, degenerate-regression safety; exact parity of the replenishment replay/optimizer with the frozen evaluator/optimizer under an empty schedule, time-respecting arrivals (future stock is never sold early), FIFO expiry (fresh deliveries never become waste), the extended accounting identity; closed-loop state transitions driven by observed (not forecast) sales, censored-day belief handling, determinism, a leakage test proving future demand cannot change today's price, bit-parity of a fixed hold path with the frozen backtest replay, the hold/open-loop/closed-loop and naive/aware-replenishment backtests under common random numbers; an end-to-end API integration test (custom data + replenishment file + confidence + closed loop); and an adversarial matrix crossing elasticity × inventory × expiry × replenishment × observed-sales scenarios, asserting constraint/accounting invariants and economically sensible directionality in every cell (never hard-coded answers).

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

Beyond data, several modeling limitations are known and deliberate:

- **Closed-loop simulation, not production streaming.** The V1.1 controller re-optimizes daily against *simulated* observations (labeled what-if environments, or the synthetic simulator in backtests). Wiring it to a live point-of-sale feed is an integration exercise, not something this repository claims to have done.
- **Candidate class.** Paths are restricted to at most two downward moves at daily granularity — operationally realistic, exhaustively searched, but not the unrestricted optimum.
- **Confidence intervals are estimates.** They are finite-sample OLS intervals under the regression's assumptions (log-linear demand, independent errors); they are honest statistics, not guarantees, and the demand-level belief update is deliberately simple exponential smoothing on uncensored days only.
- **Replenishment depends on data quality.** Only deliveries present in the supplied schedule are known to the planner; delivery shelf life defaults to "fresh within the planning window" when the data does not say otherwise. When no data is supplied, the system says so and models no replenishment.
- **Myopic customers.** Demand depends only on today's price. Strategic customers who anticipate future discounts can change optimal markdown policies; nothing here models that, and no claim is made either way.
- **No competition.** Competitor prices and cross-product cannibalization remain out of scope.

### Future work

Online elasticity re-estimation across episodes (with the uncertainty machinery guarding against overreacting to noise), an explicit uncertainty-aware conservative pricing mode (evaluating recommendations across the CI rather than at the point estimate), lot-level expiry tracking for deliveries with known shelf lives, competitor-price ingestion, authentication/multi-user deployment, and a database-backed store for large datasets.

---

## Author

**Shreshtha Bindal**

Computer Engineering
Mukesh Patel School of Technology Management & Engineering (MPSTME), NMIMS, Mumbai

GitHub: https://github.com/Solly2201

---

## License

This project is intended for academic and educational purposes.
