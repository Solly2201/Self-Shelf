# Self-Shelf v1.1.0 — Adaptive Retail Markdown Optimization

## Highlights

- **Closed-loop daily re-optimization** — the controller acts a planned
  price, observes *actual* sales, updates inventory and its demand-level
  belief, and re-optimizes the remaining path from realized state.
  Planned vs actual is explicit every day; episodes are deterministic.
- **Elasticity uncertainty** — every estimated elasticity now carries an
  OLS standard error, a 95% confidence interval, observation counts,
  distinct price levels, and a restrained High/Medium/Low confidence
  label. Point estimates are bit-identical to the frozen estimator;
  fallback values are labelled as fallbacks and never receive a
  fabricated interval.
- **Replenishment-aware optimization** — known future deliveries
  (date, product, quantity) enter the multi-period plan as
  time-respecting events: FIFO lots, no early selling of future stock,
  no resurrection of expired stock, and an exact units accounting
  identity. Absent data is stated explicitly, never assumed to be zero
  silently.
- **Adaptive dashboard** — elasticity-confidence, replenishment, and
  closed-loop simulation cards on product detail; deliveries marked on
  the price-path timeline; a hold vs open-loop vs closed-loop backtest
  (common random numbers) in Analytics; replenishment upload in the
  custom-data flow; enriched CSV exports plus a per-category
  elasticity/confidence export.
- **Custom data integration** — the full upload → map → validate →
  import flow now covers the optional replenishment file with the same
  rejected-rows transparency as products and transactions.
- **Multi-period optimization** — unchanged candidate class and
  tie-breaking; the replenishment-aware variant differs only in
  valuation and reproduces the frozen optimizer exactly when no
  deliveries exist.
- **400 tests, 0 failures** — all 302 V1.0 tests preserved unmodified,
  98 added (uncertainty statistics, frozen-parity locks, time-respecting
  inventory, closed-loop feedback/determinism/leakage, common-random-
  number backtests, end-to-end custom integration, and an adversarial
  scenario matrix).

The audited economic engine (`economics.py`, `pathopt.py`,
`optimizer.py`, `pso.py`, `backtest.py`, `pathbacktest.py`, `demand.py`)
has **zero diffs** relative to V1.0.

## Important limitations

- Closed-loop episodes run against **labeled synthetic observations**
  (what-if demand environments, or the synthetic simulator in
  backtests). This is a simulator/controller, not production event
  streaming.
- Confidence intervals are finite-sample OLS estimates under the
  regression's assumptions — honest statistics, not guarantees.
- Replenishment awareness is only as good as the supplied schedule;
  deliveries default to "fresh within the planning window" when shelf
  life is not provided.
- No competitor modeling, no strategic customer behavior.
- No result in any mode is presented as real-world retail performance.
