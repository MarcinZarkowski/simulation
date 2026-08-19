# Options Backtester — Plan to Industry Standard

Roadmap for turning this engine into one whose P&L a derivatives trader or risk
officer would act on.

Written against `BACKTESTER_GOALS.md`, clause by clause, and against the data
contract in `../optionsdata/README.md`. The engine was rewritten from a
predecessor that never read an option price; this plan covers what remains.

**Hard constraint:** free data sources only. That is why execution cost is Monte
Carlo simulated rather than reconstructed, and why the honesty of that
simulation is the single most important open item.

### Dependency on the pipeline

Two phases here cannot start until the pipeline delivers:

| Needs from `../optionsdata/PLAN.md` | Blocks |
|---|---|
| Phase 1 — quote sampler and spread calibration artifact | **B2**, and all of Phase 1 |
| Phase 2 — OCC memo ingestion, confirmed lineage | Carrying a position through an adjustment |
| Phase 1 — exercise style and settlement type columns | **M6**, index and cash-settled options |
| Phase 1 — declared dividends with declaration dates *(already produced)* | **M5**, early assignment |

Phase 0 below is entirely independent of the pipeline and is the highest
value-per-hour work in either repo: it removes silent wrongness using only code
already here.

---

## 1. Where this stands today

Verified by reading the code and running it.

### Implemented and tested

- **Exact accounting.** int64 microdollars throughout; the ledger reconciles to
  its append-only journal as an integer equality, asserted in every test harness
  teardown.
- **Contract-version identity.** Positions key on `contract_version_id`, never a
  symbol, so the same OCC symbol either side of an adjustment cannot be conflated.
- **No same-bar lookahead.** Default execution is next-bar-open; a signal from a
  bar's close fills at the following bar's open. Fills are never inferred from a
  high or low crossing a limit.
- **Atomic multi-leg groups.** All legs fill or none. Verticals, calendars,
  diagonals, butterflies, condors, straddles, strangles, ratios and rolls are all
  expressed this way.
- **Margin that pairs by expiry and nets by max loss.** A long only offsets a
  short if it lives at least as long. Spread requirements use the max-loss netting
  of FINRA 4210(f)(2)(H)(i) evaluated at every strike, so an iron condor is
  charged its wider side. Pluggable: cash, Reg-T, Robinhood.
- **Poor man's covered call works with no shares** — $0 requirement, short calls
  re-written repeatedly against one long leg.
- **Physical settlement.** An assigned short call delivers shares and establishes
  a short stock position if none are held. Exercise-by-exception at $0.01.
- **Spread cost is the only randomness**, drawn from a counter-based RNG keyed on
  `(seed, scenario, order, instrument, timestamp, leg)` — order-independent,
  reproducible from the key, common random numbers across strategies.
- **Streaming reader** with partition pruning and column projection over the
  pipeline's real 49-column schema.
- **Run manifest** with seed, models, policies, a hash of the files read and a
  hash of the config.
- **396 tests**, including an independent Python reference engine that agrees to
  the microdollar, and Monte Carlo property tests covering all eight required
  properties.

### Measured cost

| Quantity | Measured |
|---|---|
| Throughput, 1 path | **~59,000 option rows/s** |
| Throughput, 50 paths | **~38,000 rows/s** |
| Marginal cost per extra path | **~1.8e-7 s/row** |
| Peak RSS, 1 → 50 paths | **128 MB, flat** (lockstep design confirmed) |
| Extrapolated: one SPY ticker-year, 1000 paths | **~8 h** (0.7 h base + 7.4 h marginal, assuming ~1,500 quoted contracts × 390 min × 252 d) |

Memory being flat in path count is the payoff of advancing all paths over one
data pass; the cost is that runtime is linear in paths and CPU-bound in Python.

### Fixed during this audit

**Spread pairing across mismatched deliverables.** `pair_legs` balanced contract
count, but spread treatment requires equal aggregate underlying value
(FINRA 4210(f)(2)(A)(xxxii)(d)). A long delivering 100 shares paired against a
short delivering 400, leaving 300 shares of naked exposure and charging **$0**.
The failure was invisible to max-loss netting rather than merely underpriced:
with unequal deliverables the payoff slopes do not cancel, the loss is unbounded
as the underlying rises, and evaluating at the two strikes returns a net *gain*.
This is exactly the state a split produces, so it was reachable. A pair now
requires equal deliverables; a mismatch falls through to the naked charge
($6,000 in the audit case, refused under Robinhood). Regression tests added.

---

## 2. What "industry standard" requires

1. **Every declared feature works or refuses.** A silently-ignored order type is
   worse than an unsupported one.
2. **Execution cost is calibrated or labelled.** An uncalibrated spread makes
   every P&L number an estimate of unknown width.
3. **Broker fidelity validated against reality**, not only against published
   rules.
4. **A strategy can only do what a real account can.**
5. **Reproducible and auditable.** Any number traceable to code, config, data and
   seed.
6. **Independently cross-checked.** Deterministic output matching an
   implementation that shares no assumptions.
7. **Bounded cost.** Known runtime and memory, enforced in CI.

---

## 3. Gap register

Severity is risk to trusting a P&L number.

### BLOCKER

| # | Gap | Evidence | Consequence |
|---|---|---|---|
| B1 | **Stop and stop-limit orders silently execute as market orders.** `OrderType::Stop` and `StopLimit` are in the enum and bound to Python; the only price check in `execute_group` tests `OrderType::Limit` | `engine/engine.h:432` | Verified: a BUY STOP at $999 against a $5.00 market **fills immediately at $5.00**, no rejection. Any strategy using stops produces silently wrong results. Must reject explicitly. |
| B2 | **Spread cost is uncalibrated.** The pipeline supplies no quotes, so the default `conditional_lognormal` parameters are illustrative | `engine/spread.h` defaults; `optionsdata` schema has no bid/ask | Confidence intervals describe an assumed distribution, not execution. The report says so, but a number that reads as ±$3 could be ±$300. |
| B3 | **`on_fill` and `on_corporate_action` are declared in the Strategy API and never invoked** | `optionsbacktester/runner.py` — 0 call sites for either | A strategy cannot react to its own fills or to an adjustment. Any strategy written against the documented API silently never runs that logic. |
| B4 | **`EquityKind::Equity` is bindable and silently ignored.** The execution path never branches on `Order::kind`; every order is priced with the contract's quote multiplier and booked as an option | `engine/engine.h` `execute_group` — no reference to `o.kind` | Verified: an EQUITY-flagged order for **1 share at $100 moved $10,000 and created an option position**, not a share position. A strategy attempting to trade stock gets a silent 100×-levered option trade. Worse than the feature being absent. |

### MAJOR

| # | Gap | Evidence | Consequence |
|---|---|---|---|
| M1 | **Four of nine risk limits are declared but never enforced**: `max_contracts_per_underlying`, `max_notional_per_underlying`, `max_loss_per_trade`, `max_abs_delta` | zero references in `engine/engine.h` | Setting them has no effect. A risk-constrained backtest silently runs unconstrained. |
| M2 | **`max_daily_loss` is actually a max *total* loss.** `day_start_equity_` is assigned once per scenario and never per session | `engine/engine.h:155` vs `:714` | The limit triggers on cumulative loss from initial cash, so a strategy that recovers is permanently halted. |
| M3 | **No equity trading capability** (the coverage consequence of B4). Shares arrive only via exercise or assignment | — | No covered call from scratch, no protective put on existing shares, no collar, no synthetic, no delta hedging. `BACKTESTER_GOALS.md` names all of these as required. |
| M4 | **A corporate-action basis transfer loses money to integer division.** `Money{basis.micros / child_qty}` truncates | `engine/engine.h` `transfer_position` | Verified: a 1→3 conversion on a $1,000.000001 basis loses **1 microdollar**. Tiny, but it violates "transfer total economic cost basis" and produces artificial P&L at an adjustment. |
| M5 | **No early assignment.** Only expiration settles | `process_expirations`; `ConservativeEarlyAssignment` is an enum value with no distinct behavior | Short calls are never assigned before a dividend even when it is rational. Systematically flatters every short-call strategy, PMCC included. |
| M6 | **Cash-settled index options are not representable.** Settlement is always physical; there is no European exercise path and no AM settlement value | `settle_physically` | SPX/VIX/XSP/RUT/NDX cannot be backtested. `SettlementRule::CashSettlement` is declared and unused. |
| M7 | **Margin is not broker-validated.** It follows published FINRA/Cboe/Robinhood rules but has never been reconciled against a broker statement | — | Robinhood's actual house requirements range 25–100% and are model-driven and undisclosed. Requirements are a defensible approximation, not a match. |
| M8 | **No monotonic-time assertion.** `begin_bar` accepts any timestamp | `engine/engine.h` `begin_bar` | Out-of-order or duplicate bars are processed silently, producing nonsense with no error. |
| M9 | **Robinhood's expiration-day closeout is not modeled.** The real broker closes at-risk expiring positions from 3:30 PM ET | — | The engine holds to expiry and settles, which the real account often would not. The "at-risk" band is undisclosed, so this is a parameter, not a rule. |
| M10 | **Money arithmetic is unguarded.** `Money{a.micros * b}` appears throughout with no overflow check | `types.h`, `engine.h`, `margin.h` | Headroom is large (int64 max ≈ $9.2e12; the worst realistic product reaches ~1e18 of 9.2e18), so this is a latent trap rather than a live bug — but it would wrap silently. |
| M11 | **No CI**, and `requirements.txt` (62 lines) has diverged from `pyproject.toml` (10 deps); two `.so` files are tracked in git | repo root | Nothing prevents a regression. setuptools does not track header dependencies, so a header-only change silently ships a stale extension — already a live footgun, documented in CLAUDE.md. |
| M12 | **Bracket and OCO orders, and exercise and roll as first-class primitives, are absent.** Rolls are expressible as an atomic group; exercise is not expressible at all | `order.h` | `AssignmentPolicy::ExplicitExerciseOnly` is selectable but a strategy has no way to submit an exercise, so choosing it makes the engine never settle anything. |

### MINOR

- `LedgerEntryKind::EXPIRATION_SETTLEMENT` is declared and never posted (a
  worthless expiration moves no cash, so there is no entry at all).
- On exercise, premium is realized as a full loss and shares are booked at
  strike, rather than rolling premium into share basis. Total P&L is identical;
  the realized/unrealized split differs from broker convention.
- Iron-condor pairings report a per-pair `requirement` that is no longer what is
  charged, since the charge comes from joint netting. Reporting-only.
- `PortfolioApprox` margin is an alias for Reg-T.
- `build_bars` and `build_analytics` loop per row in Python — the likely hot path.
- `computation/` and `dashboard/` are the previous implementation, still present
  and still reading yfinance daily stock bars. Left in place deliberately;
  removing them is a decision, not a cleanup.
- Six root-level scripts import a module that never existed in this repo; pytest
  collection is scoped to `tests/` rather than deleting them.

---

## 4. Roadmap

### Phase 0 — Stop the silent wrongness  *(days)*

The cheapest and highest-value work: nothing here needs new data.

- **B1.** Reject `Stop` and `StopLimit` with a named reason. A trigger cannot be
  honestly simulated from OHLC without inventing an intrabar path; refusing is
  correct, executing at market is not.
- **B3.** Invoke `on_fill` after each fill and `on_corporate_action` on each
  applied transition.
- **M1.** Enforce the four dead risk limits, or delete them from the config so
  they cannot be set.
- **M2.** Reset `day_start_equity_` at each session boundary, and rename it if it
  is meant to be a total-loss limit.
- **M4.** Distribute the basis remainder across child contracts so the transfer
  is exact; assert basis conservation in a test.
- **M8.** Assert monotonic time in `begin_bar`.
- **B4.** Reject an `EquityKind::Equity` order outright until Phase 2 implements
  it. A 100x-levered option trade in place of a share purchase is the worst
  possible failure mode: it fills, it reconciles, and it is wrong.
- **M12.** Either implement exercise orders or make `ExplicitExerciseOnly`
  unselectable.

**Done when:** no declared feature silently does the wrong thing. Every enum
value and config field either works or is rejected. A test exists for each. The
guiding rule: an unsupported feature must fail loudly, never approximate.

### Phase 1 — Make execution cost defensible  *(weeks, gated on the pipeline)*

This is the difference between a number and an estimate of unknown width.

- Consume the calibration artifact from pipeline Phase 1 via
  `--spread-calibration`.
- Where a row carries real quotes, use them and mark the fill `measured`; fall
  back to the model and mark it `modeled`. Report the mix.
- Run the convergence check the spec requires (100/500/1000/5000) and record the
  standard error alongside every published result.
- Add a sensitivity report: P&L under a 2× and 0.5× spread assumption, so a
  reader sees how much of the result the assumption owns.

**Done when:** every reported P&L states what fraction of fills used measured
quotes, and a spread-sensitivity band is published alongside the point estimate.

### Phase 2 — Complete the instrument and order model  *(weeks)*

- **M3 / equity orders.** Add `EquityKind::Equity` to the execution path with
  Reg-T stock margin. Unlocks covered calls, protective puts, collars,
  synthetics and delta hedging — five structures the spec names.
- **M6 / cash settlement.** European exercise, cash settlement against a
  settlement value, AM series that stop trading the prior day.
- **M5 / early assignment.** Implement `ConservativeEarlyAssignment`: assign a
  short call when the dividend exceeds remaining extrinsic value on the last day
  before the ex-date. Requires the pipeline to supply declared dividends with
  declaration dates, which it already does.
- **M12 / order types.** Bracket and OCO, and exercise as a submittable order.
- **M9 / broker closeout.** Model expiration-day closeout as an explicit,
  documented parameter.

**Done when:** every structure named in `BACKTESTER_GOALS.md` is expressible and
exact-ledger tested, and index options run or are refused with a reason.

### Phase 3 — Earn the trust  *(weeks)*

- **M7 / broker validation.** Reconcile against real statements: fills, fees,
  margin requirements, assignment timing. Until then, label margin an
  approximation in the report itself, not only in the README.
- **Golden end-to-end replay.** Immutable fixture bundle → pipeline → backtester
  → byte-compared expected ledger and expected Monte Carlo summary. Partly built
  (`tests/fixtures.py`); needs the pipeline half and a frozen expected output.
- **Mutation testing.** Prove the suite constrains behavior. 396 passing tests is
  an input, not evidence; a mutation score is evidence. Target ≥ 80% on
  `engine/`.
- **Property-based tests** over accounting: random open/close/add/cross-zero
  sequences must always conserve the ledger identity and never produce a basis
  sign error.
- **Cross-check breadth.** The reference engine covers ten scenarios; extend it
  to margin and to multi-leg settlement.

**Done when:** a mutation score is published, a golden replay passes, and any
number in a report is traceable to code, config, data and seed.

### Phase 4 — Cost and scale  *(weeks)*

- **Profile first.** The suspicion is `build_bars`/`build_analytics` per-row
  Python looping, not the C++ core. Confirm before optimizing.
- Move snapshot construction to Arrow-native or vectorized conversion.
- Consider evaluating paths in parallel across processes; state is small and
  independent, and the counter-based RNG makes it order-safe by construction.
- **M10.** Add overflow guards or a debug-mode assertion on money products.
- **M11.** CI running build, tests, ruff and a memory/throughput ceiling.
  Reconcile dependencies; stop tracking `.so` files.
- Target: **one SPY ticker-year at 1000 paths in under 1 h**, against ~8 h today.

**Done when:** CI enforces a throughput and memory envelope, and the target
runtime is met on a real ticker-year.

---

## 5. Accepted limitations

To be restated in the README rather than quietly fixed:

- **Execution cost is modeled, not reconstructed.** Under a free-data constraint
  there is no historical option NBBO. The interval is sensitivity analysis.
- **Stop orders will not be simulated.** OHLC cannot support a trigger without
  inventing an intrabar path.
- **No market impact, queue position, or partial fills.** Bar data cannot support
  them, and inventing them would flatter results.
- **Margin is rule-derived, not broker-validated**, until Phase 3.
- **Lineage is fail-closed.** While the pipeline produces no OCC-confirmed
  transitions, a position held through an adjustment is refused. That is correct;
  guessing a conversion would corrupt everything downstream.

---

## 6. Decisions needed from a human

1. **Does the free Alpaca tier expose option quotes?** The single highest-leverage
   unknown across both repos. It determines whether Phase 1 produces calibrated
   execution cost or a permanent caveat.
2. **Is equity trading in scope?** Phase 2's largest item. Without it, five named
   structures remain unreachable — but if the intended strategies are
   options-only, it can be dropped.
3. **Are index options in scope?** Substantial work in both repos.
4. **Keep or delete `computation/` and `dashboard/`?** They are the previous
   implementation, superseded and partly broken. The dashboard is the only UI
   that exists; a replacement over the new engine is a separate project.
5. **What accuracy is required?** "Directionally right for research" and "trusted
   for capital allocation" are different programmes. The second requires Phase 3
   in full, including broker reconciliation.

---

## 7. Definition of done

This backtester is industry standard when all hold:

- Every declared feature works or refuses explicitly; nothing silently
  approximates.
- Bid/ask cost is the only Monte Carlo component, and it is calibrated or
  reported as sensitivity analysis with a published band.
- Quantity, deliverables, exercise and settlement are correct, including across a
  source-confirmed adjustment.
- A strategy can do only what a real account can.
- Every structure named in `BACKTESTER_GOALS.md` is expressible and exact-ledger
  tested.
- Deterministic output matches an independent implementation exactly.
- Every run records code, data, config, seed and output hashes.
- A golden end-to-end replay passes byte for byte.
- A mutation score demonstrates the suite constrains behavior.
- CI enforces correctness, performance and memory limits.
