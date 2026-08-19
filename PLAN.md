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

One phase here cannot start until the pipeline delivers.

| Needs from `../optionsdata/PLAN.md` | Blocks |
|---|---|
| B1 — quote sampler and spread calibration artifact | All of Phase 1. **The only remaining hard dependency.** |
| B2 — OCC memo ingestion, confirmed lineage | Carrying a position THROUGH an adjustment. The engine's fail-closed refusal is correct meanwhile, so this widens coverage rather than fixing wrongness |
| M2 — index exercise style and settlement columns | Reaching the engine's cash-settlement path with real data. The engine side is implemented and tested against synthetic index contracts |

Everything independent of the pipeline is done. The engine reads and honours every
column the pipeline currently writes, including three reference frames it
previously loaded and discarded: `option_contract_version` (point-in-time terms),
`corporate_actions` (dividends), and the stock frame's open, high, low, vwap and
trade count.

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
- **Equity trading.** Shares are buyable and shortable by symbol, on their own
  penny grid and their own per-share fee schedule, so a covered call, collar or
  protective put can be opened rather than only inherited from a settlement.
- **Cash-settled index options.** European exercise, an official settlement value
  (SET/VRO) preferred over the last observed spot, and a last-trade instant for
  AM-settled series that stop trading before expiration.
- **Dividends.** Paid on share positions at the PAY date after accruing at
  ex-date, gated on the declaration date, owed on short shares. Dividend-driven
  early assignment of short calls, decided on the prior session's close.
- **Point-in-time contract terms.** Fills are gated on when terms took effect AND
  when they became knowable, and opening a position on an adjusted contract whose
  terms are not established as point-in-time is refused.
- **Bounded mark staleness.** A carried-forward mark past its age limit falls back
  to intrinsic against a fresh spot, and the oldest mark behind any valuation is
  reported.
- **639 tests**, including an independent Python reference engine that agrees to
  the microdollar, and Monte Carlo property tests covering all eight required
  properties.

### Measured cost

| Quantity | Measured |
|---|---|
| Throughput, 1 path | **~59,000 option rows/s** |
| Throughput, 50 paths | **~38,000 rows/s** |
| Marginal cost per extra path | **~1.8e-7 s/row** |
| Peak RSS, 1 → 50 paths | **~118 MiB, flat in path count** |
| Extrapolated: one SPY ticker-year, 1000 paths | **~8 h** (0.7 h base + 7.4 h marginal, assuming ~1,500 quoted contracts × 390 min × 252 d) |

Memory being flat in path count is the payoff of advancing all paths over one
data pass; the cost is that runtime is linear in paths and CPU-bound in Python.

An earlier version of this table claimed flat memory on the strength of a fixture
holding 126 contracts, where the per-engine contract registry is too small to see.
It was not flat: each engine held its own registry by value and the runner re-set
it from the full cumulative set once per day per path, so both memory and per-day
work scaled with paths × contracts — measured at 180 MiB for 100 engines over
8,000 contracts, extrapolating to ~1.8 GiB at 1,000 paths, and a real ticker-year
carries tens of thousands of versions. One shared registry now serves every path,
which brings that term to zero and makes the claim true rather than an artifact of
the fixture. `scripts/check_budgets.py` enforces the figures in CI.

### Fixed during the audit that produced this plan

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

Severity is risk to trusting a P&L number. Everything previously listed as a
BLOCKER is closed; section 3a records what closed and how it was verified, because
a plan that only ever grows is not a plan.

### MAJOR — still open

| # | Gap | Evidence | Consequence |
|---|---|---|---|
| M-a | **Spread cost is uncalibrated.** The pipeline supplies no quotes, so the default `conditional_lognormal` parameters are illustrative | `engine/spread.h` defaults; the pipeline's schema has no bid/ask | Confidence intervals describe an assumed distribution, not execution. The report says so and the `--spread-variance-scale` knob exists to size the exposure, but a number that reads as ±$3 could be ±$300. **The single most important open item**, and it is gated on the pipeline's B1. |
| M-b | **Margin is not broker-validated.** It follows published FINRA/Cboe/Robinhood rules and has been checked against Robinhood's own help centre and fee schedule, but never against a broker statement | — | Robinhood's house requirements range 25–100% and are model-driven and undisclosed. Requirements are a defensible approximation of the published rules, not a match to an account. |
| M-c | **Robinhood's expiration-day closeout is not modelled.** The real broker closes at-risk expiring positions from 3:30 PM ET | — | The engine holds to expiry and settles, which the real account often would not. The "at-risk" band is undisclosed, so this is a parameter to expose, not a rule to implement. |
| M-d | **Bracket and OCO orders are absent, and exercise is not submittable.** Rolls are expressible as an atomic group; a stop-loss bracket is not | `engine/order.h` | `AssignmentPolicy::ExplicitExerciseOnly` is selectable but a strategy has no way to submit an exercise, so choosing it makes the engine never settle anything. Either implement it or make the policy unselectable. |
| M-e | **`config_sha256` does not cover everything that changes a result** — spread calibration, risk limits, the universe filter, strategy parameters, and the engine source itself are all outside it | `optionsbacktester/runner.py` | Two runs can report the same config hash and different numbers, which is worse than no hash. |
| M-f | **`build_bars` and `build_analytics` iterate per row in Python.** They are the hot path | `optionsbacktester/contracts.py` | Throughput is CPU-bound in Python at ~26k rows/s. A ticker-year at 1,000 paths is hours. Vectorising the frame-to-struct conversion is the largest available win. |
| M-g | **`equity_curve_`, `fills_`, `rejections_` and `trades_` accumulate without bound** | `engine/engine.h` | Memory is linear in bars × paths for the record vectors even though the registry is now shared. A long run at high path count will grow steadily. |
| M-h | **No golden end-to-end replay.** Nothing pins a full run's numbers against a stored expectation | `tests/` | A refactor that changes every P&L by a cent passes. |
| M-i | **Mutation score is not measured.** Individual fixes in this repo are mutation-verified one at a time; the suite as a whole is not scored | — | "639 tests" is a count, not evidence. A target of ≥80% killed on the accounting and settlement paths would be. |

### MINOR — still open

- `LedgerEntryKind::EXPIRATION_SETTLEMENT` is declared and never posted (a
  worthless expiration moves no cash, so there is no entry at all).
- On exercise, premium is realized as a full loss and shares are booked at strike,
  rather than rolling premium into share basis. Total P&L is identical; the
  realized/unrealized split differs from broker convention.
- Iron-condor pairings report a per-pair `requirement` that is no longer what is
  charged, since the charge comes from joint netting. Reporting-only.
- `ContractRegistry::resolve` is unused and unbound — the by-symbol index it
  maintains costs memory for nothing.
- `pricing_strike` is loaded from the lake and carried on the contract but enters
  no calculation. Either it belongs in the payoff or it should not be loaded.
- `computation/` and `dashboard/` are the previous implementation, still present
  and still reading yfinance daily stock bars. Now declared as an optional
  `legacy-dashboard` extra rather than a required dependency. Removing them is a
  decision, not a cleanup.
- Six root-level scripts import a module that never existed in this repo; pytest
  collection is scoped to `tests/` rather than deleting them.

---

## 3a. Closed, with the evidence

Each of these was verified by reverting the fix and confirming the tests fail —
mutation testing on a single change rather than a suite-wide score.

| Was | What was actually wrong | Now |
|---|---|---|
| B1 | A BUY STOP at $999 against a $5.00 market filled immediately at $5.00 with no rejection | `Stop`/`StopLimit` refused with a named reason; an intrabar path cannot be honestly inferred from OHLC |
| B3 | `on_fill` and `on_corporate_action` had zero call sites | Both invoked; adjustments delivered **before** the strategy is asked for orders, so it cannot act on a version the engine has superseded. Nine tests through the real runner |
| B4 / M3 | An EQUITY-flagged order for 1 share at $100 moved $10,000 and created an option position; later refused outright | Implemented: shares by symbol, multiplier 1, own penny grid, own per-share fee schedule. 26 tests; restoring the 100× multiplier fails 17 |
| D-2 | A 50-share deliverable at strike 100 with spot 110 paid $5,000 for $5,500 of stock, inventing $500 on a worthless contract | Aggregate exercise price is strike × **quote multiplier**, per the pipeline's `max(A·S_T + C − K·M, 0)`. Independently confirmed against OCC Rule 2803(d)(1)(iii), which freezes the unit of trading for exactly this purpose |
| D-3 | A long put whose underlying had no observed price settled at a fabricated spot: **+$9,500** and a 100-share short, with no rejection | Quarantined with a named reason and the path flagged truncated |
| D-6 | `valid_from` was hardcoded to the epoch, so a contract whose reverse split took effect on day 10 filled a buy order on day 1 at post-split terms | Terms gated on effect **and** on knowability; `terms_provenance` honoured; unknown provenance on an adjusted contract fails closed |
| M5 | `ConservativeEarlyAssignment` was byte-identical to `AutomaticITMExercise`; no early-assignment code existed | Dividend-driven assignment on the standard condition, decided on the prior session's close because the underlying opens ex-date lower |
| M6 / D-4 | Every contract settled by delivering shares, so an SPX option booked a position in an index nobody can deliver | Cash settlement, European exercise, official settlement value preferred, last-trade instant for AM-settled series |
| M8 | `begin_bar` accepted any timestamp; a repeat let an order fill at the instant it was submitted | Non-advancing timestamps refused, with a message that says why a repeat matters |
| M10 | 43 unguarded money products; the worst realistic one reaches ~1e18 of 9.2e18, and overflow wraps **negative** | All products checked. Doing it mechanically broke six multiply-then-divide sites and the suite caught all six |
| M11 | No CI; `requirements.txt` pinned 62 packages against 2 real ones; two arm64 `.so` files tracked | CI builds, tests and enforces throughput and memory ceilings. Dependencies reconciled, binaries untracked |
| E-4 | `PortfolioApprox` returned a plain Reg-T model while the manifest recorded `portfolio_approx` — a reproducibility artifact naming a methodology the run had not applied | Deleted. Robinhood does not offer portfolio margin at all; its long-option maintenance requirement of 100% is incompatible with it |
| E-7 | The audit claimed this model over-refuses, permitting short strangles at Level 3 | **The audit was wrong.** Robinhood publishes two levels, neither permits an uncovered short call, and the Level 3 menu contains no short strangle or straddle. Tests now name the source so it does not get "fixed" back |
| H-1 | Each engine held its own contract registry; 100 engines over 8,000 contracts cost 180 MiB, extrapolating to ~1.8 GiB at 1,000 paths | One shared registry. 0.0 MiB, and the per-day re-set is gone |
| J-3 | A contract that stopped printing kept its last mark forever, valuing the book and setting margin off a price of any age | Bounded; falls back to intrinsic against a fresh spot; the oldest mark behind a valuation is reported |
| — | A round trip recorded only its CLOSING leg's fees and spread, so per-trade cost read at roughly half what the path paid ($5.58 against $10.02) | Entry costs release proportionally with the basis. Now exact on both |
| — | Dividends did not exist in the engine at all, so a covered call understated its return by the entire yield | Accrued at ex-date, paid at pay date, gated on declaration, owed on short shares |
| — | The CAT fee was charged at $0.0003/contract after regulators stopped charging it on 2025-12-01 | Zero by default, still configurable for an earlier window |
| — | Robinhood short stock was charged the Reg-T 150% rather than the published 130% | Corrected against Robinhood's own worked example |

---

## 4. Roadmap

Phase 0 is complete. What remains is ordered so each phase makes the next
verifiable.

### Phase 1 — Make execution cost defensible  *(gated on the pipeline)*

The only remaining item that changes what a number MEANS rather than what it is.

- **M-a.** Consume the pipeline's spread-calibration artifact once it exists: fit
  `log_base`, `log_sigma` and the conditional betas to sampled quotes, record the
  fit's provenance in the manifest, and report the residual.
- Keep `--spread-variance-scale` as the sensitivity knob, and make the report
  state the calibration source rather than only warning that there is none.

**Done when:** the reported interval is traceable to observed quotes, and a run
that used defaults says so in its manifest.

### Phase 2 — Finish the order model  *(weeks)*

- **M-d.** Bracket and OCO groups; exercise as a submittable order so
  `ExplicitExerciseOnly` means something.
- **M-c.** Expose Robinhood's expiration-day closeout as a configurable band
  rather than pretending the account holds to expiry.

**Done when:** every selectable policy and order type has behaviour a test pins.

### Phase 3 — Earn the trust  *(weeks)*

- **M-e.** Extend `config_sha256` to cover spread calibration, risk limits, the
  universe filter, strategy parameters and a hash of the engine source.
- **M-h.** A golden end-to-end replay: a fixed lake, a fixed seed, stored
  expected numbers.
- **M-i.** Measure the mutation score on the accounting, margin and settlement
  paths. Target ≥80% killed. Publish the figure rather than the test count.
- **M-b.** Reconcile one real account statement against the model, and record
  every divergence as either a fix or a stated limitation.

**Done when:** the repo's own claims are all measured rather than asserted.

### Phase 4 — Cost and scale  *(weeks)*

- **M-f.** Vectorise `build_bars` and `build_analytics`. This is the largest
  single throughput win available.
- **M-g.** Bound or stream the per-path record vectors.
- Raise the CI budgets as they improve, so the ceiling always tracks reality.

**Done when:** a ticker-year at 1,000 paths is measured, bounded, and enforced.

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
   unknown across both repos, and now the only thing standing between this engine
   and a calibrated execution cost. It determines whether the reported interval is
   a measurement or a permanent caveat.
2. **Keep or delete `computation/` and `dashboard/`?** They are the previous
   implementation, superseded and partly broken, now behind an optional extra. The
   dashboard is the only UI that exists; a replacement over the new engine is a
   separate project.
3. **What accuracy is required?** "Directionally right for research" and "trusted
   for capital allocation" are different programmes. The second requires Phase 3
   in full, including reconciliation against a real account statement.
4. **Is portfolio margin wanted at all?** It was removed rather than implemented,
   because Robinhood does not offer it and the previous stand-in mislabelled
   Reg-T. A genuine TIMS-style scan is a real feature, but it models a broker this
   engine is not configured for.

Two questions that were open are now answered by the work rather than by a
decision: equity trading is in scope and implemented, and index options are
representable engine-side and now wait only on pipeline coverage.

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
