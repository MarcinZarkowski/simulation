# Backtester Goals and Implementation Specification

## Purpose

Build a Python research/orchestration program with a readable, efficient C++
core for chronological event processing, order fills, accounting, corporate
actions, option settlement, and results. It consumes the OptionsBackfill data
lake after the pipeline has point-in-time, source-confirmed contract lineage.

Core rule:

> Market data, contract terms, corporate actions, and settlement are
> deterministic. Only synthetic bid/ask execution cost is Monte Carlo
> simulated. Do not randomize price paths, assignment, fill probability, market
> impact, or corporate actions.

Identical data manifests, strategy version, engine version, configuration, and
Monte Carlo seed must produce identical results.

## Required Pipeline Inputs

The completed pipeline must provide:

```text
options_enriched.parquet
options_adjusted.parquet
stock.parquet
corporate_actions.parquet
corporate_action_events.parquet
option_contract_version.parquet
option_deliverable_component.parquet
option_lineage_event.parquet
occ_adjustment_memo_raw.parquet / source documents
```

Required additions to the pipeline:

- `option_contract_version`: one contract state valid over a defined interval.
- `option_lineage_event`: OCC-confirmed parent-to-child transition, quantity
  conversion, effective timestamp, settlement rule, and source memo.
- Immutable OCC memo storage: URL, retrieved timestamp, document hash, parser
  version, memo publication time, and supersession relation.
- `source_available_at` for every event, so strategies cannot know a future
  announcement.
- Provider contract ID, underlying asset ID, style, raw payload hash, provider
  retrieval time, and first/last seen timestamps.

The backtester must reject a position held through an adjustment where no
OCC-confirmed lineage event is available.

## Architecture

```text
Python CLI / experiment runner
  - validate manifests, schemas, quality, and date ranges
  - configure strategy, capital, risk, and spread Monte Carlo
  - stream Parquet / Arrow batches
  - provide batched strategy callbacks
  - write results and reports
                |
                v
C++ event engine
  - deterministic event clock
  - positions, cash, ledger, margin, and risk
  - orders, complex groups, fills, and fees
  - option exercise, assignment, expiration, settlement
  - OCC-confirmed contract transformations
  - bid/ask spread Monte Carlo scenarios
  - path-level and aggregate metrics
```

Python owns discovery, data access, research, and reporting. C++ owns
authoritative accounting and state transitions. C++ receives compact typed
batches; it does not discover cloud files itself.

## Streaming and Event Clock

Use PyArrow Dataset or Polars lazy scans with partition pruning, column
projection, predicate pushdown, fixed-size batches, and chronological file
ordering. Support universe filters for symbols, expiry/DTE, delta, strike,
moneyness, calls/puts, volume, and analytics quality.

Only keep in memory:

```text
active positions
pending orders
current relevant market snapshot
current contract versions and deliverables
current corporate-action events
running portfolio/risk state
current Monte Carlo scenario state
```

Python/C++ interface:

```python
engine.process_corporate_action_batch(events)
engine.process_market_batch(batch)
orders = strategy.on_market_snapshot(snapshot, context)
engine.submit_orders(orders)
```

Do not call Python once per option row. Call it once per timestamp or controlled
timestamp batch.

Required ordering:

```text
1. Apply OCC events effective before the session.
2. Resolve contract versions valid at the current time.
3. Process market bars at timestamp T.
4. Expose only information available at T.
5. Strategy submits orders at T.
6. Fill with the configured execution-timing rule.
7. Update ledger, margin, and risk.
8. Process expiration/exercise/assignment/settlement events.
9. Value and report at end of session.
```

Default strict timing:

```text
signal at bar close -> order fills at next eligible bar open
```

This avoids same-bar lookahead.

## C++ Core Model

Favor explicit, well-commented structs. Use integer money units for premiums,
cash, fees, cost basis, realized P&L, and settlements. IV and Greeks may be
`double` because they are analytics, not ledger values.

```cpp
struct Timestamp { int64_t epoch_nanoseconds; };
struct Money { int64_t microdollars; };
struct InstrumentId { uint64_t value; };
struct ContractVersionId { uint64_t value; };
struct PositionId { uint64_t value; };

struct OptionContractVersion {
    ContractVersionId id;
    InstrumentId instrument_id;
    std::string symbol;
    std::string underlying_symbol;
    Timestamp valid_from, valid_to, expiration;
    bool is_call;
    bool is_american;
    int64_t strike_microdollars;
    int64_t quote_multiplier;
    int64_t deliverable_equity_microshares;
    Money deliverable_cash;
    bool is_adjusted;
    bool tradable_for_new_positions;
    bool analytics_supported;
};

struct MarketBar {
    Timestamp timestamp;
    InstrumentId instrument_id;
    Money open, high, low, close, vwap;
    int64_t volume, trade_count;
    bool stale;
    bool analytics_valid;
};

struct OptionAnalytics {
    Timestamp timestamp;
    ContractVersionId contract_version_id;
    double implied_volatility;
    double delta, gamma, theta, vega, rho;
    Money theoretical_contract_value;
    bool valid;
};

struct Position {
    PositionId id;
    ContractVersionId contract_version_id;
    int64_t quantity;
    Money cost_basis, realized_pnl;
    Timestamp opened_at, last_updated_at;
};
```

```cpp
struct Order {
    uint64_t order_id;
    Timestamp submitted_at;
    InstrumentId instrument_id;
    int64_t quantity;
    OrderSide side;
    OrderType type;
    TimeInForce time_in_force;
    std::optional<Money> limit_price;
    std::optional<Money> stop_price;
    uint64_t strategy_order_group_id;
};

struct CorporateActionTransition {
    uint64_t lineage_event_id;
    Timestamp effective_at;
    ContractVersionId parent_version_id;
    ContractVersionId child_version_id;
    int64_t parent_contracts, child_contracts;
    AdjustmentType type;
    SettlementRule settlement_rule;
    bool occ_confirmed;
};

struct BacktestConfig {
    Timestamp start, end;
    Money initial_cash;
    ExecutionTimingPolicy execution_timing;
    AssignmentPolicy assignment_policy;
    uint32_t spread_mc_paths;
    uint64_t spread_mc_seed;
    SpreadModelConfig spread_model;
    bool require_occ_confirmed_lineage;
    bool reject_fallback_analytics;
    bool reject_stale_bars;
};
```

## Contract Identity and Lineage

Never use the OCC symbol as the permanent position identity.

```text
economic series -> contract version -> symbol valid for that version
```

Positions reference `contract_version_id`. A symbol is an attribute of a
version. At an OCC-confirmed action:

```text
close parent version with no artificial P&L
-> create child position(s)
-> convert quantity exactly as instructed
-> transfer total economic cost basis
-> update multiplier, deliverables, and settlement rules
```

Required events include splits, reverse splits, root changes, one-to-many
replacement contracts, same-symbol deliverable changes, stock-and-cash mergers,
spin-offs/baskets, cash settlement, accelerated expiration, and chained
adjustments. Candidate/inferred mappings are never actionable.

## Valuation Rules

Historical market mark:

```text
market mark = pipeline valuation_price
```

Model mark:

```text
model mark = theoretical_value
```

Use analytics only when `iv_failed == false`, `iv_is_model_fallback == false`,
`is_stale == false`, and the configured validation status is accepted.

Standard valuation:

```text
market contract value = valuation_price * quote_multiplier * contracts
model contract value  = theoretical_value * contracts
```

For supported adjusted contracts, use pipeline `quote_multiplier`, deliverable
amounts, pricing strike, theoretical value, and Greeks directly. They are
already scaled to the real contract; never scale them twice.

Unsupported adjusted contracts may be retained/marked/exited only under an
explicit policy. No new entry, IV/Greeks, or assumed 100-share exercise is
allowed; settlement requires a confirmed OCC rule.

## Orders and Complex Strategies

Support market, limit, stop, stop-limit, bracket, OCO, roll, exercise, and
multi-leg orders. Multi-leg order groups must be atomic: all legs fill together
or no legs fill.

Required structures include verticals, calendars, diagonals, butterflies, iron
condors, straddles, strangles, ratio spreads, covered calls, protective puts,
collars, synthetics, and custom baskets.

Strict bar-data rules:

- Market orders fill at next eligible mark plus/minus simulated half-spread.
- Limit orders fill only if the configured execution price satisfies the limit.
- Do not infer fills solely because OHLC high/low crossed a limit; intrabar path
  and queue priority are unknown.
- In strict mode, stop orders are unsupported or become deterministic next-bar
  orders after a visible trigger.
- Do not invent partial fills, queue position, fill probability, or market
  impact without additional data.

## Bid/Ask Spread Monte Carlo

This is the only stochastic component.

```bash
python -m optionsbacktester run \
  --strategy strategies/iron_condor.py \
  --tickers SPY \
  --start 2024-01-01 --end 2026-01-01 \
  --spread-mc-paths 1000 \
  --spread-mc-seed 42 \
  --spread-model conditional_lognormal \
  --spread-calibration configs/spread_model.json
```

Required arguments:

```text
--spread-mc-paths
--spread-mc-seed
--spread-model
--spread-calibration
--execution-timing
--report-confidence-level
```

Recommended final-research default: 1,000 paths, fixed recorded seed, 95%
interval, and next-bar-open execution. Require convergence runs at 100, 500,
1,000, and 5,000 paths.

Only draw nonnegative half-spread:

```text
buy fill  = mark + half_spread
sell fill = mark - half_spread
```

Spread models may be constant, piecewise, lognormal, empirical sampled, or
conditional regression/quantile distributions. Conditional inputs may include
premium, volume, trade count, DTE, moneyness, IV, underlying price, time of day,
and contract type. Without actual quote calibration, results are sensitivity
analysis—not exact execution reconstruction.

Derive draws from global seed, scenario ID, order ID, instrument ID, and
timestamp. Use common random numbers when comparing strategies.

Persist path metrics:

```text
scenario_id, net_pnl, realized_pnl, unrealized_pnl, fees, spread_cost,
max_drawdown, return, margin_usage, trade_count, assignment_count, exercise_count
```

Report mean/median P&L, standard deviation, percentiles, probability of profit,
margin-breach probability, drawdown, worst/best paths, spread-cost attribution,
and Monte Carlo standard error. Clearly separate deterministic market-data P&L
from stochastic spread cost.

## Settlement, Margin, and Risk

Exercise, assignment, expiration, and corporate actions are deterministic
policies, not Monte Carlo. Required assignment policies:

```text
expiration_only
explicit_exercise_only
automatic_ITM_exercise
conservative_early_assignment
```

Recommended research default: automatic ITM exercise at expiration and no
unobserved early assignment. This must be reported as a limitation.

Maintain a complete ledger for cash, options, equities, corporate-action
receivables/payables, realized/unrealized P&L, fees, margin, buying power, and
portfolio equity.

Risk controls must include maximum contracts/notional, Greeks, per-underlying
exposure, loss per trade, portfolio drawdown, margin usage, short-option
exposure, open positions, daily loss stop, and strategy-specific limits.

Margin must be pluggable: cash account, Reg-T approximation, broker-specific
rules, and portfolio-margin approximation. Never claim broker-accurate margin
without broker-rule validation.

## Python Strategy API

```python
class Strategy:
    def on_session_start(self, context): ...
    def on_market_snapshot(self, snapshot, context): ...
    def on_corporate_action(self, event, context): ...
    def on_fill(self, fill, context): ...
    def on_session_end(self, context): ...
```

Snapshots include only point-in-time available data: current bars, eligible
chain subset, accepted analytics, positions, cash/margin/risk, and known events.
Strategies return declarative orders; C++ controls fills and state.

## Test Plan

### Pipeline

Test raw API responses through normalized storage, full pagination, provider
contract IDs/styles/assets/raw hashes, malformed deliverables, inactive adjusted
contracts, revisions, cache refresh, all major corporate actions, OCC initial
and superseding memos, same-symbol deliverable changes, one-to-many and chained
lineage, publication/effective time separation, and no lookahead.

### Pricing

Benchmark against an independent golden source across European sanity cases,
American puts, dividend-paying calls, deep ITM/OTM, near expiry, multiple
dividends, high/low volatility, and supported stock-plus-cash adjustments.
Verify prices, IV inversion, all Greeks, and exercise-boundary behavior.

### C++ units

Test money rounding, multipliers, cost basis/P&L, exercise, assignment,
settlement, expiration, quantity conversion, lineage traversal, atomic order
groups, order rules, margin/risk, stream order, and RNG determinism.

### Strategy structures

Create exact ledger tests for long/short calls/puts, debit/credit verticals,
calendars, diagonals, straddles, strangles, iron condors, butterflies, ratios,
covered calls, protective puts, collars, synthetics, rolls, and multi-leg closes.
Assert fills, cash, quantity, margin, P&L, expiration, exercise, and adjustment
outcomes.

### Monte Carlo

1. Zero spread makes every path equal deterministic P&L.
2. Same seed gives bit-for-bit reproducibility.
3. Different seeds alter only spread-derived execution effects.
4. Market data, IV, Greeks, lineage, and settlement are identical across paths.
5. Expected spread cost converges as path count increases.
6. Common random numbers stabilize strategy comparisons.
7. Multi-leg draws are per-leg while execution remains atomic.
8. Confidence intervals narrow at the expected Monte Carlo rate.

### End to end

Create immutable fixture bundles containing raw bars, raw contract responses,
corporate-action records, OCC memo, expected pipeline Parquet, expected lineage,
expected ledger, and expected Monte Carlo summary.

```text
raw fixtures -> pipeline -> contract versions/lineage -> backtester
-> exact ledger comparison -> report comparison
```

Compare the C++ engine with a simple independent Python reference engine on
small datasets. Deterministic outputs must match exactly.

### Performance and memory

Set CI limits for resident memory per streamed day, batch size, open file
handles, bars/second, Monte Carlo paths/second, and full-year completion time.
Verify a streamed run equals an all-in-memory reference run.

## Completion Criteria

Do not call the system industry-grade until all are true:

- data is source-attributed, versioned, reproducible, and manifest-validated;
- every option transition is source-confirmed;
- historical contract terms are never inferred from future snapshots;
- strategies and event handling have no lookahead;
- quantity, deliverables, exercise, and settlement are correct;
- external analytics benchmarks pass;
- bid/ask cost is the only Monte Carlo component;
- spread assumptions are calibrated or clearly reported as sensitivity analysis;
- every run records code/data/config/seed/output hashes;
- golden end-to-end replays pass;
- C++ deterministic output matches the independent Python reference;
- CI enforces correctness, performance, and memory limits.

The target is a deterministic, auditable portfolio engine with one transparent
stochastic uncertainty: bid/ask execution cost where quote history is absent.
