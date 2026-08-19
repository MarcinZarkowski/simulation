# Options Backtester

Deterministic options portfolio engine over the OptionsBackfill data lake, with a
C++ core for accounting and state transitions and a Python layer for data access,
strategy logic, and reporting.

The governing rule is that everything except execution cost is deterministic:

> Market data, contract terms, corporate actions, and settlement are
> deterministic. Only synthetic bid/ask execution cost is Monte Carlo simulated.

Identical data, strategy, engine version, configuration, and seed produce
identical results, byte for byte.

## Layout

```text
engine/                   C++ core (header-only plus pybind11 bindings)
  types.h                 int64 microdollar money, timestamps, stable ids
  contract.h              interval-versioned contract terms, bars, analytics
  order.h                 orders, atomic groups, fills, rejections
  position.h              positions and exact cost-basis accounting
  ledger.h                append-only journal and the fee schedule
  margin.h                pluggable margin: cash, Reg-T, Robinhood
  spread.h                counter-based RNG and the spread models
  engine.h                event clock, execution, settlement, risk
  bindings.cpp            Python surface
optionsbacktester/        Python layer
  stream.py               partition-pruned streaming reader
  contracts.py            pipeline rows to engine structs
  strategy.py             Strategy base class, Chain, Context, order helpers
  runner.py               Monte Carlo runner and run manifest
  report.py               path aggregation and reporting
  cli.py                  command line entry point
  strategies/             worked strategies, including a poor man's covered call
tests/                    392 tests
  fixtures.py             deterministic synthetic lake in the pipeline's schema
```

## Build and run

```bash
uv sync
uv run python engine/build.py          # produces obt_engine.*.so at the repo root
uv run python -m pytest tests/ -q
```

`engine/build.py` compiles only `bindings.cpp`, which includes the headers. Note
that setuptools tracks the `.cpp` timestamp and not header dependencies, so touch
`bindings.cpp` after editing a header or the rebuild is skipped.

```bash
uv run python -m optionsbacktester run \
  --strategy optionsbacktester.strategies.pmcc:PoorMansCoveredCall \
  --data-root /path/to/data --tickers SPY \
  --start 2024-01-01 --end 2024-06-30 \
  --spread-mc-paths 1000 --spread-mc-seed 42 \
  --spread-model conditional_lognormal \
  --report-confidence-level 0.95
```

Before trusting an interval, check that the Monte Carlo error is actually
falling:

```bash
uv run python -m optionsbacktester convergence --strategy ... --data-root ... --tickers SPY
```

## Data input

Reads the lake the pipeline writes:

```text
data/{TICKER}/{YYYY}/{MM}/{DD}/
  options_enriched.parquet          the 49-column canonical schema
  stock.parquet
  option_contract_version.parquet   interval-versioned terms, when present
  option_lineage_event.parquet      adjustment transitions, when present
  corporate_actions.parquet
  _SUCCESS
```

Only day directories in the requested range are opened, only the columns the
engine consumes are read, and days are yielded one at a time, so memory is flat
over a multi-year run. Days without `_SUCCESS` are skipped unless
`--allow-incomplete-days` is passed.

Quality gating is expressed as filters over the pipeline's own flags rather than
reinterpreted, so the engine and the pipeline cannot disagree about which rows
are usable. By default a row is excluded when `is_stale`, `iv_failed`, or
`iv_is_model_fallback` is true, or when `adjusted_pricing_status` is
`unpriced_adjusted_contract`.

## Design decisions that matter

**Money is int64 microdollars.** Every cash, premium, fee, cost-basis, and
settlement amount is an exact integer, and the ledger reconciling to its journal
is an equality rather than a tolerance. Greeks and IV stay floating point because
they are analytics, never ledger values.

**Positions reference a contract version, not a symbol.** The same OCC symbol can
describe different economics either side of an adjustment, and one economic
series can change symbol, so a symbol is an attribute of a version rather than an
identity.

**Signals cannot see their own fill price.** Execution defaults to
next-bar-open: an order submitted from a bar's close fills at the following bar's
open. Intrabar path and queue position are unknown from OHLC data, so a fill is
never inferred from a high or low crossing a limit.

**Multi-leg groups are atomic.** All legs fill or none do. A broker does not
half-execute a spread, and a half-filled vertical is a different position with
different risk.

**Spread cost is the only randomness.** Draws are a pure function of
`(seed, scenario, order, instrument, timestamp, leg)`, so they are independent of
evaluation order, reproducible from the key alone, and give common random numbers
when comparing strategies. All paths advance in lockstep over one pass of the
data, so the tape is identical across paths by construction.

**Margin is pluggable and pairs legs by expiry.** A long option only offsets a
short if it lives at least as long, so a short "covered" by a long that expires
first is correctly treated as naked. Spread requirements use the max-loss netting
of FINRA 4210(f)(2)(H)(i), evaluated at every strike in the position, which
charges an iron condor its wider side rather than the sum of both.

**A poor man's covered call needs no shares.** The long leg collateralizes the
short, so the requirement is the debit paid. Short calls can be written against
one long leg repeatedly until it expires.

**Settlement is physical.** An assigned short call delivers shares and
establishes a short stock position if none are held, which is what makes covered
calls and diagonals behave correctly. Exercise-by-exception is $0.01 of
intrinsic.

## Reporting

Every run prints a manifest with the seed, spread model, execution and assignment
policies, fee schedule, a hash of the files actually read, and a hash of the
configuration. Deterministic market-data P&L is reported separately from
stochastic spread cost.

## Limitations, stated plainly

- **Spread cost is a modeled assumption.** Without calibration against real quote
  history the interval is sensitivity analysis, not a forecast. The pipeline does
  not supply NBBO quotes, so the default parameters are illustrative.
- **No early assignment is modeled** beyond expiration. The default policy is
  automatic ITM exercise at expiration; a short option is not assigned early even
  when a dividend would make it rational.
- **Margin is a documented approximation, not broker-validated.** The Robinhood
  model follows published rules and refuses uncovered short calls as the real
  broker does, but it has not been reconciled against live broker statements.
- **Lineage is fail-closed and mostly unconfirmed.** The pipeline currently
  produces adjustment candidates without OCC memo confirmation, so a position
  held through an adjustment is refused rather than converted. This is deliberate:
  guessing a quantity conversion would corrupt every number downstream.
- **Fee rates are point-in-time.** Defaults reflect the published 2026-04
  schedule. Section 31 was $0 from 2025-05-14 to 2026-04-03, so a run spanning
  that window should set the rate per period.
- **Stop orders are not simulated.** OHLC data cannot support a trigger without
  inventing an intrabar path.
- **The `computation/` and `dashboard/` directories are the previous
  implementation** and are not used by this engine. They read daily stock bars
  from yfinance and synthesize an option chain, so their results are statements
  about a model rather than a market. They are left in place rather than deleted.
