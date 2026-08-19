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
tests/                    750 tests
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

**Settlement follows the contract.** An equity option delivers shares: an assigned
short call establishes a short stock position if none are held, which is what makes
covered calls and diagonals behave correctly. An index option settles in cash and
touches no share position, preferring a published settlement value (SET, VRO) over
the last observed spot. Exercise-by-exception is one cent per share, so $1.00 on a
standard contract, which is also OCC's figure for a cash-settled one.

The aggregate exercise price is the listed strike times the QUOTE multiplier, not
times the delivered share count. They differ for an adjusted contract, and OCC's
reverse-split rule freezes the multiplier for exactly this purpose: the holder of a
1-for-10 adjusted $50 call still pays $5,000 and receives 10 shares.

**Shares are tradable directly.** A covered call, collar or protective put can be
opened rather than only inherited from a settlement. Equity legs use their own penny
grid and their own per-share fee schedule, because pricing shares through the option
spread model would charge a covered call more to buy its stock than to sell its call.

**Exercise is an order.** A strategy can submit one, atomically with whatever
replaces the position. Exercising an out-of-the-money contract is permitted, because
it is legal and occasionally rational and an engine that silently refuses it is
deciding strategy.

**Dividends are paid.** Accrued at ex-date on the share position, paid at the pay
date weeks later, owed on short shares, and gated on the declaration date so a
backtest cannot collect an unannounced payout. Under the conservative assignment
policy a short call is assigned before ex-date when the dividend exceeds the
extrinsic value the holder gives up, decided on the PRIOR session's close because the
underlying opens ex-date lower.

**Contract terms are point-in-time.** A fill is gated both on when terms took effect
and on when they became knowable, and opening a position on an adjusted contract whose
terms are not established as point-in-time is refused.

**Marks go stale.** A carried-forward mark past its age limit values the position at
intrinsic against a fresh spot instead, and the oldest mark behind any valuation is
reported. A contract that stops printing used to mark the book forever.

## Reporting

Every run prints a manifest with the seed, spread model, execution and assignment
policies, fee schedule, a hash of the files actually read, and a hash of the
configuration. Deterministic market-data P&L is reported separately from
stochastic spread cost.

## Limitations, stated plainly

- **Spread cost is a modeled assumption.** Without calibration against real quote
  history the interval is sensitivity analysis, not a forecast. The pipeline does
  not supply NBBO quotes, so the default parameters are illustrative.
- **Early assignment covers the dividend case only.** Under
  `conservative_early_assignment` a short call is assigned before an ex-dividend date
  when the dividend exceeds its extrinsic value. A deep in-the-money short put
  assigned for the interest on its strike is not modelled.
- **Margin is a documented approximation, not broker-validated.** The Robinhood model
  follows rules checked against Robinhood's own help centre and fee schedule --
  including that no approval level permits an uncovered short call, that short puts
  are held at the full strike in both cash and margin accounts, and that short stock
  carries the published 130% rather than the Reg-T 150%. It has not been reconciled
  against a live broker statement, and Robinhood's house requirements are
  model-driven and undisclosed.
- **There is no portfolio-margin model.** One existed as an alias for Reg-T while the
  manifest recorded it as `portfolio_approx`, which named a methodology the run had
  not applied. It was removed rather than implemented: Robinhood does not offer
  portfolio margin, and its 100% maintenance requirement on long options is
  incompatible with it.
- **The equity spread contributes no dispersion at typical prices.** At the default
  1 bp, a $100 stock's modelled half-spread is exactly the half-cent floor, so every
  draw is clamped. That is economically right -- a penny is the minimum tick and a
  liquid stock at that price quotes one cent wide -- but the stochastic term only bites
  above roughly $100.
- **Open interest is a current snapshot, not history.** The pipeline can source it
  from OCC free of charge, but OCC offers no way to ask for a past date, so it is null
  for every day already in the lake.
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
