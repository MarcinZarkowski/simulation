# Options Backtester — Claude Instructions

## Project

Deterministic options portfolio engine over the OptionsBackfill data lake. C++
core (`engine/`, header-only plus pybind11) owns accounting and state
transitions; Python (`optionsbacktester/`) owns data access, strategy logic, and
reporting. See README.md for design rationale and limitations.

## Build and test

```bash
uv sync
touch engine/bindings.cpp && uv run python engine/build.py   # see note below
uv run python -m pytest tests/ -q
```

**Always `touch engine/bindings.cpp` before rebuilding after editing a header.**
setuptools tracks the `.cpp` timestamp and not header dependencies, so a header
change alone is silently skipped and you will test a stale `.so`.

## Invariants to preserve

- **Money is int64 microdollars.** Never introduce a float into a cash, premium,
  fee, cost-basis, or settlement path. `ledger_reconciles()` is an exact equality
  and must stay one.
- **Positions key on `contract_version_id`, never a symbol.** A symbol is an
  attribute of a version.
- **Only spread cost is stochastic.** If a test shows two Monte Carlo paths
  differing in fill counts, expirations, or settlement, something other than the
  spread became random and that is a bug.
- **Draws stay counter-based.** Keep them a pure function of
  `(seed, scenario, order, instrument, timestamp, leg)`. Introducing generator
  state would make results depend on evaluation order.
- **Fills never see their own bar.** Default timing is next-bar-open. Do not add a
  fill rule that reads a high or low to infer a limit crossing.
- **Fail closed on data quality and lineage.** Unconfirmed adjustments must refuse
  the position rather than guess a quantity conversion.

## Testing conventions

- Orders submitted on bar N fill on bar N+1, so tests need an extra bar. Hold the
  price constant across the fill bar when asserting an exact entry price.
- Use `base_config(spread=E.SpreadModelKind.ZERO, fees=False)` for exact ledger
  assertions; turn on exactly one of spread or fees when testing that component.
- Assert exact money via `*_micros`, not `pytest.approx`, wherever a value is
  exact.
- Uncovered short calls are refused under `ROBINHOOD`; use `REG_T` when a test
  needs one to fill.

## Permissions

Full read and write access to this repository. Build, run, and test freely
without asking. Do not push, switch branches, or create branches.
