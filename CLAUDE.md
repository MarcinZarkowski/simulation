# Options Strategy Backtester — Claude Instructions

## Project Overview

This is an options strategy backtesting system. The core engine is a C++ Python extension (`backtest_engine.cpython-312-darwin.so`), wrapped by Python code and surfaced through a Streamlit dashboard.

## File Structure

- `computation/` — Core simulation logic
  - `cpp/` — C++ engine source (`backtest.cpp`, `position_management.cpp`, etc.) and `build.py`
  - `backtest_builder.py` — Python wrapper encoding strategies for the C++ engine
  - `config.py` — Central config for volatility windows and risk-free rates
- `dashboard/` — Streamlit frontend
  - `app.py` — Main entry point
  - `tab_backtest.py` — Strategy builder and performance visualization

## Build & Run

```bash
# Rebuild C++ extension after any changes to C++ source
uv run computation/cpp/build.py

# Run the dashboard
streamlit run dashboard/app.py
```

## Permissions

Claude has full read and write access to all files in this repository. You may:

- Read, edit, and create any file under this directory
- Run build commands (`uv run computation/cpp/build.py`)
- Run the dashboard and tests
- Modify C++ source, Python code, and configuration files without asking for confirmation

## Development Notes

- After editing any `.cpp` file, always rebuild the extension before testing
- Use `uv` (not `pip`) for Python dependency management
- The C++ engine is the performance-critical path; Python wrappers handle strategy encoding
