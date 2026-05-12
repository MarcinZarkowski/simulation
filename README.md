# Options Strategy Backtester

First simulates/recreates historical option chains using historical stock prices, calculating IV based on regime regime detection and using 
the binomial tree model to price the options.
Allows building and testing multi-leg options strategies using different entry and exit conditions.
Acts like broker in account simulation by not allowing entry without enough collateral and sharing shares if short call expires in the money.

## File Structure
- `computation/`: Core simulation logic.
  - `cpp/`: C++ engine source (`backtest.cpp`, `position_management.cpp`, etc.) and the `build.py` script.
  - `backtest_builder.py`: Python wrapper that encodes strategies for the C++ engine.
  - `config.py`: Central configuration for volatility windows and risk-free rates.
- `dashboard/`: Frontend UI.
  - `app.py`: Main dashboard entry point.
  - `tab_backtest.py`: The primary strategy builder and performance visualization interface.

## Getting Started

### 1. Installation
Clone the repository and ensure you have the necessary dependencies installed (using `uv` or `pip`).

### 2. Build the C++ Engine
If you make any changes to the backtest logic in the C++ code, you must rebuild the Python extension:
```bash
uv run computation/cpp/build.py
```

### 3. Run the Dashboard
Launch the Streamlit interface to start building and testing strategies:
```bash
streamlit run dashboard/app.py
```
