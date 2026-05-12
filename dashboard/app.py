"""
Options Simulator — Streamlit Dashboard.

Main entrypoint. Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

st.set_page_config(
    page_title="Options Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.sidebar import render_sidebar

config = render_sidebar()

if config is None:
    st.stop()

from dashboard.tab_backtest import render as render_backtest

tab1, = st.tabs([
    "Backtest Strategy"
])

with tab1:
    render_backtest(config)
