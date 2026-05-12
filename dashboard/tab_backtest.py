import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import copy
from scipy.stats import norm
from computation.backtest_builder import StrategyBuilder


def _bsm_price(S: float, K: float, r: float, T: float, sigma: float, opt_type: str) -> float:
    if T <= 0:
        if opt_type == "CALL": return max(0.0, S - K)
        return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "CALL":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _compute_strike(current_price: float, opt_type: str, moneyness: str, strikes_away: int) -> float:
    step = max(1.0, round(current_price * 0.025 / 0.5) * 0.5)
    atm = round(current_price / step) * step
    if moneyness == "ATM":
        return atm
    if moneyness == "OTM":
        return atm + strikes_away * step if opt_type == "CALL" else atm - strikes_away * step
    return atm - strikes_away * step if opt_type == "CALL" else atm + strikes_away * step


def render_group_pl_diagram(group: dict):
    S_ATM = 100.0
    STEP  = 1.0
    SIGMA = 0.25
    R     = 0.05

    legs_spec = []
    for leg in group["legs"]:
        offset = leg.get("offset", 0)
        strike = S_ATM + offset * STEP
        T      = leg.get("dte", 30) / 365.0
        premium = _bsm_price(S_ATM, strike, R, T, SIGMA, leg.get("opt_type", "CALL"))
        legs_spec.append({
            "opt_type":  leg.get("opt_type", "CALL"),
            "is_buy":    leg.get("buy", True),
            "offset":    offset,
            "strike":    strike,
            "premium":   premium,
            "contracts": leg.get("contracts", 1),
        })

    if not legs_spec:
        return

    all_offsets = [l["offset"] for l in legs_spec]
    lo = min(min(all_offsets) - 8, -10)
    hi = max(max(all_offsets) + 8, 10)
    xs_offsets = np.linspace(lo, hi, 600)
    xs_prices  = S_ATM + xs_offsets * STEP

    ys = np.zeros(len(xs_offsets))
    for leg in legs_spec:
        mult = 100 * leg["contracts"]
        if leg["opt_type"] == "CALL":
            intrinsic = np.maximum(0.0, xs_prices - leg["strike"])
        else:
            intrinsic = np.maximum(0.0, leg["strike"] - xs_prices)

        if leg["is_buy"]:
            ys += (intrinsic - leg["premium"]) * mult
        else:
            ys += (leg["premium"] - intrinsic) * mult

    max_profit = float(np.max(ys))
    max_loss   = float(np.min(ys))

    breakevens = []
    for j in range(len(ys) - 1):
        if ys[j] * ys[j + 1] < 0:
            be = xs_offsets[j] + (xs_offsets[j+1] - xs_offsets[j]) * (-ys[j] / (ys[j+1] - ys[j]))
            breakevens.append(be)

    tick_vals  = list(range(int(lo), int(hi) + 1, 2))
    tick_texts = ["ATM" if t == 0 else (f"ATM+{t}" if t > 0 else f"ATM{t}") for t in tick_vals]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs_offsets, y=np.where(ys >= 0, ys, 0),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.20)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=xs_offsets, y=np.where(ys <= 0, ys, 0),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.20)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=xs_offsets, y=ys,
        mode="lines", line=dict(color="#e2e8f0", width=2),
        hovertemplate="Strike: ATM%+.1f<br>P/L: $%{y:.2f}<extra></extra>",
        name="P/L"
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#94a3b8",
                  annotation_text="ATM", annotation_position="top")
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_width=1)
    
    for be in breakevens:
        fig.add_vline(x=be, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"BE ATM{be:+.1f}",
                      annotation_position="bottom right",
                      annotation_font_color="#fbbf24", annotation_font_size=10)
                      
    for leg in legs_spec:
        color = "#10b981" if leg["is_buy"] else "#ef4444"
        lbl   = ("Buy" if leg["is_buy"] else "Sell") + " " + leg["opt_type"]
        if leg["offset"] == 0: lbl += " ATM"
        elif leg["offset"] > 0: lbl += f" ATM+{leg['offset']}"
        else: lbl += f" ATM{leg['offset']}"
        fig.add_vline(x=leg["offset"], line_dash="dot",
                      line_color=color, line_width=1,
                      annotation_text=lbl, annotation_position="top left",
                      annotation_font_color=color, annotation_font_size=10)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis=dict(
            title="Strike Position (relative to ATM)",
            tickvals=tick_vals, ticktext=tick_texts,
            zeroline=False
        ),
        yaxis_title="Theoretical P/L ($)",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Max Profit",  f"${max_profit:+.2f}",  delta_color="normal" if max_profit >= 0 else "inverse")
    s2.metric("Max Loss",    f"${max_loss:+.2f}",    delta_color="inverse" if max_loss < 0 else "normal")
    be_str = "  /  ".join([f"ATM{b:+.1f}" for b in breakevens]) if breakevens else "None"
    s3.metric("Breakeven(s)", be_str)


def render_leg(leg: dict, g_idx: int, l_idx: int, legs_to_remove: list):
    is_buy = leg.get("buy", True)
    opt_type = leg.get("opt_type", "CALL")
    
    bg_color = "rgba(16, 185, 129, 0.05)" if is_buy else "rgba(239, 68, 68, 0.05)"
    border_color = "#10b981" if is_buy else "#ef4444"
    action_text = "BUY" if is_buy else "SELL"
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; border-left: 2px solid {border_color}; 
                padding: 4px 8px; margin-bottom: 8px;">
        <span style="color: {border_color}; font-weight: 600; font-size: 0.9em;">
            {action_text} {opt_type}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    lc1, lc2, lc3, lc4, lc5, lc6 = st.columns([1.3, 1.3, 2.5, 1.2, 0.8, 0.5])

    leg["buy"] = lc1.selectbox("Action", ["Buy", "Sell"],
                                key=f"act_{g_idx}_{l_idx}",
                                index=0 if is_buy else 1, label_visibility="collapsed") == "Buy"
    leg["opt_type"] = lc2.selectbox("Type", ["CALL", "PUT"],
                                     key=f"typ_{g_idx}_{l_idx}",
                                     index=0 if opt_type == "CALL" else 1, label_visibility="collapsed")

    current_offset = leg.get("offset", 0)
    new_offset = lc3.slider(
        "Strike vs ATM",
        min_value=-15, max_value=15,
        value=int(current_offset),
        key=f"off_{g_idx}_{l_idx}",
        label_visibility="collapsed"
    )
    leg["offset"] = new_offset

    if new_offset == 0:
        strike_label = "ATM"
    elif opt_type == "CALL":
        strike_label = f"{new_offset} above ATM ({'OTM' if new_offset > 0 else 'ITM'})"
    else:
        strike_label = f"{abs(new_offset)} below ATM ({'OTM' if new_offset < 0 else 'ITM'})" if new_offset < 0 else f"{new_offset} above ATM (ITM for PUT)"
    lc3.caption(strike_label)

    leg["dte"]       = lc4.number_input("DTE", 1, 1000, leg.get("dte", 30), key=f"dte_{g_idx}_{l_idx}", label_visibility="collapsed")
    leg["contracts"] = lc5.number_input("Qty", 1, 1000, leg.get("contracts", 1), key=f"con_{g_idx}_{l_idx}", label_visibility="collapsed")

    with lc6:
        if st.button("X", key=f"rm_l_{g_idx}_{l_idx}"):
            legs_to_remove.append(l_idx)


def render_group_conditions(group: dict, g_idx: int):
    cond_c1, cond_c2 = st.columns(2)
    
    with cond_c1:
        st.markdown("**Entry Rules (AND)**")
        entry = group["entry"]
        
        entry["use_freq"] = st.checkbox("Every N Days", value=entry.get("use_freq", True), key=f"efreq_chk_{g_idx}")
        entry["freq_days"] = st.number_input("Days", 1, 252, entry.get("freq_days", 30), disabled=not entry["use_freq"], key=f"efreq_val_{g_idx}", label_visibility="collapsed")
        
        entry["use_price_gt"] = st.checkbox("Price >", value=entry.get("use_price_gt", False), key=f"epgt_chk_{g_idx}")
        entry["price_gt"] = st.number_input("Min Price", 0.0, 10000.0, entry.get("price_gt", 100.0), disabled=not entry["use_price_gt"], key=f"epgt_val_{g_idx}", label_visibility="collapsed")
        
        entry["use_price_lt"] = st.checkbox("Price <", value=entry.get("use_price_lt", False), key=f"eplt_chk_{g_idx}")
        entry["price_lt"] = st.number_input("Max Price", 0.0, 10000.0, entry.get("price_lt", 200.0), disabled=not entry["use_price_lt"], key=f"eplt_val_{g_idx}", label_visibility="collapsed")
        
        entry["use_iv_gt"] = st.checkbox("IV >", value=entry.get("use_iv_gt", False), key=f"eigt_chk_{g_idx}")
        entry["iv_gt"] = st.number_input("Min IV", 0.0, 5.0, entry.get("iv_gt", 0.2), step=0.05, disabled=not entry["use_iv_gt"], key=f"eigt_val_{g_idx}", label_visibility="collapsed")
        
        entry["use_iv_lt"] = st.checkbox("IV <", value=entry.get("use_iv_lt", False), key=f"eilt_chk_{g_idx}")
        entry["iv_lt"] = st.number_input("Max IV", 0.0, 5.0, entry.get("iv_lt", 0.8), step=0.05, disabled=not entry["use_iv_lt"], key=f"eilt_val_{g_idx}", label_visibility="collapsed")
        
        entry["use_entry_days"] = st.checkbox("Only on weekdays", value=entry.get("use_entry_days", False), key=f"edays_chk_{g_idx}")
        if entry["use_entry_days"]:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            current_days = entry.get("entry_days", [0,1,2,3,4])
            entry["entry_days"] = [i for i, name in enumerate(day_names) if st.checkbox(name, value=i in current_days, key=f"eday_{g_idx}_{i}")]
        
    with cond_c2:
        st.markdown("**Exit Rules (OR)**")
        exit_rule = group["exit"]
        
        exit_rule["use_profit_pct"] = st.checkbox("Profit % >=", value=exit_rule.get("use_profit_pct", True), key=f"xppct_chk_{g_idx}")
        exit_rule["profit_pct"] = st.number_input("Target Profit %", 0.01, 10.0, exit_rule.get("profit_pct", 0.50), step=0.05, disabled=not exit_rule["use_profit_pct"], key=f"xppct_val_{g_idx}", label_visibility="collapsed")
        
        exit_rule["use_loss_pct"] = st.checkbox("Loss % >=", value=exit_rule.get("use_loss_pct", True), key=f"xlpct_chk_{g_idx}")
        exit_rule["loss_pct"] = st.number_input("Stop Loss %", 0.01, 10.0, exit_rule.get("loss_pct", 0.20), step=0.05, disabled=not exit_rule["use_loss_pct"], key=f"xlpct_val_{g_idx}", label_visibility="collapsed")
        
        exit_rule["use_profit_dlr"] = st.checkbox("Profit $ >=", value=exit_rule.get("use_profit_dlr", False), key=f"xpdlr_chk_{g_idx}")
        exit_rule["profit_dlr"] = st.number_input("Target Profit $", 0.0, 10000.0, exit_rule.get("profit_dlr", 500.0), disabled=not exit_rule["use_profit_dlr"], key=f"xpdlr_val_{g_idx}", label_visibility="collapsed")
        
        exit_rule["use_loss_dlr"] = st.checkbox("Loss $ >=", value=exit_rule.get("use_loss_dlr", False), key=f"xldlr_chk_{g_idx}")
        exit_rule["loss_dlr"] = st.number_input("Stop Loss $", 0.0, 10000.0, exit_rule.get("loss_dlr", 200.0), disabled=not exit_rule["use_loss_dlr"], key=f"xldlr_val_{g_idx}", label_visibility="collapsed")
        
        exit_rule["use_dte"] = st.checkbox("DTE <=", value=exit_rule.get("use_dte", False), key=f"xdte_chk_{g_idx}")
        exit_rule["dte_rem"] = st.number_input("Min DTE", 0, 365, exit_rule.get("dte_rem", 5), disabled=not exit_rule["use_dte"], key=f"xdte_val_{g_idx}", label_visibility="collapsed")
        
        exit_rule["use_exit_days"] = st.checkbox("Only exit on weekdays", value=exit_rule.get("use_exit_days", False), key=f"xdays_chk_{g_idx}")
        if exit_rule["use_exit_days"]:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            current_days = exit_rule.get("exit_days", [0,1,2,3,4])
            exit_rule["exit_days"] = [i for i, name in enumerate(day_names) if st.checkbox(name, value=i in current_days, key=f"xday_{g_idx}_{i}")]
        
        st.markdown("**Strategy Close Mode**")
        idx_close = 0 if group.get("close_mode", "ALL_TOGETHER") == "ALL_TOGETHER" else 1
        close_mode_str = st.radio("When an exit triggers:", ["Close All Legs", "Close Expired Legs Only"], key=f"cmode_{g_idx}", index=idx_close, label_visibility="collapsed")
        group["close_mode"] = "ALL_TOGETHER" if close_mode_str == "Close All Legs" else "INDIVIDUALLY"


def run_simulation(config, data, adjusted_vol, ticker, start_balance, start_shares, start_avg_cost):
    builder = StrategyBuilder()
    for group in st.session_state.bt_groups:
        entry = group["entry"]
        entries = []
        if entry.get("use_freq"): entries.append(builder.cond.every_n_days(entry.get("freq_days", 30)))
        if entry.get("use_price_gt"): entries.append(builder.cond.price_above(entry.get("price_gt", 100.0)))
        if entry.get("use_price_lt"): entries.append(builder.cond.price_below(entry.get("price_lt", 200.0)))
        if entry.get("use_iv_gt"): entries.append(builder.cond.iv_above(entry.get("iv_gt", 0.2)))
        if entry.get("use_iv_lt"): entries.append(builder.cond.iv_below(entry.get("iv_lt", 0.8)))
        if entry.get("use_entry_days"): entries.append(builder.cond.on_days(entry.get("entry_days", [0,1,2,3,4])))
        
        if len(entries) == 0:
            entry_cond = builder.cond.always()
        else:
            entry_cond = entries[0]
            for i in range(1, len(entries)):
                entry_cond = builder.cond.and_(entry_cond, entries[i])
                
        exit_rule = group["exit"]
        exits = []
        if exit_rule.get("use_profit_pct"): exits.append(builder.cond.profit_pct(exit_rule.get("profit_pct", 0.50)))
        if exit_rule.get("use_loss_pct"): exits.append(builder.cond.loss_pct(exit_rule.get("loss_pct", 0.20)))
        if exit_rule.get("use_profit_dlr"): exits.append(builder.cond.profit_dollars(exit_rule.get("profit_dlr", 500.0)))
        if exit_rule.get("use_loss_dlr"): exits.append(builder.cond.loss_dollars(exit_rule.get("loss_dlr", 200.0)))
        if exit_rule.get("use_dte"): exits.append(builder.cond.dte_remaining(exit_rule.get("dte_rem", 5)))
        if exit_rule.get("use_exit_days"): exits.append(builder.cond.on_days(exit_rule.get("exit_days", [0,1,2,3,4])))
        
        if len(exits) == 0:
            exit_cond = builder.cond.hold_to_expiry()
        else:
            exit_cond = exits[0]
            for i in range(1, len(exits)):
                exit_cond = builder.cond.or_(exit_cond, exits[i])
                
        leg_configs = []
        for leg in group["legs"]:
            leg_configs.append({
                "action":    "buy" if leg.get("buy", True) else "sell",
                "type":      leg.get("opt_type", "CALL"),
                "offset":    leg.get("offset", 0),
                "dte":       leg.get("dte", config["default_dte"]),
                "contracts": leg.get("contracts", 1)
            })
            
        builder.add_group(
            entry_condition=entry_cond,
            exit_condition=exit_cond,
            close_mode=group.get("close_mode", "ALL_TOGETHER"),
            legs=leg_configs
        )
        
    return builder.run(
        prices_df=data,
        adjusted_vol=adjusted_vol,
        ticker=ticker,
        starting_balance=float(start_balance),
        starting_shares=int(start_shares),
        starting_average_cost=float(start_avg_cost),
        risk_free_rate=config["risk_free_rate"]
    )


def render_results(data, res, initial_value):
    daily_vals = res.get("daily_values", [])
    if len(daily_vals) == 0:
        st.warning("No data returned.")
        return
        
    if len(daily_vals) != len(data.index):
        st.info("Input data range has changed. Please click 'Run Backtest' to update the results.")
        return            
        
    df_daily = pd.DataFrame(index=data.index)
    df_daily["total_value"] = daily_vals
    df_daily["balance"] = res.get("daily_balance", [])
    
    initial_stock_price = data.iloc[0].Close
    bnh_shares = initial_value / initial_stock_price
    df_daily["Buy & Hold"] = bnh_shares * data["Close"]
    
    final_strat_val = df_daily["total_value"].iloc[-1]
    final_bnh_val = df_daily["Buy & Hold"].iloc[-1]
    
    strat_return_pct = (final_strat_val - initial_value) / initial_value * 100
    bnh_return_pct = (final_bnh_val - initial_value) / initial_value * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily["total_value"],
        mode="lines", name="Strategy Account Value",
        line=dict(color="#3b82f6", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily["Buy & Hold"],
        mode="lines", name="Buy & Hold",
        line=dict(color="#10b981", width=2)
    ))
    fig.update_layout(
        title="Portfolio Growth",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        yaxis_title="Account Value ($)",
        xaxis_title="",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    strat_color = "normal" if strat_return_pct > 0 else "inverse"
    bnh_color = "normal" if bnh_return_pct > 0 else "inverse"
    
    m1.metric("Strategy Return", f"{strat_return_pct:+.2f}%", f"${final_strat_val - initial_value:+.2f}", delta_color=strat_color)
    m2.metric("Buy & Hold Return", f"{bnh_return_pct:+.2f}%", f"${final_bnh_val - initial_value:+.2f}", delta_color=bnh_color)
    m3.metric("Win Rate", f"{res.get('win_rate', 0)*100:.1f}%")
    m4.metric("Max Profit (Trade)", f"${res.get('max_profit', 0):,.2f}")
    m5.metric("Max Loss (Trade)", f"${res.get('max_loss', 0):,.2f}")
    m6.metric("Positions Opened", f"{res.get('total_positions_opened', 0)}")

    m1b, m2b, m3b, m4b, m5b, m6b = st.columns(6)
    m1b.metric("Average Win", f"${res.get('avg_win', 0):,.2f}")
    m2b.metric("Average Loss", f"${res.get('avg_loss', 0):,.2f}")


def render(config: dict):
    st.title("Strategy Backtester")
    st.markdown("Configure strategy groups, set starting balance, and test against history.")

    ticker = config["ticker"]
    data = config["data"]
    
    st.markdown(f"**Data Range:** {data.index.min().date()} to {data.index.max().date()} ({len(data)} trading days) for **{ticker}**")
    
    st.subheader("Account Setup")
    c1, c2, c3 = st.columns(3)
    start_balance = c1.number_input("Starting Cash ($)", min_value=0.0, value=10000.0, step=1000.0)
    start_shares = c2.number_input("Starting Shares", min_value=0, value=0, step=100)
    start_avg_cost = c3.number_input("Starting Stock Avg Cost ($)", min_value=0.0, value=0.0, step=10.0)
    
    st.subheader("Strategy Groups")
    
    if "bt_groups" not in st.session_state:
        st.session_state.bt_groups = [{
            "name": "Strategy Group 1",
            "entry": {"use_freq": True, "freq_days": 30},
            "exit": {"use_profit_pct": True, "profit_pct": 0.50, "use_loss_pct": True, "loss_pct": 0.20},
            "close_mode": "ALL_TOGETHER",
            "legs": [{
                "buy": True, "opt_type": "CALL", "moneyness": "ATM", "strikes": 1, "dte": 30, "contracts": 1
            }]
        }]

    groups_to_remove = []
    
    for g_idx, group in enumerate(st.session_state.bt_groups):
        with st.expander(f"{group['name']} ({len(group['legs'])} legs)", expanded=True):
            h_col1, h_col2, h_col3 = st.columns([0.6, 0.2, 0.2])
            group["name"] = h_col1.text_input("Group Name", value=group["name"], key=f"gname_{g_idx}", label_visibility="collapsed")
            
            if h_col2.button("Duplicate", key=f"dup_g_{g_idx}", use_container_width=True):
                new_group = copy.deepcopy(group)
                new_group["name"] += " (Copy)"
                st.session_state.bt_groups.append(new_group)
                st.rerun()
                
            if h_col3.button("Delete Group", key=f"rm_g_{g_idx}", use_container_width=True):
                groups_to_remove.append(g_idx)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("**Legs**")
            legs_to_remove = []
            for l_idx, leg in enumerate(group["legs"]):
                render_leg(leg, g_idx, l_idx, legs_to_remove)
                            
            for l_idx in sorted(legs_to_remove, reverse=True):
                group["legs"].pop(l_idx)
            if legs_to_remove:
                st.rerun()
                
            if st.button("Add New Leg", key=f"add_l_{g_idx}"):
                group["legs"].append({"buy": True, "opt_type": "CALL", "offset": 0, "dte": 30, "contracts": 1})
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**P/L at Expiration (Theoretical)**")
            render_group_pl_diagram(group)
                
            st.markdown("<br>", unsafe_allow_html=True)
            render_group_conditions(group, g_idx)

    for g_idx in sorted(groups_to_remove, reverse=True):
        st.session_state.bt_groups.pop(g_idx)
    if groups_to_remove:
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Add Strategy Group", use_container_width=True):
        st.session_state.bt_groups.append({
            "name": f"Strategy Group {len(st.session_state.bt_groups) + 1}",
            "entry": {"use_freq": True, "freq_days": 30},
            "exit": {"use_profit_pct": True, "profit_pct": 0.50, "use_loss_pct": True, "loss_pct": 0.20},
            "close_mode": "ALL_TOGETHER",
            "legs": [{"buy": True, "opt_type": "CALL", "offset": 0, "dte": 30, "contracts": 1}]
        })
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Backtest", use_container_width=True, type="primary"):
        if not st.session_state.bt_groups:
            st.error("Please add at least one strategy group.")
        else:
            valid = True
            for g in st.session_state.bt_groups:
                if len(g["legs"]) == 0:
                    st.error(f"Group '{g['name']}' has no legs. Please add legs or delete the group.")
                    valid = False
            
            if valid:
                with st.spinner("Running simulation..."):
                    try:
                        results = run_simulation(config, data, config["adjusted_vol"], ticker, start_balance, start_shares, start_avg_cost)
                        st.session_state.bt_results = results
                        st.session_state.bt_start_balance = start_balance
                        st.session_state.bt_start_shares = start_shares
                        st.session_state.bt_start_avg_cost = start_avg_cost
                        st.success("Simulation complete!")
                    except Exception as e:
                        st.error(f"Error running backtest: {str(e)}")

    if "bt_results" in st.session_state and st.session_state.bt_results:
        initial_value = st.session_state.bt_start_balance + (st.session_state.bt_start_shares * data.iloc[0].Close)
        render_results(data, st.session_state.bt_results, initial_value)
