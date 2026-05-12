#include "data.cpp"
#include "options_pricing.cpp"
#include <pybind11/stl.h>
#include <sstream>
#include <iostream>

#include "position_management.cpp"

// ═══════════════════════════════════════════════════════════════════════
// Daily simulation steps
// ═══════════════════════════════════════════════════════════════════════

void check_exits(vector<OpenPosition> &positions, Account &account,
                 const vector<Condition> &conditions,
                 const MarketState &market, float r) {
  vector<int> to_remove;

  for (int p = 0; p < (int)positions.size(); p++) {
    auto &pos = positions[p];
    for (auto &leg : pos.legs) { leg.dte--; reprice_leg(leg, market.close, market.iv, r); }

    GroupState gs = compute_group_state(pos, market.day_index, market.close);
    bool should_exit = evaluate_condition(conditions, pos.exit_root, gs, market);
    bool any_expired = (gs.min_dte <= 0);

    if (!should_exit && !any_expired) continue;

    if (pos.close_mode == CLOSE_ALL_TOGETHER || should_exit) {
      float pnl = close_entire_position(pos, account, market.close);
      account.record_trade(pnl);
      to_remove.push_back(p);
    } else {
      auto [pnl, num_closed, remaining] = close_expired_legs_only(pos, account, market.close);
      if (num_closed > 0) {
        account.record_trade(pnl);
      }
      if (remaining.empty()) to_remove.push_back(p);
      else                   pos.legs = remaining;
    }
  }

  for (int k = (int)to_remove.size() - 1; k >= 0; k--)
    positions.erase(positions.begin() + to_remove[k]);
}

void check_entries(vector<OpenPosition> &positions, Account &account,
                   const vector<Condition> &conditions,
                   vector<Strategy> &strategies, const vector<Leg> &blueprints,
                   const MarketState &market, float r) {
  for (int g = 0; g < (int)strategies.size(); g++) {
    GroupState dummy{};
    if (!evaluate_condition(conditions, strategies[g].entry_root, dummy, market))
      continue;

    vector<TradeLeg> priced_legs = build_trade_legs(strategies[g], blueprints, market, r);
    if (priced_legs.empty()) continue;
    if (!group_can_afford(priced_legs, account, market.close)) continue;

    OpenPosition pos;
    pos.group_id   = g;
    pos.close_mode = strategies[g].close_mode;
    pos.exit_root  = strategies[g].exit_root;
    pos.entry_day  = market.day_index;
    pos.total_entry_cost = 0;
    pos.legs       = priced_legs;

    commit_position_to_account(pos, account);
    positions.push_back(pos);
    account.total_positions_opened++;
    strategies[g].last_entry_day = market.day_index;
  }
}

void update_collateral(Account &account, const vector<OpenPosition> &positions, float close_price) {
  account.locked_collateral = compute_locked_collateral(positions, close_price);
  account.locked_shares     = compute_locked_shares(positions);
}

float compute_daily_snapshot(const vector<OpenPosition> &positions,
                             const Account &account, float close_price) {
  float options_val = 0;
  for (const auto &pos : positions)
    for (const auto &leg : pos.legs) {
      int mult = 100 * leg.contracts;
      if (leg.buy) options_val += leg.opt_price * mult;
      else         options_val -= leg.opt_price * mult;
    }

  return account.balance + options_val + account.shares * close_price;
}

// ═══════════════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════════════

py::dict build_results(const Account &account,
                       py::array_t<float> dv, py::array_t<float> db,
                       py::array_t<float> dc, float net_profit) {
  py::dict r;
  r["daily_values"]     = dv;
  r["daily_balance"]    = db;
  r["daily_collateral"] = dc;
  r["total_pnl"]        = account.total_pnl();
  r["avg_pnl"]          = account.avg_pnl();
  r["win_rate"]         = account.win_rate();
  r["max_profit"]       = (account.best_trade > -1e17f)  ? account.best_trade  : 0.0f;
  r["max_loss"]         = (account.worst_trade < 1e17f)  ? account.worst_trade : 0.0f;
  r["avg_win"]          = account.avg_win();
  r["avg_loss"]         = account.avg_loss();
  r["total_trades"]     = account.total_trades;
  r["total_positions_opened"] = account.total_positions_opened;
  r["closed_pnls"]      = account.closed_pnls;
  r["final_balance"]    = account.balance;
  r["net_strategy_profit"] = net_profit;
  return r;
}

// ═══════════════════════════════════════════════════════════════════════
// run_backtest — main entry point
// ═══════════════════════════════════════════════════════════════════════

py::dict
run_backtest(py::array_t<float> opens, py::array_t<float> highs,
             py::array_t<float> lows,  py::array_t<float> closes,
             py::array_t<float> volumes, py::array_t<float> adjusted_vol,
             py::array_t<int> day_of_week,
             float starting_balance, int starting_shares, float starting_average_cost, string ticker,
             py::array_t<int> leg_group_id, py::array_t<int> leg_buy,
             py::array_t<int> leg_opt_type, py::array_t<int> leg_strike_offset,
             py::array_t<int> leg_dte, py::array_t<int> leg_contracts,
             py::array_t<int> group_close_mode,
             py::array_t<int> group_entry_root,
             py::array_t<int> group_exit_root,
             py::array_t<int> cond_type,
             py::array_t<float> cond_param1, py::array_t<float> cond_param2,
             py::array_t<int> cond_child1, py::array_t<int> cond_child2,
             float r) {

  auto conditions = decode_conditions(cond_type, cond_param1, cond_param2, cond_child1, cond_child2);
  auto blueprints = decode_legs(leg_group_id, leg_buy, leg_opt_type, leg_strike_offset, leg_dte, leg_contracts);
  auto strategies = decode_strategies(group_close_mode, group_entry_root, group_exit_root, blueprints);

  Account account;
  account.balance = starting_balance;
  account.shares  = starting_shares;
  account.average_stock_cost = starting_average_cost;

  auto o   = opens.unchecked<1>();
  auto h   = highs.unchecked<1>();
  auto l   = lows.unchecked<1>();
  auto c   = closes.unchecked<1>();
  auto v   = volumes.unchecked<1>();
  auto vol = adjusted_vol.unchecked<1>();
  auto dow = day_of_week.unchecked<1>();
  int num_days = (int)closes.shape(0);

  py::array_t<float> out_dv(num_days), out_db(num_days), out_dc(num_days);
  auto dv = out_dv.mutable_unchecked<1>();
  auto db = out_db.mutable_unchecked<1>();
  auto dc = out_dc.mutable_unchecked<1>();

  vector<OpenPosition> positions;

  for (int i = 0; i < num_days; i++) {
    MarketState market = {o(i), h(i), l(i), c(i), v(i), vol(i), i, dow(i)};

    check_exits(positions, account, conditions, market, r);
    update_collateral(account, positions, market.close);
    check_entries(positions, account, conditions, strategies, blueprints, market, r);
    update_collateral(account, positions, market.close);
    
    dv(i) = compute_daily_snapshot(positions, account, market.close); 
    db(i) = account.balance; 
    dc(i) = account.locked_collateral;
  }

  float initial_total_value = starting_balance + starting_shares * starting_average_cost;
  float final_total_value = num_days == 0 ? initial_total_value : dv(num_days - 1);
  float net_profit = final_total_value - initial_total_value;

  return build_results(account, out_dv, out_db, out_dc, net_profit);
}

// ═══════════════════════════════════════════════════════════════════════
// pybind11 module
// ═══════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(backtest_engine, m) {
  m.doc() = "C++ backtesting engine with composable conditions";
  m.def("run_backtest", &run_backtest, "Run a strategy backtest",
      py::arg("opens"), py::arg("highs"), py::arg("lows"), py::arg("closes"),
      py::arg("volumes"), py::arg("adjusted_vol"), py::arg("day_of_week"),
      py::arg("starting_balance"), py::arg("starting_shares"), py::arg("starting_average_cost"), py::arg("ticker"),
      py::arg("leg_group_id"), py::arg("leg_buy"), py::arg("leg_opt_type"),
      py::arg("leg_strike_offset"), py::arg("leg_dte"), py::arg("leg_contracts"),
      py::arg("group_close_mode"), py::arg("group_entry_root"), py::arg("group_exit_root"),
      py::arg("cond_type"), py::arg("cond_param1"), py::arg("cond_param2"),
      py::arg("cond_child1"), py::arg("cond_child2"), py::arg("r"));

  m.attr("COND_EVERY_N_DAYS")  = (int)COND_EVERY_N_DAYS;
  m.attr("COND_DELTA_BETWEEN") = (int)COND_DELTA_BETWEEN;
  m.attr("COND_IV_ABOVE")      = (int)COND_IV_ABOVE;
  m.attr("COND_IV_BELOW")      = (int)COND_IV_BELOW;
  m.attr("COND_PRICE_ABOVE")   = (int)COND_PRICE_ABOVE;
  m.attr("COND_PRICE_BELOW")   = (int)COND_PRICE_BELOW;
  m.attr("COND_DAY_OF_WEEK")   = (int)COND_DAY_OF_WEEK;
  m.attr("COND_HOLD_TO_EXPIRY")= (int)COND_HOLD_TO_EXPIRY;
  m.attr("COND_PROFIT_PCT")    = (int)COND_PROFIT_PCT;
  m.attr("COND_LOSS_PCT")      = (int)COND_LOSS_PCT;
  m.attr("COND_PROFIT_DOLLARS")= (int)COND_PROFIT_DOLLARS;
  m.attr("COND_LOSS_DOLLARS")  = (int)COND_LOSS_DOLLARS;
  m.attr("COND_DTE_REMAINING") = (int)COND_DTE_REMAINING;
  m.attr("COND_AND")           = (int)COND_AND;
  m.attr("COND_OR")            = (int)COND_OR;
  m.attr("COND_NOT")           = (int)COND_NOT;
  m.attr("COND_TRUE")          = (int)COND_TRUE;
  m.attr("COND_FALSE")         = (int)COND_FALSE;
  m.attr("CLOSE_ALL_TOGETHER") = (int)CLOSE_ALL_TOGETHER;
  m.attr("CLOSE_INDIVIDUALLY") = (int)CLOSE_INDIVIDUALLY;
}
