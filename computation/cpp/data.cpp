#pragma once

#include <climits>
#include <limits>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <algorithm>

namespace py = pybind11;
using namespace std;

// ═══════════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════════

enum OptType { OPT_CALL = 0, OPT_PUT = 1 };
enum CloseMode { CLOSE_ALL_TOGETHER = 0, CLOSE_INDIVIDUALLY = 1 };

enum CondType {
  // Entry conditions
  COND_EVERY_N_DAYS = 0,
  COND_DELTA_BETWEEN = 1,
  COND_IV_ABOVE = 2,
  COND_IV_BELOW = 3,
  COND_PRICE_ABOVE = 4,
  COND_PRICE_BELOW = 5,
  COND_DAY_OF_WEEK = 6,

  // Exit conditions
  COND_HOLD_TO_EXPIRY = 10,
  COND_PROFIT_PCT = 11,
  COND_LOSS_PCT = 12,
  COND_PROFIT_DOLLARS = 13,
  COND_LOSS_DOLLARS = 14,
  COND_DTE_REMAINING = 15,

  // Combinators
  COND_AND = 20,
  COND_OR = 21,
  COND_NOT = 22,
  COND_TRUE = 23,
  COND_FALSE = 24,
};

// ═══════════════════════════════════════════════════════════════════════
// Core structs
// ═══════════════════════════════════════════════════════════════════════

struct Condition {
  CondType type;
  float param1, param2;
  int child1, child2; // indices into conditions vector (-1 = unused)
};

struct MarketState {
  float open, high, low, close, volume, iv;
  int day_index;
  int day_of_week; // 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
};

struct Account {
  float balance;
  int shares;

  // Collateral locked by open short positions
  float locked_collateral = 0;
  int locked_shares = 0;

  float average_stock_cost = 0;

  float add_shares(int count, float price) {
    if (count <= 0) return 0.0f;
    float realized_pnl = 0;
    
    if (shares < 0) {
        int covering = std::min(count, -shares);
        realized_pnl = covering * (average_stock_cost - price);
        shares += covering;
        count -= covering;
        if (shares == 0) average_stock_cost = 0;
    }
    
    if (count > 0) {
        float total_cost = (shares * average_stock_cost) + (count * price);
        shares += count;
        average_stock_cost = total_cost / (float)shares;
    }
    return realized_pnl;
  }

  float remove_shares(int count, float price) {
    if (count <= 0) return 0.0f;
    float realized_pnl = 0;
    
    if (shares > 0) {
        int selling = std::min(count, shares);
        realized_pnl += selling * (price - average_stock_cost);
        shares -= selling;
        count -= selling;
        if (shares == 0) average_stock_cost = 0;
    }
    
    if (count > 0) {
        float total_sale_value = ((-shares) * average_stock_cost) + (count * price);
        shares -= count;
        average_stock_cost = total_sale_value / (float)(-shares);
    }
    return realized_pnl;
  }

  int total_trades = 0;
  int winning_trades = 0;
  int total_positions_opened = 0;
  float total_win_amt = 0;
  float total_loss_amt = 0;
  float best_trade = numeric_limits<float>::lowest();
  float worst_trade = numeric_limits<float>::max();
  vector<float> closed_pnls;

  void record_trade(float pnl) {
    closed_pnls.push_back(pnl);
    total_trades++;
    if (pnl > 0) {
      winning_trades++;
      total_win_amt += pnl;
    } else if (pnl < 0) {
      total_loss_amt += pnl;
    }
    if (pnl > best_trade)
      best_trade = pnl;
    if (pnl < worst_trade)
      worst_trade = pnl;
  }

  float avg_win() const { return winning_trades > 0 ? total_win_amt / winning_trades : 0; }
  float avg_loss() const { 
    int losing_trades = total_trades - winning_trades;
    return losing_trades > 0 ? total_loss_amt / losing_trades : 0; 
  }

  float available_cash() const { return balance - locked_collateral; }
  int available_shares() const { return std::max(0, shares - locked_shares); }

  float total_pnl() const {
    float s = 0;
    for (float p : closed_pnls)
      s += p;
    return s;
  }
  float avg_pnl() const {
    return closed_pnls.empty() ? 0.0f : total_pnl() / (float)closed_pnls.size();
  }
  float win_rate() const {
    return total_trades > 0 ? (float)winning_trades / total_trades : 0.0f;
  }
};

struct Leg {
  OptType opt_type;
  bool buy;
  int strike_offset;
  int dte;
  int contracts;
  int group_id;
};

struct Strategy {
  int group_id;
  CloseMode close_mode;
  int entry_root;
  int exit_root;
  vector<int> leg_indices;
  int last_entry_day = -9999;
};

struct TradeLeg {
  OptType opt_type;
  bool buy;
  float strike_price;
  int dte, original_dte, contracts;
  float opt_price = 0;
  float delta = 0, gamma = 0, theta = 0, vega = 0;
  float entry_opt_price = 0;
  float entry_stock_price = 0;
  float entry_vol = 0;
  int entry_day = 0;
};

struct OpenPosition {
  int group_id;
  CloseMode close_mode;
  int exit_root;
  vector<TradeLeg> legs;
  float total_entry_cost = 0;
  int entry_day = 0;
};

struct GroupState {
  float unrealized_pnl = 0;
  float max_profit = 0;
  float max_loss = 0;
  float avg_delta = 0;
  int min_dte = INT_MAX;
  int days_held = 0;
};


// ═══════════════════════════════════════════════════════════════════════
// Decode logic
// ═══════════════════════════════════════════════════════════════════════

vector<Condition> decode_conditions(py::array_t<int> cond_type,
                                    py::array_t<float> cond_param1,
                                    py::array_t<float> cond_param2,
                                    py::array_t<int> cond_child1,
                                    py::array_t<int> cond_child2) {
  auto t = cond_type.unchecked<1>();
  auto p1 = cond_param1.unchecked<1>();
  auto p2 = cond_param2.unchecked<1>();
  auto c1 = cond_child1.unchecked<1>();
  auto c2 = cond_child2.unchecked<1>();
  int n = (int)cond_type.shape(0);
  vector<Condition> out(n);
  for (int i = 0; i < n; i++)
    out[i] = {static_cast<CondType>(t(i)), p1(i), p2(i), c1(i), c2(i)};
  return out;
}

vector<Leg> decode_legs(py::array_t<int> leg_group_id, py::array_t<int> leg_buy,
                        py::array_t<int> leg_opt_type,
                        py::array_t<int> leg_strike_offset,
                        py::array_t<int> leg_dte,
                        py::array_t<int> leg_contracts) {
  auto g = leg_group_id.unchecked<1>();
  auto b = leg_buy.unchecked<1>();
  auto t = leg_opt_type.unchecked<1>();
  auto o = leg_strike_offset.unchecked<1>();
  auto d = leg_dte.unchecked<1>();
  auto c = leg_contracts.unchecked<1>();
  int n = (int)leg_group_id.shape(0);
  vector<Leg> out(n);
  for (int i = 0; i < n; i++)
    out[i] = {static_cast<OptType>(t(i)), (bool)b(i), o(i), d(i), c(i), g(i)};
  return out;
}

vector<Strategy> decode_strategies(py::array_t<int> group_close_mode,
                                   py::array_t<int> group_entry_root,
                                   py::array_t<int> group_exit_root,
                                   const vector<Leg> &legs) {
  auto cm = group_close_mode.unchecked<1>();
  auto er = group_entry_root.unchecked<1>();
  auto xr = group_exit_root.unchecked<1>();
  int ng = (int)group_close_mode.shape(0);
  vector<Strategy> out(ng);
  for (int g = 0; g < ng; g++) {
    out[g].group_id = g;
    out[g].close_mode = static_cast<CloseMode>(cm(g));
    out[g].entry_root = er(g);
    out[g].exit_root = xr(g);
    out[g].last_entry_day = -9999;
  }
  for (int i = 0; i < (int)legs.size(); i++) {
    int gid = legs[i].group_id;
    if (gid >= 0 && gid < ng)
      out[gid].leg_indices.push_back(i);
  }
  return out;
}

bool evaluate_condition(const vector<Condition> &conditions, int idx,
                        const GroupState &gs, const MarketState &ms) {
  if (idx < 0 || idx >= (int)conditions.size())
    return false;
  const auto &c = conditions[idx];

  switch (c.type) {
  case COND_EVERY_N_DAYS:
    return (ms.day_index % (int)c.param1) == 0;
  case COND_DELTA_BETWEEN:
    return gs.avg_delta >= c.param1 && gs.avg_delta <= c.param2;
  case COND_IV_ABOVE:
    return ms.iv > c.param1;
  case COND_IV_BELOW:
    return ms.iv < c.param1;
  case COND_PRICE_ABOVE:
    return ms.close > c.param1;
  case COND_PRICE_BELOW:
    return ms.close < c.param1;
  case COND_DAY_OF_WEEK:
    return ms.day_of_week == (int)c.param1;
  case COND_HOLD_TO_EXPIRY:
    return false;
  case COND_PROFIT_PCT:
    return gs.max_profit > 0 && gs.unrealized_pnl >= c.param1 * gs.max_profit;
  case COND_LOSS_PCT:
    return gs.max_loss > 0 && gs.unrealized_pnl <= -(c.param1 * gs.max_loss);
  case COND_PROFIT_DOLLARS:
    return gs.unrealized_pnl >= c.param1;
  case COND_LOSS_DOLLARS:
    return gs.unrealized_pnl <= -c.param1;
  case COND_DTE_REMAINING:
    return gs.min_dte <= (int)c.param1;
  case COND_AND:
    return evaluate_condition(conditions, c.child1, gs, ms) &&
           evaluate_condition(conditions, c.child2, gs, ms);
  case COND_OR:
    return evaluate_condition(conditions, c.child1, gs, ms) ||
           evaluate_condition(conditions, c.child2, gs, ms);
  case COND_NOT:
    return !evaluate_condition(conditions, c.child1, gs, ms);
  case COND_TRUE:
    return true;
  case COND_FALSE:
    return false;
  default:
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Margin & Collateral Helpers
// ═══════════════════════════════════════════════════════════════════════

float calculate_position_margin(const OpenPosition &pos, float close_price) {
  vector<float> short_puts, long_puts, short_calls, long_calls;
  
  for (const auto &leg : pos.legs) {
    for (int i = 0; i < leg.contracts; ++i) {
      if (leg.opt_type == OPT_PUT) {
        if (leg.buy) long_puts.push_back(leg.strike_price);
        else        short_puts.push_back(leg.strike_price);
      } else {
        if (leg.buy) long_calls.push_back(leg.strike_price);
        else        short_calls.push_back(leg.strike_price);
      }
    }
  }

  // Calculate Put Margin (Vertical Put Spreads)
  sort(short_puts.rbegin(), short_puts.rend());
  sort(long_puts.rbegin(), long_puts.rend());
  float put_margin = 0;
  size_t lp_idx = 0;
  for (float s : short_puts) {
    while (lp_idx < long_puts.size() && long_puts[lp_idx] >= s) lp_idx++;
    if (lp_idx < long_puts.size()) {
      put_margin += (s - long_puts[lp_idx]) * 100.0f;
      lp_idx++;
    } else {
      put_margin += s * 100.0f;
    }
  }

  // Calculate Call Margin (Vertical Call Spreads)
  sort(short_calls.begin(), short_calls.end());
  sort(long_calls.begin(), long_calls.end());
  float call_margin = 0;
  size_t lc_idx = 0;
  for (float s : short_calls) {
    while (lc_idx < long_calls.size() && long_calls[lc_idx] <= s) lc_idx++;
    if (lc_idx < long_calls.size()) {
      call_margin += (long_calls[lc_idx] - s) * 100.0f;
      lc_idx++;
    } else {
      call_margin += close_price * 100.0f;
    }
  }

  return max(put_margin, call_margin);
}

GroupState compute_group_state(const OpenPosition &pos, int current_day, float close_price) {
  GroupState gs{};
  gs.min_dte = INT_MAX;
  gs.days_held = current_day - pos.entry_day;

  int total_contracts = 0;
  for (const auto &leg : pos.legs) {
    int mult = 100 * leg.contracts;
    float mark = leg.opt_price * mult;
    float entry = leg.entry_opt_price * mult;
    gs.unrealized_pnl += leg.buy ? (mark - entry) : (entry - mark);
    gs.avg_delta += leg.delta * leg.contracts;
    total_contracts += leg.contracts;
    if (leg.dte < gs.min_dte)
      gs.min_dte = leg.dte;
  }
  if (total_contracts > 0)
    gs.avg_delta /= total_contracts;

  float margin = calculate_position_margin(pos, close_price);
  
  if (pos.total_entry_cost < 0) {
      gs.max_profit = -pos.total_entry_cost;
      gs.max_loss   = margin; 
  } else {
      gs.max_profit = 1e9f;
      gs.max_loss   = pos.total_entry_cost;
  }
  return gs;
}

float compute_locked_collateral(const vector<OpenPosition> &positions, float close_price) {
  float total = 0;
  for (const auto &pos : positions)
    total += calculate_position_margin(pos, close_price);
  return total;
}

int compute_locked_shares(const vector<OpenPosition> &positions) {
  int total = 0;
  for (const auto &pos : positions) {
    int long_calls = 0, short_calls = 0;
    for (const auto &leg : pos.legs) {
      if (leg.opt_type == OPT_CALL) {
        if (leg.buy) long_calls += leg.contracts;
        else        short_calls += leg.contracts;
      }
    }
    int uncovered = max(0, short_calls - long_calls);
    total += uncovered * 100;
  }
  return total;
}
