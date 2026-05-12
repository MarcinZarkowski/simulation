#pragma once

#include "data.cpp"
#include "options_pricing.cpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

// ═══════════════════════════════════════════════════════════════════════
// Pricing
// ═══════════════════════════════════════════════════════════════════════

void reprice_leg(TradeLeg &leg, float stock_price, float iv, float r) {
  if (iv < 0.001f)
    iv = 0.001f; // Safeguard against zero volatility
  if (leg.dte > 0) {
    int N = max(leg.dte, 3);
    string type_str = (leg.opt_type == OPT_CALL) ? "CALL" : "PUT";
    auto [u, d] = get_tree_params(iv, leg.dte, N);
    auto result = price_binomial_tree(stock_price, leg.strike_price, r, leg.dte,
                                      N, u, d, type_str);
    leg.opt_price = result[0];
    leg.delta = result[1];
    leg.gamma = result[2];
    leg.theta = result[3];
    float bump = 0.01f;
    auto [ub, db2] = get_tree_params(iv + bump, leg.dte, N);
    auto rb = price_binomial_tree(stock_price, leg.strike_price, r, leg.dte, N,
                                  ub, db2, type_str);
    leg.vega = (rb[0] - result[0]) / bump;
  } else {
    leg.opt_price = (leg.opt_type == OPT_CALL)
                        ? max(0.0f, stock_price - leg.strike_price)
                        : max(0.0f, leg.strike_price - stock_price);
    leg.delta = (leg.opt_price > 0) ? 1.0f : 0.0f;
    leg.gamma = leg.theta = leg.vega = 0;
  }
}

vector<TradeLeg> build_trade_legs(const Strategy &strategy,
                                  const vector<Leg> &blueprints,
                                  const MarketState &market, float r) {
  float atm = round_to_grid(market.close);
  float step = strike_step(market.close);
  vector<TradeLeg> legs;
  for (int idx : strategy.leg_indices) {
    const auto &bp = blueprints[idx];
    TradeLeg tl;
    tl.opt_type = bp.opt_type;
    tl.buy = bp.buy;

    tl.strike_price = atm + bp.strike_offset * step;

    tl.dte = bp.dte;
    tl.original_dte = bp.dte;
    tl.contracts = bp.contracts;
    tl.entry_stock_price = market.close;
    tl.entry_vol = market.iv;
    tl.entry_day = market.day_index;
    reprice_leg(tl, market.close, market.iv, r);
    tl.entry_opt_price = tl.opt_price;
    legs.push_back(tl);
  }
  return legs;
}

// Affordability

// Long legs in a group "cover" short legs of the same type.
// Only truly uncovered shorts need real shares or cash collateral.
// The check also ensures the net debit is affordable.

bool group_can_afford(const vector<TradeLeg> &legs, const Account &account,
                      float close_price) {
  float net_debit = 0;
  int long_calls = 0, short_calls = 0;

  for (const auto &leg : legs) {
    int mult = 100 * leg.contracts;
    if (leg.buy) {
      net_debit += leg.opt_price * mult;
      if (leg.opt_type == OPT_CALL)
        long_calls += leg.contracts;
    } else {
      net_debit -= leg.opt_price * mult;
      if (leg.opt_type == OPT_CALL)
        short_calls += leg.contracts;
    }
  }

  // Check net debit is affordable
  if (net_debit > account.available_cash())
    return false;

  // Check uncovered short calls (covered by shares)
  int uncovered_calls = max(0, short_calls - long_calls);
  if (uncovered_calls * 100 > account.available_shares())
    return false;

  // Calculate required margin for this group using spread logic
  OpenPosition dummy;
  dummy.legs = legs;
  float required_margin = calculate_position_margin(dummy, close_price);

  // Remaining cash after debit must cover the margin
  float remaining_cash = account.available_cash() - max(0.0f, net_debit);
  if (required_margin > remaining_cash)
    return false;

  return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Position lifecycle
// ═══════════════════════════════════════════════════════════════════════

float commit_position_to_account(OpenPosition &pos, Account &account) {
  float net_premium = 0;
  for (const auto &leg : pos.legs) {
    int mult = 100 * leg.contracts;
    if (leg.buy) {
      account.balance -= leg.opt_price * mult;
      pos.total_entry_cost += leg.opt_price * mult;
      net_premium -= leg.opt_price * mult;
    } else {
      account.balance += leg.opt_price * mult;
      pos.total_entry_cost -= leg.opt_price * mult;
      net_premium += leg.opt_price * mult;
    }
  }
  return net_premium;
}

float close_entire_position(OpenPosition &pos, Account &account,
                            float stock_price) {
  float pnl = 0;
  for (auto &leg : pos.legs) {
    int mult = 100 * leg.contracts;

    // Cash settlement (early close, long options, or OTM)
    if (leg.buy) {
      pnl += (leg.opt_price - leg.entry_opt_price) * mult;
      account.balance += leg.opt_price * mult;
    } else {
      pnl += (leg.entry_opt_price - leg.opt_price) * mult;
      account.balance -= leg.opt_price * mult;
    }
  }
  return pnl;
}

struct PartialCloseResult {
  float pnl;
  int num_closed;
  vector<TradeLeg> remaining;
};

PartialCloseResult close_expired_legs_only(OpenPosition &pos, Account &account,
                                           float stock_price) {
  float pnl = 0;
  int closed = 0;
  vector<TradeLeg> remaining;
  for (auto &leg : pos.legs) {
    if (leg.dte <= 0) {
      int mult = 100 * leg.contracts;

      if (leg.buy) {
        pnl += (leg.opt_price - leg.entry_opt_price) * mult;
        account.balance += leg.opt_price * mult;
      } else {
        pnl += (leg.entry_opt_price - leg.opt_price) * mult;
        account.balance -= leg.opt_price * mult;
      }
      closed++;
    } else {
      remaining.push_back(leg);
    }
  }
  return {pnl, closed, remaining};
}
