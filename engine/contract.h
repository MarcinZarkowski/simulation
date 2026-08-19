#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace obt {

// Where a version's terms came from, carried verbatim from the pipeline's
// terms_provenance column.
//
// PointInTime means the terms were observed in a snapshot taken at the time they
// applied. Backfilled means they were copied from a LATER snapshot, so they
// describe what the contract turned out to be, not what it was known to be. The
// pipeline flags this precisely so a consumer can refuse it; the engine used to
// discard the flag.
enum class TermsProvenance : uint8_t { PointInTime, Backfilled, Unknown };

// How an in-the-money contract resolves at expiration.
//
// Equity and ETF options deliver shares. Index options (SPX, XSP, RUT, NDX, VIX)
// pay the intrinsic in cash and never touch a share position -- delivering the
// index is not a thing anyone can do.
enum class SettlementStyle : uint8_t { PhysicalDelivery, CashSettlement };

// One set of contract terms valid over an explicit interval. Mirrors the
// pipeline's option_contract_version rows so the engine never has to re-derive
// pricing terms from raw deliverables.
struct OptionContractVersion {
    ContractVersionId id{};
    InstrumentId instrument_id{};
    std::string symbol;
    std::string underlying_symbol;

    Timestamp valid_from{};
    Timestamp valid_to = Timestamp::never();
    Timestamp expiration{};

    // When these terms became knowable, as distinct from when they took effect.
    // An OCC adjustment memo is published after the fact, so a backtest that
    // acts on the adjusted terms at the effective instant is using information
    // it did not have.
    Timestamp source_available_at{};
    TermsProvenance terms_provenance = TermsProvenance::Unknown;

    OptionType type = OptionType::Call;

    // American contracts can be exercised, and therefore assigned, before
    // expiration. European ones cannot: SPX and most cash-settled index series are
    // European, so a dividend-driven early assignment on one is not conservative,
    // it is impossible.
    bool is_american = true;
    SettlementStyle settlement_style = SettlementStyle::PhysicalDelivery;

    // Last instant a new position may be opened. Equal to expiration for a
    // PM-settled contract. An AM-settled index series stops trading the business
    // day BEFORE expiration and settles against the next morning's opening prints,
    // so the two differ and a bar-based feed that ignores the distinction lets a
    // strategy trade a contract that no longer exists.
    Timestamp last_trade_at = Timestamp::never();

    // Listed strike, and the strike the pricing transform actually uses. They
    // differ for an adjusted contract whose deliverable is not 100 shares.
    Money strike{};
    Money pricing_strike{};

    // Shares per contract for quoting purposes. 100 for a standard contract.
    int64_t quote_multiplier = 100;

    // Deliverable on exercise: this many shares plus this much fixed cash.
    // Stored in millionths of a share so fractional deliverables stay exact.
    int64_t deliverable_equity_microshares = 100'000'000;
    Money deliverable_cash{};

    bool is_adjusted = false;
    bool tradable_for_new_positions = true;
    bool analytics_supported = true;

    bool covers(Timestamp t) const {
        return valid_from <= t && (valid_to.is_never() || t < valid_to);
    }

    // Whether a NEW position may be opened at this instant. Closing an existing
    // one is governed by covers() alone.
    bool tradable_at(Timestamp t) const {
        return last_trade_at.is_never() ? t < expiration : t <= last_trade_at;
    }

    bool is_cash_settled() const { return settlement_style == SettlementStyle::CashSettlement; }

    // Whether the terms were available to a participant at this moment.
    bool known_at(Timestamp t) const { return source_available_at <= t; }

    // Terms not positively established as point-in-time, for a contract whose
    // economics an adjustment changed. For an unadjusted contract a later snapshot
    // reports the same listed strike and deliverable, so backfilling it is not
    // hindsight; for an adjusted one it is exactly hindsight. Unknown provenance
    // counts as unjustified rather than fine, which is the same fail-closed
    // posture the lineage gate takes.
    bool terms_inferred_from_future() const {
        return is_adjusted && terms_provenance != TermsProvenance::PointInTime;
    }

    int64_t deliverable_shares_per_contract() const {
        return deliverable_equity_microshares / 1'000'000;
    }

    // True when the deliverable is not a whole number of shares. OCC settles the
    // fraction in cash-in-lieu, which this engine has no primitive for, so such a
    // contract is refused at settlement rather than silently truncated.
    bool has_fractional_deliverable() const {
        return deliverable_equity_microshares % 1'000'000 != 0;
    }

    // What the holder pays on exercise of a call, or receives on exercise of a
    // put: the LISTED strike times the QUOTE multiplier.
    //
    // Not strike times the delivered share count. For an adjusted contract those
    // differ, and the pipeline documents the payoff as
    //   max(A * S_T + C - K * M, 0)
    // with A delivered shares, C fixed cash, K the listed strike and M the quote
    // multiplier. Using K * A instead fabricated value: a 50-share deliverable at
    // K=100 with spot 110 paid $5,000 for $5,500 of stock on a contract whose true
    // payoff was zero.
    Money aggregate_exercise_price() const {
        return Money::scaled(strike.micros, quote_multiplier);
    }

    // Total value delivered per contract at a given underlying price.
    Money delivered_value(Money underlying) const {
        return Money::scaled(underlying.micros, deliverable_shares_per_contract()) + deliverable_cash;
    }

    // Intrinsic value per contract, following the deliverable rather than
    // assuming a 100-share standard contract.
    Money payoff_at(Money underlying) const {
        const Money delivered = delivered_value(underlying);
        const Money aggregate = aggregate_exercise_price();
        return type == OptionType::Call ? max_money(delivered - aggregate, Money::zero())
                                        : max_money(aggregate - delivered, Money::zero());
    }

    // Notional the contract controls at a given underlying price.
    Money notional(Money underlying) const {
        return Money::scaled(underlying.micros, deliverable_shares_per_contract());
    }
};

// Point-in-time market bar for one instrument.
struct MarketBar {
    Timestamp timestamp{};
    ContractVersionId contract_version_id{};
    Money open{}, high{}, low{}, close{}, vwap{};
    // The pipeline's chosen mark. Preferred over close so the engine and the
    // pipeline agree on what a contract was worth.
    Money valuation_price{};
    int64_t volume = 0;
    int64_t trade_count = 0;
    bool stale = false;
    bool analytics_valid = false;
};

// Point-in-time bar for a share. Separate from MarketBar because a share has no
// contract version, no analytics validity, and a penny grid at every price.
struct EquityBar {
    Timestamp timestamp{};
    std::string symbol;
    Money open{}, high{}, low{}, close{}, vwap{};
    int64_t volume = 0;
    int64_t trade_count = 0;
    bool stale = false;

    // The price an order executes against. Next-bar-open timing reads the open;
    // same-bar-close reads the close. VWAP is not used, because a fill at the
    // period's volume-weighted average is a price no single order could have got.
    Money execution_price(bool use_open) const {
        const Money chosen = use_open ? open : close;
        return chosen.micros > 0 ? chosen : close;
    }
};

// Analytics for one contract at one timestamp. Every field is advisory: the
// engine gates on the validity flags before letting a strategy see them.
struct OptionAnalytics {
    Timestamp timestamp{};
    ContractVersionId contract_version_id{};
    double implied_volatility = 0.0;
    double delta = 0.0, gamma = 0.0, theta = 0.0, vega = 0.0, rho = 0.0;
    Money theoretical_contract_value{};
    bool valid = false;
};

// How a parent contract becomes one or more child contracts. Only ever applied
// when the pipeline marked it OCC-confirmed.
enum class AdjustmentType : uint8_t {
    WholeShareSplit,
    FractionalSplit,
    ReverseSplit,
    RootChange,
    DeliverableChange,
    CashMerger,
    StockAndCashMerger,
    SpinOff,
    AcceleratedExpiration,
    Unknown,
};

enum class SettlementRule : uint8_t {
    PhysicalDelivery,
    CashSettlement,
    AdjustedDeliverable,
    Unknown,
};

struct CorporateActionTransition {
    uint64_t lineage_event_id = 0;
    Timestamp effective_at{};
    Timestamp source_available_at{};
    ContractVersionId parent_version_id{};
    ContractVersionId child_version_id{};
    // Quantity conversion: parent_contracts held become child_contracts held.
    int64_t parent_contracts = 0;
    int64_t child_contracts = 0;
    AdjustmentType type = AdjustmentType::Unknown;
    SettlementRule settlement_rule = SettlementRule::Unknown;
    bool occ_confirmed = false;

    // A transition is only safe to apply when the source confirmed it and gave
    // us a quantity conversion. Anything else must halt the position rather
    // than silently guess a conversion.
    bool is_actionable() const {
        return occ_confirmed && parent_contracts > 0 && child_contracts > 0;
    }
};

// Registry of contract versions, queryable by id or by (symbol, time).
class ContractRegistry {
public:
    void add(const OptionContractVersion& v) {
        by_id_[v.id.value] = v;
        by_symbol_[v.symbol].push_back(v.id);
    }

    const OptionContractVersion* find(ContractVersionId id) const {
        auto it = by_id_.find(id.value);
        return it == by_id_.end() ? nullptr : &it->second;
    }

    // Version of a symbol in force at a moment. Intervals are closed by the
    // pipeline so at most one matches.
    const OptionContractVersion* resolve(const std::string& symbol, Timestamp t) const {
        auto it = by_symbol_.find(symbol);
        if (it == by_symbol_.end()) return nullptr;
        for (ContractVersionId id : it->second) {
            const auto* v = find(id);
            if (v && v->covers(t)) return v;
        }
        return nullptr;
    }

    size_t size() const { return by_id_.size(); }

    void clear() {
        by_id_.clear();
        by_symbol_.clear();
    }

private:
    std::unordered_map<uint64_t, OptionContractVersion> by_id_;
    std::unordered_map<std::string, std::vector<ContractVersionId>> by_symbol_;
};

} // namespace obt
