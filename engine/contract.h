#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace obt {

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

    OptionType type = OptionType::Call;
    bool is_american = true;

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

    int64_t deliverable_shares_per_contract() const {
        return deliverable_equity_microshares / 1'000'000;
    }

    // Notional the contract controls at a given underlying price.
    Money notional(Money underlying) const {
        return Money{underlying.micros * deliverable_shares_per_contract()};
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
