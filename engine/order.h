#pragma once

#include <optional>
#include <string>
#include <vector>

#include "types.h"

namespace obt {

enum class OrderSide : uint8_t { Buy, Sell };
// Exercise is an order rather than a separate call so it inherits the group
// machinery: it can be submitted atomically with the hedge that replaces it, and
// it is rejected through the same path with the same named reasons.
//
// It is not priced and crosses no spread. AssignmentPolicy::ExplicitExerciseOnly
// was selectable without it, which made the engine settle nothing at all.
enum class OrderType : uint8_t { Market, Limit, Stop, StopLimit, Exercise };
enum class TimeInForce : uint8_t { Day, GoodTilCanceled, FillOrKill };

// When a signal turns into a fill. The default avoids same-bar lookahead: a
// signal computed from bar T's close can only fill at the next bar's open,
// because intrabar path and queue position are unknown from OHLC data.
enum class ExecutionTiming : uint8_t {
    NextBarOpen,
    // Only for reconciling against a reference that fills on the signal bar.
    // Reports as lookahead-contaminated.
    SameBarClose,
};

enum class OrderStatus : uint8_t { Pending, Filled, Rejected, Canceled, Expired };

enum class RejectReason : uint8_t {
    None,
    InsufficientBuyingPower,
    RiskLimitBreached,
    ContractNotTradable,
    AnalyticsRejected,
    StaleMarketData,
    NoMarketData,
    LimitNotSatisfied,
    GroupLegRejected,
    UnconfirmedLineage,
    BrokerDisallowed,
    // The order names a feature the engine does not implement. Refusing is the
    // only honest response: silently approximating it produces a fill, a
    // reconciling ledger, and a wrong answer.
    UnsupportedOrderType,
    UnsupportedInstrumentKind,
    // An exercise the holder is not in a position to make: not long, more
    // contracts than held, or a European contract before expiration.
    NotExercisable,
};

struct Order {
    uint64_t order_id = 0;
    Timestamp submitted_at{};
    ContractVersionId contract_version_id{};
    EquityKind kind = EquityKind::Option;
    // The share symbol, for an equity leg. Ignored for an option leg, which
    // identifies its instrument by contract version.
    std::string symbol;

    // Always positive; direction comes from side.
    int64_t quantity = 0;
    OrderSide side = OrderSide::Buy;
    OrderType type = OrderType::Market;
    TimeInForce time_in_force = TimeInForce::Day;

    std::optional<Money> limit_price;
    std::optional<Money> stop_price;

    // Legs sharing a group id fill together or not at all.
    uint64_t group_id = 0;
    // Marks an order that closes existing exposure, which margin treats
    // differently from one that opens it.
    bool reduce_only = false;

    std::string tag;
};

struct Fill {
    uint64_t order_id = 0;
    uint64_t group_id = 0;
    Timestamp filled_at{};
    ContractVersionId contract_version_id{};
    EquityKind kind = EquityKind::Option;
    OrderSide side = OrderSide::Buy;
    int64_t quantity = 0;

    // The mark before execution cost, and the price actually paid. Keeping both
    // lets a report separate deterministic market-data P&L from the stochastic
    // spread component.
    Money mark{};
    Money fill_price{};
    Money half_spread{};

    Money gross_cash{};   // signed premium flow, excluding fees
    Money fees{};
    Money net_cash{};     // what actually hit the ledger

    Money spread_cost() const {
        return Money::scaled(half_spread.micros, quantity * multiplier);
    }

    int64_t multiplier = 100;
};

struct OrderRejection {
    uint64_t order_id = 0;
    uint64_t group_id = 0;
    Timestamp at{};
    RejectReason reason = RejectReason::None;
    std::string detail;
};

// A set of legs that must fill atomically. Verticals, calendars, diagonals,
// butterflies, condors, straddles, strangles, ratios, collars, and rolls are
// all expressed as one of these.
struct OrderGroup {
    uint64_t group_id = 0;
    std::vector<Order> legs;
    // Reject the whole group if any leg cannot fill. Always true for real
    // multi-leg orders; a broker does not partially execute a spread.
    bool atomic = true;
};

inline const char* to_string(RejectReason r) {
    switch (r) {
        case RejectReason::None: return "none";
        case RejectReason::InsufficientBuyingPower: return "insufficient_buying_power";
        case RejectReason::RiskLimitBreached: return "risk_limit_breached";
        case RejectReason::ContractNotTradable: return "contract_not_tradable";
        case RejectReason::AnalyticsRejected: return "analytics_rejected";
        case RejectReason::StaleMarketData: return "stale_market_data";
        case RejectReason::NoMarketData: return "no_market_data";
        case RejectReason::LimitNotSatisfied: return "limit_not_satisfied";
        case RejectReason::GroupLegRejected: return "group_leg_rejected";
        case RejectReason::UnconfirmedLineage: return "unconfirmed_lineage";
        case RejectReason::BrokerDisallowed: return "broker_disallowed";
        case RejectReason::UnsupportedOrderType: return "unsupported_order_type";
        case RejectReason::UnsupportedInstrumentKind: return "unsupported_instrument_kind";
    }
    return "unknown";
}

} // namespace obt
