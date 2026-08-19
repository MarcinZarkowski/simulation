#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.h"

namespace py = pybind11;
using namespace obt;

namespace {

// Money crosses the boundary as a float in dollars for ergonomics, but is
// integral everywhere inside the engine. Exposing `micros` as well lets tests
// assert exact values without going through a float.
Money money_from(double dollars) { return Money::from_double(dollars); }

std::string assignment_policy_name(AssignmentPolicy p) {
    switch (p) {
        case AssignmentPolicy::ExpirationOnly: return "expiration_only";
        case AssignmentPolicy::ExplicitExerciseOnly: return "explicit_exercise_only";
        case AssignmentPolicy::AutomaticITMExercise: return "automatic_itm_exercise";
        case AssignmentPolicy::ConservativeEarlyAssignment: return "conservative_early_assignment";
    }
    return "unknown";
}

} // namespace

PYBIND11_MODULE(obt_engine, m) {
    m.doc() = "Deterministic options portfolio engine with Monte Carlo execution cost";

    // ---------------------------------------------------------------- money
    py::class_<Money>(m, "Money")
        .def(py::init<>())
        .def_static("from_dollars", &money_from, py::arg("dollars"))
        .def_static("from_micros", [](int64_t u) { return Money{u}; }, py::arg("micros"))
        .def_readonly("micros", &Money::micros)
        .def("to_dollars", &Money::to_double)
        .def("__repr__", [](Money m) {
            return "Money(" + std::to_string(m.to_double()) + ")";
        })
        .def("__eq__", [](Money a, Money b) { return a == b; })
        .def("__float__", &Money::to_double);

    py::class_<Timestamp>(m, "Timestamp")
        .def(py::init<>())
        .def_static("from_ns", [](int64_t ns) { return Timestamp{ns}; }, py::arg("epoch_ns"))
        .def_readonly("epoch_ns", &Timestamp::epoch_ns);

    // ---------------------------------------------------------------- enums
    py::enum_<OptionType>(m, "OptionType")
        .value("CALL", OptionType::Call)
        .value("PUT", OptionType::Put);

    py::enum_<EquityKind>(m, "EquityKind")
        .value("OPTION", EquityKind::Option)
        .value("EQUITY", EquityKind::Equity);

    py::enum_<OrderSide>(m, "OrderSide")
        .value("BUY", OrderSide::Buy)
        .value("SELL", OrderSide::Sell);

    py::enum_<OrderType>(m, "OrderType")
        .value("MARKET", OrderType::Market)
        .value("LIMIT", OrderType::Limit)
        .value("STOP", OrderType::Stop)
        .value("STOP_LIMIT", OrderType::StopLimit);

    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("DAY", TimeInForce::Day)
        .value("GTC", TimeInForce::GoodTilCanceled)
        .value("FOK", TimeInForce::FillOrKill);

    py::enum_<ExecutionTiming>(m, "ExecutionTiming")
        .value("NEXT_BAR_OPEN", ExecutionTiming::NextBarOpen)
        .value("SAME_BAR_CLOSE", ExecutionTiming::SameBarClose);

    py::enum_<AssignmentPolicy>(m, "AssignmentPolicy")
        .value("EXPIRATION_ONLY", AssignmentPolicy::ExpirationOnly)
        .value("EXPLICIT_EXERCISE_ONLY", AssignmentPolicy::ExplicitExerciseOnly)
        .value("AUTOMATIC_ITM_EXERCISE", AssignmentPolicy::AutomaticITMExercise)
        .value("CONSERVATIVE_EARLY_ASSIGNMENT", AssignmentPolicy::ConservativeEarlyAssignment);

    py::enum_<MarginModelKind>(m, "MarginModel")
        .value("CASH_ACCOUNT", MarginModelKind::CashAccount)
        .value("REG_T", MarginModelKind::RegT)
        .value("ROBINHOOD", MarginModelKind::Robinhood)
        .value("PORTFOLIO_APPROX", MarginModelKind::PortfolioApprox);

    py::enum_<SpreadModelKind>(m, "SpreadModelKind")
        .value("ZERO", SpreadModelKind::Zero)
        .value("CONSTANT_CENTS", SpreadModelKind::ConstantCents)
        .value("PROPORTIONAL_BPS", SpreadModelKind::ProportionalBps)
        .value("LOGNORMAL", SpreadModelKind::Lognormal)
        .value("CONDITIONAL_LOGNORMAL", SpreadModelKind::ConditionalLognormal)
        .value("EMPIRICAL", SpreadModelKind::Empirical);

    py::enum_<AdjustmentType>(m, "AdjustmentType")
        .value("WHOLE_SHARE_SPLIT", AdjustmentType::WholeShareSplit)
        .value("FRACTIONAL_SPLIT", AdjustmentType::FractionalSplit)
        .value("REVERSE_SPLIT", AdjustmentType::ReverseSplit)
        .value("ROOT_CHANGE", AdjustmentType::RootChange)
        .value("DELIVERABLE_CHANGE", AdjustmentType::DeliverableChange)
        .value("CASH_MERGER", AdjustmentType::CashMerger)
        .value("STOCK_AND_CASH_MERGER", AdjustmentType::StockAndCashMerger)
        .value("SPIN_OFF", AdjustmentType::SpinOff)
        .value("ACCELERATED_EXPIRATION", AdjustmentType::AcceleratedExpiration)
        .value("UNKNOWN", AdjustmentType::Unknown);

    py::enum_<SettlementRule>(m, "SettlementRule")
        .value("PHYSICAL_DELIVERY", SettlementRule::PhysicalDelivery)
        .value("CASH_SETTLEMENT", SettlementRule::CashSettlement)
        .value("ADJUSTED_DELIVERABLE", SettlementRule::AdjustedDeliverable)
        .value("UNKNOWN", SettlementRule::Unknown);

    py::enum_<RejectReason>(m, "RejectReason")
        .value("NONE", RejectReason::None)
        .value("INSUFFICIENT_BUYING_POWER", RejectReason::InsufficientBuyingPower)
        .value("RISK_LIMIT_BREACHED", RejectReason::RiskLimitBreached)
        .value("CONTRACT_NOT_TRADABLE", RejectReason::ContractNotTradable)
        .value("ANALYTICS_REJECTED", RejectReason::AnalyticsRejected)
        .value("STALE_MARKET_DATA", RejectReason::StaleMarketData)
        .value("NO_MARKET_DATA", RejectReason::NoMarketData)
        .value("LIMIT_NOT_SATISFIED", RejectReason::LimitNotSatisfied)
        .value("GROUP_LEG_REJECTED", RejectReason::GroupLegRejected)
        .value("UNCONFIRMED_LINEAGE", RejectReason::UnconfirmedLineage)
        .value("BROKER_DISALLOWED", RejectReason::BrokerDisallowed)
        .value("UNSUPPORTED_ORDER_TYPE", RejectReason::UnsupportedOrderType)
        .value("UNSUPPORTED_INSTRUMENT_KIND", RejectReason::UnsupportedInstrumentKind);

    // ------------------------------------------------------------- contract
    py::class_<OptionContractVersion>(m, "OptionContractVersion")
        .def(py::init<>())
        .def_property("id",
            [](const OptionContractVersion& c) { return c.id.value; },
            [](OptionContractVersion& c, uint64_t v) { c.id = ContractVersionId{v}; })
        .def_property("instrument_id",
            [](const OptionContractVersion& c) { return c.instrument_id.value; },
            [](OptionContractVersion& c, uint64_t v) { c.instrument_id = InstrumentId{v}; })
        .def_readwrite("symbol", &OptionContractVersion::symbol)
        .def_readwrite("underlying_symbol", &OptionContractVersion::underlying_symbol)
        .def_property("valid_from",
            [](const OptionContractVersion& c) { return c.valid_from.epoch_ns; },
            [](OptionContractVersion& c, int64_t v) { c.valid_from = Timestamp{v}; })
        .def_property("valid_to",
            [](const OptionContractVersion& c) { return c.valid_to.epoch_ns; },
            [](OptionContractVersion& c, int64_t v) { c.valid_to = Timestamp{v}; })
        .def_property("expiration",
            [](const OptionContractVersion& c) { return c.expiration.epoch_ns; },
            [](OptionContractVersion& c, int64_t v) { c.expiration = Timestamp{v}; })
        .def_readwrite("type", &OptionContractVersion::type)
        .def_readwrite("is_american", &OptionContractVersion::is_american)
        .def_property("strike",
            [](const OptionContractVersion& c) { return c.strike.to_double(); },
            [](OptionContractVersion& c, double v) { c.strike = money_from(v); })
        .def_property("pricing_strike",
            [](const OptionContractVersion& c) { return c.pricing_strike.to_double(); },
            [](OptionContractVersion& c, double v) { c.pricing_strike = money_from(v); })
        .def_readwrite("quote_multiplier", &OptionContractVersion::quote_multiplier)
        .def_readwrite("deliverable_equity_microshares",
                       &OptionContractVersion::deliverable_equity_microshares)
        .def_property("deliverable_cash",
            [](const OptionContractVersion& c) { return c.deliverable_cash.to_double(); },
            [](OptionContractVersion& c, double v) { c.deliverable_cash = money_from(v); })
        .def_readwrite("is_adjusted", &OptionContractVersion::is_adjusted)
        .def_readwrite("tradable_for_new_positions",
                       &OptionContractVersion::tradable_for_new_positions)
        .def_readwrite("analytics_supported", &OptionContractVersion::analytics_supported);

    py::class_<MarketBar>(m, "MarketBar")
        .def(py::init<>())
        .def_property("timestamp",
            [](const MarketBar& b) { return b.timestamp.epoch_ns; },
            [](MarketBar& b, int64_t v) { b.timestamp = Timestamp{v}; })
        .def_property("contract_version_id",
            [](const MarketBar& b) { return b.contract_version_id.value; },
            [](MarketBar& b, uint64_t v) { b.contract_version_id = ContractVersionId{v}; })
        .def_property("open", [](const MarketBar& b) { return b.open.to_double(); },
                      [](MarketBar& b, double v) { b.open = money_from(v); })
        .def_property("high", [](const MarketBar& b) { return b.high.to_double(); },
                      [](MarketBar& b, double v) { b.high = money_from(v); })
        .def_property("low", [](const MarketBar& b) { return b.low.to_double(); },
                      [](MarketBar& b, double v) { b.low = money_from(v); })
        .def_property("close", [](const MarketBar& b) { return b.close.to_double(); },
                      [](MarketBar& b, double v) { b.close = money_from(v); })
        .def_property("vwap", [](const MarketBar& b) { return b.vwap.to_double(); },
                      [](MarketBar& b, double v) { b.vwap = money_from(v); })
        .def_property("valuation_price",
            [](const MarketBar& b) { return b.valuation_price.to_double(); },
            [](MarketBar& b, double v) { b.valuation_price = money_from(v); })
        .def_readwrite("volume", &MarketBar::volume)
        .def_readwrite("trade_count", &MarketBar::trade_count)
        .def_readwrite("stale", &MarketBar::stale)
        .def_readwrite("analytics_valid", &MarketBar::analytics_valid);

    py::class_<OptionAnalytics>(m, "OptionAnalytics")
        .def(py::init<>())
        .def_property("timestamp",
            [](const OptionAnalytics& a) { return a.timestamp.epoch_ns; },
            [](OptionAnalytics& a, int64_t v) { a.timestamp = Timestamp{v}; })
        .def_property("contract_version_id",
            [](const OptionAnalytics& a) { return a.contract_version_id.value; },
            [](OptionAnalytics& a, uint64_t v) { a.contract_version_id = ContractVersionId{v}; })
        .def_readwrite("implied_volatility", &OptionAnalytics::implied_volatility)
        .def_readwrite("delta", &OptionAnalytics::delta)
        .def_readwrite("gamma", &OptionAnalytics::gamma)
        .def_readwrite("theta", &OptionAnalytics::theta)
        .def_readwrite("vega", &OptionAnalytics::vega)
        .def_readwrite("rho", &OptionAnalytics::rho)
        .def_readwrite("valid", &OptionAnalytics::valid);

    py::class_<CorporateActionTransition>(m, "CorporateActionTransition")
        .def(py::init<>())
        .def_readwrite("lineage_event_id", &CorporateActionTransition::lineage_event_id)
        .def_property("effective_at",
            [](const CorporateActionTransition& t) { return t.effective_at.epoch_ns; },
            [](CorporateActionTransition& t, int64_t v) { t.effective_at = Timestamp{v}; })
        .def_property("source_available_at",
            [](const CorporateActionTransition& t) { return t.source_available_at.epoch_ns; },
            [](CorporateActionTransition& t, int64_t v) { t.source_available_at = Timestamp{v}; })
        .def_property("parent_version_id",
            [](const CorporateActionTransition& t) { return t.parent_version_id.value; },
            [](CorporateActionTransition& t, uint64_t v) { t.parent_version_id = ContractVersionId{v}; })
        .def_property("child_version_id",
            [](const CorporateActionTransition& t) { return t.child_version_id.value; },
            [](CorporateActionTransition& t, uint64_t v) { t.child_version_id = ContractVersionId{v}; })
        .def_readwrite("parent_contracts", &CorporateActionTransition::parent_contracts)
        .def_readwrite("child_contracts", &CorporateActionTransition::child_contracts)
        .def_readwrite("type", &CorporateActionTransition::type)
        .def_readwrite("settlement_rule", &CorporateActionTransition::settlement_rule)
        .def_readwrite("occ_confirmed", &CorporateActionTransition::occ_confirmed)
        .def("is_actionable", &CorporateActionTransition::is_actionable);

    // ---------------------------------------------------------------- orders
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("order_id", &Order::order_id)
        .def_property("contract_version_id",
            [](const Order& o) { return o.contract_version_id.value; },
            [](Order& o, uint64_t v) { o.contract_version_id = ContractVersionId{v}; })
        .def_readwrite("kind", &Order::kind)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("side", &Order::side)
        .def_readwrite("type", &Order::type)
        .def_readwrite("time_in_force", &Order::time_in_force)
        .def_property("limit_price",
            [](const Order& o) -> py::object {
                if (!o.limit_price.has_value()) return py::none();
                return py::float_(o.limit_price->to_double());
            },
            [](Order& o, py::object v) {
                if (v.is_none()) o.limit_price.reset();
                else o.limit_price = money_from(v.cast<double>());
            })
        .def_property("stop_price",
            [](const Order& o) -> py::object {
                if (!o.stop_price.has_value()) return py::none();
                return py::float_(o.stop_price->to_double());
            },
            [](Order& o, py::object v) {
                if (v.is_none()) o.stop_price.reset();
                else o.stop_price = money_from(v.cast<double>());
            })
        .def_readwrite("group_id", &Order::group_id)
        .def_readwrite("reduce_only", &Order::reduce_only)
        .def_readwrite("tag", &Order::tag);

    py::class_<OrderGroup>(m, "OrderGroup")
        .def(py::init<>())
        .def_readwrite("group_id", &OrderGroup::group_id)
        .def_readwrite("legs", &OrderGroup::legs)
        .def_readwrite("atomic", &OrderGroup::atomic);

    py::class_<Fill>(m, "Fill")
        .def_readonly("order_id", &Fill::order_id)
        .def_readonly("group_id", &Fill::group_id)
        .def_property_readonly("filled_at", [](const Fill& f) { return f.filled_at.epoch_ns; })
        .def_property_readonly("contract_version_id",
            [](const Fill& f) { return f.contract_version_id.value; })
        .def_readonly("side", &Fill::side)
        .def_readonly("quantity", &Fill::quantity)
        .def_property_readonly("mark", [](const Fill& f) { return f.mark.to_double(); })
        .def_property_readonly("fill_price", [](const Fill& f) { return f.fill_price.to_double(); })
        .def_property_readonly("half_spread", [](const Fill& f) { return f.half_spread.to_double(); })
        .def_property_readonly("gross_cash", [](const Fill& f) { return f.gross_cash.to_double(); })
        .def_property_readonly("fees", [](const Fill& f) { return f.fees.to_double(); })
        .def_property_readonly("net_cash", [](const Fill& f) { return f.net_cash.to_double(); })
        .def_property_readonly("net_cash_micros", [](const Fill& f) { return f.net_cash.micros; })
        .def_readonly("multiplier", &Fill::multiplier);

    py::class_<OrderRejection>(m, "OrderRejection")
        .def_readonly("order_id", &OrderRejection::order_id)
        .def_readonly("group_id", &OrderRejection::group_id)
        .def_property_readonly("at", [](const OrderRejection& r) { return r.at.epoch_ns; })
        .def_readonly("reason", &OrderRejection::reason)
        .def_readonly("detail", &OrderRejection::detail)
        .def_property_readonly("reason_name",
            [](const OrderRejection& r) { return std::string(to_string(r.reason)); });

    py::class_<Position>(m, "Position")
        .def_property_readonly("position_id", [](const Position& p) { return p.id.value; })
        .def_property_readonly("contract_version_id",
            [](const Position& p) { return p.contract_version_id.value; })
        .def_readonly("kind", &Position::kind)
        .def_readonly("quantity", &Position::quantity)
        .def_property_readonly("cost_basis", [](const Position& p) { return p.cost_basis.to_double(); })
        .def_property_readonly("cost_basis_micros", [](const Position& p) { return p.cost_basis.micros; })
        .def_property_readonly("realized_pnl", [](const Position& p) { return p.realized_pnl.to_double(); })
        .def_property_readonly("average_cost", [](const Position& p) { return p.average_cost().to_double(); });

    py::class_<EquityPosition>(m, "EquityPosition")
        .def_readonly("symbol", &EquityPosition::symbol)
        .def_readonly("shares", &EquityPosition::shares)
        .def_property_readonly("cost_basis", [](const EquityPosition& e) { return e.cost_basis.to_double(); })
        .def_property_readonly("realized_pnl", [](const EquityPosition& e) { return e.realized_pnl.to_double(); })
        .def_property_readonly("average_cost", [](const EquityPosition& e) { return e.average_cost().to_double(); });

    // ---------------------------------------------------------------- config
    py::class_<SpreadFeatures>(m, "SpreadFeatures")
        .def(py::init<>())
        .def_readwrite("mark_dollars", &SpreadFeatures::mark_dollars)
        .def_readwrite("implied_volatility", &SpreadFeatures::implied_volatility)
        .def_readwrite("days_to_expiry", &SpreadFeatures::days_to_expiry)
        .def_readwrite("moneyness", &SpreadFeatures::moneyness)
        .def_readwrite("volume", &SpreadFeatures::volume)
        .def_readwrite("trade_count", &SpreadFeatures::trade_count)
        .def_readwrite("underlying_dollars", &SpreadFeatures::underlying_dollars)
        .def_readwrite("minutes_from_open", &SpreadFeatures::minutes_from_open)
        .def_readwrite("is_call", &SpreadFeatures::is_call);

    py::class_<SpreadModelConfig>(m, "SpreadModelConfig")
        .def(py::init<>())
        .def_readwrite("kind", &SpreadModelConfig::kind)
        .def_readwrite("constant_cents", &SpreadModelConfig::constant_cents)
        .def_readwrite("proportional_bps", &SpreadModelConfig::proportional_bps)
        .def_readwrite("log_base", &SpreadModelConfig::log_base)
        .def_readwrite("log_sigma", &SpreadModelConfig::log_sigma)
        .def_readwrite("ref_implied_volatility", &SpreadModelConfig::ref_implied_volatility)
        .def_readwrite("ref_days_to_expiry", &SpreadModelConfig::ref_days_to_expiry)
        .def_readwrite("ref_volume", &SpreadModelConfig::ref_volume)
        .def_readwrite("variance_scale", &SpreadModelConfig::variance_scale)
        .def_readwrite("preserve_mean_under_variance_scale",
                       &SpreadModelConfig::preserve_mean_under_variance_scale)
        .def("effective_sigma", &SpreadModelConfig::effective_sigma)
        .def_readwrite("beta_iv", &SpreadModelConfig::beta_iv)
        .def_readwrite("beta_log_dte", &SpreadModelConfig::beta_log_dte)
        .def_readwrite("beta_log_volume", &SpreadModelConfig::beta_log_volume)
        .def_readwrite("beta_abs_moneyness", &SpreadModelConfig::beta_abs_moneyness)
        .def_readwrite("beta_minutes_from_open", &SpreadModelConfig::beta_minutes_from_open)
        .def_readwrite("empirical_half_spread_cents", &SpreadModelConfig::empirical_half_spread_cents)
        .def_readwrite("min_half_spread_cents", &SpreadModelConfig::min_half_spread_cents)
        .def_readwrite("max_fraction_of_mark", &SpreadModelConfig::max_fraction_of_mark)
        .def_readwrite("round_to_tick", &SpreadModelConfig::round_to_tick);

    py::class_<FeeSchedule>(m, "FeeSchedule")
        .def(py::init<>())
        .def_static("zero", &FeeSchedule::zero)
        .def_property("commission_per_contract",
            [](const FeeSchedule& f) { return f.commission_per_contract.to_double(); },
            [](FeeSchedule& f, double v) { f.commission_per_contract = money_from(v); })
        .def_property("commission_per_trade",
            [](const FeeSchedule& f) { return f.commission_per_trade.to_double(); },
            [](FeeSchedule& f, double v) { f.commission_per_trade = money_from(v); })
        .def_readwrite("sec_fee_rate_per_dollar", &FeeSchedule::sec_fee_rate_per_dollar)
        .def_readwrite("sec_fee_on_sells_only", &FeeSchedule::sec_fee_on_sells_only)
        .def_property("finra_taf_per_contract",
            [](const FeeSchedule& f) { return f.finra_taf_per_contract.to_double(); },
            [](FeeSchedule& f, double v) { f.finra_taf_per_contract = money_from(v); })
        .def_property("finra_taf_cap_per_trade",
            [](const FeeSchedule& f) { return f.finra_taf_cap_per_trade.to_double(); },
            [](FeeSchedule& f, double v) { f.finra_taf_cap_per_trade = money_from(v); })
        .def_property("regulatory_per_contract",
            [](const FeeSchedule& f) { return f.regulatory_per_contract.to_double(); },
            [](FeeSchedule& f, double v) { f.regulatory_per_contract = money_from(v); })
        .def_property("cat_per_contract",
            [](const FeeSchedule& f) { return f.cat_per_contract.to_double(); },
            [](FeeSchedule& f, double v) { f.cat_per_contract = money_from(v); })
        .def_property("exercise_fee",
            [](const FeeSchedule& f) { return f.exercise_fee.to_double(); },
            [](FeeSchedule& f, double v) { f.exercise_fee = money_from(v); })
        .def_property("assignment_fee",
            [](const FeeSchedule& f) { return f.assignment_fee.to_double(); },
            [](FeeSchedule& f, double v) { f.assignment_fee = money_from(v); })
        .def_readwrite("schedule_id", &FeeSchedule::schedule_id)
        .def("option_fees", [](const FeeSchedule& f, OrderSide side, int64_t contracts, double notional) {
            return f.option_fees(side, contracts, money_from(notional)).to_double();
        }, py::arg("side"), py::arg("contracts"), py::arg("notional"));

    py::class_<RiskLimits>(m, "RiskLimits")
        .def(py::init<>())
        .def_readwrite("max_open_positions", &RiskLimits::max_open_positions)
        .def_readwrite("max_contracts_per_underlying", &RiskLimits::max_contracts_per_underlying)
        .def_property("max_notional_per_underlying",
            [](const RiskLimits& r) { return r.max_notional_per_underlying.to_double(); },
            [](RiskLimits& r, double v) { r.max_notional_per_underlying = money_from(v); })
        .def_property("max_loss_per_trade",
            [](const RiskLimits& r) { return r.max_loss_per_trade.to_double(); },
            [](RiskLimits& r, double v) { r.max_loss_per_trade = money_from(v); })
        .def_property("max_daily_loss",
            [](const RiskLimits& r) { return r.max_daily_loss.to_double(); },
            [](RiskLimits& r, double v) { r.max_daily_loss = money_from(v); })
        .def_readwrite("max_drawdown_fraction", &RiskLimits::max_drawdown_fraction)
        .def_readwrite("max_margin_usage_fraction", &RiskLimits::max_margin_usage_fraction)
        .def_readwrite("max_short_option_contracts", &RiskLimits::max_short_option_contracts)
        .def_readwrite("max_abs_delta", &RiskLimits::max_abs_delta);

    py::class_<BacktestConfig>(m, "BacktestConfig")
        .def(py::init<>())
        .def_property("start",
            [](const BacktestConfig& c) { return c.start.epoch_ns; },
            [](BacktestConfig& c, int64_t v) { c.start = Timestamp{v}; })
        .def_property("end",
            [](const BacktestConfig& c) { return c.end.epoch_ns; },
            [](BacktestConfig& c, int64_t v) { c.end = Timestamp{v}; })
        .def_property("initial_cash",
            [](const BacktestConfig& c) { return c.initial_cash.to_double(); },
            [](BacktestConfig& c, double v) { c.initial_cash = money_from(v); })
        .def_readwrite("execution_timing", &BacktestConfig::execution_timing)
        .def_readwrite("assignment_policy", &BacktestConfig::assignment_policy)
        .def_readwrite("spread_mc_paths", &BacktestConfig::spread_mc_paths)
        .def_readwrite("spread_mc_seed", &BacktestConfig::spread_mc_seed)
        .def_readwrite("spread_model", &BacktestConfig::spread_model)
        .def_readwrite("margin_model", &BacktestConfig::margin_model)
        .def_readwrite("fees", &BacktestConfig::fees)
        .def_readwrite("risk", &BacktestConfig::risk)
        .def_readwrite("require_occ_confirmed_lineage", &BacktestConfig::require_occ_confirmed_lineage)
        .def_readwrite("reject_fallback_analytics", &BacktestConfig::reject_fallback_analytics)
        .def_readwrite("reject_stale_bars", &BacktestConfig::reject_stale_bars);

    py::class_<PathMetrics>(m, "PathMetrics")
        .def_readonly("scenario_id", &PathMetrics::scenario_id)
        .def_property_readonly("net_pnl", [](const PathMetrics& p) { return p.net_pnl.to_double(); })
        .def_property_readonly("net_pnl_micros", [](const PathMetrics& p) { return p.net_pnl.micros; })
        .def_property_readonly("realized_pnl", [](const PathMetrics& p) { return p.realized_pnl.to_double(); })
        .def_property_readonly("unrealized_pnl", [](const PathMetrics& p) { return p.unrealized_pnl.to_double(); })
        .def_property_readonly("fees", [](const PathMetrics& p) { return p.fees.to_double(); })
        .def_property_readonly("spread_cost", [](const PathMetrics& p) { return p.spread_cost.to_double(); })
        .def_property_readonly("spread_cost_micros", [](const PathMetrics& p) { return p.spread_cost.micros; })
        .def_property_readonly("final_equity", [](const PathMetrics& p) { return p.final_equity.to_double(); })
        .def_property_readonly("final_equity_micros", [](const PathMetrics& p) { return p.final_equity.micros; })
        .def_property_readonly("peak_equity", [](const PathMetrics& p) { return p.peak_equity.to_double(); })
        .def_property_readonly("max_drawdown", [](const PathMetrics& p) { return p.max_drawdown.to_double(); })
        .def_property_readonly("peak_margin_requirement",
            [](const PathMetrics& p) { return p.peak_margin_requirement.to_double(); })
        .def_readonly("return_fraction", &PathMetrics::return_fraction)
        .def_readonly("fill_count", &PathMetrics::fill_count)
        .def_readonly("group_count", &PathMetrics::group_count)
        .def_readonly("rejection_count", &PathMetrics::rejection_count)
        .def_readonly("assignment_count", &PathMetrics::assignment_count)
        .def_readonly("exercise_count", &PathMetrics::exercise_count)
        .def_readonly("expiration_count", &PathMetrics::expiration_count)
        .def_readonly("trade_count", &PathMetrics::trade_count)
        .def_readonly("winning_trades", &PathMetrics::winning_trades)
        .def_readonly("losing_trades", &PathMetrics::losing_trades)
        .def_property_readonly("best_trade_pnl", [](const PathMetrics& p) { return p.best_trade_pnl.to_double(); })
        .def_property_readonly("worst_trade_pnl", [](const PathMetrics& p) { return p.worst_trade_pnl.to_double(); })
        .def_readonly("margin_breached", &PathMetrics::margin_breached)
        .def_readonly("ledger_reconciles", &PathMetrics::ledger_reconciles)
        .def_readonly("truncated", &PathMetrics::truncated)
        .def_readonly("quarantined_positions", &PathMetrics::quarantined_positions);

    py::enum_<CloseReason>(m, "CloseReason")
        .value("CLOSED", CloseReason::Closed)
        .value("EXPIRED", CloseReason::Expired)
        .value("EXERCISED", CloseReason::Exercised)
        .value("ASSIGNED", CloseReason::Assigned)
        .value("ADJUSTED", CloseReason::Adjusted);

    py::class_<TradeRecord>(m, "TradeRecord")
        .def_readonly("trade_id", &TradeRecord::trade_id)
        .def_property_readonly("contract_version_id",
            [](const TradeRecord& t) { return t.contract_version_id.value; })
        .def_readonly("close_group_id", &TradeRecord::close_group_id)
        .def_property_readonly("opened_at", [](const TradeRecord& t) { return t.opened_at.epoch_ns; })
        .def_property_readonly("closed_at", [](const TradeRecord& t) { return t.closed_at.epoch_ns; })
        .def_readonly("quantity", &TradeRecord::quantity)
        .def_readonly("was_short", &TradeRecord::was_short)
        .def_property_readonly("entry_price", [](const TradeRecord& t) { return t.entry_price.to_double(); })
        .def_property_readonly("exit_price", [](const TradeRecord& t) { return t.exit_price.to_double(); })
        .def_property_readonly("realized_pnl", [](const TradeRecord& t) { return t.realized_pnl.to_double(); })
        .def_property_readonly("realized_pnl_micros", [](const TradeRecord& t) { return t.realized_pnl.micros; })
        .def_property_readonly("fees", [](const TradeRecord& t) { return t.fees.to_double(); })
        .def_property_readonly("spread_cost", [](const TradeRecord& t) { return t.spread_cost.to_double(); })
        .def_readonly("reason", &TradeRecord::reason)
        .def_readonly("multiplier", &TradeRecord::multiplier)
        .def_property_readonly("holding_days", &TradeRecord::holding_days);

    py::class_<EquityPoint>(m, "EquityPoint")
        .def_property_readonly("timestamp", [](const EquityPoint& e) { return e.timestamp.epoch_ns; })
        .def_property_readonly("cash", [](const EquityPoint& e) { return e.cash.to_double(); })
        .def_property_readonly("realized_pnl", [](const EquityPoint& e) { return e.realized_pnl.to_double(); })
        .def_property_readonly("unrealized_pnl", [](const EquityPoint& e) { return e.unrealized_pnl.to_double(); })
        .def_property_readonly("equity", [](const EquityPoint& e) { return e.equity.to_double(); })
        .def_property_readonly("equity_micros", [](const EquityPoint& e) { return e.equity.micros; })
        .def_property_readonly("margin_requirement",
            [](const EquityPoint& e) { return e.margin_requirement.to_double(); })
        .def_property_readonly("position_value", [](const EquityPoint& e) { return e.position_value.to_double(); })
        .def_readonly("open_positions", &EquityPoint::open_positions);

    py::class_<AccountState>(m, "AccountState")
        .def_property_readonly("cash", [](const AccountState& s) { return s.cash.to_double(); })
        .def_property_readonly("cash_micros", [](const AccountState& s) { return s.cash.micros; })
        .def_property_readonly("equity", [](const AccountState& s) { return s.equity.to_double(); })
        .def_property_readonly("equity_micros", [](const AccountState& s) { return s.equity.micros; })
        .def_property_readonly("margin_requirement",
            [](const AccountState& s) { return s.margin_requirement.to_double(); })
        .def_property_readonly("buying_power", [](const AccountState& s) { return s.buying_power.to_double(); })
        .def_property_readonly("realized_pnl", [](const AccountState& s) { return s.realized_pnl.to_double(); })
        .def_property_readonly("unrealized_pnl", [](const AccountState& s) { return s.unrealized_pnl.to_double(); })
        .def_property_readonly("fees_paid", [](const AccountState& s) { return s.fees_paid.to_double(); })
        .def_readonly("open_position_count", &AccountState::open_position_count);

    py::class_<MarketSnapshot>(m, "MarketSnapshot")
        .def(py::init<>())
        .def_property("timestamp",
            [](const MarketSnapshot& s) { return s.timestamp.epoch_ns; },
            [](MarketSnapshot& s, int64_t v) { s.timestamp = Timestamp{v}; })
        .def_readwrite("bars", &MarketSnapshot::bars)
        .def_readwrite("analytics", &MarketSnapshot::analytics)
        .def_property("underlying_price",
            [](const MarketSnapshot& s) {
                std::unordered_map<std::string, double> out;
                for (const auto& [k, v] : s.underlying_price) out[k] = v.to_double();
                return out;
            },
            [](MarketSnapshot& s, const std::unordered_map<std::string, double>& in) {
                s.underlying_price.clear();
                for (const auto& [k, v] : in) s.underlying_price[k] = money_from(v);
            });

    py::class_<SpreadPairing>(m, "SpreadPairing")
        .def_property_readonly("short_leg", [](const SpreadPairing& p) { return p.short_leg.value; })
        .def_property_readonly("long_leg", [](const SpreadPairing& p) { return p.long_leg.value; })
        .def_readonly("contracts", &SpreadPairing::contracts)
        .def_property_readonly("requirement", [](const SpreadPairing& p) { return p.requirement.to_double(); })
        .def_readonly("covered_by_equity", &SpreadPairing::covered_by_equity)
        .def_readonly("naked", &SpreadPairing::naked);

    py::class_<MarginResult>(m, "MarginResult")
        .def_property_readonly("requirement", [](const MarginResult& r) { return r.requirement.to_double(); })
        .def_property_readonly("requirement_micros", [](const MarginResult& r) { return r.requirement.micros; })
        .def_readonly("disallowed", &MarginResult::disallowed)
        .def_readonly("disallowed_reason", &MarginResult::disallowed_reason)
        .def_readonly("pairings", &MarginResult::pairings);

    py::class_<LedgerEntry>(m, "LedgerEntry")
        .def_property_readonly("at", [](const LedgerEntry& e) { return e.at.epoch_ns; })
        .def_readonly("kind", &LedgerEntry::kind)
        .def_property_readonly("amount", [](const LedgerEntry& e) { return e.amount.to_double(); })
        .def_property_readonly("amount_micros", [](const LedgerEntry& e) { return e.amount.micros; })
        .def_readonly("memo", &LedgerEntry::memo);

    py::enum_<LedgerEntryKind>(m, "LedgerEntryKind")
        .value("DEPOSIT", LedgerEntryKind::Deposit)
        .value("OPTION_PREMIUM", LedgerEntryKind::OptionPremium)
        .value("EQUITY_TRADE", LedgerEntryKind::EquityTrade)
        .value("FEE", LedgerEntryKind::Fee)
        .value("EXERCISE_SETTLEMENT", LedgerEntryKind::ExerciseSettlement)
        .value("ASSIGNMENT_SETTLEMENT", LedgerEntryKind::AssignmentSettlement)
        .value("EXPIRATION_SETTLEMENT", LedgerEntryKind::ExpirationSettlement)
        .value("CASH_SETTLEMENT", LedgerEntryKind::CashSettlement)
        .value("CORPORATE_ACTION_CASH", LedgerEntryKind::CorporateActionCash);

    // ---------------------------------------------------------------- engine
    py::class_<Engine>(m, "Engine")
        .def(py::init<BacktestConfig>(), py::arg("config"))
        .def("add_contract", [](Engine& e, const OptionContractVersion& c) {
            ContractRegistry r = e.registry();
            r.add(c);
            e.set_registry(std::move(r));
        }, py::arg("contract"))
        .def("set_contracts", [](Engine& e, const std::vector<OptionContractVersion>& cs) {
            ContractRegistry r;
            for (const auto& c : cs) r.add(c);
            e.set_registry(std::move(r));
        }, py::arg("contracts"))
        .def("queue_corporate_actions", &Engine::queue_corporate_actions, py::arg("events"))
        .def("begin_scenario", &Engine::begin_scenario, py::arg("scenario_id"))
        .def("begin_bar", &Engine::begin_bar, py::arg("snapshot"))
        .def("submit_group", &Engine::submit_group, py::arg("group"))
        .def("end_bar", &Engine::end_bar)
        // Takes epoch nanoseconds, matching every other timestamp on this API.
        .def("end_session",
             [](Engine& e, int64_t session_close_ns) {
                 e.end_session(Timestamp{session_close_ns});
             },
             py::arg("session_close_ns"))
        .def("finalize", &Engine::finalize)
        .def("account_state", &Engine::account_state)
        .def("metrics", &Engine::metrics, py::return_value_policy::copy)
        .def("fills", &Engine::fills, py::return_value_policy::copy)
        .def("rejections", &Engine::rejections, py::return_value_policy::copy)
        .def("positions", [](const Engine& e) { return e.positions().snapshot(); })
        .def("equity_positions", [](const Engine& e) { return e.positions().equity_snapshot(); })
        .def("equity_curve", [](const Engine& e) {
            std::vector<double> out;
            for (Money m : e.equity_curve()) out.push_back(m.to_double());
            return out;
        })
        .def("trades", &Engine::trades, py::return_value_policy::copy)
        .def("equity_points", &Engine::equity_points, py::return_value_policy::copy)
        .def("ledger_entries", [](const Engine& e) { return e.ledger().entries(); })
        .def("ledger_reconciles", [](const Engine& e) { return e.ledger().reconciles(); })
        .def("cash", [](const Engine& e) { return e.ledger().cash().to_double(); })
        .def("cash_micros", [](const Engine& e) { return e.ledger().cash().micros; });

    // Exposed so margin behavior can be tested directly, without driving a
    // whole backtest, and so a report can explain how legs were paired.
    m.def("evaluate_margin",
        [](MarginModelKind kind, const std::vector<OptionContractVersion>& contracts,
           const std::vector<std::pair<uint64_t, int64_t>>& holdings,
           const std::unordered_map<std::string, double>& underlying,
           const std::unordered_map<uint64_t, double>& marks,
           const std::unordered_map<std::string, int64_t>& shares)
        {
            ContractRegistry registry;
            for (const auto& c : contracts) registry.add(c);

            PositionBook book;
            for (const auto& [cv, qty] : holdings)
                book.apply(ContractVersionId{cv}, EquityKind::Option, qty, Money::zero(), 1, Timestamp{0});
            for (const auto& [sym, n] : shares)
                book.apply_equity(sym, n, Money::zero());

            MarginContext ctx;
            for (const auto& [k, v] : underlying) ctx.underlying_price[k] = money_from(v);
            for (const auto& [k, v] : marks) ctx.mark[k] = money_from(v);

            return make_margin_model(kind)->evaluate(book, registry, ctx);
        },
        py::arg("model"), py::arg("contracts"), py::arg("holdings"),
        py::arg("underlying"), py::arg("marks") = std::unordered_map<uint64_t, double>{},
        py::arg("shares") = std::unordered_map<std::string, int64_t>{});

    m.def("hash_symbol", [](const std::string& s) { return hash_key(s); }, py::arg("key"));

    m.def("spread_draw",
        [](const SpreadModelConfig& cfg, const SpreadFeatures& f, uint64_t seed,
           uint32_t scenario, uint64_t order_id, uint64_t instrument, int64_t ts, uint32_t leg) {
            return half_spread_dollars(cfg, f, DrawKey{seed, scenario, order_id, instrument, ts, leg});
        },
        py::arg("config"), py::arg("features"), py::arg("seed"), py::arg("scenario"),
        py::arg("order_id"), py::arg("instrument"), py::arg("timestamp_ns"), py::arg("leg") = 0);

    m.attr("AUTOMATIC_EXERCISE_THRESHOLD") = automatic_exercise_threshold().to_double();
    m.def("assignment_policy_name", &assignment_policy_name, py::arg("policy"));
}
