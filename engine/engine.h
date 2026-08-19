#pragma once

#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "contract.h"
#include "ledger.h"
#include "margin.h"
#include "order.h"
#include "position.h"
#include "spread.h"
#include "types.h"

namespace obt {

// How options that finish in the money are resolved. Exercise and assignment
// are deterministic policies, never Monte Carlo: randomizing them would mix an
// unobservable into results that are otherwise reproducible.
enum class AssignmentPolicy : uint8_t {
    // Options simply expire; nothing is exercised. Useful for isolating
    // pure premium capture.
    ExpirationOnly,
    // Only a strategy's explicit exercise order does anything.
    ExplicitExerciseOnly,
    // OCC exercise-by-exception: anything at least one cent in the money at
    // expiration is exercised. This is the research default.
    AutomaticITMExercise,
    // Additionally assigns short options early when a strategy would rationally
    // be assigned, currently limited to the observable dividend case.
    ConservativeEarlyAssignment,
};

// OCC exercise-by-exception threshold, per share: one cent in the money
// (OCC Rule 805(d)(2)).
inline Money automatic_exercise_threshold() { return Money::from_double(0.01); }

// The same threshold expressed against a contract's aggregate payoff. For a
// standard 100-share contract this is $1.00, which is also what OCC Rule 1804
// specifies for a cash-settled contract with a multiplier, so one expression
// covers both. Scaling by the deliverable matters for an adjusted contract: a
// flat $0.01 against an aggregate payoff would exercise anything a hundredth of
// a cent per share in the money.
inline Money aggregate_exercise_threshold(int64_t shares_per_contract) {
    return Money{automatic_exercise_threshold().micros * shares_per_contract};
}

struct RiskLimits {
    int64_t max_open_positions = 0;          // 0 disables the limit
    int64_t max_contracts_per_underlying = 0;
    Money max_notional_per_underlying{};
    Money max_loss_per_trade{};
    Money max_daily_loss{};
    double max_drawdown_fraction = 0.0;
    double max_margin_usage_fraction = 0.0;
    int64_t max_short_option_contracts = 0;
    double max_abs_delta = 0.0;

    bool any_enabled() const {
        return max_open_positions || max_contracts_per_underlying
            || !max_notional_per_underlying.is_zero() || !max_loss_per_trade.is_zero()
            || !max_daily_loss.is_zero() || max_drawdown_fraction > 0.0
            || max_margin_usage_fraction > 0.0 || max_short_option_contracts
            || max_abs_delta > 0.0;
    }
};

struct BacktestConfig {
    Timestamp start{};
    Timestamp end = Timestamp::never();
    Money initial_cash = Money::from_double(100'000.0);

    ExecutionTiming execution_timing = ExecutionTiming::NextBarOpen;
    AssignmentPolicy assignment_policy = AssignmentPolicy::AutomaticITMExercise;

    uint32_t spread_mc_paths = 1;
    uint64_t spread_mc_seed = 42;
    SpreadModelConfig spread_model;

    MarginModelKind margin_model = MarginModelKind::Robinhood;
    FeeSchedule fees;
    RiskLimits risk;

    // Fail-closed data gates. A position held through an adjustment with no
    // OCC-confirmed lineage is rejected rather than silently carried.
    bool require_occ_confirmed_lineage = true;
    bool reject_fallback_analytics = true;
    bool reject_stale_bars = true;
};

// Why a position stopped existing. Distinguishing these matters because a
// strategy that never closes a trade voluntarily has a very different risk
// profile from one that does, and expiries are not decisions.
enum class CloseReason : uint8_t { Closed, Expired, Exercised, Assigned, Adjusted };

// One closed round trip on one contract version. Emitted on every closing event,
// including partial closes, so per-trade statistics are exact rather than
// reconstructed from fills.
struct TradeRecord {
    uint64_t trade_id = 0;
    ContractVersionId contract_version_id{};
    uint64_t open_group_id = 0;
    uint64_t close_group_id = 0;
    Timestamp opened_at{};
    Timestamp closed_at{};
    // Contracts closed by this event, always positive.
    int64_t quantity = 0;
    bool was_short = false;
    // Per-contract average entry cost and the exit price actually received.
    Money entry_price{};
    Money exit_price{};
    Money realized_pnl{};
    Money fees{};
    Money spread_cost{};
    CloseReason reason = CloseReason::Closed;
    int64_t multiplier = 100;

    int64_t holding_days() const { return opened_at.days_until(closed_at); }
};

// Account state at one instant. Realized and unrealized are carried separately so
// a report can show what has actually been banked against what is still at risk.
struct EquityPoint {
    Timestamp timestamp{};
    Money cash{};
    Money realized_pnl{};
    Money unrealized_pnl{};
    Money equity{};
    Money margin_requirement{};
    Money position_value{};
    int64_t open_positions = 0;
};

// Per-scenario results. Deterministic components are reported separately from
// the stochastic spread cost so a reader can tell them apart.
struct PathMetrics {
    uint32_t scenario_id = 0;
    Money net_pnl{};
    Money realized_pnl{};
    Money unrealized_pnl{};
    Money fees{};
    Money spread_cost{};
    Money final_equity{};
    Money peak_equity{};
    Money max_drawdown{};
    Money peak_margin_requirement{};
    double return_fraction = 0.0;
    int64_t fill_count = 0;
    int64_t group_count = 0;
    int64_t rejection_count = 0;
    int64_t assignment_count = 0;
    int64_t exercise_count = 0;
    int64_t expiration_count = 0;
    int64_t trade_count = 0;
    int64_t winning_trades = 0;
    int64_t losing_trades = 0;
    Money best_trade_pnl{};
    Money worst_trade_pnl{};
    bool margin_breached = false;
    bool ledger_reconciles = true;
    // Set when the engine could not source a contract transition and quarantined
    // the affected position. The run's P&L past that point describes a book the
    // engine has declared unsourceable, so a report must say so rather than
    // presenting the number plainly.
    bool truncated = false;
    int64_t quarantined_positions = 0;
};

// What a strategy is allowed to see at time T. Built from data already
// published at T, so a strategy cannot read ahead even accidentally.
struct MarketSnapshot {
    Timestamp timestamp{};
    std::vector<MarketBar> bars;
    std::vector<OptionAnalytics> analytics;
    std::unordered_map<std::string, Money> underlying_price;
};

struct AccountState {
    Money cash{};
    Money equity{};
    Money margin_requirement{};
    Money buying_power{};
    Money realized_pnl{};
    Money unrealized_pnl{};
    Money fees_paid{};
    int64_t open_position_count = 0;
    // The broker model says this book cannot exist. Previously a Disallow verdict
    // contributed nothing to the requirement, so a book of ten naked short calls
    // reported a $0 requirement and margin_breached false.
    bool margin_disallowed = false;
    std::string margin_disallowed_reason;
};

class Engine {
public:
    explicit Engine(BacktestConfig cfg)
        : cfg_(std::move(cfg)), margin_(make_margin_model(cfg_.margin_model)) {}

    // Shared, not copied. Every Engine previously held its own ContractRegistry by
    // value and the runner re-set it once per day per path with the full
    // cumulative contract set, so memory scaled with paths x contracts: measured
    // at ~1.8 GiB of registries alone for 8,000 contracts across 1,000 paths, and
    // a real ticker-year carries tens of thousands of versions. The registry is
    // read-only during a run, so one copy serves every path.
    void set_registry(std::shared_ptr<const ContractRegistry> registry) {
        registry_ = std::move(registry);
    }
    const ContractRegistry& registry() const {
        static const ContractRegistry kEmpty;
        return registry_ ? *registry_ : kEmpty;
    }

    // Lineage transitions are applied before any bar of the session, so a
    // position's identity is already correct when market data arrives.
    // Appends. Replacing silently dropped any transition whose effective date
    // fell later than the next day carrying a lineage row, so a split queued on
    // day 1 for day 10 never happened if day 2 also had lineage data.
    void queue_corporate_actions(std::vector<CorporateActionTransition> events) {
        for (CorporateActionTransition& t : events) {
            if (queued_event_ids_.insert(t.lineage_event_id).second)
                pending_actions_.push_back(std::move(t));
        }
    }

    void begin_scenario(uint32_t scenario_id) {
        scenario_id_ = scenario_id;
        book_.clear();
        ledger_.open(cfg_.initial_cash, cfg_.start);
        pending_.clear();
        fills_.clear();
        rejections_.clear();
        metrics_ = PathMetrics{};
        metrics_.scenario_id = scenario_id;
        metrics_.peak_equity = cfg_.initial_cash;
        equity_curve_.clear();
        next_order_id_ = 1;
        next_trade_id_ = 1;
        trades_.clear();
        equity_points_.clear();
        applied_actions_.clear();
        superseded_versions_.clear();
        current_ = MarketSnapshot{};
        day_start_equity_ = cfg_.initial_cash;
        halted_ = false;
    }

    // Steps 1-3 and 6 of the required ordering: apply lineage effective before
    // this instant, ingest the bars, then fill orders submitted earlier.
    void begin_bar(MarketSnapshot snapshot) {
        current_ = std::move(snapshot);
        now_ = current_.timestamp;

        apply_due_corporate_actions();
        index_current_bars();
        fill_pending_orders();
    }

    // Step 4: what the strategy may read.
    const MarketSnapshot& snapshot() const { return current_; }

    // Step 5: strategies submit declarative groups; the engine decides fills.
    void submit_group(OrderGroup group) {
        if (halted_) {
            for (const Order& leg : group.legs) {
                rejections_.push_back(OrderRejection{
                    leg.order_id, group.group_id, now_, RejectReason::RiskLimitBreached,
                    "trading halted by risk control"});
                metrics_.rejection_count++;
            }
            return;
        }
        for (Order& leg : group.legs) {
            if (leg.order_id == 0) leg.order_id = next_order_id_++;
            leg.submitted_at = now_;
            leg.group_id = group.group_id;
        }
        if (cfg_.execution_timing == ExecutionTiming::SameBarClose) {
            execute_group(group, /*use_open=*/false);
        } else {
            pending_.push_back(std::move(group));
        }
    }

    // Steps 7-9: settle anything already past expiration, revalue, enforce risk,
    // record the path.
    void end_bar() {
        process_expirations_through(now_);
        const AccountState state = account_state();
        enforce_risk(state);
        record_equity(state);
    }

    // Closes a trading session at `session_close`, settling every position whose
    // expiration falls on or before it.
    //
    // This exists because expiration is an instant no bar occupies. Contracts
    // expire at the 16:00 ET close, while minute bars are stamped at minute
    // start, so the last bar of the day is 15:59 and `expiration <= now_` is never
    // satisfied on the expiration date. Settling on bar timestamps alone therefore
    // deferred every expiration to the *next* session's open, injecting an
    // overnight or weekend gap into the settlement price of every expiring
    // position -- measured at one point as -$1,500 where the correct answer was
    // +$1,800.
    //
    // The last observed spot of the session is used, which is the closing price a
    // real settlement would reference.
    void end_session(Timestamp session_close) {
        process_expirations_through(session_close);
        const AccountState state = account_state();
        enforce_risk(state);
        // Reset the daily loss baseline so max_daily_loss is genuinely daily
        // rather than a cumulative loss from initial cash.
        day_start_equity_ = state.equity;
    }

    AccountState account_state() const {
        AccountState s;
        s.cash = ledger_.cash();
        s.realized_pnl = book_.realized_pnl_total() + book_.equity_realized_pnl();
        s.unrealized_pnl = unrealized_pnl();
        const MarginResult margin = current_margin();
        s.margin_requirement = margin.requirement;
        s.margin_disallowed = margin.disallowed;
        s.margin_disallowed_reason = margin.disallowed_reason;
        s.equity = s.cash + position_market_value();
        s.buying_power = s.equity - s.margin_requirement;
        s.fees_paid = ledger_.fees_paid();
        s.open_position_count = static_cast<int64_t>(book_.open_count());
        return s;
    }

    const PathMetrics& metrics() const { return metrics_; }
    const std::vector<Fill>& fills() const { return fills_; }
    const std::vector<OrderRejection>& rejections() const { return rejections_; }
    const std::vector<Money>& equity_curve() const { return equity_curve_; }
    const std::vector<EquityPoint>& equity_points() const { return equity_points_; }
    const std::vector<TradeRecord>& trades() const { return trades_; }
    const Ledger& ledger() const { return ledger_; }
    const PositionBook& positions() const { return book_; }
    const BacktestConfig& config() const { return cfg_; }

    PathMetrics finalize() {
        const AccountState s = account_state();
        metrics_.final_equity = s.equity;
        metrics_.realized_pnl = s.realized_pnl;
        metrics_.unrealized_pnl = s.unrealized_pnl;
        metrics_.fees = s.fees_paid;
        metrics_.net_pnl = s.equity - cfg_.initial_cash;
        metrics_.ledger_reconciles = ledger_.reconciles();
        if (!cfg_.initial_cash.is_zero()) {
            metrics_.return_fraction =
                static_cast<double>(metrics_.net_pnl.micros)
                / static_cast<double>(cfg_.initial_cash.micros);
        }
        return metrics_;
    }

private:
    // -----------------------------------------------------------------------
    // Market indexing
    // -----------------------------------------------------------------------
    void index_current_bars() {
        bar_index_.clear();
        analytics_index_.clear();
        for (const MarketBar& b : current_.bars) bar_index_[b.contract_version_id.value] = &b;
        for (const OptionAnalytics& a : current_.analytics)
            analytics_index_[a.contract_version_id.value] = &a;
    }

    const MarketBar* bar_for(ContractVersionId cv) const {
        auto it = bar_index_.find(cv.value);
        return it == bar_index_.end() ? nullptr : it->second;
    }

    const OptionAnalytics* analytics_for(ContractVersionId cv) const {
        auto it = analytics_index_.find(cv.value);
        return it == analytics_index_.end() ? nullptr : it->second;
    }

    // The mark the pipeline chose, so engine and pipeline agree on value.
    Money mark_for(ContractVersionId cv) const {
        const MarketBar* b = bar_for(cv);
        if (b == nullptr) return last_mark(cv);
        return b->valuation_price.is_zero() ? b->close : b->valuation_price;
    }

    Money last_mark(ContractVersionId cv) const {
        auto it = last_mark_.find(cv.value);
        return it == last_mark_.end() ? Money::zero() : it->second;
    }

    // -----------------------------------------------------------------------
    // Corporate actions
    // -----------------------------------------------------------------------
    void apply_due_corporate_actions() {
        for (const CorporateActionTransition& t : pending_actions_) {
            if (t.effective_at > now_) continue;
            if (t.source_available_at > now_) continue;
            if (applied_actions_.count(t.lineage_event_id)) continue;

            // The parent version stops being tradable at the effective date
            // whether or not anything is held. Recording that separately closes
            // the hole where an order already in flight filled onto a dead
            // version, because corporate actions are applied before pending
            // fills within the same bar.
            superseded_versions_.insert(t.parent_version_id.value);

            const int64_t held = book_.quantity_of(t.parent_version_id);
            if (held == 0) {
                // Nothing to convert now, but a pending order may still land on
                // the parent this bar, so do not mark it applied yet.
                continue;
            }

            if (!t.is_actionable()) {
                if (cfg_.require_occ_confirmed_lineage) {
                    // Refuse to carry a position through an adjustment we cannot
                    // source. Guessing the conversion would corrupt every
                    // downstream number, so the position is quarantined at its
                    // last observed mark and the run is flagged truncated -- one
                    // rejection, not one per bar forever.
                    quarantine(t.parent_version_id, held,
                               "adjustment has no OCC-confirmed lineage");
                    applied_actions_.insert(t.lineage_event_id);
                }
                continue;
            }

            // A conversion that does not divide evenly cannot be applied without
            // inventing or destroying exposure. OCC settles the remainder in cash,
            // which is a primitive this engine does not have, so refuse rather
            // than truncate: held=3 under a 2-for-3 conversion is 4.5 contracts,
            // and truncating to 3 silently discarded a third of the position.
            if (held % t.parent_contracts != 0) {
                quarantine(t.parent_version_id, held,
                           "quantity conversion does not divide the holding evenly");
                applied_actions_.insert(t.lineage_event_id);
                continue;
            }

            transfer_position(t, held);
            applied_actions_.insert(t.lineage_event_id);
        }
    }

    // Closes a position the engine cannot legitimately carry forward, at its last
    // observed mark, and stops marking it. Leaving it open would keep producing a
    // P&L for a book the engine has already declared unsourceable.
    void quarantine(ContractVersionId cv, int64_t held, const std::string& why) {
        const Position* p = book_.find(cv);
        const OptionContractVersion* c = registry().find(cv);
        const int64_t multiplier = c ? c->quote_multiplier : 100;
        const Money mark = mark_for(cv);
        const Money entry_avg = p ? p->average_cost() : Money::zero();
        const Timestamp opened_at = p ? p->opened_at : now_;

        const ApplyFillResult applied = book_.apply(
            cv, EquityKind::Option, -held, mark, multiplier, now_);
        ledger_.post(now_, LedgerEntryKind::CorporateActionCash,
                     Money{mark.micros * multiplier * held}, cv, 0,
                     "quarantined at last mark: " + why);

        record_trade(TradeRecord{
            next_trade_id_++, cv, 0, 0, opened_at, now_,
            held < 0 ? -held : held, held < 0,
            Money{entry_avg.micros / multiplier}, mark,
            applied.realized_pnl, Money::zero(), Money::zero(),
            CloseReason::Adjusted, multiplier,
        });

        rejections_.push_back(OrderRejection{
            0, 0, now_, RejectReason::UnconfirmedLineage, why});
        metrics_.rejection_count++;
        metrics_.truncated = true;
        metrics_.quarantined_positions++;
    }

    // Closes the parent and opens the child, moving the whole economic basis
    // across so the adjustment itself produces no artificial P&L.
    void transfer_position(const CorporateActionTransition& t, int64_t held) {
        Position* parent = book_.find(t.parent_version_id);
        if (parent == nullptr) return;
        const Money basis = parent->cost_basis;

        const int64_t child_qty = held / t.parent_contracts * t.child_contracts;
        const OptionContractVersion* child = registry().find(t.child_version_id);
        if (child == nullptr || child_qty == 0) return;

        // Remove the parent at its own average cost, which realizes nothing.
        book_.apply(t.parent_version_id, EquityKind::Option, -held,
                    parent->average_cost(), 1, now_);

        // Open the child carrying the identical total basis.
        const Money per_unit = Money{basis.micros / child_qty};
        book_.apply(t.child_version_id, EquityKind::Option, child_qty, per_unit, 1, now_);
    }

    // -----------------------------------------------------------------------
    // Order execution
    // -----------------------------------------------------------------------
    void fill_pending_orders() {
        std::vector<OrderGroup> due = std::move(pending_);
        pending_.clear();
        for (OrderGroup& g : due) execute_group(g, /*use_open=*/true);
    }

    // Reference price for a fill. Under next-bar-open timing this is the open
    // of the bar following the signal, which is why the signal cannot see it.
    Money execution_mark(ContractVersionId cv, bool use_open) const {
        const MarketBar* b = bar_for(cv);
        if (b == nullptr) return Money::zero();
        if (use_open && !b->open.is_zero()) return b->open;
        return b->valuation_price.is_zero() ? b->close : b->valuation_price;
    }

    SpreadFeatures features_for(const Order& o, Money mark) const {
        const OptionContractVersion* c = registry().find(o.contract_version_id);
        const MarketBar* b = bar_for(o.contract_version_id);
        const OptionAnalytics* a = analytics_for(o.contract_version_id);

        SpreadFeatures f;
        f.mark_dollars = mark.to_double();
        if (b != nullptr) {
            f.volume = static_cast<double>(b->volume);
            f.trade_count = static_cast<double>(b->trade_count);
        }
        if (a != nullptr) f.implied_volatility = a->implied_volatility;
        if (c != nullptr) {
            f.is_call = c->type == OptionType::Call;
            f.days_to_expiry = static_cast<double>(now_.days_until(c->expiration));
            const Money spot = underlying_of(c->underlying_symbol);
            f.underlying_dollars = spot.to_double();
            if (spot.micros > 0 && c->strike.micros > 0)
                f.moneyness = std::log(c->strike.to_double() / spot.to_double());
        }
        return f;
    }

    // Whether a price for this underlying was actually observed, as opposed to
    // defaulting to zero.
    bool has_observed_underlying(const std::string& sym) const {
        return current_.underlying_price.count(sym) || last_underlying_.count(sym);
    }

    Money underlying_of(const std::string& sym) const {
        auto it = current_.underlying_price.find(sym);
        if (it != current_.underlying_price.end()) return it->second;
        auto fallback = last_underlying_.find(sym);
        return fallback == last_underlying_.end() ? Money::zero() : fallback->second;
    }

    // Validates every leg, then commits all of them or none. A broker does not
    // partially execute a spread, so neither does this.
    void execute_group(OrderGroup& group, bool use_open) {
        struct Planned {
            Order order;
            const OptionContractVersion* contract = nullptr;
            Money mark{};
            Money half_spread{};
            Money fill_price{};
            Money gross_cash{};
            Money fees{};
            int64_t multiplier = 100;
        };

        std::vector<Planned> plan;
        plan.reserve(group.legs.size());
        RejectReason reason = RejectReason::None;
        std::string detail;

        uint32_t leg_index = 0;
        for (const Order& o : group.legs) {
            Planned p;
            p.order = o;

            // A stop trigger cannot be honestly simulated from OHLC bars: the
            // intrabar path is unknown, so there is no defensible instant at
            // which the stop became live. Refusing is correct; the previous
            // behavior silently ignored the stop price and executed at market,
            // so a BUY STOP at $999 filled immediately against a $5.00 market.
            if (o.type == OrderType::Stop || o.type == OrderType::StopLimit) {
                reason = RejectReason::UnsupportedOrderType;
                detail = "stop triggers cannot be simulated from bar data; use a limit";
                break;
            }
            // Equity legs are not implemented. The kind flag was accepted and
            // ignored, so an order for one share was priced with the contract's
            // 100x multiplier and booked as an option.
            if (o.kind != EquityKind::Option) {
                reason = RejectReason::UnsupportedInstrumentKind;
                detail = "equity orders are not implemented; shares arrive only via settlement";
                break;
            }

            p.contract = registry().find(o.contract_version_id);
            if (p.contract == nullptr) { reason = RejectReason::ContractNotTradable; detail = "unknown contract version"; break; }
            if (!p.contract->covers(now_)) { reason = RejectReason::ContractNotTradable; detail = "contract version not valid at fill time"; break; }
            if (superseded_versions_.count(o.contract_version_id.value)) {
                reason = RejectReason::ContractNotTradable;
                detail = "contract version superseded by a contract adjustment";
                break;
            }

            const bool opening = !o.reduce_only;
            if (opening && !p.contract->tradable_for_new_positions) {
                reason = RejectReason::ContractNotTradable;
                detail = "contract not tradable for new positions";
                break;
            }

            const MarketBar* b = bar_for(o.contract_version_id);
            if (b == nullptr) { reason = RejectReason::NoMarketData; detail = "no bar at fill time"; break; }
            if (cfg_.reject_stale_bars && b->stale) { reason = RejectReason::StaleMarketData; detail = "stale bar"; break; }
            if (cfg_.reject_fallback_analytics && opening && !b->analytics_valid) {
                reason = RejectReason::AnalyticsRejected;
                detail = "analytics rejected by configuration";
                break;
            }

            p.mark = execution_mark(o.contract_version_id, use_open);
            if (p.mark.micros <= 0) { reason = RejectReason::NoMarketData; detail = "non-positive mark"; break; }
            p.multiplier = p.contract->quote_multiplier;

            DrawKey key{cfg_.spread_mc_seed, scenario_id_, o.order_id,
                        p.contract->instrument_id.value, now_.epoch_ns, leg_index};
            const double half = half_spread_dollars(
                cfg_.spread_model, features_for(o, p.mark), key);
            p.half_spread = Money::from_double(half);

            // Buys cross the ask, sells cross the bid. Never the other way.
            p.fill_price = (o.side == OrderSide::Buy) ? p.mark + p.half_spread
                                                      : p.mark - p.half_spread;
            if (p.fill_price.micros < 0) p.fill_price = Money::zero();

            if (o.type == OrderType::Limit && o.limit_price.has_value()) {
                const bool ok = (o.side == OrderSide::Buy)
                    ? p.fill_price <= *o.limit_price
                    : p.fill_price >= *o.limit_price;
                if (!ok) { reason = RejectReason::LimitNotSatisfied; detail = "limit not satisfied at execution price"; break; }
            }

            const int64_t signed_qty = (o.side == OrderSide::Buy) ? o.quantity : -o.quantity;
            // Cash out for a buy, cash in for a sell.
            p.gross_cash = Money{-p.fill_price.micros * p.multiplier * signed_qty};
            p.fees = cfg_.fees.option_fees(
                o.side, o.quantity,
                Money{p.fill_price.micros * p.multiplier * o.quantity});

            plan.push_back(p);
            leg_index++;
        }

        if (reason != RejectReason::None) {
            reject_group(group, reason, detail);
            return;
        }

        // Buying power and broker rules are checked against the portfolio the
        // group would produce, not leg by leg, so a spread is not rejected for
        // the naked requirement of its short leg alone.
        if (!passes_margin_after(plan.size(), [&](PositionBook& probe) {
                for (const Planned& p : plan) {
                    const int64_t q = (p.order.side == OrderSide::Buy) ? p.order.quantity
                                                                       : -p.order.quantity;
                    probe.apply(p.order.contract_version_id, EquityKind::Option, q,
                                p.fill_price, p.multiplier, now_);
                }
            }, plan, &reason, &detail)) {
            reject_group(group, reason, detail);
            return;
        }

        for (const Planned& p : plan) {
            const int64_t signed_qty = (p.order.side == OrderSide::Buy) ? p.order.quantity
                                                                        : -p.order.quantity;
            // Capture the pre-trade state so a close can be recorded with the
            // entry price it is closing against.
            const Position* existing = book_.find(p.order.contract_version_id);
            const Money prior_avg = existing ? existing->average_cost() : Money::zero();
            const Timestamp opened_at = existing ? existing->opened_at : now_;
            const bool was_short = existing && existing->quantity < 0;

            const ApplyFillResult applied = book_.apply(
                p.order.contract_version_id, EquityKind::Option, signed_qty,
                p.fill_price, p.multiplier, now_);
            book_.add_fees(p.order.contract_version_id, p.fees);

            if (applied.closed_quantity > 0) {
                record_trade(TradeRecord{
                    next_trade_id_++, p.order.contract_version_id,
                    /*open_group_id=*/0, group.group_id, opened_at, now_,
                    applied.closed_quantity, was_short,
                    Money{prior_avg.micros / p.multiplier}, p.fill_price,
                    applied.realized_pnl, p.fees,
                    Money{p.half_spread.micros * applied.closed_quantity * p.multiplier},
                    CloseReason::Closed, p.multiplier,
                });
            }

            ledger_.post(now_, LedgerEntryKind::OptionPremium, p.gross_cash,
                         p.order.contract_version_id, p.order.order_id, "option premium");
            ledger_.post(now_, LedgerEntryKind::Fee, -p.fees,
                         p.order.contract_version_id, p.order.order_id,
                         cfg_.fees.schedule_id);

            Fill f;
            f.order_id = p.order.order_id;
            f.group_id = group.group_id;
            f.filled_at = now_;
            f.contract_version_id = p.order.contract_version_id;
            f.side = p.order.side;
            f.quantity = p.order.quantity;
            f.mark = p.mark;
            f.fill_price = p.fill_price;
            f.half_spread = p.half_spread;
            f.multiplier = p.multiplier;
            f.gross_cash = p.gross_cash;
            f.fees = p.fees;
            f.net_cash = p.gross_cash - p.fees;
            fills_.push_back(f);

            metrics_.fill_count++;
            metrics_.spread_cost += f.spread_cost();
        }
        metrics_.group_count++;
    }

    void reject_group(const OrderGroup& group, RejectReason reason, const std::string& detail) {
        for (const Order& o : group.legs) {
            rejections_.push_back(OrderRejection{o.order_id, group.group_id, now_, reason, detail});
            metrics_.rejection_count++;
        }
    }

    template <typename Mutate, typename PlanVec>
    bool passes_margin_after(size_t, Mutate mutate, const PlanVec& plan,
                             RejectReason* reason, std::string* detail) const
    {
        PositionBook probe = book_;
        mutate(probe);

        const MarginResult res = margin_->evaluate(probe, registry(), margin_context());
        if (res.disallowed) {
            *reason = RejectReason::BrokerDisallowed;
            *detail = res.disallowed_reason;
            return false;
        }

        // Long options carry no loan value at any broker, so they cannot
        // collateralize their own purchase: cash has to cover the debit in
        // full. Only stock lends, and only in a margin account. Counting option
        // market value here would let an empty account buy an unlimited premium.
        Money available = ledger_.cash();
        for (const auto& p : plan) available += p.gross_cash - p.fees;
        available += scale(probe_equity_value(probe), equity_loan_fraction());

        if (available - res.requirement < Money::zero()) {
            *reason = RejectReason::InsufficientBuyingPower;
            *detail = "requirement exceeds available buying power";
            return false;
        }

        if (cfg_.risk.max_open_positions > 0
            && static_cast<int64_t>(probe.open_count()) > cfg_.risk.max_open_positions) {
            *reason = RejectReason::RiskLimitBreached;
            *detail = "max open positions";
            return false;
        }
        if (cfg_.risk.max_short_option_contracts > 0) {
            int64_t shorts = 0;
            for (const Position& p : probe.snapshot())
                if (p.kind == EquityKind::Option && p.quantity < 0) shorts += -p.quantity;
            if (shorts > cfg_.risk.max_short_option_contracts) {
                *reason = RejectReason::RiskLimitBreached;
                *detail = "max short option contracts";
                return false;
            }
        }

        // These four were declared, bound to Python, and never referenced, so
        // configuring them silently ran the backtest unconstrained.
        if (cfg_.risk.max_contracts_per_underlying > 0
            || !cfg_.risk.max_notional_per_underlying.is_zero()
            || cfg_.risk.max_abs_delta > 0.0) {
            std::unordered_map<std::string, int64_t> contracts_by_underlying;
            std::unordered_map<std::string, Money> notional_by_underlying;
            double abs_delta = 0.0;

            for (const Position& p : probe.snapshot()) {
                if (p.kind != EquityKind::Option || p.quantity == 0) continue;
                const OptionContractVersion* c = registry().find(p.contract_version_id);
                if (c == nullptr) continue;
                const int64_t qty = p.abs_quantity();
                contracts_by_underlying[c->underlying_symbol] += qty;
                notional_by_underlying[c->underlying_symbol] +=
                    Money{c->notional(underlying_of(c->underlying_symbol)).micros * qty};
                const OptionAnalytics* a = analytics_for(p.contract_version_id);
                if (a != nullptr && a->valid) abs_delta += a->delta * p.quantity;
            }

            if (cfg_.risk.max_contracts_per_underlying > 0) {
                for (const auto& [sym, n] : contracts_by_underlying) {
                    if (n > cfg_.risk.max_contracts_per_underlying) {
                        *reason = RejectReason::RiskLimitBreached;
                        *detail = "max contracts per underlying";
                        return false;
                    }
                }
            }
            if (!cfg_.risk.max_notional_per_underlying.is_zero()) {
                for (const auto& [sym, value] : notional_by_underlying) {
                    if (value > cfg_.risk.max_notional_per_underlying) {
                        *reason = RejectReason::RiskLimitBreached;
                        *detail = "max notional per underlying";
                        return false;
                    }
                }
            }
            if (cfg_.risk.max_abs_delta > 0.0
                && std::fabs(abs_delta) > cfg_.risk.max_abs_delta) {
                *reason = RejectReason::RiskLimitBreached;
                *detail = "max absolute portfolio delta";
                return false;
            }
        }
        return true;
    }

    MarginContext margin_context() const {
        MarginContext ctx;
        ctx.underlying_price = current_.underlying_price;
        for (const auto& [sym, px] : last_underlying_)
            ctx.underlying_price.emplace(sym, px);

        // Marks come from every contract quoted at this instant, not just the
        // ones already held. A pre-trade probe evaluates a position that is not
        // in the book yet, and without its mark the Reg-T naked requirement
        // would silently drop its premium term and under-charge the order.
        for (const MarketBar& b : current_.bars)
            ctx.mark[b.contract_version_id.value] =
                b.valuation_price.is_zero() ? b.close : b.valuation_price;
        for (const Position& p : book_.snapshot())
            ctx.mark.emplace(p.contract_version_id.value, mark_for(p.contract_version_id));
        return ctx;
    }

    MarginResult current_margin() const {
        return margin_->evaluate(book_, registry(), margin_context());
    }

    // -----------------------------------------------------------------------
    // Valuation
    // -----------------------------------------------------------------------
    // Reg-T initial requirement for stock is 50%, so half the position lends.
    // A cash account lends nothing.
    double equity_loan_fraction() const {
        return cfg_.margin_model == MarginModelKind::CashAccount ? 0.0 : 0.5;
    }

    Money probe_equity_value(const PositionBook& book) const {
        Money total = Money::zero();
        for (const EquityPosition& e : book.equity_snapshot())
            total += Money{underlying_of(e.symbol).micros * e.shares};
        return total;
    }

    Money position_market_value() const { return probe_market_value(book_); }

    Money probe_market_value(const PositionBook& book) const {
        Money total = Money::zero();
        for (const Position& p : book.snapshot()) {
            const OptionContractVersion* c = registry().find(p.contract_version_id);
            const int64_t mult = c ? c->quote_multiplier : 100;
            total += Money{mark_for(p.contract_version_id).micros * mult * p.quantity};
        }
        for (const EquityPosition& e : book.equity_snapshot())
            total += Money{underlying_of(e.symbol).micros * e.shares};
        return total;
    }

    Money unrealized_pnl() const {
        Money total = Money::zero();
        for (const Position& p : book_.snapshot()) {
            const OptionContractVersion* c = registry().find(p.contract_version_id);
            const int64_t mult = c ? c->quote_multiplier : 100;
            total += Money{mark_for(p.contract_version_id).micros * mult * p.quantity} - p.cost_basis;
        }
        for (const EquityPosition& e : book_.equity_snapshot())
            total += Money{underlying_of(e.symbol).micros * e.shares} - e.cost_basis;
        return total;
    }

    // -----------------------------------------------------------------------
    // Expiration and settlement
    // -----------------------------------------------------------------------
    void process_expirations_through(Timestamp cutoff) {
        if (cfg_.assignment_policy == AssignmentPolicy::ExplicitExerciseOnly) return;

        for (const Position& p : book_.snapshot()) {
            if (p.kind != EquityKind::Option || p.quantity == 0) continue;
            const OptionContractVersion* c = registry().find(p.contract_version_id);
            if (c == nullptr || c->expiration > cutoff) continue;

            // Settlement needs an observed underlying price. Falling back to zero
            // made a put maximally in the money and settled it at a fabricated
            // price -- measured at +$9,500 of invented profit and a 100-share
            // short -- while a call silently expired worthless. Neither was
            // flagged, so refuse instead.
            if (!has_observed_underlying(c->underlying_symbol)) {
                quarantine(p.contract_version_id, p.quantity,
                           "no observed underlying price at settlement");
                continue;
            }
            if (c->has_fractional_deliverable()) {
                quarantine(p.contract_version_id, p.quantity,
                           "fractional deliverable requires cash-in-lieu settlement");
                continue;
            }

            const Money spot = underlying_of(c->underlying_symbol);
            // Per-contract payoff from the actual deliverable, aggregated against
            // the listed strike times the quote multiplier.
            const Money intrinsic = c->payoff_at(spot);

            const bool exercise =
                cfg_.assignment_policy != AssignmentPolicy::ExpirationOnly
                && intrinsic >= aggregate_exercise_threshold(
                       c->deliverable_shares_per_contract());

            const Money entry_avg = p.average_cost();
            const bool was_short = p.quantity < 0;
            const int64_t closed = p.abs_quantity();

            // Remove the option at zero, realizing the full remaining basis.
            const ApplyFillResult applied = book_.apply(
                p.contract_version_id, EquityKind::Option, -p.quantity,
                Money::zero(), c->quote_multiplier, now_);
            metrics_.expiration_count++;

            const CloseReason reason = !exercise ? CloseReason::Expired
                : (p.quantity > 0 ? CloseReason::Exercised : CloseReason::Assigned);
            record_trade(TradeRecord{
                next_trade_id_++, p.contract_version_id, 0, 0,
                p.opened_at, now_, closed, was_short,
                Money{entry_avg.micros / c->quote_multiplier},
                // An expiring option is closed at zero; the intrinsic it settles
                // into shows up in the settlement cash flow, not here.
                Money::zero(), applied.realized_pnl, Money::zero(), Money::zero(),
                reason, c->quote_multiplier,
            });

            if (!exercise) continue;

            settle_physically(*c, p.quantity, spot);
            if (p.quantity > 0) metrics_.exercise_count++;
            else metrics_.assignment_count++;
        }
    }

    // Physical delivery, which is what actually happens to equity options and
    // what makes a covered call or an assigned short call behave correctly. A
    // short call assigned without shares establishes a short stock position
    // rather than settling to cash.
    void settle_physically(const OptionContractVersion& c, int64_t contracts, Money spot) {
        const int64_t count = contracts < 0 ? -contracts : contracts;
        const int64_t shares = c.deliverable_shares_per_contract() * count;
        // The aggregate exercise price is the listed strike times the quote
        // multiplier, not the strike times the delivered share count.
        const Money strike_cash = Money{c.aggregate_exercise_price().micros * count};

        const bool long_position = contracts > 0;
        const bool call = c.type == OptionType::Call;

        // Long call and short put receive shares; long put and short call deliver.
        const bool receives_shares = (long_position && call) || (!long_position && !call);
        const int64_t share_delta = receives_shares ? shares : -shares;
        const Money cash_delta = receives_shares ? -strike_cash : strike_cash;

        // Book the shares at the effective per-share cost implied by the
        // aggregate exercise price, so basis is right for a non-standard
        // deliverable rather than assuming the listed strike per share.
        const Money share_cost = shares > 0
            ? Money{strike_cash.micros / shares} : c.strike;
        book_.apply_equity(c.underlying_symbol, share_delta, share_cost);
        ledger_.post(now_,
                     long_position ? LedgerEntryKind::ExerciseSettlement
                                   : LedgerEntryKind::AssignmentSettlement,
                     cash_delta, c.id, 0,
                     receives_shares ? "received shares at strike" : "delivered shares at strike");

        const Money fee = long_position ? cfg_.fees.exercise_fee : cfg_.fees.assignment_fee;
        if (!fee.is_zero())
            ledger_.post(now_, LedgerEntryKind::Fee, -fee, c.id, 0, "exercise/assignment fee");

        if (!c.deliverable_cash.is_zero()) {
            const Money extra = Money{c.deliverable_cash.micros * (contracts < 0 ? -contracts : contracts)};
            ledger_.post(now_, LedgerEntryKind::CashSettlement,
                         receives_shares ? extra : -extra, c.id, 0, "deliverable cash component");
        }
    }

    // -----------------------------------------------------------------------
    // Risk and reporting
    // -----------------------------------------------------------------------
    void enforce_risk(const AccountState& s) {
        if (s.equity > metrics_.peak_equity) metrics_.peak_equity = s.equity;
        const Money drawdown = metrics_.peak_equity - s.equity;
        if (drawdown > metrics_.max_drawdown) metrics_.max_drawdown = drawdown;
        if (s.margin_requirement > metrics_.peak_margin_requirement)
            metrics_.peak_margin_requirement = s.margin_requirement;

        // A book the broker model refuses is a breach whatever the arithmetic
        // requirement came to, since the requirement is not defined for a position
        // that cannot be held.
        if (s.margin_requirement > s.equity || s.margin_disallowed)
            metrics_.margin_breached = true;

        const RiskLimits& r = cfg_.risk;
        if (r.max_drawdown_fraction > 0.0 && !metrics_.peak_equity.is_zero()) {
            const double frac = static_cast<double>(drawdown.micros)
                              / static_cast<double>(metrics_.peak_equity.micros);
            if (frac > r.max_drawdown_fraction) halted_ = true;
        }
        if (r.max_margin_usage_fraction > 0.0 && !s.equity.is_zero()) {
            const double usage = static_cast<double>(s.margin_requirement.micros)
                               / static_cast<double>(s.equity.micros);
            if (usage > r.max_margin_usage_fraction) halted_ = true;
        }
        if (!r.max_daily_loss.is_zero() && (day_start_equity_ - s.equity) > r.max_daily_loss)
            halted_ = true;
    }

    void record_trade(TradeRecord t) {
        if (t.realized_pnl > metrics_.best_trade_pnl) metrics_.best_trade_pnl = t.realized_pnl;
        if (t.realized_pnl < metrics_.worst_trade_pnl) metrics_.worst_trade_pnl = t.realized_pnl;
        metrics_.trade_count++;
        if (t.realized_pnl > Money::zero()) metrics_.winning_trades++;
        else if (t.realized_pnl < Money::zero()) metrics_.losing_trades++;
        trades_.push_back(std::move(t));
    }

    void record_equity(const AccountState& s) {
        equity_curve_.push_back(s.equity);
        equity_points_.push_back(EquityPoint{
            now_, s.cash, s.realized_pnl, s.unrealized_pnl, s.equity,
            s.margin_requirement, position_market_value(), s.open_position_count,
        });
        for (const MarketBar& b : current_.bars)
            last_mark_[b.contract_version_id.value] =
                b.valuation_price.is_zero() ? b.close : b.valuation_price;
        for (const auto& [sym, px] : current_.underlying_price) last_underlying_[sym] = px;
    }

    BacktestConfig cfg_;
    std::unique_ptr<MarginModel> margin_;
    std::shared_ptr<const ContractRegistry> registry_;

    PositionBook book_;
    Ledger ledger_;
    std::vector<OrderGroup> pending_;
    std::vector<Fill> fills_;
    std::vector<OrderRejection> rejections_;
    std::vector<CorporateActionTransition> pending_actions_;
    std::set<uint64_t> applied_actions_;
    std::set<uint64_t> queued_event_ids_;
    std::set<uint64_t> superseded_versions_;

    MarketSnapshot current_;
    std::unordered_map<uint64_t, const MarketBar*> bar_index_;
    std::unordered_map<uint64_t, const OptionAnalytics*> analytics_index_;
    std::unordered_map<uint64_t, Money> last_mark_;
    std::unordered_map<std::string, Money> last_underlying_;

    std::vector<Money> equity_curve_;
    std::vector<EquityPoint> equity_points_;
    std::vector<TradeRecord> trades_;
    PathMetrics metrics_;
    Timestamp now_{};
    uint32_t scenario_id_ = 0;
    uint64_t next_order_id_ = 1;
    uint64_t next_trade_id_ = 1;
    Money day_start_equity_{};
    bool halted_ = false;
};

} // namespace obt
