#pragma once

#include <string>
#include <vector>

#include "order.h"
#include "types.h"

namespace obt {

// Fee schedule.
//
// Rates are configuration, not constants, because regulatory rates are revised
// on their own schedule and a backtest has to be able to state which schedule it
// applied. Defaults model Robinhood's published equity/ETF options rates as of
// 2026-08. Applicability follows the actual rules: Section 31 and the FINRA
// Trading Activity Fee are charged on sales only, per-contract regulatory and
// clearing fees on both sides.
//
// The Section 31 rate in particular is not a constant over history -- it was
// literally $0 from 2025-05-14 to 2026-04-03 -- so a run spanning that window
// needs the rate set per period rather than left at the default.
struct FeeSchedule {
    // Broker commission per contract. Zero for equity and ETF options at a
    // zero-commission broker; index options are charged per contract.
    Money commission_per_contract{};
    Money commission_per_trade{};

    // SEC Section 31, applied to the dollar value of sales. $20.60 per $1M
    // effective 2026-04-04.
    double sec_fee_rate_per_dollar = 0.0000206;
    bool sec_fee_on_sells_only = true;

    // FINRA Trading Activity Fee, per contract sold, effective 2026-01-01. The
    // cap is a broker convention: FINRA's rule caps the equity TAF only.
    Money finra_taf_per_contract = Money::from_double(0.00329);
    Money finra_taf_cap_per_trade = Money::from_double(9.79);

    // Robinhood bills a single blended $0.04 covering the Options Regulatory Fee
    // and OCC clearing, both sides, with no cap. The real components sum to
    // about $0.0375, so this is the broker's rounded pass-through rather than an
    // exact reconstruction.
    Money regulatory_per_contract = Money::from_double(0.04);

    // Consolidated Audit Trail, both sides.
    Money cat_per_contract = Money::from_double(0.0003);

    // Exercise, assignment, and worthless expiration are free at Robinhood.
    Money exercise_fee{};
    Money assignment_fee{};

    std::string schedule_id = "robinhood_equity_options_2026_04";

    // A charge below a cent is dropped rather than rounded up, matching how the
    // per-contract regulatory fees are actually billed.
    static Money drop_sub_cent(Money amount) {
        return amount < Money::from_double(0.01) ? Money::zero() : amount;
    }

    // Fees on one option trade. `notional` is the total dollar value of the
    // trade, used only for the ad-valorem Section 31 fee.
    Money option_fees(OrderSide side, int64_t contracts, Money notional) const {
        Money total = commission_per_trade + commission_per_contract * contracts;
        total += regulatory_per_contract * contracts;
        total += drop_sub_cent(cat_per_contract * contracts);

        if (side == OrderSide::Sell || !sec_fee_on_sells_only) {
            // Section 31 is rounded up to the next cent.
            const Money raw = scale(notional.abs(), sec_fee_rate_per_dollar);
            const int64_t cent = 10'000;
            total += Money{((raw.micros + cent - 1) / cent) * cent};
            total += drop_sub_cent(
                min_money(finra_taf_per_contract * contracts, finra_taf_cap_per_trade));
        }
        return total;
    }

    static FeeSchedule zero() {
        FeeSchedule f;
        f.sec_fee_rate_per_dollar = 0.0;
        f.finra_taf_per_contract = Money::zero();
        f.regulatory_per_contract = Money::zero();
        f.cat_per_contract = Money::zero();
        f.schedule_id = "zero_fees";
        return f;
    }
};

enum class LedgerEntryKind : uint8_t {
    Deposit,
    OptionPremium,
    EquityTrade,
    Fee,
    ExerciseSettlement,
    AssignmentSettlement,
    ExpirationSettlement,
    CashSettlement,
    CorporateActionCash,
};

// Append-only journal. Every cash movement is recorded, so the closing balance
// is reconstructible from the entries alone and any discrepancy between the
// running balance and the journal sum is a bug rather than rounding.
struct LedgerEntry {
    Timestamp at{};
    LedgerEntryKind kind = LedgerEntryKind::Deposit;
    Money amount{};
    ContractVersionId contract_version_id{};
    uint64_t order_id = 0;
    std::string memo;
};

class Ledger {
public:
    void open(Money initial_cash, Timestamp at) {
        entries_.clear();
        cash_ = Money::zero();
        post(at, LedgerEntryKind::Deposit, initial_cash, ContractVersionId{}, 0, "initial cash");
        initial_cash_ = initial_cash;
    }

    void post(Timestamp at, LedgerEntryKind kind, Money amount,
              ContractVersionId cv = ContractVersionId{}, uint64_t order_id = 0,
              const std::string& memo = "")
    {
        cash_ += amount;
        entries_.push_back(LedgerEntry{at, kind, amount, cv, order_id, memo});
        if (kind == LedgerEntryKind::Fee) fees_ += -amount;
    }

    Money cash() const { return cash_; }
    Money initial_cash() const { return initial_cash_; }
    Money fees_paid() const { return fees_; }
    const std::vector<LedgerEntry>& entries() const { return entries_; }

    // The journal must always reproduce the running balance exactly. Integer
    // money makes this an equality, not an approximation.
    bool reconciles() const {
        Money sum = Money::zero();
        for (const LedgerEntry& e : entries_) sum += e.amount;
        return sum == cash_;
    }

    void clear() {
        entries_.clear();
        cash_ = Money::zero();
        fees_ = Money::zero();
        initial_cash_ = Money::zero();
    }

private:
    std::vector<LedgerEntry> entries_;
    Money cash_{};
    Money fees_{};
    Money initial_cash_{};
};

} // namespace obt
