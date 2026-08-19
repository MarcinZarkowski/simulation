#pragma once

#include <string>
#include <vector>

#include "order.h"
#include "types.h"

namespace obt {

// Fee schedule. Rates are configuration, not constants, because regulatory
// rates are revised on their own schedule and a backtest must be able to state
// which schedule it applied. Defaults model a zero-commission US retail broker
// passing through the standard regulatory fees.
//
// Applicability follows the actual rules: Section 31 and the FINRA Trading
// Activity Fee are charged on sales only; per-contract exchange and clearing
// fees are charged on both sides.
struct FeeSchedule {
    // Broker commission per contract. Zero at Robinhood.
    Money commission_per_contract{};
    Money commission_per_trade{};

    // SEC Section 31: a rate applied to the dollar value of sales.
    double sec_fee_rate_per_dollar = 0.0000278;
    bool sec_fee_on_sells_only = true;

    // FINRA Trading Activity Fee: per contract sold, capped per trade.
    Money finra_taf_per_contract = Money::from_double(0.00279);
    Money finra_taf_cap_per_trade = Money::from_double(8.30);

    // Options Regulatory Fee plus exchange and clearing fees, both sides.
    Money orf_per_contract = Money::from_double(0.02685);
    Money clearing_per_contract = Money::from_double(0.02);
    Money clearing_cap_per_trade = Money::from_double(55.0);

    // Exercise and assignment. Zero at Robinhood.
    Money exercise_fee{};
    Money assignment_fee{};

    std::string schedule_id = "us_retail_zero_commission";

    // Fees on one option trade. `notional` is the total dollar value of the
    // trade, used only for the ad-valorem Section 31 fee.
    Money option_fees(OrderSide side, int64_t contracts, Money notional) const {
        Money total = commission_per_trade + commission_per_contract * contracts;

        total += orf_per_contract * contracts;
        total += min_money(clearing_per_contract * contracts, clearing_cap_per_trade);

        if (side == OrderSide::Sell || !sec_fee_on_sells_only) {
            total += scale(notional.abs(), sec_fee_rate_per_dollar);
            total += min_money(finra_taf_per_contract * contracts, finra_taf_cap_per_trade);
        }
        return total;
    }

    static FeeSchedule zero() {
        FeeSchedule f;
        f.sec_fee_rate_per_dollar = 0.0;
        f.finra_taf_per_contract = Money::zero();
        f.orf_per_contract = Money::zero();
        f.clearing_per_contract = Money::zero();
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
