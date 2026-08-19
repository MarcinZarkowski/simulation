#pragma once

#include <string>

#include "types.h"

namespace obt {

// A cash dividend on an underlying.
//
// Three dates, all distinct and all load-bearing:
//
//   declared_at  when the dividend became knowable. A backtest that accrues
//                before this is trading on an unannounced payout.
//   ex_date      the first session the share trades without the dividend.
//                Whoever holds shares at the close BEFORE this earns it.
//   pay_date     when the cash actually arrives, typically two to four weeks
//                after ex-date. Accruing at ex-date overstates cash for that
//                whole window, which matters when the account is near a margin
//                boundary.
struct DividendEvent {
    std::string underlying_symbol;
    Money amount_per_share{};
    Timestamp declared_at{};
    Timestamp ex_date{};
    Timestamp pay_date{};

    // A stable id so the same dividend queued on consecutive days is applied
    // once. Derived from the terms rather than supplied, because the pipeline's
    // corporate-action rows carry no event id.
    uint64_t event_id() const {
        return hash_key(underlying_symbol + "|" + std::to_string(ex_date.epoch_ns) + "|"
                        + std::to_string(amount_per_share.micros));
    }

    bool known_at(Timestamp t) const { return declared_at <= t; }
};

// A dividend earned and not yet paid. Shares held short produce a negative
// amount, which the holder owes.
struct DividendAccrual {
    std::string underlying_symbol;
    Money amount{};
    int64_t shares = 0;
    Timestamp ex_date{};
    Timestamp pay_date{};
};

} // namespace obt
