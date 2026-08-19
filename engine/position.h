#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "contract.h"
#include "order.h"
#include "types.h"

namespace obt {

// A holding in one contract version. Quantity is signed: positive long,
// negative short. Cost basis is the signed cash actually paid or received, so
// realized P&L never depends on a separately tracked average price.
struct Position {
    PositionId id{};
    ContractVersionId contract_version_id{};
    EquityKind kind = EquityKind::Option;
    int64_t quantity = 0;
    Money cost_basis{};
    Money realized_pnl{};
    // Entry-side costs on the quantity still open, held here until the round trip
    // closes and can be charged for both legs. A TradeRecord is written on the
    // closing fill, so without this the opening leg's fees and spread cost never
    // reached the trade -- per-trade fees read as roughly half the fees the path
    // actually paid.
    Money open_fees{};
    Money open_spread_cost{};
    Timestamp opened_at{};
    Timestamp last_updated_at{};

    bool is_long() const { return quantity > 0; }
    bool is_short() const { return quantity < 0; }
    bool is_flat() const { return quantity == 0; }
    int64_t abs_quantity() const { return quantity < 0 ? -quantity : quantity; }

    // Average entry cost per contract, signed like cost_basis.
    Money average_cost() const {
        if (quantity == 0) return Money::zero();
        return Money{cost_basis.micros / quantity};
    }
};

// Equity holding in the underlying, needed for covered calls, protective puts,
// collars, and share delivery on assignment.
struct EquityPosition {
    std::string symbol;
    int64_t shares = 0;
    Money cost_basis{};
    Money realized_pnl{};

    Money average_cost() const {
        if (shares == 0) return Money::zero();
        return Money{cost_basis.micros / shares};
    }
};

// Result of applying a fill to a position: how much closed, and the realized
// P&L that closing produced.
struct ApplyFillResult {
    int64_t closed_quantity = 0;
    int64_t opened_quantity = 0;
    Money realized_pnl{};
    // Both legs' costs on the quantity this fill closed: the entry-side costs
    // released proportionally, plus this fill's own share.
    Money round_trip_fees{};
    Money round_trip_spread_cost{};
};

class PositionBook {
public:
    // Applies a signed quantity change at a given per-contract price.
    //
    // A partial close releases a proportional share of the basis, so repeated
    // open and close cycles sum exactly to the difference between cash in and
    // cash out -- including when the position was built at several prices.
    ApplyFillResult apply(
        ContractVersionId cv, EquityKind kind, int64_t signed_qty,
        Money price_per_unit, int64_t multiplier, Timestamp at,
        Money fees = Money::zero(), Money spread_cost = Money::zero())
    {
        Position& p = positions_[cv.value];
        if (p.contract_version_id.value == 0 && p.quantity == 0) {
            p.contract_version_id = cv;
            p.kind = kind;
            p.id = PositionId{next_position_id_++};
            p.opened_at = at;
        }
        p.last_updated_at = at;

        ApplyFillResult out;
        // Cash value of one unit of quantity.
        const Money unit_value = Money{price_per_unit.micros * multiplier};

        const bool opposing = (p.quantity > 0 && signed_qty < 0) || (p.quantity < 0 && signed_qty > 0);
        if (opposing) {
            const int64_t closable = std::min(p.abs_quantity(),
                                              signed_qty < 0 ? -signed_qty : signed_qty);
            // Release basis with ONE exact division rather than truncating an
            // average and then multiplying it back up. Truncating first amplified
            // the error by the closed quantity, so realized P&L drifted from the
            // cash it produced -- about a microdollar per partial close, and only
            // once a position was built at more than one price, which is why
            // single-price tests never saw it. Dividing once leaves the remainder
            // in cost_basis, where the final close releases it exactly.
            const Money released = Money{p.cost_basis.micros * closable / p.abs_quantity()};
            // Proceeds of the closing trade, signed against the position.
            const Money proceeds = Money{unit_value.micros * (p.quantity > 0 ? closable : -closable)};

            // Entry-side costs release on the same proportion as the basis.
            const Money released_fees = Money{p.open_fees.micros * closable / p.abs_quantity()};
            const Money released_spread =
                Money{p.open_spread_cost.micros * closable / p.abs_quantity()};
            p.open_fees -= released_fees;
            p.open_spread_cost -= released_spread;

            out.closed_quantity = closable;
            out.realized_pnl = proceeds - released;
            p.realized_pnl += out.realized_pnl;
            p.cost_basis -= released;
            p.quantity += (signed_qty > 0 ? closable : -closable);

            const int64_t total = signed_qty < 0 ? -signed_qty : signed_qty;
            // A fill that crosses through zero splits its own cost between the
            // part that closed and the part that opened.
            const Money closing_fees = Money{fees.micros * closable / total};
            const Money closing_spread = Money{spread_cost.micros * closable / total};
            out.round_trip_fees = released_fees + closing_fees;
            out.round_trip_spread_cost = released_spread + closing_spread;

            const int64_t remainder = total - closable;
            if (remainder > 0) {
                const int64_t dir = signed_qty > 0 ? 1 : -1;
                p.quantity += dir * remainder;
                p.cost_basis += Money{unit_value.micros * dir * remainder};
                p.open_fees += fees - closing_fees;
                p.open_spread_cost += spread_cost - closing_spread;
                out.opened_quantity = remainder;
                p.opened_at = at;
            }
        } else {
            p.quantity += signed_qty;
            p.cost_basis += Money{unit_value.micros * signed_qty};
            p.open_fees += fees;
            p.open_spread_cost += spread_cost;
            out.opened_quantity = signed_qty < 0 ? -signed_qty : signed_qty;
        }

        if (p.quantity == 0) {
            realized_closed_ += p.realized_pnl;
            positions_.erase(cv.value);
        }
        return out;
    }

    Position* find(ContractVersionId cv) {
        auto it = positions_.find(cv.value);
        return it == positions_.end() ? nullptr : &it->second;
    }

    const Position* find(ContractVersionId cv) const {
        auto it = positions_.find(cv.value);
        return it == positions_.end() ? nullptr : &it->second;
    }

    int64_t quantity_of(ContractVersionId cv) const {
        const Position* p = find(cv);
        return p ? p->quantity : 0;
    }

    std::vector<Position> snapshot() const {
        std::vector<Position> out;
        out.reserve(positions_.size());
        for (const auto& [_, p] : positions_) out.push_back(p);
        std::sort(out.begin(), out.end(),
                  [](const Position& a, const Position& b) { return a.id.value < b.id.value; });
        return out;
    }

    size_t open_count() const { return positions_.size(); }

    Money realized_pnl_total() const {
        Money total = realized_closed_;
        for (const auto& [_, p] : positions_) total += p.realized_pnl;
        return total;
    }

    // Equity leg handling. Average-cost basis, and a sale that crosses through
    // zero opens a short rather than silently clamping.
    void apply_equity(const std::string& symbol, int64_t signed_shares, Money price) {
        EquityPosition& e = equities_[symbol];
        e.symbol = symbol;
        const bool opposing = (e.shares > 0 && signed_shares < 0) || (e.shares < 0 && signed_shares > 0);
        if (opposing) {
            const int64_t held = e.shares < 0 ? -e.shares : e.shares;
            const int64_t closable = std::min(
                held, signed_shares < 0 ? -signed_shares : signed_shares);
            const int64_t dir = e.shares > 0 ? 1 : -1;
            // Same single-division release as the option path.
            const Money released = Money{e.cost_basis.micros * closable / held};
            const Money proceeds = Money{price.micros * dir * closable};
            e.realized_pnl += proceeds - released;
            e.cost_basis -= released;
            e.shares -= dir * closable;
            const int64_t remainder = (signed_shares < 0 ? -signed_shares : signed_shares) - closable;
            if (remainder > 0) {
                const int64_t open_dir = signed_shares > 0 ? 1 : -1;
                e.shares += open_dir * remainder;
                e.cost_basis += Money{price.micros * open_dir * remainder};
            }
        } else {
            e.shares += signed_shares;
            e.cost_basis += Money{price.micros * signed_shares};
        }
        if (e.shares == 0) {
            equity_realized_closed_ += e.realized_pnl;
            equities_.erase(symbol);
        }
    }

    int64_t shares_of(const std::string& symbol) const {
        auto it = equities_.find(symbol);
        return it == equities_.end() ? 0 : it->second.shares;
    }

    const EquityPosition* find_equity(const std::string& symbol) const {
        auto it = equities_.find(symbol);
        return it == equities_.end() ? nullptr : &it->second;
    }

    std::vector<EquityPosition> equity_snapshot() const {
        std::vector<EquityPosition> out;
        for (const auto& [_, e] : equities_) out.push_back(e);
        std::sort(out.begin(), out.end(),
                  [](const EquityPosition& a, const EquityPosition& b) { return a.symbol < b.symbol; });
        return out;
    }

    Money equity_realized_pnl() const {
        Money total = equity_realized_closed_;
        for (const auto& [_, e] : equities_) total += e.realized_pnl;
        return total;
    }

    void clear() {
        positions_.clear();
        equities_.clear();
        realized_closed_ = Money::zero();
        equity_realized_closed_ = Money::zero();
        next_position_id_ = 1;
    }

private:
    std::unordered_map<uint64_t, Position> positions_;
    std::unordered_map<std::string, EquityPosition> equities_;
    Money realized_closed_{};
    Money equity_realized_closed_{};
    uint64_t next_position_id_ = 1;
};

} // namespace obt
