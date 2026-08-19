#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "contract.h"
#include "position.h"
#include "types.h"

namespace obt {

enum class MarginModelKind : uint8_t { CashAccount, RegT, Robinhood, PortfolioApprox };

// One short option and the long option, if any, that offsets it.
struct SpreadPairing {
    ContractVersionId short_leg{};
    ContractVersionId long_leg{};   // zero value when unpaired
    int64_t contracts = 0;
    Money requirement{};
    bool covered_by_equity = false;
    bool naked = false;
};

struct MarginResult {
    Money requirement{};
    // Set when the broker model forbids the position outright rather than
    // merely charging for it, e.g. a retail naked short call.
    bool disallowed = false;
    std::string disallowed_reason;
    std::vector<SpreadPairing> pairings;
};

// Everything the margin model needs about the market at valuation time.
struct MarginContext {
    // Underlying price per share, keyed by underlying symbol.
    std::unordered_map<std::string, Money> underlying_price;
    // Current mark per share for each contract version.
    std::unordered_map<uint64_t, Money> mark;

    Money underlying_or_zero(const std::string& sym) const {
        auto it = underlying_price.find(sym);
        return it == underlying_price.end() ? Money::zero() : it->second;
    }
    Money mark_or_zero(ContractVersionId cv) const {
        auto it = mark.find(cv.value);
        return it == mark.end() ? Money::zero() : it->second;
    }
};

namespace detail {

struct LegView {
    ContractVersionId id{};
    const OptionContractVersion* contract = nullptr;
    int64_t contracts = 0;   // always positive
};

// Reg-T / CBOE minimum for an uncovered short option.
//
//   call: max(20% * underlying - out_of_the_money, 10% * underlying) * shares
//   put:  max(20% * underlying - out_of_the_money, 10% * strike)     * shares
//
// The premium received is added, because the requirement is stated against the
// proceeds of the sale.
inline Money reg_t_naked_requirement(
    const OptionContractVersion& c, Money underlying, Money premium_per_share)
{
    const int64_t shares = c.deliverable_shares_per_contract();
    const Money strike = c.strike;

    Money out_of_the_money = Money::zero();
    if (c.type == OptionType::Call) {
        if (strike > underlying) out_of_the_money = strike - underlying;
    } else {
        if (underlying > strike) out_of_the_money = underlying - strike;
    }

    const Money primary = scale(underlying, 0.20) - out_of_the_money;
    const Money floor_amount = (c.type == OptionType::Call) ? scale(underlying, 0.10)
                                                            : scale(strike, 0.10);
    const Money per_share = max_money(max_money(primary, floor_amount), Money::zero())
                          + premium_per_share;
    return Money{per_share.micros * shares};
}

// Pairs each short option with the long option that best offsets it.
//
// A long only offsets a short if it lives at least as long: a short call
// "covered" by a long call that expires first is genuinely naked for the
// remaining period, which is the trap the previous engine fell into. Among
// eligible longs, the one whose strike minimizes the residual risk is used.
//
// Residual per-contract risk for a call pair is max(0, K_long - K_short) and
// for a put pair max(0, K_short - K_long). A poor man's covered call therefore
// pairs to zero residual, because its long LEAP sits below the short strike and
// expires later. A bear call spread pairs to the strike width.
inline std::vector<SpreadPairing> pair_legs(
    std::vector<LegView> shorts, std::vector<LegView> longs, OptionType type)
{
    // Shortest-dated shorts first, so a scarce long-dated long is offered to
    // the short that most needs it.
    std::sort(shorts.begin(), shorts.end(), [](const LegView& a, const LegView& b) {
        if (a.contract->expiration != b.contract->expiration)
            return a.contract->expiration < b.contract->expiration;
        return a.contract->strike < b.contract->strike;
    });

    std::vector<SpreadPairing> out;
    for (LegView& s : shorts) {
        int64_t remaining = s.contracts;
        while (remaining > 0) {
            LegView* best = nullptr;
            Money best_residual = Money::zero();
            for (LegView& l : longs) {
                if (l.contracts <= 0) continue;
                if (l.contract->expiration < s.contract->expiration) continue;
                const Money residual = (type == OptionType::Call)
                    ? max_money(l.contract->strike - s.contract->strike, Money::zero())
                    : max_money(s.contract->strike - l.contract->strike, Money::zero());
                if (best == nullptr || residual < best_residual) {
                    best = &l;
                    best_residual = residual;
                }
            }
            if (best == nullptr) {
                SpreadPairing p;
                p.short_leg = s.id;
                p.contracts = remaining;
                p.naked = true;
                out.push_back(p);
                break;
            }
            const int64_t matched = std::min(remaining, best->contracts);
            const int64_t shares = s.contract->deliverable_shares_per_contract();
            SpreadPairing p;
            p.short_leg = s.id;
            p.long_leg = best->id;
            p.contracts = matched;
            p.requirement = Money{best_residual.micros * shares * matched};
            out.push_back(p);
            best->contracts -= matched;
            remaining -= matched;
        }
    }
    return out;
}

// Splits open option positions into long and short views per underlying+type.
struct SplitLegs {
    std::vector<LegView> long_calls, short_calls, long_puts, short_puts;
};

inline std::unordered_map<std::string, SplitLegs> split_by_underlying(
    const std::vector<Position>& positions, const ContractRegistry& registry)
{
    std::unordered_map<std::string, SplitLegs> out;
    for (const Position& p : positions) {
        if (p.kind != EquityKind::Option || p.quantity == 0) continue;
        const OptionContractVersion* c = registry.find(p.contract_version_id);
        if (c == nullptr) continue;
        LegView v{p.contract_version_id, c, p.abs_quantity()};
        SplitLegs& s = out[c->underlying_symbol];
        if (c->type == OptionType::Call) {
            (p.quantity > 0 ? s.long_calls : s.short_calls).push_back(v);
        } else {
            (p.quantity > 0 ? s.long_puts : s.short_puts).push_back(v);
        }
    }
    return out;
}

} // namespace detail

class MarginModel {
public:
    virtual ~MarginModel() = default;
    virtual MarginModelKind kind() const = 0;
    virtual const char* name() const = 0;
    virtual MarginResult evaluate(
        const PositionBook& book, const ContractRegistry& registry,
        const MarginContext& ctx) const = 0;
};

// Long options paid in full, short options fully collateralized: short puts by
// cash equal to the strike, short calls by shares. No borrowing, so an
// unsecured short is disallowed rather than charged.
class CashAccountMargin : public MarginModel {
public:
    MarginModelKind kind() const override { return MarginModelKind::CashAccount; }
    const char* name() const override { return "cash_account"; }

    MarginResult evaluate(
        const PositionBook& book, const ContractRegistry& registry,
        const MarginContext& ctx) const override
    {
        MarginResult res;
        auto groups = detail::split_by_underlying(book.snapshot(), registry);
        for (auto& [underlying, legs] : groups) {
            int64_t free_shares = book.shares_of(underlying);

            for (auto& p : detail::pair_legs(legs.short_calls, legs.long_calls, OptionType::Call)) {
                if (!p.naked) { res.requirement += p.requirement; res.pairings.push_back(p); continue; }
                const OptionContractVersion* c = registry.find(p.short_leg);
                const int64_t need = c->deliverable_shares_per_contract() * p.contracts;
                if (free_shares >= need) {
                    free_shares -= need;
                    p.covered_by_equity = true;
                } else {
                    res.disallowed = true;
                    res.disallowed_reason = "cash account cannot hold an uncovered short call";
                }
                res.pairings.push_back(p);
            }

            for (auto& p : detail::pair_legs(legs.short_puts, legs.long_puts, OptionType::Put)) {
                if (!p.naked) { res.requirement += p.requirement; res.pairings.push_back(p); continue; }
                const OptionContractVersion* c = registry.find(p.short_leg);
                // Cash-secured: the full strike must be set aside.
                p.requirement = Money{c->strike.micros * c->deliverable_shares_per_contract() * p.contracts};
                res.requirement += p.requirement;
                res.pairings.push_back(p);
            }
        }
        return res;
    }
};

// Regulation T minimums as published by CBOE/FINRA. Spreads are charged their
// maximum loss; uncovered shorts use the 20%/10% formula.
class RegTMargin : public MarginModel {
public:
    MarginModelKind kind() const override { return MarginModelKind::RegT; }
    const char* name() const override { return "reg_t"; }

    MarginResult evaluate(
        const PositionBook& book, const ContractRegistry& registry,
        const MarginContext& ctx) const override
    {
        MarginResult res;
        auto groups = detail::split_by_underlying(book.snapshot(), registry);
        for (auto& [underlying, legs] : groups) {
            const Money spot = ctx.underlying_or_zero(underlying);
            int64_t free_shares = book.shares_of(underlying);

            auto charge = [&](std::vector<SpreadPairing> pairings) {
                for (auto& p : pairings) {
                    if (!p.naked) {
                        res.requirement += p.requirement;
                        res.pairings.push_back(p);
                        continue;
                    }
                    const OptionContractVersion* c = registry.find(p.short_leg);
                    const int64_t need = c->deliverable_shares_per_contract() * p.contracts;
                    if (c->type == OptionType::Call && free_shares >= need) {
                        free_shares -= need;
                        p.covered_by_equity = true;
                    } else {
                        p.requirement = Money{
                            detail::reg_t_naked_requirement(*c, spot, ctx.mark_or_zero(p.short_leg)).micros
                            * p.contracts};
                        res.requirement += p.requirement;
                    }
                    res.pairings.push_back(p);
                }
            };
            charge(detail::pair_legs(legs.short_calls, legs.long_calls, OptionType::Call));
            charge(detail::pair_legs(legs.short_puts, legs.long_puts, OptionType::Put));
        }
        return res;
    }
};

// Robinhood: Reg-T spread and cash-secured treatment, but retail accounts are
// not permitted uncovered short calls at any approval level, so those are
// refused rather than margined.
class RobinhoodMargin : public MarginModel {
public:
    explicit RobinhoodMargin(bool allow_uncovered_calls = false)
        : allow_uncovered_calls_(allow_uncovered_calls) {}

    MarginModelKind kind() const override { return MarginModelKind::Robinhood; }
    const char* name() const override { return "robinhood"; }

    MarginResult evaluate(
        const PositionBook& book, const ContractRegistry& registry,
        const MarginContext& ctx) const override
    {
        MarginResult res;
        auto groups = detail::split_by_underlying(book.snapshot(), registry);
        for (auto& [underlying, legs] : groups) {
            const Money spot = ctx.underlying_or_zero(underlying);
            int64_t free_shares = book.shares_of(underlying);

            for (auto& p : detail::pair_legs(legs.short_calls, legs.long_calls, OptionType::Call)) {
                if (!p.naked) { res.requirement += p.requirement; res.pairings.push_back(p); continue; }
                const OptionContractVersion* c = registry.find(p.short_leg);
                const int64_t need = c->deliverable_shares_per_contract() * p.contracts;
                if (free_shares >= need) {
                    free_shares -= need;
                    p.covered_by_equity = true;
                } else if (allow_uncovered_calls_) {
                    p.requirement = Money{
                        detail::reg_t_naked_requirement(*c, spot, ctx.mark_or_zero(p.short_leg)).micros
                        * p.contracts};
                    res.requirement += p.requirement;
                } else {
                    res.disallowed = true;
                    res.disallowed_reason =
                        "uncovered short call is not permitted at any retail approval level";
                }
                res.pairings.push_back(p);
            }

            for (auto& p : detail::pair_legs(legs.short_puts, legs.long_puts, OptionType::Put)) {
                if (!p.naked) { res.requirement += p.requirement; res.pairings.push_back(p); continue; }
                const OptionContractVersion* c = registry.find(p.short_leg);
                // Short puts are collateralized at the full strike.
                p.requirement = Money{
                    c->strike.micros * c->deliverable_shares_per_contract() * p.contracts};
                res.requirement += p.requirement;
                res.pairings.push_back(p);
            }
        }
        return res;
    }

private:
    bool allow_uncovered_calls_;
};

inline std::unique_ptr<MarginModel> make_margin_model(MarginModelKind kind) {
    switch (kind) {
        case MarginModelKind::CashAccount: return std::make_unique<CashAccountMargin>();
        case MarginModelKind::RegT: return std::make_unique<RegTMargin>();
        case MarginModelKind::Robinhood: return std::make_unique<RobinhoodMargin>();
        case MarginModelKind::PortfolioApprox: return std::make_unique<RegTMargin>();
    }
    return std::make_unique<RobinhoodMargin>();
}

} // namespace obt
