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

// Consumes shares to cover as many contracts of a naked short call as the
// holding allows, splitting the pairing when it covers only part of it. Without
// the split, 100 shares against 2 short calls would cover neither and both
// contracts would be charged as naked.
//
// Returns the still-uncovered contract count, appending a covered pairing to
// `out` when at least one contract was covered.
inline int64_t consume_share_coverage(
    SpreadPairing& p, int64_t shares_per_contract, int64_t& free_shares,
    std::vector<SpreadPairing>& out)
{
    if (shares_per_contract <= 0 || free_shares < shares_per_contract) return p.contracts;

    const int64_t coverable = std::min(p.contracts, free_shares / shares_per_contract);
    free_shares -= coverable * shares_per_contract;
    if (coverable == 0) return p.contracts;

    SpreadPairing covered = p;
    covered.contracts = coverable;
    covered.covered_by_equity = true;
    covered.naked = false;
    covered.requirement = Money::zero();
    out.push_back(covered);

    return p.contracts - coverable;
}

struct LegView {
    ContractVersionId id{};
    const OptionContractVersion* contract = nullptr;
    int64_t contracts = 0;   // always positive
};

// A leg with its signed quantity, for max-loss evaluation.
struct SignedLeg {
    const OptionContractVersion* contract = nullptr;
    int64_t quantity = 0;   // negative for short
};

inline Money intrinsic_at(const OptionContractVersion& c, Money underlying) {
    return c.type == OptionType::Call
        ? max_money(underlying - c.strike, Money::zero())
        : max_money(c.strike - underlying, Money::zero());
}

// Greatest loss of a spread, evaluated at every strike present in it.
//
// This is the method FINRA 4210(f)(2)(H)(i) prescribes: compute the intrinsic
// value of each leg at price points corresponding to every exercise price in the
// spread, net them at each point, and take the worst result. Summing pairwise
// widths instead would double-charge an iron condor, whose two wings cannot both
// lose at once -- the requirement is the wider side, not the sum.
//
// Payoff is piecewise linear in the underlying with breakpoints only at strikes,
// so the extremum over the whole price line is attained at one of them.
inline Money max_potential_loss(const std::vector<SignedLeg>& legs) {
    Money worst = Money::zero();
    for (const SignedLeg& point : legs) {
        Money net = Money::zero();
        for (const SignedLeg& leg : legs) {
            const Money value = intrinsic_at(*leg.contract, point.contract->strike);
            net += Money{value.micros * leg.contract->deliverable_shares_per_contract()
                         * leg.quantity};
        }
        if (net < worst) worst = net;
    }
    return Money{-worst.micros};
}

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
struct PairingOutcome {
    std::vector<SpreadPairing> pairings;
    // Legs that entered a spread, for joint max-loss evaluation.
    std::vector<SignedLeg> matched;
};

inline std::vector<SpreadPairing> pair_legs(
    std::vector<LegView> shorts, std::vector<LegView> longs, OptionType type,
    std::vector<SignedLeg>* matched = nullptr)
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
                // Spread treatment requires the aggregate underlying value of the
                // long and short sides to be equal (FINRA 4210(f)(2)(A)(xxxii)(d)).
                // Pairing one contract against one contract only achieves that when
                // both deliver the same number of shares. A 100-share long against a
                // 400-share short leaves 300 shares of naked exposure whose loss is
                // unbounded, and because the payoff slopes then fail to cancel, the
                // max-loss netting below would not even see it: evaluating at the two
                // strikes returns a net gain. Refuse the pairing so the short falls
                // through to the naked charge.
                if (l.contract->deliverable_shares_per_contract()
                    != s.contract->deliverable_shares_per_contract()) continue;
                const Money residual = (type == OptionType::Call)
                    ? max_money(l.contract->strike - s.contract->strike, Money::zero())
                    : max_money(s.contract->strike - l.contract->strike, Money::zero());
                // Lowest residual wins; ties go to the EARLIEST-expiring eligible
                // long, which conserves long-dated longs for the shorts that need
                // them. Without the tie-break the winner was whichever long
                // happened to be encountered first -- position-id order, hence
                // the order the legs were opened -- so the same portfolio was
                // legal or refused depending on how it was assembled.
                const bool better =
                    best == nullptr
                    || residual < best_residual
                    || (residual == best_residual
                        && l.contract->expiration < best->contract->expiration);
                if (better) {
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
            const int64_t paired = std::min(remaining, best->contracts);
            const int64_t shares = s.contract->deliverable_shares_per_contract();
            SpreadPairing p;
            p.short_leg = s.id;
            p.long_leg = best->id;
            p.contracts = paired;
            // Pairwise residual, kept for reporting. The charged amount comes
            // from joint max-loss netting, which is smaller when wings offset.
            p.requirement = Money{best_residual.micros * shares * paired};
            out.push_back(p);
            if (matched != nullptr) {
                matched->push_back(SignedLeg{s.contract, -paired});
                matched->push_back(SignedLeg{best->contract, paired});
            }
            best->contracts -= paired;
            remaining -= paired;
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
// How a short leg with no long partner and no share coverage is treated.
enum class NakedPolicy : uint8_t {
    // The account cannot carry it at all, so the position is refused.
    Disallow,
    // Regulation T / FINRA 4210(f)(2)(E)(i) minimum.
    RegTMinimum,
    // Fully collateralized at the strike, which is what a retail broker holds
    // for a short put.
    FullStrike,
};

struct NakedRules {
    NakedPolicy call = NakedPolicy::Disallow;
    NakedPolicy put = NakedPolicy::FullStrike;
    const char* call_refusal = "uncovered short call is not permitted";
    const char* put_refusal = "uncovered short put is not permitted";

    // Equity legs. Stock previously carried no requirement at all, long or
    // short, so $500k of short stock required nothing -- and shares arrive on
    // every assignment, which is how the covered call, PMCC and collar families
    // all ended up with an unmargined book.
    //
    // Reg-T initial margin on long stock is 50% (12 CFR 220.12(a)); a cash
    // account pays in full. A short sale requires 100% of the proceeds plus 50%
    // margin, so 150% of market value.
    double long_stock_fraction = 0.50;
    double short_stock_fraction = 1.50;
    bool allow_short_stock = true;
    const char* short_stock_refusal = "short stock is not permitted";
};

// Shared evaluation. The three published models differ only in what they do
// with an uncovered short, so the pairing, share-coverage, and max-loss netting
// logic lives here once rather than three times.
inline MarginResult evaluate_with(
    const PositionBook& book, const ContractRegistry& registry,
    const MarginContext& ctx, const NakedRules& rules)
{
    MarginResult res;
    auto groups = detail::split_by_underlying(book.snapshot(), registry);

    // Equity legs are charged on the full holding, independently of whether some
    // of it also collateralizes a short call: a covered call requires margin on
    // the stock and nothing extra on the option, not the reverse.
    for (const EquityPosition& e : book.equity_snapshot()) {
        if (e.shares == 0) continue;
        const Money spot = ctx.underlying_or_zero(e.symbol);
        if (e.shares > 0) {
            res.requirement += scale(Money{spot.micros * e.shares}, rules.long_stock_fraction);
        } else {
            if (!rules.allow_short_stock) {
                res.disallowed = true;
                res.disallowed_reason = rules.short_stock_refusal;
            }
            res.requirement += scale(Money{spot.micros * (-e.shares)},
                                     rules.short_stock_fraction);
        }
    }

    for (auto& [underlying, legs] : groups) {
        const Money spot = ctx.underlying_or_zero(underlying);
        int64_t free_shares = book.shares_of(underlying);

        std::vector<detail::SignedLeg> matched;
        auto call_pairs = detail::pair_legs(legs.short_calls, legs.long_calls,
                                            OptionType::Call, &matched);
        auto put_pairs = detail::pair_legs(legs.short_puts, legs.long_puts,
                                           OptionType::Put, &matched);

        // Every spread leg on this underlying is netted together, so offsetting
        // wings are not both charged.
        if (!matched.empty()) res.requirement += detail::max_potential_loss(matched);

        auto settle_naked = [&](std::vector<SpreadPairing>& pairings, OptionType type) {
            const NakedPolicy policy = type == OptionType::Call ? rules.call : rules.put;
            const char* refusal = type == OptionType::Call ? rules.call_refusal
                                                           : rules.put_refusal;
            for (auto& p : pairings) {
                if (!p.naked) {
                    res.pairings.push_back(p);
                    continue;
                }
                const OptionContractVersion* c = registry.find(p.short_leg);
                if (c == nullptr) continue;

                if (type == OptionType::Call) {
                    p.contracts = detail::consume_share_coverage(
                        p, c->deliverable_shares_per_contract(), free_shares, res.pairings);
                    if (p.contracts == 0) continue;
                }

                switch (policy) {
                    case NakedPolicy::Disallow:
                        res.disallowed = true;
                        res.disallowed_reason = refusal;
                        break;
                    case NakedPolicy::RegTMinimum:
                        p.requirement = Money{
                            detail::reg_t_naked_requirement(
                                *c, spot, ctx.mark_or_zero(p.short_leg)).micros * p.contracts};
                        res.requirement += p.requirement;
                        break;
                    case NakedPolicy::FullStrike:
                        p.requirement = Money{c->strike.micros
                            * c->deliverable_shares_per_contract() * p.contracts};
                        res.requirement += p.requirement;
                        break;
                }
                res.pairings.push_back(p);
            }
        };
        settle_naked(call_pairs, OptionType::Call);
        settle_naked(put_pairs, OptionType::Put);
    }
    return res;
}

// Long options paid in full, short options fully collateralized: short puts by
// cash equal to the strike, short calls by shares. No borrowing, so an
// unsecured short call is refused rather than charged.
class CashAccountMargin : public MarginModel {
public:
    MarginModelKind kind() const override { return MarginModelKind::CashAccount; }
    const char* name() const override { return "cash_account"; }

    MarginResult evaluate(const PositionBook& book, const ContractRegistry& registry,
                          const MarginContext& ctx) const override
    {
        NakedRules rules{
            NakedPolicy::Disallow, NakedPolicy::FullStrike,
            "cash account cannot hold an uncovered short call",
            "cash account short put must be cash secured"};
        rules.long_stock_fraction = 1.00;
        rules.allow_short_stock = false;
        rules.short_stock_refusal = "a cash account cannot sell stock short";
        return evaluate_with(book, registry, ctx, rules);
    }
};

// Regulation T / FINRA 4210 minimums. Spreads are charged their maximum loss;
// uncovered shorts use the 20%/10% formula.
class RegTMargin : public MarginModel {
public:
    MarginModelKind kind() const override { return MarginModelKind::RegT; }
    const char* name() const override { return "reg_t"; }

    MarginResult evaluate(const PositionBook& book, const ContractRegistry& registry,
                          const MarginContext& ctx) const override
    {
        return evaluate_with(book, registry, ctx, NakedRules{
            NakedPolicy::RegTMinimum, NakedPolicy::RegTMinimum, "", ""});
    }
};

// Robinhood: Reg-T spread treatment, but retail accounts are not permitted
// uncovered short calls at any approval level, and a short put is held at the
// full strike rather than the Reg-T percentage.
class RobinhoodMargin : public MarginModel {
public:
    explicit RobinhoodMargin(bool allow_uncovered_calls = false)
        : allow_uncovered_calls_(allow_uncovered_calls) {}

    MarginModelKind kind() const override { return MarginModelKind::Robinhood; }
    const char* name() const override { return "robinhood"; }

    MarginResult evaluate(const PositionBook& book, const ContractRegistry& registry,
                          const MarginContext& ctx) const override
    {
        return evaluate_with(book, registry, ctx, NakedRules{
            allow_uncovered_calls_ ? NakedPolicy::RegTMinimum : NakedPolicy::Disallow,
            NakedPolicy::FullStrike,
            "uncovered short call is not permitted at any retail approval level",
            ""});
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
