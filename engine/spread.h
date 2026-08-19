#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "types.h"

namespace obt {

// ---------------------------------------------------------------------------
// Counter-based RNG
// ---------------------------------------------------------------------------
// Draws are a pure function of the key (seed, scenario, order, instrument,
// timestamp, leg). Nothing is carried between calls, which buys three
// properties a stateful generator cannot:
//
//   - order independence: evaluating orders in a different sequence, or in
//     parallel, yields identical draws
//   - common random numbers: two strategies that place the same order at the
//     same instant on the same scenario draw the same spread, so a comparison
//     between them is not polluted by execution-cost noise
//   - resumability: a scenario can be recomputed from its key alone
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct DrawKey {
    uint64_t seed = 0;
    uint32_t scenario_id = 0;
    uint64_t order_id = 0;
    uint64_t instrument = 0;
    int64_t timestamp_ns = 0;
    uint32_t leg_index = 0;
};

inline uint64_t draw_bits(const DrawKey& k) {
    uint64_t h = splitmix64(k.seed ^ 0xa5a5a5a5a5a5a5a5ULL);
    h = splitmix64(h ^ (static_cast<uint64_t>(k.scenario_id) * 0x9e3779b1ULL));
    h = splitmix64(h ^ k.order_id);
    h = splitmix64(h ^ k.instrument);
    h = splitmix64(h ^ static_cast<uint64_t>(k.timestamp_ns));
    h = splitmix64(h ^ (static_cast<uint64_t>(k.leg_index) + 0x165667b1ULL));
    return h;
}

// Uniform on [0,1) with 53 bits of resolution.
inline double uniform01(const DrawKey& k) {
    return static_cast<double>(draw_bits(k) >> 11) * (1.0 / 9007199254740992.0);
}

// Standard normal via Box-Muller on two independent uniforms from the same key.
inline double standard_normal(const DrawKey& k) {
    DrawKey k2 = k;
    k2.leg_index = k.leg_index ^ 0x5bf03635u;
    const double u1 = std::max(uniform01(k), 1e-300);
    const double u2 = uniform01(k2);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// ---------------------------------------------------------------------------
// Spread models
// ---------------------------------------------------------------------------
enum class SpreadModelKind : uint8_t {
    // Every path equals the deterministic no-cost result. Used to prove that
    // market-data P&L is separable from execution cost.
    Zero,
    ConstantCents,
    ProportionalBps,
    Lognormal,
    ConditionalLognormal,
    Empirical,
};

// Observable features a conditional model may use. All are known at or before
// the fill timestamp.
struct SpreadFeatures {
    double mark_dollars = 0.0;
    double implied_volatility = 0.0;
    double days_to_expiry = 0.0;
    double moneyness = 0.0;      // log(strike / underlying)
    double volume = 0.0;
    double trade_count = 0.0;
    double underlying_dollars = 0.0;
    double minutes_from_open = 0.0;
    bool is_call = true;
};

struct SpreadModelConfig {
    SpreadModelKind kind = SpreadModelKind::ConditionalLognormal;

    // ConstantCents
    double constant_cents = 5.0;

    // ProportionalBps: full spread in basis points of the mark.
    double proportional_bps = 60.0;

    // Lognormal / ConditionalLognormal, in log-basis-point space.
    //
    // log_base is the log of the median full spread in basis points AT THE
    // REFERENCE POINT below. The conditional terms are deviations from that
    // reference, so this constant is directly interpretable and the documented
    // magnitude is true by construction. Previously the betas were absolute, so
    // log_base = 4.0 was documented as ~55 bps while actually producing 13 bps at
    // any realistic volume -- which the min-spread floor then clamped away
    // entirely, collapsing the whole Monte Carlo onto a single value.
    double log_base = 4.007;        // exp(4.007) = 55 bps at the reference point
    double log_sigma = 0.45;

    // Reference point the conditional betas are measured against.
    double ref_implied_volatility = 0.15;
    double ref_days_to_expiry = 30.0;
    double ref_volume = 5000.0;

    // Scales the dispersion of the drawn spread without touching its central
    // tendency. 1.0 leaves the model as calibrated; 2.0 doubles the log-space
    // standard deviation; 0.0 collapses every draw onto the mean, which turns the
    // Monte Carlo into a single deterministic path.
    //
    // This is the knob for asking "how much of my result is owned by the spread
    // assumption?". Because a lognormal's mean is exp(mu + sigma^2/2), changing
    // sigma alone would also move the mean and confound level with dispersion, so
    // preserve_mean_under_variance_scale compensates mu to hold E[spread] fixed.
    double variance_scale = 1.0;
    bool preserve_mean_under_variance_scale = true;
    double beta_iv = 0.90;
    double beta_log_dte = -0.15;
    double beta_log_volume = -0.12;
    double beta_abs_moneyness = 1.80;
    double beta_minutes_from_open = -0.0004;

    // Empirical: sampled half-spreads in cents, drawn uniformly.
    std::vector<double> empirical_half_spread_cents;

    // Equity legs. A share quote is a different animal from an option quote: the
    // grid is a penny at every price, there is no expiry or implied volatility to
    // condition on, and a liquid name trades a penny wide -- which on a $100 stock
    // is one basis point of full spread, two orders of magnitude tighter than the
    // 55 bps the option model centres on. Pricing shares through the option model
    // would have charged a covered call more to buy its stock than to sell its
    // call.
    double equity_full_spread_bps = 1.0;
    double equity_log_sigma = 0.35;
    double equity_min_half_spread_cents = 0.5;

    // Guards. A real quote is never tighter than a tick and a spread wider than
    // the option is worth would let a fill go through zero.
    double min_half_spread_cents = 0.5;
    double max_fraction_of_mark = 0.50;
    // Quote grid: $0.01 below $3.00, $0.05 at or above, per exchange convention.
    bool round_to_tick = true;

    double effective_sigma() const {
        return log_sigma * (variance_scale < 0.0 ? 0.0 : variance_scale);
    }

    // Shifts mu so that scaling sigma does not move the distribution's mean.
    // For a lognormal, mean = exp(mu + sigma^2/2), so holding the mean fixed
    // while moving sigma requires mu' = mu + (sigma_0^2 - sigma^2)/2.
    double mean_preserving_mu(double mu, double sigma) const {
        if (!preserve_mean_under_variance_scale) return mu;
        return mu + 0.5 * (log_sigma * log_sigma - sigma * sigma);
    }
};

inline double tick_size_dollars(double mark_dollars) {
    return mark_dollars < 3.0 ? 0.01 : 0.05;
}

// Half of the bid/ask spread, in dollars per share. Always nonnegative, so a
// buy never fills below the mark and a sell never fills above it.
inline double half_spread_dollars(
    const SpreadModelConfig& cfg, const SpreadFeatures& f, const DrawKey& key)
{
    double half = 0.0;
    switch (cfg.kind) {
        case SpreadModelKind::Zero:
            return 0.0;

        case SpreadModelKind::ConstantCents:
            half = cfg.constant_cents / 200.0;
            break;

        case SpreadModelKind::ProportionalBps:
            half = f.mark_dollars * cfg.proportional_bps / 20000.0;
            break;

        case SpreadModelKind::Lognormal: {
            const double sigma = cfg.effective_sigma();
            const double bps = std::exp(cfg.mean_preserving_mu(cfg.log_base, sigma)
                                        + sigma * standard_normal(key));
            half = f.mark_dollars * bps / 20000.0;
            break;
        }

        case SpreadModelKind::ConditionalLognormal: {
            const double mu = cfg.log_base
                + cfg.beta_iv * (f.implied_volatility - cfg.ref_implied_volatility)
                + cfg.beta_log_dte * (std::log1p(std::max(0.0, f.days_to_expiry))
                                      - std::log1p(cfg.ref_days_to_expiry))
                + cfg.beta_log_volume * (std::log1p(std::max(0.0, f.volume))
                                         - std::log1p(cfg.ref_volume))
                + cfg.beta_abs_moneyness * std::fabs(f.moneyness)
                + cfg.beta_minutes_from_open * f.minutes_from_open;
            const double sigma = cfg.effective_sigma();
            const double bps = std::exp(cfg.mean_preserving_mu(mu, sigma)
                                        + sigma * standard_normal(key));
            half = f.mark_dollars * bps / 20000.0;
            break;
        }

        case SpreadModelKind::Empirical: {
            if (cfg.empirical_half_spread_cents.empty()) return 0.0;
            const size_t n = cfg.empirical_half_spread_cents.size();
            const size_t i = static_cast<size_t>(uniform01(key) * static_cast<double>(n));
            half = cfg.empirical_half_spread_cents[std::min(i, n - 1)] / 100.0;
            break;
        }
    }

    // Cap first, then quantize to the quote grid, then enforce the floor. The
    // floor has to come last: applying it before rounding let half a cent round
    // down to zero on the five-cent grid, so every option priced at or above
    // $3.00 executed at exactly the mark with no spread cost. The floor is itself
    // snapped up to the grid so the result stays a legal quote.
    half = std::min(half, f.mark_dollars * cfg.max_fraction_of_mark);
    half = std::max(half, 0.0);

    const double tick = tick_size_dollars(f.mark_dollars);
    if (cfg.round_to_tick && half > 0.0) {
        // The full spread lands on the grid, so a half-spread lands on a half-tick.
        half = std::round(2.0 * half / tick) * tick / 2.0;
    }

    double floor_dollars = cfg.min_half_spread_cents / 100.0;
    if (cfg.round_to_tick && floor_dollars > 0.0) {
        floor_dollars = std::ceil(2.0 * floor_dollars / tick) * tick / 2.0;
    }
    return std::max(half, floor_dollars);
}

// Half of the bid/ask spread on an equity leg, in dollars per share.
//
// Drawn through the same counter-based generator as the option spread, so an
// equity fill participates in the Monte Carlo and stays a pure function of
// (seed, scenario, order, instrument, timestamp, leg) rather than depending on
// evaluation order. The share grid is a penny at every price, so the half-spread
// lands on a half-cent.
inline double equity_half_spread_dollars(
    const SpreadModelConfig& cfg, double price_dollars, const DrawKey& key)
{
    if (cfg.kind == SpreadModelKind::Zero || price_dollars <= 0.0) return 0.0;

    double half = price_dollars * cfg.equity_full_spread_bps / 20000.0;
    const bool stochastic = cfg.kind == SpreadModelKind::Lognormal
                         || cfg.kind == SpreadModelKind::ConditionalLognormal;
    if (stochastic) {
        const double sigma = cfg.equity_log_sigma
            * (cfg.variance_scale < 0.0 ? 0.0 : cfg.variance_scale);
        // Same mean-preserving correction as the option path, so scaling the
        // variance moves dispersion without moving the expected cost.
        double mu = std::log(std::max(half, 1e-12));
        if (cfg.preserve_mean_under_variance_scale)
            mu += 0.5 * (cfg.equity_log_sigma * cfg.equity_log_sigma - sigma * sigma);
        half = std::exp(mu + sigma * standard_normal(key));
    }

    half = std::min(half, price_dollars * cfg.max_fraction_of_mark);
    half = std::max(half, 0.0);

    constexpr double kEquityTick = 0.01;
    if (cfg.round_to_tick && half > 0.0)
        half = std::round(2.0 * half / kEquityTick) * kEquityTick / 2.0;

    double floor_dollars = cfg.equity_min_half_spread_cents / 100.0;
    if (cfg.round_to_tick && floor_dollars > 0.0)
        floor_dollars = std::ceil(2.0 * floor_dollars / kEquityTick) * kEquityTick / 2.0;
    return std::max(half, floor_dollars);
}

} // namespace obt
