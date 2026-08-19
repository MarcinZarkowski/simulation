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
    double log_base = 4.0;          // exp(4.0) ~ 55 bps at the reference point
    double log_sigma = 0.45;
    double beta_iv = 0.90;
    double beta_log_dte = -0.15;
    double beta_log_volume = -0.12;
    double beta_abs_moneyness = 1.80;
    double beta_minutes_from_open = -0.0004;

    // Empirical: sampled half-spreads in cents, drawn uniformly.
    std::vector<double> empirical_half_spread_cents;

    // Guards. A real quote is never tighter than a tick and a spread wider than
    // the option is worth would let a fill go through zero.
    double min_half_spread_cents = 0.5;
    double max_fraction_of_mark = 0.50;
    // Quote grid: $0.01 below $3.00, $0.05 at or above, per exchange convention.
    bool round_to_tick = true;
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
            const double bps = std::exp(cfg.log_base + cfg.log_sigma * standard_normal(key));
            half = f.mark_dollars * bps / 20000.0;
            break;
        }

        case SpreadModelKind::ConditionalLognormal: {
            const double mu = cfg.log_base
                + cfg.beta_iv * f.implied_volatility
                + cfg.beta_log_dte * std::log1p(std::max(0.0, f.days_to_expiry))
                + cfg.beta_log_volume * std::log1p(std::max(0.0, f.volume))
                + cfg.beta_abs_moneyness * std::fabs(f.moneyness)
                + cfg.beta_minutes_from_open * f.minutes_from_open;
            const double bps = std::exp(mu + cfg.log_sigma * standard_normal(key));
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

    half = std::max(half, cfg.min_half_spread_cents / 100.0);
    half = std::min(half, f.mark_dollars * cfg.max_fraction_of_mark);
    half = std::max(half, 0.0);

    if (cfg.round_to_tick && half > 0.0) {
        const double tick = tick_size_dollars(f.mark_dollars);
        // Round the full spread to the grid, so half lands on a half-tick.
        half = std::round(2.0 * half / tick) * tick / 2.0;
    }
    return half;
}

} // namespace obt
