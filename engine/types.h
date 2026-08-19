#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <string>

namespace obt {

// ---------------------------------------------------------------------------
// Money
// ---------------------------------------------------------------------------
// Ledger values are exact integers in microdollars. The previous engine
// accumulated cash in float32, where one ULP is $1.00 at a $10M balance, so
// balances silently lost cents. Every cash, premium, fee, cost-basis, and
// settlement amount below is integral; only IV and Greeks stay floating point
// because they are analytics, never ledger entries.
struct Money {
    int64_t micros = 0;

    static constexpr int64_t kPerDollar = 1'000'000;

    static Money zero() { return Money{0}; }

    // Nearest-microdollar conversion from a vendor price. Ties go away from
    // zero so a half-microdollar never depends on the sign.
    static Money from_double(double dollars) {
        if (!std::isfinite(dollars)) return Money{0};
        return Money{static_cast<int64_t>(std::llround(dollars * static_cast<double>(kPerDollar)))};
    }

    static Money from_cents(int64_t cents) { return Money{cents * 10'000}; }

    double to_double() const { return static_cast<double>(micros) / static_cast<double>(kPerDollar); }

    Money operator+(Money o) const { return Money{micros + o.micros}; }
    Money operator-(Money o) const { return Money{micros - o.micros}; }
    Money operator-() const { return Money{-micros}; }
    Money& operator+=(Money o) { micros += o.micros; return *this; }
    Money& operator-=(Money o) { micros -= o.micros; return *this; }

    // Scaling by a contract count or share amount stays exact.
    Money operator*(int64_t n) const { return Money{micros * n}; }

    bool operator<(Money o) const { return micros < o.micros; }
    bool operator<=(Money o) const { return micros <= o.micros; }
    bool operator>(Money o) const { return micros > o.micros; }
    bool operator>=(Money o) const { return micros >= o.micros; }
    bool operator==(Money o) const { return micros == o.micros; }
    bool operator!=(Money o) const { return micros != o.micros; }

    bool is_zero() const { return micros == 0; }
    Money abs() const { return Money{micros < 0 ? -micros : micros}; }
};

inline Money operator*(int64_t n, Money m) { return m * n; }

inline Money min_money(Money a, Money b) { return a < b ? a : b; }
inline Money max_money(Money a, Money b) { return a > b ? a : b; }

// Multiply a money amount by a ratio, rounding to the nearest microdollar.
// Used only for fee rates and margin percentages, never for premium math.
inline Money scale(Money m, double ratio) {
    return Money{static_cast<int64_t>(std::llround(static_cast<double>(m.micros) * ratio))};
}

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------
// Epoch nanoseconds, tz-naive UTC to match the pipeline's timestamp columns.
struct Timestamp {
    int64_t epoch_ns = 0;

    static Timestamp never() { return Timestamp{std::numeric_limits<int64_t>::max()}; }
    static Timestamp min() { return Timestamp{std::numeric_limits<int64_t>::min()}; }

    bool is_never() const { return epoch_ns == std::numeric_limits<int64_t>::max(); }

    bool operator<(Timestamp o) const { return epoch_ns < o.epoch_ns; }
    bool operator<=(Timestamp o) const { return epoch_ns <= o.epoch_ns; }
    bool operator>(Timestamp o) const { return epoch_ns > o.epoch_ns; }
    bool operator>=(Timestamp o) const { return epoch_ns >= o.epoch_ns; }
    bool operator==(Timestamp o) const { return epoch_ns == o.epoch_ns; }
    bool operator!=(Timestamp o) const { return epoch_ns != o.epoch_ns; }

    int64_t days_until(Timestamp o) const {
        return (o.epoch_ns - epoch_ns) / (86'400LL * 1'000'000'000LL);
    }
};

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------
// A position references a contract *version*, never a symbol. The same OCC
// symbol can describe different economics either side of an adjustment, and one
// economic series can change symbol, so the symbol is an attribute of a version
// rather than an identity.
struct InstrumentId {
    uint64_t value = 0;
    bool operator==(InstrumentId o) const { return value == o.value; }
    bool operator<(InstrumentId o) const { return value < o.value; }
};

struct ContractVersionId {
    uint64_t value = 0;
    bool operator==(ContractVersionId o) const { return value == o.value; }
    bool operator!=(ContractVersionId o) const { return value != o.value; }
    bool operator<(ContractVersionId o) const { return value < o.value; }
};

struct PositionId {
    uint64_t value = 0;
    bool operator==(PositionId o) const { return value == o.value; }
    bool operator<(PositionId o) const { return value < o.value; }
};

// Stable 64-bit id from a string key, so the same symbol maps to the same id
// across runs and across processes.
inline uint64_t hash_key(const std::string& key) {
    // FNV-1a, then a splitmix64 finalizer to spread low-entropy inputs.
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : key) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27; h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

enum class OptionType : uint8_t { Call, Put };
enum class EquityKind : uint8_t { Option, Equity };

} // namespace obt
