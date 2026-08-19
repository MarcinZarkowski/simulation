"""
Monte Carlo properties.

The engine has exactly one stochastic component: bid/ask execution cost. These
tests pin that claim down. If any of them fails, either something else in the
engine became random, or the spread draws are not independent, and in both cases
a reported confidence interval would be meaningless.
"""
from __future__ import annotations

import math
import statistics

import pytest

import obt_engine as E
from optionsbacktester.strategy import buy, group, sell
from tests.conftest import (
    EngineHarness,
    base_config,
    day_ns,
    make_bar,
    make_contract,
)

CALL = 1
PUT = 2
HOUR_NS = 3600 * 10**9


def features(mark: float = 5.0, iv: float = 0.25, dte: float = 30.0,
             volume: float = 500.0, moneyness: float = 0.0) -> E.SpreadFeatures:
    f = E.SpreadFeatures()
    f.mark_dollars = mark
    f.implied_volatility = iv
    f.days_to_expiry = dte
    f.volume = volume
    f.moneyness = moneyness
    return f


def lognormal_model(sigma: float = 0.45, tick: bool = False) -> E.SpreadModelConfig:
    cfg = E.SpreadModelConfig()
    cfg.kind = E.SpreadModelKind.LOGNORMAL
    cfg.log_sigma = sigma
    # Tick rounding quantizes the draw, which would mask small differences the
    # distribution tests are trying to observe.
    cfg.round_to_tick = tick
    cfg.min_half_spread_cents = 0.0
    return cfg


def run_round_trip(*, paths: int, seed: int, spread: E.SpreadModelKind,
                   sigma: float = 0.45, price: float = 5.0,
                   legs: int = 1) -> list[E.PathMetrics]:
    """
    Open and close a position on consecutive bars across `paths` scenarios.

    With zero spread and zero fees the round trip is exactly flat, so any P&L is
    attributable to execution cost alone.
    """
    contracts = [make_contract(CALL, strike=100.0, expiry_day=60)]
    if legs > 1:
        contracts.append(make_contract(PUT, strike=95.0, expiry_day=60, is_call=False))

    out: list[E.PathMetrics] = []
    for scenario in range(paths):
        cfg = base_config(paths=paths, seed=seed, spread=spread)
        if spread == E.SpreadModelKind.LOGNORMAL:
            cfg.spread_model = lognormal_model(sigma)
        engine = E.Engine(cfg)
        engine.set_contracts(contracts)
        engine.begin_scenario(scenario)

        def bars(ts):
            return [make_bar(c.id, timestamp_ns=ts, price=price) for c in contracts]

        opening = group(*[buy(c.id, 1) for c in contracts])
        closing = group(*[sell(c.id, 1, reduce_only=True) for c in contracts])

        engine.begin_bar(_snapshot(day_ns(1), bars(day_ns(1))))
        engine.submit_group(opening)
        engine.end_bar()

        engine.begin_bar(_snapshot(day_ns(2), bars(day_ns(2))))
        engine.submit_group(closing)
        engine.end_bar()

        engine.begin_bar(_snapshot(day_ns(3), bars(day_ns(3))))
        engine.end_bar()

        out.append(engine.finalize())
    return out


def _snapshot(ts: int, bars: list[E.MarketBar]) -> E.MarketSnapshot:
    s = E.MarketSnapshot()
    s.timestamp = ts
    s.bars = bars
    s.underlying_price = {"TEST": 100.0}
    return s


class TestZeroSpreadIsDeterministic:
    """Property 1: with no execution cost, every path is the same number."""

    def test_all_paths_share_one_pnl(self):
        paths = run_round_trip(paths=25, seed=7, spread=E.SpreadModelKind.ZERO)
        assert len({p.net_pnl_micros for p in paths}) == 1

    def test_round_trip_pnl_is_exactly_zero(self):
        """Buying and selling at the same mark with no friction must net to nothing."""
        paths = run_round_trip(paths=5, seed=7, spread=E.SpreadModelKind.ZERO)
        assert all(p.net_pnl_micros == 0 for p in paths)
        assert all(p.spread_cost_micros == 0 for p in paths)

    def test_zero_spread_reports_no_execution_cost(self):
        paths = run_round_trip(paths=5, seed=1, spread=E.SpreadModelKind.ZERO)
        assert all(p.spread_cost_micros == 0 for p in paths)


class TestSeedReproducibility:
    """Property 2: the same seed reproduces results bit for bit."""

    def test_same_seed_gives_identical_pnl(self):
        a = run_round_trip(paths=12, seed=99, spread=E.SpreadModelKind.LOGNORMAL)
        b = run_round_trip(paths=12, seed=99, spread=E.SpreadModelKind.LOGNORMAL)
        assert [p.net_pnl_micros for p in a] == [p.net_pnl_micros for p in b]

    def test_same_seed_gives_identical_spread_cost(self):
        a = run_round_trip(paths=12, seed=99, spread=E.SpreadModelKind.LOGNORMAL)
        b = run_round_trip(paths=12, seed=99, spread=E.SpreadModelKind.LOGNORMAL)
        assert [p.spread_cost_micros for p in a] == [p.spread_cost_micros for p in b]

    def test_draw_is_a_pure_function_of_its_key(self):
        cfg = lognormal_model()
        f = features()
        first = E.spread_draw(cfg, f, 42, 3, 100, 555, day_ns(5), 0)
        second = E.spread_draw(cfg, f, 42, 3, 100, 555, day_ns(5), 0)
        assert first == second

    def test_draw_is_independent_of_call_order(self):
        """
        Counter-based draws must not depend on how many draws preceded them, or
        results would change when orders are evaluated in a different sequence.
        """
        cfg = lognormal_model()
        f = features()
        keys = [(42, s, o, 7, day_ns(1), 0) for s in range(4) for o in range(4)]
        forward = [E.spread_draw(cfg, f, *k) for k in keys]
        backward = [E.spread_draw(cfg, f, *k) for k in reversed(keys)]
        assert forward == list(reversed(backward))


class TestSeedIndependence:
    """Property 3: a different seed changes execution cost and nothing else."""

    def test_different_seeds_change_pnl(self):
        a = run_round_trip(paths=20, seed=1, spread=E.SpreadModelKind.LOGNORMAL)
        b = run_round_trip(paths=20, seed=2, spread=E.SpreadModelKind.LOGNORMAL)
        assert [p.net_pnl_micros for p in a] != [p.net_pnl_micros for p in b]

    def test_different_seeds_leave_trade_counts_untouched(self):
        """
        Fill counts, expirations, and rejections come from market data and
        policy, so they must be identical no matter the seed.
        """
        a = run_round_trip(paths=20, seed=1, spread=E.SpreadModelKind.LOGNORMAL)
        b = run_round_trip(paths=20, seed=2, spread=E.SpreadModelKind.LOGNORMAL)
        for x, y in zip(a, b, strict=True):
            assert (x.fill_count, x.group_count, x.rejection_count) == (
                y.fill_count, y.group_count, y.rejection_count)
            assert (x.exercise_count, x.assignment_count, x.expiration_count) == (
                y.exercise_count, y.assignment_count, y.expiration_count)

    def test_pnl_difference_is_exactly_the_spread_difference(self):
        """
        Since spread cost is the only stochastic term, removing it must recover
        the same deterministic number on every path and every seed.
        """
        for seed in (1, 2, 3):
            paths = run_round_trip(paths=15, seed=seed, spread=E.SpreadModelKind.LOGNORMAL)
            gross = {p.net_pnl_micros + p.spread_cost_micros for p in paths}
            assert gross == {0}


class TestScenariosShareMarketData:
    """Property 4: only the draw varies; the tape does not."""

    def test_every_scenario_sees_the_same_fills_and_marks(self):
        contracts = [make_contract(CALL, strike=100.0, expiry_day=60)]
        marks_by_scenario = []
        for scenario in range(6):
            cfg = base_config(paths=6, seed=11, spread=E.SpreadModelKind.LOGNORMAL)
            cfg.spread_model = lognormal_model()
            engine = E.Engine(cfg)
            engine.set_contracts(contracts)
            engine.begin_scenario(scenario)
            for day in (1, 2):
                engine.begin_bar(_snapshot(day_ns(day), [
                    make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)]))
                if day == 1:
                    engine.submit_group(group(buy(CALL, 1)))
                engine.end_bar()
            engine.finalize()
            marks_by_scenario.append([f.mark for f in engine.fills()])
        assert len({tuple(m) for m in marks_by_scenario}) == 1

    def test_fill_price_differs_while_mark_agrees(self):
        """The mark is market data; the fill price is the mark plus a draw."""
        contracts = [make_contract(CALL, strike=100.0, expiry_day=60)]
        seen = []
        for scenario in range(6):
            cfg = base_config(paths=6, seed=11, spread=E.SpreadModelKind.LOGNORMAL)
            cfg.spread_model = lognormal_model()
            engine = E.Engine(cfg)
            engine.set_contracts(contracts)
            engine.begin_scenario(scenario)
            for day in (1, 2):
                engine.begin_bar(_snapshot(day_ns(day), [
                    make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)]))
                if day == 1:
                    engine.submit_group(group(buy(CALL, 1)))
                engine.end_bar()
            engine.finalize()
            fill = engine.fills()[0]
            seen.append((fill.mark, fill.fill_price))
        assert len({m for m, _ in seen}) == 1
        assert len({p for _, p in seen}) > 1


class TestSpreadDirection:
    """A drawn spread is a cost, never a windfall."""

    def test_half_spread_is_never_negative(self):
        cfg = lognormal_model(sigma=1.2)
        for scenario in range(400):
            assert E.spread_draw(cfg, features(), 5, scenario, 1, 1, day_ns(1), 0) >= 0.0

    def test_buys_fill_at_or_above_the_mark_and_sells_at_or_below(self):
        contracts = [make_contract(CALL, strike=100.0, expiry_day=60)]
        for scenario in range(12):
            cfg = base_config(paths=12, seed=3, spread=E.SpreadModelKind.LOGNORMAL)
            cfg.spread_model = lognormal_model()
            engine = E.Engine(cfg)
            engine.set_contracts(contracts)
            engine.begin_scenario(scenario)
            for day in (1, 2, 3):
                engine.begin_bar(_snapshot(day_ns(day), [
                    make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)]))
                if day == 1:
                    engine.submit_group(group(buy(CALL, 1)))
                if day == 2:
                    engine.submit_group(group(sell(CALL, 1, reduce_only=True)))
                engine.end_bar()
            engine.finalize()
            for fill in engine.fills():
                if fill.side == E.OrderSide.BUY:
                    assert fill.fill_price >= fill.mark
                else:
                    assert fill.fill_price <= fill.mark

    def test_spread_cost_is_never_negative(self):
        paths = run_round_trip(paths=30, seed=17, spread=E.SpreadModelKind.LOGNORMAL)
        assert all(p.spread_cost_micros >= 0 for p in paths)

    def test_execution_cost_only_reduces_pnl(self):
        """A round trip that is flat before cost cannot be profitable after it."""
        paths = run_round_trip(paths=30, seed=17, spread=E.SpreadModelKind.LOGNORMAL)
        assert all(p.net_pnl_micros <= 0 for p in paths)


class TestConvergence:
    """Properties 5 and 8: the estimator behaves like a Monte Carlo estimator."""

    # Analytic mean of the lognormal spread model. The half-spread in basis
    # points is lognormal(log_base, log_sigma), so its mean is
    # exp(mu + sigma^2/2); a round trip pays it on two fills of 100 shares.
    LOG_BASE = 4.0
    LOG_SIGMA = 0.45
    MARK = 5.0

    @classmethod
    def _analytic_mean_cost(cls) -> float:
        mean_bps = math.exp(cls.LOG_BASE + 0.5 * cls.LOG_SIGMA**2)
        half_spread = cls.MARK * mean_bps / 20000.0
        return 2 * half_spread * 100

    @staticmethod
    def _mean_cost(paths: int, seed: int = 21) -> float:
        results = run_round_trip(paths=paths, seed=seed, spread=E.SpreadModelKind.LOGNORMAL,
                                 price=TestConvergence.MARK)
        return statistics.fmean(p.spread_cost for p in results)

    def test_mean_cost_converges_to_the_analytic_expectation(self):
        """
        Averaged over independent seeds, the deviation from the closed-form mean
        must shrink as paths increase. Comparing two sample means from one seed
        would measure sampling noise instead.
        """
        target = self._analytic_mean_cost()
        seeds = (11, 23, 37, 51, 67)

        def mean_abs_error(n: int) -> float:
            return statistics.fmean(abs(self._mean_cost(n, seed) - target) for seed in seeds)

        assert mean_abs_error(1000) < mean_abs_error(50)

    def test_large_sample_mean_lands_within_a_few_standard_errors(self):
        target = self._analytic_mean_cost()
        results = run_round_trip(paths=4000, seed=77, spread=E.SpreadModelKind.LOGNORMAL,
                                 price=self.MARK)
        costs = [p.spread_cost for p in results]
        stderr = statistics.stdev(costs) / math.sqrt(len(costs))
        assert abs(statistics.fmean(costs) - target) < 4 * stderr

    def test_standard_error_falls_like_one_over_root_n(self):
        """
        Quadrupling the sample should roughly halve the standard error. A ratio
        far from 2 would mean the draws are correlated rather than independent.
        """
        def stderr(n: int) -> float:
            results = run_round_trip(paths=n, seed=31, spread=E.SpreadModelKind.LOGNORMAL)
            costs = [p.spread_cost for p in results]
            return statistics.stdev(costs) / math.sqrt(len(costs))

        ratio = stderr(250) / stderr(1000)
        assert 1.4 < ratio < 2.8, f"standard error scaled by {ratio:.2f}, expected about 2"

    def test_draws_are_not_all_identical(self):
        costs = {p.spread_cost_micros for p in
                 run_round_trip(paths=200, seed=5, spread=E.SpreadModelKind.LOGNORMAL)}
        assert len(costs) > 50


class TestCommonRandomNumbers:
    """Property 6: comparisons between strategies are not polluted by draw noise."""

    def test_same_order_on_the_same_instant_draws_the_same_spread(self):
        cfg = lognormal_model()
        a = E.spread_draw(cfg, features(mark=5.0), 42, 3, 77, 900, day_ns(4), 0)
        b = E.spread_draw(cfg, features(mark=5.0), 42, 3, 77, 900, day_ns(4), 0)
        assert a == b

    def test_draw_varies_across_instruments_and_instants(self):
        cfg = lognormal_model()
        base = E.spread_draw(cfg, features(), 42, 3, 77, 900, day_ns(4), 0)
        assert E.spread_draw(cfg, features(), 42, 3, 77, 901, day_ns(4), 0) != base
        assert E.spread_draw(cfg, features(), 42, 3, 77, 900, day_ns(5), 0) != base
        assert E.spread_draw(cfg, features(), 42, 4, 77, 900, day_ns(4), 0) != base
        assert E.spread_draw(cfg, features(), 43, 3, 77, 900, day_ns(4), 0) != base


class TestPerLegDraws:
    """Property 7: each leg draws its own spread, but the group still fills atomically."""

    def test_legs_of_one_group_draw_independently(self):
        cfg = lognormal_model()
        f = features()
        first = E.spread_draw(cfg, f, 42, 0, 100, 500, day_ns(1), 0)
        second = E.spread_draw(cfg, f, 42, 0, 100, 500, day_ns(1), 1)
        assert first != second

    def test_two_leg_group_fills_both_legs_together(self):
        contracts = [
            make_contract(CALL, strike=100.0, expiry_day=60),
            make_contract(PUT, strike=95.0, expiry_day=60, is_call=False),
        ]
        cfg = base_config(spread=E.SpreadModelKind.LOGNORMAL)
        cfg.spread_model = lognormal_model()
        h = EngineHarness(cfg, contracts)
        bars = lambda ts: [make_bar(c.id, timestamp_ns=ts, price=5.0) for c in contracts]

        h.bar(day_ns(1), bars(day_ns(1)), groups=[group(buy(CALL, 1), sell(PUT, 1))])
        h.bar(day_ns(2), bars(day_ns(2)))
        fills = h.fills()
        assert len(fills) == 2
        assert len({f.group_id for f in fills}) == 1
        assert len({f.half_spread for f in fills}) == 2


class TestSpreadModels:
    """Each model responds to its own parameters and respects its guards."""

    def test_constant_model_ignores_the_draw(self):
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.CONSTANT_CENTS
        cfg.constant_cents = 10.0
        cfg.round_to_tick = False
        draws = {E.spread_draw(cfg, features(), 1, s, 1, 1, day_ns(1), 0) for s in range(50)}
        assert draws == {0.05}

    def test_proportional_model_scales_with_the_mark(self):
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.PROPORTIONAL_BPS
        cfg.proportional_bps = 100.0
        cfg.round_to_tick = False
        cfg.min_half_spread_cents = 0.0
        cheap = E.spread_draw(cfg, features(mark=1.0), 1, 0, 1, 1, day_ns(1), 0)
        rich = E.spread_draw(cfg, features(mark=10.0), 1, 0, 1, 1, day_ns(1), 0)
        assert rich == pytest.approx(10 * cheap)

    def test_conditional_model_widens_with_volatility(self):
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.CONDITIONAL_LOGNORMAL
        cfg.log_sigma = 0.0
        cfg.round_to_tick = False
        low = E.spread_draw(cfg, features(iv=0.15), 1, 0, 1, 1, day_ns(1), 0)
        high = E.spread_draw(cfg, features(iv=0.90), 1, 0, 1, 1, day_ns(1), 0)
        assert high > low

    def test_conditional_model_tightens_with_volume(self):
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.CONDITIONAL_LOGNORMAL
        cfg.log_sigma = 0.0
        cfg.round_to_tick = False
        thin = E.spread_draw(cfg, features(volume=1), 1, 0, 1, 1, day_ns(1), 0)
        liquid = E.spread_draw(cfg, features(volume=100_000), 1, 0, 1, 1, day_ns(1), 0)
        assert liquid < thin

    def test_conditional_model_widens_away_from_the_money(self):
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.CONDITIONAL_LOGNORMAL
        cfg.log_sigma = 0.0
        cfg.round_to_tick = False
        atm = E.spread_draw(cfg, features(moneyness=0.0), 1, 0, 1, 1, day_ns(1), 0)
        wing = E.spread_draw(cfg, features(moneyness=0.40), 1, 0, 1, 1, day_ns(1), 0)
        assert wing > atm

    def test_empirical_model_only_returns_sampled_values(self):
        cfg = E.SpreadModelConfig()
        cfg.kind = E.SpreadModelKind.EMPIRICAL
        cfg.empirical_half_spread_cents = [1.0, 3.0, 7.0]
        cfg.round_to_tick = False
        cfg.min_half_spread_cents = 0.0
        draws = {round(E.spread_draw(cfg, features(), 1, s, 1, 1, day_ns(1), 0), 6)
                 for s in range(200)}
        assert draws <= {0.01, 0.03, 0.07}
        assert len(draws) == 3

    def test_spread_never_exceeds_its_cap_fraction_of_the_mark(self):
        cfg = lognormal_model(sigma=3.0)
        cfg.max_fraction_of_mark = 0.25
        for scenario in range(300):
            draw = E.spread_draw(cfg, features(mark=2.0), 9, scenario, 1, 1, day_ns(1), 0)
            assert draw <= 2.0 * 0.25 + 1e-12

    def test_tick_rounding_quantizes_the_spread(self):
        cfg = lognormal_model(sigma=0.6, tick=True)
        for scenario in range(100):
            draw = E.spread_draw(cfg, features(mark=1.5), 4, scenario, 1, 1, day_ns(1), 0)
            # Below $3 the grid is a penny, so a half-spread lands on a half-cent.
            assert abs(draw * 200 - round(draw * 200)) < 1e-9


class TestLedgerUnderRandomness:
    def test_ledger_reconciles_on_every_path(self):
        paths = run_round_trip(paths=40, seed=8, spread=E.SpreadModelKind.LOGNORMAL)
        assert all(p.ledger_reconciles for p in paths)

    def test_reported_spread_cost_matches_the_fills(self):
        contracts = [make_contract(CALL, strike=100.0, expiry_day=60)]
        cfg = base_config(spread=E.SpreadModelKind.LOGNORMAL)
        cfg.spread_model = lognormal_model()
        h = EngineHarness(cfg, contracts)
        for day in (1, 2, 3):
            groups = []
            if day == 1:
                groups = [group(buy(CALL, 2))]
            if day == 2:
                groups = [group(sell(CALL, 2, reduce_only=True))]
            h.bar(day_ns(day), [make_bar(CALL, timestamp_ns=day_ns(day), price=5.0)],
                  groups=groups)
        metrics = h.finalize()
        expected = sum(round(f.half_spread * f.quantity * f.multiplier * 1_000_000)
                       for f in h.fills())
        assert metrics.spread_cost_micros == expected
