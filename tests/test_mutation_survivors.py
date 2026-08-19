"""
Gaps found by mutation testing the engine, and closed here.

`scripts/mutation_score.py` broke the engine on purpose and reported 18 mutants the
suite did not notice. Most were equivalent -- `>` to `>=` where the value can never be
zero, so there is no behaviour to detect -- but these were real: a specific change to
the engine that nothing objected to, on a path a real run reaches.

Each test names the mutation it kills, so a later reader can see why an otherwise
unremarkable assertion is here.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester.strategy import buy, buy_shares, group, sell

from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

CALL = 1
LONG = 2
SPOT = 100.0
SESSION = 3_600_000_000_000


def equity_bar(day: int, price: float = SPOT) -> E.EquityBar:
    b = E.EquityBar()
    b.timestamp = day_ns(day)
    b.symbol = "TEST"
    b.open = b.high = b.low = b.close = price
    b.volume = 1_000_000
    return b


def snapshot(day: int, prices: dict[int, float], *, spot: float = SPOT,
             with_equity: bool = False, analytics: list | None = None):
    s = E.MarketSnapshot()
    s.timestamp = day_ns(day)
    s.bars = [make_bar(cv, timestamp_ns=day_ns(day), price=p) for cv, p in prices.items()]
    s.underlying_price = {"TEST": spot}
    if with_equity:
        s.equity_bars = [equity_bar(day, spot)]
    if analytics is not None:
        s.analytics = analytics
    return s


class TestACashAccountCanBuyStockItHasPaidFor:
    """
    A real bug, found by chasing a mutation survivor on `equity_loan_fraction`.

    Order admission credited share positions at a LOAN fraction -- 50% under Reg-T, zero
    under a cash account -- while the margin model separately charged them 50% and 100%.
    For a cash account that is a double charge: it lends nothing and is charged in full,
    so it needed TWICE the purchase price. Measured: $12,000 of cash could not buy
    $10,000 of shares, but $100,000 of cash could.

    Shares are now credited at full market value and the margin model does the
    charging, so the two offset to exactly the model's own fraction -- net zero under
    Reg-T's 50% and net zero under a cash account's 100%. The loan-fraction function is
    deleted; nothing calls it.

    Long OPTION value is still excluded from the credit, because the margin model
    charges nothing for a long option and crediting it would let the option
    collateralize its own purchase.
    """

    def _buys(self, model, *, cash: float, shares: int) -> bool:
        contract = make_contract(CALL, strike=100.0, expiry_day=400)
        h = EngineHarness(base_config(cash=cash, margin=model), [contract])
        for day in (1, 2):
            h.engine.begin_bar(snapshot(day, {CALL: 5.0}, with_equity=True))
            if day == 1:
                h.engine.submit_group(group(buy_shares("TEST", shares)))
            h.engine.end_bar()
        return h.shares_of("TEST") == shares

    def test_a_cash_account_can_buy_shares_it_can_pay_for(self):
        """$12,000 of cash, $10,000 of stock. Refused before this fix."""
        assert self._buys(E.MarginModel.CASH_ACCOUNT, cash=12_000.0, shares=100) is True

    def test_a_cash_account_cannot_buy_shares_it_cannot_pay_for(self):
        """The control: paying in full still means paying in full."""
        assert self._buys(E.MarginModel.CASH_ACCOUNT, cash=9_000.0, shares=100) is False

    def test_a_margin_account_can_buy_twice_as_much(self):
        """
        Where the models genuinely differ: 50% against 100%. $12,000 of cash carries
        $20,000 of stock on margin and $10,000 in a cash account.
        """
        assert self._buys(E.MarginModel.REG_T, cash=12_000.0, shares=200) is True
        assert self._buys(E.MarginModel.CASH_ACCOUNT, cash=12_000.0, shares=200) is False

    def test_a_long_option_still_cannot_collateralize_its_own_purchase(self):
        """
        The reason the credit is not simply "all position value". A long option carries
        no requirement, so crediting its market value would make any option affordable.
        """
        pricey = make_contract(LONG, strike=50.0, expiry_day=400)
        h = EngineHarness(base_config(cash=1_000.0, margin=E.MarginModel.REG_T), [pricey])
        for day in (1, 2):
            h.engine.begin_bar(snapshot(day, {LONG: 50.0}))
            if day == 1:
                h.engine.submit_group(group(buy(LONG, 1)))      # $5,000
            h.engine.end_bar()

        assert h.quantity_of(LONG) == 0
        assert h.rejections()[0].reason == E.RejectReason.INSUFFICIENT_BUYING_POWER


class TestAMarginBreachIsRecorded:
    """
    Kills: engine.h `metrics_.margin_breached = true` -> `false`.

    Nothing asserted the flag. A run that blew through its requirement reported a
    clean path.
    """

    def _breach(self) -> E.PathMetrics:
        # A short put needs the full strike under Robinhood: $10,000 against $600.
        short_put = make_contract(CALL, strike=100.0, expiry_day=400, is_call=False)
        h = EngineHarness(base_config(cash=600.0, margin=E.MarginModel.ROBINHOOD),
                          [short_put])
        for day in (1, 2, 3):
            h.engine.begin_bar(snapshot(day, {CALL: 6.0}))
            if day == 1:
                h.engine.submit_group(group(sell(CALL, 1)))
            h.engine.end_bar()
        return h.finalize()

    def test_an_unaffordable_position_is_refused_rather_than_breaching(self):
        """
        The engine checks margin BEFORE filling, so the ordinary path is a rejection
        and the flag stays clear. That is what makes the next test meaningful.
        """
        metrics = self._breach()

        assert metrics.rejection_count > 0
        assert not metrics.margin_breached

    def test_a_book_that_becomes_unaffordable_after_filling_breaches(self):
        """
        Fill while affordable, then move the market against it.

        A naked short call is the case where this can happen at all: its Reg-T
        requirement grows with the underlying, so a position that fitted at a spot of
        100 does not at 500. A vertical spread cannot breach this way -- its
        requirement is its max loss, which does not move.
        """
        far_otm = make_contract(CALL, strike=150.0, expiry_day=400)
        h = EngineHarness(base_config(cash=5_000.0, margin=E.MarginModel.REG_T),
                          [far_otm])
        for day in (1, 2):
            h.engine.begin_bar(snapshot(day, {CALL: 1.0}))
            if day == 1:
                h.engine.submit_group(group(sell(CALL, 1)))
            h.engine.end_bar()
        assert h.quantity_of(CALL) == -1
        assert not h.finalize().margin_breached

        h.engine.begin_bar(snapshot(3, {CALL: 355.0}, spot=500.0))
        h.engine.end_bar()

        assert h.finalize().margin_breached


class TestTheEquitySpreadIsStochasticOnlyAboveTheTick:
    """
    Kills: spread.h `kind == Lognormal || kind == ConditionalLognormal` -> inverted.

    Writing this test found something the earlier commit overstated. At the default
    1 bp full spread, a $100 stock's modelled half-spread is exactly $0.005 -- which is
    the half-cent floor -- so every draw is clamped and the equity spread contributes
    NOTHING to the Monte Carlo. The floor binds at every price up to about $100.

    That is economically right rather than a bug: a penny is the minimum tick, and a
    liquid stock at those prices really does quote one cent wide with no dispersion to
    model. But it means the stochastic term only bites on a higher-priced or
    wider-spread name, and a reader comparing a run's interval against its equity
    activity should know that. So the dispersion is tested at $500, where the modelled
    spread exceeds the tick, and the pinning is tested at $100 as the deliberate
    behaviour it is.
    """

    def _half_spreads(self, kind, paths: int = 6, spot: float = SPOT) -> list[float]:
        out = []
        for scenario in range(paths):
            cfg = base_config(cash=100_000.0, spread=kind, margin=E.MarginModel.REG_T)
            contract = make_contract(CALL, strike=100.0, expiry_day=400)
            engine = E.Engine(cfg)
            registry = E.ContractRegistry()
            registry.add(contract)
            engine.share_registry(registry)
            engine.begin_scenario(scenario)
            for day in (1, 2):
                engine.begin_bar(snapshot(day, {CALL: 5.0}, spot=spot, with_equity=True))
                if day == 1:
                    engine.submit_group(group(buy_shares("TEST", 100)))
                engine.end_bar()
            fills = [f for f in engine.fills() if f.kind == E.EquityKind.EQUITY]
            assert fills, "no equity fill"
            out.append(fills[0].half_spread)
        return out

    def test_the_lognormal_models_vary_the_spread_where_it_exceeds_the_tick(self):
        for kind in (E.SpreadModelKind.LOGNORMAL, E.SpreadModelKind.CONDITIONAL_LOGNORMAL):
            spreads = self._half_spreads(kind, spot=500.0)

            assert len(set(spreads)) > 1, f"{kind} produced one value across paths"

    def test_the_constant_model_does_not(self):
        """The control. A deterministic model must be deterministic."""
        spreads = self._half_spreads(E.SpreadModelKind.PROPORTIONAL_BPS, spot=500.0)

        assert len(set(spreads)) == 1

    def test_at_a_hundred_dollars_the_tick_floor_pins_every_draw(self):
        """
        Deliberate, and worth stating: the modelled half-spread at 1 bp on a $100
        stock is exactly the half-cent floor, so the equity spread contributes no
        Monte Carlo dispersion there. A liquid stock at that price quotes one cent
        wide and there is nothing to disperse.
        """
        spreads = self._half_spreads(E.SpreadModelKind.CONDITIONAL_LOGNORMAL, spot=100.0)

        assert set(spreads) == {0.005}

    def test_zero_means_zero(self):
        assert self._half_spreads(E.SpreadModelKind.ZERO) == [0.0] * 6


class TestADeltaLimitToleratesMissingAnalytics:
    """
    Kills: engine.h `if (a != nullptr && a->valid)` -> `||`.

    With `||` the expression dereferences a null pointer whenever analytics are
    absent, which is undefined behaviour. Nothing configured a delta limit alongside a
    position whose analytics were missing, so the guard was never exercised.
    """

    def _run(self, *, with_analytics: bool, limit: float) -> EngineHarness:
        contract = make_contract(CALL, strike=100.0, expiry_day=400)
        cfg = base_config(cash=100_000.0, margin=E.MarginModel.REG_T)
        cfg.risk.max_abs_delta = limit
        h = EngineHarness(cfg, [contract])
        for day in (1, 2, 3):
            analytics = None
            if with_analytics:
                a = E.OptionAnalytics()
                a.timestamp = day_ns(day)
                a.contract_version_id = CALL
                a.delta = 0.9
                a.valid = True
                analytics = [a]
            h.engine.begin_bar(snapshot(day, {CALL: 5.0}, analytics=analytics))
            if day == 1:
                h.engine.submit_group(group(buy(CALL, 5)))
            h.engine.end_bar()
        return h

    def test_a_position_with_no_analytics_does_not_crash_the_delta_check(self):
        h = self._run(with_analytics=False, limit=1.0)

        assert h.engine.ledger_reconciles()

    def test_a_position_with_no_analytics_contributes_no_delta(self):
        """
        So it cannot breach the limit. Counting an absent delta as anything would be
        inventing exposure.
        """
        h = self._run(with_analytics=False, limit=0.5)

        assert h.quantity_of(CALL) == 5

    def test_a_position_with_valid_analytics_does_breach_it(self):
        """The control: the limit works when there is a delta to measure."""
        h = self._run(with_analytics=True, limit=0.5)

        assert h.quantity_of(CALL) == 0
        assert any("delta" in r.detail for r in h.rejections())


class TestADisabledNotionalLimitIsInert:
    """
    Kills: engine.h `value > max_notional_per_underlying` -> `>=`.

    With `>=` and the limit left at its disabled default of zero, every position has
    notional at least zero and so every order is refused. The outer guard makes this
    unreachable today, but nothing asserted that the guard is what protects it.
    """

    def _fills(self, limit: float) -> int:
        contract = make_contract(CALL, strike=100.0, expiry_day=400)
        cfg = base_config(cash=100_000.0, margin=E.MarginModel.REG_T)
        cfg.risk.max_notional_per_underlying = limit
        h = EngineHarness(cfg, [contract])
        for day in (1, 2):
            h.engine.begin_bar(snapshot(day, {CALL: 5.0}))
            if day == 1:
                h.engine.submit_group(group(buy(CALL, 1)))
            h.engine.end_bar()
        return len(h.fills())

    def test_zero_means_no_limit_rather_than_a_limit_of_zero(self):
        assert self._fills(0.0) == 1

    def test_a_limit_above_the_position_permits_it(self):
        assert self._fills(50_000.0) == 1

    def test_a_limit_below_the_position_refuses_it(self):
        """$10,000 of notional against a $5,000 limit."""
        assert self._fills(5_000.0) == 0


class TestPairingPrefersTheEarliestExpiringShort:
    """
    Kills: margin.h `a.expiration < b.expiration` -> `<=` in the sort comparator.

    A non-strict comparator is undefined behaviour in std::sort. The ordering is also
    load-bearing on its own terms: a scarce long-dated long should be offered to the
    short that most needs it.
    """

    def test_a_single_long_covers_the_nearest_short(self):
        near = make_contract(1, strike=105.0, expiry_day=30)
        far = make_contract(2, strike=105.0, expiry_day=90)
        leap = make_contract(3, strike=100.0, expiry_day=400)
        result = E.evaluate_margin(
            E.MarginModel.REG_T, [near, far, leap],
            [(1, -1), (2, -1), (3, 1)], {"TEST": SPOT},
            {1: 2.0, 2: 4.0, 3: 8.0}, {},
        )

        paired = [p for p in result.pairings if not p.naked]
        assert len(paired) == 1
        assert paired[0].short_leg == 1        # the 30-day short, not the 90-day one


class TestADeliverableCashComponentIsAdded:
    """
    Kills: contract.h `delivered_value` `+ deliverable_cash` -> `-`.

    A stock-and-cash merger leaves a contract delivering shares PLUS a fixed cash
    amount, and the sign of that cash decides whether the holder receives it or pays
    it. No test had a non-zero cash component reach the payoff, so flipping the sign
    was invisible -- on precisely the contracts whose terms are already unusual.
    """

    def _contract(self, cash: float):
        c = make_contract(CALL, strike=100.0, expiry_day=6, multiplier=100,
                          deliverable_shares=50, is_adjusted=True)
        c.deliverable_cash = cash
        c.terms_provenance = E.TermsProvenance.POINT_IN_TIME
        return c

    def _settle(self, cash: float, spot: float = 200.0) -> float:
        contract = self._contract(cash)
        h = EngineHarness(base_config(cash=100_000.0, margin=E.MarginModel.REG_T),
                          [contract])
        for day in (1, 2):
            h.engine.begin_bar(snapshot(day, {CALL: 20.0}, spot=spot))
            if day == 1:
                h.engine.submit_group(group(buy(CALL, 1)))
            h.engine.end_bar()
        assert h.quantity_of(CALL) == 1
        h.engine.begin_bar(snapshot(6, {CALL: 20.0}, spot=spot))
        h.engine.end_bar()
        h.engine.end_session(day_ns(6) + SESSION)
        return h.engine.account_state().equity

    def test_the_cash_component_is_worth_its_face_value_at_settlement(self):
        """
        50 shares at 200 is $10,000 delivered, plus $500 of cash, against an aggregate
        exercise price of 100 x 100 = $10,000. So the cash component is the ENTIRE
        payoff: with it the contract exercises and is worth $500, without it the payoff
        is zero and it expires worthless.

        Compared on EQUITY rather than cash, because the two cases end in different
        shapes -- the exercised one spends cash to acquire shares -- and comparing cash
        alone made the profitable case look worse.
        """
        with_cash = self._settle(500.0)
        without = self._settle(0.0)

        assert with_cash - without == pytest.approx(500.0)

    def test_the_payoff_itself_includes_it(self):
        """Asserted directly on the contract, not only through settlement."""
        assert self._contract(500.0).payoff_at(200.0) == pytest.approx(500.0)
        assert self._contract(0.0).payoff_at(200.0) == pytest.approx(0.0)

    def test_a_cash_component_can_make_an_otherwise_worthless_contract_pay(self):
        plain = self._contract(0.0)
        with_cash = self._contract(1_000.0)

        assert plain.payoff_at(190.0) == pytest.approx(0.0)
        assert with_cash.payoff_at(190.0) == pytest.approx(500.0)


class TestTheNotionalLimitBoundary:
    """
    Kills: engine.h `value > max_notional_per_underlying` -> `>=`.

    The earlier test only covered a limit well above and well below the position, so
    the boundary itself -- notional exactly equal to the limit -- was never exercised
    and the comparison could be either strict or not.
    """

    def _fills(self, limit: float) -> int:
        contract = make_contract(CALL, strike=100.0, expiry_day=400)
        cfg = base_config(cash=100_000.0, margin=E.MarginModel.REG_T)
        cfg.risk.max_notional_per_underlying = limit
        h = EngineHarness(cfg, [contract])
        for day in (1, 2):
            h.engine.begin_bar(snapshot(day, {CALL: 5.0}))
            if day == 1:
                h.engine.submit_group(group(buy(CALL, 1)))
            h.engine.end_bar()
        return len(h.fills())

    def test_notional_exactly_at_the_limit_is_permitted(self):
        """
        One contract on a $100 underlying controls $10,000. A limit OF $10,000 is a
        limit the position meets, not one it breaches.
        """
        assert self._fills(10_000.0) == 1

    def test_one_microdollar_over_the_limit_is_refused(self):
        assert self._fills(9_999.999999) == 0
