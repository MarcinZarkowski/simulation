"""
Order execution semantics: no-lookahead timing, group atomicity, limits, data
quality gates, buying power, and exact cash/position bookkeeping.
"""
from __future__ import annotations

import pytest

import obt_engine as E
from optionsbacktester.strategy import buy, group, sell
from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

CALL_100, CALL_105, CALL_110 = 1, 2, 3
PUT_100, PUT_95, PUT_90 = 4, 5, 6
UNKNOWN_CV = 999

INITIAL_CASH_MICROS = 100_000 * 10**6
PER_CONTRACT = 100


def universe() -> list[E.OptionContractVersion]:
    return [
        make_contract(CALL_100, strike=100.0),
        make_contract(CALL_105, strike=105.0),
        make_contract(CALL_110, strike=110.0),
        make_contract(PUT_100, strike=100.0, is_call=False),
        make_contract(PUT_95, strike=95.0, is_call=False),
        make_contract(PUT_90, strike=90.0, is_call=False),
    ]


def bars_at(day: int, prices: dict[int, float], **kw) -> list[E.MarketBar]:
    return [make_bar(cv, timestamp_ns=day_ns(day), price=p, **kw) for cv, p in prices.items()]


def reasons(harness: EngineHarness) -> list[E.RejectReason]:
    return [r.reason for r in harness.rejections()]


@pytest.fixture
def engine():
    """
    Factory for harnesses that reconciles every ledger it created at teardown.

    Reconciliation is an invariant of every scenario rather than a property of
    one, so it is asserted here instead of repeated in each test.
    """
    built: list[EngineHarness] = []

    def build(contracts: list[E.OptionContractVersion] | None = None, **cfg) -> EngineHarness:
        h = EngineHarness(base_config(**cfg), universe() if contracts is None else contracts)
        built.append(h)
        return h

    yield build

    for h in built:
        assert h.engine.ledger_reconciles()
        assert h.finalize().ledger_reconciles


def opened_long(engine_factory, cv: int = CALL_100, qty: int = 1, price: float = 5.00,
                **cfg) -> EngineHarness:
    """Harness holding `qty` long contracts of `cv`, filled at `price` on day 2."""
    h = engine_factory(**cfg)
    h.bar(day_ns(1), bars_at(1, {cv: price}), groups=[group(buy(cv, qty))])
    h.bar(day_ns(2), bars_at(2, {cv: price}))
    assert h.quantity_of(cv) == qty
    return h


class TestExecutionTiming:
    def test_fill_price_is_next_bar_open_not_signal_bar_close(self, engine):
        h = engine()
        h.bar(day_ns(1), [make_bar(CALL_100, timestamp_ns=day_ns(1), price=5.00)],
              groups=[group(buy(CALL_100))])
        h.bar(day_ns(2), [make_bar(CALL_100, timestamp_ns=day_ns(2), price=9.00,
                                   open_price=7.00)])
        (fill,) = h.fills()
        assert fill.fill_price == 7.00

    def test_nothing_fills_on_the_bar_the_order_was_submitted(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        assert len(h.fills()) == 0
        assert h.quantity_of(CALL_100) == 0
        assert h.cash_micros == INITIAL_CASH_MICROS

    def test_fill_is_stamped_with_the_execution_bar_not_the_signal_bar(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))
        assert h.fills()[0].filled_at == day_ns(2)

    def test_same_bar_close_fills_at_a_price_next_bar_open_cannot_see(self, engine):
        """The gap between the two timings is exactly the lookahead the default avoids."""
        filled = {}
        for timing in (E.ExecutionTiming.NEXT_BAR_OPEN, E.ExecutionTiming.SAME_BAR_CLOSE):
            h = engine(timing=timing)
            h.bar(day_ns(1), [make_bar(CALL_100, timestamp_ns=day_ns(1), price=5.00)],
                  groups=[group(buy(CALL_100))])
            h.bar(day_ns(2), [make_bar(CALL_100, timestamp_ns=day_ns(2), price=9.00,
                                       open_price=7.00)])
            filled[timing] = h.fills()[0].fill_price

        assert filled[E.ExecutionTiming.SAME_BAR_CLOSE] == 5.00
        assert filled[E.ExecutionTiming.NEXT_BAR_OPEN] == 7.00
        assert filled[E.ExecutionTiming.SAME_BAR_CLOSE] != filled[E.ExecutionTiming.NEXT_BAR_OPEN]

    def test_same_bar_close_fills_on_the_submitting_bar(self, engine):
        h = engine(timing=E.ExecutionTiming.SAME_BAR_CLOSE)
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        assert len(h.fills()) == 1
        assert h.quantity_of(CALL_100) == 1

    def test_order_submitted_on_the_last_bar_never_fills(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        metrics = h.finalize()
        assert len(h.fills()) == 0
        assert metrics.fill_count == 0
        assert h.cash_micros == INITIAL_CASH_MICROS


class TestGroupAtomicity:
    def test_two_valid_legs_both_fill(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00, CALL_105: 3.00}),
              groups=[group(buy(CALL_100), sell(CALL_105))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00, CALL_105: 3.00}))
        assert len(h.fills()) == 2
        assert h.quantity_of(CALL_100) == 1
        assert h.quantity_of(CALL_105) == -1

    def test_one_leg_without_market_data_rejects_both_legs(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00, CALL_105: 3.00}),
              groups=[group(buy(CALL_100), sell(CALL_105))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))
        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.NO_MARKET_DATA] * 2
        assert h.quantity_of(CALL_100) == 0

    def test_four_leg_condor_fills_all_legs(self, engine):
        h = engine()
        prices = {CALL_105: 2.00, CALL_110: 1.00, PUT_95: 2.00, PUT_90: 1.00}
        h.bar(day_ns(1), bars_at(1, prices),
              groups=[group(sell(CALL_105), buy(CALL_110), sell(PUT_95), buy(PUT_90))])
        h.bar(day_ns(2), bars_at(2, prices))
        assert len(h.fills()) == 4
        assert [h.quantity_of(cv) for cv in (CALL_105, CALL_110, PUT_95, PUT_90)] == [-1, 1, -1, 1]

    def test_four_leg_condor_rejects_every_leg_when_one_is_unpriced(self, engine):
        h = engine()
        prices = {CALL_105: 2.00, CALL_110: 1.00, PUT_95: 2.00, PUT_90: 1.00}
        h.bar(day_ns(1), bars_at(1, prices),
              groups=[group(sell(CALL_105), buy(CALL_110), sell(PUT_95), buy(PUT_90))])
        h.bar(day_ns(2), bars_at(2, {CALL_105: 2.00, CALL_110: 1.00, PUT_95: 2.00}))
        assert len(h.fills()) == 0
        assert len(h.rejections()) == 4
        assert h.engine.account_state().open_position_count == 0

    def test_rejected_group_leaves_the_position_book_untouched(self, engine):
        h = opened_long(engine)
        before_cash = h.cash_micros
        before = [(p.contract_version_id, p.quantity, p.cost_basis_micros) for p in h.positions()]

        h.bar(day_ns(3), bars_at(3, {CALL_100: 5.00}),
              groups=[group(buy(CALL_100), buy(CALL_105))])
        h.bar(day_ns(4), bars_at(4, {CALL_100: 5.00}))

        assert len(h.rejections()) == 2
        assert h.cash_micros == before_cash
        assert [(p.contract_version_id, p.quantity, p.cost_basis_micros) for p in h.positions()] == before

    def test_rejection_reason_is_propagated_to_every_leg(self, engine):
        h = engine()
        g = group(buy(CALL_100), sell(CALL_105), buy(CALL_110))
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00, CALL_105: 3.00, CALL_110: 1.00}), groups=[g])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00, CALL_110: 1.00}))

        assert len(h.rejections()) == 3
        assert {r.reason for r in h.rejections()} == {E.RejectReason.NO_MARKET_DATA}
        assert {r.group_id for r in h.rejections()} == {g.group_id}
        assert len({r.order_id for r in h.rejections()}) == 3


class TestLimitOrders:
    @pytest.mark.parametrize(
        "side, limit, fills",
        [
            (buy, 6.00, True),
            (buy, 4.00, False),
            (sell, 4.00, True),
            (sell, 6.00, False),
            (buy, 5.00, True),
            (sell, 5.00, True),
        ],
        ids=["buy_above", "buy_below", "sell_below", "sell_above",
             "buy_at_price", "sell_at_price"],
    )
    def test_limit_is_satisfied_only_on_the_favorable_side_inclusive(
        self, engine, side, limit, fills
    ):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}),
              groups=[group(side(CALL_100, limit_price=limit))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))

        if fills:
            assert len(h.fills()) == 1
            assert h.fills()[0].fill_price == 5.00
        else:
            assert len(h.fills()) == 0
            assert reasons(h) == [E.RejectReason.LIMIT_NOT_SATISFIED]

    def test_one_unsatisfied_limit_rejects_the_whole_group(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00, CALL_105: 3.00}),
              groups=[group(buy(CALL_100, limit_price=6.00),
                            buy(CALL_105, limit_price=1.00))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00, CALL_105: 3.00}))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.LIMIT_NOT_SATISFIED] * 2
        assert h.cash_micros == INITIAL_CASH_MICROS


class TestDataQualityGates:
    @pytest.mark.parametrize("reject_stale, fills", [(True, 0), (False, 1)])
    def test_stale_bar_is_rejected_only_when_configured(self, engine, reject_stale, fills):
        h = engine(reject_stale=reject_stale)
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}, stale=True))

        assert len(h.fills()) == fills
        assert reasons(h) == ([E.RejectReason.STALE_MARKET_DATA] if fills == 0 else [])

    def test_invalid_analytics_rejects_an_opening_order(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}, analytics_valid=False))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.ANALYTICS_REJECTED]

    def test_invalid_analytics_blocks_opening_but_never_exiting(self, engine):
        """A strategy must always be able to close what it already holds."""
        h = opened_long(engine)
        h.bar(day_ns(3), bars_at(3, {CALL_100: 5.00}),
              groups=[group(buy(CALL_100)), group(sell(CALL_100, reduce_only=True))])
        h.bar(day_ns(4), bars_at(4, {CALL_100: 5.00}, analytics_valid=False))

        assert reasons(h) == [E.RejectReason.ANALYTICS_REJECTED]
        assert h.quantity_of(CALL_100) == 0
        assert h.cash_micros == INITIAL_CASH_MICROS

    def test_untradable_contract_blocks_opening_but_never_exiting(self, engine):
        """A contract that stops trading while held must still be exitable."""
        h = opened_long(engine)
        h.engine.add_contract(make_contract(CALL_100, strike=100.0, tradable=False))

        h.bar(day_ns(3), bars_at(3, {CALL_100: 5.00}),
              groups=[group(buy(CALL_100)), group(sell(CALL_100, reduce_only=True))])
        h.bar(day_ns(4), bars_at(4, {CALL_100: 5.00}))

        assert reasons(h) == [E.RejectReason.CONTRACT_NOT_TRADABLE]
        assert h.quantity_of(CALL_100) == 0

    def test_non_positive_mark_is_rejected_as_missing_market_data(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 0.00}))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.NO_MARKET_DATA]


class TestBuyingPower:
    def test_short_put_beyond_collateral_is_rejected(self, engine):
        """
        A cash-secured put must set aside the whole strike. The $500 premium
        counts toward it, so $9,499 falls one dollar short of the $10,000
        requirement.
        """
        h = engine(cash=9_499.0, margin=E.MarginModel.CASH_ACCOUNT)
        h.bar(day_ns(1), bars_at(1, {PUT_100: 5.00}), groups=[group(sell(PUT_100))])
        h.bar(day_ns(2), bars_at(2, {PUT_100: 5.00}))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.INSUFFICIENT_BUYING_POWER]
        assert h.cash_micros == 9_499 * 10**6

    def test_order_that_exactly_fits_available_cash_fills(self, engine):
        h = engine(cash=9_500.0, margin=E.MarginModel.CASH_ACCOUNT)
        h.bar(day_ns(1), bars_at(1, {PUT_100: 5.00}), groups=[group(sell(PUT_100))])
        h.bar(day_ns(2), bars_at(2, {PUT_100: 5.00}))

        assert len(h.fills()) == 1
        assert h.cash_micros == 10_000 * 10**6

    def test_buying_power_is_evaluated_on_the_whole_group(self, engine):
        """
        The short leg alone needs the Reg-T naked requirement, which $1,000 cannot
        cover; paired with the long put the requirement is only the strike width.
        """
        naked = engine(cash=1_000.0, margin=E.MarginModel.REG_T)
        naked.bar(day_ns(1), bars_at(1, {PUT_100: 5.00}), groups=[group(sell(PUT_100))])
        naked.bar(day_ns(2), bars_at(2, {PUT_100: 5.00}))
        assert reasons(naked) == [E.RejectReason.INSUFFICIENT_BUYING_POWER]

        spread = engine(cash=1_000.0, margin=E.MarginModel.REG_T)
        prices = {PUT_100: 5.00, PUT_95: 3.00}
        spread.bar(day_ns(1), bars_at(1, prices), groups=[group(sell(PUT_100), buy(PUT_95))])
        spread.bar(day_ns(2), bars_at(2, prices))
        assert len(spread.fills()) == 2
        assert spread.cash_micros == 1_200 * 10**6

    @pytest.mark.xfail(reason="engine.h:526-532 credits the bought option's own market "
                              "value back into the buying-power check, and both RegT and "
                              "CashAccount charge zero for long options, so a long "
                              "purchase can never fail and cash goes negative")
    def test_buy_costing_more_than_cash_is_rejected(self, engine):
        h = engine(cash=400.0)
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(CALL_100))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.INSUFFICIENT_BUYING_POWER]
        assert h.cash_micros == 400 * 10**6

    @pytest.mark.xfail(reason="engine.h:553-561 builds the margin context from the live "
                              "book, so a probe position has no mark and the Reg-T naked "
                              "requirement drops its premium term: $2,000 is charged at "
                              "submission but $2,500 once the position is held")
    def test_naked_requirement_at_submission_matches_the_held_requirement(self, engine):
        h = engine(cash=2_499.0, margin=E.MarginModel.REG_T)
        h.bar(day_ns(1), bars_at(1, {PUT_100: 5.00}), groups=[group(sell(PUT_100))])
        h.bar(day_ns(2), bars_at(2, {PUT_100: 5.00}))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.INSUFFICIENT_BUYING_POWER]


class TestCashAndPositionBookkeeping:
    def test_buying_one_contract_debits_price_times_multiplier(self, engine):
        h = opened_long(engine, price=5.00)
        assert h.cash_micros == INITIAL_CASH_MICROS - 5 * PER_CONTRACT * 10**6

    def test_selling_one_contract_credits_price_times_multiplier(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(sell(CALL_100))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))
        assert h.cash_micros == INITIAL_CASH_MICROS + 5 * PER_CONTRACT * 10**6

    @pytest.mark.parametrize("side, quantity", [(buy, 2), (sell, -2)])
    def test_quantity_sign_follows_the_side(self, engine, side, quantity):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(side(CALL_100, 2))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))
        assert h.quantity_of(CALL_100) == quantity

    @pytest.mark.parametrize("side", [buy, sell], ids=["buy", "sell"])
    def test_cost_basis_equals_the_signed_cash_paid(self, engine, side):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(side(CALL_100))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))
        (fill,) = h.fills()
        (position,) = h.positions()
        assert position.cost_basis_micros == -fill.net_cash_micros

    def test_round_trip_at_the_same_price_returns_cash_exactly(self, engine):
        h = opened_long(engine, price=5.00)
        h.bar(day_ns(3), bars_at(3, {CALL_100: 5.00}),
              groups=[group(sell(CALL_100, reduce_only=True))])
        h.bar(day_ns(4), bars_at(4, {CALL_100: 5.00}))

        assert h.cash_micros == INITIAL_CASH_MICROS
        assert h.quantity_of(CALL_100) == 0
        assert h.positions() == []

    def test_partial_close_realizes_a_proportional_share_of_average_cost(self, engine):
        h = opened_long(engine, qty=3, price=5.00)
        h.bar(day_ns(3), bars_at(3, {CALL_100: 5.00}),
              groups=[group(sell(CALL_100, 1, reduce_only=True))])
        h.bar(day_ns(4), bars_at(4, {CALL_100: 7.00}))

        (position,) = h.positions()
        assert position.quantity == 2
        assert position.average_cost == 5.00 * PER_CONTRACT
        assert position.cost_basis_micros == 2 * 5 * PER_CONTRACT * 10**6
        assert position.realized_pnl == (7.00 - 5.00) * PER_CONTRACT

    def test_selling_through_zero_opens_the_opposite_side_with_a_new_basis(self, engine):
        h = opened_long(engine, qty=1, price=5.00)
        h.bar(day_ns(3), bars_at(3, {CALL_100: 5.00}), groups=[group(sell(CALL_100, 3))])
        h.bar(day_ns(4), bars_at(4, {CALL_100: 7.00}))

        (position,) = h.positions()
        assert position.quantity == -2
        assert position.realized_pnl == (7.00 - 5.00) * PER_CONTRACT
        assert position.cost_basis_micros == -2 * 7 * PER_CONTRACT * 10**6

    def test_fill_cash_flow_sums_to_the_ledger_balance(self, engine):
        h = engine()
        prices = {CALL_100: 5.00, CALL_105: 3.00}
        h.bar(day_ns(1), bars_at(1, prices), groups=[group(buy(CALL_100), sell(CALL_105))])
        h.bar(day_ns(2), bars_at(2, prices))

        assert h.cash_micros == INITIAL_CASH_MICROS + sum(f.net_cash_micros for f in h.fills())


class TestRejectionsDoNotCorruptState:
    def test_a_series_of_rejections_leaves_cash_and_ledger_intact(self, engine):
        h = engine(cash=400.0, margin=E.MarginModel.CASH_ACCOUNT)
        submissions = [
            group(buy(UNKNOWN_CV)),
            group(buy(CALL_100, limit_price=1.00)),
            group(sell(PUT_100)),
            group(buy(CALL_100), buy(CALL_110)),
        ]
        for day, g in enumerate(submissions, start=1):
            h.bar(day_ns(day), bars_at(day, {CALL_100: 5.00, PUT_100: 5.00}), groups=[g])
        h.bar(day_ns(len(submissions) + 1), bars_at(len(submissions) + 1,
                                                   {CALL_100: 5.00, PUT_100: 5.00}))

        assert len(h.fills()) == 0
        assert len(h.rejections()) == 5
        assert h.cash_micros == 400 * 10**6
        assert h.positions() == []
        assert h.engine.ledger_reconciles()


class TestOrderIdsAndGrouping:
    def test_every_leg_gets_a_unique_id_and_shares_the_group_id(self, engine):
        h = engine()
        g = group(buy(CALL_100), sell(CALL_105), buy(CALL_110))
        prices = {CALL_100: 5.00, CALL_105: 3.00, CALL_110: 1.00}
        h.bar(day_ns(1), bars_at(1, prices), groups=[g])
        h.bar(day_ns(2), bars_at(2, prices))

        fills = h.fills()
        assert len({f.order_id for f in fills}) == 3
        assert all(f.order_id != 0 for f in fills)
        assert {f.group_id for f in fills} == {g.group_id}

    def test_order_ids_stay_unique_across_groups(self, engine):
        h = engine()
        first, second = group(buy(CALL_100), buy(CALL_105)), group(buy(CALL_110))
        prices = {CALL_100: 5.00, CALL_105: 3.00, CALL_110: 1.00}
        h.bar(day_ns(1), bars_at(1, prices), groups=[first, second])
        h.bar(day_ns(2), bars_at(2, prices))

        fills = h.fills()
        assert len({f.order_id for f in fills}) == 3
        assert {f.group_id for f in fills} == {first.group_id, second.group_id}


class TestUnknownContract:
    def test_order_on_an_unregistered_contract_is_rejected(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}), groups=[group(buy(UNKNOWN_CV))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.CONTRACT_NOT_TRADABLE]
        assert h.cash_micros == INITIAL_CASH_MICROS

    def test_unregistered_leg_rejects_its_whole_group(self, engine):
        h = engine()
        h.bar(day_ns(1), bars_at(1, {CALL_100: 5.00}),
              groups=[group(buy(CALL_100), buy(UNKNOWN_CV))])
        h.bar(day_ns(2), bars_at(2, {CALL_100: 5.00}))

        assert len(h.fills()) == 0
        assert reasons(h) == [E.RejectReason.CONTRACT_NOT_TRADABLE] * 2


class TestLedgerReconciliation:
    def test_journal_sum_equals_the_running_balance_exactly(self, engine):
        h = engine()
        prices = {CALL_100: 5.00, CALL_105: 3.00}
        h.bar(day_ns(1), bars_at(1, prices), groups=[group(buy(CALL_100), sell(CALL_105))])
        h.bar(day_ns(2), bars_at(2, prices), groups=[group(sell(CALL_100, reduce_only=True))])
        h.bar(day_ns(3), bars_at(3, prices), groups=[group(buy(UNKNOWN_CV))])
        h.bar(day_ns(4), bars_at(4, prices))

        assert sum(e.amount_micros for e in h.engine.ledger_entries()) == h.cash_micros
        assert h.engine.ledger_reconciles()
