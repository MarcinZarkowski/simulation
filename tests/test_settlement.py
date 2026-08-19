"""Expiration and settlement of equity options under physical delivery."""
from __future__ import annotations

import pytest

import obt_engine as E
from optionsbacktester.strategy import buy, group, sell
from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

SIGNAL_DAY = 1
FILL_DAY = 2
EXPIRY_DAY = 30

STRIKE = 100.0
PREMIUM = 5.0
INITIAL_CASH = 100_000.0


def dollars(amount: float) -> int:
    """Microdollars, the engine's exact integer money unit."""
    return round(amount * 1_000_000)


def open_position(contracts, *legs, config=None, premiums=None) -> EngineHarness:
    """
    Fill `legs` and return the harness holding the resulting option position.

    Execution is next-bar-open, so a signal on the first bar needs a second bar
    before it becomes a position.
    """
    prices = premiums or {c.id: PREMIUM for c in contracts}

    def bars(at: int) -> list:
        return [make_bar(c.id, timestamp_ns=at, price=prices[c.id]) for c in contracts]

    h = EngineHarness(config or base_config(), contracts)
    h.bar(day_ns(SIGNAL_DAY), bars(day_ns(SIGNAL_DAY)),
          groups=[group(*legs)], underlying={"TEST": STRIKE})
    h.bar(day_ns(FILL_DAY), bars(day_ns(FILL_DAY)), underlying={"TEST": STRIKE})
    assert not h.rejections()
    return h


def expire(h: EngineHarness, spot: float, *, day: int = EXPIRY_DAY):
    return h.bar(day_ns(day), [], underlying={"TEST": spot})


def entries_of(h: EngineHarness, kind: E.LedgerEntryKind) -> list:
    return [e for e in h.engine.ledger_entries() if e.kind == kind]


class TestExpirationWithoutExercise:
    def test_otm_long_call_is_removed_and_keeps_only_the_premium_debit(self):
        h = open_position([make_contract(1, strike=STRIKE)], buy(1, 1))
        expire(h, spot=90.0)

        assert h.quantity_of(1) == 0
        assert h.shares_of("TEST") == 0
        assert h.cash_micros == dollars(INITIAL_CASH - PREMIUM * 100)
        assert h.engine.metrics().exercise_count == 0
        assert h.engine.ledger_reconciles()

    def test_worthless_expiration_realizes_exactly_minus_the_premium_paid(self):
        h = open_position([make_contract(1, strike=STRIKE)], buy(1, 1))
        expire(h, spot=90.0)
        metrics = h.finalize()

        assert metrics.realized_pnl == -PREMIUM * 100
        assert metrics.net_pnl == -PREMIUM * 100
        assert not h.positions()
        assert h.engine.ledger_reconciles()

    def test_expiration_count_covers_itm_and_otm_positions_alike(self):
        itm = make_contract(1, strike=90.0)
        otm = make_contract(2, strike=110.0)
        h = open_position([itm, otm], buy(1, 1), buy(2, 1))
        expire(h, spot=STRIKE)

        metrics = h.engine.metrics()
        assert metrics.expiration_count == 2
        assert metrics.exercise_count == 1
        assert h.engine.ledger_reconciles()


DIRECTIONS = [
    pytest.param(True, True, 110.0, 100, -1, id="long_call_pays_strike_and_receives_shares"),
    pytest.param(True, False, 110.0, -100, 1, id="short_call_receives_strike_and_delivers_shares"),
    pytest.param(False, True, 90.0, -100, 1, id="long_put_receives_strike_and_delivers_shares"),
    pytest.param(False, False, 90.0, 100, -1, id="short_put_pays_strike_and_receives_shares"),
]


class TestPhysicalDelivery:
    @pytest.mark.parametrize("is_call,is_long,spot,shares,cash_sign", DIRECTIONS)
    def test_itm_settlement_moves_shares_and_strike_cash_in_opposite_directions(
        self, is_call, is_long, spot, shares, cash_sign
    ):
        contract = make_contract(1, strike=STRIKE, is_call=is_call)
        leg = buy(1, 1) if is_long else sell(1, 1)
        h = open_position([contract], leg)
        cash_before = h.cash_micros

        expire(h, spot=spot)

        assert h.shares_of("TEST") == shares
        assert h.cash_micros - cash_before == cash_sign * dollars(STRIKE * 100)
        assert h.quantity_of(1) == 0

        metrics = h.engine.metrics()
        assert metrics.exercise_count == (1 if is_long else 0)
        assert metrics.assignment_count == (0 if is_long else 1)
        assert h.engine.ledger_reconciles()

    def test_exercising_three_contracts_delivers_three_hundred_shares(self):
        h = open_position([make_contract(1, strike=STRIKE)], buy(1, 3))
        cash_before = h.cash_micros

        expire(h, spot=110.0)

        assert h.shares_of("TEST") == 300
        assert h.cash_micros - cash_before == -dollars(STRIKE * 300)
        assert h.engine.metrics().exercise_count == 1
        assert h.engine.ledger_reconciles()

    def test_assigned_covered_call_nets_shares_to_zero_instead_of_going_short(self):
        """
        The shares acquired by exercising the long call are the ones delivered
        when the later short call is assigned, so the account never goes short.
        """
        long_call = make_contract(1, strike=STRIKE, expiry_day=20)
        short_call = make_contract(2, strike=105.0, expiry_day=EXPIRY_DAY)
        h = open_position([long_call, short_call], buy(1, 1), sell(2, 1),
                          premiums={1: PREMIUM, 2: 3.0})

        expire(h, spot=110.0, day=20)
        assert h.shares_of("TEST") == 100

        expire(h, spot=110.0, day=EXPIRY_DAY)

        assert h.shares_of("TEST") == 0
        assert h.equity_positions() == []
        assert h.cash_micros == dollars(
            INITIAL_CASH - PREMIUM * 100 + 3.0 * 100 - STRIKE * 100 + 105.0 * 100)

        metrics = h.engine.metrics()
        assert metrics.exercise_count == 1
        assert metrics.assignment_count == 1
        assert h.engine.ledger_reconciles()


class TestExerciseByExceptionThreshold:
    @pytest.mark.parametrize("is_call,spot,exercised", [
        pytest.param(True, 100.01, True, id="call_one_cent_itm_exercises"),
        pytest.param(True, 100.005, False, id="call_half_cent_itm_expires"),
        pytest.param(False, 99.99, True, id="put_one_cent_itm_exercises"),
        pytest.param(False, 99.995, False, id="put_half_cent_itm_expires"),
    ])
    def test_one_cent_of_intrinsic_is_the_exercise_boundary(self, is_call, spot, exercised):
        h = open_position([make_contract(1, strike=STRIKE, is_call=is_call)], buy(1, 1))
        expire(h, spot=spot)

        assert h.engine.metrics().exercise_count == int(exercised)
        assert (h.shares_of("TEST") != 0) is exercised
        assert h.engine.metrics().expiration_count == 1
        assert h.engine.ledger_reconciles()

    def test_threshold_constant_is_one_cent(self):
        assert dollars(E.AUTOMATIC_EXERCISE_THRESHOLD) == 10_000


class TestAssignmentPolicies:
    def test_expiration_only_lets_an_itm_long_call_expire_worthless(self):
        config = base_config(assignment=E.AssignmentPolicy.EXPIRATION_ONLY)
        h = open_position([make_contract(1, strike=STRIKE)], buy(1, 1), config=config)
        expire(h, spot=120.0)

        metrics = h.engine.metrics()
        assert metrics.exercise_count == 0
        assert metrics.expiration_count == 1
        assert h.shares_of("TEST") == 0
        assert h.quantity_of(1) == 0
        assert h.cash_micros == dollars(INITIAL_CASH - PREMIUM * 100)
        assert h.engine.ledger_reconciles()

    def test_explicit_exercise_only_leaves_the_position_open_past_expiration(self):
        """Nothing is settled automatically, so the contract outlives its own expiry."""
        config = base_config(assignment=E.AssignmentPolicy.EXPLICIT_EXERCISE_ONLY)
        h = open_position([make_contract(1, strike=STRIKE)], buy(1, 1), config=config)
        expire(h, spot=120.0, day=EXPIRY_DAY + 1)

        metrics = h.engine.metrics()
        assert h.quantity_of(1) == 1
        assert metrics.expiration_count == 0
        assert metrics.exercise_count == 0
        assert h.shares_of("TEST") == 0
        assert h.engine.ledger_reconciles()


class TestAdjustedDeliverable:
    def test_exercise_uses_the_deliverable_rather_than_a_hundred_shares(self):
        """
        Post 4:1 split terms: strike 25 against a 400-share deliverable.

        The quote multiplier moves with the deliverable, which is what conserves
        the aggregate exercise price: 25 x 400 equals the original 100 x 100. A
        400-share deliverable left at a multiplier of 100 would make the aggregate
        $2,500 for $12,000 of stock, which is not a contract OCC would issue.
        """
        contract = make_contract(1, strike=25.0, deliverable_shares=400, multiplier=400)
        h = open_position([contract], buy(1, 1))
        cash_before = h.cash_micros

        expire(h, spot=30.0)

        assert h.shares_of("TEST") == 400
        assert h.cash_micros - cash_before == -dollars(25.0 * 400)
        assert h.cash_micros - cash_before != -dollars(25.0 * 100)
        assert h.engine.ledger_reconciles()

    @pytest.mark.parametrize("is_long,spot,sign", [
        pytest.param(True, 110.0, 1, id="exercise_receives_the_cash_component"),
        pytest.param(False, 110.0, -1, id="assignment_delivers_the_cash_component"),
    ])
    def test_deliverable_cash_follows_the_share_leg(self, is_long, spot, sign):
        contract = make_contract(1, strike=STRIKE, deliverable_cash=12.5)
        leg = buy(1, 1) if is_long else sell(1, 1)
        h = open_position([contract], leg)

        expire(h, spot=spot)

        cash_entries = entries_of(h, E.LedgerEntryKind.CASH_SETTLEMENT)
        assert len(cash_entries) == 1
        assert cash_entries[0].amount_micros == sign * dollars(12.5)
        assert h.engine.ledger_reconciles()


class TestLedgerEntries:
    def test_exercise_posts_one_exercise_settlement_for_the_strike_amount(self):
        h = open_position([make_contract(1, strike=STRIKE)], buy(1, 1))
        expire(h, spot=110.0)

        settlements = entries_of(h, E.LedgerEntryKind.EXERCISE_SETTLEMENT)
        assert len(settlements) == 1
        assert settlements[0].amount_micros == -dollars(STRIKE * 100)
        assert entries_of(h, E.LedgerEntryKind.ASSIGNMENT_SETTLEMENT) == []
        assert h.engine.ledger_reconciles()

    def test_assignment_posts_one_assignment_settlement_for_the_strike_amount(self):
        h = open_position([make_contract(1, strike=STRIKE)], sell(1, 1))
        expire(h, spot=110.0)

        settlements = entries_of(h, E.LedgerEntryKind.ASSIGNMENT_SETTLEMENT)
        assert len(settlements) == 1
        assert settlements[0].amount_micros == dollars(STRIKE * 100)
        assert entries_of(h, E.LedgerEntryKind.EXERCISE_SETTLEMENT) == []
        assert h.engine.ledger_reconciles()

    def test_worthless_expiration_posts_no_settlement_entry(self):
        h = open_position([make_contract(1, strike=STRIKE)], buy(1, 1))
        before = len(h.engine.ledger_entries())
        expire(h, spot=90.0)

        assert len(h.engine.ledger_entries()) == before
        assert h.engine.ledger_reconciles()

    def test_journal_sum_equals_cash_after_a_mixed_settlement(self):
        long_call = make_contract(1, strike=90.0)
        short_put = make_contract(2, strike=110.0, is_call=False)
        h = open_position([long_call, short_put], buy(1, 1), sell(2, 1))
        expire(h, spot=STRIKE)

        metrics = h.engine.metrics()
        assert (metrics.exercise_count, metrics.assignment_count) == (1, 1)
        assert h.shares_of("TEST") == 200

        journal = sum(e.amount_micros for e in h.engine.ledger_entries())
        assert journal == h.cash_micros
        assert h.engine.ledger_reconciles()
        assert h.finalize().ledger_reconciles


class TestSpreadExpiration:
    def test_vertical_settles_each_leg_and_pays_the_spread_intrinsic(self):
        """
        Long 95 call exercised, short 105 call expires. Physical delivery pays
        the strike in cash and hands over shares, so the spread's $5 intrinsic
        shows up as strike cash plus the market value of the delivered shares.
        """
        spot = STRIKE
        lower = make_contract(1, strike=95.0)
        upper = make_contract(2, strike=105.0)
        h = open_position([lower, upper], buy(1, 1), sell(2, 1),
                          premiums={1: 7.0, 2: 2.0})
        cash_before = h.cash_micros

        expire(h, spot=spot)

        settlement_cash = h.cash_micros - cash_before
        assert settlement_cash == -dollars(95.0 * 100)
        assert h.shares_of("TEST") == 100
        assert settlement_cash + h.shares_of("TEST") * dollars(spot) == dollars(5.0 * 100)

        metrics = h.engine.metrics()
        assert metrics.expiration_count == 2
        assert metrics.exercise_count == 1
        assert metrics.assignment_count == 0
        assert h.quantity_of(1) == 0
        assert h.quantity_of(2) == 0
        assert h.engine.ledger_reconciles()


class TestSessionCloseSettlement:
    """
    Expiration is an instant no bar occupies.

    Contracts expire at the 16:00 ET close while minute bars are stamped at minute
    start, so the last bar of the day is 15:59 and a settlement rule keyed on
    `expiration <= bar_timestamp` never fires on the expiration date. It deferred
    every expiry to the next session's open, injecting an overnight or weekend gap
    into the settlement price of every expiring position.
    """

    CALL = 1
    EXPIRATION = day_ns(19) + 21 * 3600 * 10**9      # 16:00 ET on day 19
    LAST_BAR = day_ns(19) + 20 * 3600 * 10**9 + 59 * 60 * 10**9   # 15:59 ET
    NEXT_SESSION_CLOSE = day_ns(20)

    def _contract(self):
        c = make_contract(self.CALL, strike=100.0, expiry_day=19)
        c.expiration = self.EXPIRATION
        c.valid_to = self.EXPIRATION
        return c

    def _held_position(self, spot_at_last_bar: float):
        """Buys one call, then walks to the final 15:59 bar of expiration day."""
        h = EngineHarness(base_config(cash=100_000.0), [self._contract()])
        first = day_ns(19) + 20 * 3600 * 10**9
        h.bar(first, [make_bar(self.CALL, timestamp_ns=first, price=5.00)],
              underlying={"TEST": 100.0}, groups=[group(buy(self.CALL, 1))])
        h.bar(self.LAST_BAR, [make_bar(self.CALL, timestamp_ns=self.LAST_BAR, price=5.00)],
              underlying={"TEST": spot_at_last_bar})
        return h

    def test_position_is_still_open_after_the_last_bar(self):
        """No bar reaches the expiration instant, so end_bar alone cannot settle."""
        h = self._held_position(120.0)
        assert h.quantity_of(self.CALL) == 1

    def test_session_close_settles_on_the_expiration_date(self):
        h = self._held_position(120.0)
        h.engine.end_session(self.NEXT_SESSION_CLOSE)
        assert h.quantity_of(self.CALL) == 0
        assert h.finalize().exercise_count == 1

    def test_settlement_uses_the_expiration_day_spot_not_the_next_session(self):
        """
        The shares must be acquired at the strike against the expiration day's
        closing spot. Settling a session later would substitute the next open,
        which is where the overnight gap entered.
        """
        h = self._held_position(120.0)
        h.engine.end_session(self.NEXT_SESSION_CLOSE)
        assert h.shares_of("TEST") == 100
        assert h.engine.ledger_reconciles()

    def test_out_of_the_money_at_the_close_expires_worthless(self):
        h = self._held_position(95.0)
        h.engine.end_session(self.NEXT_SESSION_CLOSE)
        metrics = h.finalize()
        assert metrics.exercise_count == 0
        assert metrics.expiration_count == 1
        assert h.shares_of("TEST") == 0

    def test_a_session_close_before_expiration_settles_nothing(self):
        h = self._held_position(120.0)
        h.engine.end_session(day_ns(19))
        assert h.quantity_of(self.CALL) == 1

    def test_session_close_is_idempotent(self):
        h = self._held_position(120.0)
        h.engine.end_session(self.NEXT_SESSION_CLOSE)
        cash_after_first = h.cash_micros
        h.engine.end_session(self.NEXT_SESSION_CLOSE)
        assert h.cash_micros == cash_after_first
        assert h.finalize().expiration_count == 1


class TestCorporateActionFailClosed:
    """
    An adjustment the engine cannot source must reduce the book, not silently
    reshape it. Two of these paths previously failed OPEN while the documentation
    promised fail-closed.
    """

    PARENT = 1
    CHILD = 2
    EFFECTIVE = 10

    def _transition(self, parent_contracts, child_contracts, *, confirmed=True,
                    effective=EFFECTIVE, event_id=1):
        t = E.CorporateActionTransition()
        t.lineage_event_id = event_id
        t.effective_at = day_ns(effective)
        t.source_available_at = day_ns(effective)
        t.parent_version_id = self.PARENT
        t.child_version_id = self.CHILD
        t.parent_contracts = parent_contracts
        t.child_contracts = child_contracts
        t.occ_confirmed = confirmed
        return t

    def _held_through(self, held, transitions, *, bars_after=2):
        parent = make_contract(self.PARENT, symbol="P", strike=100.0, expiry_day=400)
        child = make_contract(self.CHILD, symbol="C", strike=25.0, expiry_day=400)
        h = EngineHarness(base_config(cash=500_000.0), [parent, child])
        h.engine.queue_corporate_actions(transitions)
        h.bar(day_ns(1), [make_bar(self.PARENT, timestamp_ns=day_ns(1), price=10.0)],
              groups=[group(buy(self.PARENT, held))])
        h.bar(day_ns(2), [make_bar(self.PARENT, timestamp_ns=day_ns(2), price=10.0)])
        for d in range(self.EFFECTIVE + 1, self.EFFECTIVE + 1 + bars_after):
            h.bar(day_ns(d), [make_bar(self.PARENT, timestamp_ns=day_ns(d), price=10.0),
                              make_bar(self.CHILD, timestamp_ns=day_ns(d), price=2.50)])
        return h

    def test_an_even_conversion_transfers_the_position(self):
        h = self._held_through(2, [self._transition(1, 4)])
        assert h.quantity_of(self.CHILD) == 8
        assert h.quantity_of(self.PARENT) == 0
        assert not h.finalize().truncated

    def test_an_uneven_conversion_quarantines_rather_than_truncating(self):
        """
        Three contracts under a 2-for-3 conversion is 4.5 contracts. Truncating to
        3 silently discarded a third of the exposure; OCC settles the remainder in
        cash, which this engine has no primitive for, so it refuses.
        """
        h = self._held_through(3, [self._transition(2, 3)])
        metrics = h.finalize()
        assert h.positions() == []
        assert metrics.truncated
        assert metrics.quarantined_positions == 1

    def test_a_holding_too_small_to_convert_is_not_stranded(self):
        """One contract under a 2-for-3 conversion rounded to zero children."""
        h = self._held_through(1, [self._transition(2, 3)])
        assert h.positions() == []
        assert h.finalize().truncated

    def test_an_unconfirmed_adjustment_emits_one_rejection_not_one_per_bar(self):
        h = self._held_through(1, [self._transition(1, 4, confirmed=False)],
                               bars_after=10)
        assert len(h.rejections()) == 1
        assert h.rejections()[0].reason == E.RejectReason.UNCONFIRMED_LINEAGE

    def test_an_unconfirmed_adjustment_closes_the_position(self):
        """
        Leaving it open kept marking a book the engine had declared unsourceable,
        so the run went on producing a P&L for it.
        """
        h = self._held_through(1, [self._transition(1, 4, confirmed=False)],
                               bars_after=10)
        assert h.positions() == []
        assert h.finalize().truncated
        assert h.engine.ledger_reconciles()

    def test_quarantining_records_a_trade_with_an_adjusted_reason(self):
        h = self._held_through(1, [self._transition(1, 4, confirmed=False)])
        assert any(t.reason == E.CloseReason.ADJUSTED for t in h.engine.trades())

    def test_queueing_twice_keeps_both_events(self):
        """
        Replacing rather than appending silently dropped any transition whose
        effective date fell after the next day carrying lineage data.
        """
        parent = make_contract(self.PARENT, symbol="P", strike=100.0, expiry_day=400)
        child = make_contract(self.CHILD, symbol="C", strike=25.0, expiry_day=400)
        h = EngineHarness(base_config(cash=500_000.0), [parent, child])
        h.engine.queue_corporate_actions([self._transition(1, 4, event_id=1)])
        h.engine.queue_corporate_actions(
            [self._transition(1, 4, effective=20, event_id=2)])

        h.bar(day_ns(1), [make_bar(self.PARENT, timestamp_ns=day_ns(1), price=10.0)],
              groups=[group(buy(self.PARENT, 1))])
        h.bar(day_ns(2), [make_bar(self.PARENT, timestamp_ns=day_ns(2), price=10.0)])
        for d in (11, 12):
            h.bar(day_ns(d), [make_bar(self.PARENT, timestamp_ns=day_ns(d), price=10.0),
                              make_bar(self.CHILD, timestamp_ns=day_ns(d), price=2.50)])
        assert h.quantity_of(self.CHILD) == 4

    def test_a_superseded_version_cannot_be_traded(self):
        """
        Adjustments are applied before pending fills within a bar, so an order
        already in flight could land on a version the adjustment had just retired.
        """
        parent = make_contract(self.PARENT, symbol="P", strike=100.0, expiry_day=400)
        child = make_contract(self.CHILD, symbol="C", strike=25.0, expiry_day=400)
        h = EngineHarness(base_config(cash=500_000.0), [parent, child])
        h.engine.queue_corporate_actions([self._transition(1, 4)])

        h.bar(day_ns(self.EFFECTIVE),
              [make_bar(self.PARENT, timestamp_ns=day_ns(self.EFFECTIVE), price=10.0)],
              groups=[group(buy(self.PARENT, 1))])
        h.bar(day_ns(self.EFFECTIVE + 1),
              [make_bar(self.PARENT, timestamp_ns=day_ns(self.EFFECTIVE + 1), price=10.0)])

        assert h.quantity_of(self.PARENT) == 0
        assert any(r.reason == E.RejectReason.CONTRACT_NOT_TRADABLE
                   for r in h.rejections())


class TestSettlementRequiresAnObservedSpot:
    """
    Settlement without a price is refused rather than guessed.

    Falling back to zero made a put maximally in the money and settled it at a
    fabricated price: measured at +$9,500 of invented profit and a 100-share
    short position, with no rejection. A call silently expired worthless.
    """

    CV = 1

    def _held_with_no_spot(self, is_call: bool):
        # The contract's underlying is never priced in any snapshot.
        c = make_contract(self.CV, strike=100.0, expiry_day=5, is_call=is_call,
                          underlying="UNPRICED")
        h = EngineHarness(base_config(cash=500_000.0), [c])
        for day in (1, 2):
            h.bar(day_ns(day),
                  [make_bar(self.CV, timestamp_ns=day_ns(day), price=5.00)],
                  underlying={"TEST": 100.0},
                  groups=[group(buy(self.CV, 1))] if day == 1 else None)
        h.engine.end_session(day_ns(6))
        return h

    @pytest.mark.parametrize("is_call", [True, False])
    def test_the_position_is_quarantined_not_settled(self, is_call):
        h = self._held_with_no_spot(is_call)
        assert h.positions() == []
        assert h.shares_of("UNPRICED") == 0
        assert h.finalize().exercise_count == 0

    @pytest.mark.parametrize("is_call", [True, False])
    def test_it_is_flagged_rather_than_silent(self, is_call):
        h = self._held_with_no_spot(is_call)
        metrics = h.finalize()
        assert metrics.truncated
        assert len(h.rejections()) == 1
        assert "underlying" in h.rejections()[0].detail

    def test_no_value_is_fabricated(self):
        """A put with no spot previously produced +$9,500 out of nothing."""
        h = self._held_with_no_spot(is_call=False)
        assert h.finalize().net_pnl <= 0.0
        assert h.engine.ledger_reconciles()


class TestAdjustedAggregateExercisePrice:
    """
    The aggregate exercise price is the listed strike times the QUOTE multiplier,
    not the strike times the delivered share count.

    Using strike x shares fabricated value: a 50-share deliverable at strike 100
    with spot 110 paid $5,000 for $5,500 of stock on a contract whose true payoff
    under max(A*S_T + C - K*M, 0) was zero.
    """

    CV = 1
    STRIKE = 100.0
    MULTIPLIER = 100
    DELIVERED = 50

    def _run(self, spot: float):
        c = make_contract(self.CV, strike=self.STRIKE, expiry_day=5,
                          multiplier=self.MULTIPLIER, deliverable_shares=self.DELIVERED)
        h = EngineHarness(base_config(cash=500_000.0), [c])
        for day in (1, 2):
            h.bar(day_ns(day), [make_bar(self.CV, timestamp_ns=day_ns(day), price=5.00)],
                  underlying={"TEST": spot},
                  groups=[group(buy(self.CV, 1))] if day == 1 else None)
        h.engine.end_session(day_ns(6))
        return h

    def test_it_expires_worthless_when_delivered_value_is_below_the_aggregate(self):
        """50 x 110 = 5,500 against an aggregate of 100 x 100 = 10,000."""
        h = self._run(110.0)
        assert h.finalize().exercise_count == 0
        assert h.shares_of("TEST") == 0

    def test_it_exercises_when_delivered_value_exceeds_the_aggregate(self):
        """50 x 250 = 12,500 against 10,000, so the payoff is 2,500."""
        h = self._run(250.0)
        metrics = h.finalize()
        assert metrics.exercise_count == 1
        assert h.shares_of("TEST") == self.DELIVERED
        # Payoff 2,500 less the 5.00 x 100 premium paid.
        assert metrics.net_pnl == pytest.approx(2_500.0 - 5.00 * self.MULTIPLIER)

    def test_a_fractional_deliverable_is_refused(self):
        """OCC settles the fraction in cash-in-lieu, which this engine lacks."""
        c = make_contract(self.CV, strike=100.0, expiry_day=5, multiplier=100)
        c.deliverable_equity_microshares = 66_666_667
        h = EngineHarness(base_config(cash=500_000.0), [c])
        for day in (1, 2):
            h.bar(day_ns(day), [make_bar(self.CV, timestamp_ns=day_ns(day), price=5.00)],
                  underlying={"TEST": 200.0},
                  groups=[group(buy(self.CV, 1))] if day == 1 else None)
        h.engine.end_session(day_ns(6))
        assert h.finalize().truncated
        assert h.shares_of("TEST") == 0

    def test_the_exercise_threshold_scales_with_the_deliverable(self):
        """One cent per share, so $1.00 on a standard contract and $0.50 here."""
        assert E.aggregate_exercise_threshold(100) == pytest.approx(1.00)
        assert E.aggregate_exercise_threshold(50) == pytest.approx(0.50)
        assert E.aggregate_exercise_threshold(400) == pytest.approx(4.00)
