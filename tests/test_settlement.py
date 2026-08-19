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
        """Post 4:1 split terms: strike 25 against a 400-share deliverable."""
        contract = make_contract(1, strike=25.0, deliverable_shares=400)
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
