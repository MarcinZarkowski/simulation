"""
The C++ engine against an independent Python reference, compared on exact integers.

The reference below is written from the settlement rules rather than from the
engine's source: an agreement between the two is only evidence if the second
implementation shares no arithmetic with the first. Every comparison is in
microdollars, so a rounding difference of one millionth of a dollar fails.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pytest

import obt_engine as E
from optionsbacktester.strategy import buy, group, sell
from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

MICROS_PER_DOLLAR = 1_000_000
EXERCISE_THRESHOLD_MICROS = 10_000
INITIAL_CASH = 100_000.0
UNDERLYING = "TEST"


def micros(dollars: float) -> int:
    """Nearest microdollar, ties away from zero, matching the engine's Money."""
    scaled = dollars * MICROS_PER_DOLLAR
    return int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)


def truncated(numerator: int, denominator: int) -> int:
    """Integer division truncating toward zero, as C++ does."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator >= 0) == (denominator > 0) else -quotient


@dataclass(frozen=True)
class Terms:
    """Contract terms shared by both engines."""
    cv_id: int
    strike: float
    is_call: bool
    expiry_day: int
    multiplier: int = 100
    shares: int = 100


@dataclass(frozen=True)
class Step:
    """
    One bar: the tape, and the order groups signalled on it.

    Orders signalled here fill on the following step's bar, which is the engine's
    next-bar-open rule.
    """
    day: int
    underlying: float
    prices: dict[int, float] = field(default_factory=dict)
    groups: tuple[tuple[tuple[int, int], ...], ...] = ()


# ---------------------------------------------------------------------------
# Independent reference
# ---------------------------------------------------------------------------
class ReferenceEngine:
    """
    Minimal replica of the ledger: cash in microdollars, options, and shares.

    Knows nothing about spreads, fees, margin, or risk, which is why the
    comparison runs with a zero spread model and a zero fee schedule.
    """

    def __init__(self, terms: list[Terms], cash: float = INITIAL_CASH):
        self.terms = {t.cv_id: t for t in terms}
        self.cash = micros(cash)
        self.positions: dict[int, tuple[int, int]] = {}
        self.shares = 0
        self.spot = 0
        self.marks: dict[int, int] = {}
        self.pending: list[tuple[int, int]] = []

    def step(self, step: Step) -> int:
        """Advance one bar and return the equity at its close."""
        prices = {cv: micros(price) for cv, price in step.prices.items()}
        self.spot = micros(step.underlying)

        for cv, quantity in self.pending:
            self._trade(cv, quantity, prices[cv])
        self.pending = [leg for legs in step.groups for leg in legs]

        self._settle_expirations(day_ns(step.day))
        equity = self.equity(prices)
        self.marks.update(prices)
        return equity

    def equity(self, prices: dict[int, int]) -> int:
        total = self.cash + self.spot * self.shares
        for cv, (quantity, _) in self.positions.items():
            mark = prices.get(cv, self.marks.get(cv, 0))
            total += mark * self.terms[cv].multiplier * quantity
        return total

    def _trade(self, cv: int, quantity: int, price: int) -> None:
        """Premium leaves on a buy and arrives on a sell; basis is average cost."""
        unit = price * self.terms[cv].multiplier
        self.cash -= unit * quantity
        held, basis = self.positions.get(cv, (0, 0))

        closed = min(abs(held), abs(quantity)) if held * quantity < 0 else 0
        if closed:
            basis -= truncated(basis, held) * (closed if held > 0 else -closed)
        opened = abs(quantity) - closed
        basis += unit * opened * (1 if quantity > 0 else -1)

        held += quantity
        if held:
            self.positions[cv] = (held, basis)
        else:
            self.positions.pop(cv, None)

    def _settle_expirations(self, now_ns: int) -> None:
        """
        Physical delivery at expiry, exercised from one cent of intrinsic.

        A long call and a short put receive shares against paying and receiving
        the strike respectively; a long put and a short call deliver them.
        """
        for cv in list(self.positions):
            terms = self.terms[cv]
            if day_ns(terms.expiry_day) > now_ns:
                continue
            held, _ = self.positions.pop(cv)
            strike = micros(terms.strike)
            intrinsic = self.spot - strike if terms.is_call else strike - self.spot
            if intrinsic < EXERCISE_THRESHOLD_MICROS:
                continue

            shares = terms.shares * abs(held)
            receives = (held > 0) == terms.is_call
            self.shares += shares if receives else -shares
            self.cash += -strike * shares if receives else strike * shares

    def quantities(self) -> dict[int, int]:
        return {cv: quantity for cv, (quantity, _) in self.positions.items()}


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------
def run_engine(terms: list[Terms], steps: list[Step], margin) -> tuple[EngineHarness, list[int]]:
    contracts = [
        make_contract(t.cv_id, strike=t.strike, is_call=t.is_call, expiry_day=t.expiry_day,
                      multiplier=t.multiplier, deliverable_shares=t.shares)
        for t in terms
    ]
    harness = EngineHarness(base_config(margin=margin), contracts)
    equity: list[int] = []
    for step in steps:
        bars = [make_bar(cv, timestamp_ns=day_ns(step.day), price=price)
                for cv, price in step.prices.items()]
        groups = [group(*(buy(cv, qty) if qty > 0 else sell(cv, -qty) for cv, qty in legs))
                  for legs in step.groups]
        state = harness.bar(day_ns(step.day), bars, groups=groups,
                            underlying={UNDERLYING: step.underlying})
        equity.append(state.equity_micros)
    return harness, equity


def assert_engines_agree(
    terms: list[Terms], steps: list[Step], *, margin=E.MarginModel.REG_T
) -> tuple[EngineHarness, ReferenceEngine]:
    harness, engine_equity = run_engine(terms, steps, margin)
    reference = ReferenceEngine(terms)
    reference_equity = [reference.step(step) for step in steps]

    assert [(r.reason_name, r.detail) for r in harness.rejections()] == []
    assert engine_equity == reference_equity
    assert harness.cash_micros == reference.cash
    assert {p.contract_version_id: p.quantity for p in harness.positions()} \
        == reference.quantities()
    assert harness.shares_of(UNDERLYING) == reference.shares
    assert harness.finalize().final_equity_micros == reference_equity[-1]
    return harness, reference


class TestSingleLegToExpiry:
    def test_long_call_expiring_worthless_costs_exactly_the_premium(self):
        terms = [Terms(1, 100.0, is_call=True, expiry_day=5)]
        steps = [
            Step(1, 100.0, {1: 4.00}, groups=(((1, 1),),)),
            Step(2, 100.0, {1: 5.00}),
            Step(3, 98.0, {1: 3.50}),
            Step(4, 96.0, {1: 1.20}),
            Step(5, 95.0),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(INITIAL_CASH - 500.0)
        assert reference.shares == 0
        assert harness.engine.metrics().exercise_count == 0

    def test_long_call_exercised_at_expiry_leaves_shares_worth_the_final_price(self):
        terms = [Terms(1, 100.0, is_call=True, expiry_day=5)]
        steps = [
            Step(1, 100.0, {1: 4.00}, groups=(((1, 1),),)),
            Step(2, 100.0, {1: 5.00}),
            Step(3, 102.0, {1: 6.50}),
            Step(4, 105.0, {1: 8.00}),
            Step(5, 110.0),
            Step(6, 112.0),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(INITIAL_CASH - 500.0 - 10_000.0)
        assert reference.shares == 100
        assert harness.finalize().final_equity_micros == micros(89_500.0 + 112.0 * 100)

    def test_settlement_uses_the_deliverable_while_premium_uses_the_multiplier(self):
        """Post 4:1 split terms: quoted per 100 shares, delivering 400 at strike 25."""
        terms = [Terms(1, 25.0, is_call=True, expiry_day=5, multiplier=100, shares=400)]
        steps = [
            Step(1, 26.0, {1: 3.00}, groups=(((1, 1),),)),
            Step(2, 26.0, {1: 4.00}),
            Step(3, 28.0, {1: 5.00}),
            Step(4, 30.0, {1: 6.00}),
            Step(5, 30.0),
            Step(6, 31.0),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(INITIAL_CASH - 400.0 - 25.0 * 400)
        assert reference.shares == 400
        assert harness.finalize().final_equity_micros == micros(89_600.0 + 31.0 * 400)

    @pytest.mark.parametrize("spot,exercised", [
        pytest.param(100.01, True, id="one_cent_itm"),
        pytest.param(100.005, False, id="half_cent_itm"),
    ])
    def test_both_engines_put_the_exercise_boundary_at_one_cent_of_intrinsic(
        self, spot, exercised
    ):
        terms = [Terms(1, 100.0, is_call=True, expiry_day=3)]
        steps = [
            Step(1, 100.0, {1: 4.00}, groups=(((1, 1),),)),
            Step(2, 100.0, {1: 5.00}),
            Step(3, spot),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.shares == (100 if exercised else 0)
        assert harness.engine.metrics().exercise_count == int(exercised)

    def test_short_put_assigned_at_expiry_pays_the_strike_and_receives_shares(self):
        terms = [Terms(1, 100.0, is_call=False, expiry_day=5)]
        steps = [
            Step(1, 100.0, {1: 3.00}, groups=(((1, -1),),)),
            Step(2, 100.0, {1: 4.00}),
            Step(3, 97.0, {1: 5.50}),
            Step(4, 95.0, {1: 6.50}),
            Step(5, 94.0),
            Step(6, 94.0),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(INITIAL_CASH + 400.0 - 10_000.0)
        assert reference.shares == 100
        assert harness.engine.metrics().assignment_count == 1

    def test_short_call_assigned_at_expiry_delivers_shares_it_does_not_own(self):
        terms = [Terms(1, 100.0, is_call=True, expiry_day=5)]
        steps = [
            Step(1, 100.0, {1: 4.00}, groups=(((1, -1),),)),
            Step(2, 100.0, {1: 5.00}),
            Step(3, 104.0, {1: 6.50}),
            Step(4, 108.0, {1: 9.00}),
            Step(5, 110.0),
            Step(6, 111.0),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(INITIAL_CASH + 500.0 + 10_000.0)
        assert reference.shares == -100
        assert harness.finalize().final_equity_micros == micros(110_500.0 - 111.0 * 100)


class TestSpreadToExpiry:
    def test_debit_vertical_with_one_leg_itm_settles_each_leg_independently(self):
        terms = [Terms(1, 95.0, is_call=True, expiry_day=5),
                 Terms(2, 105.0, is_call=True, expiry_day=5)]
        steps = [
            Step(1, 100.0, {1: 6.50, 2: 2.40}, groups=(((1, 1), (2, -1)),)),
            Step(2, 100.0, {1: 7.00, 2: 2.00}),
            Step(3, 100.0, {1: 7.20, 2: 2.10}),
            Step(4, 100.0, {1: 6.00, 2: 1.50}),
            Step(5, 100.0),
            Step(6, 100.0),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(INITIAL_CASH - 700.0 + 200.0 - 9_500.0)
        assert reference.shares == 100
        metrics = harness.engine.metrics()
        assert (metrics.expiration_count, metrics.exercise_count) == (2, 1)


class TestRoundTrips:
    def test_a_closed_round_trip_keeps_only_the_difference_in_premium(self):
        terms = [Terms(1, 100.0, is_call=True, expiry_day=20)]
        steps = [
            Step(1, 100.0, {1: 4.00}, groups=(((1, 1),),)),
            Step(2, 100.0, {1: 5.00}),
            Step(3, 103.0, {1: 6.00}, groups=(((1, -1),),)),
            Step(4, 103.0, {1: 6.50}),
            Step(5, 104.0, {1: 7.00}),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(INITIAL_CASH - 500.0 + 650.0)
        assert reference.quantities() == {}
        assert harness.finalize().final_equity_micros == micros(100_150.0)

    def test_several_opens_and_closes_leave_the_surviving_position_only(self):
        terms = [Terms(1, 100.0, is_call=True, expiry_day=30),
                 Terms(2, 100.0, is_call=False, expiry_day=30)]
        steps = [
            Step(1, 100.0, {1: 4.50, 2: 3.50}, groups=(((1, 2),),)),
            Step(2, 100.0, {1: 5.00, 2: 4.00}, groups=(((2, -1),),)),
            Step(3, 101.0, {1: 5.50, 2: 4.50}, groups=(((1, -1),),)),
            Step(4, 102.0, {1: 6.00, 2: 5.00}, groups=(((2, 1),),)),
            Step(5, 103.0, {1: 6.50, 2: 5.50}, groups=(((1, 1),),)),
            Step(6, 104.0, {1: 7.00, 2: 6.00}),
        ]

        harness, reference = assert_engines_agree(terms, steps)

        assert reference.cash == micros(
            INITIAL_CASH - 1_000.0 + 450.0 + 600.0 - 550.0 - 700.0)
        assert reference.quantities() == {1: 2}
        assert harness.finalize().final_equity_micros == micros(98_800.0 + 1_400.0)


class TestMovingUnderlying:
    """Twelve bars on a rising tape, with an expiry in the middle of the run."""

    @staticmethod
    def tape() -> tuple[list[Terms], list[Step]]:
        terms = [Terms(1, 105.0, is_call=True, expiry_day=10),
                 Terms(2, 95.0, is_call=False, expiry_day=10)]
        steps = []
        for day in range(1, 13):
            spot = 100.0 + day
            prices = {} if day >= 10 else {
                1: round(2.0 + 0.1 * day + max(0.0, spot - 105.0), 2),
                2: round(2.0 + 0.05 * day + max(0.0, 95.0 - spot), 2),
            }
            groups = (((1, 1), (2, -1)),) if day == 1 else ()
            steps.append(Step(day, spot, prices, groups))
        return terms, steps

    def test_a_twelve_bar_run_with_a_mid_run_expiry_matches_bar_for_bar(self):
        _, reference = assert_engines_agree(*self.tape())

        assert reference.cash == micros(INITIAL_CASH - 220.0 + 210.0 - 10_500.0)
        assert reference.shares == 100
        assert reference.quantities() == {}

    def test_the_long_call_is_exercised_and_the_short_put_expires_worthless(self):
        harness, _ = assert_engines_agree(*self.tape())
        metrics = harness.engine.metrics()

        assert (metrics.expiration_count, metrics.exercise_count) == (2, 1)
        assert metrics.assignment_count == 0
        assert harness.finalize().final_equity_micros == micros(89_490.0 + 112.0 * 100)


def test_the_reference_implementation_names_no_engine_symbol():
    """A cross-check that reused the engine's own logic would prove nothing."""
    source = inspect.getsource(ReferenceEngine)

    assert "obt_engine" not in source
    assert "E." not in source
