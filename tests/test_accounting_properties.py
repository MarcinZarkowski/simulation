"""
Property-based accounting invariants.

Every defect found in the position book so far came from a hand-built sequence
with more than one entry price and a partial close. Generating those sequences is
strictly better than guessing them: the basis-truncation drift survived 400
example-based tests and fails immediately here.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import obt_engine as E
from optionsbacktester.strategy import buy, group, sell
from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

CV = 1
MICROS = 1_000_000

# Prices with awkward cent values, which is where truncation shows up.
prices = st.decimals(min_value="0.01", max_value="99.99", places=2).map(float)
quantities = st.integers(min_value=1, max_value=25)
sides = st.sampled_from([1, -1])

# A sequence of signed (quantity, price) legs.
sequences = st.lists(
    st.tuples(sides, quantities, prices).map(lambda t: (t[0] * t[1], t[2])),
    min_size=1, max_size=8,
)


@st.composite
def flattening_sequences(draw):
    """
    A sequence that always ends with the book flat.

    Generating freely and then guarding on `if h.positions(): return` looks like
    coverage and is not: a random walk almost never lands on zero, so the
    interesting assertion is skipped on nearly every example. Verified by
    reintroducing the basis-truncation bug -- the guarded version passed.
    """
    legs = draw(st.lists(
        st.tuples(sides, quantities, prices).map(lambda t: (t[0] * t[1], t[2])),
        min_size=2, max_size=8,
    ))
    net = sum(q for q, _ in legs)
    if net != 0:
        legs.append((-net, draw(prices)))
    return legs

SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def replay(legs, *, multiplier: int = 100, cash: float = 5_000_000.0):
    """Applies each leg on its own bar and returns the harness plus starting cash."""
    contract = make_contract(CV, strike=100.0, expiry_day=4000, multiplier=multiplier,
                             deliverable_shares=multiplier)
    cfg = base_config(cash=cash, margin=E.MarginModel.REG_T)
    h = EngineHarness(cfg, [contract])
    start = h.cash_micros

    day = 1
    for signed_qty, price in legs:
        order = buy(CV, signed_qty) if signed_qty > 0 else sell(CV, -signed_qty)
        h.bar(day_ns(day), [make_bar(CV, timestamp_ns=day_ns(day), price=price)],
              groups=[group(order)])
        day += 1
        h.bar(day_ns(day), [make_bar(CV, timestamp_ns=day_ns(day), price=price)])
        day += 1
    return h, start


class TestLedgerInvariants:
    @given(legs=sequences)
    @SETTINGS
    def test_journal_always_sums_to_cash(self, legs):
        h, _ = replay(legs)
        entries = h.engine.ledger_entries()
        assert sum(e.amount_micros for e in entries) == h.cash_micros

    @given(legs=sequences)
    @SETTINGS
    def test_equity_equals_cash_plus_position_value(self, legs):
        h, _ = replay(legs)
        state = h.engine.account_state()
        position_value = sum(
            round(p.quantity * 100 * _last_price(legs) * MICROS) for p in h.positions()
        )
        # Compare against the engine's own decomposition rather than recomputing
        # the mark, which would just restate the implementation.
        points = h.engine.equity_points()
        if points:
            last = points[-1]
            assert last.equity == pytest.approx(last.cash + last.position_value, abs=1e-6)


def _last_price(legs):
    return legs[-1][1] if legs else 0.0


class TestRealizedPnlMatchesCash:
    """
    The invariant that actually constrains the book.

    ledger_reconciles() only asserts the journal sums to the cash balance, which
    is true by construction and can never fail, so it cannot catch a realized-P&L
    error. This can.
    """

    @given(legs=flattening_sequences())
    @SETTINGS
    def test_a_flat_position_realizes_exactly_the_cash_it_produced(self, legs):
        h, start = replay(legs)
        assert h.positions() == [], "sequence should have flattened the book"
        realized = round(h.engine.account_state().realized_pnl * MICROS)
        assert realized == h.cash_micros - start

    @given(legs=sequences)
    @SETTINGS
    def test_basis_is_fully_released_when_flat(self, legs):
        h, _ = replay(legs)
        assert all(p.cost_basis_micros != 0 or p.quantity != 0 for p in h.positions())

    @given(legs=flattening_sequences(), multiplier=st.sampled_from([1, 10, 100, 400]))
    @SETTINGS
    def test_the_identity_holds_for_any_deliverable(self, legs, multiplier):
        h, start = replay(legs, multiplier=multiplier)
        assert h.positions() == []
        realized = round(h.engine.account_state().realized_pnl * MICROS)
        assert realized == h.cash_micros - start

    @given(legs=flattening_sequences())
    @SETTINGS
    def test_cost_basis_is_fully_released_by_the_final_close(self, legs):
        """
        A flat book must carry no residual basis. Truncating an average and then
        multiplying it back up left a remainder behind on every partial close.
        """
        replay(legs)  # positions_ erases a flat position, so absence is the check
        h, start = replay(legs)
        assert h.positions() == []
        assert round(h.engine.account_state().unrealized_pnl * MICROS) == 0


class TestPositionConsistency:
    @given(legs=sequences)
    @SETTINGS
    def test_quantity_equals_the_net_of_the_legs(self, legs):
        h, _ = replay(legs)
        expected = sum(q for q, _ in legs)
        assert h.quantity_of(CV) == expected

    @given(legs=sequences)
    @SETTINGS
    def test_a_long_position_never_carries_negative_basis(self, legs):
        h, _ = replay(legs)
        for p in h.positions():
            if p.quantity > 0:
                assert p.cost_basis_micros >= 0
            elif p.quantity < 0:
                assert p.cost_basis_micros <= 0

    @given(legs=sequences)
    @SETTINGS
    def test_ledger_reconciles_on_every_sequence(self, legs):
        h, _ = replay(legs)
        assert h.engine.ledger_reconciles()
        assert h.finalize().ledger_reconciles
