"""
Trading the underlying directly.

Equity orders were refused outright -- "shares arrive only via settlement" -- so a
covered call, a collar, or a protective put could not be OPENED at all, only
inherited from an option that happened to settle into stock. Before that, the kind
flag was accepted and ignored, which priced one share with the contract's 100x
multiplier and booked it as an option position.
"""
from __future__ import annotations

import obt_engine as E
import pytest
from optionsbacktester.strategy import buy, buy_shares, group, sell, sell_shares

from tests.conftest import EngineHarness, base_config, day_ns, make_bar, make_contract

SYMBOL = "TEST"
SPOT = 100.0
STRIKE = 105.0
PREMIUM = 3.0
CASH = 100_000.0
CALL = 1
MICROS = 1_000_000


def dollars(amount: float) -> int:
    return round(amount * MICROS)


def equity_bar(day: int, *, price: float = SPOT, open_price: float | None = None,
               stale: bool = False, volume: int = 1_000_000) -> E.EquityBar:
    b = E.EquityBar()
    b.timestamp = day_ns(day)
    b.symbol = SYMBOL
    b.open = price if open_price is None else open_price
    b.high = b.low = b.close = b.vwap = price
    b.volume = volume
    b.trade_count = 5_000
    b.stale = stale
    return b


class EquityHarness(EngineHarness):
    """EngineHarness that also feeds a share bar on every bar."""

    def session(self, day: int, *, price: float = SPOT, open_price: float | None = None,
                option_price: float = PREMIUM, groups=None, stale: bool = False):
        snap = E.MarketSnapshot()
        snap.timestamp = day_ns(day)
        snap.bars = [make_bar(CALL, timestamp_ns=day_ns(day), price=option_price)]
        snap.underlying_price = {SYMBOL: price}
        snap.equity_bars = [equity_bar(day, price=price, open_price=open_price,
                                       stale=stale)]
        self.engine.begin_bar(snap)
        for g in groups or []:
            self.engine.submit_group(g)
        self.engine.end_bar()
        return self


def harness(*, cash: float = CASH, margin=E.MarginModel.REG_T, **cfg) -> EquityHarness:
    contract = make_contract(CALL, strike=STRIKE, expiry_day=60, underlying=SYMBOL)
    return EquityHarness(base_config(cash=cash, margin=margin, **cfg), [contract])


class TestBuyingAndSellingShares:
    def test_a_buy_fills_at_the_next_bar_open_and_costs_price_times_shares(self):
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2)

        assert h.shares_of(SYMBOL) == 100
        assert h.cash_micros == dollars(CASH - 100 * SPOT)

    def test_the_multiplier_is_one_not_one_hundred(self):
        """
        The original defect: one share priced with the contract's 100x multiplier
        moved $10,000 instead of $100.
        """
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 1))])
        h.session(2)

        assert h.cash_micros == dollars(CASH - SPOT)

    def test_no_option_position_is_created(self):
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2)

        assert h.positions() == []
        assert h.shares_of(SYMBOL) == 100

    def test_a_sale_realizes_exactly_the_price_difference(self):
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        # Bought at the day-2 open of $100, sold at the day-3 open of $110.
        h.session(2, price=SPOT, groups=[group(sell_shares(SYMBOL, 100))])
        h.session(3, price=110.0)

        assert h.shares_of(SYMBOL) == 0
        assert h.cash_micros == dollars(CASH - 100 * SPOT + 100 * 110.0)
        assert h.engine.account_state().realized_pnl == pytest.approx(1_000.0)
        assert h.engine.ledger_reconciles()

    def test_a_short_sale_covered_at_a_lower_price_realizes_a_gain(self):
        h = harness()
        h.session(1, groups=[group(sell_shares(SYMBOL, 100))])
        h.session(2, price=SPOT, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(3, price=90.0)

        assert h.shares_of(SYMBOL) == 0
        assert h.engine.account_state().realized_pnl == pytest.approx(1_000.0)

    def test_a_short_sale_opens_a_negative_position(self):
        h = harness()
        h.session(1, groups=[group(sell_shares(SYMBOL, 100))])
        h.session(2)

        assert h.shares_of(SYMBOL) == -100
        assert h.cash_micros == dollars(CASH + 100 * SPOT)

    def test_the_fill_price_comes_from_the_open_not_the_close(self):
        """Next-bar-open timing has to read the open, or the fill sees its own bar."""
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2, price=200.0, open_price=150.0)

        assert h.cash_micros == dollars(CASH - 100 * 150.0)

    def test_the_fill_is_recorded_as_equity(self):
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2)
        fills = h.fills()

        assert len(fills) == 1
        assert fills[0].kind == E.EquityKind.EQUITY
        assert fills[0].multiplier == 1
        assert fills[0].quantity == 100

    def test_a_ledger_entry_names_the_symbol(self):
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2)
        entries = [e for e in h.engine.ledger_entries()
                   if e.kind == E.LedgerEntryKind.EQUITY_TRADE]

        assert len(entries) == 1
        assert entries[0].memo == SYMBOL


class TestRefusals:
    def test_a_symbol_with_no_bar_is_refused(self):
        h = harness()
        h.session(1, groups=[group(buy_shares("NOSUCH", 100))])
        h.session(2)

        assert h.shares_of("NOSUCH") == 0
        assert h.rejections()[0].reason == E.RejectReason.NO_MARKET_DATA

    def test_a_stale_bar_is_refused_by_default(self):
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2, stale=True)

        assert h.shares_of(SYMBOL) == 0
        assert h.rejections()[0].reason == E.RejectReason.STALE_MARKET_DATA

    def test_a_limit_above_the_market_does_not_fill_a_sale(self):
        h = harness()
        h.session(1, groups=[group(sell_shares(SYMBOL, 100, limit_price=120.0))])
        h.session(2)

        assert h.shares_of(SYMBOL) == 0
        assert h.rejections()[0].reason == E.RejectReason.LIMIT_NOT_SATISFIED

    def test_a_limit_at_the_market_fills(self):
        h = harness()
        h.session(1, groups=[group(buy_shares(SYMBOL, 100, limit_price=120.0))])
        h.session(2)

        assert h.shares_of(SYMBOL) == 100

    def test_a_purchase_beyond_buying_power_is_refused(self):
        """
        $1,000 of cash gives $2,000 of Reg-T buying power, well short of $100,000 of
        stock.
        """
        h = harness(cash=1_000.0)
        h.session(1, groups=[group(buy_shares(SYMBOL, 1_000))])
        h.session(2)

        assert h.shares_of(SYMBOL) == 0
        assert h.rejections()[0].reason == E.RejectReason.INSUFFICIENT_BUYING_POWER

    def test_a_cash_account_cannot_sell_short(self):
        h = harness(margin=E.MarginModel.CASH_ACCOUNT)
        h.session(1, groups=[group(sell_shares(SYMBOL, 100))])
        h.session(2)

        assert h.shares_of(SYMBOL) == 0


class TestSpreadAndFees:
    def test_shares_cross_a_much_tighter_spread_than_options(self):
        """
        A share quote is a penny wide on a liquid name, which on a $100 stock is one
        basis point. Pricing shares through the option model would have charged a
        covered call more to buy its stock than to sell its call.
        """
        h = harness(spread=E.SpreadModelKind.CONDITIONAL_LOGNORMAL)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100), buy(CALL, 1))])
        h.session(2)
        equity, option = ([f for f in h.fills() if f.kind == E.EquityKind.EQUITY][0],
                          [f for f in h.fills() if f.kind == E.EquityKind.OPTION][0])

        # Per share, and against a $100 stock versus a $3.00 option.
        assert equity.half_spread <= 0.05
        assert equity.half_spread / SPOT < option.half_spread / PREMIUM

    def test_a_buy_crosses_up_and_a_sale_crosses_down(self):
        h = harness(spread=E.SpreadModelKind.CONDITIONAL_LOGNORMAL)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2)
        bought = h.fills()[0]
        assert bought.fill_price > bought.mark

        h2 = harness(spread=E.SpreadModelKind.CONDITIONAL_LOGNORMAL)
        h2.session(1, groups=[group(sell_shares(SYMBOL, 100))])
        h2.session(2)
        sold = h2.fills()[0]
        assert sold.fill_price < sold.mark

    def test_zero_spread_fills_exactly_at_the_mark(self):
        h = harness(spread=E.SpreadModelKind.ZERO)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2)

        assert h.fills()[0].half_spread == 0.0
        assert h.cash_micros == dollars(CASH - 100 * SPOT)

    def test_equity_fees_are_per_share_not_per_contract(self):
        schedule = E.FeeSchedule()
        # $0.000195 per share sold, with no per-contract regulatory pass-through.
        assert schedule.equity_fees(E.OrderSide.SELL, 1_000, 100_000.0) \
            == pytest.approx(0.20 + 2.06, abs=0.01)
        assert schedule.equity_fees(E.OrderSide.BUY, 1_000, 100_000.0) == 0.0

    def test_a_small_sale_is_exempt_from_both_ad_valorem_fees(self):
        """
        Section 31 is not passed through on a sale of $500 or less, and FINRA
        excludes a sale of 50 shares or fewer.
        """
        schedule = E.FeeSchedule()

        assert schedule.equity_fees(E.OrderSide.SELL, 5, 500.0) == 0.0


class TestCoveredCallFromShares:
    """
    The strategy family the refusal blocked entirely. A covered call needs shares
    first; before this, the only way to hold stock was to have an option settle
    into it.
    """

    def test_shares_bought_then_a_call_written_against_them(self):
        h = harness(margin=E.MarginModel.ROBINHOOD)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2, groups=[group(sell(CALL, 1))])
        h.session(3)

        assert h.shares_of(SYMBOL) == 100
        assert h.quantity_of(CALL) == -1
        assert h.cash_micros == dollars(CASH - 100 * SPOT + PREMIUM * 100)

    def test_the_short_call_is_covered_so_robinhood_permits_it(self):
        """Uncovered short calls are refused at Robinhood; this one is covered."""
        h = harness(margin=E.MarginModel.ROBINHOOD)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100))])
        h.session(2, groups=[group(sell(CALL, 1))])
        h.session(3)

        assert h.quantity_of(CALL) == -1
        assert not h.engine.account_state().margin_disallowed

    def test_writing_the_call_first_is_refused_because_it_is_uncovered(self):
        h = harness(margin=E.MarginModel.ROBINHOOD)
        h.session(1, groups=[group(sell(CALL, 1))])
        h.session(2)

        assert h.quantity_of(CALL) == 0

    def test_shares_and_the_call_can_be_opened_atomically(self):
        """
        A buy-write is one order at a broker, and submitting it as one group means
        the call is never momentarily uncovered.
        """
        h = harness(margin=E.MarginModel.ROBINHOOD)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100), sell(CALL, 1))])
        h.session(2)

        assert h.shares_of(SYMBOL) == 100
        assert h.quantity_of(CALL) == -1
        assert len(h.fills()) == 2

    def test_the_stock_carries_margin_and_the_covered_call_does_not(self):
        h = harness(margin=E.MarginModel.ROBINHOOD)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100), sell(CALL, 1))])
        h.session(2)

        # Reg-T 50% on the long stock; nothing extra for the covered call.
        assert h.engine.account_state().margin_requirement == pytest.approx(
            0.50 * SPOT * 100)

    def test_the_ledger_reconciles_across_the_whole_structure(self):
        h = harness(margin=E.MarginModel.ROBINHOOD, fees=True,
                    spread=E.SpreadModelKind.CONDITIONAL_LOGNORMAL)
        h.session(1, groups=[group(buy_shares(SYMBOL, 100), sell(CALL, 1))])
        h.session(2)
        h.session(3, price=110.0, groups=[group(sell_shares(SYMBOL, 100),
                                                buy(CALL, 1, reduce_only=True))])
        h.session(4, price=110.0)

        assert h.shares_of(SYMBOL) == 0
        assert h.quantity_of(CALL) == 0
        assert h.engine.ledger_reconciles()
