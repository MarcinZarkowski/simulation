"""
Strategy API.

A strategy sees a point-in-time chain and returns declarative order groups. It
never mutates positions, cash, or fills: the engine owns all state transitions,
so a strategy cannot accidentally produce an unreachable portfolio.

The snapshot is built from data already published at the current timestamp, and
the engine fills on the following bar, so a strategy cannot act on a price it
would not have seen.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

import obt_engine as E

_group_counter = itertools.count(1)


def next_group_id() -> int:
    return next(_group_counter)


@dataclass(frozen=True)
class ChainRow:
    """
    One tradable contract as a strategy sees it.

    Greeks are per share, not per contract. The pipeline stores them scaled to a
    100-share contract, but strike selection is universally expressed per share
    -- a "30 delta call" means 0.30 -- so they are normalized here by the
    contract's actual deliverable. That also makes selection behave correctly for
    an adjusted contract whose deliverable is not 100 shares.
    """
    symbol: str
    contract_version_id: int
    flag: str
    strike: float
    expiration: datetime
    dte: float
    mark: float
    underlying_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    volume: int
    contract_delta: float = 0.0
    open_interest: int = 0
    moneyness: float = 0.0

    @property
    def is_call(self) -> bool:
        return self.flag == "c"

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


class Chain:
    """
    Queryable view of the contracts available at one instant.

    Selection helpers exist because picking a strike by delta or by moneyness is
    what strategies actually express; the previous engine only offered an
    integer offset from a fabricated at-the-money strike, which is not a
    quantity any real strategy is defined in terms of.
    """

    def __init__(self, rows: Sequence[ChainRow], underlying_price: float | None):
        self._rows = list(rows)
        self.underlying_price = underlying_price

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    @property
    def rows(self) -> list[ChainRow]:
        return list(self._rows)

    def calls(self) -> Chain:
        return Chain([r for r in self._rows if r.is_call], self.underlying_price)

    def puts(self) -> Chain:
        return Chain([r for r in self._rows if not r.is_call], self.underlying_price)

    def expiring_in(self, min_dte: float, max_dte: float) -> Chain:
        return Chain([r for r in self._rows if min_dte <= r.dte <= max_dte], self.underlying_price)

    def with_volume_at_least(self, volume: int) -> Chain:
        return Chain([r for r in self._rows if r.volume >= volume], self.underlying_price)

    def expirations(self) -> list[datetime]:
        return sorted({r.expiration for r in self._rows})

    def at_expiration(self, expiration: datetime) -> Chain:
        return Chain([r for r in self._rows if r.expiration == expiration], self.underlying_price)

    def nearest_delta(self, target: float) -> ChainRow | None:
        """Contract whose delta is closest to the target, comparing signed values."""
        if not self._rows:
            return None
        return min(self._rows, key=lambda r: abs(r.delta - target))

    def nearest_strike(self, strike: float) -> ChainRow | None:
        if not self._rows:
            return None
        return min(self._rows, key=lambda r: abs(r.strike - strike))

    def nearest_moneyness(self, target: float) -> ChainRow | None:
        """Target expressed as strike / underlying, so 1.05 is 5% out for a call."""
        if not self._rows or not self.underlying_price:
            return None
        want = target * self.underlying_price
        return self.nearest_strike(want)

    def find(self, symbol: str) -> ChainRow | None:
        for r in self._rows:
            if r.symbol == symbol:
                return r
        return None


@dataclass
class Context:
    """Account state and history a strategy may read."""
    timestamp: datetime
    account: E.AccountState
    positions: list[E.Position]
    equity_positions: list[E.EquityPosition]
    contracts: dict[int, E.OptionContractVersion]
    session_day: datetime
    scenario_id: int = 0
    user_state: dict = field(default_factory=dict)

    def position_in(self, contract_version_id: int) -> E.Position | None:
        for p in self.positions:
            if p.contract_version_id == contract_version_id:
                return p
        return None

    def quantity_in(self, contract_version_id: int) -> int:
        p = self.position_in(contract_version_id)
        return p.quantity if p else 0

    def open_option_count(self) -> int:
        return sum(1 for p in self.positions if p.kind == E.EquityKind.OPTION and p.quantity != 0)

    def shares_of(self, symbol: str) -> int:
        for e in self.equity_positions:
            if e.symbol == symbol:
                return e.shares
        return 0

    def contract(self, contract_version_id: int) -> E.OptionContractVersion | None:
        return self.contracts.get(contract_version_id)

    def symbol_of(self, contract_version_id: int) -> str | None:
        c = self.contract(contract_version_id)
        return c.symbol if c else None


def leg(
    contract_version_id: int,
    side: E.OrderSide,
    quantity: int,
    *,
    limit_price: float | None = None,
    reduce_only: bool = False,
    tag: str = "",
) -> E.Order:
    """One leg of an order group."""
    o = E.Order()
    o.contract_version_id = contract_version_id
    o.side = side
    o.quantity = quantity
    o.type = E.OrderType.LIMIT if limit_price is not None else E.OrderType.MARKET
    if limit_price is not None:
        o.limit_price = limit_price
    o.reduce_only = reduce_only
    o.tag = tag
    return o


def buy(cv: int, qty: int = 1, **kw) -> E.Order:
    return leg(cv, E.OrderSide.BUY, qty, **kw)


def sell(cv: int, qty: int = 1, **kw) -> E.Order:
    return leg(cv, E.OrderSide.SELL, qty, **kw)


def shares(
    symbol: str,
    side: E.OrderSide,
    quantity: int,
    *,
    limit_price: float | None = None,
    reduce_only: bool = False,
    tag: str = "",
) -> E.Order:
    """
    An equity leg. Identifies its instrument by symbol, not by contract version.

    Shares used to arrive only via assignment, so a covered call or a collar could
    not be opened at all -- only inherited from an option that settled.
    """
    o = E.Order()
    o.kind = E.EquityKind.EQUITY
    o.symbol = symbol
    o.side = side
    o.quantity = quantity
    o.type = E.OrderType.LIMIT if limit_price is not None else E.OrderType.MARKET
    if limit_price is not None:
        o.limit_price = limit_price
    o.reduce_only = reduce_only
    o.tag = tag
    return o


def buy_shares(symbol: str, quantity: int, **kw) -> E.Order:
    return shares(symbol, E.OrderSide.BUY, quantity, **kw)


def sell_shares(symbol: str, quantity: int, **kw) -> E.Order:
    return shares(symbol, E.OrderSide.SELL, quantity, **kw)


def group(*legs: E.Order, group_id: int | None = None) -> E.OrderGroup:
    """
    Bundle legs into one atomic order.

    All legs fill or none do. A broker does not partially execute a spread, and
    a half-filled vertical is a different position with different risk.
    """
    g = E.OrderGroup()
    g.group_id = group_id if group_id is not None else next_group_id()
    g.legs = list(legs)
    g.atomic = True
    return g


class Strategy:
    """
    Base class. Override the callbacks that matter; the rest are no-ops.

    ``on_market_snapshot`` returns order groups rather than performing trades,
    which is what keeps the engine authoritative over fills and state.
    """

    name: str = "strategy"

    def on_session_start(self, context: Context) -> None:
        pass

    def on_market_snapshot(self, chain: Chain, context: Context) -> Sequence[E.OrderGroup]:
        return ()

    def on_corporate_action(self, event: E.CorporateActionTransition, context: Context) -> None:
        pass

    def on_fill(self, fill: E.Fill, context: Context) -> None:
        pass

    def on_session_end(self, context: Context) -> None:
        pass


def chain_from_batch(
    batch: pl.DataFrame,
    contracts: dict[int, E.OptionContractVersion],
    key_of,
) -> Chain:
    """
    Build a Chain from one timestamp batch.

    Only contracts the pipeline priced are included, so a strategy cannot select
    a contract whose Greeks are absent and then act on a zero delta.
    """
    rows: list[ChainRow] = []
    underlying: float | None = None
    for row in batch.iter_rows(named=True):
        key = key_of(row)
        contract = contracts.get(key)
        if contract is None or not contract.analytics_supported:
            continue
        price = row.get("underlying_price")
        if price is not None:
            underlying = float(price)
        mark = row.get("valuation_price") or row.get("close") or 0.0
        # Divide by the real deliverable so per-share Greeks stay correct even
        # when a contract does not deliver 100 shares.
        shares = float(row.get("deliverable_equity_amount") or 100.0) or 100.0
        contract_delta = float(row.get("delta") or 0.0)
        rows.append(ChainRow(
            symbol=row["symbol"],
            contract_version_id=key,
            flag=row["flag"],
            strike=float(row["strike"]),
            expiration=row["expiration"],
            dte=float(row.get("dte") or 0.0),
            mark=float(mark),
            underlying_price=float(price) if price is not None else 0.0,
            delta=contract_delta / shares,
            gamma=float(row.get("gamma") or 0.0) / shares,
            theta=float(row.get("theta") or 0.0) / shares,
            vega=float(row.get("vega") or 0.0) / shares,
            implied_volatility=float(row.get("smoothed_iv") or 0.0),
            volume=int(row.get("volume") or 0),
            contract_delta=contract_delta,
            moneyness=float(row.get("moneyness") or 0.0),
        ))
    return Chain(rows, underlying)
