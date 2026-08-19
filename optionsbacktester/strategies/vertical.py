"""Short put spread: sell a put, buy a lower-strike put at the same expiry."""
from __future__ import annotations

from collections.abc import Sequence

import obt_engine as E

from ..strategy import Chain, Context, Strategy, buy, group, sell


class ShortPutSpread(Strategy):
    name = "short_put_spread"

    def __init__(self, *, min_dte: float = 25.0, max_dte: float = 45.0,
                 short_delta: float = -0.30, width: float = 5.0, contracts: int = 1):
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.short_delta = short_delta
        self.width = width
        self.contracts = contracts
        self.legs: tuple[int, int] | None = None

    def on_market_snapshot(self, chain: Chain, context: Context) -> Sequence[E.OrderGroup]:
        if self.legs and all(context.quantity_in(cv) == 0 for cv in self.legs):
            self.legs = None
        if self.legs is not None:
            return ()

        puts = chain.puts().expiring_in(self.min_dte, self.max_dte)
        if len(puts) < 2:
            return ()
        short = puts.nearest_delta(self.short_delta)
        if short is None:
            return ()

        # The long leg must share the short's expiry, or this is a diagonal with
        # different risk than a vertical.
        same_expiry = puts.at_expiration(short.expiration)
        long = same_expiry.nearest_strike(short.strike - self.width)
        if long is None or long.contract_version_id == short.contract_version_id:
            return ()

        self.legs = (short.contract_version_id, long.contract_version_id)
        return (group(
            sell(short.contract_version_id, self.contracts, tag="short_put"),
            buy(long.contract_version_id, self.contracts, tag="long_put"),
        ),)
