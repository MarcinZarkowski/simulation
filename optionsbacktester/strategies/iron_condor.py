"""Iron condor: short put spread plus short call spread at one expiry."""
from __future__ import annotations

from collections.abc import Sequence

import obt_engine as E

from ..strategy import Chain, Context, Strategy, buy, group, sell


class IronCondor(Strategy):
    name = "iron_condor"

    def __init__(self, *, min_dte: float = 25.0, max_dte: float = 45.0,
                 short_delta: float = 0.20, width: float = 5.0, contracts: int = 1):
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.short_delta = short_delta
        self.width = width
        self.contracts = contracts
        self.legs: tuple[int, ...] | None = None

    def on_market_snapshot(self, chain: Chain, context: Context) -> Sequence[E.OrderGroup]:
        if self.legs and all(context.quantity_in(cv) == 0 for cv in self.legs):
            self.legs = None
        if self.legs is not None:
            return ()

        window = chain.expiring_in(self.min_dte, self.max_dte)
        expirations = window.expirations()
        if not expirations:
            return ()

        # All four legs share one expiry, which is what makes the two wings net
        # against each other instead of behaving as two separate diagonals.
        for expiration in expirations:
            at = window.at_expiration(expiration)
            calls, puts = at.calls(), at.puts()
            short_call = calls.nearest_delta(self.short_delta)
            short_put = puts.nearest_delta(-self.short_delta)
            if short_call is None or short_put is None:
                continue
            long_call = calls.nearest_strike(short_call.strike + self.width)
            long_put = puts.nearest_strike(short_put.strike - self.width)
            if long_call is None or long_put is None:
                continue
            ids = {short_call.contract_version_id, long_call.contract_version_id,
                   short_put.contract_version_id, long_put.contract_version_id}
            if len(ids) < 4:
                continue

            self.legs = tuple(ids)
            return (group(
                sell(short_put.contract_version_id, self.contracts, tag="short_put"),
                buy(long_put.contract_version_id, self.contracts, tag="long_put"),
                sell(short_call.contract_version_id, self.contracts, tag="short_call"),
                buy(long_call.contract_version_id, self.contracts, tag="long_call"),
            ),)
        return ()
