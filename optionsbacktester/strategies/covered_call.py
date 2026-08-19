"""Covered call: short a call against shares already held."""
from __future__ import annotations

from collections.abc import Sequence

import obt_engine as E

from ..strategy import Chain, Context, Strategy, buy, group, sell


class CoveredCall(Strategy):
    name = "covered_call"

    def __init__(self, *, min_dte: float = 20.0, max_dte: float = 45.0,
                 target_delta: float = 0.30, contracts: int = 1,
                 roll_at_dte: float = 2.0):
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.target_delta = target_delta
        self.contracts = contracts
        self.roll_at_dte = roll_at_dte
        self.short_cv: int | None = None

    def on_market_snapshot(self, chain: Chain, context: Context) -> Sequence[E.OrderGroup]:
        if self.short_cv is not None and context.quantity_in(self.short_cv) == 0:
            self.short_cv = None

        # One contract per 100 shares; without the shares this would be a naked
        # call, which the broker model refuses.
        shares = context.shares_of(context.contracts[next(iter(context.contracts))].underlying_symbol) \
            if context.contracts else 0
        if shares < 100 * self.contracts:
            return ()

        calls = chain.calls().expiring_in(self.min_dte, self.max_dte)
        if len(calls) == 0:
            return ()

        if self.short_cv is not None:
            row = next((r for r in calls if r.contract_version_id == self.short_cv), None)
            if row is not None and row.dte <= self.roll_at_dte:
                return (group(buy(self.short_cv, self.contracts, reduce_only=True)),)
            return ()

        pick = calls.nearest_delta(self.target_delta)
        if pick is None or pick.mark <= 0:
            return ()
        self.short_cv = pick.contract_version_id
        return (group(sell(pick.contract_version_id, self.contracts, tag="covered_call")),)
