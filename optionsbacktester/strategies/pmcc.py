"""
Poor man's covered call.

A long-dated call stands in for the 100 shares a covered call would need, and
shorter-dated calls are written against it repeatedly until the long leg
expires. The account never needs to own shares.

Two properties matter and both are enforced by the engine rather than here:

  - the short call is collateralized by the long call, not by stock, so the
    margin requirement is the debit paid rather than the naked requirement
  - a short call is only treated as covered while the long leg still outlives
    it; once the long is closer to expiry than the short, the position is
    genuinely naked and the broker model refuses it

The strategy re-writes the short as often as the schedule allows, which is the
whole point of the structure: income is collected many times over the life of a
single long call.
"""
from __future__ import annotations

from collections.abc import Sequence

import obt_engine as E

from ..strategy import Chain, Context, Strategy, buy, group, sell


class PoorMansCoveredCall(Strategy):
    name = "poor_mans_covered_call"

    def __init__(
        self,
        *,
        long_min_dte: float = 180.0,
        long_max_dte: float = 500.0,
        long_target_delta: float = 0.80,
        short_min_dte: float = 20.0,
        short_max_dte: float = 45.0,
        short_target_delta: float = 0.30,
        contracts: int = 1,
        # Roll the short once it is nearly worthless or nearly expired, which is
        # what frees capacity to write the next one.
        roll_at_dte: float = 5.0,
        roll_at_profit_fraction: float = 0.80,
        min_short_premium: float = 0.05,
    ):
        self.long_min_dte = long_min_dte
        self.long_max_dte = long_max_dte
        self.long_target_delta = long_target_delta
        self.short_min_dte = short_min_dte
        self.short_max_dte = short_max_dte
        self.short_target_delta = short_target_delta
        self.contracts = contracts
        self.roll_at_dte = roll_at_dte
        self.roll_at_profit_fraction = roll_at_profit_fraction
        self.min_short_premium = min_short_premium

        self.long_cv: int | None = None
        self.short_cv: int | None = None
        self.short_entry_credit: float = 0.0
        self.shorts_written: int = 0

    def on_market_snapshot(self, chain: Chain, context: Context) -> Sequence[E.OrderGroup]:
        calls = chain.calls()
        if len(calls) == 0:
            return ()

        self._forget_closed_legs(context)

        if self.long_cv is None:
            return self._open_long(calls)

        long_contract = context.contract(self.long_cv)
        if long_contract is None:
            return ()

        if self.short_cv is not None:
            close = self._maybe_close_short(calls, context)
            if close is not None:
                return (close,)
            return ()

        return self._write_short(calls, context, long_contract)

    # The engine closes an expired or fully-exited position, so the strategy
    # notices by finding no quantity rather than by being told.
    def _forget_closed_legs(self, context: Context) -> None:
        if self.long_cv is not None and context.quantity_in(self.long_cv) == 0:
            self.long_cv = None
        if self.short_cv is not None and context.quantity_in(self.short_cv) == 0:
            self.short_cv = None
            self.short_entry_credit = 0.0

    def _open_long(self, calls: Chain) -> Sequence[E.OrderGroup]:
        candidates = calls.expiring_in(self.long_min_dte, self.long_max_dte)
        if len(candidates) == 0:
            return ()
        pick = candidates.nearest_delta(self.long_target_delta)
        if pick is None or pick.mark <= 0:
            return ()
        self.long_cv = pick.contract_version_id
        return (group(buy(pick.contract_version_id, self.contracts, tag="pmcc_long")),)

    def _write_short(
        self, calls: Chain, context: Context, long_contract: E.OptionContractVersion
    ) -> Sequence[E.OrderGroup]:
        candidates = calls.expiring_in(self.short_min_dte, self.short_max_dte)
        if len(candidates) == 0:
            return ()

        eligible = [
            row for row in candidates
            # Strike above the long leg keeps the structure's risk capped, and
            # the long must outlive the short for it to count as coverage.
            if row.strike > long_contract.strike
            and row.mark >= self.min_short_premium
            and row.contract_version_id != self.long_cv
            and E.Timestamp.from_ns(long_contract.expiration).epoch_ns
            >= _expiration_ns(context, row.contract_version_id)
        ]
        if not eligible:
            return ()

        pick = min(eligible, key=lambda r: abs(r.delta - self.short_target_delta))
        self.short_cv = pick.contract_version_id
        self.short_entry_credit = pick.mark
        self.shorts_written += 1
        return (group(sell(pick.contract_version_id, self.contracts, tag="pmcc_short")),)

    def _maybe_close_short(self, calls: Chain, context: Context) -> E.OrderGroup | None:
        assert self.short_cv is not None
        row = next((r for r in calls if r.contract_version_id == self.short_cv), None)
        if row is None:
            return None

        near_expiry = row.dte <= self.roll_at_dte
        captured = (
            self.short_entry_credit > 0
            and (self.short_entry_credit - row.mark) / self.short_entry_credit
            >= self.roll_at_profit_fraction
        )
        if not (near_expiry or captured):
            return None

        return group(buy(self.short_cv, self.contracts, reduce_only=True, tag="pmcc_short_close"))


def _expiration_ns(context: Context, contract_version_id: int) -> int:
    contract = context.contract(contract_version_id)
    return contract.expiration if contract else 0
