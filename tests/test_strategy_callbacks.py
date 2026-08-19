"""
The Strategy callbacks, driven through the real runner.

``on_fill``, ``on_corporate_action`` and ``on_session_start`` were declared in the
Strategy API and never invoked, so a strategy written against the documented
interface silently never ran that logic. ``on_session_start`` was worse than absent
for a while: it fired once for the entire run while ``on_session_end`` fired daily,
so per-session state was reset on a schedule that matched nothing.
"""
from __future__ import annotations

from datetime import date, datetime

import obt_engine as E
import pytest
from optionsbacktester import runner
from optionsbacktester.strategy import Strategy, buy, group, sell

from tests import fixtures as F

TICKER = "TEST"


class Recorder(Strategy):
    """Records every callback it receives, and buys one call on the first bar."""

    name = "recorder"

    def __init__(self, *, trade: bool = True):
        self.trade = trade
        self.sessions_started: list[date] = []
        self.sessions_ended: list[date] = []
        self.fills: list = []
        self.actions: list = []
        self.snapshots = 0
        self.bought = False

    def on_session_start(self, context):
        self.sessions_started.append(context.session_day)

    def on_session_end(self, context):
        self.sessions_ended.append(context.session_day)

    def on_fill(self, fill, context):
        self.fills.append(fill)

    def on_corporate_action(self, event, context):
        self.actions.append(event)

    def on_market_snapshot(self, chain, context):
        self.snapshots += 1
        if not self.trade or self.bought:
            return ()
        pick = chain.calls().expiring_in(10, 70).nearest_delta(0.40)
        if pick is None:
            return ()
        self.bought = True
        return [group(buy(pick.contract_version_id, 1))]


def run(root, *, factory=Recorder, **cfg_overrides) -> runner.RunResult:
    cfg = E.BacktestConfig()
    cfg.initial_cash = 100_000.0
    for name, value in cfg_overrides.items():
        setattr(cfg, name, value)
    return runner.run(factory, data_root=root, ticker=TICKER, config=cfg)


class TestSessionCallbacks:
    def test_a_session_starts_and_ends_once_per_trading_day(self, tmp_path):
        """
        on_session_start fired once for the entire run while on_session_end fired
        daily, so they did not pair.
        """
        F.write_lake(tmp_path, F.LakeSpec(trading_days=4, bars_per_day=3))
        strategies: list[Recorder] = []

        def factory():
            s = Recorder()
            strategies.append(s)
            return s

        run(tmp_path, factory=factory)
        first = strategies[0]

        assert len(first.sessions_started) == 4
        assert first.sessions_started == first.sessions_ended

    def test_every_bar_reaches_on_market_snapshot(self, tmp_path):
        F.write_lake(tmp_path, F.LakeSpec(trading_days=3, bars_per_day=5))
        strategies: list[Recorder] = []

        def factory():
            s = Recorder(trade=False)
            strategies.append(s)
            return s

        run(tmp_path, factory=factory)

        assert strategies[0].snapshots == 15


class TestFillCallback:
    def test_a_fill_is_delivered_exactly_once(self, tmp_path):
        F.write_lake(tmp_path, F.LakeSpec(trading_days=3, bars_per_day=4))
        strategies: list[Recorder] = []

        def factory():
            s = Recorder()
            strategies.append(s)
            return s

        result = run(tmp_path, factory=factory)
        recorder = strategies[0]

        assert len(recorder.fills) == 1
        assert result.paths[0].fill_count == 1
        assert recorder.fills[0].quantity == 1

    def test_the_fill_carries_a_price_and_a_signed_cash_flow(self, tmp_path):
        F.write_lake(tmp_path, F.LakeSpec(trading_days=3, bars_per_day=4))
        strategies: list[Recorder] = []

        def factory():
            s = Recorder()
            strategies.append(s)
            return s

        run(tmp_path, factory=factory)
        fill = strategies[0].fills[0]

        assert fill.fill_price > 0.0
        assert fill.gross_cash < 0.0      # a purchase costs money

    def test_each_monte_carlo_path_gets_its_own_strategy_and_its_own_fills(self, tmp_path):
        """
        A shared strategy instance would leak state across paths, and the fills a
        path saw would be another path's.
        """
        F.write_lake(tmp_path, F.LakeSpec(trading_days=3, bars_per_day=4))
        strategies: list[Recorder] = []

        def factory():
            s = Recorder()
            strategies.append(s)
            return s

        run(tmp_path, factory=factory, spread_mc_paths=3)

        # Three paths plus the zero-spread reference engine.
        assert len(strategies) == 4
        assert all(len(s.fills) == 1 for s in strategies)


def lineage_row(*, parent: str, child: str, effective: datetime,
                confirmed: bool, parent_contracts: int = 1,
                child_contracts: int = 1) -> dict:
    return {
        "lineage_event_id": "evt-1",
        "effective_at": effective,
        "source_available_at": effective,
        "parent_symbol": parent,
        "child_symbol": child,
        "parent_contracts": parent_contracts,
        "child_contracts": child_contracts,
        "occ_confirmed": confirmed,
    }


class TestCorporateActionCallback:
    """
    A strategy that is not told its position was converted keeps referencing a
    version the engine has already superseded, and every later order on it is
    rejected for reasons the strategy cannot see.
    """

    def _lake(self, tmp_path, *, confirmed: bool):
        spec = F.LakeSpec(trading_days=5, bars_per_day=4)
        symbols = sorted({
            F.occ_symbol(TICKER, exp, flag, strike)
            for exp in [spec.start] for flag in ("c",) for strike in spec.strikes
        })
        # The parent is whatever the strategy will actually buy; use every listed
        # call so the event matches regardless of which strike is selected.
        first_day = F._trading_days(spec.start, spec.trading_days)[0]
        options, _ = F.build_day_frames(spec, 0, first_day)
        parent = options["symbol"][0]
        spec.lineage_events = [lineage_row(
            parent=parent, child=parent, effective=datetime(2024, 1, 4),
            confirmed=confirmed,
        )]
        F.write_lake(tmp_path, spec)
        return parent

    def test_an_applied_adjustment_reaches_the_strategy(self, tmp_path):
        parent = self._lake(tmp_path, confirmed=True)
        strategies: list[Recorder] = []

        class BuysTheParent(Recorder):
            def on_market_snapshot(self, chain, context):
                self.snapshots += 1
                if self.bought:
                    return ()
                row = chain.find(parent)
                if row is None:
                    return ()
                self.bought = True
                return [group(buy(row.contract_version_id, 1))]

        def factory():
            s = BuysTheParent()
            strategies.append(s)
            return s

        run(tmp_path, factory=factory)

        assert strategies[0].actions, "on_corporate_action was never invoked"
        assert strategies[0].actions[0].occ_confirmed

    def test_an_unconfirmed_adjustment_also_reaches_the_strategy(self, tmp_path):
        """
        The position is quarantined rather than converted, which the strategy needs
        to know even more urgently than a clean conversion.
        """
        parent = self._lake(tmp_path, confirmed=False)
        strategies: list[Recorder] = []

        class BuysTheParent(Recorder):
            def on_market_snapshot(self, chain, context):
                self.snapshots += 1
                if self.bought:
                    return ()
                row = chain.find(parent)
                if row is None:
                    return ()
                self.bought = True
                return [group(buy(row.contract_version_id, 1))]

        def factory():
            s = BuysTheParent()
            strategies.append(s)
            return s

        result = run(tmp_path, factory=factory)

        assert strategies[0].actions
        assert not strategies[0].actions[0].occ_confirmed
        assert result.paths[0].truncated

    def test_no_adjustment_means_no_callback(self, tmp_path):
        F.write_lake(tmp_path, F.LakeSpec(trading_days=4, bars_per_day=3))
        strategies: list[Recorder] = []

        def factory():
            s = Recorder()
            strategies.append(s)
            return s

        run(tmp_path, factory=factory)

        assert strategies[0].actions == []

    def test_the_callback_fires_before_the_strategy_is_asked_for_orders(self, tmp_path):
        """
        Adjustments are applied at the start of the bar, so a strategy has to hear
        about one before it can reference a version the engine has superseded.
        """
        parent = self._lake(tmp_path, confirmed=True)
        order: list[str] = []

        class Ordered(Recorder):
            def on_corporate_action(self, event, context):
                order.append("action")

            def on_market_snapshot(self, chain, context):
                order.append("snapshot")
                if self.bought:
                    return ()
                row = chain.find(parent)
                if row is None:
                    return ()
                self.bought = True
                return [group(buy(row.contract_version_id, 1))]

        run(tmp_path, factory=Ordered)

        assert "action" in order
        # Every action is immediately followed by that bar's snapshot call.
        assert order[order.index("action") + 1] == "snapshot"
