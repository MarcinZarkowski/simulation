"""
Two throughput changes, and the correctness they must not cost.

``build_bars``, ``build_analytics`` and ``chain_from_batch`` each iterated the same
frame independently, so every bar was materialized into Python dicts three times and
the contract-version key recomputed three times per row. The runner now takes one
pass; the separate builders remain for callers that want one of them.

Reading a Parquet file's schema costs about 0.8 ms and happened twice per trading
day, which was 15% of a run's wall clock spent re-establishing that today's columns
match yesterday's. It is cached per lake and per file name, with a fallback that
re-reads on a stale hit -- so the cache is a performance assumption and not a
correctness one, and this file proves that.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from optionsbacktester.contracts import (
    build_analytics,
    build_bar_view,
    build_bars,
    build_snapshot,
    contract_version_key,
)
from optionsbacktester.strategy import chain_from_batch
from optionsbacktester.stream import (
    _SCHEMA_CACHE,
    OPTION_COLUMNS,
    DataLake,
    iter_days,
    iter_timestamp_batches,
    load_day,
)

from tests import fixtures as F

TICKER = "TEST"


def _row_key(row: dict) -> int:
    return contract_version_key(
        row["symbol"],
        float(row["strike"]),
        float(row.get("deliverable_equity_amount") or 100.0),
        float(row.get("quote_multiplier") or 100.0),
    )


@pytest.fixture
def day_and_contracts(tmp_path):
    from optionsbacktester.contracts import build_contracts

    F.write_lake(tmp_path, F.LakeSpec(trading_days=1, bars_per_day=4))
    day = next(iter_days(DataLake(tmp_path, TICKER)))
    return day, build_contracts(day.options, TICKER)


class TestTheSinglePassAgreesWithTheSeparateBuilders:
    """
    The whole point of merging them: the same numbers, from one pass instead of
    three.
    """

    def test_bars_are_identical(self, day_and_contracts):
        day, contracts = day_and_contracts
        for ts, batch in iter_timestamp_batches(day):
            snap, _ = build_bar_view(ts, batch, contracts, TICKER)
            separate = build_bars(batch, contracts)

            assert len(snap.bars) == len(separate)
            for merged, alone in zip(snap.bars, separate, strict=True):
                assert merged.contract_version_id == alone.contract_version_id
                assert merged.open == alone.open
                assert merged.close == alone.close
                assert merged.valuation_price == alone.valuation_price
                assert merged.volume == alone.volume
                assert merged.stale == alone.stale
                assert merged.analytics_valid == alone.analytics_valid

    def test_analytics_are_identical(self, day_and_contracts):
        day, contracts = day_and_contracts
        for ts, batch in iter_timestamp_batches(day):
            snap, _ = build_bar_view(ts, batch, contracts, TICKER)
            separate = build_analytics(batch, contracts)

            assert len(snap.analytics) == len(separate)
            for merged, alone in zip(snap.analytics, separate, strict=True):
                assert merged.contract_version_id == alone.contract_version_id
                assert merged.delta == alone.delta
                assert merged.implied_volatility == alone.implied_volatility
                assert merged.valid == alone.valid

    def test_the_chain_is_identical(self, day_and_contracts):
        day, contracts = day_and_contracts
        for ts, batch in iter_timestamp_batches(day):
            _, merged = build_bar_view(ts, batch, contracts, TICKER)
            separate = chain_from_batch(batch, contracts, _row_key)

            assert len(merged) == len(separate)
            assert merged.underlying_price == separate.underlying_price
            for a, b in zip(merged.rows, separate.rows, strict=True):
                assert a == b

    def test_the_underlying_price_matches_build_snapshot(self, day_and_contracts):
        day, contracts = day_and_contracts
        for ts, batch in iter_timestamp_batches(day):
            snap, _ = build_bar_view(ts, batch, contracts, TICKER)
            separate = build_snapshot(ts, batch, contracts, TICKER)

            assert snap.underlying_price == separate.underlying_price
            assert snap.timestamp == separate.timestamp

    def test_equity_bars_are_carried_through(self, day_and_contracts):
        day, contracts = day_and_contracts
        ts, batch = next(iter_timestamp_batches(day))
        stock = day.stock.filter(pl.col("timestamp") == ts)

        snap, _ = build_bar_view(ts, batch, contracts, TICKER, stock)

        assert len(snap.equity_bars) == stock.height
        assert snap.equity_bars[0].symbol == TICKER

    def test_a_contract_missing_from_the_registry_is_skipped_in_both(self, day_and_contracts):
        day, contracts = day_and_contracts
        ts, batch = next(iter_timestamp_batches(day))
        partial = dict(list(contracts.items())[:2])

        snap, chain = build_bar_view(ts, batch, contracts=partial, underlying_symbol=TICKER)

        assert len(snap.bars) == 2
        assert len(chain) == 2


class TestTheSchemaCacheIsNotACorrectnessAssumption:
    def test_a_second_day_does_not_reread_the_schema(self, tmp_path, monkeypatch):
        F.write_lake(tmp_path, F.LakeSpec(trading_days=3, bars_per_day=2))
        _SCHEMA_CACHE.clear()
        reads = {"n": 0}
        real = pl.LazyFrame.collect_schema

        def counted(self):
            reads["n"] += 1
            return real(self)

        monkeypatch.setattr(pl.LazyFrame, "collect_schema", counted)
        list(iter_days(DataLake(tmp_path, TICKER)))

        # Two files projected per day -- options and stock -- read once each, not
        # once per day.
        assert reads["n"] == 2

    def test_a_column_set_that_shrinks_mid_history_still_loads(self, tmp_path):
        """
        The fallback. A day written by an older pipeline version has fewer columns
        than the cached set, and a select against the stale set raises -- so the entry
        is re-derived for that file rather than the read failing.
        """
        F.write_lake(tmp_path, F.LakeSpec(trading_days=2, bars_per_day=2))
        lake = DataLake(tmp_path, TICKER)
        days = sorted(lake.day_dirs())
        _SCHEMA_CACHE.clear()

        # Warm the cache on the full schema.
        first = load_day(days[0], TICKER)
        assert not first.options.is_empty()

        # Rewrite the second day without a column the cache believes exists.
        path = days[1] / "options_enriched.parquet"
        reduced = pl.read_parquet(path).drop("rho")
        reduced.write_parquet(path)

        second = load_day(days[1], TICKER)

        assert not second.options.is_empty()
        assert "rho" not in second.options.columns

    def test_a_column_set_that_grows_is_projected_to_what_is_asked_for(self, tmp_path):
        """
        A later day with EXTRA columns is not a problem: the projection asks for a
        fixed list, so anything new is simply not selected.
        """
        F.write_lake(tmp_path, F.LakeSpec(trading_days=2, bars_per_day=2))
        lake = DataLake(tmp_path, TICKER)
        days = sorted(lake.day_dirs())
        _SCHEMA_CACHE.clear()
        load_day(days[0], TICKER)

        path = days[1] / "options_enriched.parquet"
        grown = pl.read_parquet(path).with_columns(pl.lit(1.0).alias("brand_new"))
        grown.write_parquet(path)

        second = load_day(days[1], TICKER)

        assert "brand_new" not in second.options.columns
        assert set(second.options.columns) & set(OPTION_COLUMNS)

    def test_two_lakes_do_not_share_a_cache_entry(self, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        F.write_lake(one, F.LakeSpec(trading_days=1, bars_per_day=2))
        F.write_lake(two, F.LakeSpec(trading_days=1, bars_per_day=2))
        _SCHEMA_CACHE.clear()

        list(iter_days(DataLake(one, TICKER)))
        list(iter_days(DataLake(two, TICKER)))

        lakes = {key[0] for key in _SCHEMA_CACHE}
        assert len(lakes) == 2
