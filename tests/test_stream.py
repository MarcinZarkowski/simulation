"""Discovery, pruning, projection, and quality gating over the pipeline's lake layout."""
from __future__ import annotations

import inspect
import math
from datetime import date

import polars as pl
import pytest

from optionsbacktester.contracts import (
    build_contracts,
    build_snapshot,
    contract_version_key,
)
from optionsbacktester.stream import (
    OPTION_COLUMNS,
    SUCCESS_MARKER,
    DataLake,
    UniverseFilter,
    iter_days,
    iter_timestamp_batches,
    load_day,
)
from tests import fixtures as F

TICKER = "TEST"
DERIVED_COLUMNS = {"dte", "moneyness", "delta_per_share"}

# A deliberately small lake: every assertion below is about layout or predicates,
# and a three-strike single-expiry tape exercises those as well as a full chain.
SMALL = {"strikes": (95.0, 100.0, 105.0), "expiry_offsets": (30,),
         "trading_days": 3, "bars_per_day": 1}


def small_lake(tmp_lake, **kw) -> DataLake:
    return DataLake(tmp_lake(**{**SMALL, **kw}), TICKER)


def only_day(lake: DataLake, universe: UniverseFilter | None = None):
    return load_day(lake.day_dirs()[0], TICKER, universe)


def day_options(lake: DataLake, universe: UniverseFilter | None = None) -> pl.DataFrame:
    return only_day(lake, universe).options


def symbols(frame: pl.DataFrame) -> set[str]:
    return set(frame["symbol"].to_list())


def flip_iv_failed(day_dir, symbol: str) -> None:
    """
    Mark one symbol's rows as failed-IV.

    ``LakeSpec`` has no injection field for ``iv_failed``, so the flag is set by
    rewriting the day's Parquet in place rather than by widening the generator.
    """
    path = day_dir / "options_enriched.parquet"
    pl.read_parquet(path).with_columns(
        pl.when(pl.col("symbol") == symbol).then(True)
        .otherwise(pl.col("iv_failed")).alias("iv_failed")
    ).write_parquet(path)


class TestDiscovery:
    def test_day_dirs_returns_every_written_day_in_chronological_order(self, tmp_lake):
        lake = small_lake(tmp_lake, trading_days=5)
        days = [DataLake.day_of(d) for d in lake.day_dirs()]

        assert days == sorted(days)
        assert days == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
                        date(2024, 1, 5), date(2024, 1, 8)]

    def test_complete_day_dirs_excludes_a_day_whose_success_marker_is_gone(self, tmp_lake):
        lake = small_lake(tmp_lake, trading_days=5)
        dropped = lake.day_dirs()[2]
        (dropped / SUCCESS_MARKER).unlink()

        assert dropped in lake.day_dirs()
        assert dropped not in lake.complete_day_dirs()
        assert len(lake.complete_day_dirs()) == 4

    def test_day_of_parses_the_partition_path_back_to_its_date(self, tmp_lake):
        lake = small_lake(tmp_lake)
        day_dir = lake.day_dirs()[0]

        assert DataLake.day_of(day_dir) == date(2024, 1, 2)
        assert day_dir.parts[-3:] == ("2024", "01", "02")

    def test_manifest_reads_the_success_marker_as_json(self, tmp_lake):
        lake = small_lake(tmp_lake)
        day_dir = lake.day_dirs()[0]
        manifest = DataLake.manifest(day_dir)

        assert manifest["status"] == "COMPLETE"
        assert manifest["pipeline_version"] == "2.0.0"
        assert manifest["row_counts"]["options_enriched.parquet"] == 6

    def test_manifest_of_an_unmarked_day_is_empty_rather_than_an_error(self, tmp_lake):
        lake = small_lake(tmp_lake)
        day_dir = lake.day_dirs()[0]
        (day_dir / SUCCESS_MARKER).unlink()

        assert DataLake.manifest(day_dir) == {}


class TestDateRangePruning:
    """The lake spans 2024-12-20 into 2025-01-09, so both boundaries are crossed."""

    @pytest.fixture
    def spanning(self, tmp_lake) -> DataLake:
        return small_lake(tmp_lake, start=date(2024, 12, 20), trading_days=15)

    def test_full_range_covers_both_the_month_and_the_year_boundary(self, spanning):
        days = [DataLake.day_of(d) for d in spanning.day_dirs()]

        assert days[0] == date(2024, 12, 20)
        assert days[-1] == date(2025, 1, 9)
        assert {d.year for d in days} == {2024, 2025}
        assert {d.month for d in days} == {12, 1}

    def test_a_range_inside_one_month_returns_only_that_month(self, spanning):
        days = [DataLake.day_of(d)
                for d in spanning.day_dirs(date(2024, 12, 23), date(2024, 12, 27))]

        assert days == [date(2024, 12, 23), date(2024, 12, 24), date(2024, 12, 25),
                        date(2024, 12, 26), date(2024, 12, 27)]

    def test_a_range_crossing_the_year_boundary_returns_days_from_both_years(self, spanning):
        days = [DataLake.day_of(d)
                for d in spanning.day_dirs(date(2024, 12, 30), date(2025, 1, 2))]

        assert days == [date(2024, 12, 30), date(2024, 12, 31),
                        date(2025, 1, 1), date(2025, 1, 2)]

    def test_out_of_range_days_are_never_returned(self, spanning):
        start, end = date(2024, 12, 26), date(2025, 1, 2)
        days = [DataLake.day_of(d) for d in spanning.day_dirs(start, end)]

        assert days
        assert all(start <= d <= end for d in days)
        assert date(2024, 12, 20) not in days
        assert date(2025, 1, 9) not in days

    def test_a_range_before_the_lake_starts_returns_nothing(self, spanning):
        assert spanning.day_dirs(date(2024, 1, 1), date(2024, 6, 30)) == []

    def test_a_range_after_the_lake_ends_returns_nothing(self, spanning):
        assert spanning.day_dirs(date(2025, 6, 1), date(2025, 12, 31)) == []


class TestColumnProjection:
    def test_load_day_returns_only_the_projected_columns_plus_the_derived_ones(self, tmp_lake):
        day = only_day(small_lake(tmp_lake))

        assert set(day.options.columns) == set(OPTION_COLUMNS) | DERIVED_COLUMNS
        assert len(day.options.columns) == len(OPTION_COLUMNS) + len(DERIVED_COLUMNS)
        assert len(day.options.columns) < len(F.OPTIONS_ENRICHED_SCHEMA)

    def test_unprojected_pipeline_columns_are_absent(self, tmp_lake):
        day = only_day(small_lake(tmp_lake))
        unprojected = set(F.OPTIONS_ENRICHED_SCHEMA) - set(OPTION_COLUMNS)

        assert {"T", "r", "iv_status", "pricing_model"} <= unprojected
        assert unprojected & set(day.options.columns) == set()

    def test_the_stock_frame_is_projected_to_the_underlying_columns(self, tmp_lake):
        day = only_day(small_lake(tmp_lake))

        assert day.stock.columns == ["timestamp", "underlying_open", "underlying_high",
                                     "underlying_low", "underlying_close", "underlying_volume"]


class TestDerivedColumns:
    @pytest.fixture
    def rows(self, tmp_lake) -> list[dict]:
        return only_day(small_lake(tmp_lake)).options.to_dicts()

    def test_dte_is_the_expiration_minus_the_bar_timestamp_in_days(self, rows):
        for row in rows:
            expected = (row["expiration"] - row["timestamp"]).total_seconds() / 86400.0
            assert row["dte"] == pytest.approx(expected, abs=1e-12)
        assert rows[0]["dte"] == pytest.approx(30.270833333333332)

    def test_moneyness_is_the_log_of_strike_over_underlying(self, rows):
        for row in rows:
            assert row["moneyness"] == pytest.approx(
                math.log(row["strike"] / row["underlying_price"]), abs=1e-12)
        assert {round(r["moneyness"], 10) for r in rows} == {
            round(math.log(k / 100.0), 10) for k in (95.0, 100.0, 105.0)}

    def test_delta_per_share_is_the_contract_delta_over_the_deliverable(self, rows):
        for row in rows:
            assert row["delta_per_share"] == pytest.approx(
                row["delta"] / row["deliverable_equity_amount"], abs=1e-12)
            assert abs(row["delta_per_share"]) <= 1.0
            assert abs(row["delta"]) > 1.0


class TestQualityGates:
    STALE = F.occ_symbol(TICKER, date(2024, 2, 1), "c", 100.0)
    FALLBACK = F.occ_symbol(TICKER, date(2024, 2, 1), "p", 100.0)
    UNPRICED = F.occ_symbol(TICKER, date(2024, 2, 1), "c", 105.0)

    @pytest.fixture
    def injected(self, tmp_lake) -> DataLake:
        return small_lake(
            tmp_lake,
            stale_symbols=(self.STALE,),
            fallback_iv_symbols=(self.FALLBACK,),
            unpriced_adjusted_symbols=(self.UNPRICED,),
        )

    @pytest.mark.parametrize("flag,symbol", [
        pytest.param("exclude_stale", STALE, id="stale_bar"),
        pytest.param("exclude_fallback_iv", FALLBACK, id="fallback_iv"),
        pytest.param("exclude_unpriced_adjusted", UNPRICED, id="unpriced_adjusted"),
    ])
    def test_each_gate_removes_only_its_own_bad_rows_and_keeps_them_when_off(
        self, injected, flag, symbol
    ):
        permissive = {"exclude_stale": False, "exclude_fallback_iv": False,
                      "exclude_unpriced_adjusted": False}

        gated = day_options(injected, UniverseFilter(**{**permissive, flag: True}))
        kept = day_options(injected, UniverseFilter(**permissive))

        assert symbol not in symbols(gated)
        assert symbol in symbols(kept)
        assert symbols(kept) - symbols(gated) == {symbol}

    def test_the_default_filter_removes_every_injected_bad_row(self, injected):
        gated = symbols(day_options(injected))

        assert gated.isdisjoint({self.STALE, self.FALLBACK, self.UNPRICED})
        assert len(gated) == 6 - 3

    def test_exclude_failed_iv_removes_a_row_whose_iv_solve_failed(self, injected):
        flip_iv_failed(injected.day_dirs()[0], self.STALE)
        permissive = {"exclude_stale": False, "exclude_fallback_iv": False,
                      "exclude_unpriced_adjusted": False}

        gated = day_options(injected, UniverseFilter(**permissive, exclude_failed_iv=True))
        kept = day_options(injected, UniverseFilter(**permissive, exclude_failed_iv=False))

        assert self.STALE not in symbols(gated)
        assert self.STALE in symbols(kept)

    def test_exclude_adjusted_is_off_by_default_and_drops_adjusted_rows_when_set(self, injected):
        default = day_options(injected, UniverseFilter(exclude_unpriced_adjusted=False))
        strict = day_options(injected, UniverseFilter(exclude_unpriced_adjusted=False,
                                                   exclude_adjusted=True))

        assert self.UNPRICED in symbols(default)
        assert self.UNPRICED not in symbols(strict)


class TestUniverseSelection:
    SELECTION = {"strikes": (80.0, 95.0, 100.0, 105.0, 120.0),
                 "expiry_offsets": (30, 60), "trading_days": 1, "bars_per_day": 1}

    @pytest.fixture
    def lake(self, tmp_lake) -> DataLake:
        return DataLake(tmp_lake(**self.SELECTION), TICKER)

    @pytest.fixture
    def unfiltered(self, lake) -> pl.DataFrame:
        return day_options(lake)

    def test_option_types_restricts_the_frame_to_the_requested_flags(self, lake, unfiltered):
        calls = day_options(lake, UniverseFilter(option_types=("c",)))

        assert set(calls["flag"].to_list()) == {"c"}
        assert calls.height == unfiltered.height // 2

    def test_dte_bounds_keep_only_the_expirations_inside_the_window(self, lake, unfiltered):
        near = day_options(lake, UniverseFilter(max_dte=40))
        far = day_options(lake, UniverseFilter(min_dte=40))

        assert near["dte"].max() < 40
        assert far["dte"].min() > 40
        assert near.height + far.height == unfiltered.height
        assert near["expiration"].n_unique() == 1

    def test_min_volume_drops_the_zero_volume_rows_the_pipeline_marked_stale(self, tmp_lake):
        quiet = F.occ_symbol(TICKER, date(2024, 2, 1), "c", 100.0)
        lake = small_lake(tmp_lake, stale_symbols=(quiet,))

        kept = day_options(lake, UniverseFilter(exclude_stale=False))
        gated = day_options(lake, UniverseFilter(exclude_stale=False, min_volume=1))

        assert kept.filter(pl.col("symbol") == quiet)["volume"].to_list() == [0]
        assert symbols(kept) - symbols(gated) == {quiet}
        assert gated["volume"].min() >= 1

    def test_min_volume_above_the_tape_volume_empties_the_frame(self, lake):
        assert day_options(lake, UniverseFilter(min_volume=501)).is_empty()
        assert not day_options(lake, UniverseFilter(min_volume=500)).is_empty()

    def test_abs_delta_bounds_select_on_per_share_delta(self, lake, unfiltered):
        selected = day_options(lake, UniverseFilter(min_abs_delta=0.2, max_abs_delta=0.6))
        expected = unfiltered.filter(pl.col("delta_per_share").abs().is_between(0.2, 0.6))

        assert symbols(selected) == symbols(expected)
        assert 0 < selected.height < unfiltered.height
        assert selected["delta_per_share"].abs().max() <= 0.6

    def test_moneyness_bounds_select_on_log_strike_over_underlying(self, lake, unfiltered):
        selected = day_options(lake, UniverseFilter(min_moneyness=-0.06, max_moneyness=0.06))

        assert set(selected["strike"].to_list()) == {95.0, 100.0, 105.0}
        assert selected.height < unfiltered.height

    def test_an_explicit_symbols_tuple_restricts_the_frame_to_those_contracts(self, lake):
        wanted = tuple(sorted(symbols(day_options(lake)))[:2])
        selected = day_options(lake, UniverseFilter(symbols=wanted))

        assert symbols(selected) == set(wanted)


class TestIterDays:
    def test_days_are_yielded_in_chronological_order(self, tmp_lake):
        lake = small_lake(tmp_lake, trading_days=5)
        days = [d.day for d in iter_days(lake)]

        assert days == sorted(days)
        assert len(days) == 5

    def test_a_day_directory_with_no_option_file_is_skipped(self, tmp_lake):
        lake = small_lake(tmp_lake, trading_days=3)
        empty = lake.ticker_dir / "2024" / "01" / "31"
        empty.mkdir(parents=True)
        (empty / SUCCESS_MARKER).write_text("{}")

        assert empty in lake.complete_day_dirs()
        assert date(2024, 1, 31) not in [d.day for d in iter_days(lake)]

    def test_a_day_filtered_down_to_nothing_is_skipped(self, tmp_lake):
        lake = small_lake(tmp_lake, trading_days=3)

        assert list(iter_days(lake, universe=UniverseFilter(min_volume=10_000))) == []

    def test_require_complete_false_includes_a_day_that_lacks_a_success_marker(self, tmp_lake):
        lake = small_lake(tmp_lake, trading_days=5)
        (lake.day_dirs()[1] / SUCCESS_MARKER).unlink()

        assert len(list(iter_days(lake))) == 4
        assert len(list(iter_days(lake, require_complete=False))) == 5

    def test_a_start_and_end_prune_the_iteration(self, tmp_lake):
        lake = small_lake(tmp_lake, trading_days=5)
        days = [d.day for d in iter_days(lake, date(2024, 1, 3), date(2024, 1, 5))]

        assert days == [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]


class TestTimestampBatches:
    @pytest.fixture
    def day(self, tmp_lake):
        return only_day(small_lake(tmp_lake, bars_per_day=4, trading_days=1))

    def test_there_is_exactly_one_batch_per_distinct_timestamp(self, day):
        batches = list(iter_timestamp_batches(day))

        assert len(batches) == 4
        assert len(batches) == day.options["timestamp"].n_unique()
        assert sum(b.height for _, b in batches) == day.options.height

    def test_batches_arrive_in_ascending_time_order(self, day):
        stamps = [ts for ts, _ in iter_timestamp_batches(day)]

        assert stamps == sorted(stamps)
        assert stamps == day.timestamps()

    def test_every_row_in_a_batch_carries_that_batch_timestamp(self, day):
        for ts, batch in iter_timestamp_batches(day):
            assert batch["timestamp"].unique().to_list() == [ts]
            assert batch.height == 6

    def test_an_empty_day_yields_no_batches(self, tmp_lake):
        empty = only_day(small_lake(tmp_lake), UniverseFilter(min_volume=10_000))

        assert list(iter_timestamp_batches(empty)) == []
        assert empty.timestamps() == []


class TestMissingInputs:
    def test_a_nonexistent_ticker_directory_returns_no_days_rather_than_raising(self, tmp_lake):
        root = tmp_lake(**SMALL)
        absent = DataLake(root, "NOSUCHTICKER")

        assert absent.day_dirs() == []
        assert absent.complete_day_dirs() == []
        assert list(iter_days(absent)) == []

    def test_a_nonexistent_root_returns_no_days(self, tmp_path):
        assert DataLake(tmp_path / "missing", TICKER).day_dirs() == []

    def test_a_day_without_an_options_file_loads_as_an_empty_slice(self, tmp_lake):
        lake = small_lake(tmp_lake)
        day_dir = lake.day_dirs()[0]
        (day_dir / "options_enriched.parquet").unlink()

        day = load_day(day_dir, TICKER)

        assert day.day == date(2024, 1, 2)
        assert day.ticker == TICKER
        assert day.options.is_empty()
        assert day.stock.is_empty()


class TestContractMapping:
    @pytest.fixture
    def day(self, tmp_lake):
        return only_day(small_lake(tmp_lake, bars_per_day=5, trading_days=1))

    def test_a_contract_quoted_across_many_bars_maps_to_one_version(self, day):
        contracts = build_contracts(day.options, TICKER)

        assert day.options.height == 30
        assert len(contracts) == 6
        assert len(contracts) == day.options["symbol"].n_unique()
        assert {c.symbol for c in contracts.values()} == symbols(day.options)

    def test_version_ids_are_the_keys_and_carry_the_pipeline_terms(self, day):
        contracts = build_contracts(day.options, TICKER)

        for key, contract in contracts.items():
            assert contract.id == key
            assert contract.underlying_symbol == TICKER
            assert contract.quote_multiplier == 100
            assert contract.deliverable_equity_microshares == 100_000_000
            assert contract.strike in (95.0, 100.0, 105.0)

    def test_contract_version_key_is_stable_for_identical_terms(self):
        first = contract_version_key("TEST240202C00100000", 100.0, 100.0, 100.0)
        second = contract_version_key("TEST240202C00100000", 100.0, 100.0, 100.0)

        assert first == second

    @pytest.mark.parametrize("strike,shares,multiplier", [
        pytest.param(101.0, 100.0, 100.0, id="different_strike"),
        pytest.param(100.0, 400.0, 100.0, id="different_deliverable"),
        pytest.param(100.0, 100.0, 10.0, id="different_multiplier"),
    ])
    def test_contract_version_key_differs_when_any_term_differs(self, strike, shares, multiplier):
        baseline = contract_version_key("TEST240202C00100000", 100.0, 100.0, 100.0)

        assert contract_version_key("TEST240202C00100000", strike, shares, multiplier) != baseline

    def test_build_snapshot_sets_the_underlying_price_and_one_bar_per_row(self, day):
        contracts = build_contracts(day.options, TICKER)
        timestamp, batch = next(iter_timestamp_batches(day))

        snapshot = build_snapshot(timestamp, batch, contracts, TICKER)

        assert snapshot.underlying_price == {TICKER: 100.0}
        assert len(snapshot.bars) == batch.height == 6
        assert len(snapshot.analytics) == batch.height
        assert {b.contract_version_id for b in snapshot.bars} == set(contracts)

    def test_an_unpriced_adjusted_contract_is_not_tradable_for_new_positions(self, tmp_lake):
        unpriced = F.occ_symbol(TICKER, date(2024, 2, 1), "c", 100.0)
        lake = small_lake(tmp_lake, unpriced_adjusted_symbols=(unpriced,))
        day = only_day(lake, UniverseFilter(exclude_unpriced_adjusted=False))

        by_symbol = {c.symbol: c for c in build_contracts(day.options, TICKER).values()}

        assert by_symbol[unpriced].tradable_for_new_positions is False
        assert by_symbol[unpriced].analytics_supported is False
        assert all(c.tradable_for_new_positions
                   for s, c in by_symbol.items() if s != unpriced)

    def test_one_symbol_with_two_deliverables_maps_to_two_versions(self, day):
        row = day.options.head(1)
        adjusted = row.with_columns(
            pl.lit(400.0).alias("deliverable_equity_amount"),
            pl.lit(400.0).alias("quote_multiplier"),
        )

        contracts = build_contracts(pl.concat([row, adjusted]), TICKER)

        assert len(contracts) == 2


def test_iter_days_is_a_generator_so_memory_stays_flat_over_a_long_run(tmp_lake):
    lake = small_lake(tmp_lake, trading_days=5)
    days = iter_days(lake)

    assert inspect.isgenerator(days)
    assert next(days).day == date(2024, 1, 2)
