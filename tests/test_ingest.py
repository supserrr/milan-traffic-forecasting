"""The reduction itself: does it produce the numbers it claims to?"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from milan_traffic.config import N_SQUARES, SLOT_MS, SLOTS_PER_DAY
from milan_traffic.ingest import date_from_path, day_start_ms, ingest, reduce_day


def test_date_is_parsed_from_the_filename():
    assert date_from_path(Path("sms-call-internet-mi-2013-12-25.txt")) == date(2013, 12, 25)


def test_day_origin_is_local_midnight():
    """2013-11-01 00:00 CET is 2013-10-31 23:00 UTC - the value seen in the raw files."""
    assert day_start_ms(date(2013, 11, 1)) == 1383260400000


def test_reduce_day_sums_over_country_codes(tiny_raw_file):
    path, expected = tiny_raw_file
    result = reduce_day(path, activities=("internet", "sms_in"), chunksize=500)

    assert result.arrays["internet"].shape == (SLOTS_PER_DAY, N_SQUARES)
    assert result.rows == 3 * SLOTS_PER_DAY * 2
    np.testing.assert_allclose(
        result.arrays["internet"][:, :3], expected.astype(np.float32), rtol=1e-5
    )
    # every square beyond the three present must be exactly zero
    assert result.arrays["internet"][:, 3:].sum() == 0.0


def test_reduce_day_treats_missing_fields_as_no_activity(tiny_raw_file):
    path, _ = tiny_raw_file
    result = reduce_day(path, activities=("sms_in",), chunksize=500)
    # half the rows carry sms_in=1.5, the rest are empty -> 1 contribution per (slot, square)
    assert np.isfinite(result.arrays["sms_in"]).all()
    assert result.arrays["sms_in"][:, :3].sum() == pytest.approx(1.5 * SLOTS_PER_DAY * 3)


def test_reduce_day_counts_observed_cells(tiny_raw_file):
    path, _ = tiny_raw_file
    result = reduce_day(path, activities=("internet",), chunksize=500)
    assert result.observed_cells == 3 * SLOTS_PER_DAY
    assert result.empty_cells == SLOTS_PER_DAY * (N_SQUARES - 3)


def test_chunksize_does_not_change_the_result(tiny_raw_file):
    path, _ = tiny_raw_file
    small = reduce_day(path, activities=("internet",), chunksize=97)
    large = reduce_day(path, activities=("internet",), chunksize=10_000_000)
    np.testing.assert_allclose(small.arrays["internet"], large.arrays["internet"], rtol=1e-6)


def test_bad_day_origin_is_detected(tmp_path):
    """A timestamp outside the file's own day must fail loudly, not fold into slot 0."""
    origin = day_start_ms(date(2013, 11, 1))
    bad = origin + 200 * SLOT_MS  # a slot on the following day
    path = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    path.write_text(f"1\t{bad}\t39\t\t\t\t\t1.0\n")
    with pytest.raises(ValueError, match="slot index out of range"):
        reduce_day(path, activities=("internet",))


def test_ingest_writes_a_complete_store(tiny_raw_file, tmp_path):
    path, _ = tiny_raw_file
    manifest = ingest(
        path.parent,
        tmp_path / "processed",
        activities=("internet",),
        target="internet",
        chunksize=500,
    )

    out = tmp_path / "processed"
    assert (out / "internet.npy").exists()
    assert (out / "timestamps.npy").exists()
    assert (out / "square_totals.csv").exists()
    assert (out / "daily" / "2013-11-01.npz").exists()

    matrix = np.load(out / "internet.npy")
    assert matrix.shape == (SLOTS_PER_DAY, N_SQUARES)

    ts = np.load(out / "timestamps.npy")
    assert np.all(np.diff(ts) == SLOT_MS)  # uniform, gap-free time axis
    assert manifest["n_slots"] == SLOTS_PER_DAY
    assert manifest["total_rows"] == 3 * SLOTS_PER_DAY * 2


@pytest.mark.slow
def test_real_processed_store_is_consistent():
    """Guards the actual store once it has been built. Skipped if it is absent."""
    from milan_traffic import dataio

    try:
        X = dataio.load_matrix("internet")
        index = dataio.load_index()
    except FileNotFoundError:
        pytest.skip("processed store not built yet - run `make ingest`")

    assert X.shape == (len(index), N_SQUARES)
    assert index.is_monotonic_increasing
    # Cast to a fixed unit: pandas >= 3.0 keeps datetime64[ms] here, 2.x promoted to ns,
    # so a bare .view("int64") means different things on different pandas versions.
    stamps_ms = index.values.astype("datetime64[ms]").astype("int64")
    assert (np.diff(stamps_ms) == SLOT_MS).all()
    assert np.isfinite(np.asarray(X[:, :50])).all()
