"""Reading the processed store: the clock, the column lookup, and the memmap promise.

Every notebook, experiment and figure reaches the data through these five functions, so
a silent regression here (a dropped timezone, an off-by-one column, a matrix quietly
read into RAM) corrupts results everywhere else without failing anything else. The
store used below is synthetic and lives in ``tmp_path``: the suite must stay fast and
must keep working on a clean clone where ``data/processed/`` does not exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from milan_traffic import dataio
from milan_traffic.config import SLOT_MS, SLOTS_PER_DAY, TIMEZONE, square_to_rowcol

#: Deliberately not 1, 2, 3: an id that equals its column index cannot catch a lookup
#: that returns the id itself.
SQUARE_IDS = (11, 22, 33)
DAY_START = "2013-11-01 00:00"


@pytest.fixture
def store(tmp_path: Path):
    """A miniature processed store (one day, three squares) and its target matrix."""
    root = tmp_path / "processed"
    root.mkdir()

    rng = np.random.default_rng(0)
    internet = rng.uniform(1.0, 100.0, size=(SLOTS_PER_DAY, len(SQUARE_IDS))).astype(np.float32)
    # Skew two columns so the traffic ranking is 11 > 33 > 22. On a flat matrix a
    # "ranking" that just returned the first k columns, or the first k rows of the
    # totals table, would pass; here it cannot.
    internet[:, 0] *= 3.0
    internet[:, 2] *= 2.0
    np.save(root / "internet.npy", internet)

    origin = pd.Timestamp(DAY_START, tz=TIMEZONE).value // 1_000_000
    np.save(root / "timestamps.npy", origin + SLOT_MS * np.arange(SLOTS_PER_DAY, dtype=np.int64))
    np.save(root / "square_ids.npy", np.asarray(SQUARE_IDS, dtype=np.int32))

    totals = internet.astype(np.float64).sum(axis=0)
    rowcols = [square_to_rowcol(s) for s in SQUARE_IDS]
    pd.DataFrame(
        {
            "square_id": SQUARE_IDS,
            "row": [rc[0] for rc in rowcols],
            "col": [rc[1] for rc in rowcols],
            # Reversed, so sms_in ranks the squares the other way round and a ranking
            # that ignores its `activity` argument gives the wrong answer.
            "sms_in": totals[::-1],
            "internet": totals,
        }
    ).to_csv(root / "square_totals.csv", index=False)

    # load_index and load_square_ids are lru_cached on the directory string; clearing
    # around each test keeps one test's store from answering another's question.
    dataio.load_index.cache_clear()
    dataio.load_square_ids.cache_clear()
    yield root, internet
    dataio.load_index.cache_clear()
    dataio.load_square_ids.cache_clear()


# ------------------------------------------------------------------------ load_matrix


def test_load_matrix_returns_a_read_only_memmap(store):
    root, internet = store
    X = dataio.load_matrix("internet", processed_dir=root)

    assert isinstance(X, np.memmap)  # the whole point: no copy in RAM
    assert Path(X.filename).resolve() == (root / "internet.npy").resolve()
    assert X.shape == (SLOTS_PER_DAY, len(SQUARE_IDS))
    assert X.dtype == np.float32
    np.testing.assert_allclose(np.asarray(X), internet)

    with pytest.raises(ValueError):  # mode "r": an accidental write cannot corrupt it
        X[0, 0] = 1.0


def test_memmap_does_not_pull_the_file_into_resident_memory(tmp_path: Path):
    """The memory claim this project is partly graded on, measured rather than asserted.

    A 21 MiB matrix is opened both ways and the resident set is sampled around each. The
    memmap pages in only what is read; ``mmap=False`` brings the whole file into RAM.

    The slice read here is a block of rows, not a column. The matrix is C-ordered, so a
    column stride touches a byte on every page in the file: one cell's series is 36 KB
    of data spread across the whole mapping. That is fine on the real store (the pages
    are clean and the OS can evict them), but it is not what "reads a little" means, and
    a test that pretended otherwise would be measuring the page cache.
    """
    from milan_traffic.utils.memory import rss_mb

    root = tmp_path / "processed"
    root.mkdir()
    shape = (8928, 600)  # 21.4 MB of float32: 600 cells of the real (8928, 10000) matrix
    mm = np.lib.format.open_memmap(root / "internet.npy", mode="w+", dtype=np.float32, shape=shape)
    for i in range(0, shape[0], 1000):  # filled in chunks: building it must not need 21 MiB either
        mm[i : i + 1000] = 1.0
    mm.flush()
    del mm
    file_mb = (root / "internet.npy").stat().st_size / 2**20
    assert file_mb > 20.0

    before = rss_mb()
    X = dataio.load_matrix("internet", processed_dir=root)
    block = np.asarray(X[:64])  # 64 rows x 600 cells = 150 KB, contiguous on disk
    mapped_growth = rss_mb() - before
    assert block.sum() == pytest.approx(64 * shape[1])  # data really was read
    assert mapped_growth < 5.0, f"opening and sampling the memmap added {mapped_growth:.1f} MiB"

    before = rss_mb()
    full = dataio.load_matrix("internet", mmap=False, processed_dir=root)
    copied_growth = rss_mb() - before
    assert full.sum() == pytest.approx(shape[0] * shape[1])
    assert copied_growth > 10.0, f"a full load should cost ~{file_mb:.0f} MiB of RSS"
    assert copied_growth > 2 * mapped_growth


def test_load_matrix_can_be_read_into_ram_on_request(store):
    root, internet = store
    X = dataio.load_matrix("internet", mmap=False, processed_dir=root)

    assert not isinstance(X, np.memmap)
    np.testing.assert_allclose(X, internet)


def test_load_matrix_rejects_an_unknown_activity(store):
    root, _ = store
    with pytest.raises(ValueError, match="Unknown activity"):
        dataio.load_matrix("bandwidth", processed_dir=root)


def test_an_unmaterialised_channel_names_the_archives_it_needs(store):
    """sms_in is not written eagerly; without the daily archives, say so plainly."""
    root, _ = store
    with pytest.raises(FileNotFoundError, match="daily"):
        dataio.load_matrix("sms_in", processed_dir=root)


# ------------------------------------------------------------------------- load_index


def test_load_index_is_localised_strictly_increasing_and_evenly_spaced(store):
    root, _ = store
    index = dataio.load_index(str(root))

    assert isinstance(index, pd.DatetimeIndex)
    assert index.name == "time"
    assert index.tz is not None and str(index.tz) == TIMEZONE
    assert len(index) == SLOTS_PER_DAY
    assert index.is_monotonic_increasing and index.is_unique

    # Compared as Timedeltas rather than raw int64, which would silently depend on
    # whichever datetime resolution pandas happens to store.
    gaps = index.to_series().diff().dropna()
    assert set(gaps.unique()) == {pd.Timedelta(SLOT_MS, unit="ms")}
    assert index[0].strftime("%Y-%m-%d %H:%M") == DAY_START


def test_load_index_points_at_make_ingest_when_the_store_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="make ingest"):
        dataio.load_index(str(tmp_path / "nothing-here"))


# -------------------------------------------------------------------------- column_of


def test_column_of_maps_each_square_to_its_own_column(store):
    root, _ = store
    for expected_col, square_id in enumerate(SQUARE_IDS):
        assert dataio.column_of(square_id, root) == expected_col


def test_column_of_refuses_a_square_the_store_does_not_hold(store):
    root, _ = store
    with pytest.raises(KeyError):
        dataio.column_of(9999, root)


# ------------------------------------------------------------------------ load_series


def test_load_series_carries_the_column_the_index_and_a_name(store):
    root, internet = store
    s = dataio.load_series(22, processed_dir=root)

    assert isinstance(s, pd.Series)
    assert s.name == "internet_sq22"
    assert len(s) == SLOTS_PER_DAY
    np.testing.assert_allclose(s.to_numpy(), internet[:, 1])  # square 22 is column 1
    pd.testing.assert_index_equal(pd.DatetimeIndex(s.index), dataio.load_index(str(root)))


def test_load_series_slicing_is_inclusive_at_both_ends(store):
    root, internet = store
    s = dataio.load_series(11, processed_dir=root, start="2013-11-01 01:00", end="2013-11-01 01:50")

    assert len(s) == 6  # 01:00, 01:10, ..., 01:50
    assert s.index[0].strftime("%H:%M") == "01:00"
    assert s.index[-1].strftime("%H:%M") == "01:50"
    np.testing.assert_allclose(s.to_numpy(), internet[6:12, 0])


def test_load_series_accepts_an_open_ended_window(store):
    root, internet = store
    s = dataio.load_series(33, processed_dir=root, start="2013-11-01 23:00")

    assert len(s) == 6
    np.testing.assert_allclose(s.to_numpy(), internet[-6:, 2])


# ------------------------------------------------------------------------ top_squares


def test_top_squares_ranks_by_total_and_is_read_from_the_table(store):
    root, internet = store
    totals = internet.astype(np.float64).sum(axis=0)
    expected = [sq for _, sq in sorted(zip(totals, SQUARE_IDS, strict=True), reverse=True)]

    assert dataio.top_squares(3, processed_dir=root) == expected
    assert dataio.top_squares(2, processed_dir=root) == expected[:2]
    assert dataio.top_squares(1, processed_dir=root) == expected[:1]
    # The ranking is not the column order, so it cannot have come from the file layout.
    assert expected != list(SQUARE_IDS)


def test_top_squares_reflects_the_activity_it_was_asked_about(store):
    root, _ = store
    assert dataio.top_squares(1, processed_dir=root) != dataio.top_squares(
        1, activity="sms_in", processed_dir=root
    )


def test_top_squares_follows_the_table_it_is_given(store):
    """Change the totals on disk and the answer must change with them.

    This is the property the brief actually needs: the evaluation areas are whatever
    ``square_totals.csv`` says they are, so ids can never be carried over from memory or
    from a previous run of the ingestion.
    """
    root, _ = store
    df = pd.read_csv(root / "square_totals.csv")
    df["internet"] = [1.0, 500.0, 2.0]  # square 22 now dominates
    df.to_csv(root / "square_totals.csv", index=False)

    assert dataio.top_squares(2, processed_dir=root) == [22, 33]
