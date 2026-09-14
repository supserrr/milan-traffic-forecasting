"""Read access to the processed store.

Everything here memory-maps by default. A cell's two-month series is 8 928 float32
values - about 36 KB - so slicing a memmap costs a page fault or two and nothing else.
There is never a reason to hold the full 357 MB matrix resident, and code that does so
by accident is the single easiest way to blow the memory budget this assignment is
partly graded on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ACTIVITIES,
    PROCESSED_DIR,
    SLOTS_PER_DAY,
    TARGET,
    TIMEZONE,
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `make ingest` to build the processed store."
        )
    return path


def materialise(activity: str, processed_dir: Path | str = PROCESSED_DIR) -> Path:
    """Build ``<activity>.npy`` from the per-day archives if it does not exist yet.

    Only the forecasting target is written eagerly by the ingestion; the other four
    activity channels are assembled the first time something asks for them. That keeps
    the default on-disk footprint to one matrix instead of five.
    """
    processed_dir = Path(processed_dir)
    out = processed_dir / f"{activity}.npy"
    if out.exists():
        return out
    if activity not in ACTIVITIES:
        raise ValueError(f"Unknown activity {activity!r}; expected one of {ACTIVITIES}.")

    daily = sorted((processed_dir / "daily").glob("*.npz"))
    if not daily:
        raise FileNotFoundError(f"No daily archives under {processed_dir / 'daily'}.")

    n_slots = len(daily) * SLOTS_PER_DAY
    with np.load(daily[0]) as z:
        n_squares = z[activity].shape[1]

    # Write through a memmap so assembly itself never holds the whole matrix in RAM.
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(n_slots, n_squares))
    for i, path in enumerate(daily):
        with np.load(path) as z:
            mm[i * SLOTS_PER_DAY : (i + 1) * SLOTS_PER_DAY] = z[activity]
    mm.flush()
    del mm
    return out


def load_matrix(
    activity: str = TARGET,
    *,
    mmap: bool = True,
    processed_dir: Path | str = PROCESSED_DIR,
) -> np.ndarray:
    """The ``(n_slots, n_squares)`` matrix for ``activity``.

    Returns a read-only ``np.memmap`` unless ``mmap=False``.
    """
    path = materialise(activity, processed_dir)
    return np.load(path, mmap_mode="r" if mmap else None)


@lru_cache(maxsize=4)
def load_index(processed_dir: str | None = None) -> pd.DatetimeIndex:
    """Localised ``DatetimeIndex`` aligned with the rows of the matrices."""
    root = Path(processed_dir) if processed_dir else PROCESSED_DIR
    ts = np.load(_require(root / "timestamps.npy"))
    return pd.DatetimeIndex(
        pd.to_datetime(ts, unit="ms", utc=True).tz_convert(TIMEZONE), name="time"
    )


@lru_cache(maxsize=4)
def load_square_ids(processed_dir: str | None = None) -> np.ndarray:
    """Column index -> square id."""
    root = Path(processed_dir) if processed_dir else PROCESSED_DIR
    return np.load(_require(root / "square_ids.npy"))


def column_of(square_id: int, processed_dir: Path | str = PROCESSED_DIR) -> int:
    """Matrix column holding ``square_id``."""
    ids = load_square_ids(str(processed_dir))
    hits = np.flatnonzero(ids == square_id)
    if hits.size == 0:
        raise KeyError(f"square_id {square_id} not present in the processed store.")
    return int(hits[0])


def load_series(
    square_id: int,
    activity: str = TARGET,
    *,
    start: str | None = None,
    end: str | None = None,
    processed_dir: Path | str = PROCESSED_DIR,
) -> pd.Series:
    """One cell's time series, optionally restricted to ``[start, end]`` (inclusive)."""
    matrix = load_matrix(activity, processed_dir=processed_dir)
    index = load_index(str(processed_dir))
    col = column_of(square_id, processed_dir)
    series = pd.Series(
        np.asarray(matrix[:, col], dtype=np.float32), index=index, name=f"{activity}_sq{square_id}"
    )
    if start or end:
        series = series.loc[start:end]
    return series


def load_square_totals(processed_dir: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    """Per-square totals with grid coordinates."""
    return pd.read_csv(_require(Path(processed_dir) / "square_totals.csv"))


def top_squares(
    n: int = 3, activity: str = TARGET, processed_dir: Path | str = PROCESSED_DIR
) -> list[int]:
    """The ``n`` highest-total squares for ``activity``, read from the totals table."""
    df = load_square_totals(processed_dir)
    return df.nlargest(n, activity)["square_id"].astype(int).tolist()


def to_grid(values: np.ndarray) -> np.ndarray:
    """Reshape a per-square vector onto the square spatial grid, for heatmaps.

    Square ids run row-major over a 100x100 lattice, so ``values[k]`` (the value for
    ``square_id == k + 1``) lands at ``grid[k // 100, k % 100]``. Needs nothing from the
    store beyond the vector itself, so it also works on derived quantities.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    side = int(round(float(v.size) ** 0.5))
    if side * side != v.size:
        raise ValueError(f"{v.size} values do not form a square grid.")
    return v.reshape(side, side)


def manifest(processed_dir: Path | str = PROCESSED_DIR) -> dict:
    """The ingestion manifest: row counts, timings, memory, compression ratio."""
    import json

    return json.loads(_require(Path(processed_dir) / "ingest_manifest.json").read_text())
