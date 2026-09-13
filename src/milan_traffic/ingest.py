"""Reduce the raw Telecom Italia dumps to a compact, memory-mappable store.

This is the only module that reads ``../Dataverse/``. Everything downstream reads
``data/processed/``.

Why it is written this way
--------------------------
The 62 daily files hold ~300 M rows (~21 GB of text) at a granularity the assignment
does not need: each row is one *(square, 10-minute slot, country)* triple. The target
quantity is traffic per square per slot, so the country dimension is summed out. That
turns the problem from "300 M rows" into "8 928 x 10 000 floats" - about 357 MB as
float32, small enough to memory-map and slice for free.

The reduction is done as a **streaming accumulation** rather than a load-then-group:

1. each file is read in chunks, never in full;
2. only the seven needed columns are parsed (``country_code`` is skipped outright,
   which is the cheapest possible way to handle a column you intend to aggregate away);
3. dtypes are pinned narrow (``int32`` / ``float32``) instead of pandas' default
   ``int64`` / ``float64``;
4. each chunk is folded into a preallocated ``(144, 10000)`` accumulator with
   ``np.bincount`` on a flattened index, so peak memory is set by the chunk size and
   not by the file size.

The consequence is that peak RSS is roughly constant in the size of the input - the
property that matters when the dataset is larger than RAM. ``scripts/benchmark_memory.py``
measures the difference against a naive full-file read; the numbers it produces are the
evidence the report needs.

Trade-off worth stating in the write-up: summing over ``country_code`` is lossy and
irreversible without re-ingesting. It is the right call for this research question
(total traffic per cell) and the wrong one for, say, a study of roaming behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ACTIVITIES,
    GRID_SIDE,
    N_SQUARES,
    PROCESSED_DIR,
    RAW_COLUMNS,
    RAW_DIR,
    SLOT_MS,
    SLOTS_PER_DAY,
    TARGET,
    TIMEZONE,
)
from .utils.logging import get_logger
from .utils.memory import MemoryTracker
from .utils.timing import Timer, hardware_string

LOG = get_logger(__name__)

FILE_GLOB = "sms-call-internet-mi-*.txt"

#: Narrow dtypes: ~2x smaller than pandas' defaults for the same information.
RAW_DTYPES: dict[str, str] = {
    "square_id": "int32",
    "time_interval": "int64",
    "country_code": "int32",
    **{a: "float32" for a in ACTIVITIES},
}


# --------------------------------------------------------------------------- helpers


def date_from_path(path: Path) -> _date:
    """Extract the observation date encoded in a raw filename."""
    stem = path.stem  # sms-call-internet-mi-2013-11-01
    return datetime.strptime(stem.split("mi-")[-1], "%Y-%m-%d").date()


def day_start_ms(day: _date) -> int:
    """Epoch milliseconds of local midnight for ``day`` in the dataset's timezone.

    Falls back to a fixed UTC+1 offset if the system has no timezone database; the
    observation window (Nov 2013 - Jan 2014) is entirely within CET, so the fallback is
    exact for this dataset. Slot indices are validated afterwards either way.
    """
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(TIMEZONE)
        dt = datetime(day.year, day.month, day.day, tzinfo=tz)
    except Exception:  # pragma: no cover - only on systems without tzdata
        dt = datetime(day.year, day.month, day.day, tzinfo=timezone(timedelta(hours=1)))
    return int(dt.timestamp() * 1000)


def iter_raw_files(raw_dir: Path | str = RAW_DIR) -> list[Path]:
    """All daily raw files, in chronological order."""
    paths = sorted(Path(raw_dir).glob(FILE_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {FILE_GLOB!r} under {raw_dir}. "
            "Expected the Dataverse dump alongside the repository."
        )
    return paths


# ----------------------------------------------------------------------- day reducer


@dataclass(slots=True)
class DayResult:
    """One day reduced to ``(144, 10000)`` arrays plus the stats the manifest records."""

    day: _date
    arrays: dict[str, np.ndarray]
    rows: int
    observed_cells: int
    wall_s: float
    peak_rss_mb: float
    totals: dict[str, float] = field(default_factory=dict)

    @property
    def empty_cells(self) -> int:
        """(slot, square) pairs with no contributing raw row - zero by convention."""
        return SLOTS_PER_DAY * N_SQUARES - self.observed_cells


def reduce_day(
    path: Path,
    *,
    activities: Sequence[str] = ACTIVITIES,
    chunksize: int = 2_000_000,
) -> DayResult:
    """Stream one raw file into per-activity ``(144, 10000)`` float32 arrays.

    Parameters
    ----------
    path:
        A ``sms-call-internet-mi-YYYY-MM-DD.txt`` file.
    activities:
        Which activity columns to accumulate. Parsing fewer columns is faster.
    chunksize:
        Rows per pandas chunk. This, not the file size, sets peak memory.
    """
    day = date_from_path(path)
    origin = day_start_ms(day)
    n_cells = SLOTS_PER_DAY * N_SQUARES

    # float64 accumulators: the sum over ~480 country rows per cell is where precision
    # is lost if you accumulate in float32. Cast down once, at the end.
    acc = {a: np.zeros(n_cells, dtype=np.float64) for a in activities}
    observed = np.zeros(n_cells, dtype=np.int32)

    usecols = ["square_id", "time_interval", *activities]
    rows = 0

    with MemoryTracker() as mem, Timer(path.name) as timer:
        reader = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=RAW_COLUMNS,
            usecols=usecols,
            dtype={k: v for k, v in RAW_DTYPES.items() if k in usecols},
            chunksize=chunksize,
            engine="c",
        )
        for chunk in reader:
            rows += len(chunk)

            ts = chunk["time_interval"].to_numpy()
            slot = (ts - origin) // SLOT_MS
            square = chunk["square_id"].to_numpy().astype(np.int64) - 1

            if slot.min() < 0 or slot.max() >= SLOTS_PER_DAY:
                raise ValueError(
                    f"{path.name}: slot index out of range "
                    f"[{slot.min()}, {slot.max()}] - day origin {origin} looks wrong."
                )
            if square.min() < 0 or square.max() >= N_SQUARES:
                raise ValueError(f"{path.name}: square_id outside 1..{N_SQUARES}.")

            flat = slot * N_SQUARES + square

            for activity in activities:
                values = chunk[activity].to_numpy(dtype=np.float64)
                present = ~np.isnan(values)
                if activity == TARGET:
                    observed += np.bincount(flat[present], minlength=n_cells).astype(np.int32)
                np.nan_to_num(values, copy=False)
                acc[activity] += np.bincount(flat, weights=values, minlength=n_cells)

            del chunk, ts, slot, square, flat

    arrays = {a: acc[a].astype(np.float32).reshape(SLOTS_PER_DAY, N_SQUARES) for a in activities}
    totals = {a: float(acc[a].sum()) for a in activities}

    return DayResult(
        day=day,
        arrays=arrays,
        rows=rows,
        observed_cells=int((observed > 0).sum()),
        wall_s=timer.wall_s,
        peak_rss_mb=mem.peak_mb,
        totals=totals,
    )


# ------------------------------------------------------------------------- pipeline


def ingest(
    raw_dir: Path | str = RAW_DIR,
    out_dir: Path | str = PROCESSED_DIR,
    *,
    activities: Sequence[str] = ACTIVITIES,
    target: str = TARGET,
    chunksize: int = 2_000_000,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Run the full reduction and write the processed store.

    Writes
    ------
    ``daily/YYYY-MM-DD.npz``
        Per-day ``(144, 10000)`` arrays for every activity, compressed. These are the
        durable intermediate: any full matrix can be rebuilt from them without touching
        the raw text again.
    ``<target>.npy``
        The full ``(n_slots, 10000)`` matrix for the forecasting target, written
        eagerly because every downstream step needs it. Other activities stay in the
        daily archives and are materialised on demand by :func:`dataio.load_matrix`,
        which keeps the eager footprint at ~357 MB instead of ~1.8 GB.
    ``timestamps.npy`` / ``square_ids.npy``
        Axis labels for the matrix.
    ``square_totals.csv``
        Per-square totals for each activity, plus grid coordinates - the source of
        truth for "the three areas with the highest total Internet traffic".
    ``ingest_manifest.json``
        Row counts, totals, timings and peak RSS per day, plus the hardware string.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_raw_files(raw_dir)[:limit]
    LOG.info("Ingesting %d files from %s", len(paths), raw_dir)
    LOG.info("Hardware: %s", hardware_string())

    n_days = len(paths)
    target_matrix = np.zeros((n_days * SLOTS_PER_DAY, N_SQUARES), dtype=np.float32)
    totals = {a: np.zeros(N_SQUARES, dtype=np.float64) for a in activities}
    timestamps = np.empty(n_days * SLOTS_PER_DAY, dtype=np.int64)

    records: list[dict] = []
    with Timer("ingest") as overall:
        for i, path in enumerate(paths):
            cache = daily_dir / f"{date_from_path(path):%Y-%m-%d}.npz"
            if cache.exists() and not overwrite:
                LOG.info("[%2d/%d] %s (cached)", i + 1, n_days, cache.name)
                with np.load(cache) as z:
                    arrays = {a: z[a] for a in activities}
                meta = (
                    json.loads(cache.with_suffix(".json").read_text())
                    if cache.with_suffix(".json").exists()
                    else {}
                )
                result = DayResult(
                    day=date_from_path(path),
                    arrays=arrays,
                    rows=int(meta.get("rows", 0)),
                    observed_cells=int(meta.get("observed_cells", 0)),
                    wall_s=float(meta.get("wall_s", 0.0)),
                    peak_rss_mb=float(meta.get("peak_rss_mb", 0.0)),
                    totals={a: float(arrays[a].sum()) for a in activities},
                )
            else:
                result = reduce_day(path, activities=activities, chunksize=chunksize)
                np.savez_compressed(cache, **result.arrays)
                cache.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "rows": result.rows,
                            "observed_cells": result.observed_cells,
                            "wall_s": result.wall_s,
                            "peak_rss_mb": result.peak_rss_mb,
                        }
                    )
                )
                LOG.info(
                    "[%2d/%d] %s  rows=%s  %.1fs  peakRSS=%.0fMB  empty_cells=%s",
                    i + 1,
                    n_days,
                    path.name,
                    f"{result.rows:,}",
                    result.wall_s,
                    result.peak_rss_mb,
                    f"{result.empty_cells:,}",
                )

            lo = i * SLOTS_PER_DAY
            target_matrix[lo : lo + SLOTS_PER_DAY] = result.arrays[target]
            for a in activities:
                totals[a] += result.arrays[a].sum(axis=0, dtype=np.float64)

            origin = day_start_ms(result.day)
            timestamps[lo : lo + SLOTS_PER_DAY] = origin + np.arange(SLOTS_PER_DAY) * SLOT_MS

            records.append(
                {
                    "date": f"{result.day:%Y-%m-%d}",
                    "rows": result.rows,
                    "observed_cells": result.observed_cells,
                    "empty_cells": result.empty_cells,
                    "wall_s": round(result.wall_s, 3),
                    "peak_rss_mb": round(result.peak_rss_mb, 1),
                    "totals": {a: result.totals[a] for a in activities},
                }
            )
            del result

    # --- axis labels and the eager target matrix ---------------------------------
    square_ids = np.arange(1, N_SQUARES + 1, dtype=np.int32)
    np.save(out_dir / f"{target}.npy", target_matrix)
    np.save(out_dir / "timestamps.npy", timestamps)
    np.save(out_dir / "square_ids.npy", square_ids)

    gaps = np.unique(np.diff(timestamps))
    if not (gaps.size == 1 and gaps[0] == SLOT_MS):
        LOG.warning("Non-uniform time axis; unique gaps (ms): %s", gaps.tolist())

    # --- per-square totals --------------------------------------------------------
    rows_, cols_ = np.divmod(square_ids.astype(np.int64) - 1, GRID_SIDE)
    totals_df = pd.DataFrame({"square_id": square_ids, "row": rows_, "col": cols_})
    for a in activities:
        totals_df[a] = totals[a]
    totals_df.to_csv(out_dir / "square_totals.csv", index=False)

    raw_bytes = sum(p.stat().st_size for p in paths)
    processed_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware_string(),
        "raw_dir": str(raw_dir),
        "n_days": n_days,
        "n_slots": int(n_days * SLOTS_PER_DAY),
        "n_squares": N_SQUARES,
        "activities": list(activities),
        "target": target,
        "chunksize": chunksize,
        "total_rows": int(sum(r["rows"] for r in records)),
        "raw_bytes": raw_bytes,
        "processed_bytes": processed_bytes,
        "compression_ratio": round(raw_bytes / max(processed_bytes, 1), 1),
        "wall_s": round(overall.wall_s, 1),
        "peak_rss_mb_max": max((r["peak_rss_mb"] for r in records), default=0.0),
        "days": records,
    }
    (out_dir / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2))

    LOG.info(
        "Done in %.1fs. %s raw rows -> %s matrix. %.1f GB -> %.0f MB (%.0fx).",
        overall.wall_s,
        f"{manifest['total_rows']:,}",
        target_matrix.shape,
        raw_bytes / 2**30,
        processed_bytes / 2**20,
        manifest["compression_ratio"],
    )
    return manifest
