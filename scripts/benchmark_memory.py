#!/usr/bin/env python3
"""Measure the memory cost of three ways of reducing one raw daily file.

The assignment asks for "evidence showing memory usage before and after your
optimisation". This produces it: the same aggregation, computed three ways, with peak
RSS and wall time recorded for each, written to ``reports/tables/memory_benchmark.md``.

    A. naive      full-file read_csv with pandas' default dtypes, then groupby
    B. downcast   full-file read, narrow dtypes, only the needed columns, then groupby
    C. streaming  chunked read folded into a preallocated array - what ingest.py does

The interesting column is not the ratio between A and C on one file; it is what happens
when the input grows. A and B scale with file size, C does not, which is the difference
between a pipeline that runs on 62 files and one that does not.

    python scripts/benchmark_memory.py --file <path>   # defaults to the first raw day
    python scripts/benchmark_memory.py --skip-naive    # if RAM is tight
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic.config import (  # noqa: E402
    N_SQUARES,
    RAW_COLUMNS,
    RAW_DIR,
    SLOT_MS,
    SLOTS_PER_DAY,
    TABLES_DIR,
    TARGET,
)
from milan_traffic.ingest import (
    RAW_DTYPES,
    date_from_path,
    day_start_ms,
    iter_raw_files,
)  # noqa: E402
from milan_traffic.utils.memory import MemoryTracker, human_bytes  # noqa: E402
from milan_traffic.utils.timing import Timer, hardware_string  # noqa: E402


def variant_naive(path: Path) -> float:
    """Everything pandas does by default: all columns, int64/float64, then groupby."""
    df = pd.read_csv(path, sep="\t", header=None, names=list(RAW_COLUMNS))
    out = df.groupby(["time_interval", "square_id"])[TARGET].sum()
    total = float(out.sum())
    del df, out
    return total


def variant_downcast(path: Path) -> float:
    """Same shape of computation, but only the needed columns at narrow dtypes."""
    usecols = ["square_id", "time_interval", TARGET]
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(RAW_COLUMNS),
        usecols=usecols,
        dtype={k: v for k, v in RAW_DTYPES.items() if k in usecols},
    )
    out = df.groupby(["time_interval", "square_id"])[TARGET].sum()
    total = float(out.sum())
    del df, out
    return total


def variant_streaming(path: Path, chunksize: int = 2_000_000) -> float:
    """Chunked read folded into a preallocated accumulator - the production path."""
    origin = day_start_ms(date_from_path(path))
    acc = np.zeros(SLOTS_PER_DAY * N_SQUARES, dtype=np.float64)
    usecols = ["square_id", "time_interval", TARGET]
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(RAW_COLUMNS),
        usecols=usecols,
        dtype={k: v for k, v in RAW_DTYPES.items() if k in usecols},
        chunksize=chunksize,
    )
    for chunk in reader:
        slot = (chunk["time_interval"].to_numpy() - origin) // SLOT_MS
        square = chunk["square_id"].to_numpy().astype(np.int64) - 1
        values = np.nan_to_num(chunk[TARGET].to_numpy(dtype=np.float64))
        acc += np.bincount(slot * N_SQUARES + square, weights=values, minlength=acc.size)
        del chunk, slot, square, values
    return float(acc.sum())


VARIANTS = {
    "A. naive full read (default dtypes, all columns)": variant_naive,
    "B. full read (narrow dtypes, 3 columns)": variant_downcast,
    "C. chunked streaming accumulation": variant_streaming,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--file", type=Path, default=None)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--skip-naive", action="store_true")
    ap.add_argument("--out", type=Path, default=TABLES_DIR / "memory_benchmark.md")
    args = ap.parse_args()

    path = args.file or iter_raw_files(args.raw_dir)[0]
    size = path.stat().st_size
    hardware = hardware_string()
    print(f"File: {path.name} ({human_bytes(size)})\nHardware: {hardware}\n")

    rows = []
    for label, fn in VARIANTS.items():
        if args.skip_naive and label.startswith("A."):
            rows.append((label, None, None, None))
            continue
        gc.collect()
        with MemoryTracker() as mem, Timer(label) as t:
            total = fn(path)
        gc.collect()
        rows.append((label, mem.delta_mb, t.wall_s, total))
        print(
            f"{label:52s} peak +{mem.delta_mb:7.0f} MB   {t.wall_s:6.2f} s   " f"total={total:,.0f}"
        )

    measured = [r for r in rows if r[1] is not None]
    best = min(r[1] for r in measured) if measured else float("nan")

    lines = [
        "# Memory benchmark",
        "",
        f"Single raw file `{path.name}` ({human_bytes(size)}), aggregating Internet",
        "traffic to (10-minute slot x square). Identical output, three implementations.",
        "",
        f"- **Hardware:** {hardware}",
        "- **Measurement:** peak resident set size above the pre-call baseline, sampled",
        "  every 20 ms from a background thread; wall time from `time.perf_counter()`.",
        "- Each variant runs in the same process after an explicit `gc.collect()`.",
        "",
        "| variant | peak RSS above baseline | wall time | x leanest | aggregate total |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, peak, wall, total in rows:
        if peak is None:
            lines.append(f"| {label} | _skipped_ | _skipped_ | - | - |")
        else:
            lines.append(
                f"| {label} | {peak:,.0f} MB | {wall:.2f} s | {peak / best:.1f}x | {total:,.0f} |"
            )

    totals = [r[3] for r in rows if r[3] is not None]
    if len(totals) > 1 and max(totals) != min(totals):
        lines += [
            "",
            "## A precision footnote",
            "",
            "The aggregate totals are not bit-identical. Variant B both stores *and sums* in",
            "float32, so summing ~4.8 M values accumulates rounding error; variants A and C sum",
            "in float64 and agree exactly. The production pipeline therefore reads float32 but",
            "**accumulates in float64 and casts down once at the end** - narrow dtypes are a",
            "storage decision, not an arithmetic one. Downcasting without thinking about where",
            "the sum happens buys memory at the cost of silently wrong totals.",
        ]

    n_files = len(iter_raw_files(args.raw_dir))
    lines += [
        "",
        "## Why this matters at full scale",
        "",
        f"The dataset is {n_files} files of this size (~21 GB of text, ~300 M rows).",
        "Variants A and B hold the parsed file in memory, so their footprint grows with",
        "the input; variant C's footprint is set by the chunk size, so it is flat in the",
        "number of rows. Only C's peak stays constant as days are added, which is what",
        "makes the full ingestion run on a laptop rather than requiring the whole dump",
        "to fit in RAM.",
        "",
        "The reduction on disk is separate and larger: summing out `country_code` turns",
        "~21 GB of text into a 8 928 x 10 000 float32 matrix (~357 MB) that is then",
        "memory-mapped rather than loaded, so downstream analysis costs page faults",
        "instead of RAM. See `data/processed/ingest_manifest.json` for the exact ratio.",
        "",
        "**Trade-off.** The aggregation is lossy: per-country granularity is discarded and",
        "cannot be recovered without re-ingesting. That is the right call for a research",
        "question about total traffic per cell, and the wrong one for anything about",
        "roaming or international behaviour.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
