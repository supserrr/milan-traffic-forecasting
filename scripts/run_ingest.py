#!/usr/bin/env python3
"""Build the processed store from the raw Dataverse dump.

    python scripts/run_ingest.py                 # all 62 days, resuming from cache
    python scripts/run_ingest.py --limit 3       # smoke test on three days
    python scripts/run_ingest.py --overwrite     # ignore the per-day cache

Per-day archives are cached, so an interrupted run resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic.config import ACTIVITIES, PROCESSED_DIR, RAW_DIR, TARGET  # noqa: E402
from milan_traffic.ingest import ingest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--activities", nargs="+", default=list(ACTIVITIES))
    ap.add_argument("--target", default=TARGET)
    ap.add_argument(
        "--chunksize",
        type=int,
        default=2_000_000,
        help="rows per pandas chunk; this sets peak memory, not file size",
    )
    ap.add_argument("--limit", type=int, default=None, help="process only the first N days")
    ap.add_argument("--overwrite", action="store_true", help="ignore cached daily archives")
    args = ap.parse_args()

    manifest = ingest(
        args.raw_dir,
        args.out_dir,
        activities=tuple(args.activities),
        target=args.target,
        chunksize=args.chunksize,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(
        f"\n{manifest['total_rows']:,} raw rows -> "
        f"{manifest['n_slots']} x {manifest['n_squares']} matrix  "
        f"({manifest['compression_ratio']}x smaller on disk, "
        f"peak RSS {manifest['peak_rss_mb_max']:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
