"""Turn the ingestion manifest into a committed evidence table.

    python scripts/make_ingest_table.py

`scripts/run_ingest.py` writes `data/processed/ingest_manifest.json` as it goes,
recording per-day rows, wall time and peak resident set size. That file is a data
artefact and is not tracked, so this script lifts the numbers the report quotes
at full scale into `reports/tables/ingest_scale.md`, which is.

The one-day before/after comparison lives in `reports/tables/memory_benchmark.md`
and answers a different question: it isolates the aggregation, measured as a
delta above a pre-call baseline. The numbers here are the whole 62-day run,
measured as absolute process RSS, so they also carry the output matrix under
construction. The table states both bases, because the two figures are otherwise
easy to read as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "data" / "processed" / "ingest_manifest.json"
DEFAULT_OUT = REPO / "reports" / "tables" / "ingest_scale.md"

GIB = 1024**3
EXPECTED_DAYS = 62
#: Below this, the run resumed from the per-day cache instead of reading raw text.
MIN_S_PER_DAY = 0.5


def render(manifest: dict, matrix_bytes: int) -> str:
    days = manifest["days"]
    n_days = manifest["n_days"]
    wall = manifest["wall_s"]
    peak_max = manifest["peak_rss_mb_max"]
    peak_min = min(d["peak_rss_mb"] for d in days)
    raw_bytes = manifest["raw_bytes"]
    matrix_mb = matrix_bytes / 1e6
    working_set = peak_max - matrix_mb

    n_rows = f"{manifest['total_rows']:,}".replace(",", " ")
    chunk = f"{manifest['chunksize']:,}".replace(",", " ")

    rows = [
        ("Raw input", f"{n_days} daily files, {raw_bytes / GIB:.1f} GiB, {n_rows} rows"),
        (
            "Target matrix",
            f"{manifest['n_slots']} x {manifest['n_squares']} float32, "
            f"{matrix_mb:.0f} MB ({raw_bytes / matrix_bytes:.0f}x smaller)",
        ),
        (
            "Full store",
            f"{manifest['processed_bytes'] / 1e9:.2f} GB including the per-day "
            f"archives of all five channels ({manifest['compression_ratio']:.1f}x)",
        ),
        ("Wall time", f"{wall:.1f} s for the run, {wall / n_days:.2f} s per day"),
        ("Peak RSS", f"{peak_max:.1f} MB highest day, {peak_min:.1f} MB lowest day"),
        (
            "Of which output",
            f"{matrix_mb:.0f} MB is the matrix under construction, so the "
            f"per-file working set peaks at {working_set:.0f} MB",
        ),
    ]

    out = [
        "# Ingestion at full scale",
        "",
        "Source: `data/processed/ingest_manifest.json`, written by "
        "`scripts/run_ingest.py` during the run it describes. Regenerate with "
        "`make ingest && make ingest-table`.",
        "",
        f"Hardware: {manifest['hardware']}",
        "",
        "| Quantity | Measurement |",
        "|---|---|",
    ]
    out += [f"| {k} | {v} |" for k, v in rows]
    out += [
        "",
        "**Two bases, not one scale.** `reports/tables/memory_benchmark.md` reports "
        "peak RSS *above a pre-call baseline* for a single day's aggregation, which "
        "is what isolates the effect of chunking. The figures here are *absolute* "
        "process RSS over the whole run, and the process holds the "
        f"{matrix_mb:.0f} MB output matrix from the first file to the last. The "
        "per-file working set is the difference, and it is flat in the number of "
        "days because peak memory follows the chunk size rather than the file.",
        "",
        f"Chunk size: {chunk} rows.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="write the table even if the manifest is not a cold run over all 62 files",
    )
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"{args.manifest} not found. Run `make ingest` first; the manifest is "
            "written as the ingestion proceeds."
        )
    manifest = json.loads(args.manifest.read_text())
    matrix = args.manifest.parent / f"{manifest['target']}.npy"
    if not matrix.exists():
        raise SystemExit(f"{matrix} not found beside the manifest.")

    # A manifest from `--limit 3`, or from a run that resumed over a warm cache, carries
    # the same field names and would quietly replace the full-scale numbers the report
    # quotes with a partial run's. Refuse both unless the caller says otherwise.
    if not args.allow_partial:
        if manifest["n_days"] != EXPECTED_DAYS:
            raise SystemExit(
                f"{args.manifest} describes {manifest['n_days']} days, not {EXPECTED_DAYS}. "
                "This table reports the full ingestion; rerun `make ingest` over all 62 "
                "files, or pass --allow-partial to write it anyway."
            )
        per_day = manifest["wall_s"] / manifest["n_days"]
        if per_day < MIN_S_PER_DAY:
            raise SystemExit(
                f"{args.manifest} reports {per_day:.2f} s per day, below {MIN_S_PER_DAY} s, "
                "which means the run resumed from the per-day cache rather than reading the "
                "raw files. Rerun with `python scripts/run_ingest.py --overwrite`, or pass "
                "--allow-partial to write it anyway."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(manifest, matrix.stat().st_size))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
