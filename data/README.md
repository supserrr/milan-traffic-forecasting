# data/

**Nothing here is tracked in git.** Everything is rebuildable — see the root `README.md`.

```
data/
├── raw/           (unused — raw .txt files live in ../../Dataverse/)
├── interim/       scratch space for intermediate artefacts
├── external/      anything not from the primary dataset (e.g. grid geometry, holidays)
└── processed/     ← the store every downstream module reads
    ├── internet.npy            (8928, 10000) float32   time × square, the target — 357 MB
    ├── timestamps.npy          (8928,) int64           epoch ms, 10-min spacing
    ├── square_ids.npy          (10000,) int32          column → square id
    ├── square_totals.csv       per-square totals + grid coordinates
    ├── ingest_manifest.json    row counts, timings, peak RSS, compression ratio
    └── daily/YYYY-MM-DD.npz    (144, 10000) × 5 activities, compressed — 1.3 GB total
```

The whole store is 1.68 GB (1.57 GiB), 12.4 times smaller than the 20.8 GB of raw text it was
built from; `ingest_manifest.json` records the exact byte counts.

Only the forecasting target is written as a full matrix. The other four activity channels stay in
the daily archives and are assembled on first request by `dataio.load_matrix`, which keeps the
default footprint at one matrix instead of five.

Rebuild with `make ingest` (resumable — per-day archives are cached) or wipe with
`make clean-processed`.
