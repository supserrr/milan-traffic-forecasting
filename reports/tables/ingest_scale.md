# Ingestion at full scale

Source: `data/processed/ingest_manifest.json`, written by `scripts/run_ingest.py` during the run it describes. Regenerate with `make ingest && make ingest-table`.

Hardware: Darwin | 27.0.0 | arm64 | Apple M1 Pro | 8P/8L cores | 16 GiB RAM | torch:mps

| Quantity | Measurement |
|---|---|
| Raw input | 62 daily files, 19.4 GiB, 319 896 289 rows |
| Target matrix | 8928 x 10000 float32, 357 MB (58x smaller) |
| Full store | 1.68 GB including the per-day archives of all five channels (12.4x) |
| Wall time | 161.4 s for the run, 2.60 s per day |
| Peak RSS | 623.2 MB highest day, 386.4 MB lowest day |
| Of which output | 357 MB is the matrix under construction, so the per-file working set peaks at 266 MB |

**Two bases, not one scale.** `reports/tables/memory_benchmark.md` reports peak RSS *above a pre-call baseline* for a single day's aggregation, which is what isolates the effect of chunking. The figures here are *absolute* process RSS over the whole run, and the process holds the 357 MB output matrix from the first file to the last. The per-file working set is the difference, and it is flat in the number of days because peak memory follows the chunk size rather than the file.

Chunk size: 2 000 000 rows.
