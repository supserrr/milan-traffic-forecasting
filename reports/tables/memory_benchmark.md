# Memory benchmark

Single raw file `sms-call-internet-mi-2013-11-01.txt` (307.9 MiB), aggregating Internet
traffic to (10-minute slot x square). Identical output, three implementations.

- **Hardware:** Linux | 6.8.0-136-generic | aarch64 | 4P/4L cores | 4 GiB RAM
- **Measurement:** peak resident set size above the pre-call baseline, sampled
  every 20 ms from a background thread; wall time from `time.perf_counter()`.
- Each variant runs in the same process after an explicit `gc.collect()`.

| variant | peak RSS above baseline | wall time | x leanest | aggregate total |
|---|---:|---:|---:|---:|
| A. naive full read (default dtypes, all columns) | 559 MB | 2.03 s | 5.1x | 82,479,247 |
| B. full read (narrow dtypes, 3 columns) | 313 MB | 1.57 s | 2.9x | 82,479,264 |
| C. chunked streaming accumulation | 109 MB | 1.41 s | 1.0x | 82,479,247 |

## A precision footnote

The aggregate totals are not bit-identical. Variant B both stores *and sums* in
float32, so summing ~4.8 M values accumulates rounding error; variants A and C sum
in float64 and agree exactly. The production pipeline therefore reads float32 but
**accumulates in float64 and casts down once at the end** - narrow dtypes are a
storage decision, not an arithmetic one. Downcasting without thinking about where
the sum happens buys memory at the cost of silently wrong totals.

## Why this matters at full scale

The dataset is 62 files of this size (~21 GB of text, ~300 M rows).
Variants A and B hold the parsed file in memory, so their footprint grows with
the input; variant C's footprint is set by the chunk size, so it is flat in the
number of rows. Only C's peak stays constant as days are added, which is what
makes the full ingestion run on a laptop rather than requiring the whole dump
to fit in RAM.

The reduction on disk is separate and larger: summing out `country_code` turns
~21 GB of text into a 8 928 x 10 000 float32 matrix (~357 MB) that is then
memory-mapped rather than loaded, so downstream analysis costs page faults
instead of RAM. See `data/processed/ingest_manifest.json` for the exact ratio.

**Trade-off.** The aggregation is lossy: per-country granularity is discarded and
cannot be recovered without re-ingesting. That is the right call for a research
question about total traffic per cell, and the wrong one for anything about
roaming or international behaviour.
