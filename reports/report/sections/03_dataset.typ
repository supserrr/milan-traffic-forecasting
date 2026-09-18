#import "../lib.typ": *

= Dataset and Data Preparation

== Provenance and schema

The data are the Milan Call Detail Record (CDR) aggregates published for the Telecom Italia Big Data Challenge [3], [16]. Milan is covered by a 100 × 100 grid of cells about 235 m on a side [17], with activity aggregated into 10-minute slots from 1 November 2013 to 1 January 2014. The release is 62 daily tab-separated files, 19.4 GiB and 319 896 289 rows. Each row is one (cell, slot, country) triple: `square_id` (1 to 10 000), `time_interval` (epoch milliseconds, CET), `country_code`, then `sms_in`, `sms_out`, `call_in`, `call_out` and `internet`, the target. An empty field is a missing record, not a zero.

== What the values measure

The values are scaled CDR activity counts published without the scaling constant, so they are comparable across cells and across time but have no physical unit. An Internet CDR is written when a connection starts or ends and, during a session, every 15 minutes or every 5 MB [3], so the series measures connection activity more than transferred volume.

== Reduction to a memory-mapped store

The research question needs traffic per cell per slot, so `country_code` is summed out, collapsing the 319 896 289 rows into an 8 928 × 10 000 float32 matrix: 357 MB on disk, a 58-fold reduction, with the run that produced it recorded in `reports/tables/ingest_scale.md`. Per-day archives keep all five channels and bring the store to 1.68 GB. The matrix is memory-mapped with NumPy [18], so analysis code slices columns without loading the array.

The reduction is a streaming accumulation using the three levers the pandas documentation gives for larger-than-memory data: fewer columns, narrower dtypes and chunking [19], [20]. Each file is read in chunks of 2 000 000 rows; `country_code` is never parsed; `square_id` is pinned to int32 and the activity columns to float32, while `time_interval` has to stay int64 because epoch milliseconds overflow int32; each chunk is folded into a preallocated (144, 10 000) float64 accumulator with `np.bincount`. Peak memory follows the chunk, not the file.

== Memory evidence

Table 1 compares three implementations of the one-day aggregation, with peak resident set size sampled every 20 ms above a pre-call baseline. The naive full read needs 559 MB; narrow dtypes and three columns cut that to 313 MB; the chunked accumulator needs 109 MB, 5.1 times less, and is the fastest at 1.41 s. Measuring the delta above a pre-call baseline for a single day is what isolates the effect of chunking, and it is a different basis from the one used at full scale. The 62-day ingestion is reported as absolute process RSS, over a process that holds the 357 MB output matrix from the first file to the last: the run took 161.4 s, 2.60 s per day, and peak RSS was 623.2 MB on the busiest day and 386.4 MB on the quietest, of that peak, 357 MB is the matrix being built, so the per-file working set peaks at 266 MB and does not grow with the number of days. The per-day record is in `reports/tables/ingest_scale.md`.

#apa-table(1, [Peak resident memory above baseline and wall time, three implementations of the one-day aggregation (file of 1 November 2013, 307.9 MiB; Linux aarch64, 4 GiB RAM).], [
  #apa-body(
    (1fr, auto, auto, auto),
    align: (left, right, right, right),
    rule,
    [*Variant*], [*Peak RSS*], [*Wall time*], [*Day total*],
    rule,
    [A. Naive full read, all columns], [559 MB], [2.03 s], [82 479 247],
    [B. Full read, narrow dtypes, 3 columns], [313 MB], [1.57 s], [82 479 264],
    [C. Chunked accumulation], [109 MB], [1.41 s], [82 479 247],
    rule,
  )
])

The last column carries a precision warning: variant B sums in float32 and its total differs by 17 units from the float64 sums of A and C. Narrow dtypes are a storage decision, not an arithmetic one; the pipeline accumulates in float64 and casts once.

== Trade-offs and limitations

Summing over country is lossy and irreversible without re-ingesting the 19.4 GiB, so the store answers nothing about roaming. A zero in the store means no row carried a non-missing Internet value for that cell and slot, indistinguishable from genuine silence: 152 527 such pairs, 0.17 percent of the matrix, concentrated in 98 low-traffic cells, five of which record nothing after early December; none of the five modelled cells contains a zero. The matrix is also time-major, so reading one cell's 36 KB series touches about 146 MB of file-backed cache; the process's physical footprint is unchanged, but the layout remains a limitation.

== Hardware

The one-day comparison of Table 1 was measured in the 4 GiB Linux sandbox, which is the constraint the memory strategy was sized for. The full 62-day ingestion, all model training and every timing in Section 6 ran on an Apple M1 Pro with 16 GB of unified memory under macOS, using PyTorch 2.14 [21] with the MPS backend. Every quoted number names the machine it was measured on.
