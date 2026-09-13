# Data dictionary

## Raw files - `../Dataverse/sms-call-internet-mi-YYYY-MM-DD.txt`

62 files, `2013-11-01` … `2014-01-01`, ~21 GB total, ~4.8 M rows each.
Tab-separated, **no header**, missing values are **empty fields** (not zeros - the absence of a
record means no activity was registered for that country in that slot).

| idx | name | type | description |
|----:|------|------|-------------|
| 0 | `square_id` | int 1–10000 | Grid cell. The grid is 100×100 over Milan; ids run row-major. `row = (id-1)//100`, `col = (id-1)%100`. |
| 1 | `time_interval` | int64 | Epoch **milliseconds** at the start of a 10-minute slot. Local time base is CET (UTC+1); the observation window contains no DST transition. |
| 2 | `country_code` | int | Country of the other party. `0` appears for some records. Summed out in this project. |
| 3 | `sms_in` | float | SMS reception activity |
| 4 | `sms_out` | float | SMS sending activity |
| 5 | `call_in` | float | Incoming call activity |
| 6 | `call_out` | float | Outgoing call activity |
| 7 | `internet` | float | Internet CDR activity - **the forecasting target** |

**Units.** These are *scaled* activity measures derived from Call Detail Records, published
without the scaling constant to protect commercial confidentiality. They are comparable across
cells and across time but have no absolute physical interpretation. Do not report them as MB or
Mbps.

**Sampling caveat from the source paper.** An Internet CDR is generated when a user starts or ends
a connection, or every 15 min / 5 MB during a session. Very short or very light sessions are
therefore under-represented, and the measure is closer to *connection activity* than to volume.

## Processed store - `data/processed/`

| file | shape | dtype | description |
|---|---|---|---|
| `internet.npy` | (8928, 10000) | float32 | time × square, country codes summed out |
| `sms_in.npy` / `sms_out.npy` / `call_in.npy` / `call_out.npy` | (8928, 10000) | float32 | as above; not written by the ingest, assembled on first request by `dataio.load_matrix` |
| `timestamps.npy` | (8928,) | int64 | epoch ms, ascending, uniform 600 000 ms spacing |
| `square_ids.npy` | (10000,) | int32 | column index → square id |
| `square_totals.csv` | 10000 rows | - | `square_id, row, col, sms_in, sms_out, call_in, call_out, internet` totals over the full period |
| `daily/YYYY-MM-DD.npz` | (144, 10000) ×5 | float32 | one compressed archive per day, each with a `.json` sidecar of that day's counts and totals |
| `ingest_manifest.json` | - | - | per-day row counts, totals, timings, peak RSS; used to verify integrity |

**Reading the manifest's timings.** The top-level `wall_s` is the wall time of the *last*
invocation of `run_ingest.py`, which is normally a resumed one that rebuilds only what the
per-day cache was missing; on the store shipped with this work it reads 9.7 s. The cost of a
full ingestion is the **sum of the 62 per-day `wall_s` entries**, 129.4 s (2.09 s per day),
and that is the figure the report and the README quote. `peak_rss_mb_max` is already a maximum
over the per-day entries, so it needs no such care: 377.9 MB either way.

**Time axis.** 62 days × 144 slots = **8928** rows. Row `t` corresponds to
`timestamps[t]`; convert with `pd.to_datetime(ts, unit="ms", utc=True).tz_convert("Europe/Rome")`.

**Missing data policy.** Empty raw fields are treated as "no recorded activity" and contribute
zero to the aggregate. A slot with no rows at all for a square is therefore exactly `0.0`, which
is indistinguishable from genuinely zero activity - a known limitation of this dataset.
`ingest_manifest.json` records how many (slot, square) pairs had no contributing rows, so the
extent of the ambiguity is measurable; report it rather than ignoring it.
