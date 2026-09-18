# Forecasting Mobile Network Traffic in Milan

Comparative study of three sequential models for **one-step-ahead** forecasting of mobile Internet
traffic, on the Telecom Italia Big Data Challenge dataset for Milan (Nov 2013 to Jan 2014).

> How do different sequential models compare for one-step-ahead mobile network traffic
> forecasting, and how does their performance vary across geographical areas with different
> traffic characteristics?

| Deliverable | Where |
|---|---|
| **Report** | [`reports/report/report.pdf`](reports/report/report.pdf), sources in [`reports/report/`](reports/report) |
| **Code** | <https://github.com/supserrr/milan-traffic-forecasting> (MIT, see `LICENSE`) |
| **Video** | _pending: add the URL here and rebuild the report with `make report VIDEO_URL=...`_ |

An LSTM, a dilated TCN and gradient-boosted trees are trained per area on an identical 144-slot
window and scored on 16 to 22 December against persistence, seasonal-naive and linear
autoregressive baselines, on the three highest-traffic cells. The finding is that the ordering of
the three architectures reverses between cells 500 m apart (p = 0.004), while no architecture
betters a 145-parameter linear autoregression on average.

## Setup

Python >= 3.10, ~1 GB RAM, ~2 GB disk for the processed store. No GPU required.

```bash
make setup && source .venv/bin/activate
```

Dependencies are declared in `requirements.txt`; `requirements-lock.txt` pins the exact
environment every reported number was produced in. `make help` lists every target.

## Data

The raw data is not in this repository (~19 GiB, 62 daily files). Download
`sms-call-internet-mi-YYYY-MM-DD.txt` (2013-11-01 to 2014-01-01) from
[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
into a `Dataverse/` directory next to this repository, then:

```bash
make ingest-smoke     # 3 days, ~15 s, checks the pipeline end to end
make ingest           # all 62 days, ~2.7 min, resumable
```

Ingestion streams each file in 2 M-row chunks into a preallocated accumulator, reducing 19.4 GiB
of text and 319.9 M rows to a memory-mapped 8928 x 10000 float32 matrix (357 MB). Peak memory
follows the chunk rather than the file: `reports/tables/memory_benchmark.md` has the one-day
before/after comparison and `reports/tables/ingest_scale.md` the full-run record. Schemas are in
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Reproducing the results

```bash
make bench ingest-table tsa eda-stats section4-checks figures   # evidence tables and figures
python scripts/run_baselines.py --extra-areas --run-id EXP-000
python scripts/run_experiment.py --models lstm tcn gbt \
    --baselines naive seasonal_naive_daily linear_ar_144 \
    --seq-len 144 --scaling log1p-standard --seeds 42 1337 2024
python scripts/make_results.py --run EXP-008
make significance period-breakdown capacity-table
make report VIDEO_URL=https://...                               # -> reports/report/report.pdf
```

The reported run is **EXP-008** (three architectures, three areas, three seeds; ~20 min on an
M1 Pro). Its predictions are tracked, so every table and figure regenerates without retraining.
`run_experiment.py` claims the next free id, so on a clean clone training writes EXP-019: pass
that id via `--run` to the three scripts above, or keep the defaults to rebuild the cited
evidence. `make check-log` and `make check-model-selection` re-derive the logged numbers and the
model choice from the run artefacts and fail on drift.

`make report` needs the video URL because the reference list has to carry it; `make report-draft`
builds a proofreading copy without that reference.

## Layout

```
configs/   run configuration (data.yaml, experiment.yaml)
data/      the processed store is built here; nothing in it is tracked
docs/      DATA_DICTIONARY.md, the raw and processed schemas
experiments/  EXPERIMENT_LOG.md and one directory per run
notebooks/ scratch space, deliberately empty
reports/   figures/, tables/ and the report with its Typst sources
scripts/   one entry point per artefact; `make help` lists them
src/milan_traffic/   the package: ingest, features, models, metrics, evaluate
tests/     283 fast tests on synthetic data, plus one that reads the built store
```

`experiments/EXPERIMENT_LOG.md` carries one row per run, written as each run finished, with the
reasoning behind each change. Every figure and table under `reports/` is written by a script in
`scripts/`.

`make test` (synthetic data, fast), `make test-all`, `make lint`.

## Citation

G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of Milan and the Province
of Trentino," *Scientific Data*, vol. 2, 150055, 2015. doi:10.1038/sdata.2015.55
