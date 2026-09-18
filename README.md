# Forecasting Mobile Network Traffic in Milan

Comparative study of three sequential models for **one-step-ahead** forecasting of mobile Internet
traffic, on the Telecom Italia Big Data Challenge dataset for Milan (Nov 2013 - Jan 2014).

> How do different sequential models compare for one-step-ahead mobile network traffic
> forecasting, and how does their performance vary across geographical areas with different
> traffic characteristics?

Report: [`reports/report/report.pdf`](reports/report/report.pdf).

## Setup

Python >= 3.10, ~1 GB RAM, ~1.7 GB disk. No GPU required.

```bash
make setup && source .venv/bin/activate
```

`requirements-lock.txt` pins the environment every reported number was produced in.

## Data

The raw data is not in this repository (~21 GB, 62 daily files). Download
`sms-call-internet-mi-YYYY-MM-DD.txt` (2013-11-01 -> 2014-01-01) from
[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
into a `Dataverse/` directory next to this repository, then:

```bash
make ingest-smoke     # 3 days, ~15 s - checks the pipeline
make ingest           # all 62 days, ~2.2 min, resumable
```

Ingestion streams each file in chunks into a preallocated array, reducing 19.4 GiB of text and
319.9 M rows to a memory-mapped 8928 x 10000 float32 matrix (357 MB) at 377.9 MB peak RSS.
Schemas: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Reproducing the results

```bash
make bench tsa eda-stats figures          # evidence tables and figures
python scripts/run_baselines.py --extra-areas --run-id EXP-000
python scripts/run_experiment.py --models lstm tcn gbt \
    --baselines naive seasonal_naive_daily linear_ar_144 \
    --seq-len 144 --scaling log1p-standard --seeds 42 1337 2024
python scripts/make_results.py --run EXP-008
make significance period-breakdown capacity-table
```

The reported run is **EXP-008** (three architectures, three areas, three seeds; ~20 min on an
M1 Pro). Its predictions are tracked, so every table and figure regenerates without retraining.
`run_experiment.py` claims the next free id, so on a clean clone training writes EXP-019: pass that
id via `--run` to the three scripts above, or keep the defaults to rebuild the cited evidence.
`make check-log` and `make check-model-selection` re-derive the logged numbers and the model
choice. `make help` lists every target.

## Layout

```
configs/  docs/  experiments/  reports/  scripts/  src/milan_traffic/  tests/
```

`make test` (synthetic data, fast), `make test-all`, `make lint`.

## Citation

G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of Milan and the Province
of Trentino," *Scientific Data*, vol. 2, 150055, 2015. doi:10.1038/sdata.2015.55
