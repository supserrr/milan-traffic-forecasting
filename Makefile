# Convenience targets. Everything here is also runnable directly with python.
#
# PY bootstraps the virtualenv and is used for nothing else. pyproject requires Python
# >= 3.10, and on macOS bare `python3` is usually the 3.9 system build, so an explicit
# modern interpreter is preferred when one is on PATH. Override with `make setup PY=...`.
# Every other target runs VPY, the venv's own interpreter, so a target can never silently
# execute against system site-packages.
PY      ?= $(shell command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)
VENV    ?= .venv
BIN      = $(VENV)/bin
VPY      = $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup check-venv ingest ingest-smoke ingest-table bench eda-stats section4-checks tsa figures significance period-breakdown capacity-table check-log check-model-selection report report-draft test test-all lint format clean clean-processed

help:  ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

setup:  ## create .venv and install dependencies
	@$(PY) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
	  || { echo "$(PY) is $$($(PY) -V 2>&1), but pyproject requires >= 3.10."; \
	       echo "Install a newer Python and retry, e.g. make setup PY=python3.11"; exit 1; }
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .
	@echo "\nActivate with: source $(VENV)/bin/activate"

check-venv:
	@test -x $(VPY) || { echo "No virtualenv at $(VENV). Run 'make setup' first."; exit 1; }

ingest: check-venv  ## build data/processed/ from ../Dataverse (resumes from cache)
	$(VPY) scripts/run_ingest.py

ingest-smoke: check-venv  ## ingest three days only, to check the pipeline end to end
	$(VPY) scripts/run_ingest.py --limit 3

ingest-table: check-venv  ## full-run ingestion evidence -> reports/tables/ingest_scale.md
	$(VPY) scripts/make_ingest_table.py

bench: check-venv  ## memory benchmark -> reports/tables/memory_benchmark.md
	$(VPY) scripts/benchmark_memory.py

eda-stats: check-venv  ## exploratory summary statistics -> reports/tables/eda_stats.md
	$(VPY) scripts/make_eda_stats.py

section4-checks: check-venv  ## the Section 4 claims no other table carries -> reports/tables/section4_checks.md
	$(VPY) scripts/make_section4_checks.py

tsa: check-venv  ## time-series diagnostics -> reports/tables/tsa_5161.md + 3 figures
	$(VPY) scripts/run_tsa.py

figures: check-venv  ## regenerate every exploratory figure -> reports/figures/
	$(VPY) scripts/make_figures.py

significance: check-venv  ## predictive-accuracy tests -> reports/tables/significance_dm.md + figure
	$(VPY) scripts/make_significance.py

period-breakdown: check-venv  ## per-period failure evidence -> reports/tables/per_period_*.md + figure
	$(VPY) scripts/make_period_breakdown.py

capacity-table: check-venv  ## capacity grid EXP-010..018 -> reports/tables/capacity_grid.md
	$(VPY) scripts/make_capacity_table.py

check-log: check-venv  ## re-derive EXPERIMENT_LOG.md's numbers from metrics.json (fails on drift)
	$(VPY) scripts/check_log_numbers.py

check-model-selection: check-venv  ## top-cell ranking + SARIMAX cost -> reports/tables/model_selection_checks.md
	$(VPY) scripts/check_model_selection.py

# The video URL is a build input rather than a string in the sources, so the reference
# list cannot be published with a stand-in inside it. `report` refuses to build without
# one; `report-draft` omits reference [28] altogether instead of printing a placeholder.
VIDEO_URL ?=

report: check-venv  ## build reports/report/report.pdf (requires VIDEO_URL=...)
	@test -n "$(VIDEO_URL)" || { \
	  echo "VIDEO_URL is not set, and the submitted report has to cite the presentation."; \
	  echo "Upload the video, then:  make report VIDEO_URL=https://..."; \
	  echo "For a proofreading copy without that reference:  make report-draft"; \
	  exit 1; }
	$(VPY) scripts/build_report.py --video "$(VIDEO_URL)"

report-draft: check-venv  ## build the report with no video reference, for proofreading only
	$(VPY) scripts/build_report.py

test: check-venv  ## run the test suite (skips anything touching raw data)
	$(VPY) -m pytest -m "not slow"

test-all: check-venv  ## run every test, including the one that reads the built store
	$(VPY) -m pytest

lint: check-venv  ## ruff + black --check
	$(VPY) -m ruff check src scripts tests
	$(VPY) -m black --check src scripts tests

format: check-venv  ## apply black and ruff --fix
	$(VPY) -m black src scripts tests
	$(VPY) -m ruff check --fix src scripts tests

clean:  ## remove caches and build artefacts
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info

clean-processed:  ## delete the processed store (forces a full re-ingest)
	rm -rf data/processed/*.npy data/processed/*.csv data/processed/*.json data/processed/daily
