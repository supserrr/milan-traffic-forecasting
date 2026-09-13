"""Project paths and experiment constants.

Defaults live here so that importing the package never requires a YAML parser; the
values can be overridden from `configs/*.yaml` when one is supplied. Keeping the
canonical constants in one place is what makes "the evaluation week" mean the same
thing in the ingestion code, the notebooks and the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- paths

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

RAW_DIR = (PROJECT_ROOT.parent / "Dataverse").resolve()
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
DAILY_DIR = PROCESSED_DIR / "daily"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"

# ----------------------------------------------------------------------- constants

GRID_SIDE = 100
N_SQUARES = GRID_SIDE * GRID_SIDE  # 10_000 cells
SLOT_MS = 10 * 60 * 1000  # 10-minute sampling interval
SLOTS_PER_DAY = 24 * 60 // 10  # 144
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7  # 1008

TIMEZONE = "Europe/Rome"

ACTIVITIES: tuple[str, ...] = ("sms_in", "sms_out", "call_in", "call_out", "internet")
TARGET = "internet"

#: Raw column order. ``country_code`` is summed out during ingestion.
RAW_COLUMNS: tuple[str, ...] = (
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
)

#: Cells the brief requires in the two-week exploratory figure, alongside the top three.
REQUIRED_EDA_SQUARES: tuple[int, ...] = (4159, 4556)

#: Window the brief fixes for evaluation (inclusive, local time).
EVAL_WEEK_START = "2013-12-16 00:00"
EVAL_WEEK_END = "2013-12-22 23:50"

#: First two weeks, used for the exploratory time-series figure.
EDA_WINDOW_START = "2013-11-01 00:00"
EDA_WINDOW_END = "2013-11-14 23:50"


@dataclass(slots=True)
class DataConfig:
    """Resolved configuration for a run. Override via ``configs/data.yaml``."""

    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    activities: tuple[str, ...] = ACTIVITIES
    target: str = TARGET
    chunksize: int = 1_000_000
    eval_start: str = EVAL_WEEK_START
    eval_end: str = EVAL_WEEK_END
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        import yaml  # imported lazily: ingestion must work without PyYAML

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in cls.__slots__ if f != "extra"}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        for key in ("raw_dir", "processed_dir"):
            if key in kwargs:
                kwargs[key] = Path(kwargs[key]).expanduser()
        kwargs["extra"] = {k: v for k, v in raw.items() if k not in known}
        return cls(**kwargs)


def square_to_rowcol(square_id: int) -> tuple[int, int]:
    """Map a 1-based square id to its (row, col) position on the 100x100 grid."""
    zero_based = square_id - 1
    return divmod(zero_based, GRID_SIDE)


def rowcol_to_square(row: int, col: int) -> int:
    """Inverse of :func:`square_to_rowcol`."""
    return row * GRID_SIDE + col + 1
