"""Shared fixtures.

Tests run against synthetic data only, so the suite is fast and works on a clean clone
with no dataset present. Anything that needs the real 21 GB is marked ``slow``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic.config import SLOT_MS, SLOTS_PER_DAY  # noqa: E402
from milan_traffic.ingest import day_start_ms  # noqa: E402


@pytest.fixture
def tiny_raw_file(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A miniature raw file with a known answer.

    Three squares x 144 slots x 2 country codes, plus deliberately missing fields, so
    the reducer's aggregation and NaN handling can both be checked against a value
    computed independently.
    """
    day = date(2013, 11, 1)
    origin = day_start_ms(day)
    rng = np.random.default_rng(0)

    expected = np.zeros((SLOTS_PER_DAY, 3), dtype=np.float64)
    lines = []
    for square in (1, 2, 3):
        for slot in range(SLOTS_PER_DAY):
            ts = origin + slot * SLOT_MS
            for country in (0, 39):
                internet = float(rng.uniform(0, 10))
                expected[slot, square - 1] += internet
                # sms_in missing on every other row, to exercise NaN handling
                sms_in = "" if (slot + country) % 2 == 0 else "1.5"
                lines.append(f"{square}\t{ts}\t{country}\t{sms_in}\t\t\t\t{internet!r}")

    path = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    path.write_text("\n".join(lines) + "\n")
    return path, expected


@pytest.fixture
def seasonal_series() -> np.ndarray:
    """A daily-seasonal series with noise: 14 days at 10-minute resolution."""
    t = np.arange(14 * SLOTS_PER_DAY)
    daily = 50 + 40 * np.sin(2 * np.pi * t / SLOTS_PER_DAY - np.pi / 2)
    weekly = 8 * np.sin(2 * np.pi * t / (7 * SLOTS_PER_DAY))
    noise = np.random.default_rng(1).normal(0, 2.0, size=t.size)
    return (daily + weekly + noise).astype(np.float32)
