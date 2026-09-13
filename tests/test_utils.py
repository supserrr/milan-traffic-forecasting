"""Seeding, timing, memory and logging.

Four small modules that nothing else in the suite exercises, and that underpin three
claims the brief grades directly: that a reported number can be reproduced from its
seed, that the training and inference times are measurements rather than estimates, and
that the memory reduction is evidence rather than description. If these drift, nothing
else in the test suite notices.
"""

from __future__ import annotations

import logging
import platform
import random
from pathlib import Path

import numpy as np
import pytest

from milan_traffic.utils import MemoryTracker, Timer, human_bytes, peak_rss_mb, rss_mb, timed
from milan_traffic.utils.logging import get_logger
from milan_traffic.utils.seed import set_seed
from milan_traffic.utils.timing import hardware_string, median_ms, repeat_timed

# ------------------------------------------------------------------------------- seed


@pytest.fixture
def torch_determinism_restored():
    """Yield torch with its global determinism flags restored afterwards.

    ``set_seed`` flips process-wide switches. Leaving them flipped would make this file
    change how every later test in the session behaves, which is exactly the kind of
    hidden coupling the seeding exists to avoid.
    """
    torch = pytest.importorskip("torch")
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield torch
    if was_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=was_warn_only)
    else:
        torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _draws(torch) -> tuple[float, np.ndarray, np.ndarray]:
    """One sample from each generator ``set_seed`` claims to cover."""
    return (
        random.random(),
        np.random.random(4),  # noqa: NPY002 - legacy global stream is what set_seed seeds
        torch.rand(4).numpy(),
    )


def test_the_same_seed_reproduces_every_generator(torch_determinism_restored):
    torch = torch_determinism_restored

    set_seed(1234)
    first = _draws(torch)
    set_seed(1234)
    second = _draws(torch)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])


def test_different_seeds_give_different_draws(torch_determinism_restored):
    torch = torch_determinism_restored

    set_seed(0)
    first = _draws(torch)
    set_seed(1)
    second = _draws(torch)

    assert first[0] != second[0]
    assert not np.array_equal(first[1], second[1])
    assert not np.array_equal(first[2], second[2])


def test_set_seed_returns_the_seed_for_the_run_record(torch_determinism_restored):
    assert set_seed(7) == 7
    assert set_seed() == 42  # the documented default, recorded in run metadata


def test_set_seed_leaves_a_deterministic_torch_behind(torch_determinism_restored):
    torch = torch_determinism_restored
    set_seed(3, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()  # MPS gaps warn, not raise
    assert torch.backends.cudnn.benchmark is False


# ----------------------------------------------------------------------------- timing


def _burn_cpu(n: int = 200_000) -> int:
    """Enough work that ``process_time`` resolution cannot round it to zero."""
    return sum(i * i for i in range(n))


def test_timer_records_positive_wall_and_cpu_seconds():
    with Timer("fit") as t:
        _burn_cpu()

    assert t.label == "fit"
    assert t.wall_s > 0.0
    assert t.cpu_s > 0.0
    assert t.cpu_s <= t.wall_s * 4  # single-threaded work: CPU time cannot run away
    assert "fit" in str(t) and "wall" in str(t) and "cpu" in str(t)


def test_timer_synchronises_before_reading_the_clock():
    """On an async backend the clock must be read after the device has caught up."""
    calls: list[str] = []

    with Timer("mps-block", sync=lambda: calls.append("sync")):
        calls.append("work")

    assert calls == ["sync", "work", "sync"]


def test_timed_decorator_attaches_the_last_timing():
    @timed
    def train(n: int) -> int:
        return _burn_cpu(n)

    assert train.last_timing is None  # nothing measured until it runs
    out = train(50_000)

    assert out == sum(i * i for i in range(50_000))
    assert isinstance(train.last_timing, Timer)
    assert train.last_timing.label == "train"
    assert train.last_timing.wall_s > 0.0


def test_repeat_timed_reports_only_the_repetitions_not_the_warmup():
    calls = []

    timings = repeat_timed(lambda: calls.append(1), reps=3, warmup=2)

    assert len(calls) == 5  # 2 warm-up + 3 timed
    assert len(timings) == 3  # the warm-up ones are not reported
    assert all(t >= 0.0 for t in timings)


def test_repeat_timed_refuses_to_report_zero_repetitions():
    with pytest.raises(ValueError, match="reps"):
        repeat_timed(lambda: None, reps=0)


def test_median_ms_converts_seconds_and_survives_an_empty_run():
    assert median_ms([0.001, 0.003, 0.002]) == pytest.approx(2.0)
    assert np.isnan(median_ms([]))


def test_hardware_string_names_the_device_the_run_actually_used():
    declared = hardware_string("mps")

    assert declared.startswith(platform.system())
    assert declared.endswith("device:mps")
    # Without a declared device the string may only report what is available, which is a
    # property of the machine and must not be mistaken for what the run used.
    assert "device:" not in hardware_string()


# ----------------------------------------------------------------------------- memory


def test_rss_and_peak_rss_are_positive_and_use_the_same_unit():
    current = rss_mb()
    peak = peak_rss_mb()

    assert current > 0.0
    assert peak > 0.0
    # A unit mix-up (KiB vs bytes vs MiB) would put these three orders of magnitude
    # apart; normal drift between the two measures cannot.
    assert 0.01 < current / peak < 100.0


def test_memory_tracker_sees_an_allocation_made_inside_the_block():
    with MemoryTracker(interval=0.005) as m:
        block = np.ones(8_000_000, dtype=np.float64)  # 64 MiB, every page written
        block[::1000] += 1.0

    assert m.baseline_mb > 0.0
    assert m.peak_mb >= m.baseline_mb
    assert m.delta_mb > 20.0, f"64 MiB allocation showed as {m.delta_mb:.1f} MiB"
    del block


def test_memory_tracker_stops_sampling_at_the_end_of_the_block():
    with MemoryTracker(interval=0.005) as m:
        pass

    assert m._thread is not None and not m._thread.is_alive()
    assert m.delta_mb >= 0.0


def test_human_bytes_scales_through_the_units():
    assert human_bytes(512) == "512.0 B"
    assert human_bytes(1536) == "1.5 KiB"
    assert human_bytes(2**20) == "1.0 MiB"
    assert human_bytes(21.4 * 2**30) == "21.4 GiB"
    assert human_bytes(-(2**20)) == "-1.0 MiB"  # a delta can be negative


# ---------------------------------------------------------------------------- logging


def test_get_logger_mirrors_to_a_file_and_configures_itself_once(tmp_path: Path):
    path = tmp_path / "runs" / "EXP-000" / "run.log"
    logger = get_logger("milan_traffic.test_logging", logfile=path)
    try:
        logger.info("EXP-000 started")
        for handler in logger.handlers:
            handler.flush()

        assert path.exists()  # the parent directories are created for it
        assert "EXP-000 started" in path.read_text(encoding="utf-8")
        assert len(logger.handlers) == 2  # console + file
        assert logger.level == logging.INFO
        assert logger.propagate is False  # no duplicate lines via the root logger

        again = get_logger("milan_traffic.test_logging")
        assert again is logger
        assert len(again.handlers) == 2  # a second call must not stack handlers
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
