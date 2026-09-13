"""Wall-clock and CPU timing.

The brief requires "exact statistics on the training and execution time of each model,
with details on the process used to compute such statistics and on the hardware". Both
wall and CPU time are recorded so that multi-threaded BLAS work is visible, and the
hardware string is captured alongside them so a table can never drift from the machine
that produced it.

Single-shot timings were found to vary five-fold for identical operations in EXP-000,
and the first call on an MPS device at a new tensor shape is roughly thirty times slower
than a warm one. :func:`repeat_timed` therefore exists to take a warm-up call and then a
run of repetitions whose median is what gets reported; the caller decides how many.
"""

from __future__ import annotations

import functools
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _cpu_brand() -> str | None:
    """The chip's marketing name, or ``None`` where the platform does not expose one.

    Audit note, 2026-09-18. Every run's ``metrics.json`` recorded the machine as
    ``Darwin | arm64 | 8P/8L cores | 16 GiB RAM``, while the report and the README name the
    chip as an Apple M1 Pro. Both are true and neither artefact carried the other's wording,
    so the chip name could not be checked against a generated file. ``sysctl`` supplies it on
    macOS; elsewhere this returns ``None`` and the string is unchanged. Runs recorded before
    this date keep their original hardware strings, since the runs are append-only history.
    """
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except Exception:  # pragma: no cover - absent or slow sysctl is not worth failing a run
        return None
    return out.stdout.strip() or None


def hardware_string(device: str | None = None) -> str:
    """One-line description of the machine, for inclusion in results tables.

    ``device`` names the compute device a run actually used (``"mps"``, ``"cpu"``, or a
    comma-joined list when models differ). When it is given the string ends in
    ``device:<name>``; when it is omitted the string falls back to reporting which torch
    backend is *available*, which is a property of the machine rather than of the run
    and should not be read as "this run used it".
    """
    bits = [platform.system(), platform.release(), platform.machine()]
    cpu = _cpu_brand()
    if cpu:
        bits.append(cpu)
    try:
        import psutil

        bits.append(f"{psutil.cpu_count(logical=False) or '?'}P/{psutil.cpu_count()}L cores")
        bits.append(f"{psutil.virtual_memory().total / 2**30:.0f} GiB RAM")
    except Exception:  # pragma: no cover
        pass
    if device is not None:
        bits.append(f"device:{device}")
        return " | ".join(str(b) for b in bits)
    try:
        import torch

        if torch.backends.mps.is_available():
            bits.append("torch:mps")
        elif torch.cuda.is_available():
            bits.append(f"torch:cuda({torch.cuda.get_device_name(0)})")
        else:
            bits.append("torch:cpu")
    except Exception:
        pass
    return " | ".join(str(b) for b in bits)


@dataclass
class Timer(ContextDecorator):
    """Context manager and decorator recording wall and CPU seconds.

    ``sync`` is called immediately before the clocks are read on exit. Pass the device
    synchronisation call for asynchronous backends (``torch.mps.synchronize``), or the
    timer stops when the work was *queued* rather than when it finished.
    """

    label: str = "block"
    sync: Callable[[], None] | None = field(default=None, repr=False)
    wall_s: float = field(default=0.0, init=False)
    cpu_s: float = field(default=0.0, init=False)
    _t0: float = field(default=0.0, init=False, repr=False)
    _c0: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> Timer:
        if self.sync is not None:
            self.sync()
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.sync is not None:
            self.sync()
        self.wall_s = time.perf_counter() - self._t0
        self.cpu_s = time.process_time() - self._c0

    def __str__(self) -> str:
        return f"{self.label}: {self.wall_s:.3f}s wall / {self.cpu_s:.3f}s cpu"


def repeat_timed(
    fn: Callable[[], Any],
    reps: int,
    *,
    warmup: int = 1,
    sync: Callable[[], None] | None = None,
) -> list[float]:
    """Wall seconds of ``reps`` calls to ``fn`` after ``warmup`` untimed calls.

    Returns the individual timings so the caller can report a median and keep the
    spread. ``sync`` is passed to :class:`Timer` and runs before each stop.
    """
    if reps < 1:
        raise ValueError("reps must be at least 1.")
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(reps):
        with Timer(sync=sync) as t:
            fn()
        out.append(t.wall_s)
    return out


def median_ms(seconds: list[float]) -> float:
    """Median of a list of wall seconds, in milliseconds."""
    return statistics.median(seconds) * 1e3 if seconds else float("nan")


def timed(fn: F) -> F:
    """Decorator attaching ``.last_timing`` (a :class:`Timer`) to the wrapped function."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer(fn.__name__) as t:
            out = fn(*args, **kwargs)
        wrapper.last_timing = t  # type: ignore[attr-defined]
        return out

    wrapper.last_timing = None  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
