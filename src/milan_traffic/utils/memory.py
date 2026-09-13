"""Memory accounting.

The assignment asks for *evidence* of memory reduction, not a description of it, so
measurement is a first-class part of the pipeline rather than something bolted on for
the write-up. Resident set size is sampled from a background thread because peak usage
during a chunked read happens between statements, not at them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

try:  # psutil gives a portable RSS; fall back to resource on POSIX if absent
    import psutil

    _PROC: psutil.Process | None = psutil.Process()
except Exception:  # pragma: no cover - exercised only on stripped environments
    psutil = None  # type: ignore[assignment]
    _PROC = None


def rss_mb() -> float:
    """Current resident set size of this process, in MiB."""
    if _PROC is not None:
        return _PROC.memory_info().rss / 2**20
    import resource  # POSIX only

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    import sys

    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def peak_rss_mb() -> float:
    """Peak RSS the OS has observed for this process, in MiB."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def human_bytes(n: float) -> str:
    """Format a byte count for tables and log lines."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PiB"


@dataclass
class MemoryTracker:
    """Sample RSS on a background thread for the duration of a ``with`` block.

    Example
    -------
    >>> with MemoryTracker() as m:
    ...     df = expensive_load()
    >>> m.peak_mb, m.delta_mb
    """

    interval: float = 0.02
    baseline_mb: float = field(default=0.0, init=False)
    peak_mb: float = field(default=0.0, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            time.sleep(self.interval)

    def __enter__(self) -> MemoryTracker:
        self.baseline_mb = rss_mb()
        self.peak_mb = self.baseline_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_mb = max(self.peak_mb, rss_mb())

    @property
    def delta_mb(self) -> float:
        """Peak usage attributable to the block, above the entry baseline."""
        return self.peak_mb - self.baseline_mb
