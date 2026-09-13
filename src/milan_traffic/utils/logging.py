"""Console + file logging with a consistent format across scripts and notebooks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "milan_traffic", *, level: int = logging.INFO, logfile: str | Path | None = None
) -> logging.Logger:
    """Return a configured logger, optionally mirroring output to ``logfile``."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(stream)

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(fh)

    logger.propagate = False
    return logger
