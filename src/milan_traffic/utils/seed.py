"""Deterministic seeding.

Every reported number should be reproducible from its run config, so seeding covers the
Python, NumPy and (when present) PyTorch generators in one call. Determinism on MPS/CUDA
is best-effort: the flags below remove the easy sources of run-to-run variation, not all
of them, which is why the experiment protocol asks for several seeds rather than one.

``PYTHONHASHSEED`` is deliberately *not* set here. Hash randomisation is fixed when the
interpreter starts, so assigning the variable from inside a running process changes
nothing in that process; it would only record an intention. Nothing in this project
iterates over a set of strings in an order that reaches a reported number, so the
variable is left alone rather than documented as doing something it does not.
"""

from __future__ import annotations

import random


def set_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch. Returns the seed, for logging."""
    random.seed(seed)

    try:
        import numpy as np

        # Legacy global seeding is deliberate: it is what makes any third-party code
        # calling np.random.* reproducible. A Generator would only seed its own stream.
        np.random.seed(seed)  # noqa: NPY002
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only: an op without a deterministic kernel on the current device
            # (several exist on MPS) logs a warning instead of raising, so a run still
            # completes and the warning is the record that bitwise repeatability was
            # not guaranteed there.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed
