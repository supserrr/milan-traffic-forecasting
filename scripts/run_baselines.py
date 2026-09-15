"""EXP-000: measure the baseline error floor on the fixed evaluation week.

Thin CLI. All logic lives in ``milan_traffic.evaluate`` and ``milan_traffic.models``.

    python scripts/run_baselines.py                 # top-3 areas, writes EXP-000
    python scripts/run_baselines.py --areas 5161 --run-id EXP-000b --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import get_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic.config import REQUIRED_EDA_SQUARES, SLOTS_PER_DAY, SLOTS_PER_WEEK  # noqa: E402
from milan_traffic.dataio import top_squares  # noqa: E402
from milan_traffic.evaluate import prepare_area, results_frame, run_model, write_run  # noqa: E402
from milan_traffic.features import ScaleMethod  # noqa: E402
from milan_traffic.models.baselines import (  # noqa: E402
    LinearAR,
    Persistence,
    SeasonalNaive,
    TimeOfDayMean,
)
from milan_traffic.utils.logging import get_logger  # noqa: E402
from milan_traffic.utils.seed import set_seed  # noqa: E402

LOG = get_logger("run_baselines")

#: Every baseline is fed one 1008-step window so the weekly lag is reachable and the
#: input representation is identical across models.
SEQ_LEN = SLOTS_PER_WEEK

FACTORIES = {
    "naive": Persistence,
    "seasonal_naive_144": lambda: SeasonalNaive(season=SLOTS_PER_DAY),
    "seasonal_naive_1008": lambda: SeasonalNaive(season=SLOTS_PER_WEEK),
    "time_of_day_mean_4d": lambda: TimeOfDayMean(season=SLOTS_PER_DAY, n_seasons=4),
    "linear_ar_6": lambda: LinearAR(order=6),
    "linear_ar_36": lambda: LinearAR(order=36),
    "linear_ar_144": lambda: LinearAR(order=144),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="EXP-000")
    parser.add_argument(
        "--areas",
        type=int,
        nargs="*",
        default=None,
        help="square ids; default is the top 3 by total Internet traffic",
    )
    parser.add_argument(
        "--extra-areas",
        action="store_true",
        help="also run squares 4159 and 4556 for cross-area context",
    )
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--scaling", default="none", choices=list(get_args(ScaleMethod)))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="print results, write nothing")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    areas = args.areas or top_squares(3)
    if args.extra_areas:
        areas = list(dict.fromkeys([*areas, *REQUIRED_EDA_SQUARES]))

    results = []
    for square_id in areas:
        area = prepare_area(
            square_id, seq_len=args.seq_len, scaling=args.scaling, val_frac=args.val_frac
        )
        LOG.info(area.describe())
        for name, factory in FACTORIES.items():
            results.append(run_model(name, factory, area))
            LOG.info(
                "  %-20s test MAE %9.2f  RMSE %9.2f  R2 %.4f",
                name,
                results[-1].test["MAE"],
                results[-1].test["RMSE"],
                results[-1].test["R2"],
            )

    print(results_frame(results).to_string(index=False))
    if args.dry_run:
        return 0

    config = {
        "run": {
            "id": args.run_id,
            "seed": args.seed,
            "device": "cpu",
            "notes": "Baseline floor. No trained sequence model yet.",
        },
        "data": {
            "areas": areas,
            "activity": "internet",
            "seq_len": args.seq_len,
            "horizon": 1,
            "scaling": args.scaling,
            "val_frac": args.val_frac,
            "test_window": "eval_week",
        },
        "models": sorted(FACTORIES),
        "evaluation": {
            "metrics": ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE", "R2"],
            "mape_mode": "epsilon",
            "mase_scale": "training period (Hyndman)",
            "season": SLOTS_PER_DAY,
        },
    }
    out = write_run(args.run_id, results, config=config, notes=__doc__)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
