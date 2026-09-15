"""One training run: fit the selected models on the selected areas and write EXP-NNN.

Thin CLI. Every default comes from ``configs/experiment.yaml`` so that a run really is
reproducible from its config rather than from whatever the flags happened to be, and the
resolved configuration, including the command line itself, is written back into the run
directory before training starts.

    python scripts/run_experiment.py --notes "EXP-001: LSTM at seq_len 144, standard scaling"
    python scripts/run_experiment.py --models gbt --areas 5161 --dry-run
    python scripts/run_experiment.py --models lstm tcn gbt --seeds 42 1337 2024

The run id is allocated automatically and the directory is claimed before the first fit,
so a collision fails in the first second rather than after twenty minutes of training.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, get_args

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic.config import PROJECT_ROOT, SLOTS_PER_DAY  # noqa: E402
from milan_traffic.dataio import top_squares  # noqa: E402
from milan_traffic.evaluate import (  # noqa: E402
    VAL_FRAC,
    aggregate_seeds,
    log_row,
    next_run_id,
    prepare_area,
    reserve_run,
    results_frame,
    run_model,
    write_run,
)
from milan_traffic.features import ScaleMethod  # noqa: E402
from milan_traffic.models import build  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402

LOG = get_logger("run_experiment")

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

#: Flags that reach the model constructor, per model. Anything not listed here is a
#: property of the run rather than of a model, and is recorded in the config instead.
#:
#: The last four were added on 2026-09-18. The gradient clip, the early-stopping tolerance
#: and the learning-rate schedule were active in every neural fit in this repository, at the
#: values ``TorchForecaster`` defaults to, but they were not on this list, so no run
#: directory recorded them and editing ``configs/experiment.yaml`` could not change them.
#: The report discloses the schedule, so the artefact has to carry it. The defaults below
#: are the values the runs already used, so nothing about EXP-000 to EXP-018 changes.
TORCH_KEYS = (
    "loss",
    "lr",
    "batch_size",
    "max_epochs",
    "patience",
    "device",
    "grad_clip",
    "min_delta",
    "scheduler",
    "scheduler_patience",
)
MODEL_KEYS: dict[str, tuple[str, ...]] = {
    "lstm": (*TORCH_KEYS, "hidden", "layers", "dropout"),
    "tcn": (*TORCH_KEYS, "channels", "levels", "kernel_size", "dropout"),
    "gbt": ("max_iter", "learning_rate", "max_leaf_nodes"),
}


def load_defaults(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Defaults from ``configs/experiment.yaml``, flattened to the CLI's names."""
    if not path.exists():  # pragma: no cover - the file ships with the repository
        return {}
    cfg = yaml.safe_load(path.read_text()) or {}
    data, training, opt = cfg.get("data", {}), cfg.get("training", {}), {}
    opt = training.get("optimizer", {}) or {}
    early = training.get("early_stopping", {}) or {}
    sched = training.get("lr_scheduler", {}) or {}
    return {
        "seq_len": data.get("seq_len", SLOTS_PER_DAY),
        "scaling": data.get("scaling", "standard"),
        "batch_size": training.get("batch_size", 128),
        "max_epochs": training.get("max_epochs", 100),
        "patience": early.get("patience", 10),
        "min_delta": early.get("min_delta", 1e-4),
        "loss": training.get("loss", "mse"),
        "lr": opt.get("lr", 1e-3),
        "grad_clip": training.get("grad_clip", 1.0),
        "scheduler": sched.get("name", "reduce_on_plateau"),
        "scheduler_patience": sched.get("patience", 5),
        "seeds": cfg.get("run", {}).get("seeds", [42]),
        "device": cfg.get("run", {}).get("device", "auto"),
    }


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", default=None, help="default: the next free EXP-NNN")
    p.add_argument("--models", nargs="+", default=["lstm", "tcn", "gbt"])
    p.add_argument("--baselines", nargs="*", default=[], help="reference models to run alongside")
    p.add_argument("--areas", type=int, nargs="*", default=None, help="default: top 3 by traffic")
    p.add_argument("--seq-len", type=int, default=defaults.get("seq_len", SLOTS_PER_DAY))
    p.add_argument(
        "--scaling",
        default=defaults.get("scaling", "standard"),
        choices=list(get_args(ScaleMethod)),
    )
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--seeds", type=int, nargs="+", default=[defaults.get("seeds", [42])[0]])
    p.add_argument(
        "--no-calendar",
        action="store_true",
        help="ablate the five target-slot calendar features for every model",
    )
    p.add_argument("--loss", default=defaults.get("loss", "mse"), choices=["mse", "mae", "huber"])
    p.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3))
    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 128))
    p.add_argument("--max-epochs", type=int, default=defaults.get("max_epochs", 100))
    p.add_argument("--patience", type=int, default=defaults.get("patience", 10))
    p.add_argument("--min-delta", type=float, default=defaults.get("min_delta", 1e-4))
    p.add_argument("--grad-clip", type=float, default=defaults.get("grad_clip", 1.0))
    p.add_argument(
        "--scheduler",
        default=defaults.get("scheduler", "reduce_on_plateau"),
        choices=["reduce_on_plateau", "none"],
    )
    p.add_argument("--scheduler-patience", type=int, default=defaults.get("scheduler_patience", 5))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gbt-max-iter", type=int, default=300)
    p.add_argument("--gbt-lr", type=float, default=0.05)
    p.add_argument("--gbt-leaves", type=int, default=31)
    p.add_argument("--device", default=defaults.get("device", "auto"))
    p.add_argument("--timing-reps", type=int, default=20)
    p.add_argument("--latency-reps", type=int, default=50)
    p.add_argument("--notes", default="", help="why this run exists; copied into the log")
    p.add_argument("--dry-run", action="store_true", help="print results, write nothing")
    return p


def model_kwargs(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """The subset of the flags that belongs to ``name``'s constructor."""
    source = {
        "loss": args.loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "device": args.device,
        "min_delta": args.min_delta,
        "grad_clip": args.grad_clip,
        "scheduler": None if args.scheduler == "none" else args.scheduler,
        "scheduler_patience": args.scheduler_patience,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "channels": args.channels,
        "levels": args.levels,
        "kernel_size": args.kernel_size,
        "max_iter": args.gbt_max_iter,
        "learning_rate": args.gbt_lr,
        "max_leaf_nodes": args.gbt_leaves,
    }
    return {k: source[k] for k in MODEL_KEYS.get(name, ()) if k in source}


def main(argv: list[str] | None = None) -> int:
    defaults = load_defaults()
    args = build_parser(defaults).parse_args(argv)

    run_id = args.run_id or next_run_id()
    areas = args.areas or top_squares(3)
    models = [*args.models, *args.baselines]
    calendar = not args.no_calendar

    config = {
        "run": {
            "id": run_id,
            "seeds": args.seeds,
            "device": args.device,
            "notes": args.notes,
            "cli": " ".join(sys.argv[1:]),
            "config_source": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        },
        "data": {
            "areas": list(areas),
            "activity": "internet",
            "seq_len": args.seq_len,
            "horizon": 1,
            "scaling": args.scaling,
            "val_frac": args.val_frac,
            "calendar_features": calendar,
            "test_window": "eval_week",
        },
        "models": {name: model_kwargs(name, args) for name in models},
        "training": {
            "loss": args.loss,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "early_stopping": {"monitor": "val_loss", "patience": args.patience},
        },
        "evaluation": {
            "metrics": ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE", "R2"],
            "mape_mode": "epsilon",
            "mase_scale": "training period (Hyndman)",
            "season": SLOTS_PER_DAY,
            "timing_reps": args.timing_reps,
            "latency_reps": args.latency_reps,
        },
    }

    reserved: Path | None = None
    if not args.dry_run:
        # Claim the directory before the first fit, so a collision costs a second and not
        # a whole training run, and an interrupted run still leaves its config behind.
        reserved = reserve_run(run_id, config)

    try:
        results = []
        for square_id in areas:
            area = prepare_area(
                square_id,
                seq_len=args.seq_len,
                scaling=args.scaling,
                val_frac=args.val_frac,
                calendar=calendar,
            )
            LOG.info(area.describe())
            for name in models:
                for seed in args.seeds:
                    kwargs = model_kwargs(name, args)
                    result = run_model(
                        name,
                        lambda name=name, kwargs=kwargs: build(name, **kwargs),
                        area,
                        seed=seed,
                        n_timing_reps=args.timing_reps,
                        n_latency_reps=args.latency_reps,
                    )
                    results.append(result)
                    LOG.info(
                        "  %-6s seed %-5d test MAE %9.2f  RMSE %9.2f  R2 %.4f  "
                        "fit %6.1fs (%s epochs)  predict %6.2f ms",
                        name,
                        seed,
                        result.test["MAE"],
                        result.test["RMSE"],
                        result.test["R2"],
                        result.fit_s,
                        result.n_epochs if result.n_epochs is not None else "-",
                        result.predict_ms_batch_median,
                    )
    except BaseException:
        if reserved is not None:
            LOG.error(
                "run failed; %s keeps its config so the attempt stays on the record", reserved
            )
        raise

    print(results_frame(results).to_string(index=False))
    if len(args.seeds) > 1:
        print("\nmean and std over seeds:")
        print(aggregate_seeds(results).to_string(index=False))

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    out = write_run(run_id, results, config=config, notes=args.notes or __doc__)
    print(f"\nwrote {out}")
    print("\nEXPERIMENT_LOG.md row:\n")
    print(log_row(run_id, date.today().isoformat(), results, changed=args.notes or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
