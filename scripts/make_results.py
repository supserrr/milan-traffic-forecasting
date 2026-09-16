"""Turn a finished run into the report's result tables and figures.

Reads ``experiments/runs/EXP-NNN/`` and writes, under ``reports/``:

* ``tables/results_<area>.md``, one per area, with the metrics the brief requires
  (MAE, MAPE, RMSE) plus the scale-free companions, for every model and baseline;
* ``tables/timing.md``, the training and execution times with the hardware string and
  the measurement protocol stated beneath it;
* ``figures/actual_vs_predicted.pdf``, the nine superposed panels the brief asks for,
  drawn from one seed (the run's first, seed 42) because a line has to be a forecaster;
* ``figures/error_by_hour.pdf``, mean absolute error by hour of day, the error taken
  per seed and then averaged, which is the evidence for the failure discussion.

Both figures keep the seed convention of the tables and of ``periods.py``: the mean over
seeds of a per-seed statistic, never a statistic of the seed-averaged forecast. Averaging
the forecasts first would draw a three-member ensemble, a fourth forecaster that appears in
no table here and that beats its own tabulated MAE in all nine model-area cells of EXP-008.

Thin CLI: every computation lives in ``milan_traffic``. Run it twice on the same run and
it overwrites its own outputs, because these are derived views rather than run evidence.

    python scripts/make_results.py --run EXP-001
    python scripts/make_results.py --run EXP-004 --baseline-run EXP-001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic import viz  # noqa: E402
from milan_traffic.config import RUNS_DIR, TABLES_DIR  # noqa: E402
from milan_traffic.dataio import load_index, load_series  # noqa: E402
from milan_traffic.evaluate import EVAL_WEEK_END, EVAL_WEEK_START  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402

LOG = get_logger("make_results")

#: Printed names, in the order the report's tables use: trained models first, then the
#: references they are judged against.
DISPLAY = {
    "lstm": "LSTM",
    "tcn": "TCN",
    "gbt": "GBT",  # one name for this model in every table, figure legend and sentence (V.F)
    "naive": "Persistence",
    "seasonal_naive_daily": "Seasonal naive (daily)",
    "seasonal_naive_144": "Seasonal naive (daily)",
    "seasonal_naive_weekly": "Seasonal naive (weekly)",
    "seasonal_naive_1008": "Seasonal naive (weekly)",
    "linear_ar_144": "Linear AR(144)",
    "linear_ar_36": "Linear AR(36)",
    "linear_ar_6": "Linear AR(6)",
    "time_of_day_mean_4d": "Time-of-day mean",
}
ORDER = list(DISPLAY)

#: Position along the seed axis of the single forecast drawn in ``actual_vs_predicted``.
#: Zero is the run's first seed, 42 for EXP-008. One line per panel has to come from one
#: forecaster, and the mean of the three seeds is not one of the three.
FIGURE_SEED_POSITION = 0


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> dict:
    path = Path(runs_dir) / run_id / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; has {run_id} finished?")
    return json.loads(path.read_text())


def frame_from(payload: dict, *, split: str = "test") -> pd.DataFrame:
    """Long table of every (area, model, seed) row of a run."""
    rows = []
    for r in payload["results"]:
        scores = r[split]
        rows.append(
            {
                "area": r["area"],
                "model": r["model"],
                "seed": r.get("seed", 0),
                "n_params": r["n_params"],
                **{k: v for k, v in scores.items() if not k.startswith("n_")},
                "fit_s": r["fit_s"],
                "fit_cpu_s": r.get("fit_cpu_s"),
                "epochs": r.get("n_epochs"),
                "s_per_epoch": r.get("s_per_epoch"),
                "predict_ms": r.get("predict_ms_batch_median", r["predict_s"] * 1e3),
                "latency_ms": r.get("latency_ms_single_median"),
                "device": r.get("device", "cpu"),
            }
        )
    return pd.DataFrame(rows)


def merge_runs(payloads: list[dict], *, split: str = "test") -> pd.DataFrame:
    """One frame from several runs, keeping the first occurrence of each model."""
    frames = [frame_from(p, split=split) for p in payloads]
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["area", "model", "seed"], keep="first")


def per_area_table(frame: pd.DataFrame, area: int) -> str:
    """Markdown table for one area: the brief's three metrics plus the companions."""
    sub = frame[frame["area"] == area].copy()
    sub = sub.groupby("model", as_index=False).mean(numeric_only=True)
    sub["order"] = sub["model"].map(lambda m: ORDER.index(m) if m in ORDER else 99)
    sub = sub.sort_values("order")
    cols = ["MAE", "MAPE", "RMSE", "WAPE", "MASE", "R2"]
    best = {c: (sub[c].max() if c == "R2" else sub[c].min()) for c in cols}

    head = "| Model | " + " | ".join(["MAE", "MAPE (%)", "RMSE", "WAPE (%)", "MASE", "R2"]) + " |"
    rule = "|---" * 7 + "|"
    lines = [head, rule]
    for _, r in sub.iterrows():
        cells = []
        for c in cols:
            value = f"{r[c]:.2f}" if c not in {"MASE", "R2"} else f"{r[c]:.4f}"
            cells.append(f"**{value}**" if np.isclose(r[c], best[c]) else value)
        lines.append(f"| {DISPLAY.get(r['model'], r['model'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def timing_table(frame: pd.DataFrame, payload: dict) -> str:
    """Training and execution time, averaged across areas, with the protocol stated."""
    trained = frame[frame["epochs"].notna()]
    rows = []
    for model, g in frame.groupby("model", sort=False):
        t = trained[trained["model"] == model]
        rows.append(
            {
                "order": ORDER.index(model) if model in ORDER else 99,
                "Model": DISPLAY.get(model, model),
                "Parameters": int(g["n_params"].mean()),
                "Device": g["device"].iloc[0],
                "Epochs": f"{t['epochs'].mean():.0f}" if len(t) else "-",
                "s / epoch": f"{t['s_per_epoch'].mean():.2f}" if len(t) else "-",
                "Training (s)": f"{g['fit_s'].mean():.2f}",
                "Training CPU (s)": f"{g['fit_cpu_s'].mean():.2f}",
                "Batch (ms)": f"{g['predict_ms'].mean():.2f}",
                "Latency (ms)": f"{g['latency_ms'].mean():.3f}",
            }
        )
    table = pd.DataFrame(rows).sort_values("order").drop(columns="order")
    proto = payload.get("timing_protocol", {})
    reps = payload.get("config", {}).get("evaluation", {})
    note = (
        f"\n\n**Hardware.** {payload['hardware']}\n\n"
        "**Method.** Training time is the wall time of one fit call, reported with the "
        "number of epochs actually run under early stopping and the mean seconds per "
        "epoch, because a bare wall time under early stopping is not representative of "
        "the model. Execution time is reported twice: as the median of "
        f"{reps.get('timing_reps', 20)} repetitions of predicting the whole 1008-slot "
        "evaluation week after one untimed warm-up call, and as single-window latency, "
        f"the median of {reps.get('latency_reps', 50)} single-window predictions. The "
        "device is synchronised before every timer stops. "
        + str(proto.get("why", ""))
        + " Values are averaged over the three areas.\n"
    )
    return _markdown(table) + note


def _markdown(table: pd.DataFrame) -> str:
    """Pipe-delimited markdown without the optional tabulate dependency."""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="the run holding the trained models, e.g. EXP-004")
    p.add_argument(
        "--baseline-run",
        default=None,
        help="an earlier run whose baselines should be merged in, e.g. EXP-001",
    )
    p.add_argument("--models", nargs="*", default=None, help="restrict the figures to these models")
    args = p.parse_args(argv)

    payloads = [load_run(args.run)]
    if args.baseline_run:
        payloads.append(load_run(args.baseline_run))
    frame = merge_runs(payloads)
    areas = sorted(frame["area"].unique())
    LOG.info("areas %s, models %s", areas, sorted(frame["model"].unique()))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for area in areas:
        text = (
            f"# Evaluation week results, square {area}\n\n"
            f"One-step-ahead forecasts of Internet activity, "
            f"{EVAL_WEEK_START} to {EVAL_WEEK_END} (1008 slots), all models fed the "
            f"identical 144-slot window. Best value in each column in bold. "
            f"Source: `experiments/runs/{args.run}/metrics.json`.\n\n"
            + per_area_table(frame, area)
            + "\n"
        )
        path = TABLES_DIR / f"results_{area}.md"
        path.write_text(text)
        LOG.info("wrote %s", path)

    timing = TABLES_DIR / "timing.md"
    timing.write_text(
        f"# Training and execution time\n\nSource: `experiments/runs/{args.run}/`.\n\n"
        + timing_table(frame, payloads[0])
    )
    LOG.info("wrote %s", timing)

    # --- figures -------------------------------------------------------------------
    viz.use_style()
    index = load_index()
    week = slice(*_eval_bounds(index))
    truth = {a: load_series(a).to_numpy()[week] for a in areas}

    models = args.models or [m for m in ORDER if m in set(frame["model"])][:3]
    preds, seeds = _predictions(payloads, areas, models)
    Y = np.stack([truth[a] for a in areas])

    # One drawn line is one forecaster, so the panels show a single seed. The seed mean
    # would be an ensemble, and on EXP-008 it beats its own tabulated MAE in all nine cells.
    drawn = {m: seeds[m][FIGURE_SEED_POSITION] for m in models}
    labelled = {DISPLAY.get(m, m): preds[m][:, FIGURE_SEED_POSITION] for m in models}
    fig = viz.plot_actual_vs_predicted(Y, labelled, index[week], areas)
    viz.save_figure(fig, "actual_vs_predicted")
    LOG.info("wrote reports/figures/actual_vs_predicted.{pdf,png}, seed drawn %s", drawn)

    # Take the error per seed and average the three, so the plotted curve is the mean over
    # seeds of the per-seed hourly MAE. plot_error_by_hour takes the absolute value itself,
    # which leaves an already non-negative array alone, and averaging over the slots of an
    # hour commutes with averaging over seeds.
    abs_error = {
        DISPLAY.get(m, m): np.nanmean(np.abs(Y[:, None, :] - preds[m]), axis=1) for m in models
    }
    fig = viz.plot_error_by_hour(abs_error, index[week], y_true=Y)
    viz.save_figure(fig, "error_by_hour")
    LOG.info("wrote reports/figures/error_by_hour.{pdf,png}, seeds averaged %s", seeds)
    return 0


def _eval_bounds(index: pd.DatetimeIndex) -> tuple[int, int]:
    mask = (index >= pd.Timestamp(EVAL_WEEK_START, tz=index.tz)) & (
        index <= pd.Timestamp(EVAL_WEEK_END, tz=index.tz)
    )
    pos = np.flatnonzero(mask)
    return int(pos[0]), int(pos[-1]) + 1


def _predictions(
    payloads: list[dict], areas: list[int], models: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """``{model: array(n_areas, n_seeds, 1008)}`` with ``{model: [seed, ...]}`` beside it.

    The seeds are kept apart rather than averaged here. A forecast averaged over seeds is a
    three-member ensemble, which is a fourth forecaster: averaging cancels part of the
    seed-to-seed spread, so its MAE sits below the mean over seeds that the tables report,
    and a figure built on it would show a better forecaster than the one being reported.
    Each caller reduces over the seed axis the way its statistic requires, or picks a seed.
    """
    out: dict[str, np.ndarray] = {}
    seeds: dict[str, list[int]] = {}
    for payload in payloads:
        run_dir = RUNS_DIR / payload["run_id"]
        stack = np.load(run_dir / "predictions.npy")
        p_areas, p_models = payload["areas"], payload["models"]
        p_seeds = payload.get("seeds", [0])
        for m in models:
            if m in out or m not in p_models:
                continue
            mi = p_models.index(m)
            rows = []
            for a in areas:
                if a not in p_areas:
                    break
                ai = p_areas.index(a)
                block = stack[ai, mi] if stack.ndim == 4 else stack[ai, mi][None, :]
                rows.append(block[: len(p_seeds)])
            if len(rows) == len(areas):
                out[m] = np.stack(rows)
                seeds[m] = list(p_seeds)
    missing = [m for m in models if m not in out]
    if missing:
        raise KeyError(f"no predictions for {missing} in the runs given")
    return out, seeds


if __name__ == "__main__":
    raise SystemExit(main())
