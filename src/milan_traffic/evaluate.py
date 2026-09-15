"""Run a set of forecasters over a set of areas and write the run's evidence.

The experiment protocol asks every run to leave behind a config, a metrics file, its
predictions and a log, and to do so *as the run finishes*. That is only realistic if
producing those artefacts is one function call, so it lives here rather than in a script
or a notebook: the same path produces EXP-000's baselines and every model run after it,
which is also what keeps the comparison honest. A difference between two rows of the
results table can then only come from the model, never from how it was fed or scored.

Four properties of this module exist because their absence cost something, and each is
worth a sentence in the report.

*The window layout.* Training and validation windows are built strictly inside their own
split (``features.split_windows``), so no window straddles a boundary. The evaluation
window is different: at forecast time the history immediately before the evaluation week
is legitimately available, so test windows reach back into it (``features.context_windows``)
and every one of the 1008 slots gets a prediction while the targets stay inside the week.
This is rolling-origin one-step evaluation with observed history, and no held-out target
is ever an input to its own forecast.

*The scale contract.* Models are fed scaled inputs and return predictions in that same
scaled space. :func:`run_model` inverts the transform before scoring, so every reported
number is in the original units of the series whatever ``scaling`` was used. Keeping the
inversion here rather than inside each model is what makes that guarantee hold for all
models at once instead of depending on each one remembering to do it.

*The timing protocol.* A single wall-clock reading is not a measurement. EXP-000's
single-shot numbers varied five-fold for identical operations, and on the MPS backend the
first call at a new batch shape is roughly thirty times slower than a warm one, while an
unsynchronised timer stops when work was queued rather than when it finished. Execution
time is therefore the median of ``n_timing_reps`` warm repetitions with the device
synchronised, reported twice: over the full batch of evaluation windows, and as
single-window latency, because the two orderings differ by device.

*The run lifecycle.* :func:`reserve_run` claims the directory and writes the config
*before* training starts. Refusing to overwrite only at the end means a twenty-minute run
that collides with an existing id dies having written nothing, and the artefacts are the
evidence the experimentation criterion is graded on.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .config import EVAL_WEEK_END, EVAL_WEEK_START, RUNS_DIR, SLOTS_PER_DAY
from .dataio import load_index, load_series
from .features import (
    Scaler,
    Split,
    context_windows,
    eval_week_split,
    make_windows,
    target_calendar,
    window_target_rows,
)
from .training import synchronize
from .utils.logging import get_logger
from .utils.seed import set_seed
from .utils.timing import Timer, hardware_string, median_ms, repeat_timed

LOG = get_logger(__name__)

#: Slots before the evaluation week, i.e. rows 0:6480. ``VAL_FRAC`` carves exactly one
#: 1008-slot week off the end of it, so validation is Mon 2013-12-09 00:00 to Sun
#: 2013-12-15 23:50. The previous default of 0.15 gave 972 rows starting on a Monday at
#: 06:00, which is neither a week nor aligned to one.
PRE_EVAL_SLOTS = 6480
SLOTS_PER_WEEK = 1008
VAL_FRAC = SLOTS_PER_WEEK / PRE_EVAL_SLOTS


@dataclass
class AreaData:
    """Everything one geographical area contributes to a run."""

    square_id: int
    seq_len: int
    horizon: int
    scaling: str
    split: Split
    scaler: Scaler
    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    y_val_true: np.ndarray
    y_test_true: np.ndarray
    mase_scale: float
    train_exog: np.ndarray | None = None
    val_exog: np.ndarray | None = None
    test_exog: np.ndarray | None = None

    @property
    def calendar(self) -> bool:
        """Whether this area's models are fed calendar features."""
        return self.train_exog is not None

    def describe(self) -> str:
        return (
            f"square {self.square_id}: train {self.train[0].shape[0]} windows, "
            f"val {self.val[0].shape[0]}, test {self.test[0].shape[0]} "
            f"(seq_len={self.seq_len}, scaling={self.scaling}, "
            f"calendar={'on' if self.calendar else 'off'})"
        )


def prepare_area(
    square_id: int,
    *,
    seq_len: int,
    horizon: int = 1,
    scaling: str = "none",
    val_frac: float = VAL_FRAC,
    season: int = SLOTS_PER_DAY,
    calendar: bool = True,
) -> AreaData:
    """Windowed train / validation / evaluation data for one area, leakage-safe.

    The default ``val_frac`` makes validation exactly one calendar week, the week
    immediately before the evaluation window. That week is the busiest of the record on
    all three reported cells, so validation error will exceed test error for every model;
    that is a property of the calendar and is stated in the report rather than discovered
    in the results table.
    """
    series = load_series(square_id).to_numpy(dtype=np.float64)
    index = load_index()
    split = eval_week_split(index, EVAL_WEEK_START, EVAL_WEEK_END, val_frac=val_frac)

    scaler = Scaler(scaling).fit(series[split.train])
    # Training windows are built strictly inside the training slice, so none of them can
    # reach across a split boundary.
    train = make_windows(scaler.transform(series[split.train]), seq_len, horizon=horizon)
    # Both held-out periods use the same convention: inputs may reach into the period
    # before, targets stay strictly inside. Recorded in the run config.
    val = context_windows(
        series, split, seq_len, horizon=horizon, scaler=scaler, target_slice=split.val
    )
    test = context_windows(series, split, seq_len, horizon=horizon, scaler=scaler)

    rows = window_target_rows(split, seq_len, horizon=horizon)
    exog = {k: target_calendar(index, v) for k, v in rows.items()} if calendar else {}
    for name, (X, _) in (("train", train), ("val", val), ("test", test)):
        if calendar and len(exog[name]) != len(X):
            raise ValueError(
                f"{name} exog has {len(exog[name])} rows against {len(X)} windows; "
                "the calendar features would be silently offset from the slots they describe."
            )

    return AreaData(
        square_id=square_id,
        seq_len=seq_len,
        horizon=horizon,
        scaling=scaling,
        split=split,
        scaler=scaler,
        train=train,
        val=val,
        test=test,
        # Taken from the series directly rather than by inverting float32 windows, which
        # differed from the truth by up to 2.2e-3 under log1p and made the ground truth
        # depend on which scaling had been chosen.
        y_val_true=series[split.val],
        y_test_true=series[split.test],
        # Hyndman scaling: the denominator comes from the TRAINING period, never from the
        # window being scored, or "below 1 beats seasonal persistence" stops being true.
        mase_scale=M.seasonal_naive_scale(series[split.train], season=season),
        train_exog=exog.get("train"),
        val_exog=exog.get("val"),
        test_exog=exog.get("test"),
    )


@dataclass
class ModelResult:
    """One model on one area: metrics plus the cost of producing them."""

    area: int
    model: str
    n_params: int
    val: dict[str, float]
    test: dict[str, float]
    fit_s: float
    fit_cpu_s: float
    predict_s: float
    predict_us_per_step: float
    predict_ms_batch_median: float
    predict_ms_batch_reps: int
    latency_ms_single_median: float
    latency_reps: int
    seed: int
    device: str
    n_epochs: int | None
    best_epoch: int | None
    s_per_epoch: float | None
    stopped_early: bool | None
    predictions: np.ndarray = field(repr=False)

    def summary(self) -> dict[str, Any]:
        """Everything except the prediction array, for ``metrics.json``."""
        out = {k: v for k, v in asdict(self).items() if k != "predictions"}
        return out


def run_model(
    name: str,
    factory: Callable[[], object],
    area: AreaData,
    *,
    seed: int = 42,
    n_timing_reps: int = 20,
    n_latency_reps: int = 50,
    season: int = SLOTS_PER_DAY,
    mape_mode: str = "epsilon",
) -> ModelResult:
    """Fit one model on one area's training windows and score it on both held-out sets."""
    set_seed(seed)
    model = factory()
    # Seeding the global generators is not enough. Both trainable models re-seed from
    # their own constructor default inside `fit` (`TorchForecaster.fit` calls `set_seed`,
    # and the tree model passes `random_state`), so without this every "three seed" run
    # silently repeated one seed: EXP-004 and EXP-005 returned a standard deviation of
    # exactly 0.00 over three seeds, which is what exposed it. The run's seed is
    # therefore pushed onto the model before it is fitted, and the ModelResult records
    # the value that was actually used.
    if hasattr(model, "seed"):
        model.seed = seed
    X_tr, y_tr = area.train

    device = str(getattr(model, "device", "cpu"))

    def sync() -> None:
        synchronize(device)

    with Timer(f"fit:{name}", sync=sync) as fit_timer:
        model.fit(
            X_tr,
            y_tr,
            X_val=area.val[0],
            y_val=area.val[1],
            exog=area.train_exog,
            exog_val=area.val_exog,
        )
    # A torch model resolves its device during fit, so read it again afterwards.
    device = str(getattr(model, "device", device))

    # Predictions come back in the scaled space the model was fed, so they are inverted
    # here before they meet the original-scale truth. Without this, any `scaling` other
    # than "none" silently compares two different unit systems and still returns
    # plausible finite numbers: on square 5161 with `scaling: standard` persistence
    # scored MAE 1445.19 and R2 -1.12 in place of 92.80 and 0.9902.
    val_pred = area.scaler.inverse_transform(model.predict(area.val[0], area.val_exog))
    test_pred = area.scaler.inverse_transform(model.predict(area.test[0], area.test_exog))

    # Execution time, measured twice. The first call at a new batch shape on MPS is about
    # thirty times slower than a warm one, so it is spent as the warm-up and excluded.
    batch_times = repeat_timed(
        lambda: model.predict(area.test[0], area.test_exog),
        n_timing_reps,
        warmup=1,
        sync=sync,
    )
    one_X = area.test[0][:1]
    one_exog = None if area.test_exog is None else area.test_exog[:1]
    latency_times = repeat_timed(
        lambda: model.predict(one_X, one_exog), n_latency_reps, warmup=2, sync=sync
    )
    batch_ms = median_ms(batch_times)

    score = dict(season=season, mape_mode=mape_mode, mase_scale=area.mase_scale)
    val_scores = M.evaluate(area.y_val_true, val_pred, **score)
    test_scores = M.evaluate(area.y_test_true, test_pred, **score)
    for split_name, scores in (("val", val_scores), ("test", test_scores)):
        if scores["n_dropped"]:
            LOG.warning(
                "%s on square %d: %d of %d %s predictions are non-finite, so every "
                "metric for that split is NaN. A diverged model is not scored on the "
                "part of the window it survived.",
                name,
                area.square_id,
                scores["n_dropped"],
                scores["n_dropped"] + scores["n_scored"],
                split_name,
            )

    history: Mapping[str, Any] = getattr(model, "history", {}) or {}
    n_steps = max(len(test_pred), 1)
    return ModelResult(
        area=area.square_id,
        model=name,
        n_params=int(getattr(model, "n_params", 0)),
        val=val_scores,
        test=test_scores,
        fit_s=fit_timer.wall_s,
        fit_cpu_s=fit_timer.cpu_s,
        predict_s=batch_ms / 1e3,
        predict_us_per_step=batch_ms * 1e3 / n_steps,
        predict_ms_batch_median=batch_ms,
        predict_ms_batch_reps=len(batch_times),
        latency_ms_single_median=median_ms(latency_times),
        latency_reps=len(latency_times),
        seed=seed,
        device=str(history.get("device", device)),
        n_epochs=history.get("n_epochs"),
        best_epoch=history.get("best_epoch"),
        s_per_epoch=history.get("s_per_epoch"),
        stopped_early=history.get("stopped_early"),
        predictions=test_pred.astype(np.float32),
    )


#: What the timing columns mean, written into every ``metrics.json`` so a table lifted
#: from a run can always be traced back to how it was measured.
TIMING_PROTOCOL = {
    "fit_s": "wall seconds of one fit call, device synchronised before the clock stops",
    "fit_cpu_s": "process CPU seconds of the same call",
    "predict_ms_batch_median": (
        "median of n repetitions of predicting the whole evaluation window, after one "
        "untimed warm-up call, device synchronised before each stop"
    ),
    "latency_ms_single_median": (
        "median of n repetitions of predicting a single window, after two untimed warm-ups"
    ),
    "why": (
        "single-shot timings varied five-fold for identical operations in EXP-000, and "
        "the first MPS call at a new batch shape is about thirty times slower than a warm one"
    ),
}


def results_frame(results: Sequence[ModelResult], *, split: str = "test") -> pd.DataFrame:
    """Long-format table of every (area, model) pair, for the per-area report tables."""
    rows = []
    for r in results:
        scores = r.test if split == "test" else r.val
        rows.append(
            {
                "area": r.area,
                "model": r.model,
                "seed": r.seed,
                "n_params": r.n_params,
                **{k: round(v, 4) for k, v in scores.items()},
                "device": r.device,
                "epochs": r.n_epochs,
                "best_epoch": r.best_epoch,
                "s_per_epoch": None if r.s_per_epoch is None else round(r.s_per_epoch, 3),
                "fit_s": round(r.fit_s, 4),
                "fit_cpu_s": round(r.fit_cpu_s, 4),
                "predict_ms": round(r.predict_ms_batch_median, 3),
                "latency_ms": round(r.latency_ms_single_median, 4),
                "predict_us_per_step": round(r.predict_us_per_step, 3),
            }
        )
    return pd.DataFrame(rows)


def aggregate_seeds(results: Sequence[ModelResult], *, split: str = "test") -> pd.DataFrame:
    """Mean and standard deviation of every metric over seeds, per (area, model).

    A single run is a weak claim when the model is stochastic, so the final
    configurations are repeated and reported as mean and spread. With one seed the
    standard deviation column is zero and the table reads the same way.
    """
    frame = results_frame(results, split=split)
    metric_cols = [
        c
        for c in frame.columns
        if c not in {"area", "model", "seed", "device", "epochs", "best_epoch"}
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    grouped = frame.groupby(["area", "model"], sort=False)[metric_cols]
    out = grouped.agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


# ------------------------------------------------------------------- run artefacts


RUN_ID_RE = re.compile(r"^EXP-(\d{3,})$")


def next_run_id(runs_dir: Path | str = RUNS_DIR, *, prefix: str = "EXP") -> str:
    """The next unused ``EXP-NNN`` in ``runs_dir``, one above the highest present."""
    directory = Path(runs_dir)
    highest = -1
    if directory.exists():
        for child in directory.iterdir():
            match = RUN_ID_RE.match(child.name)
            if child.is_dir() and match:
                highest = max(highest, int(match.group(1)))
    return f"{prefix}-{highest + 1:03d}"


def reserve_run(
    run_id: str, config: Mapping[str, object], *, runs_dir: Path | str = RUNS_DIR
) -> Path:
    """Claim ``runs_dir/run_id`` and write its config, before any training starts.

    Checking for a collision only when the results are ready means a long run can die
    having written nothing at all. Reserving first also leaves a config behind when a run
    is interrupted, which is what makes an abandoned experiment legible later.
    """
    import yaml

    out = Path(runs_dir) / run_id
    if out.exists():
        raise FileExistsError(f"{out} already exists; runs are append-only.")
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    LOG.info("reserved %s", out)
    return out


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``json.dump`` writes the bare literal ``NaN`` by default, which no strict JSON parser
    accepts. A diverged model is exactly the case that produces NaN metrics, so the
    artefact recording that failure must not itself be unreadable.
    """
    if isinstance(obj, Mapping):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def write_run(
    run_id: str,
    results: Sequence[ModelResult],
    *,
    config: Mapping[str, object],
    notes: str = "",
    runs_dir: Path | str = RUNS_DIR,
) -> Path:
    """Write ``config.yaml``, ``metrics.json``, ``predictions.npy`` and ``run.log``.

    Fills a directory already claimed by :func:`reserve_run`, or creates one. Never
    overwrites results that are already there: the log is append-only history, and a run
    that can be silently replaced is not evidence of anything.
    """
    import yaml

    out = Path(runs_dir) / run_id
    existing = [n for n in ("metrics.json", "predictions.npy", "run.log") if (out / n).exists()]
    if existing:
        raise FileExistsError(f"{out} already holds {', '.join(existing)}; runs are append-only.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))

    areas = sorted({r.area for r in results})
    models = list(dict.fromkeys(r.model for r in results))
    seeds = sorted({r.seed for r in results})
    n_steps = len(results[0].predictions)
    stack = np.full((len(areas), len(models), len(seeds), n_steps), np.nan, np.float32)
    for r in results:
        stack[areas.index(r.area), models.index(r.model), seeds.index(r.seed)] = r.predictions
    np.save(out / "predictions.npy", stack)

    devices = {r.model: r.device for r in results}
    payload = {
        "run_id": run_id,
        "hardware": hardware_string(",".join(sorted(set(devices.values())))),
        "predictions_axes": ["area", "model", "seed", "step"],
        "areas": areas,
        "models": models,
        "seeds": seeds,
        "device_by_model": devices,
        "timing_protocol": TIMING_PROTOCOL,
        "config": dict(config),
        "results": [r.summary() for r in results],
    }
    (out / "metrics.json").write_text(json.dumps(_json_safe(payload), indent=2))

    table = results_frame(results).to_string(index=False)
    header = f"{run_id}\nhardware: {payload['hardware']}\n\n{notes.strip()}\n\n"
    body = table
    if len(seeds) > 1:
        body += "\n\nmean and std over seeds:\n" + aggregate_seeds(results).to_string(index=False)
    (out / "run.log").write_text(header + body + "\n")
    LOG.info("wrote %s", out)
    return out


def log_row(run_id: str, date: str, results: Sequence[ModelResult], *, changed: str) -> str:
    """One pipe-delimited EXPERIMENT_LOG.md row, ready to paste.

    The rationale is deliberately left as a placeholder. A log whose "why and next"
    column was generated alongside the numbers reads exactly like what it is, and that
    column is the only part of the row a reader cannot reconstruct from the artefacts.
    """
    frame = results_frame(results)
    val = results_frame(results, split="val")
    best = frame["MAE"].idxmin() if "MAE" in frame and len(frame) else None
    if best is None:
        return f"| {run_id} | {date} | - | {changed} | - | - | - | - | - | |"
    row = frame.loc[best]
    return (
        f"| {run_id} | {date} | {row['model']} | {changed} | "
        f"{val.loc[best, 'MAE']:.2f} | {row['MAE']:.2f} | {row['RMSE']:.2f} | "
        f"{row['fit_s']:.2f} | {row['predict_ms']:.2f} | "
        "<why and next: to be written by hand> |"
    )


def load_run(run_id: str, *, runs_dir: Path | str = RUNS_DIR) -> dict[str, Any]:
    """The stored ``metrics.json`` and ``config.yaml`` of a finished run, as one mapping.

    The counterpart to :func:`write_run`, kept beside it so the read and write formats cannot
    drift apart. Returns ``{"metrics": ..., "config": ...}``; the config is the resolved one the
    run actually used, not the defaults, which is what makes a cross-run comparison trustworthy.
    """
    import yaml

    base = Path(runs_dir) / run_id
    metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    config_path = base / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {"metrics": metrics, "config": config}


def compare_runs(
    run_ids: Sequence[str],
    *,
    runs_dir: Path | str = RUNS_DIR,
    metric: str = "MAE",
    select_on: str = "val",
) -> pd.DataFrame:
    """One row per (run, area, model) for a set of runs, with what changed between them.

    Why this rather than reading nine ``metrics.json`` files in a script: a capacity grid is only
    interpretable if the row labels are the hyperparameters that actually differ, and a label
    typed by hand is a label that can be wrong. The ``changed`` column is derived by diffing each
    run's resolved model config against the others in the set, so a row cannot claim a variable
    the run did not move.

    ``select_on`` names the split the ``selected`` flag is computed from, and defaults to
    validation. Selecting on ``"test"`` is available for diagnosis but is not a selection a
    result may rest on, and the docstring says so because the temptation is real: on this dataset
    the transform comparison of EXP-002 looks decisive on test and is close to a wash on
    validation.
    """
    loaded = {rid: load_run(rid, runs_dir=runs_dir) for rid in run_ids}

    params: dict[str, dict[str, dict[str, Any]]] = {}
    for rid, payload in loaded.items():
        models_cfg = (payload["config"].get("models") or {}) if payload["config"] else {}
        params[rid] = {m: dict(cfg or {}) for m, cfg in models_cfg.items()}

    def changed_for(rid: str, model: str) -> str:
        mine = params.get(rid, {}).get(model, {})
        others = [params[r].get(model, {}) for r in run_ids if r != rid]
        if not others or not mine:
            return ""
        keys = [k for k in mine if any(o.get(k) != mine[k] for o in others if o)]
        drop = {"device", "loss", "max_epochs", "patience"}
        keys = [k for k in sorted(keys) if k not in drop]
        return ", ".join(f"{k}={mine[k]}" for k in keys)

    rows: list[dict[str, Any]] = []
    for rid, payload in loaded.items():
        by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for r in payload["metrics"].get("results", []):
            by_key.setdefault((r["area"], r["model"]), []).append(r)
        for (area, model), group in by_key.items():
            row: dict[str, Any] = {
                "run": rid,
                "area": area,
                "model": model,
                "changed": changed_for(rid, model),
                "n_params": int(group[0].get("n_params", 0)),
                "n_seeds": len(group),
            }
            for split in ("val", "test"):
                values = [g[split][metric] for g in group if split in g]
                if values:
                    row[f"{split}_{metric}"] = float(np.mean(values))
                    row[f"{split}_{metric}_sd"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    )
            row["s_per_epoch"] = float(np.mean([g.get("s_per_epoch") or 0.0 for g in group]))
            row["fit_s"] = float(np.mean([g.get("fit_s") or 0.0 for g in group]))
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    column = f"{select_on}_{metric}"
    frame["selected"] = False
    for _key, sub in frame.groupby(["area", "model"]):
        if column in sub and sub[column].notna().any():
            frame.loc[sub[column].idxmin(), "selected"] = True
    return frame.sort_values(["area", "model", column]).reset_index(drop=True)
