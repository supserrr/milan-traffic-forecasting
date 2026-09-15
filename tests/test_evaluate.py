"""The evaluation harness: the layer that turns model output into reported numbers.

The unit tests elsewhere cover the pieces. These cover the assembly, which is where two
defects lived undetected behind a green suite: predictions were scored without inverting
the scaler, and non-finite predictions were silently dropped. Both produced plausible
finite numbers rather than an error, so only an assertion about the *relationship*
between inputs and reported metrics catches them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from milan_traffic import metrics as M
from milan_traffic.evaluate import (
    ModelResult,
    compare_runs,
    load_run,
    next_run_id,
    reserve_run,
    results_frame,
    run_model,
    write_run,
)
from milan_traffic.features import (
    Scaler,
    chronological_split,
    context_windows,
    make_windows,
    target_calendar,
    window_target_rows,
)


class _Persistence:
    """Scale-equivariant reference: repeat the last observed input.

    Equivariance is the property the scaling test leans on. Persistence commutes with any
    monotone affine transform, so its inverse-transformed predictions must be identical
    whatever scaler was used - unless the harness forgets to invert them.
    """

    name = "persistence"
    n_params = 0

    def fit(self, X, y, *, X_val=None, y_val=None, exog=None, exog_val=None, **kw):
        return self

    def predict(self, X, exog=None):
        return np.asarray(X)[:, -1]


class _Diverged(_Persistence):
    """Persistence that emits NaN from ``after`` onward, like a run with a NaN loss."""

    name = "diverged"

    def __init__(self, after: int = 200):
        self.after = after

    def predict(self, X, exog=None):
        out = np.asarray(X, dtype=np.float64)[:, -1].copy()
        out[self.after :] = np.nan
        return out


def _area(
    series: np.ndarray,
    *,
    seq_len: int = 12,
    scaling: str = "none",
    season: int = 24,
    calendar: bool = False,
):
    """A minimal AreaData built by the same helpers ``prepare_area`` uses.

    Built here rather than via ``prepare_area`` so the test needs no processed store and
    no fixed evaluation week, while still exercising the real scaler/window/inverse path.
    The truth arrays are taken from the series directly, as ``prepare_area`` now does,
    rather than by inverting float32 windows.
    """
    from milan_traffic.evaluate import AreaData

    split = chronological_split(len(series), test_slice=slice(len(series) - 120, len(series)))
    scaler = Scaler(scaling).fit(series[split.train])
    train = make_windows(scaler.transform(series[split.train]), seq_len)
    val = context_windows(series, split, seq_len, scaler=scaler, target_slice=split.val)
    test = context_windows(series, split, seq_len, scaler=scaler)
    exog = {}
    if calendar:
        rows = window_target_rows(split, seq_len)
        index = pd.date_range("2013-11-01", periods=len(series), freq="10min", tz="Europe/Rome")
        exog = {k: target_calendar(index, v) for k, v in rows.items()}
    return AreaData(
        square_id=1,
        seq_len=seq_len,
        horizon=1,
        scaling=scaling,
        split=split,
        scaler=scaler,
        train=train,
        val=val,
        test=test,
        y_val_true=np.asarray(series[split.val], dtype=np.float64),
        y_test_true=np.asarray(series[split.test], dtype=np.float64),
        mase_scale=M.seasonal_naive_scale(series[split.train], season=season),
        train_exog=exog.get("train"),
        val_exog=exog.get("val"),
        test_exog=exog.get("test"),
    )


# --- scaling ------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", ["none", "standard", "minmax", "log1p", "log1p-standard"])
def test_metrics_are_scale_invariant_for_a_scale_equivariant_model(seasonal_series, scaling):
    """The regression test for the inverse-transform bug.

    A scale-equivariant model must score identically under every normalisation, because
    the choice of scaling is a detail of how the model was fed, not of what it predicted.
    Before the fix this passed only for ``scaling="none"``.
    """
    reference = run_model("naive", _Persistence, _area(seasonal_series, scaling="none"), season=24)
    got = run_model("naive", _Persistence, _area(seasonal_series, scaling=scaling), season=24)
    for metric in ("MAE", "RMSE", "WAPE", "R2"):
        # Loose enough for the float32 rounding make_windows introduces in the scaled
        # space, orders of magnitude tighter than the defect (MAE 92.80 -> 1445.19).
        assert got.test[metric] == pytest.approx(reference.test[metric], rel=1e-3, abs=1e-6)
        assert got.val[metric] == pytest.approx(reference.val[metric], rel=1e-3, abs=1e-6)


def test_predictions_are_reported_in_the_original_scale(seasonal_series):
    """Predictions land in the units of the series, not of the scaled space."""
    result = run_model("naive", _Persistence, _area(seasonal_series, scaling="standard"), season=24)
    truth = _area(seasonal_series, scaling="standard").y_test_true
    assert result.predictions.mean() == pytest.approx(truth.mean(), rel=0.05)
    # Standardised space would be centred near zero; the original scale is not.
    assert abs(result.predictions.mean()) > 1.0


def test_scaler_is_fit_on_the_training_slice_only(seasonal_series):
    area = _area(seasonal_series, scaling="standard")
    train_only = Scaler("standard").fit(seasonal_series[area.split.train])
    assert area.scaler.center_ == pytest.approx(train_only.center_)
    assert area.scaler.scale_ == pytest.approx(train_only.scale_)
    assert area.scaler.center_ != pytest.approx(float(np.mean(seasonal_series)))


# --- non-finite predictions ----------------------------------------------------------


def test_a_diverged_model_does_not_beat_an_intact_one(seasonal_series):
    """The regression test for the silent-drop bug.

    Scored on the subset it survived, a model that emits NaN for most of the window
    returns a *better* MAE than an intact forecast. It must score NaN instead.
    """
    area = _area(seasonal_series)
    honest = run_model("naive", _Persistence, area, season=24)
    broken = run_model("diverged", lambda: _Diverged(after=20), area, season=24)

    assert np.isfinite(honest.test["MAE"])
    for metric in ("MAE", "RMSE", "MASE", "R2"):
        assert np.isnan(broken.test[metric]), f"{metric} was scored on the surviving subset"


def test_dropped_points_are_counted_in_the_run_artefacts(seasonal_series):
    area = _area(seasonal_series)
    n_test = len(area.test[1])
    broken = run_model("diverged", lambda: _Diverged(after=20), area, season=24)

    assert broken.test["n_dropped"] == n_test - 20
    assert broken.test["n_scored"] == 20
    honest = run_model("naive", _Persistence, area, season=24)
    assert honest.test["n_dropped"] == 0
    assert honest.test["n_scored"] == n_test


def test_a_single_nan_is_enough_to_poison_the_metric():
    """No threshold, no tolerance: one non-finite point means the window is not scored."""
    y = np.arange(100.0)
    p = y.copy()
    assert M.mae(y, p) == pytest.approx(0.0)
    p[57] = np.nan
    assert np.isnan(M.mae(y, p))
    assert M.mae(y, p, allow_missing=True) == pytest.approx(0.0)
    assert M.finite_counts(y, p) == (99, 1)


def test_results_frame_exposes_the_scored_count(seasonal_series):
    area = _area(seasonal_series)
    frame = results_frame([run_model("naive", _Persistence, area, season=24)])
    assert "n_scored" in frame.columns and "n_dropped" in frame.columns
    assert int(frame.loc[0, "n_dropped"]) == 0


def test_run_model_returns_a_result_for_every_test_target(seasonal_series):
    """Every slot of the evaluation period gets a prediction - that is why context
    windows reach back rather than discarding the first ``seq_len`` points."""
    area = _area(seasonal_series)
    result = run_model("naive", _Persistence, area, season=24)
    assert isinstance(result, ModelResult)
    assert len(result.predictions) == area.split.test.stop - area.split.test.start


# --- the run lifecycle ---------------------------------------------------------------


def test_next_run_id_counts_from_the_highest_present(tmp_path):
    assert next_run_id(tmp_path) == "EXP-000"
    for name in ("EXP-000", "EXP-003", "not-a-run", "EXP-abc"):
        (tmp_path / name).mkdir()
    (tmp_path / "EXP-009.txt").write_text("a file, not a run")
    assert next_run_id(tmp_path) == "EXP-004"


def test_reserve_run_writes_the_config_before_any_training(tmp_path):
    """An interrupted run must still leave evidence that it was attempted."""
    out = reserve_run("EXP-001", {"run": {"id": "EXP-001"}}, runs_dir=tmp_path)
    assert (out / "config.yaml").exists()
    assert not (out / "metrics.json").exists()
    with pytest.raises(FileExistsError):
        reserve_run("EXP-001", {}, runs_dir=tmp_path)


def test_write_run_fills_a_reserved_directory_but_never_overwrites_results(
    seasonal_series, tmp_path
):
    area = _area(seasonal_series)
    results = [run_model("naive", _Persistence, area, season=24, n_timing_reps=2, n_latency_reps=2)]
    reserve_run("EXP-001", {"run": {"id": "EXP-001"}}, runs_dir=tmp_path)
    out = write_run("EXP-001", results, config={"run": {"id": "EXP-001"}}, runs_dir=tmp_path)
    for name in ("config.yaml", "metrics.json", "predictions.npy", "run.log"):
        assert (out / name).exists(), name
    with pytest.raises(FileExistsError, match="append-only"):
        write_run("EXP-001", results, config={}, runs_dir=tmp_path)


def test_metrics_json_is_strict_json_even_when_a_model_diverged(seasonal_series, tmp_path):
    """A diverged run produces NaN metrics, and json.dump writes a bare NaN literal that
    no strict parser accepts. The artefact recording the failure must stay readable."""
    area = _area(seasonal_series)
    results = [
        run_model(
            "diverged",
            lambda: _Diverged(after=20),
            area,
            season=24,
            n_timing_reps=2,
            n_latency_reps=2,
        )
    ]
    out = write_run("EXP-002", results, config={}, runs_dir=tmp_path)
    payload = json.loads((out / "metrics.json").read_text())  # strict: rejects NaN
    assert payload["results"][0]["test"]["MAE"] is None
    assert payload["predictions_axes"] == ["area", "model", "seed", "step"]
    assert "timing_protocol" in payload


def test_predictions_axes_match_the_documented_order(seasonal_series, tmp_path):
    area = _area(seasonal_series)
    results = [
        run_model(name, _Persistence, area, season=24, seed=seed, n_timing_reps=2, n_latency_reps=2)
        for name in ("naive", "other")
        for seed in (42, 7)
    ]
    out = write_run("EXP-003", results, config={}, runs_dir=tmp_path)
    stack = np.load(out / "predictions.npy")
    assert stack.shape == (1, 2, 2, len(area.y_test_true))
    np.testing.assert_allclose(stack[0, 0, 0], results[0].predictions)


def test_timing_fields_are_medians_over_the_requested_repetitions(seasonal_series):
    area = _area(seasonal_series)
    result = run_model("naive", _Persistence, area, season=24, n_timing_reps=5, n_latency_reps=3)
    assert result.predict_ms_batch_reps == 5
    assert result.latency_reps == 3
    assert result.predict_ms_batch_median > 0
    assert result.latency_ms_single_median > 0
    # A single window must not cost more than the whole batch.
    assert result.latency_ms_single_median <= result.predict_ms_batch_median * 2
    assert result.seed == 42
    assert result.device == "cpu"


def test_calendar_features_are_optional_and_reach_the_model(seasonal_series):
    """With calendar off the exog arrays are absent; with it on they are aligned."""
    off = _area(seasonal_series, calendar=False)
    on = _area(seasonal_series, calendar=True)
    assert off.calendar is False and on.calendar is True
    assert on.train_exog.shape == (len(on.train[0]), 5)
    assert on.test_exog.shape == (len(on.test[0]), 5)
    seen = {}

    class _Spy(_Persistence):
        def predict(self, X, exog=None):
            seen["exog"] = None if exog is None else exog.shape
            return np.asarray(X)[:, -1]

    run_model("spy", _Spy, on, season=24, n_timing_reps=2, n_latency_reps=2)
    assert seen["exog"] == (1, 5)  # the last call is the single-window latency probe


def test_the_runs_seed_reaches_a_model_that_seeds_itself(seasonal_series):
    """Both trainable models re-seed from their own constructor default inside ``fit``.

    Seeding the global generators in ``run_model`` is therefore not enough, and without
    pushing the seed onto the model a multi-seed run silently repeats one seed. The
    symptom was a standard deviation of exactly 0.00 over three seeds.
    """

    class _SelfSeeding(_Persistence):
        def __init__(self):
            self.seed = 42
            self.seen: int | None = None

        def fit(self, X, y, **kw):
            self.seen = self.seed
            return self

    captured = []

    def factory():
        model = _SelfSeeding()
        captured.append(model)
        return model

    area = _area(seasonal_series)
    result = run_model("spy", factory, area, seed=1337, n_timing_reps=2, n_latency_reps=2)
    assert captured[0].seen == 1337, "the run's seed never reached the model"
    assert result.seed == 1337


def _fake_run(base, run_id, *, model, params, val, test, n_params=100, area=5161):
    """A run directory reduced to what the readers need: metrics plus the resolved config."""
    import yaml

    out = base / run_id
    out.mkdir(parents=True)
    results = [
        {
            "area": area,
            "model": model,
            "n_params": n_params,
            "seed": seed,
            "val": {"MAE": v},
            "test": {"MAE": t},
            "s_per_epoch": 1.0,
            "fit_s": 10.0,
        }
        for seed, v, t in zip([42, 1337, 2024], val, test, strict=True)
    ]
    payload = {"areas": [area], "models": [model], "seeds": [42, 1337, 2024], "results": results}
    (out / "metrics.json").write_text(json.dumps(payload))
    (out / "config.yaml").write_text(yaml.safe_dump({"models": {model: params}}))
    return out


def test_load_run_returns_metrics_and_the_resolved_config(tmp_path):
    _fake_run(
        tmp_path,
        "EXP-900",
        model="tcn",
        params={"levels": 6, "channels": 32},
        val=[100.0, 101.0, 102.0],
        test=[80.0, 81.0, 82.0],
    )
    payload = load_run("EXP-900", runs_dir=tmp_path)
    assert payload["metrics"]["models"] == ["tcn"]
    assert payload["config"]["models"]["tcn"]["levels"] == 6


def test_compare_runs_derives_what_changed_instead_of_trusting_a_label(tmp_path):
    """A capacity grid is only readable if the row label is the variable the run actually moved.
    A hand-typed label can claim one the run held fixed, which is what makes a grid untrustworthy,
    so the label is diffed out of the stored configs and a held-constant key must not appear."""
    _fake_run(
        tmp_path,
        "EXP-900",
        model="tcn",
        params={"levels": 6, "channels": 32},
        val=[100.0] * 3,
        test=[80.0] * 3,
    )
    _fake_run(
        tmp_path,
        "EXP-901",
        model="tcn",
        params={"levels": 5, "channels": 32},
        val=[105.0] * 3,
        test=[78.0] * 3,
    )
    frame = compare_runs(["EXP-900", "EXP-901"], runs_dir=tmp_path)
    changed = dict(zip(frame["run"], frame["changed"], strict=True))
    assert changed == {"EXP-900": "levels=6", "EXP-901": "levels=5"}


def test_compare_runs_selects_on_validation_not_on_the_evaluation_week(tmp_path):
    """The flag that keeps a selection honest. Here the two splits disagree and validation wins;
    selecting on test is reachable for diagnosis and must be an explicit request."""
    _fake_run(
        tmp_path, "EXP-900", model="tcn", params={"levels": 6}, val=[100.0] * 3, test=[80.0] * 3
    )
    _fake_run(
        tmp_path, "EXP-901", model="tcn", params={"levels": 5}, val=[105.0] * 3, test=[70.0] * 3
    )
    on_val = compare_runs(["EXP-900", "EXP-901"], runs_dir=tmp_path, select_on="val")
    assert on_val[on_val["selected"]]["run"].tolist() == ["EXP-900"]
    on_test = compare_runs(["EXP-900", "EXP-901"], runs_dir=tmp_path, select_on="test")
    assert on_test[on_test["selected"]]["run"].tolist() == ["EXP-901"]


def test_compare_runs_reports_seed_spread_and_selects_once_per_architecture(tmp_path):
    _fake_run(
        tmp_path,
        "EXP-900",
        model="tcn",
        params={"levels": 6},
        val=[100.0, 102.0, 104.0],
        test=[80.0] * 3,
    )
    _fake_run(
        tmp_path, "EXP-901", model="lstm", params={"hidden": 64}, val=[90.0] * 3, test=[70.0] * 3
    )
    frame = compare_runs(["EXP-900", "EXP-901"], runs_dir=tmp_path)
    tcn = frame[frame["model"] == "tcn"].iloc[0]
    assert tcn["val_MAE"] == pytest.approx(102.0)
    assert tcn["val_MAE_sd"] == pytest.approx(2.0)
    assert tcn["n_seeds"] == 3
    assert int(frame["selected"].sum()) == 2
