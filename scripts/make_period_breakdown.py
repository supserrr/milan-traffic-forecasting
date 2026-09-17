"""Cut a finished run's evaluation week into calendar periods, for the failure analysis.

A weekly MAE per model per area cannot answer the brief's requirement for a period on which a
model performs poorly. On square 5059 the convolutional model's whole deficit to linear AR(144)
is two days wide: outside Thursday 19 and Friday 20 December the two are level, and on those two
days the network loses to persistence. This script produces that evidence, the diagnostics that
say why those days are different, and the weekend comparison across the three areas. It also
sizes the two days against the calendar-off control run, pooled over seeds and per seed, so the
share quoted in the report is an artefact of the repository rather than a number typed into it.

Thin CLI: every computation lives in ``milan_traffic.periods``. Re-running overwrites its own
outputs, because these are derived views of existing runs rather than run evidence, and it
appends no row to the experiment log.

    python scripts/make_period_breakdown.py --run EXP-008 --ablation-run EXP-007

Outputs: ``reports/tables/per_period_{area}.md``, ``per_period_wape.md``,
``per_period_stats.md`` and ``reports/figures/per_period_error.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic import periods as PD  # noqa: E402
from milan_traffic import viz  # noqa: E402
from milan_traffic.config import FIGURES_DIR, RUNS_DIR, TABLES_DIR  # noqa: E402
from milan_traffic.dataio import load_index, load_series  # noqa: E402
from milan_traffic.evaluate import EVAL_WEEK_END, EVAL_WEEK_START  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402

LOG = get_logger("make_period_breakdown")

#: Printed names, matching the report's per-area tables.
DISPLAY = {
    "lstm": "LSTM",
    "tcn": "TCN",
    "gbt": "GBT",
    "naive": "Persistence",
    "linear_ar_144": "Linear AR(144)",
}
ABLATION_LABEL = "TCN, no calendar"

#: The seasonal-naive baseline is excluded from the pooled WAPE panel: its weekend WAPE on
#: square 5259 reaches 142 percent, which would set the axis and hide the effect being shown.
WAPE_MODELS = ("lstm", "tcn", "gbt", "naive", "linear_ar_144")


def _eval_slice(index: pd.DatetimeIndex) -> slice:
    mask = (index >= pd.Timestamp(EVAL_WEEK_START, tz=index.tz)) & (
        index <= pd.Timestamp(EVAL_WEEK_END, tz=index.tz)
    )
    pos = np.flatnonzero(mask)
    return slice(int(pos[0]), int(pos[-1]) + 1)


def _load(run_dir: Path) -> tuple[dict, np.ndarray]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return payload, np.load(run_dir / "predictions.npy").astype(np.float64)


def _try_load(run_dir: Path) -> tuple[dict, np.ndarray] | None:
    """:func:`_load`, or ``None`` when the run has not been produced on this machine."""
    if not (run_dir / "metrics.json").exists() or not (run_dir / "predictions.npy").exists():
        return None
    return _load(run_dir)


def _resolve_area(requested: int | None, areas: list[int], control: dict | None) -> int:
    """The cell the per-day panel covers, derived rather than typed.

    ``--area`` wins when given. Otherwise the calendar-off control run names the failure case by
    construction, because it exists only to re-run one architecture on the one cell whose deficit
    is being explained; falling back to the run's own first area keeps the script usable when no
    control is present. Nothing here hard-codes a square id, which is the rule the areas
    themselves are read from ``square_totals.csv`` for.
    """
    if requested is not None:
        return int(requested)
    if control is not None:
        control_areas = [int(a) for a in control["areas"]]
        if len(control_areas) == 1 and control_areas[0] in areas:
            return control_areas[0]
    return int(areas[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="EXP-008")
    ap.add_argument("--ablation-run", default="EXP-007", help="the calendar-off control")
    ap.add_argument(
        "--area",
        type=int,
        default=None,
        help="area the per-day MAE panel covers; default: the control run's cell, else the "
        "first area of --run",
    )
    ap.add_argument("--zoom", nargs=2, default=("2013-12-19", "2013-12-20"))
    ap.add_argument("--k-worst", type=int, default=8)
    ap.add_argument("--lag", type=int, default=None)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    ap.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    payload, preds = _load(Path(args.runs_dir) / args.run)
    areas, models = list(payload["areas"]), list(payload["models"])
    window = _eval_slice(load_index())
    index = load_index()[window]
    truth = {a: load_series(square_id=a).to_numpy(dtype=np.float64)[window] for a in areas}
    LOG.info("%s: areas %s, %d slots", args.run, areas, len(index))

    control = _try_load(Path(args.runs_dir) / args.ablation_run)
    if control is None:
        LOG.warning(
            "%s has no metrics.json or predictions.npy under %s: the calendar-off control and "
            "the calendar-penalty section are skipped. Produce it with "
            "`python scripts/run_experiment.py --models tcn --no-calendar` to restore them.",
            args.ablation_run,
            args.runs_dir,
        )
    focus = _resolve_area(args.area, areas, control[0] if control else None)
    fi = areas.index(focus)
    named = {DISPLAY.get(m, m): preds[fi, j] for j, m in enumerate(models) if m in DISPLAY}
    LOG.info("per-day panel on square %d", focus)

    ablation, control_covers_focus = None, False
    if control is not None:
        ab_payload, ab_preds = control
        control_covers_focus = [int(a) for a in ab_payload["areas"]] == [focus]
        if not control_covers_focus or list(ab_payload["models"]) != ["tcn"]:
            LOG.warning(
                "%s holds areas %s models %s, not the single (%d, tcn) control expected",
                args.ablation_run,
                ab_payload["areas"],
                ab_payload["models"],
                focus,
            )
        ablation = ab_preds[0, 0]
        if np.allclose(ablation, preds[fi, models.index("tcn")]):
            LOG.warning(
                "%s is identical to the reported TCN on square %d: the calendar-off control is "
                "not a control. Check the run configs.",
                args.ablation_run,
                focus,
            )

    with_ablation = dict(named) if ablation is None else {**named, ABLATION_LABEL: ablation}
    order = ["LSTM", "TCN", ABLATION_LABEL, "GBT", "Persistence", "Linear AR(144)"]
    order = [k for k in order if k in with_ablation]

    per_day = PD.per_period_errors(truth[focus], with_ablation, index)
    labels = list(dict.fromkeys(per_day["period"]))
    daily_mae = per_day.pivot(index="period", columns="model", values="MAE").loc[labels, order]
    by_type = PD.per_period_errors(truth[focus], with_ablation, index, period="daytype")

    reference = "Linear AR(144)"
    excess = PD.excess_error_share(truth[focus], with_ablation["TCN"], named[reference], index)

    zoom_days = pd.to_datetime(list(args.zoom)).tz_localize(index.tz)
    zoom_mask = np.asarray(index.normalize().isin(zoom_days))
    masks = {
        "whole week": np.ones(len(index), dtype=bool),
        "zoom days": zoom_mask,
        "other days": ~zoom_mask,
    }
    dm_linear = PD.compare_on_mask(
        truth[focus], with_ablation["TCN"], named[reference], index, masks, lag=args.lag
    )
    dm_naive = PD.compare_on_mask(
        truth[focus], with_ablation["TCN"], named["Persistence"], index, masks, lag=args.lag
    )

    reason = (
        f"No control run at `{args.runs_dir}/{args.ablation_run}`"
        if control is None
        else f"`{args.ablation_run}` does not cover square {focus}"
    )
    penalty_section: list[str] = [
        f"## Calendar-feature penalty, {', '.join(args.zoom)}",
        "",
        f"{reason}, so the calendar-feature penalty is not computed: a penalty read off a "
        "control trained on another cell would not be one. Re-run with `--ablation-run` "
        "pointing at a calendar-off control for this cell to restore this section.",
    ]
    if ablation is not None and control_covers_focus:
        seeds = [int(s) for s in payload.get("seeds", [])]
        if [int(s) for s in ab_payload.get("seeds", [])] != seeds:
            LOG.warning(
                "seed lists differ between %s (%s) and %s (%s); the per-seed penalty pairs rows "
                "by position, so the pairing is only meaningful if the orders agree",
                args.run,
                seeds,
                args.ablation_run,
                ab_payload.get("seeds"),
            )
        pen_day, pen_seed = PD.calendar_penalty(
            truth[focus],
            with_ablation["TCN"],
            ablation,
            index,
            zoom_mask,
            seeds=seeds,
        )
        zoom_row = pen_day.iloc[-2]
        pooled_share = float(zoom_row["share_of_week_pct"])
        LOG.info(
            "calendar penalty on %s: %.2f percent pooled, per seed %s",
            " + ".join(args.zoom),
            pooled_share,
            ", ".join(f"{v:.2f}" for v in pen_seed["share_of_week_pct"]),
        )
        penalty_section = [
            f"## Calendar-feature penalty, {zoom_row['period']}",
            "",
            f"The penalty at slot *t* is `|y - TCN| - |y - {ABLATION_LABEL}|`: the pointwise "
            "absolute error of the model given the calendar features minus that of the same "
            f"architecture trained without them (`{args.ablation_run}`). Positive means the "
            "features cost accuracy at that slot. Units are scaled CDR activity counts.",
            "",
            f"**{pooled_share:.1f} percent** of the week's net calendar penalty falls on "
            f"{zoom_row['period']}, computed on the seed-mean pointwise penalty. Per seed the "
            "share is "
            + ", ".join(f"{v:.1f}" for v in pen_seed["share_of_week_pct"])
            + f" percent, mean {pen_seed['share_of_week_pct'].mean():.1f}. "
            + (
                "A share above 100 percent means that seed gains on the remaining days, so its "
                "ratio is not bounded and the mean of the three is not a share of anything. "
                if (pen_seed["share_of_week_pct"] > 100.0).any()
                else ""
            )
            + f"Quote the pooled {pooled_share:.1f} percent, or the per-seed range with the "
            "pooled value beside it; the mean of the per-seed shares is not a share.",
            "",
            "### Per day, seed-mean penalty",
            "",
            PD.to_markdown(pen_day),
            "",
            "### Per seed",
            "",
            PD.to_markdown(pen_seed),
        ]
    diagnostics = PD.residual_diagnostics(truth[focus], with_ablation, mask=zoom_mask)
    concentration = {
        a: PD.error_concentration(
            truth[a], {m: preds[i, models.index(m)] for m in WAPE_MODELS}
        ).assign(area=a)
        for i, a in enumerate(areas)
    }
    worst = {
        a: PD.worst_slots(
            truth[a],
            {DISPLAY[m]: preds[i, models.index(m)] for m in WAPE_MODELS},
            index,
            k=args.k_worst,
        ).assign(area=a)
        for i, a in enumerate(areas)
    }

    wape_rows, mae_rows = {}, {}
    for i, a in enumerate(areas):
        frame = PD.per_period_errors(
            truth[a], {DISPLAY[m]: preds[i, models.index(m)] for m in WAPE_MODELS}, index
        )
        wape_rows[a] = frame.groupby("period", sort=False)["WAPE"].mean().loc[labels]
        mae_rows[a] = frame.groupby("period", sort=False)["MAE"].mean().loc[labels]
    daily_wape = pd.DataFrame(wape_rows).loc[labels]
    daily_level = pd.DataFrame(
        {
            a: PD.per_period_errors(truth[a], {"x": truth[a]}, index).set_index("period")[
                "mean_level"
            ]
            for a in areas
        }
    ).loc[labels]

    tables = Path(args.tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    head = [
        f"Source: `experiments/runs/{args.run}/` and `{args.ablation_run}/`.",
        "Generated by `scripts/make_period_breakdown.py`.",
        "",
        PD.SEED_CONVENTION,
        "",
        f"Evaluation week {EVAL_WEEK_START} to {EVAL_WEEK_END} CET, {len(index)} slots.",
        "Values are scaled CDR activity counts; WAPE is a percentage.",
        "",
    ]
    (tables / f"per_period_{focus}.md").write_text(
        "\n".join(
            [
                f"# Per-day error, square {focus}",
                "",
                *head,
                "## Per-day MAE by model",
                "",
                PD.to_markdown(daily_mae.reset_index()),
                "",
                "## By day type",
                "",
                PD.to_markdown(by_type),
                "",
                f"## Where the TCN's deficit to {reference} falls",
                "",
                "`share_of_net` and `share_of_positive` use different denominators and differ",
                "materially; `_min` and `_max` are taken across seeds, so a concentration that",
                "holds only on the seed mean is visible as a wide range.",
                "",
                PD.to_markdown(excess),
                "",
                *penalty_section,
            ]
        ),
        encoding="utf-8",
    )
    (tables / "per_period_wape.md").write_text(
        "\n".join(
            [
                "# Per-day relative and absolute error by area",
                "",
                *head,
                "Mean over the five non-seasonal forecasters. The daily seasonal naive is excluded:",
                "its weekend WAPE on square 5259 exceeds 140 percent and would set the axis.",
                "",
                "## WAPE (%)",
                "",
                PD.to_markdown(daily_wape.reset_index(names="period")),
                "",
                "## MAE, the counterpoint",
                "",
                "Absolute error and relative error move in opposite directions at the weekend on",
                "square 5259, which is why the weekend is a property of the cell and not a model",
                "failure.",
                "",
                PD.to_markdown(pd.DataFrame(mae_rows).loc[labels].reset_index(names="period")),
                "",
                "## Mean level",
                "",
                PD.to_markdown(daily_level.reset_index(names="period")),
            ]
        ),
        encoding="utf-8",
    )
    (tables / "per_period_stats.md").write_text(
        "\n".join(
            [
                f"# Period diagnostics, square {focus}",
                "",
                *head,
                f"## Diebold-Mariano, TCN against {reference}",
                "",
                "A 144-slot or 288-slot subset is short, so the asymptotic p-value on the",
                "restricted periods is indicative rather than exact.",
                "",
                PD.to_markdown(dm_linear, floatfmt="{:.4f}"),
                "",
                "## Diebold-Mariano, TCN against persistence",
                "",
                PD.to_markdown(dm_naive, floatfmt="{:.4f}"),
                "",
                f"## Residual diagnostics on {', '.join(args.zoom)}",
                "",
                "`bias2_share` is the percentage of MSE attributable to the mean error rather",
                "than to dispersion; `corr_resid_diff` is 1.0 by construction for persistence, so",
                "a high value means a model is following the series one step late.",
                "",
                PD.to_markdown(diagnostics, floatfmt="{:.3f}"),
                "",
                "## Error concentration",
                "",
                PD.to_markdown(
                    pd.concat(concentration.values(), ignore_index=True)[
                        ["area", "n_slots", "pct_of_week", "pct_of_error"]
                    ]
                ),
                "",
                "## Worst slots",
                "",
                PD.to_markdown(pd.concat(worst.values(), ignore_index=True)),
            ]
        ),
        encoding="utf-8",
    )
    for name in (f"per_period_{focus}.md", "per_period_wape.md", "per_period_stats.md"):
        LOG.info("wrote %s", tables / name)

    if not args.no_figure:
        viz.use_style()
        zoom_models = [k for k in ("TCN", ABLATION_LABEL, reference) if k in with_ablation]
        panel = [k for k in ("TCN", ABLATION_LABEL, "Persistence", reference) if k in daily_mae]
        fig = viz.plot_per_period_error(
            zoom_truth=truth[focus][zoom_mask],
            zoom_predictions={
                k: np.atleast_2d(with_ablation[k]).mean(axis=0)[zoom_mask] for k in zoom_models
            },
            zoom_index=index[zoom_mask],
            daily_mae=daily_mae[panel],
            daily_wape=daily_wape,
        )
        for path in viz.save_figure(fig, "per_period_error", directory=args.figures_dir):
            LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
