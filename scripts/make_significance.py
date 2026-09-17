"""Turn a finished run's stored predictions into the report's significance evidence.

The per-area tables rank models on gaps of two to three MAE units against seed spreads of the
same size. That is a comparison of point estimates, not a result, so this script tests every
pairwise margin the report claims and writes the outcome as an auditable artefact plus one figure.

Thin CLI: every computation lives in ``milan_traffic.significance``. Re-running overwrites its
own outputs, because these are derived views of an existing run rather than run evidence, and it
appends no row to the experiment log.

    python scripts/make_significance.py --run EXP-008

Outputs: ``reports/tables/significance_dm.md`` and
``reports/figures/dm_intervals.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic import significance as S  # noqa: E402
from milan_traffic import viz  # noqa: E402
from milan_traffic.config import FIGURES_DIR, RUNS_DIR, TABLES_DIR  # noqa: E402
from milan_traffic.dataio import load_index, load_series  # noqa: E402
from milan_traffic.evaluate import EVAL_WEEK_END, EVAL_WEEK_START  # noqa: E402
from milan_traffic.periods import to_markdown  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402

LOG = get_logger("make_significance")

#: Printed names, matching the report's per-area tables. "GBT" rather than the tables' longer
#: label, because a forest-plot row label has to fit a single column.
DISPLAY = {
    "lstm": "LSTM",
    "tcn": "TCN",
    "gbt": "GBT",
    "naive": "persistence",
    "linear_ar_144": "linear AR(144)",
}

#: The three families were fixed before the tests were run, and are written here rather than
#: chosen in the caller so that the pre-specification is visible in the repository. Holm is
#: applied within each, because a family boundary is a claim about what was planned.
FAMILY_TRAINED_VS_LINEAR = "trained vs linear AR(144)"
FAMILY_RANKING_REVERSAL = "TCN vs LSTM"
FAMILY_REFERENCE = "linear AR(144) vs persistence"

FAMILIES: dict[str, tuple[tuple[int, str, str], ...]] = {
    FAMILY_TRAINED_VS_LINEAR: (
        (5059, "lstm", "linear_ar_144"),
        (5059, "tcn", "linear_ar_144"),
        (5059, "gbt", "linear_ar_144"),
        (5161, "lstm", "linear_ar_144"),
        (5161, "tcn", "linear_ar_144"),
        (5161, "gbt", "linear_ar_144"),
        (5259, "lstm", "linear_ar_144"),
        (5259, "tcn", "linear_ar_144"),
        (5259, "gbt", "linear_ar_144"),
    ),
    FAMILY_RANKING_REVERSAL: (
        (5059, "tcn", "lstm"),
        (5161, "tcn", "lstm"),
        (5259, "tcn", "lstm"),
    ),
    FAMILY_REFERENCE: (
        (5059, "linear_ar_144", "naive"),
        (5161, "linear_ar_144", "naive"),
        (5259, "linear_ar_144", "naive"),
    ),
}

#: The conjunction the study's contribution rests on: the TCN-LSTM ordering reverses between
#: these two cells. Stated as (area, favours_first) so the direction cannot drift from the claim.
REVERSAL_LEGS = ((5059, False), (5259, True))

#: The pair whose detection floor is drawn as the grey bar in the figure. The minimum detectable
#: difference is ``(z_.975 + z_.80) * se`` of one differential, so it belongs to a comparison and
#: not to a cell: on square 5059 it is 10.67 units for TCN versus linear AR(144) but 6.32 for
#: LSTM and 6.03 for GBT against the same reference. The TCN pair is drawn because the study's
#: headline claim is about the TCN; every comparison's own floor is the `mdd` column of the
#: generated table, and the note written beside the figure's inputs says which pair the bar is.
MDD_PAIR = ("tcn", "linear_ar_144")


def _check_prespecified_areas(areas: list[int]) -> None:
    """Fail loudly if the run's areas are not the ones :data:`FAMILIES` was written against.

    The families are hard-coded so that the pre-specification is visible in the repository, which
    only helps if it cannot silently outlive the run it was written for. A run over different
    squares would otherwise produce a correctly computed table with the wrong ids printed in it.
    """
    spec = sorted({area for pairs in FAMILIES.values() for area, _, _ in pairs})
    legs = sorted({area for area, _ in REVERSAL_LEGS})
    if spec != sorted(areas) or not set(legs) <= set(spec):
        raise SystemExit(
            f"Pre-specified areas {spec} (reversal legs {legs}) do not match the run's "
            f"{sorted(areas)}. Edit FAMILIES and REVERSAL_LEGS deliberately, or point --run at "
            "the run they were written for."
        )


def _lag_sensitive(frame: pd.DataFrame, lag_cols: list[str], *, alpha: float = 0.05) -> list[str]:
    """One phrase per row whose verdict at ``alpha`` is not the same at every lag in the grid.

    The report claims that only two of the fifteen conclusions are lag-sensitive and that both
    are named. Deriving them here rather than transcribing them means the claim is recomputed
    with the table and cannot drift from it.
    """
    named = []
    for row in frame.itertuples(index=False):
        ps = [float(getattr(row, col)) for col in lag_cols]
        if (min(ps) < alpha) == (max(ps) < alpha):
            continue
        col, p = next((c, v) for c, v in zip(lag_cols, ps, strict=True) if v >= alpha)
        lag = col.removeprefix("p_lag_")
        named.append(
            f"square {int(row.area)}, `{row.model_a}` versus `{row.model_b}` "
            f"(p = {p:.4f} at lag {lag})"
        )
    return named


def _eval_slice(index: pd.DatetimeIndex) -> slice:
    mask = (index >= pd.Timestamp(EVAL_WEEK_START, tz=index.tz)) & (
        index <= pd.Timestamp(EVAL_WEEK_END, tz=index.tz)
    )
    pos = np.flatnonzero(mask)
    return slice(int(pos[0]), int(pos[-1]) + 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="EXP-008")
    ap.add_argument("--lag", type=int, default=None, help="Newey-West lag; default is the rule.")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--block", type=int, default=144)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    ap.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.run
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    preds = np.load(run_dir / "predictions.npy").astype(np.float64)
    areas, models = list(payload["areas"]), list(payload["models"])
    _check_prespecified_areas(areas)
    LOG.info("%s: areas %s, models %s, seeds %s", args.run, areas, models, payload["seeds"])

    window = _eval_slice(load_index())
    truth = {a: load_series(square_id=a).to_numpy(dtype=np.float64)[window] for a in areas}
    by_area = {a: {m: preds[i, j] for j, m in enumerate(models)} for i, a in enumerate(areas)}

    frames, adjusted = [], []
    for family, pairs in FAMILIES.items():
        rows = []
        for area, a, b in pairs:
            frame = S.compare(
                truth[area],
                by_area[area],
                [(a, b)],
                lag=args.lag,
                block=args.block,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            frame.insert(0, "area", area)
            frame.insert(0, "family", family)
            rows.append(frame)
        family_frame = pd.concat(rows, ignore_index=True)
        absolute = family_frame[family_frame["loss"] == "absolute"].copy()
        absolute["p_holm_family"] = S.holm(absolute["p"].to_numpy())
        squared = family_frame[family_frame["loss"] == "squared"].copy()
        squared["p_holm_family"] = np.nan
        frames.append(pd.concat([absolute, squared], ignore_index=True))
        adjusted.append(absolute)

    full = pd.concat(frames, ignore_index=True)
    headline = pd.concat(adjusted, ignore_index=True)
    headline["p_holm_pooled"] = S.holm(headline["p"].to_numpy())

    results = {}
    for area, favours_first in REVERSAL_LEGS:
        d = S.seed_averaged_loss(truth[area], by_area[area]["tcn"]) - S.seed_averaged_loss(
            truth[area], by_area[area]["lstm"]
        )
        results[area] = (S.dm_test(d, lag=args.lag), favours_first)
    iu_p = S.intersection_union_p(
        [r for r, _ in results.values()], [f for _, f in results.values()]
    )
    LOG.info("intersection-union p for the TCN/LSTM reversal: %.5f", iu_p)

    show = [
        "family",
        "area",
        "model_a",
        "model_b",
        "d",
        "se_hac",
        "ci_low",
        "ci_high",
        "t",
        "p",
        "p_holm_family",
        "p_holm_pooled",
        "mdd",
        "required_slots",
        "boot_ci_low",
        "boot_ci_high",
        "acf_lag1",
        "ljung_box_p",
        "andrews_lag",
        "ens_d",
    ]
    lag_cols = [c for c in full.columns if c.startswith("p_lag_")]
    lag_frame = full[full["loss"] == "absolute"][
        ["area", "model_a", "model_b", *lag_cols]
    ].reset_index(drop=True)
    sensitive = _lag_sensitive(lag_frame, lag_cols)
    legs = " and ".join(str(a) for a, _ in REVERSAL_LEGS)
    lines = [
        "# Predictive-accuracy tests, evaluation week",
        "",
        f"Source: `experiments/runs/{args.run}/`. Generated by `scripts/make_significance.py`.",
        "",
        "Diebold-Mariano on paired **seed-averaged** pointwise losses, so the quantity tested is",
        "exactly the seed-mean metric the per-area tables print. Bartlett kernel at Newey-West lag",
        f"{S.newey_west_lag(len(truth[areas[0]]))} (`{S.NW_LAG_RULE}`), Harvey-Leybourne-Newbold",
        "correction, **standard normal** reference. statsmodels uses no Student t correction with",
        "a robust covariance, so `p` and the interval are normal-based; a t reference on 1007",
        "degrees of freedom would move a p-value in this table by at most 3.0e-04. Negative `d`",
        "favours the first model named.",
        "",
        f"Intersection-union p for the TCN/LSTM reversal between squares {legs}: **{iu_p:.5f}**.",
        "Its components are **one-sided**, because each leg of the conjunction asserts a direction",
        "rather than a difference; the lag table below prints two-sided p, so a component read off",
        "it is half the printed value. No multiplicity correction applies to the joint test: the",
        "claim is a conjunction, so its size is bounded by the largest component (Berger, 1982).",
        "",
        "## Absolute-error loss, with Holm adjustment within and across families",
        "",
        to_markdown(headline[[c for c in show if c in headline.columns]], floatfmt="{:.4f}"),
        "",
        textwrap.fill(
            "`mdd` is the minimum detectable difference at 80 percent power. It belongs to the "
            "comparison and not to the cell, so the single grey bar behind each area's rows in "
            f"`reports/figures/dm_intervals` is the `{MDD_PAIR[0]}` versus `{MDD_PAIR[1]}` floor "
            "for that area; every other row's own floor is the value in this column.",
            width=88,
        ),
        "",
        "## Lag sensitivity (two-sided p at each Newey-West lag)",
        "",
        to_markdown(lag_frame, floatfmt="{:.4f}"),
        "",
        textwrap.fill(
            f"{len(sensitive)} of the {len(lag_frame)} conclusions change at the 0.05 level "
            f"somewhere in this grid: {'; '.join(sensitive)}. Every other row keeps its verdict "
            "from lag 0 to lag 144.",
            width=88,
        ),
        "",
        "## Squared-error loss, as a robustness check against RMSE",
        "",
        to_markdown(
            full[full["loss"] == "squared"][
                ["area", "model_a", "model_b", "d", "t", "p"]
            ].reset_index(drop=True),
            floatfmt="{:.4f}",
        ),
    ]
    out = Path(args.tables_dir) / "significance_dm.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s", out)

    if not args.no_figure:
        viz.use_style()
        plot = headline[headline["family"] != FAMILY_REFERENCE].reset_index(drop=True)
        mdd_keys = plot.copy()
        plot["model_a"] = plot["model_a"].map(lambda m: DISPLAY.get(m, m))
        plot["model_b"] = plot["model_b"].map(lambda m: DISPLAY.get(m, m))
        mdd = {
            int(a): float(
                mdd_keys[
                    (mdd_keys.area == a)
                    & (mdd_keys.model_a == MDD_PAIR[0])
                    & (mdd_keys.model_b == MDD_PAIR[1])
                ]["mdd"].iloc[0]
            )
            for a in areas
        }
        # One bar per area, and it is one pair's floor rather than the area's: the caption and
        # the table both have to say so, because the figure draws it behind every row.
        LOG.info("grey bar = %s vs %s minimum detectable difference: %s", *MDD_PAIR, mdd)
        fig = viz.plot_dm_intervals(plot, mdd_by_area=mdd)
        for path in viz.save_figure(fig, "dm_intervals", directory=args.figures_dir):
            LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
