"""Plot styling, figure export, and every figure the report uses.

Figures go to ``reports/figures/`` as vector PDF (for the report) and PNG (for the
slides) in one call, at a size that stays legible in a two-column layout. Saving rather
than showing is deliberate: every figure in the report must be regenerable by running
the code, not by re-executing a notebook cell from memory.

Each plotting function documents *why the figure exists*, that is, which claim in the
report it evidences. The brief says a figure the text cannot interpret should be cut, so
a function without such a claim does not belong here. Functions take plain arrays and
frames, never file paths, so the same code serves the real store, the tests' synthetic
data and, later, the experiment outputs.

Conventions shared by every figure:

* a figure is drawn at the width it is *placed* at in the report, :data:`COLUMN_WIDTH_IN`
  for one text column and :data:`PAGE_WIDTH_IN` for a figure spanning both, so the
  typesetter scales the PDF 1:1 and a 7 pt label in the figure is 7 pt on the page. A
  wide figure shrunk into a column loses its labels: at the old 7 in width a 7 pt legend
  landed at 3.5 pt;
* colours are assigned per entity (a grid cell, a model) in a fixed order from
  :data:`PALETTE`, so the same cell keeps the same colour across the report;
* the y axis is always the raw, unitless scaled CDR count, never a physical unit;
* time axes are drawn in local wall-clock time. The tz is stripped before plotting so
  matplotlib does not silently relabel midnight as 23:00 UTC;
* no dual y axes. Two measures of different scale get two panels or a common index.

The two small data-preparation helpers at the end (:func:`grid_totals`,
:func:`weekday_means_by_iso_week`, :func:`daily_peak_table`) live here rather than in a
script so that they are tested and so that ``scripts/make_figures.py`` stays glue.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory
from numpy.typing import ArrayLike

from .config import FIGURES_DIR, GRID_SIDE, SLOTS_PER_DAY, SLOTS_PER_WEEK

#: Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

#: Width of one text column of the two-column A4 report, in inches (244.9 pt, 8.64 cm:
#: A4 less 1.5 cm margins, halved, less the column gutter). Figures placed at column
#: width are drawn at this width so the typesetter neither shrinks nor stretches them.
COLUMN_WIDTH_IN = 3.4

#: Width of the whole text block, both columns and the gutter (510.2 pt, 18 cm). Only
#: for figures the report places with ``scope: "parent"``.
PAGE_WIDTH_IN = 7.0


def use_style(*, base_size: int = 8) -> None:
    """Apply the project's matplotlib defaults. Call once per notebook or script.

    ``base_size`` is read as a size *on the printed page*, because figures are drawn at
    the width they are placed at (see :data:`COLUMN_WIDTH_IN`). At the default the
    smallest text set from these defaults, tick labels and legends, is 7 pt on the page.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            # A column-width figure is half as wide as the old full-width one, so 300 dpi
            # would halve the pixel width of the PNGs the slides use. 400 keeps them
            # above 1300 px without doubling the file size.
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            # The default 0.1 in of padding on each side is 6 percent of a column-width
            # figure, so it shrank everything by that much once the page scaled the PDF
            # back to the column. The report's own figure gap supplies the white space.
            "savefig.pad_inches": 0.02,
            "font.size": base_size,
            "axes.titlesize": base_size + 1,
            "axes.labelsize": base_size,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": base_size - 1,
            "lines.linewidth": 1.2,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
        }
    )


def save_figure(
    fig: plt.Figure,
    name: str,
    *,
    formats: Iterable[str] = ("pdf", "png"),
    directory: Path | str = FIGURES_DIR,
) -> list[Path]:
    """Write ``fig`` to ``directory/name.<ext>`` for each format. Returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = directory / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(path)
    return paths


# ------------------------------------------------------------------ shared constants

#: Public holidays inside the observation window (national unless noted). Non-working
#: weekdays behave like weekends in most cells, so they are shaded and named wherever a
#: calendar effect is the point of the figure.
HOLIDAYS: dict[str, str] = {
    "2013-11-01": "Ognissanti",
    "2013-12-07": "Sant'Ambrogio (Milan only)",
    "2013-12-08": "Immacolata",
    "2013-12-25": "Natale",
    "2013-12-26": "Santo Stefano",
    "2014-01-01": "Capodanno",
}

Y_LABEL = "Internet activity (scaled CDR count)"
ACTUAL_COLOUR = "#333333"
WEEKEND_COLOUR = "#7f7f7f"
HOLIDAY_COLOUR = PALETTE[4]
HIGHLIGHT_COLOUR = PALETTE[1]
#: Single-hue ramp for magnitude. Never a rainbow: lightness alone must carry the value.
SEQUENTIAL_CMAP = "Blues"


# --------------------------------------------------------------------------- helpers


def colour_for(entities: Sequence[object]) -> dict[object, str]:
    """Fixed ``entity -> colour`` mapping in the order given.

    Colours follow the entity, never its rank or position in a filtered list, and the
    palette is never cycled: a ninth series is a design problem, not a colour problem.
    """
    if len(entities) > len(PALETTE):
        raise ValueError(f"{len(entities)} series but only {len(PALETTE)} palette colours.")
    return {e: PALETTE[i] for i, e in enumerate(entities)}


def _local_naive(index: ArrayLike) -> pd.DatetimeIndex:
    """Local wall-clock time with the tz dropped, so matplotlib labels it as-is."""
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _thousands(axis: mpl.axis.Axis) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def _day_ticks_at_noon(ax: Axes, fmt: str = "%a\n%d %b", *, label_every: int = 1) -> None:
    """Tick marks at midnight, labels centred on the day they describe.

    ``label_every`` labels only every n-th day. The tick marks stay daily, so the day
    grid is unchanged; only the labels thin out, which is what a narrow axis needs when
    a fortnight of days will not fit side by side.
    """
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
    stamp = mdates.DateFormatter(fmt)
    if label_every > 1:
        ax.xaxis.set_minor_formatter(
            FuncFormatter(
                lambda v, _: stamp(v) if mdates.num2date(v).toordinal() % label_every == 0 else ""
            )
        )
    else:
        ax.xaxis.set_minor_formatter(stamp)
    ax.tick_params(axis="x", which="minor", length=0)


def shade_non_working_days(
    ax: Axes,
    index: ArrayLike,
    *,
    holidays: Mapping[str, str] = HOLIDAYS,
    label_holidays: bool = False,
) -> None:
    """Grey spans over Saturdays and Sundays, amber over public holidays.

    Drawn behind the data (``zorder=0``) so the series stays legible. Holidays are
    checked first: a holiday that falls on a weekday is the interesting case.
    """
    idx = _local_naive(index)
    one_day = pd.Timedelta(days=1)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for day in pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D"):
        key = day.strftime("%Y-%m-%d")
        if key in holidays:
            ax.axvspan(day, day + one_day, color=HOLIDAY_COLOUR, alpha=0.2, lw=0, zorder=0)
            if label_holidays:
                ax.text(
                    day + one_day / 2,
                    0.96,
                    f"{day:%a %d %b}: {holidays[key]}",
                    transform=trans,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#6b4300",
                )
        elif day.dayofweek >= 5:
            ax.axvspan(day, day + one_day, color=WEEKEND_COLOUR, alpha=0.12, lw=0, zorder=0)


# ------------------------------------------------------------ concentration statistics


def lorenz_curve(totals: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """``(share_of_cells, share_of_traffic)`` with cells sorted ascending, both from 0."""
    t = np.sort(np.asarray(totals, dtype=np.float64).ravel())
    if t.size == 0 or t.sum() <= 0:
        raise ValueError("Lorenz curve needs a non-empty vector with a positive total.")
    x = np.linspace(0.0, 1.0, t.size + 1)
    y = np.concatenate([[0.0], np.cumsum(t) / t.sum()])
    return x, y


def gini(totals: ArrayLike) -> float:
    """Gini coefficient of the per-cell totals: 0 is uniform, 1 is one cell with everything."""
    x, y = lorenz_curve(totals)
    area = float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))
    return 1.0 - 2.0 * area


def top_share(totals: ArrayLike, frac: float = 0.01) -> float:
    """Share of the total carried by the top ``frac`` of cells."""
    t = np.sort(np.asarray(totals, dtype=np.float64).ravel())[::-1]
    k = max(1, int(round(frac * t.size)))
    return float(t[:k].sum() / t.sum())


def cells_to_share(totals: ArrayLike, share: float = 0.5) -> int:
    """Smallest number of top cells whose combined total reaches ``share`` of the whole."""
    t = np.sort(np.asarray(totals, dtype=np.float64).ravel())[::-1]
    cum = np.cumsum(t) / t.sum()
    return int(np.searchsorted(cum, share) + 1)


# ------------------------------------------------------------------- EDA figures 1-5


def plot_total_traffic_distribution(totals: ArrayLike, *, top_frac: float = 0.01) -> Figure:
    """Why: the report claims traffic is spatially concentrated, and the claim needs a number.

    Panel (a) is the histogram of per-cell totals on a log axis, with the median and the
    top-``top_frac`` cut-off marked. Panel (b) is the Lorenz curve with the Gini
    coefficient, the share carried by the top cells, and the number of cells needed to
    reach half of all traffic. Every annotated figure is computed from ``totals``.

    The panels are stacked rather than side by side: at column width two panels abreast
    leave 1.6 in each, which neither the log-scale x labels nor the two callouts fit.
    """
    t = np.asarray(totals, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    n = t.size
    g = gini(t)
    share = top_share(t, top_frac)
    k_half = cells_to_share(t, 0.5)
    pct = f"{top_frac * 100:g} %"

    fig, (ax_h, ax_l) = plt.subplots(
        2,
        1,
        figsize=(COLUMN_WIDTH_IN, 4.7),
        layout="constrained",
        height_ratios=[1.0, 1.3],
    )

    positive = t[t > 0]
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60)
    ax_h.hist(positive, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.3)
    ax_h.set_xscale("log")
    median = float(np.median(t))
    cut = float(np.sort(t)[::-1][max(1, int(round(top_frac * n))) - 1])
    ax_h.axvline(median, color="#555555", ls="--", lw=0.9, label=f"median {median:,.0f}")
    ax_h.axvline(
        cut, color=HIGHLIGHT_COLOUR, ls="--", lw=0.9, label=f"top {pct} cut-off {cut:,.0f}"
    )
    ax_h.set_xlabel("Total Internet activity per cell, 62 days\n(scaled CDR count, log scale)")
    ax_h.set_ylabel("Number of cells")
    ax_h.set_title("(a) Per-cell totals", loc="left")
    ax_h.legend(loc="upper left")

    x, y = lorenz_curve(t)
    ax_l.plot([0, 1], [0, 1], color="#9a9a9a", lw=0.8, label="perfect equality")
    ax_l.plot(x, y, color=PALETTE[0], lw=1.4, label="observed")
    x_top = 1.0 - top_frac
    y_top = float(np.interp(x_top, x, y))
    ax_l.plot([x_top], [y_top], "o", color=HIGHLIGHT_COLOUR, ms=5, mec="white", mew=0.8, zorder=4)
    ax_l.annotate(
        f"top {pct} of cells carry\n{share * 100:.2f} % of traffic",
        xy=(x_top, y_top),
        xytext=(0.20, 0.84),
        textcoords="axes fraction",
        fontsize=7.5,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#555555"},
    )
    x_half = 1.0 - k_half / n
    ax_l.plot([x_half], [0.5], "o", color=PALETTE[2], ms=5, mec="white", mew=0.8, zorder=4)
    ax_l.annotate(
        f"{k_half:,} cells ({k_half / n * 100:.1f} %)\ncarry half of all traffic",
        xy=(x_half, 0.5),
        xytext=(0.10, 0.62),
        textcoords="axes fraction",
        fontsize=7.5,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#555555"},
    )
    ax_l.text(
        0.03, 0.96, f"Gini = {g:.3f}", transform=ax_l.transAxes, ha="left", va="top", fontsize=8.5
    )
    ax_l.set_xlabel("Share of cells, sorted by increasing total")
    ax_l.set_ylabel("Cumulative share of Internet activity")
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.set_aspect("equal")
    ax_l.set_title("(b) Lorenz curve", loc="left")
    ax_l.legend(loc="lower right")
    return fig


def plot_grid_heatmap(
    totals: ArrayLike,
    square_ids: ArrayLike,
    highlight: Sequence[int] = (5161, 5059, 5259, 4159, 4556),
    *,
    zoom_pad: int = 4,
) -> Figure:
    """Why: the report must show where the study cells sit and that the top three are not
    one contiguous hotspot.

    Panel (a) is the full 100 x 100 grid of ``log1p`` totals; the study cells are ringed
    and the zoom window outlined. Panel (b) zooms on that window with one cell per square
    so that adjacency can be read directly, and carries the id labels (labelling five
    cells that are two cells apart on a 100-cell axis would be illegible).

    Orientation: rows are drawn with row 0 at the **bottom** (``origin="lower"``). The
    data dictionary records only that ids are row-major (``row = (id-1)//100``,
    ``col = (id-1)%100``); the source grid's GeoJSON places square 1 in the south-west
    corner with ids increasing eastward then northward, so this orientation reads
    north-up if that holds. The axes are therefore labelled "grid row" and "grid column",
    not with compass points, and no landmark is asserted.

    Both panels are square, so side by side at column width they would be 1.2 in across
    and the id labels in (b) would overlap the cells they point at. They are stacked.
    """
    tot = np.asarray(totals, dtype=np.float64).ravel()
    ids = np.asarray(square_ids).astype(int).ravel()
    if tot.size != ids.size:
        raise ValueError("totals and square_ids must have the same length.")
    grid = np.full((GRID_SIDE, GRID_SIDE), np.nan)
    grid[(ids - 1) // GRID_SIDE, (ids - 1) % GRID_SIDE] = np.log1p(tot)
    norm = Normalize(vmin=float(np.nanmin(grid)), vmax=float(np.nanmax(grid)))

    h_rows = [(s - 1) // GRID_SIDE for s in highlight]
    h_cols = [(s - 1) % GRID_SIDE for s in highlight]
    half = max(max(h_rows) - min(h_rows), max(h_cols) - min(h_cols)) / 2 + zoom_pad
    r_mid = (max(h_rows) + min(h_rows)) / 2
    c_mid = (max(h_cols) + min(h_cols)) / 2
    zr0, zr1 = max(0, int(np.floor(r_mid - half))), min(GRID_SIDE - 1, int(np.ceil(r_mid + half)))
    zc0, zc1 = max(0, int(np.floor(c_mid - half))), min(GRID_SIDE - 1, int(np.ceil(c_mid + half)))

    fig, (ax, axz) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH_IN, 5.8), layout="constrained", height_ratios=[1.0, 1.0]
    )
    im = ax.imshow(grid, origin="lower", cmap=SEQUENTIAL_CMAP, norm=norm, interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row (0 at the bottom)")
    ax.set_title("(a) All 10 000 cells, log(1 + total)", loc="left")
    ax.add_patch(
        Rectangle(
            (zc0 - 0.5, zr0 - 0.5),
            zc1 - zc0 + 1,
            zr1 - zr0 + 1,
            fill=False,
            ec="#222222",
            lw=0.8,
            label="zoom window (b)",
        )
    )
    ax.plot(
        h_cols,
        h_rows,
        "o",
        mfc="none",
        mec=HIGHLIGHT_COLOUR,
        mew=1.1,
        ms=5.5,
        ls="none",
        label="study cells",
    )
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9, edgecolor="none")

    sub = grid[zr0 : zr1 + 1, zc0 : zc1 + 1]
    axz.imshow(
        sub,
        origin="lower",
        cmap=SEQUENTIAL_CMAP,
        norm=norm,
        interpolation="nearest",
        extent=(zc0 - 0.5, zc1 + 0.5, zr0 - 0.5, zr1 + 0.5),
    )
    axz.set_xticks(np.arange(zc0, zc1 + 1, 2))
    axz.set_yticks(np.arange(zr0, zr1 + 1, 2))
    axz.set_xticks(np.arange(zc0 - 0.5, zc1 + 1, 1.0), minor=True)
    axz.set_yticks(np.arange(zr0 - 0.5, zr1 + 1, 1.0), minor=True)
    axz.grid(which="major", visible=False)
    axz.grid(which="minor", color="white", lw=0.4, alpha=0.9)
    axz.tick_params(which="minor", length=0)
    axz.set_xlabel("Grid column")
    axz.set_ylabel("Grid row")
    axz.set_title("(b) Zoom on the study cells", loc="left")
    offsets = [
        (16, 0, "left"),
        (-16, -7, "right"),
        (-16, 7, "right"),
        (0, -16, "center"),
        (-16, 0, "right"),
    ]
    for i, (s, r, c) in enumerate(zip(highlight, h_rows, h_cols, strict=True)):
        axz.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=HIGHLIGHT_COLOUR, lw=1.3))
        dx, dy, ha = offsets[i % len(offsets)]
        axz.annotate(
            str(s),
            xy=(c, r),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#222222",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#333333"},
        )

    cbar = fig.colorbar(im, ax=[ax, axz], shrink=0.9, pad=0.02)
    cbar.set_label("log(1 + total Internet activity), 62 days")
    return fig


def plot_five_cells_two_weeks(
    series_by_square: Mapping[int, ArrayLike],
    index: ArrayLike,
    *,
    labels: Mapping[int, str] | None = None,
    holidays: Mapping[str, str] = HOLIDAYS,
) -> Figure:
    """Why: the brief requires the top-3 and squares 4159 and 4556 over the first two weeks,
    and the report's cross-area argument (different daily *shapes*, not just levels)
    rests on this figure.

    One panel per cell, x shared, y **not** shared: levels differ five-fold and the point
    is the shape of each series, so each panel gets its own scale. Weekends are shaded
    grey and public holidays amber; Fri 1 Nov 2013 (Ognissanti) is the holiday that
    makes a weekday look like a Sunday in the working-district cells.
    """
    squares = list(series_by_square)
    colours = colour_for(squares)
    idx = _local_naive(index)
    n = len(squares)
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        figsize=(COLUMN_WIDTH_IN, 0.95 * n + 1.0),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[:, 0]
    for ax, s in zip(axes, squares, strict=True):
        y = np.asarray(series_by_square[s], dtype=np.float64)
        if y.size != idx.size:
            raise ValueError(f"square {s}: {y.size} values but {idx.size} timestamps.")
        ax.plot(idx, y, color=colours[s], lw=0.9)
        shade_non_working_days(ax, idx, holidays=holidays)
        title = labels.get(s, f"square {s}") if labels else f"square {s}"
        ax.set_title(title, loc="left", fontsize=8, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        _thousands(ax.yaxis)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
    # A fortnight of day labels needs about 4 in at 7 pt; the axis has under 3. Every
    # second day is labelled, the tick marks stay daily and the shading carries the rest.
    _day_ticks_at_noon(axes[-1], label_every=2)
    fig.supylabel(Y_LABEL, fontsize=8)
    present = [
        f"{pd.Timestamp(key):%a %d %b}: {name}"
        for key, name in holidays.items()
        if idx[0].normalize() <= pd.Timestamp(key) <= idx[-1].normalize()
    ]
    holiday_label = "public holiday" + (f" ({'; '.join(present)})" if present else "")
    handles = [
        Patch(fc=WEEKEND_COLOUR, alpha=0.3, label="Saturday / Sunday"),
        Patch(fc=HOLIDAY_COLOUR, alpha=0.45, label=holiday_label),
    ]
    # One column per entry: the holiday label names the holiday, so the two side by side
    # are wider than the column. Constrained layout reserves the strip above the panels.
    fig.legend(handles=handles, loc="outside upper left", ncol=1, fontsize=7)
    return fig


def _iso_monday(week: int, year: int, base_week: int) -> date:
    """Monday of ISO ``week``; weeks far below ``base_week`` belong to the following year."""
    yy = year + 1 if (base_week - week) > 26 else year
    return date.fromisocalendar(yy, week, 1)


def plot_weekly_weekday_means(
    table: pd.DataFrame,
    *,
    base_week: int = 45,
    highlight_week: int | None = 51,
    year: int = 2013,
    grid_column: str = "grid",
) -> Figure:
    """Why: the training period is not stationary. Traffic drifts down towards Christmas,
    and the drift is not the same everywhere, so a model fitted on November sees a level
    the evaluation week no longer has.

    ``table`` holds the Mon to Fri mean per 10-minute slot, one row per ISO week, one
    column per series (``grid_column`` for the whole grid, the rest for single cells).
    Everything is indexed to ``base_week`` = 1 on one axis (no dual scales), and the
    legend states each series' change by ``highlight_week``, the evaluation week.
    """
    if base_week not in table.index:
        raise ValueError(f"base week {base_week} not in table index {list(table.index)}.")
    norm = table.div(table.loc[base_week], axis=1)
    weeks = [int(w) for w in norm.index]
    last = highlight_week if highlight_week in norm.index else int(norm.index[-1])
    cells = [c for c in norm.columns if c != grid_column]
    colours = colour_for(cells)

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.0), layout="constrained")
    if highlight_week is not None and highlight_week in norm.index:
        ax.axvspan(
            highlight_week - 0.5, highlight_week + 0.5, color=WEEKEND_COLOUR, alpha=0.12, lw=0
        )
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            highlight_week,
            0.98,
            "evaluation\nweek",
            transform=trans,
            ha="center",
            va="top",
            fontsize=7,
            color="#555555",
        )
    ax.axhline(1.0, color="#9a9a9a", lw=0.8, zorder=1)
    for col in norm.columns:
        change = (float(norm.loc[last, col]) - 1.0) * 100.0
        name = "all 10 000 cells" if col == grid_column else f"square {col}"
        label = f"{name} ({change:+.1f} % by week {last})"
        if col == grid_column:
            ax.plot(
                weeks, norm[col], color="#000000", lw=1.8, marker="o", ms=4, label=label, zorder=3
            )
        else:
            ax.plot(weeks, norm[col], color=colours[col], lw=1.1, marker="o", ms=3.2, label=label)
    ax.set_xticks(weeks)
    ax.set_xticklabels([f"W{w}\n{_iso_monday(w, year, base_week):%d %b}" for w in weeks])
    ax.set_xlabel("ISO week (label gives the Monday)")
    ax.set_ylabel(f"Mon to Fri mean per 10-min slot,\nrelative to week {base_week} (= 1)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    return fig


def plot_grid_spike(
    grid_totals: ArrayLike,
    index: ArrayLike,
    *,
    center_row: int = 5165,
    halo: int = 9,
    reference_lag: int | None = SLOTS_PER_WEEK,
) -> Figure:
    """Why: the largest single-slot jump in the whole series is a one-slot event at
    Fri 6 Dec 2013 20:50, and a one-step-ahead model cannot anticipate it. It is the
    named failure case for every model and needs to be seen, not described.

    Plots the grid total for ``halo`` slots either side of ``center_row``, with the same
    slots ``reference_lag`` slots earlier as a dashed reference (one week by default, so
    the weekday and time of day match). Annotates the spike as a multiple of the mean of
    its two neighbours and the dip in the following slot, both computed from the data.
    """
    g = np.asarray(grid_totals, dtype=np.float64).ravel()
    idx = _local_naive(index)
    lo, hi = center_row - halo, center_row + halo + 1
    if lo < 0 or hi > g.size or g.size != idx.size:
        raise ValueError("center_row/halo fall outside the series, or index length mismatch.")
    t, y = idx[lo:hi], g[lo:hi]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.9), layout="constrained")
    if reference_lag and lo - reference_lag >= 0:
        ref_day = idx[center_row - reference_lag]
        ax.plot(
            t,
            g[lo - reference_lag : hi - reference_lag],
            color="#9a9a9a",
            lw=1.0,
            ls="--",
            marker="o",
            ms=2.5,
            label=f"same slots {reference_lag // SLOTS_PER_DAY} days earlier ({ref_day:%a %d %b})",
        )
    ax.plot(
        t,
        y,
        color=PALETTE[0],
        lw=1.3,
        marker="o",
        ms=3,
        label=f"grid total, {idx[center_row]:%a %d %b %Y}",
    )

    neighbours = (g[center_row - 1] + g[center_row + 1]) / 2.0
    ratio = g[center_row] / neighbours
    dip = (g[center_row + 1] / g[center_row - 1] - 1.0) * 100.0
    # The spike sits mid-axis, so the callouts are pulled in close: the offsets are in
    # points and the axis is only about 200 pt wide.
    ax.annotate(
        f"{idx[center_row]:%H:%M}: {ratio:.2f}x the mean\nof the adjacent slots",
        xy=(t[halo], y[halo]),
        xytext=(9, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#555555"},
    )
    ax.annotate(
        f"{idx[center_row + 1]:%H:%M}: {dip:+.0f} % vs {idx[center_row - 1]:%H:%M}",
        xy=(t[halo + 1], y[halo + 1]),
        xytext=(8, -20),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#555555"},
    )
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel(f"Local time, {idx[center_row]:%A %d %B %Y}")
    ax.set_ylabel("Grid total per 10-min slot\n(scaled CDR count)")
    _thousands(ax.yaxis)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    return fig


# ------------------------------------------------------- experiment figures 6-8 (later)


def plot_actual_vs_predicted(
    y_true: ArrayLike,
    preds_by_model: Mapping[str, ArrayLike],
    index: ArrayLike,
    area_ids: Sequence[int],
) -> Figure:
    """Why: a metrics table says *how much* a model misses; only the overlay says *where*
    (peaks, regime changes, the spike), which is what the failure analysis is built on.

    Rows are areas, columns are models, y is shared within a row so models on the same
    area are directly comparable. ``y_true`` is ``(n_areas, n)`` and each entry of
    ``preds_by_model`` is ``(n_areas, n)``; a 1-D input is treated as one area.
    """
    Y = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    idx = _local_naive(index)
    n_areas, n = Y.shape
    if n != idx.size or n_areas != len(area_ids):
        raise ValueError("y_true must be (n_areas, len(index)) with one row per area id.")
    models = list(preds_by_model)
    colours = colour_for(models)

    fig, axes = plt.subplots(
        n_areas,
        len(models),
        sharex=True,
        sharey="row",
        figsize=(PAGE_WIDTH_IN, 1.7 * n_areas + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for i, area in enumerate(area_ids):
        for j, m in enumerate(models):
            ax = axes[i, j]
            P = np.atleast_2d(np.asarray(preds_by_model[m], dtype=np.float64))
            ax.plot(idx, Y[i], color=ACTUAL_COLOUR, lw=0.7)
            ax.plot(idx, P[i], color=colours[m], lw=0.7, alpha=0.9)
            if i == 0:
                ax.set_title(m, loc="left", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"square {area}\n(scaled CDR count)", fontsize=8)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            _thousands(ax.yaxis)
            ax.margins(x=0)
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d"))
    handles = [Line2D([], [], color=ACTUAL_COLOUR, lw=1.2, label="actual")]
    handles += [Line2D([], [], color=colours[m], lw=1.2, label=f"{m} (predicted)") for m in models]
    fig.legend(handles=handles, loc="outside upper right", ncol=len(handles), fontsize=7)
    fig.supxlabel(f"{idx[0]:%d %b} to {idx[-1]:%d %b %Y}, one-step-ahead forecasts", fontsize=8)
    return fig


def _resolve_span(zoom: slice | tuple[object, object], idx: pd.DatetimeIndex) -> tuple[int, int]:
    if isinstance(zoom, slice):
        lo, hi, _ = zoom.indices(idx.size)
        return lo, hi
    start, end = _local_naive(pd.DatetimeIndex([pd.Timestamp(zoom[0]), pd.Timestamp(zoom[1])]))
    return int(idx.searchsorted(start)), int(idx.searchsorted(end, side="right"))


def plot_actual_vs_predicted_single(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    index: ArrayLike,
    *,
    area_id: int,
    model_name: str,
    zoom: slice | tuple[object, object] | None = None,
    colour: str | None = None,
    inset_pos: tuple[float, float, float, float] = (0.58, 0.52, 0.40, 0.44),
) -> Figure:
    """Why: one area, one model, full width, with an optional zoom inset on the hours
    where the model fails (a peak, the spike, a regime boundary).

    ``zoom`` is either an integer ``slice`` over the rows or a ``(start, end)`` pair of
    timestamps or strings; the inset shows that span and the main axes outline it.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    idx = _local_naive(index)
    if y.size != idx.size or p.size != idx.size:
        raise ValueError("y_true, y_pred and index must have the same length.")
    colour = colour or PALETTE[0]

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 2.9), layout="constrained")
    ax.plot(idx, y, color=ACTUAL_COLOUR, lw=0.9, label="actual")
    ax.plot(idx, p, color=colour, lw=0.9, label=f"{model_name} (one step ahead)")
    ax.set_title(f"square {area_id}", loc="left")
    ax.set_ylabel(Y_LABEL)
    ax.margins(x=0)
    _thousands(ax.yaxis)
    _day_ticks_at_noon(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    if zoom is not None:
        lo, hi = _resolve_span(zoom, idx)
        if hi - lo < 2:
            raise ValueError("zoom span must cover at least two slots.")
        axins = ax.inset_axes(inset_pos)
        axins.plot(idx[lo:hi], y[lo:hi], color=ACTUAL_COLOUR, lw=0.9)
        axins.plot(idx[lo:hi], p[lo:hi], color=colour, lw=0.9)
        axins.set_xlim(idx[lo], idx[hi - 1])
        axins.tick_params(labelsize=6)
        axins.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        axins.yaxis.set_major_locator(MaxNLocator(3))
        _thousands(axins.yaxis)
        axins.set_facecolor("white")
        ax.indicate_inset_zoom(axins, edgecolor="#555555", lw=0.7)
    return fig


def plot_error_by_hour(
    residuals_by_model: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    y_true: ArrayLike | None = None,
) -> Figure:
    """Why: forecast error is not constant over the day. If MAE tracks the traffic level
    (large at the afternoon peak, small at night) the error is heteroscedastic, which
    changes how a single headline MAE should be read and argues for scale-free metrics.

    Residuals are ``y_true - y_pred`` per model, ``(n,)`` or ``(n_areas, n)`` (pooled over
    areas). If ``y_true`` is given, a second panel below shows the mean actual level by
    hour on its own axis, so the two can be compared without a dual y axis.
    """
    idx = pd.DatetimeIndex(index)
    hours = idx.hour.to_numpy()
    models = list(residuals_by_model)
    colours = colour_for(models)
    hour_range = np.arange(24)

    if y_true is None:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.7), layout="constrained")
        ax_lvl = None
    else:
        fig, (ax, ax_lvl) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(COLUMN_WIDTH_IN, 3.7),
            height_ratios=[2, 1],
            layout="constrained",
        )
    for m in models:
        r = np.abs(np.asarray(residuals_by_model[m], dtype=np.float64)).reshape(-1, hours.size)
        per_hour = np.array([np.nanmean(r[:, hours == h]) for h in hour_range])
        ax.plot(hour_range, per_hour, color=colours[m], marker="o", ms=3, lw=1.1, label=m)
    ax.set_ylabel("Mean absolute error\n(scaled CDR count)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    bottom = ax if ax_lvl is None else ax_lvl
    bottom.set_xticks(range(0, 24, 6))
    bottom.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 6)])
    bottom.set_xlabel("Hour of day (local time)")
    if ax_lvl is not None:
        Y = np.asarray(y_true, dtype=np.float64).reshape(-1, hours.size)
        level = np.array([np.nanmean(Y[:, hours == h]) for h in hour_range])
        ax_lvl.fill_between(hour_range, level, color="#bbbbbb", alpha=0.7, lw=0)
        ax_lvl.set_ylabel("Mean actual\nlevel")
        ax_lvl.set_ylim(bottom=0)
    return fig


def plot_daily_peak_times(
    peak_table: pd.DataFrame,
    *,
    square_id: int | None = None,
    holidays: Mapping[str, str] = HOLIDAYS,
) -> Figure:
    """Why: on square 5259 the daily maximum moves from early afternoon on working days to
    the small hours on weekends and holidays. That is a regime change, not a seasonal
    amplitude change, and it is what breaks the daily seasonal-naive baseline there.

    ``peak_table`` needs ``date`` and ``peak_hour`` (fractional hours, 0 to 24). An
    optional ``kind`` column in {weekday, weekend, holiday} overrides the calendar-based
    classification. Weekdays, weekends and holidays get distinct colours *and* markers.
    """
    dates = _local_naive(pd.DatetimeIndex(pd.to_datetime(peak_table["date"]))).normalize()
    hours = peak_table["peak_hour"].to_numpy(dtype=np.float64)
    if "kind" in peak_table.columns:
        kinds = peak_table["kind"].to_numpy()
    else:
        kinds = np.array(
            [
                (
                    "holiday"
                    if d.strftime("%Y-%m-%d") in holidays
                    else ("weekend" if d.dayofweek >= 5 else "weekday")
                )
                for d in dates
            ]
        )
    styles = {
        "weekday": (PALETTE[0], "o", "working day"),
        "weekend": (PALETTE[1], "s", "Saturday / Sunday"),
        "holiday": (HOLIDAY_COLOUR, "D", "public holiday"),
    }
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 2.7), layout="constrained")
    x = dates + pd.Timedelta(hours=12)
    for kind, (colour, marker, label) in styles.items():
        mask = kinds == kind
        if mask.any():
            ax.scatter(
                x[mask],
                hours[mask],
                color=colour,
                marker=marker,
                s=26,
                ec="white",
                lw=0.6,
                label=label,
                zorder=3,
            )
    shade_non_working_days(ax, dates, holidays=holidays)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 4))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax.set_ylabel("Local time of the daily maximum")
    ax.set_xlabel("Date")
    step = max(1, int(np.ceil(dates.size / 16)))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=1))
    if square_id is not None:
        ax.set_title(f"square {square_id}", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    return fig


# ------------------------------------------------------------------------- tables


def metrics_table_markdown(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("MAE", "RMSE", "MAPE", "WAPE", "MASE", "R2"),
    *,
    higher_is_better: Sequence[str] = ("R2",),
    extra_cols: Sequence[str] = (),
    formats: Mapping[str, str] | None = None,
) -> str:
    """Why: the report needs one comparison table per area with the best model per metric
    visible at a glance, generated from the results frame rather than typed by hand.

    ``df`` is the long frame from ``evaluate.results_frame`` (columns ``area``, ``model``
    and the metrics). Rows sharing an area and model (several seeds) are averaged. The
    best value per metric is bold: the minimum, or the maximum for ``higher_is_better``.
    ``extra_cols`` (numeric, e.g. ``n_params``) are appended without highlighting.
    """
    for col in ("area", "model"):
        if col not in df.columns:
            raise ValueError(f"results frame needs an {col!r} column.")
    present = [m for m in metrics if m in df.columns]
    extras = [c for c in extra_cols if c in df.columns]
    columns = [*present, *extras]
    fmt = {"MASE": ".3f", "R2": ".3f", "n_params": ",.0f"}
    fmt.update(formats or {})

    blocks = []
    for area, sub in df.groupby("area", sort=True):
        agg = sub.groupby("model", sort=False)[columns].mean(numeric_only=True)
        best: dict[str, object] = {}
        for m in present:
            col = agg[m].dropna()
            if not col.empty:
                best[m] = col.idxmax() if m in higher_is_better else col.idxmin()
        header = "| Model | " + " | ".join(columns) + " |"
        rule = "|:--|" + "|".join("--:" for _ in columns) + "|"
        rows = []
        for model, row in agg.iterrows():
            cells = []
            for c in columns:
                v = row[c]
                text = "n/a" if pd.isna(v) else format(float(v), fmt.get(c, ".2f"))
                if best.get(c) == model:
                    text = f"**{text}**"
                cells.append(text)
            rows.append(f"| {model} | " + " | ".join(cells) + " |")
        blocks.append("\n".join([f"### Square {area}", "", header, rule, *rows]))
    return "\n\n".join(blocks) + "\n"


# ------------------------------------------------------------ figure-input helpers


def grid_totals(matrix: np.ndarray, *, chunk_rows: int = 2 * SLOTS_PER_DAY) -> np.ndarray:
    """Per-slot sum over all cells, computed in row chunks.

    Works on the memory-mapped ``(n_slots, n_squares)`` matrix without ever holding more
    than ``chunk_rows`` rows (288 x 10 000 float32 is about 11 MB) in RAM.
    """
    n = matrix.shape[0]
    out = np.empty(n, dtype=np.float64)
    for lo in range(0, n, chunk_rows):
        block = np.asarray(matrix[lo : lo + chunk_rows], dtype=np.float64)
        out[lo : lo + chunk_rows] = block.sum(axis=1)
    return out


def weekday_means_by_iso_week(frame: pd.DataFrame, *, full_weeks_only: bool = True) -> pd.DataFrame:
    """Mon to Fri mean of every column, one row per ISO week number.

    ``frame`` is indexed by a ``DatetimeIndex``. With ``full_weeks_only`` (default) weeks
    that do not contain all five weekdays are dropped: a week represented by a single
    holiday Friday is not a level estimate. Week numbers alone are a valid key here only
    because the observation window is 62 days and cannot repeat a week number.
    """
    idx = pd.DatetimeIndex(frame.index)
    wd = frame[idx.dayofweek < 5]
    wd_idx = pd.DatetimeIndex(wd.index)
    weeks = wd_idx.isocalendar()["week"].to_numpy().astype(int)
    means = wd.groupby(weeks).mean()
    if full_weeks_only:
        n_days = pd.Series(wd_idx.normalize(), index=wd.index).groupby(weeks).nunique()
        means = means[n_days.reindex(means.index) == 5]
    means.index.name = "iso_week"
    return means


def daily_peak_table(series: pd.Series, *, holidays: Mapping[str, str] = HOLIDAYS) -> pd.DataFrame:
    """``date, peak_hour, peak_value, kind`` for every day of ``series`` (a DatetimeIndex).

    Input for :func:`plot_daily_peak_times`. ``peak_hour`` is fractional local hours.
    """
    s = pd.Series(np.asarray(series, dtype=np.float64), index=_local_naive(series.index))
    days = s.index.normalize()
    rows = []
    for day, sub in s.groupby(days):
        t = sub.idxmax()
        key = day.strftime("%Y-%m-%d")
        kind = "holiday" if key in holidays else ("weekend" if day.dayofweek >= 5 else "weekday")
        rows.append(
            {
                "date": day,
                "peak_hour": t.hour + t.minute / 60.0,
                "peak_value": float(sub.max()),
                "kind": kind,
            }
        )
    return pd.DataFrame(rows)


def plot_dm_intervals(
    frame: pd.DataFrame,
    *,
    mdd_by_area: Mapping[int, float] | None = None,
) -> Figure:
    """Why: the per-area tables bold the lowest number in each column, which reads as a verdict
    and is only a point estimate. This figure puts each pairwise margin beside the interval it
    sits in, so a reader can see at a glance which of them are separable from zero and which are
    not. Two of the comparisons the study's headline once rested on are the two whose intervals
    straddle zero.

    ``frame`` needs ``area``, ``model_a``, ``model_b``, ``d``, ``ci_low``, ``ci_high`` and ``p``,
    one row per comparison, already restricted to a single loss. Rows are drawn in the order
    given, grouped by area, most recent at the bottom. ``mdd_by_area`` shades each area's
    minimum detectable difference behind its rows, so the detection floor is visible beside the
    estimate rather than quoted only in the text.
    """
    needed = {"area", "model_a", "model_b", "d", "ci_low", "ci_high", "p"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"plot_dm_intervals needs columns {sorted(missing)}.")
    rows = list(frame.itertuples(index=False))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(3.42, 0.30 * n + 1.05), layout="constrained")
    ax.axvline(0.0, color="#333333", lw=0.9, ls="--", zorder=1)

    areas = list(dict.fromkeys(r.area for r in rows))
    band = {a: c for a, c in zip(areas, ("#f2f2f2", "#ffffff", "#f2f2f2"), strict=False)}
    ys = np.arange(n)[::-1]
    for y, r in zip(ys, rows, strict=True):
        ax.axhspan(y - 0.5, y + 0.5, color=band.get(r.area, "#ffffff"), lw=0, zorder=0)
        if mdd_by_area and r.area in mdd_by_area:
            m = float(mdd_by_area[r.area])
            ax.axhspan(y - 0.5, y + 0.5, xmin=0, xmax=1, color="none", lw=0, zorder=0)
            ax.plot([-m, m], [y, y], color="#cccccc", lw=5.5, solid_capstyle="butt", zorder=1)
        significant = float(r.p) < 0.05
        colour = PALETTE[0] if significant else "#8a8a8a"
        ax.plot([r.ci_low, r.ci_high], [y, y], color=colour, lw=1.4, zorder=3)
        ax.plot(
            [r.d],
            [y],
            marker="o" if significant else "o",
            ms=4.5 if significant else 3.5,
            mfc=colour if significant else "white",
            mec=colour,
            mew=1.2,
            zorder=4,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.area}  {r.model_a} vs {r.model_b}" for r in rows], fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    # Reserve a right-hand margin for the p annotations: without it the widest interval's
    # point estimate sits underneath its own label.
    spread = [float(r.ci_low) for r in rows] + [float(r.ci_high) for r in rows]
    if mdd_by_area:
        spread += [v for a in mdd_by_area for v in (-float(mdd_by_area[a]), float(mdd_by_area[a]))]
    lo, hi = min(spread), max(spread)
    span = max(hi - lo, 1e-9)
    ax.set_xlim(lo - 0.06 * span, hi + 0.32 * span)
    for y, r in zip(ys, rows, strict=True):
        ax.annotate(
            f"p={float(r.p):.3f}" if float(r.p) >= 0.001 else "p<0.001",
            xy=(1.0, y),
            xycoords=blended_transform_factory(ax.transAxes, ax.transData),
            xytext=(-2, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            # 6.5 pt here landed under the legibility floor once the page scaled the
            # figure into the column; 7 pt is the smallest text in the report's figures.
            fontsize=7,
            color="#333333",
        )
    ax.set_xlabel("MAE difference (scaled CDR count)\nnegative favours the first model named")
    ax.tick_params(axis="y", length=0)
    return fig


def plot_per_period_error(
    *,
    zoom_truth: ArrayLike,
    zoom_predictions: Mapping[str, ArrayLike],
    zoom_index: ArrayLike,
    daily_mae: pd.DataFrame,
    daily_wape: pd.DataFrame,
) -> Figure:
    """Why: a weekly MAE per model hides where a model actually fails. The convolutional model's
    whole deficit on square 5059 is two days wide, and it is an over-predicted afternoon plateau
    rather than a noisy week. The lower panels put those days in the context of the week and of
    the other two cells, where the weekend moves relative and absolute error in opposite
    directions.

    Panel (a) plots the **signed error**, truth minus prediction, rather than the level. On a
    y axis that spans the diurnal cycle a 200-unit bias on a 2500-unit plateau is invisible, and
    the bias is the entire claim; the levels themselves are already in the actual-versus-predicted
    figure. Zero is drawn, so over-prediction is everything below the line.

    ``daily_mae`` is indexed by day label with one column per model; ``daily_wape`` likewise with
    one column per area. Both use a positional x axis with the day labels as ticks, so the
    weekend shading aligns with the day columns rather than with a timestamp.
    """
    idx = _local_naive(zoom_index)
    fig, axes = plt.subplots(3, 1, figsize=(3.42, 5.4), layout="constrained")
    ax_resid, ax_mae, ax_wape = axes

    truth = np.asarray(zoom_truth, dtype=np.float64).ravel()
    names = list(zoom_predictions)
    # One palette over every entity in the figure, so a model keeps its colour between panel
    # (a) and panel (b). colour_for assigns by position, so passing each panel's own list
    # separately would give the same model two colours.
    entities = list(dict.fromkeys([*names, *(str(c) for c in daily_mae.columns)]))
    colours = colour_for(entities)
    for day in pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D"):
        ax_resid.axvspan(
            day + pd.Timedelta(hours=12),
            day + pd.Timedelta(hours=18),
            color="#d9d9d9",
            alpha=0.5,
            lw=0,
            zorder=0,
        )
    ax_resid.axhline(0.0, color=ACTUAL_COLOUR, lw=1.0, zorder=2)
    for name in names:
        resid = truth - np.asarray(zoom_predictions[name], dtype=np.float64).ravel()
        ax_resid.plot(idx, resid, color=colours[name], lw=1.0, label=name, zorder=3)
    ax_resid.set_ylabel("Signed error\n(actual minus forecast)", fontsize=7)
    _thousands(ax_resid.yaxis)
    _day_ticks_at_noon(ax_resid, fmt="%a %d %b")
    lo, hi = ax_resid.get_ylim()
    ax_resid.set_ylim(lo, hi + 0.42 * (hi - lo))
    ax_resid.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper left")
    ax_resid.annotate(
        "(a)",
        xy=(0.985, 0.96),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        weight="bold",
    )

    for ax, table, ylabel, tag, ncol in (
        (ax_mae, daily_mae, "MAE", "(b)", 2),
        (ax_wape, daily_wape, "WAPE (%)", "(c)", 3),
    ):
        labels = [str(v) for v in table.index]
        x = np.arange(len(labels))
        cols = list(table.columns)
        palette = {c: colours.get(str(c), PALETTE[i % len(PALETTE)]) for i, c in enumerate(cols)}
        if ax is ax_wape:  # areas, not models: they share no entity with the panels above
            palette = colour_for(cols)
        starts = pd.to_datetime([f"2013-12-{16 + i:02d}" for i in range(len(labels))])
        for i, day in enumerate(starts):
            if day.dayofweek >= 5:
                ax.axvspan(i - 0.5, i + 0.5, color=WEEKEND_COLOUR, alpha=0.14, lw=0, zorder=0)
        for col in cols:
            ax.plot(
                x,
                table[col].to_numpy(dtype=np.float64),
                marker="o",
                ms=3,
                lw=1.1,
                color=palette[col],
                label=str(col),
                zorder=2,
            )
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
        ax.set_xlim(-0.5, len(labels) - 0.5)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.40 * (top - bottom))
        ax.legend(fontsize=6.5, ncol=ncol, frameon=False, loc="upper left")
        ax.annotate(
            tag,
            xy=(0.985, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8,
            weight="bold",
        )
    return fig
