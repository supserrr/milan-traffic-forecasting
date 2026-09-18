#import "../lib.typ": *

= Exploratory Analysis

Every statistic in this section is computed on the training rows alone, 1 November to 8 December, unless it is explicitly descriptive of the whole record. The distinction matters: searching lags 1 to 400 on square 5259, the last partial autocorrelation above a practical 0.05 falls at lag 310 on the training slice and at lag 159 on the full 8928 slots, so a sequence length read off the full series would have been set partly by the week it is evaluated on.

== Spatial concentration

Per-cell Internet totals over the 62 days span almost five orders of magnitude, from 214 to 12 740 060. The top 1 percent of the 10 000 cells carry 11.05 percent of all traffic, the Gini coefficient is 0.608, and 1075 cells reach half the total (Figure 1). On a logarithmic axis the histogram is unimodal and near-symmetric, skewness 0.02, so the concentration is the ordinary consequence of a log-normal spread; the left tail is the longer one, a handful of near-empty cells 3.1 decades below the median against 1.7 above.

The two cells the brief names sit in the upper tail without being extreme: 4556 ranks 109th and 4159 ranks 424th, at 16.5 and 8.8 times the median cell. The three highest-traffic cells are squares 5161, 5059 and 5259, at 12.74, 11.17 and 10.49 million, and the ranking is stable under pre-evaluation totals, under totals excluding the Christmas collapse, and under the mean over non-zero slots. The fourth cell, 5061, is 8.7 percent below the third, so the boundary is not a near-tie.

#apa-figure(1, [Total Internet activity per cell over the 62 days. Panel (a): histogram on a logarithmic axis, with the median and the top-1-percent cut-off. Panel (b): Lorenz curve, annotated with the top-1-percent share and the number of cells needed to reach half of all traffic.], "/reports/figures/total_traffic_distribution.png", width: 92%)

The three evaluation areas are the most extreme cells of a heavy-tailed distribution, so nothing measured on them transfers to a typical cell, and their levels differ by orders of magnitude, which makes MAE and RMSE comparable only within an area. Section 6 therefore compares areas on the scale-free WAPE and MASE, and every scaler is fitted per area.

== The three areas are neighbours with different regimes

All five study cells lie in the dark central core, inside a window of roughly 2.8 km by 1.4 km (Figure 2). The three top cells sit at grid positions (51, 60), (50, 58) and (52, 58), every pair at Chebyshev distance 2, so no two share an edge or a corner, yet all three lie within about 470 to 525 m of each other. The research question's "geographical areas with different traffic characteristics" therefore means neighbouring cells of one core, which is the stronger test: whatever separates them is land use at cell scale.

The grid geometry says which land use. The published lattice has square 1 in the south-west corner with ids running east before north [17], so a centroid follows from the id alone, and that orientation is measured rather than assumed: under it the agricultural belt south of the city carries 52 000 activity units against 1 195 000 in Sesto San Giovanni, while the reflected reading would make the farmland the busier of the two, and the traffic mass has its mean row 1.5 km north of the Duomo rather than in open fields. Read that way, square 5161 centres 320 m east-north-east of the Duomo on the Corso Vittorio Emanuele retail axis, square 5059 206 m west-south-west on Via Torino, and square 5259 409 m north-north-west at Cordusio, between La Scala and the Piazza Affari banking district; square 4556 centres on the Darsena at the head of the Navigli, and square 4159 at Porta Lodovica on the edge of the Bocconi campus. The coordinates were resolved against OpenStreetMap, and the land uses are the hypothesis those addresses support rather than a measurement.

What separates them is the weekly regime (Figure 3). Over 1 to 14 November the weekday-to-weekend mean ratio is 0.79 for square 5161, which is busier at weekends and peaks between 13:30 and 18:30; 1.29 for 5059; 2.98 for 5259, whose weekday mean of 1597 falls to 536 at weekends; 1.93 for 4159; and 0.90 for 4556, which peaks between 19:00 and 23:40 on 13 of the 14 days. Pearson correlation over the fortnight orders the pairs the same way: 0.84 for 5161 and 5059, 0.78 for 5059 and 5259, but 0.43 for 5161 and 5259.

On square 5259 the weekday pattern does not shrink at weekends but disappears. The daily maximum falls between 13:00 and 17:20 on all nine working days of the fortnight and between 22:50 and 02:10 on all five non-working days, the weekends plus Friday 1 November, the Ognissanti public holiday. The regime follows the working calendar and not the day-of-week label, which is why Section 5's calendar features encode a working-day indicator with the window's holidays marked, in place of a weekend flag. Square 4159 reproduces the same shape at a fifth of the amplitude, a mean daily profile over the fortnight correlating at 0.980 with 5259's, so what separates the two is amplitude rather than a second regime.

Every one of the five regimes is what its address would predict. An office and banking block empties at weekends, which is 5259 at 2.98 and, at a fifth of the volume, the university quarter at 4159; a retail and tourist frontage is busier on a Saturday afternoon than a Tuesday morning, which is 5161 at 0.79; a nightlife basin peaks after dinner on almost every day, which is 4556 at 0.90; and a mixed retail and through-traffic street, 5059 at 1.29, has no strong contrast either way. That last cell is where the calendar features later fail: the working-day indicator encodes exactly the contrast 5059 does not have.

#apa-figure(2, [Total Internet activity across the 100 by 100 grid on a log scale, north up with square 1 in the south-west corner, the five study cells marked, and a zoom on the central core. The three evaluation areas are neighbours inside one dark core rather than separate districts, so their differing weekly regimes are the finding.], "/reports/figures/grid_heatmap.png", width: 78%)

#apa-figure(3, [Internet activity over the first two weeks for the three highest-traffic cells and the two squares the brief names. Weekends are shaded grey and the 1 November public holiday amber. Square 5259 loses its daytime peak entirely on non-working days; square 5161 is busiest at weekends.], "/reports/figures/five_cells_first_two_weeks.png", width: 78%)

The heterogeneity costs the baselines immediately. A seasonal naive forecast at the daily lag scores MAE 470.3 on square 5259 against persistence at 76.0, because the daily lag crosses a regime boundary twice a week; the weekly lag, which preserves the day of week, more than halves that error to 210.2. One model per area, each with its own scaler, is the only defensible setup, and a shared architecture must cope with a weekend-heavy and a weekday-only cell at once.

== Temporal dependence and the sequence length

The two additional analyses the brief asks for on the area with the highest total traffic are both carried out on square 5161: the autocorrelation structure below and the decomposition, stationarity and transform diagnostics of Section 4.4, while Section 4.5 is supporting context on the whole grid rather than a third analysis of that cell.

The autocorrelation function of the highest-traffic cell is dominated by the daily cycle, at 0.854 at lag 144 and still 0.761 at the weekly lag 1008 (Figure 4). The partial autocorrelation sets the input window: against a 95 percent band of ±0.0265 at n = 5472, the last partial above a practical magnitude of 0.05 over lags 1 to 300 is lag 149, the partial at the daily lag itself is only −0.037, and at lag 288 only −0.002.

Beyond lag 300 exactly two partials still exceed that threshold anywhere in the first 1100, at lags 865 and 1011, the latter three slots past the weekly lag: a small and real weekly signal that the 144-slot window leaves to the model. Across lags 301 to 1100 only 5.9 percent exceed the band against the 5 percent expected by chance. Linear autoregressions corroborate the choice: on square 5161 validation MAE falls 115.1, 114.6, 109.2 across orders 6, 36 and 144, while order 216 improves the three-cell mean by 0.6 percent for 1.5 times the input. A window of 144 slots, one day, is a cost and accuracy trade rather than a located optimum.

#apa-figure(4, [Autocorrelation and partial autocorrelation of square 5161 on the training slice. The daily lag 144 and the weekly lag 1008 are marked. Over lags 1 to 300 the partial autocorrelation is negligible beyond about lag 150, which sets the input window; the only exceedances further out are the two isolated partials at lags 865 and 1011 next to the weekly lag.], "/reports/figures/acf_pacf_5161.png", width: 92%)

== Seasonality, stationarity and the transform

An STL decomposition at period 144 [22] puts 89.5 percent of the variance in the daily seasonal component, 3.4 percent in the trend and 4.8 percent in the remainder. A multi-seasonal decomposition at periods 144 and 1008 splits that further, 88.0 percent daily and 11.1 percent weekly, while a classical centred moving-average decomposition puts the remainder at 11.8 percent (Table 2). The method changes that number by a factor of six, so the method is named wherever a share is quoted; what survives is the ordering, and a weekly component that is real but roughly eight times smaller than the daily one.

The 3.4 percent STL trend is mostly the weekly cycle, which a decomposition at period 144 cannot separate from a slow level. What either method calls irregular is not noise: the STL remainder is autocorrelated at 0.78 at lag 1, so most of the 4.8 percent is predictable one step ahead from the preceding value, which is the persistence floor of Section 6 seen from the decomposition side.

Both stationarity tests agree, though their nulls are opposites. The augmented Dickey-Fuller test rejects its unit-root null decisively on the raw series (p = 3.3 × 10⁻²⁸); the KPSS test does not reject its level-stationarity null (p > 0.10), nor does its trend-stationary variant. The same verdict holds after a log transform, first differencing and seasonal differencing at lag 144. The series is therefore level-stationary over the training period and needs no differencing, but neither test addresses the daily cycle, which the input window handles instead.

#apa-table(2, [Variance shares in percent, var(component) / var(series), square 5161 on the training slice. Shares need not sum to 100 because the components are not orthogonal; the classical filter loses 72 rows at each end. Reading down the trend column shows what the period-144 methods misattribute: the weekly cycle has nowhere else to go. Reading down the remainder column under log1p is the transform decision of Section 5.4.], [
  #apa-body(
    (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    rule,
    [*Series*], [*Method*], [*Trend*], [*Daily*], [*Weekly*], [*Rem.*], [*n*],
    rule,
    [raw], [STL (144)], [3.4], [89.5], [-], [4.8], [5472],
    [raw], [MSTL (144, 1008)], [0.4], [88.0], [11.1], [2.0], [5472],
    [raw], [classical (144)], [4.0], [84.4], [-], [11.8], [5328],
    [log1p], [STL (144)], [0.5], [96.5], [-], [2.3], [5472],
    [log1p], [MSTL (144, 1008)], [0.1], [95.9], [3.2], [1.2], [5472],
    [log1p], [classical (144)], [0.6], [94.7], [-], [5.0], [5328],
    rule,
  )
])

Level-stationary is not the same as unchanging. The grid's Monday-to-Friday mean per slot falls 18.2 percent between the week of 4 November and the evaluation week, before any holiday, and the decline is uneven: square 5161 ends 0.1 percent below its starting level, 5059 and 5259 lose 7.4 and 7.7 percent, and 4159 loses 24.7 percent. Any model that learns a level will overshoot on the drifting cells, while persistence is immune to slow drift, part of why it is hard to beat one step ahead.

The transform decision rests on heteroscedasticity. The trailing 144-slot rolling mean and rolling standard deviation correlate at 0.973 on the raw training series and at 0.235 after a log(1+x) transform: the noise scales with the level. Both transforms are therefore carried into the experiments, and the remainder share falls under the log transform for every decomposition method tried.

== One anomaly, and what it means for training

A scan of every slot against the median of the same slot of the week over the training rows flags exactly one event on the grid total: Friday 6 December at 20:50, where the grid jumps from 712 060 to 1 234 958 and falls back to 615 431 (Figure 5). That is 1.86 times the mean of its neighbouring slots and the largest single-step rise in the record, at 522 898 against a second-largest of 139 442. It is not local: 95.5 percent of the 10 000 cells are higher at 20:50 than at 20:40, and 76.9 percent exceed 1.5 times their own neighbouring slots. Only about a sixth of the excess is given back afterwards, so the event is an additive burst rather than traffic moved between slots.

#apa-figure(5, [Grid-wide Internet activity around Friday 6 December 20:50 against the same hour one week earlier. The grid total reaches 1.86 times the level of its neighbouring slots, 77 percent of cells sit above 1.5 times their own, and the step into that slot is the largest single-step rise in the record.], "/reports/figures/grid_spike_dec06.png", width: 92%)

Nothing in the preceding slots announces it, so no one-step-ahead model can predict it, and the rebound produces a second large error of opposite sign one step later. Row 5165 falls inside the training slice, so the reported metrics are unaffected, but the training loss is not: a squared-error objective sees one target above 1.5 times its context in more than three quarters of the cells. That is an argument for the gradient clipping Section 5 adopts.

The Christmas collapse is severe: on square 5161 the 24 to 26 December block falls 65.1 percent against the same weekdays a fortnight earlier, where the grid as a whole falls 36.5 percent, and 25 December alone falls 88.6 percent against the grid's 38.0. The brief's evaluation week of 16 to 22 December is therefore the last normal week of the record, and no result here is extrapolated past it. Minima over that week on the three study cells are 108, 182 and 161, so no percentage error divides by anything near zero.
