#import "../lib.typ": *

= Results and Discussion

All numbers below come from run EXP-008: three architectures on three areas at three seeds each, log1p-standard scaling, calendar features on, evaluated on 16 to 22 December against reference forecasters fed the identical 144-slot window. Trained models are reported as the seed mean, and every pairwise margin is tested on the paired error differentials of that seed-averaged loss path.

== Per-area accuracy

#apa-table(4, [Square 5161, the highest-traffic cell. Bold marks the lowest point estimate in each column, not a demonstrated win; the standard deviation over the three seeds follows the MAE in brackets. The TCN's 2.85-unit lead over the linear model is inside its own 6.65 seed spread and inside the test's interval, [−9.01, +3.31] at p = 0.36. Nothing here separates the two.], [
  #apa-body(
    (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    rule,
    [*Model*], [*MAE*], [*MAPE (%)*], [*RMSE*], [*WAPE (%)*], [*MASE*], [*R²*],
    rule,
    [LSTM], [84.42 (1.41)], [8.02], [129.35], [5.84], [0.243], [0.9910],
    [TCN], [*82.13* (6.65)], [7.83], [*124.35*], [*5.68*], [*0.236*], [*0.9916*],
    [GBT], [87.08 (1.87)], [8.41], [130.39], [6.03], [0.251], [0.9908],
    [Persistence], [92.80], [9.19], [134.88], [6.42], [0.267], [0.9902],
    [Seasonal naive (daily)], [338.59], [25.94], [619.04], [23.43], [0.975], [0.7934],
    [Linear AR(144)], [84.98], [*7.74*], [130.92], [5.88], [0.245], [0.9908],
    rule,
  )
])

#apa-table(5, [Square 5059. The linear model beats all three trained models on every metric, by margins of 7.1 to 16.0 percent of the trained model's MAE that exceed every seed spread on the cell. They are also the only trained-versus-linear margins in the study that survive a test, and only against the TCN; the values are in Figure 6 and in Section 6.1.], [
  #apa-body(
    (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    rule,
    [*Model*], [*MAE*], [*MAPE (%)*], [*RMSE*], [*WAPE (%)*], [*MASE*], [*R²*],
    rule,
    [LSTM], [71.84 (2.97)], [6.77], [103.68], [5.68], [0.266], [0.9878],
    [TCN], [78.88 (3.58)], [7.22], [115.32], [6.23], [0.292], [0.9849],
    [GBT], [71.32 (0.53)], [6.79], [102.68], [5.64], [0.264], [0.9880],
    [Persistence], [81.52], [7.96], [114.38], [6.44], [0.302], [0.9851],
    [Seasonal naive (daily)], [171.74], [18.02], [245.87], [13.57], [0.635], [0.9313],
    [Linear AR(144)], [*66.25*], [*6.30*], [*96.67*], [*5.24*], [*0.245*], [*0.9894*],
    rule,
  )
])

#apa-table(6, [Square 5259, the weekday-only cell. Bold marks the lowest point estimate; seed standard deviation in brackets after the MAE. The TCN's 2.25-unit lead over the linear model clears its seed spread of 0.46 but not the week's sampling variation: the interval is [−6.72, +2.21] at p = 0.32. Seed stability and separability are different questions, and this cell answers the first yes and the second no. What does hold here is the TCN's 4.26-unit margin over the LSTM, p = 0.009, the reverse of the ordering on square 5059.], [
  #apa-body(
    (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    rule,
    [*Model*], [*MAE*], [*MAPE (%)*], [*RMSE*], [*WAPE (%)*], [*MASE*], [*R²*],
    rule,
    [LSTM], [65.70 (0.29)], [7.11], [94.13], [5.19], [0.125], [0.9930],
    [TCN], [*61.43* (0.46)], [*6.65*], [*89.29*], [*4.86*], [*0.117*], [*0.9937*],
    [GBT], [64.08 (0.51)], [6.92], [92.93], [5.06], [0.122], [0.9931],
    [Persistence], [75.97], [8.11], [109.58], [6.00], [0.145], [0.9904],
    [Seasonal naive (daily)], [470.32], [71.62], [861.62], [37.18], [0.898], [0.4093],
    [Linear AR(144)], [63.69], [6.66], [93.74], [5.03], [0.122], [0.9930],
    rule,
  )
])

#apa-figure(6, [Difference in mean absolute error between the paired forecasters of the three results tables, with 95 percent Newey-West intervals at lag 15 over the 1008 evaluation slots. Negative favours the first model named; filled markers are the comparisons with p < 0.05. The grey bar behind each area is the minimum detectable difference at 80 percent power of that area's TCN-versus-linear comparison, 10.67, 8.81 and 6.38 units for squares 5059, 5161 and 5259. The two comparisons the study's headline once rested on are among those whose intervals cross zero.], "/reports/figures/dm_intervals.png", width: 78%)

The ranking changes with the area. The TCN is best on squares 5161 and 5259 and worst of the three on 5059, the only trained model there that fails to beat persistence on RMSE. The LSTM and the GBT swap places between cells. Answering the research question from one area, which the brief allows, would have been confident and wrong. The TCN beats the LSTM by 4.26 units on 5259, p = 0.009, and loses to it by 7.03 units on 5059, p = 0.002 (Figure 6); as a conjunction the intersection-union p is 0.004, and it rejects at every lag from 0 to 144. The margins over the linear model do not survive: 2.85 units on 5161 at p = 0.36 and 2.25 on 5259 at p = 0.32. One week on these cells resolves 8.81 and 6.38 units, 10.4 and 10.0 percent of the linear model's MAE, so margins of 3.4 and 3.5 percent were never within reach; detecting them would take about 67 and 56 days. The ordering is real and the podium is not.

A 145-parameter linear model is the real competitor, and on average it wins. Linear AR(144) wins on square 5059 on all six metrics, beating the best trained model there by 7.1 percent of that model's MAE, and it is the only cell where a trained-versus-linear margin separates from zero (p = 0.0009 against the TCN, 0.008 after Holm adjustment). On 5161 and 5259 it loses to the TCN by 3.4 and 3.5 percent. Aggregate the three cells and the advantage disappears: the per-area best trained model averages 71.63 MAE against the linear model's 71.64, a difference of 0.015 percent. Architecture by architecture the linear model is ahead of all three, by 3.3 percent over the LSTM, 3.5 percent over the TCN and 3.5 percent over the GBT. Per-cell selection does not quite reach that parity without the test set. Validation agrees with the evaluation week on squares 5161 and 5259, the TCN both times, but on 5059 it ranks the LSTM first where the evaluation week ranks the GBT, and selecting on validation alone averages 71.80 MAE against the linear model's 71.64. This matches the critical literature [12], [13], [11].

The best-performing architecture, and the grounds for naming it: of the three the study names the TCN, the dilated convolutional network, on one quantitative ground and one qualitative one. Quantitatively it holds the lowest error on two of the three cells, and it is in both halves of the one architecture-versus-architecture comparison the study pre-specified. Qualitatively that win is exactly where Section 4 says it should be. Square 5259 is the cell whose daily shape changes between regimes rather than merely shrinking: its mean working-day and non-working-day profiles correlate at −0.28, against 0.95 on 5161 and 0.91 on 5059, and 59 percent of its profile variance lies outside its own mean profile, against 6 and 9 percent. On that cell, knowing which of two shapes the previous day had is worth something, and a receptive field of 253 steps reads it off the window in a single pass, where a recurrent state must carry it through 144 sequential updates and a linear map has one fixed set of coefficients for both regimes. The same reading accounts for the loss on 5059: a shape that barely changes leaves no regime to identify, and the calendar block that encodes the regime becomes the noise of Section 6.4.

Two qualifications belong with the name. The TCN's second win, 2.85 units on square 5161 at p = 0.36, is a point estimate rather than a result, and the argument above does not extend to it. And the best of three architectures is not a recommended forecaster: averaged over the three cells none of them betters the 145-parameter linear model.

The daily seasonal naive is not a weak baseline, it is a wrong one. It scores 338.6, 171.7 and 470.3 against persistence at 92.8, 81.5 and 76.0. The failure is worst on square 5259, the cell whose regime changes twice a week: at a one-step horizon a ten-minute-old observation beats a day-old one, and the age of the information dominates the seasonal structure.

#apa-figure(7, [One-step-ahead forecasts against observed traffic for each model and area over the evaluation week, drawn for seed 42 rather than the seed mean, which would be a three-member ensemble appearing in no table here. Rows are areas, columns are models, and the vertical scale is shared within each row. All nine panels track the daily cycle closely; the visible errors are at the sharp afternoon ramps and, on square 5259, in the flat weekend of 21 and 22 December.], "/reports/figures/actual_vs_predicted.png", width: 100%)

== Why the errors sit where they do

Figure 7 shows nine forecasts that are, at this resolution, hard to distinguish from the truth, which is why a table of R² values above 0.98 carries so little information here. The differences live in the ramps.

Error is strongly heteroscedastic across the day (Figure 8). For persistence on square 5161 the mean absolute error at 13:00 is 164 against 19 between 02:00 and 05:00, a factor of 8.7, and the busiest 5 percent of slots in the evaluation week carry about a tenth of all absolute error on each of the three cells, twice their share of the week. Every model inherits this shape. A headline MAE is therefore dominated by a few hours and describes peak-hour behaviour more than average behaviour. MAPE, which weights the quiet 01:00 to 06:00 slots at up to 1.3 times their share of the week, is the metric furthest from what an operator provisioning capacity cares about, which is why WAPE is reported beside it.

#apa-figure(8, [Mean absolute error by hour of day, pooled over the three areas and averaged over the three seeds of each trained model, with the mean observed level beneath it. Error tracks the level: the models are accurate at night because the series is flat at night.], "/reports/figures/error_by_hour.png", width: 80%)

== Computational cost

#apa-table(7, [Training and execution time, averaged over the three areas and three seeds. Training is the wall time of one fit under early stopping, reported with the epochs actually run; the area-to-area spread behind those means is wide, 25.0 to 29.9 s for the LSTM and 38.4 to 53.1 s for the TCN. Batch is the median of 20 warm repetitions over all 1008 evaluation slots; latency is the median of 50 single-window predictions, the device synchronised before each timer stops. Params counts learned weights for the two neural models and the 145 coefficients of linear AR(144), while for the GBT it is the total number of leaves across the fitted trees, so the column is not comparable row to row. Hardware: Apple M1 Pro, 8 cores, 16 GB, macOS, PyTorch 2.14 with MPS.], [
  #apa-body(
    (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, left, right, right, right, right),
    rule,
    [*Model*], [*Params*], [*Device*], [*Epochs*], [*Train (s)*], [*Batch (ms)*], [*Latency (ms)*],
    rule,
    [LSTM], [17 222], [mps], [42], [28.2], [14.10], [3.332],
    [TCN], [34 758], [mps], [34], [44.1], [20.89], [3.305],
    [GBT], [3 795], [cpu], [122], [2.1], [9.21], [7.118],
    [Persistence], [0], [cpu], [-], [0.00], [0.02], [0.002],
    [Linear AR(144)], [145], [cpu], [-], [0.02], [0.06], [0.004],
    rule,
  )
])

The cost spread in Table 7 is three orders of magnitude wider than the accuracy spread. Training the TCN takes 44 s against 0.02 s for the linear model, and batch inference over a week of slots 20.9 ms against 0.06 ms, for 3.4 percent of MAE at best and a loss at worst. The GBT is the compromise: 2.1 s to fit, 9.2 ms to predict, within about 3 percent of the LSTM on every cell and ahead of it on two.

Single-window latency inverts the batch ordering: among the trained models the GBT is the fastest in a batch and the slowest per window, 7.1 ms against 3.3 ms, because its per-call overhead does not amortise. The first MPS call at a new batch shape runs about thirty times slower than a warm one, so a single unsynchronised timing would have reported the neural models as roughly an order of magnitude slower than they are. Both are artefacts of how the question is asked, not properties of the models.

== Failure case: the calendar features, square 5059, and two days in December

The clearest failure is not a model losing to a baseline but an input representation that helps two cells and hurts a third. Switching off the five calendar features costs 0.8 to 2.6 percent of MAE on seven of the nine model-area pairs on a single seed, inside the seed spread of two of the three architectures and so weak on its own. Both exceptions are on square 5059, and the larger is not weak at all. Repeated over three seeds, the TCN on 5059 improves from 78.88 to 68.42 MAE without the features, a gain of 13 percent, and its standard deviation across seeds falls from 3.58 to 0.97. The features are worse on validation as well as on test, so the model trains worse; early stopping is not the culprit.

That deficit is not spread across the week (Figure 9). Outside Thursday 19 and Friday 20 December the TCN and the linear model are level, 70.14 against 67.43 (p = 0.29), and over those five days the TCN beats persistence by 11.90 units (p < 0.0001). On the two days it scores 100.72 against persistence's 80.21 and loses significantly (p = 0.010): a trained convolutional network beaten by repeating the last observation. Between 79.5 and 95.5 percent of its net excess over the linear model falls on those two days in each of the three seeds. This is the period the brief asks for, and the same event as the ablation above.

The cause is not turbulence: Thursday is the calmest weekday of the week, and persistence, whose absolute error is by definition the mean absolute ten-minute change, scores 69.8 there against 87.4 on Monday to Wednesday. It is not lag either, since the residual's correlation with the first difference of the series is lower for the TCN than for the linear model, 0.617 against 0.710. It is bias in the afternoon plateau. Over the two days the TCN's signed error averages −78.4 against the linear model's −5.2, and squared bias accounts for 30.4 percent of its mean squared error against 0.3 percent for the linear model. Withholding the calendar block removes 34.2 percent of the Thursday error and 25.0 percent of the Friday error, and 81.2 percent of the whole calendar penalty on this cell falls on those two days.

#apa-figure(9, [Per-period error on the evaluation week. (a) Signed error, actual minus forecast, on square 5059 over Thursday 19 and Friday 20 December; the shaded bands mark 12:00 to 18:00, where the TCN sits below zero and so over-predicts. (b) Per-day MAE on square 5059, mean over three seeds, with the calendar-off control. (c) Per-day WAPE by area, mean over the five non-seasonal forecasters. Saturday and Sunday are shaded in both lower panels.], "/reports/figures/per_period_error.png", width: 78%)

Square 5059 has the weakest calendar signal of the three, though not in the way the weekday-to-weekend ratios suggest: 1.29 against 0.79 and 2.98 puts it between the other two. The quantity that matters is what the working-day flag adds on top of time of day, because time of day is already in the window. Regressed on the 143 time-of-day dummies alone, the training series reaches R² 0.884 on 5059, 0.839 on 5161 and 0.438 on 5259; adding the working-day indicator and its interaction lifts those to 0.945, 0.942 and 0.925. The flag is worth 0.06 of R² on 5059 against 0.10 on 5161 and 0.49 on 5259, so on this cell it carries three fifths of what it carries next door and an eighth of what it carries on 5259. The features there are close to noise, and the two models that degrade are the two fed that noise on the cell where it carries least.

The architectural half is subtler, and the obvious explanation is wrong. Both neural models concatenate the calendar block to a single summary vector and pass it through one linear head, so neither folds it into the recurrence. The asymmetry is in what the summary can do about it. The LSTM's 64-unit hidden state is free to shrink its own scale relative to a fixed calendar term, and the GBT can award those five columns no splits, which it largely does. The TCN's last-step channel vector is the output of a fixed convolutional stack whose scale is tied to the input, leaving the head less freedom to attenuate a term it should ignore. The seed instability, 3.58 against 0.97, is consistent with that reading, which remains a hypothesis about optimisation rather than a demonstrated mechanism.

The features were kept on for every model. Turning them off for one architecture on one cell would buy about 13 percent of MAE there and destroy the equal-information comparison the study rests on. An input representation justified by an exploratory finding on one cell, here the night-time peaks of square 5259 on non-working days, does not automatically transfer to a neighbouring cell 500 m away.

Panel (c) of Figure 9 carries a warning about the metric. On square 5259 every forecaster roughly doubles its WAPE at the weekend, from 4.8 percent on Monday to Friday to 9.4 percent on Saturday and Sunday, which invites the reading that the weekend is hard. Absolute error moves the other way over the same slots, from 76.2 to 41.2, because the level collapses from 1595 to 441: the denominator falls faster than the error does. No model is significantly worse than any other there. The weekend on 5259 is a property of the cell and of the choice of metric, not a failure of any forecaster.

A second failure is shared by every model and fixable by none. The grid-wide spike on 6 December at 20:50, described in Section 4, is unannounced by anything in the preceding window, so a one-step-ahead forecaster must miss it and then miss again in the opposite direction when the series falls back. It lies in the training period, outside the evaluation week, but it is the clearest illustration of the ceiling on this task: a model that has learned the daily cycle perfectly still cannot forecast an event that the recent past does not foreshadow.

== Personal considerations and margins for improvement

Across the three cells the gap between a well-tuned linear model and the three trained models lies between minus 7.7 and plus 3.5 percent of the linear model's mean absolute error, a range taken over the per-cell best trained model, the same convention Section 7 uses. The worst single model-area pair is far wider: the TCN on square 5059 is 19.1 percent behind the linear model. Persistence reaches R² 0.99 because ten-minute traffic is smooth, and almost all of the remaining error is concentrated in a few ramp hours per day. A model can only compete for that residue, the component Xu et al. separate out and declare unpredictable [1]; the measurement here is that the predictable part is largely already in the previous observation.

Section 5 recorded an expectation for each model before any of them was run, and two of the three were wrong. The LSTM was expected to clear persistence by a few percent and it did, by 9 to 14 percent, though persistence mimicry was not the failure mode that appeared. The TCN was expected to match the LSTM at 0.40 s per epoch against 0.58 s; it matched on accuracy but ran at 1.28 s per epoch against 0.67 s, 1.9 times slower, not faster. The GBT was expected to be best or joint-best on square 5259 and came second of the three trained models there, while winning on 5059 instead, the cell nobody predicted.

The margin for improvement differs by model. The GBT is under-capacity: 63 leaves improve validation MAE by 2.3 percent over the reported 31, beyond any seed spread in that block, and the wider setting was not adopted because the campaign and its tests were already closed (Section 5.9, `reports/tables/capacity_grid.md`). The LSTM at 128 hidden units improves validation MAE by 0.6 percent, which is inside one seed spread, so its margin is real but not demonstrated. The TCN's margin is representational rather than one of capacity: feeding the calendar block at every time step instead of concatenating it to a single summary vector is the change the square 5059 failure of Section 6.4 implicates.
