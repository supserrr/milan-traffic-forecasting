#import "../lib.typ": *

= Methodology

== The forecasting task

At time t the algorithm maps the L most recent observations of one area to the
next slot:

$ hat(x)_a (t+1) = f(x_a (t - L + 1), dots, x_a (t)), quad L = 144 $

The 10-minute horizon is fixed by the brief; L is one day, chosen below.

== Splits, and a validation week harder than the test week

Rows 0:5472 train (1 November to 8 December), 5472:6480 validate (Monday 9 to
Sunday 15 December, one full week), 6480:7488 test (16 to 22 December, fixed by
the brief). The split is chronological, because a forecaster is deployed
forward in time.

The validation week is the busiest of the record on all three cells, 15.3
percent above the five training weeks by weekday mean on 5161, so every
validation error below is inflated relative to its test counterpart:
persistence scores MAE 116.8 against 92.8 on 5161, 99.9 against 81.5 on 5059
and 89.9 against 76.0 on 5259. The gap is a calendar effect, not overfitting.

== Input representation and sequence length

The input is the 144 most recent slots of the target cell, one day, derived in
Section 4.3 from the partial autocorrelation and corroborated there by an AR
order sweep; Section 5.9 tests it a third time through the convolutional receptive field.

== Normalisation

The scaler is fitted on the training rows only; models predict in the scaled
space and the harness inverts once, centrally, before any metric. Standard and
log1p-standard run as separate logged configurations, not one assumed;
log1p-standard maps a traffic value $y$ to $(log(1 + y) - mu) / sigma$, where
$mu$ and $sigma$ are the mean and standard deviation of $log(1 + y)$ over the
same training rows. The choice rests on the training-slice
heteroscedasticity of Section 4.4: the rolling mean and standard deviation
decouple under the transform, and every decomposition method leaves a smaller
remainder (Table 2). On validation the transform is close to a wash, lowering
MAE on six of the nine fits by 1.1 to 4.0 percent and raising it on three, mean
change −0.33 percent. It lowers evaluation-week MAE on all nine, a property of
that week and not what the choice was made on.

== Calendar features

Five features describe the target slot, all known at the forecast origin:
time-of-day and day-of-week sine and cosine pairs, and a working-day indicator
treating the public holidays observed in Milan inside the window (1 November, 7
and 8 December, 25 and 26 December, 1 January) as non-working. The regime
finding of Section 4.2 rules out a plain weekend flag: on square 5259 the shape
follows the working calendar, and Friday 1 November behaves as a Sunday. All
three models get the same lags and features; one logged run ablates the
features for all three, keeping the comparison at equal information.

== The three models

The LSTM is one layer of 64 hidden units, its final hidden state concatenated
with the five calendar features and mapped to one output, about 17 000
parameters. Gated recurrence carries a regime across many steps without a
hand-specified lag structure [8], which the three regimes of Section 4.2 ask
for and the closest prior work used [7]. It reads all 144 steps in sequence,
forgoing the parallelism the other two have, and it has a known degenerate
solution in which the gates saturate and the output tracks the last input,
which would show up as persistence mimicry.

The TCN, a temporal convolutional network, is six residual blocks of two causal
convolutions, kernel 3, dilations 1 to 32, 32 channels, with ReLU activations,
weight normalisation and dropout 0.1, following [9]; about 35 000 parameters,
the largest of the three trained models, so the comparison is matched neither
on capacity nor on regularisation. Its receptive field of 253 steps answers the
lag-149 cut-off of Section 4.3: everything the partial autocorrelation locates
sits inside one parallel forward pass. The cost is a fixed field: nothing
beyond 253 steps exists for this model at all.

The gradient-boosted trees, GBT in the tables and figures, are a
`HistGradientBoostingRegressor` [23] with 300 boosting iterations, learning
rate 0.05 and at most 31 leaves, taking those 149 columns as they come and
dropping sequence structure. It is still a sequential forecaster in the sense
the research question uses; what it lacks is an ordering prior over those lags,
exactly the inductive bias the comparison isolates [11]. Five low-cardinality
calendar columns are the kind of split variable a tree handles natively, which
should matter most on square 5259. Its output is piecewise constant, so it
cannot go above the training maximum, and it stops on a different rule from the
networks: scikit-learn holds out a random tenth of the training windows, which
is optimistic against the chronological validation week the networks stop on
because adjacent windows share 143 of their 144 values.

The predictions recorded in advance were: a few percent over persistence for
the LSTM with persistence mimicry as its failure mode; LSTM-level accuracy for
the TCN at 0.40 s per epoch against 0.58 s; and best or joint-best on 5259 for
the GBT, with no extrapolation above the training maximum.

== Rejected alternatives

SARIMA was rejected on a measurement, not on principle: at seasonal period 144
on the 5472-point training slice one likelihood evaluation costs about 0.8 s
undifferenced and close to four times that once the seasonal difference takes
the state vector from 146 to 290, and a fit of the cheapest order got through
about 150 evaluations in 120 s without converging. The linear family stays in
through AR(144). A GRU is a variant of the same recurrent architecture, and
Transformers lose to simple linear models at short horizons [13], the regime
here.

== Baselines

Seven reference forecasters were fitted first, in EXP-000: persistence,
seasonal naive at the daily lag of 144 and the weekly lag of 1008, a four-day
time-of-day mean and linear autoregression at orders 6, 36 and 144. Three carry
through into the reported run and into every table of Section 6, persistence,
the daily seasonal naive and AR(144), each refitted on the same 144-slot window
as the trained models. Persistence reaches R² 0.9902 and WAPE 6.42 percent on
5161, and AR(144) lowers its MAE by 8.4 percent and its RMSE by 2.9 percent,
the headroom the trained models compete for.

Copying a past value commutes with a monotone transform, so persistence and the
seasonal naive are scale invariant, but AR(144) is fitted in whatever space the
run uses: in Section 6 a linear map of the log1p-standard series. It also
receives the 144 lags only, 145 parameters, without the calendar block the
three trained models get, so its margins bound the architectural gain from
above: part of the leads reported in Section 6.1 could be the features, which
Section 6.4 measures at 0.8 to 2.6 percent of MAE.

== Training and tuning

Adam at 1e-3, batch 128, gradient clipping at 1.0, up to 100 epochs, early
stopping on validation MSE with patience 10 and best-epoch weights restored,
and the learning rate halved once six consecutive epochs pass without
validation improvement. Loss and early stopping work on the scaled target,
every reported metric in original units. The optimiser, batch size, clipping
threshold and loss were held at these values in all 69 neural fits, so every
logged change is attributable to representation or capacity, never to a moving
optimiser; the price is that none of the four is justified by an experiment of
its own. Seed 42 while iterating, three seeds for the final configuration,
reported as mean and standard deviation.

Every model is retrained per area with its own scaler, because the cells run on
different regimes; tuning has two scopes. The input representation was decided
across all three cells by manual one-variable changes (Table 3, EXP-001 to
EXP-003), the rationale written before each: with under 10 percent of headroom
over the linear baseline, why a change helps is the finding, not the change
itself. Capacity was then swept on the highest-traffic cell alone (EXP-010 to
EXP-018), as a one-variable grid at three seeds, nine runs, the TCN over five,
six and seven levels and 16, 32 and 64 channels, the LSTM over 32, 64 and 128
hidden units and the GBT over 15, 31 and 63 leaves, selected on validation.

#apa-table(3, [
  The run-by-run trail behind this section, one row per run or run group.
  `experiments/EXPERIMENT_LOG.md` holds each row's reasoning, written before the
  run, and each run directory holds its own config, metrics and log. The
  validation column reads the TCN on square 5161 throughout, except EXP-000
  (linear AR(144) on the same cell) and EXP-006 and EXP-007 (the TCN on square
  5059); it identifies a row rather than ranking rows against each other.
], [
  #apa-body(
    (auto, 1fr, auto, 1fr),
    align: (left, left, right, left),
    rule,
    [*Run*], [*What changed*], [*Val MAE*], [*What it settled*],
    rule,
    [EXP-000],
    [seven reference forecasters, 1008-slot window, unscaled],
    [109.16],
    [the floor, fixed before any architecture was chosen],

    [EXP-001],
    [first trained run: window 1008 to 144, standard scaling, calendar on],
    [102.19],
    [all three clear persistence on 5161 and 5259; on test MAE only the TCN
     clears AR(144) there],

    [EXP-002],
    [standard to log1p-standard],
    [98.10],
    [adopted, on the training-slice evidence of Section 4.4],

    [EXP-003],
    [calendar features off],
    [99.52],
    [kept on; the two exceptions are both on square 5059],

    [EXP-004, 005],
    [three seeds on 5059, block on and off],
    [-],
    [invalid: sd exactly 0.000 exposed a seed that never reached the model],

    [EXP-006, 007],
    [the same pair, after the fix],
    [88.30, 83.28],
    [the 5059 effect survives three seeds and becomes the failure case],

    [EXP-008],
    [the reported configuration: three areas, three seeds],
    [100.11],
    [every number in Section 6],

    [EXP-010 to 018],
    [capacity grid, one variable per run, three seeds],
    [100.11],
    [confirms the convolutional depth; two models left at a default],
    rule,
  )
])

The TCN's reported configuration is the validation pick, and it confirms the
window the exploratory analysis derived: five levels reach only 125 of the 144
slots and cost 1.5 percent of validation MAE, while seven reach 509 and buy
nothing for 13 percent more time per epoch. The LSTM and the GBT were left at
configurations validation does not select. A wider GBT, 63 leaves, improves
validation MAE by 2.3 percent, beyond any seed spread in that block; 128 hidden
units improve it by 0.6 percent, which is inside one. Neither was adopted,
because the reported campaign and its tests were already closed and the larger
gain would change no conclusion. The shortfall is stated in Section 7.

== Metrics

The brief requires MAE, RMSE and MAPE; WAPE, MASE and R² are reported beside
them. MASE takes its denominator from the training-period seasonal-naive error
[14]; scaling by the scored window inflates every MASE, by a factor of 1.38 on
square 5161, which made a pure seasonal-naive forecast read 1.34 there instead
of 0.975. MAPE is well defined here, traffic never falling below 108 in the
evaluation week, so the epsilon guard never binds. It still over-weights the
01:00 to 06:00 slots by up to 1.3 times their share of the week, hence WAPE
beside it.

== Separating two forecasters

A margin of two or three units between table columns is not a result until it
is tested, so every pairwise claim in Section 6 carries a Diebold-Mariano test
on the paired absolute-error differentials of the 1008 evaluation slots [24].
The differential is taken between the means over the three seeds of each slot's
absolute error, so the quantity tested is exactly the seed-mean MAE the tables
print; testing the mean of the three forecasts instead would score a
three-member ensemble that appears in no table here, and would widen the TCN's
margin on square 5161 from 2.85 to 5.53 units.

Serial correlation in the loss differentials rules out the uncorrected variance
(Ljung-Box p < 0.002 for all fifteen comparisons), so the variance of the mean
uses a Bartlett kernel at lag 15 with the Harvey-Leybourne-Newbold small-sample
correction [25] against a standard normal reference. Only two of fifteen
conclusions change anywhere between lag 0 and lag 144, and both are named where
they arise. Three comparison families were fixed before the tests ran, with
Holm-adjusted values within each. The reordering claim is a conjunction, not a
selection, so it takes an intersection-union test and no multiplicity
correction [26].

== Evaluation convention and timing

Scoring implements Equation (1) as rolling-origin one-step forecasting with
observed history [15]: 1008 origins, none recursive. Inputs may reach back
before 16 December because they are available at the forecast origin; targets
never do, so no held-out target is an input to its own forecast. Validation
follows the same convention.

Training time is fit wall and CPU seconds with epochs and seconds per epoch,
since under early stopping a bare wall time is not representative. Execution
time is the median of 20 warm repetitions over the 1008 evaluation windows,
plus a single-window latency, the device synchronised before the timer stops: a
first call at a new batch shape on MPS is about 30 times slower than warm.
