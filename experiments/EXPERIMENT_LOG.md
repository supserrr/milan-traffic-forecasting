# Experiment log

Append-only. **One row per training run, written as the run finishes** - not reconstructed
afterwards. This file is the evidence for the "Experimentation & Hyperparameter Tuning" criterion
(10 pts), and a log written in one sitting at the end reads exactly like one.

Each run also writes `experiments/runs/EXP-NNN/` containing `config.yaml`, `metrics.json`,
`predictions.npy` and `run.log`. Never overwrite a previous run.

## How to fill this in

- **Changed** - only what differs from the previous run. One variable at a time while reasoning
  manually; a grid search is a single row with the search space in *Changed*.
- **Val / Test MAE** - test metrics on the fixed evaluation week, in the original scale.
- **Why & next** - the sentence that makes the log worth reading: why this change was made, what
  the result showed, and what it implies for the next run. "Tried it" is not a reason.

---

## Runs

| ID | Date | Model | Changed | Val MAE | Test MAE | Test RMSE | Train (s) | Infer (ms) | Why & next |
|----|------|-------|---------|--------:|---------:|----------:|----------:|-----------:|------------|
| EXP-000 | 2026-09-15 | 7 baselines, none trained on a sequence model | first run: fixes the floor before any architecture is chosen (row shows `linear_ar_144` on square 5161; full table below) | 109.16 | 84.64 | 123.54 | 0.020 | 0.22 | Persistence alone reaches R2 0.9902 and WAPE 6.42% on square 5161, so the task is far easier than the daily cycle suggests. Linear AR(144) is the best baseline and beats persistence by only 8.4% RMSE, which is the budget a neural model competes for. Next: EXP-001, one architecture at seq_len 144, judged against these numbers and not against zero. |

---

## Decision trail

Narrative notes that do not fit a table row: dead ends, surprises, and the reasoning behind
changes of direction. The report's Methodology and Results sections are written from here.

### EXP-000: the baseline floor (2026-09-15)

Reported areas, the three with the highest total Internet traffic. Squares 4159 and 4556 were run
as well and appear below the fold; they are context for the cross-area question, not part of the
required three. Every model receives an identical 1008-step window, unscaled, and predicts one step
ahead over 2013-12-16 00:00 to 2013-12-22 23:50 (1008 slots, all of them predicted). MASE is scaled
by the seasonal-naive error of the training period, not of the evaluation week.

| area | model | test MAE | test RMSE | test MAPE % | test WAPE % | MASE | R2 | fit s | infer ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5161 | `naive` | 92.80 | 134.88 | 9.19 | 6.42 | 0.268 | 0.9902 | 0.000 | 0.95 |
| 5161 | `seasonal_naive_144` | 338.59 | 619.04 | 25.94 | 23.43 | 0.980 | 0.7934 | 0.000 | 0.16 |
| 5161 | `seasonal_naive_1008` | 300.75 | 432.95 | 28.97 | 20.81 | 0.870 | 0.8990 | 0.000 | 0.21 |
| 5161 | `time_of_day_mean_4d` | 360.53 | 587.04 | 26.93 | 24.95 | 1.043 | 0.8143 | 0.000 | 0.22 |
| 5161 | `linear_ar_6` | 90.79 | 129.30 | 10.05 | 6.28 | 0.263 | 0.9910 | 0.007 | 0.95 |
| 5161 | `linear_ar_36` | 91.74 | 127.81 | 13.17 | 6.35 | 0.265 | 0.9912 | 0.006 | 0.39 |
| 5161 | `linear_ar_144` | 84.64 | 123.54 | 9.91 | 5.86 | 0.245 | 0.9918 | 0.020 | 0.22 |
| 5059 | `naive` | 81.52 | 114.38 | 7.96 | 6.44 | 0.303 | 0.9851 | 0.000 | 0.97 |
| 5059 | `seasonal_naive_144` | 171.74 | 245.87 | 18.02 | 13.57 | 0.638 | 0.9313 | 0.000 | 0.17 |
| 5059 | `seasonal_naive_1008` | 259.95 | 361.12 | 25.73 | 20.54 | 0.966 | 0.8519 | 0.000 | 0.15 |
| 5059 | `time_of_day_mean_4d` | 190.51 | 272.80 | 20.87 | 15.05 | 0.708 | 0.9155 | 0.000 | 0.21 |
| 5059 | `linear_ar_6` | 82.84 | 112.97 | 8.45 | 6.55 | 0.308 | 0.9855 | 0.007 | 0.53 |
| 5059 | `linear_ar_36` | 79.33 | 104.56 | 10.19 | 6.27 | 0.295 | 0.9876 | 0.004 | 0.36 |
| 5059 | `linear_ar_144` | 71.28 | 99.34 | 7.88 | 5.63 | 0.265 | 0.9888 | 0.022 | 0.22 |
| 5259 | `naive` | 75.97 | 109.58 | 8.11 | 6.00 | 0.144 | 0.9904 | 0.000 | 0.31 |
| 5259 | `seasonal_naive_144` | 470.32 | 861.62 | 71.62 | 37.18 | 0.891 | 0.4093 | 0.000 | 0.15 |
| 5259 | `seasonal_naive_1008` | 210.24 | 289.50 | 28.56 | 16.62 | 0.398 | 0.9333 | 0.000 | 0.14 |
| 5259 | `time_of_day_mean_4d` | 653.14 | 937.61 | 103.18 | 51.63 | 1.237 | 0.3006 | 0.000 | 0.17 |
| 5259 | `linear_ar_6` | 70.70 | 100.83 | 7.90 | 5.59 | 0.134 | 0.9919 | 0.001 | 0.33 |
| 5259 | `linear_ar_36` | 67.64 | 95.70 | 7.82 | 5.35 | 0.128 | 0.9927 | 0.004 | 0.31 |
| 5259 | `linear_ar_144` | 66.22 | 93.42 | 7.77 | 5.23 | 0.125 | 0.9931 | 0.019 | 0.26 |

Also run, for cross-area context:

| area | model | test MAE | test RMSE | test MAPE % | test WAPE % | MASE | R2 | fit s | infer ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4159 | `naive` | 15.95 | 21.54 | 6.98 | 6.57 | 0.196 | 0.9688 | 0.000 | 0.21 |
| 4159 | `seasonal_naive_144` | 51.19 | 84.64 | 21.80 | 21.08 | 0.629 | 0.5179 | 0.000 | 0.15 |
| 4159 | `seasonal_naive_1008` | 73.41 | 97.38 | 27.96 | 30.23 | 0.902 | 0.3619 | 0.000 | 0.14 |
| 4159 | `time_of_day_mean_4d` | 64.24 | 91.99 | 31.54 | 26.46 | 0.789 | 0.4307 | 0.000 | 0.17 |
| 4159 | `linear_ar_6` | 15.34 | 20.69 | 6.80 | 6.32 | 0.188 | 0.9712 | 0.001 | 0.20 |
| 4159 | `linear_ar_36` | 15.69 | 20.39 | 7.31 | 6.46 | 0.193 | 0.9720 | 0.004 | 0.25 |
| 4159 | `linear_ar_144` | 14.91 | 19.95 | 6.68 | 6.14 | 0.183 | 0.9732 | 0.021 | 0.20 |
| 4556 | `naive` | 28.86 | 39.62 | 6.60 | 6.36 | 0.256 | 0.9407 | 0.000 | 0.20 |
| 4556 | `seasonal_naive_144` | 76.34 | 108.35 | 17.46 | 16.82 | 0.677 | 0.5568 | 0.000 | 0.15 |
| 4556 | `seasonal_naive_1008` | 96.39 | 125.86 | 21.69 | 21.24 | 0.855 | 0.4019 | 0.000 | 0.14 |
| 4556 | `time_of_day_mean_4d` | 80.35 | 108.17 | 19.32 | 17.70 | 0.712 | 0.5582 | 0.000 | 0.17 |
| 4556 | `linear_ar_6` | 28.26 | 37.41 | 6.67 | 6.22 | 0.251 | 0.9472 | 0.002 | 0.21 |
| 4556 | `linear_ar_36` | 28.06 | 36.89 | 6.68 | 6.18 | 0.249 | 0.9486 | 0.004 | 0.23 |
| 4556 | `linear_ar_144` | 26.60 | 35.64 | 6.18 | 5.86 | 0.236 | 0.9521 | 0.020 | 0.25 |

Four readings that change what comes next:

1. **Persistence is the real competitor, not zero.** It reaches R2 0.9902 / 0.9851 / 0.9904 on the
   three required areas and WAPE near 6.4 percent on all five. Any reported model that does not
   clear it has learned nothing that the previous sample did not already carry.
2. **Seasonal repetition is much weaker than the variance decomposition implies.** Seasonal naive
   at the daily lag is 3.6x worse than persistence on square 5161 and 6.2x worse on 5259, and the
   4-day time-of-day mean is worse still (MASE 1.04 and 1.24, i.e. beaten by a single lagged day).
   That comparison measures information age, not seasonal strength: a 24-hour-old observation
   losing to a 10-minute-old one at h=1 is near-inevitable. Tested directly, adding daily and weekly
   lag terms (143 to 145 and 1007 to 1009) to a 6-lag linear model cuts validation MAE by 11.4, 10.4
   and 4.9 percent on 5161, 5059 and 5259, more than the whole AR(144)-over-persistence headroom, so
   the calendar is worth modelling explicitly rather than through a single seasonal lag. The test
   week disagrees on 5059 (weekly lags worsen RMSE there), which is itself a finding for the report.
3. **Square 5259 breaks the seasonal baselines specifically** (seasonal naive MAE 470.3 against
   persistence 76.0, R2 0.41), and the cause is now identified. The cell is weekday-only: weekday
   mean 1597 against weekend mean 536, and on the five non-working days of the first fortnight (Fri
   Nov 1, the Ognissanti public holiday, and both weekends) its daily maximum falls at night, four of
   them between 00:50 and 02:10 and Sat Nov 9 at 22:50, against 13:00 to 17:20 on every working day.
   The regime follows the working calendar, not the day-of-week label. The daily lag therefore
   crosses a regime boundary twice a week, and **75 percent of its evaluation-week error falls on
   the Monday and the Saturday** alone. The weekly lag, which preserves day of week, more than
   halves the error (210.2). See finding F-02 in the EDA working notes. This makes 5259 the natural
   failure case, and argues for explicit calendar features over a single seasonal lag.
4. **Linear AR(144) beats persistence by 8.4 percent RMSE on square 5161** (123.54 against 134.88)
   and by 14.7 percent on 5259. The headroom between "repeat the last value" and "fit 145 linear
   coefficients to one day of history" is therefore small, and it bounds what a nonlinear model can
   claim. Fitting costs 20 ms.

### Sequence length

PACF on the full 8928-slot series (Levinson-Durbin, 95 percent band +/-0.0207) has its last
partial correlation above 0.05 at lag 149 on square 5161, and at 146, 159, 151 and 146 on the other
four cells. The partial at the daily lag itself is only -0.045, and at lag 288 it is -0.005. The
evidence therefore supports a window of roughly 144 and gives no support for 1008.

Two disclosures. The 0.05 cutoff is a practical magnitude threshold, 2.4x the +/-0.0207 band; by
the band alone the PACF stays significant past lag 400 on every cell at n=8928, so the band is not
what chose the window. And these statistics were computed on the full series, including the
held-out week; the report quotes the training-slice versions produced by `scripts/run_tsa.py`
(`reports/tables/tsa_5161.md`), and the AR-order sweep below, which is fit on training windows
only, is the primary evidence. Orders between 144 and 1008 were fitted afterwards: 216 gives the
best three-cell mean validation MAE (92.75 against 93.34 for 144, a 0.6 percent gain at 1.5x the
window), 288 is worse again and 1008 is clearly worse, so 144 is kept as a cost/accuracy trade
rather than as the located optimum.

EXP-000 agrees from the other direction: validation MAE on square 5161 falls 115.08, 114.59, 109.16
across AR orders 6, 36 and 144, so information is still being added out to a full day, but the gain
from 6 to 36 lags is 0.4 percent against 4.8 percent from 36 to 144.

EXP-000 itself uses 1008 only because the weekly seasonal naive needs that lag to exist. Trained
models should use 144 and say so. The cost of the longer window is recorded below.

### Validation window

A 1008-step window does not fit inside the validation split: 15 percent of the 6480 pre-evaluation
slots is 972 slots, which is shorter than one window, so zero validation windows can be built
strictly inside it. Rather than shrink the window or inflate val_frac, validation now uses the same
convention already used for the evaluation week: inputs may reach back into the period before,
targets stay strictly inside. Inputs then come from training data, which is exactly what is
available when selecting a model, and no evaluation-week observation is touched.
`features.context_windows` takes a `target_slice` argument for this, and both held-out periods go
through the same code path. At seq_len 144 the constraint disappears, so this note applies mainly
to EXP-000.

### Normalisation

EXP-000 runs unscaled (`scaling: none`) on purpose: a reference forecast whose value depended on a
transform fitted to the training period would be a poor reference, and all seven baselines are
scale-equivariant, so the choice cannot flatter them.

For trained models the evidence points at log1p. The correlation between the rolling 144-slot mean
and the rolling 144-slot standard deviation is 0.940 on square 5161 and 0.900 to 0.975 across the
five cells, i.e. the noise is multiplicative; on log1p the same correlation falls to 0.177 and
0.165 to 0.871 (those figures were computed on the full series; on the training slice alone, rows
0:5472, the raw correlation on 5161 is 0.973 and the log1p one 0.235, per `reports/tables/tsa_5161.md`).
Decomposition agrees in direction whatever the method: on the top cell, training slice, the
remainder share of variance falls under log1p for STL at period 144 (0.048 raw to 0.023), for MSTL
at periods 144 and 1008 (0.020 to 0.012, with the weekly component alone carrying 0.111 of the raw
variance) and for a classical centred moving-average decomposition (0.118 to 0.050). The method
changes the level of the number by a factor of six, so the report names the method it quotes.
Standardisation and log1p-standard should be compared explicitly in an experiment rather than
assumed.

### Harness corrections before EXP-001 (2026-09-15)

An audit of the evaluation path before the first trained run found three defects, in
`evaluate.run_model`, `metrics._prep` and `features.Scaler`. None of them was visible to the 43
tests then passing: the suite covered the pieces carefully and never once imported the module that
assembles them. None of the three changes EXP-000 either, because all three are latent under
`scaling: none`; the run was re-scored against its stored artefacts afterwards to confirm it, at 490
metric values with a maximum absolute difference of 0.000e+00.

**Predictions were never inverted.** `run_model` scored `model.predict(...)` directly against
`y_test_true`, which had already been inverse-transformed. Under `scaling: none` the scaler is the
identity, so EXP-000 is untouched, but `configs/experiment.yaml` defaults trained runs to
`standard`, under which persistence on square 5161 scored MAE 1445.19 and R2 -1.1243 in place of
92.80 and 0.9902. The failure mode is the dangerous kind: finite, plausible numbers in the wrong
units rather than an error that stops the run. `run_model` now inverts both prediction arrays, and
persistence scores 92.802 / 134.878 / 0.9902 identically under all five scaling methods, which is
the invariance any scale-equivariant model must show. The timer still covers `predict` alone, so the
inversion is not charged to whichever model happens to run it.

That settles a contract question the Methodology section should state outright: models are fed
scaled inputs and return predictions in that same space, and the harness inverts once, centrally.
Inverting inside each model was never possible anyway, since no model is handed the scaler.

**Non-finite predictions were silently dropped.** `metrics._prep` masked out every non-finite pair
and averaged whatever survived. Scored that way, a forecast emitting NaN from step 200 of 1008
returns MAE 69.04 against intact persistence's 92.80, beating it on RMSE and MASE as well. A NaN
loss is the commonest way an unstabilised training run fails, so the harness was arranged to report
precisely that failure as the best result in the table. One non-finite pair now makes every metric
NaN, and `n_scored` and `n_dropped` are carried in every result row and written to `metrics.json`,
so the artefacts record how many of the 1008 slots actually produced a number.

**A misspelled scaling name disabled scaling instead of failing.** `Scaler` validated nothing, so
any unrecognised method fell through to the identity branch and returned a fitted scaler that did
nothing. Combined with the first defect this was perverse: a typo in `--scaling` was the only
setting under which the reported metrics were in the right units. The method is now checked against
`ScaleMethod` and the CLI flag carries `choices=`.

`tests/test_evaluate.py` pins all three at the integration level, and each fix was confirmed by
reverting it and watching between one and six of those tests fail. Worth a paragraph in the report
rather than a quiet fix: the episode is a concrete argument for feeding all three models through one
shared path, and for the distinction between a green test suite and correct numbers.
