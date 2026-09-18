# Model selection checks

Produced by `scripts/check_model_selection.py`, which reads `data/processed/` and
writes this file. It exists to put a script behind two claims that were otherwise
sourced to an interactive session: the stability of the three evaluated cells, and
the measured cost of a seasonal ARIMA at period 144.

## 1. Does the same top three come first under other definitions of busiest?

Four definitions over the same processed matrix. All 62 days is the definition the
project uses. Rows before the evaluation week is the leakage-free version of it:
nothing after 15 December contributes. Excluding 24 to 26 December removes the
holiday collapse. The mean over non-zero slots divides out how often a cell is
active at all, so it cannot be won on volume alone.

| rank | total, all 62 days | total, rows before the evaluation week | total, excluding 24 to 26 December | mean over non-zero slots |
|--:|--:|--:|--:|--:|
| 1 | 5161 | 5161 | 5161 | 5161 |
| 2 | 5059 | 5059 | 5059 | 5059 |
| 3 | 5259 | 5259 | 5259 | 5259 |
| 4 | 5061 | 5061 | 5061 | 5061 |
| 5 | 5258 | 5258 | 5258 | 5258 |

**The top three agree under all four definitions: True.** They are 5161, 5059 and 5259.

Scores for every cell that appears above. Sums are in millions of activity units,
the mean is in activity units per active slot. The values are scaled CDR counts and
carry no physical unit.

| square | total, all 62 days | total, rows before the evaluation week | total, excluding 24 to 26 December | mean over non-zero slots |
|:--|--:|--:|--:|--:|
| 5161 | 12.74 | 9.98 | 12.50 | 1426.98 |
| 5059 | 11.17 | 8.85 | 10.97 | 1251.22 |
| 5259 | 10.49 | 8.65 | 10.38 | 1174.48 |
| 5061 | 9.58 | 7.53 | 9.40 | 1073.51 |
| 5258 | 8.71 | 7.18 | 8.60 | 975.30 |


## 2. What does one SARIMA likelihood cost at period 144?

Square 5161, training slice rows 0:5472
(1 November to 8 December), the same slice every model is fitted on. A daily
seasonal term at period 144 puts about 144 lags into the
state vector, and the Kalman filter carries a matrix of that size across every
observation, once per likelihood evaluation. Each row is the median of
3 evaluations at `start_params` after one untimed warm-up.

| order | seasonal order | state dim | free params | min (s) | median (s) | max (s) |
|:--|:--|:--|:--|--:|--:|--:|
| (1, 0, 1) | (1, 0, 1, 144) | 146 | 5 | 0.77 | 0.77 | 0.77 |
| (2, 0, 2) | (1, 0, 1, 144) | 147 | 7 | 0.82 | 0.83 | 0.87 |
| (1, 0, 1) | (0, 1, 1, 144) | 290 | 4 | 2.84 | 2.84 | 2.85 |
| (1, 1, 1) | (1, 1, 1, 144) | 291 | 5 | 2.54 | 2.56 | 3.05 |

**One likelihood evaluation costs 0.77 to 3.05 s** across the 4 candidate orders. Differencing is what moves it: the undifferenced orders carry a state vector of 146 to 147, a difference at the seasonal period takes it to 290 to 291, and the likelihood then costs 3.4 times as much.

A fit needs many of those. Below is what a single fit of the cheapest order does
with a fixed budget. It cannot hang: the likelihood refuses to run once the budget
is spent, and the optimiser is capped at a fixed number of iterations besides.

| quantity | value |
|:--|:--|
| order | (1, 0, 1) x (1, 0, 1, 144) |
| observations in the training slice | 5472 |
| state dimension | 146 |
| time budget (s) | 120 |
| iteration cap | 50 |
| likelihood evaluations completed | 155 |
| wall time of the attempt (s) | 120.7 |
| outcome | stopped by the 120 s budget, unconverged |

**Hardware.** Darwin | 27.0.0 | arm64
