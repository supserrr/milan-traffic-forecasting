# Evaluation week results, square 5259

One-step-ahead forecasts of Internet activity, 2013-12-16 00:00 to 2013-12-22 23:50 (1008 slots), all models fed the identical 144-slot window. Best value in each column in bold. Source: `experiments/runs/EXP-008/metrics.json`.

| Model | MAE | MAPE (%) | RMSE | WAPE (%) | MASE | R2 |
|---|---|---|---|---|---|---|
| LSTM | 65.70 | 7.11 | 94.13 | 5.19 | 0.1255 | 0.9930 |
| TCN | **61.43** | **6.65** | **89.29** | **4.86** | **0.1173** | **0.9937** |
| GBT | 64.08 | 6.92 | 92.93 | 5.06 | 0.1224 | 0.9931 |
| Persistence | 75.97 | 8.11 | 109.58 | 6.00 | 0.1451 | 0.9904 |
| Seasonal naive (daily) | 470.32 | 71.62 | 861.62 | 37.18 | 0.8981 | 0.4093 |
| Linear AR(144) | 63.69 | 6.66 | 93.74 | 5.03 | 0.1216 | 0.9930 |
