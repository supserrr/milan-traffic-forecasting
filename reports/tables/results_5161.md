# Evaluation week results, square 5161

One-step-ahead forecasts of Internet activity, 2013-12-16 00:00 to 2013-12-22 23:50 (1008 slots), all models fed the identical 144-slot window. Best value in each column in bold. Source: `experiments/runs/EXP-008/metrics.json`.

| Model | MAE | MAPE (%) | RMSE | WAPE (%) | MASE | R2 |
|---|---|---|---|---|---|---|
| LSTM | 84.42 | 8.02 | 129.35 | 5.84 | 0.2430 | 0.9910 |
| TCN | **82.13** | 7.83 | **124.35** | **5.68** | **0.2364** | **0.9916** |
| GBT | 87.08 | 8.41 | 130.39 | 6.03 | 0.2507 | 0.9908 |
| Persistence | 92.80 | 9.19 | 134.88 | 6.42 | 0.2671 | 0.9902 |
| Seasonal naive (daily) | 338.59 | 25.94 | 619.04 | 23.43 | 0.9747 | 0.7934 |
| Linear AR(144) | 84.98 | **7.74** | 130.92 | 5.88 | 0.2446 | 0.9908 |
