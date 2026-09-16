# Evaluation week results, square 5059

One-step-ahead forecasts of Internet activity, 2013-12-16 00:00 to 2013-12-22 23:50 (1008 slots), all models fed the identical 144-slot window. Best value in each column in bold. Source: `experiments/runs/EXP-008/metrics.json`.

| Model | MAE | MAPE (%) | RMSE | WAPE (%) | MASE | R2 |
|---|---|---|---|---|---|---|
| LSTM | 71.84 | 6.77 | 103.68 | 5.68 | 0.2658 | 0.9878 |
| TCN | 78.88 | 7.22 | 115.32 | 6.23 | 0.2918 | 0.9849 |
| GBT | 71.32 | 6.79 | 102.68 | 5.64 | 0.2639 | 0.9880 |
| Persistence | 81.52 | 7.96 | 114.38 | 6.44 | 0.3016 | 0.9851 |
| Seasonal naive (daily) | 171.74 | 18.02 | 245.87 | 13.57 | 0.6354 | 0.9313 |
| Linear AR(144) | **66.25** | **6.30** | **96.67** | **5.24** | **0.2451** | **0.9894** |
