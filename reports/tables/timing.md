# Training and execution time

Source: `experiments/runs/EXP-008/`.

| Model | Parameters | Device | Epochs | s / epoch | Training (s) | Training CPU (s) | Batch (ms) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| LSTM | 17222 | mps | 42 | 0.67 | 28.17 | 12.16 | 14.10 | 3.332 |
| TCN | 34758 | mps | 34 | 1.28 | 44.06 | 18.91 | 20.89 | 3.305 |
| GBT | 3795 | cpu | 122 | 0.02 | 2.05 | 10.16 | 9.21 | 7.118 |
| Persistence | 0 | cpu | - | - | 0.00 | 0.00 | 0.02 | 0.002 |
| Seasonal naive (daily) | 0 | cpu | - | - | 0.00 | 0.00 | 0.02 | 0.002 |
| Linear AR(144) | 145 | cpu | - | - | 0.02 | 0.09 | 0.06 | 0.004 |

**Hardware.** Darwin | 27.0.0 | arm64 | 8P/8L cores | 16 GiB RAM | device:cpu,mps

**Method.** Training time is the wall time of one fit call, reported with the number of epochs actually run under early stopping and the mean seconds per epoch, because a bare wall time under early stopping is not representative of the model. Execution time is reported twice: as the median of 20 repetitions of predicting the whole 1008-slot evaluation week after one untimed warm-up call, and as single-window latency, the median of 50 single-window predictions. The device is synchronised before every timer stops. single-shot timings varied five-fold for identical operations in EXP-000, and the first MPS call at a new batch shape is about thirty times slower than a warm one Values are averaged over the three areas.
