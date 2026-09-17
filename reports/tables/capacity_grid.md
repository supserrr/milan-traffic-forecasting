# Capacity grid, square 5161

Runs EXP-010 to EXP-018, three seeds each (42, 1337, 2024), one variable moved per run
from the reported configuration, everything else held at EXP-008's values (seq_len 144,
log1p-standard scaling, calendar features on). The EXP-008 rows are the reported
configuration itself, at the same seeds on the same cell. EXP-010 re-ran that
configuration for the TCN as a control and reproduced it bit-for-bit on all three seeds,
which is what licenses reading the two runs as one grid; its duplicate row is dropped.

`changed` is derived by diffing each run's resolved model config against the others in
the set, so a row cannot claim a variable its run did not move. The arrow marks the
configuration **validation** selects per architecture; the evaluation-week column is
shown for completeness and decides nothing.

The TCN receptive field is `1 + 2(k-1) * sum(2^i)` at kernel 3. Five levels reach 125
steps and so cannot see the whole 144-slot window; six is the first depth that covers it.

| run | model | changed | n_params | receptive_field | val_MAE | val_MAE_sd | test_MAE | test_MAE_sd | s_per_epoch | pick |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|:--|
| EXP-018 | GBT | max_leaf_nodes=63 | 9765.00 |  | 103.98 | 0.80 | 86.86 | 0.38 | 0.02 | <-- validation selects |
| EXP-017 | GBT | max_leaf_nodes=15 | 2040.00 |  | 105.70 | 1.33 | 87.48 | 1.42 | 0.01 |  |
| EXP-008 (reported) | GBT | max_leaf_nodes=31 | 3720.00 |  | 106.42 | 1.30 | 87.08 | 1.87 | 0.02 |  |
| EXP-016 | LSTM | hidden=128 | 67206.00 |  | 103.65 | 3.02 | 83.56 | 1.63 | 0.95 | <-- validation selects |
| EXP-008 (reported) | LSTM | hidden=64 | 17222.00 |  | 104.28 | 2.42 | 84.42 | 1.41 | 0.71 |  |
| EXP-015 | LSTM | hidden=32 | 4518.00 |  | 106.20 | 1.31 | 88.38 | 3.92 | 0.63 |  |
| EXP-008 (reported) | TCN | channels=32, levels=6 | 34758.00 | 253 | 100.11 | 1.75 | 82.13 | 6.65 | 1.28 | <-- validation selects |
| EXP-014 | TCN | channels=64, levels=6 | 137094.00 | 253 | 100.52 | 1.70 | 81.73 | 6.29 | 1.90 |  |
| EXP-011 | TCN | channels=32, levels=5 | 28486.00 | 125 | 101.66 | 2.29 | 82.86 | 5.04 | 1.14 |  |
| EXP-012 | TCN | channels=32, levels=7 | 41030.00 | 509 | 101.71 | 3.72 | 84.36 | 4.85 | 1.45 |  |
| EXP-013 | TCN | channels=16, levels=6 | 8934.00 | 253 | 105.95 | 2.44 | 85.47 | 2.31 | 1.02 |  |
