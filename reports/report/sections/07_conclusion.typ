#import "../lib.typ": *

= Conclusion and Future Work

The research question asked how sequential models compare for one-step-ahead mobile
traffic forecasting, and how that comparison varies across areas. They compare closely,
and the ordering is not stable. The dilated temporal convolutional network had the lowest
error on two of the three cells, by 3.4 and 3.5 percent of MAE over a linear autoregression
on 144 lags, but neither lead separates from zero (p = 0.36 and 0.32), and one week on
these cells resolves only differences of about 10 percent. The reordering does separate:
the convolutional network beats the recurrent network on square 5259 and loses to it on
square 5059, jointly at p = 0.004. The exploratory analysis predicts that pair: square 5259
is the one cell whose daily shape changes between regimes instead of shrinking, which is
what a 253-step receptive field is for, while the shape of square 5059 barely moves, making
its calendar features noise. On square 5059 the linear autoregression beat all three trained
architectures on every metric, by 7.1 to 16.0 percent; the widest is the study's one
trained-versus-linear gap that a test separates from zero (p = 0.0009). Aggregated, the
advantage vanishes: the per-area best trained model averages 71.63 MAE against 71.64 for
the linear autoregression, and parity is bought only by picking the winning architecture per
cell, at between 96 and 2400 times the training time, a ratio formed on each cell between
that cell's best trained model and the linear autoregression on the same cell, not across
all model-area pairs.

Information age dominates seasonal structure at this horizon: a forecast from a ten-minute-old
observation reaches R² 0.99, one from a day-old observation at the same time of day between 0.41
and 0.93, so the daily cycle, 89.5 percent of the variance under STL at period 144, is
already contained in the recent past. Heterogeneity between cells 500 m apart is enough to
reorder the models, so cross-area evaluation is not a formality. An input representation can
also harm a cell whose signal it was not derived from.

The evaluation covers one week on the three most extreme cells of a heavy-tailed distribution,
so nothing here generalises to a typical cell without retesting. A defect that let both
trainable models re-seed themselves meant several early runs repeated a single seed; the
reported configuration was rerun at three seeds after the fix, but the intermediate comparisons
carry unmeasured variance. Capacity was swept only after the campaign closed, and it found two
of the three architectures at configurations validation does not select, so they carry a
reasonable configuration rather than the best one tried; the convolutional network, whose
configuration the sweep does select, may still be under-tuned on the cell where it failed,
since the sweep ran on a different cell. Finally, the memory and timing evidence come from two
different machines.

Each limitation suggests a next step. Feeding the calendar block at every time step, instead of
concatenating it to a single summary vector as both neural models do, would test whether the
convolutional failure is about the head's inability to attenuate a term it should ignore.
Evaluating at horizons of one hour and one day would show whether the small gaps found here are
a property of the models or of the horizon. Using neighbouring cells as covariates [5] would ask
whether the regime differences documented in Section 4 are predictable from spatial context.
Repeating the comparison on cells from across the traffic distribution, not its extreme tail,
would establish whether the ranking instability is general or a feature of the grid's busiest
corner.
