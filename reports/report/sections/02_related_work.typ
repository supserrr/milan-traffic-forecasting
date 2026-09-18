#import "../lib.typ": *

= Related Work

== One-step forecasting is not where this dataset has been studied

The Milan release is a dataset, not a benchmark [3], and the forecasting work built on it has run
towards longer horizons and spatial structure. Zhang and Patras forecast the whole grid at horizons
up to ten hours, reporting up to 61 percent lower NRMSE than Holt-Winters, ARIMA, MLP and SVM, the
margin growing with the horizon [2]; Chen et al. judge a multivariate LSTM on the same two months by
a downstream planning objective [4]; and operator data from other cities points the same way [5],
[1]. The closest short-horizon neighbour, STDenseNet, is citywide and spatial where this study is
per-cell and univariate [6], and that is the scope decision: holding the input to one cell's own
history makes the comparison about temporal inductive bias rather than spatial context. Trinh et al.
are closest to the present setting, a single-cell LSTM at one and fifteen steps ahead [7]. This study
fixes the horizon at one 10-minute slot, models three cells separately, and tests on 16 to 22
December instead of the window of [2], which straddles Christmas.

== Recurrence, convolution, and a model with no sequence bias at all

The LSTM cell holds information across long lags through gated constant-error carousels,
demonstrated on minimal lags in excess of 1000 steps [8]; a 144-step window sits well inside that
reach, so if the recurrent model degenerates here, memory is not the explanation. The convolutional
alternative makes memory an explicit parameter instead of a learned property: Bai, Kolter and Koltun
stack dilated causal convolutions in residual blocks, reporting that this generic architecture
outperforms recurrent networks across the standard benchmark suite [9]. The third family drops
sequence structure entirely: gradient boosting fits additive expansions of regression trees by
steepest descent in function space [10], and Elsayed et al. feed such a model a flattened window of
lagged targets to beat eight deep forecasters, the gain coming from the windowing rather than from
the trees [11]. That result dictates the comparison: the tree ensemble receives the same 144 lags and
calendar features as the two networks, so any gap is attributable to inductive bias.

== The case that simple methods are hard to beat

Makridakis et al. ran ten machine-learning methods against eight statistical ones on 1045 monthly M3
series under one protocol; the six most accurate were statistical, and at one step ahead an MLP
scored 8.39 percent sMAPE against 7.19 percent for ARIMA [12]. Zeng et al. make the same argument
against Transformers: a single linear layer from look-back window to forecast beats five of them on
all nine benchmarks [13]. The measurement here agrees: persistence alone reaches R² 0.9902 and
WAPE 6.42 percent on square 5161. A linear AR over the same 144 lags is therefore reported in every
table, so a nonlinear model must beat a linear map of its own inputs and not merely the previous
value. Transformers were excluded on this evidence and on their cost over a univariate 10-minute
series.

== Measuring accuracy when the series never repeats its scale

Percentage errors degenerate near zero and scale-dependent errors cannot be pooled across series, so
Hyndman and Koehler propose the mean absolute scaled error [14]; accuracy is assessed with a rolling
forecasting origin [15]. MASE is therefore reported with a training-period denominator rather than
one taken from the window being scored, a distinction Section 5.10 quantifies. Seasonal naive is run
at both the daily lag of 144 and the weekly lag of 1008, which is how the weekday regime of square
5259 surfaced, at MAE 470.3 against 210.2.

== The gap

Work on the Milan release concentrates on spatio-temporal architectures at multi-hour horizons [2],
[4]; the 10-minute one-step horizon belongs to the regime the critical literature describes [12],
[13], [11]. No study on this dataset compares a recurrent model, a convolutional model and a tree
ensemble on identical inputs, against persistence, seasonal-naive and linear baselines, across cells
whose weekday-to-weekend ratios span 0.79 to 2.98, using scale-free errors and recorded compute cost.
