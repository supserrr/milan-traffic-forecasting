#import "../lib.typ": *

#align(center)[#text(weight: "bold")[Abstract]]
#v(0.3em)

#block[
#set par(first-line-indent: 0pt)
Capacity assignment, overnight base-station sleep scheduling and anomaly alarms all act on a
ten-minute horizon, and each acts on a forecast of the next slot. This study compares three
architecturally distinct sequential models for one-step-ahead forecasting of mobile Internet
traffic on the Telecom Italia Milan grid, and asks whether their ranking survives a change of
geographical area. A streaming aggregation reduces 19.4 GiB of Call Detail Record text to a
357 MB memory-mapped matrix at a peak of 109 MB above baseline per daily file, a fifth of a naive
load. A long short-term memory network, a dilated temporal convolutional network and
gradient-boosted regression trees are trained per area on an identical 144-slot window and
evaluated on 16 to 22 December against persistence, seasonal naive and linear autoregressive
baselines. The ordering of the three architectures reverses between cells 500 m apart, and that
reversal is the one result a predictive-accuracy test supports: the convolutional network beats
the recurrent network on one cell and loses to it on another, jointly at p = 0.004. Its 3.4 and 3.5
percent leads over a 145-parameter linear autoregression are not separable from zero (p = 0.36
and 0.32), while on the third cell that linear model beats all three trained architectures on
every metric, by 7.1 to 16.0 percent of the trained model's error, and the widest of those
margins is the study's one trained-versus-linear gap that a test separates from zero
(p = 0.0009). No architecture betters
it on average, it trains in 0.02 s against 44 s, and persistence alone reaches R² 0.99. The
calendar features that help seven of nine model-area pairs cost the convolutional model 13
percent on the cell whose working-day contrast is weakest, the reported failure case.
]

#block[#h(0.5in)#emph[Keywords:] time series forecasting, mobile network traffic, sequential
models, call detail records, temporal convolutional network, forecast evaluation]

#pagebreak()
