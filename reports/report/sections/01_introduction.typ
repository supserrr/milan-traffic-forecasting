#import "../lib.typ": *

= Introduction

A mobile operator reads its network in ten-minute slots. Capacity assignment to a
cell, the decision to put a base station into a sleep mode overnight to save
energy, and the alarm that flags an outage or a flash crowd all act on that
horizon, and each acts on a forecast of the next slot [1], [2]. A forecast at
10-minute resolution decides how much capacity to switch on before demand
arrives, and how much to switch off when it will not.

This study takes that one-step problem to the Milan Call Detail Record aggregates
released for the Telecom Italia Big Data Challenge [3]: 62 days of Internet
activity for 10 000 grid cells at 10-minute resolution, 319 896 289 raw rows
in 19.4 GiB of text. The research question is fixed by the brief and is quoted
verbatim:

#block(inset: (left: 0.5in, right: 0.5in))[
  #set par(first-line-indent: 0pt)
  #emph[How do different sequential models compare for one-step-ahead mobile network traffic
  forecasting, and how does their performance vary across geographical areas with different
  traffic characteristics?]
]

The objective is to defend the decisions around three architecturally distinct
models run under one protocol on three areas, and to explain the differences from
the data, rather than to reach the lowest error. Two measurements set the terms.
First, persistence, the forecast ŷ(t+1) = y(t), already reaches R² 0.99 on the
highest-traffic cell, and the best linear baseline lowers its mean absolute error
by only 8.4 percent. Second, the three highest-traffic cells, squares 5161, 5059
and 5259, lie within about 500 m of each other yet run on different weekly
regimes, with weekday-to-weekend traffic ratios of 0.79, 1.29 and 2.98. The
second half of the research question is therefore a question about neighbours.
