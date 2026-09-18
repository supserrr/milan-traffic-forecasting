#import "../lib.typ": *

= Use of AI Assistance

The assignment permits AI tools and requires substantial use to be disclosed.

The tool was Claude Code, Anthropic's command-line coding assistant, running Claude
Opus. It wrote the first draft of most of the Python package, including the streaming
ingestion, the windowing and scaling helpers, the training loop, the three model modules
and the test suite, and it drafted prose for this report from the author's working notes.
It was also used as a reviewer: independent audit passes were run over the repository,
and the numerical claims of Sections 4 and 6 were recomputed from the stored matrix by a
further pass instructed to refute the first, not to confirm it.

On the prose: the argument of each section, the decisions it defends and every
interpretive claim were written by the author first, in the working notes and the
decision register the assistant then drafted from. Every drafted paragraph was rewritten
by the author against the stored numbers, and every quantitative claim was checked
against the artefact that produces it. The abstract and this section were written by the
author directly.

Review mattered more than generation. Three of the four defects were in the harness:
metrics scored in the scaled space against ground truth in the original scale, which
would have reported MAE 1445 in place of 92.8 for persistence; non-finite predictions
dropped silently from the average; and an unrecognised scaling name falling through to an
identity transform instead of raising. All three passed a suite of 43 component tests
that never imported the module assembling them. The fourth was found by an experiment:
both trainable models re-seeded themselves inside their fitting routine, and the symptom
was a standard deviation of exactly zero across three nominally different seeds. Each is
recorded in `experiments/EXPERIMENT_LOG.md` and pinned by an integration test that fails
if the fix is reverted.

Every figure and table cell is produced by a script in the repository and can be
regenerated from a clean clone; the numbers quoted in prose are copied from those tables,
except the descriptive statistics of Section 4, which `scripts/make_eda_stats.py` writes
to `reports/tables/eda_stats.md`. The interpretations, the argument and the conclusions
are the author's own.

== Resources

The source code, the experiment log with every run's configuration and metrics, and the
scripts that regenerate every figure and table are in the project repository [27]#if video-url != "" [, and the video presentation accompanying this report is at [28]].
