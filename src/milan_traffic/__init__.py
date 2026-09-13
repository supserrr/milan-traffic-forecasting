"""Milan mobile network traffic forecasting.

A comparative study of sequential models for one-step-ahead forecasting of Internet
traffic activity on the Telecom Italia Milan grid.

The package is deliberately layered so that no analysis or modelling code ever touches
the 21 GB of raw text: `ingest` is the only module that reads `../Dataverse/`, and
everything downstream reads the compact, memory-mappable store it produces.
"""

__version__ = "0.1.0"

#: Every module and subpackage, alphabetically. Kept complete so that
#: ``from milan_traffic import *`` exposes the modelling and evaluation half of the
#: project and not only the data half.
__all__ = [
    "config",
    "dataio",
    "evaluate",
    "features",
    "ingest",
    "metrics",
    "models",
    "periods",
    "significance",
    "training",
    "tsa",
    "utils",
    "viz",
]
