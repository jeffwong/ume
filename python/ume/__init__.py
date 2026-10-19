"""Universal Model for Experimentation.

One model, weighted two stage least squares with every covariate interacted
with treatment and cluster-robust covariance, covers AB tests, cluster
randomized tests, encouragement designs, adaptive assignment, and mean or ratio
metrics as special cases. The analysis is a short grammar:

1. :func:`annotate` a frame with the roles of its columns;
2. declare the :func:`mean` and :func:`ratio` metrics;
3. :meth:`~Experiment.fit` the universal model, on the rows, or
   :meth:`~Experiment.compress` them first and fit the compressed data;
4. read the :meth:`~UniversalFit.effects`, with sandwich standard errors or with
   :class:`Draws` from :meth:`~Experiment.bootstrap` or :meth:`~CompressedData.posterior`.

The public API is Python; the numeric engine is Rust (``ume._core``).
"""

from ume._compress import (
    CompressedData,
    CompressedRow,
    Compressor,
    Stratum,
    bootstrap,
    bootstrap_sums,
    compress,
    effects,
    load,
)
from ume._config import get_threads, set_threads
from ume._core import __version__
from ume._design import Design, DesignFeatures
from ume._draws import Draws
from ume._drawsums import DrawSums
from ume._effects import CustomEffect, Effect, Levels, effect
from ume._experiment import Experiment, annotate, design
from ume._fit import EffectTable, UniversalFit
from ume._metrics import Mean, Ratio, mean, ratio
from ume._roles import Categorical, Covariates, Numeric, categorical, covariates
from ume._table import Table

__all__ = [
    "Categorical",
    "CompressedData",
    "CompressedRow",
    "Compressor",
    "Covariates",
    "CustomEffect",
    "Design",
    "DesignFeatures",
    "DrawSums",
    "Draws",
    "Effect",
    "EffectTable",
    "Experiment",
    "Levels",
    "Mean",
    "Numeric",
    "Ratio",
    "Stratum",
    "Table",
    "UniversalFit",
    "__version__",
    "annotate",
    "bootstrap",
    "bootstrap_sums",
    "categorical",
    "compress",
    "covariates",
    "design",
    "effect",
    "effects",
    "get_threads",
    "load",
    "mean",
    "ratio",
    "set_threads",
]
