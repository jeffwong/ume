"""Draws of the universal model: bootstrap replicates and posterior draws.

Every engine produces the same object. Each draw $b$ is the universal model
refit with one weight per cluster, $\\xi^{(b)}_a$, recomputing the propensities,
$\\hat\\beta$ and $\\bar x_C$. An effect's draws are the effect's function applied to
each draw, so any function of the effects has draws too.

=====================  ======================  ===================================
Made by                Weights                 Distribution
=====================  ======================  ===================================
``exp.bootstrap()``    ``"bayesian"`` Exp(1)   posterior (Bayesian bootstrap)
``exp.bootstrap()``    ``"poisson"``           sampling (Poisson bootstrap)
``exp.bootstrap()``    ``"multinomial"``       sampling (classical bootstrap)
``cd.posterior()``     $N(1, 1)$ multiplier    posterior, from the compressed rows
=====================  ======================  ===================================
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ume._design import Design
from ume._effects import AnyEffect, Levels, default_effects, point
from ume._table import Table


class Draws:
    """$B$ draws of the universal model.

    Made by :meth:`ume.Experiment.bootstrap`, :func:`ume.bootstrap`,
    :meth:`ume.CompressedData.posterior` and :meth:`ume.DrawSums.draws`, rather
    than built directly.

    Parameters
    ----------
    raw : mapping
        The engine's output (``ume._core.fit_draws``): ``beta``, ``x_c``,
        ``pi_c``, ``e_hat`` and ``ok`` (False for degenerate draws, which are dropped).
    design : Design
        The resolved design, which labels the coefficients and reads the effects.
    engine : str
        ``"exact"`` or ``"gaussian"``, for display.
    weights : str
        The weight scheme, for display.
    seed : int
        The seed that produced the draws.

    Attributes
    ----------
    beta : numpy.ndarray
        $\\hat\\beta^{(b)}$ of each draw, shape ``(B, 2d, o)``.
    x_C : numpy.ndarray
        The compliers' mean of each draw (original units), shape ``(B, k)``.
    pi_C : numpy.ndarray
        The compliance rate of each draw.
    e_hat : numpy.ndarray
        The propensities of each draw, shape ``(B, cohorts)``.
    engine : str
        ``"exact"`` (one pass over the rows) or ``"gaussian"`` (from compressed rows).
    weights : str
        The weight scheme.
    n_failed : int
        Draws dropped because their model was degenerate (for example, a Poisson
        draw that left a cohort with weight in one arm only).
    """

    def __init__(self, raw: Mapping[str, Any], design: Design, engine: str, weights: str, seed: int) -> None:
        ok = np.asarray(raw["ok"], dtype=bool)
        self.design = design
        self.engine = engine
        self.weights = weights
        self.seed = seed
        self.n_failed = int((~ok).sum())
        self.beta: NDArray[np.float64] = np.asarray(raw["beta"])[ok]
        self._x_c: NDArray[np.float64] = np.asarray(raw["x_c"])[ok]
        self.pi_C: NDArray[np.float64] = np.asarray(raw["pi_c"])[ok]
        self.e_hat: NDArray[np.float64] = np.asarray(raw["e_hat"])[ok]

    @property
    def B(self) -> int:
        """Number of draws kept."""
        return int(self.beta.shape[0])

    @property
    def x_C(self) -> NDArray[np.float64]:
        """The compliers' mean of each draw, in original units."""
        return self._x_c + np.asarray(self.design.covariates.x_ref())

    @property
    def label(self) -> str:
        """A short description, such as ``"bayesian bootstrap (exact, B = 1000)"``."""
        kind = "gaussian posterior" if self.engine == "gaussian" else f"{self.weights} bootstrap"
        return f"{kind} ({self.engine}, B = {self.B:,})"

    def values(self, effect: AnyEffect, at: Mapping[str, Any] | None = None) -> NDArray[np.float64]:
        """The draws of one effect, shape ``(B,)``.

        Each draw is read at its own compliers' mean (or at the segment `at`).

        Parameters
        ----------
        effect : Effect or CustomEffect
            The effect.
        at : mapping, optional
            A segment (covariate values) for an effect that does not set its own.

        Returns
        -------
        numpy.ndarray
        """
        where = effect.at if effect.at is not None else at
        x0 = point(self.design.covariates, self._x_c, where)
        return np.asarray(effect(Levels(self.beta, x0, self.design.layout)), dtype=np.float64)

    def prob(self, effect: AnyEffect, *, below: float | None = None, above: float | None = None) -> float:
        """The share of draws with the effect below and/or above thresholds.

        For posterior draws this is a posterior probability, such as
        $P(\\rho_R < -10\\%)$.

        Parameters
        ----------
        effect : Effect or CustomEffect
            The effect.
        below : float, optional
            Count draws whose effect is below this value.
        above : float, optional
            Count draws whose effect is above this value. With both, the draws
            strictly between the two.

        Returns
        -------
        float
        """
        v = self.values(effect)
        mask = np.ones(v.size, dtype=bool)
        if below is not None:
            mask &= v < below
        if above is not None:
            mask &= v > above
        return float(mask.mean())

    def apply(self, fn: Callable[[NDArray[np.float64]], NDArray[np.float64]], effect: AnyEffect) -> NDArray[np.float64]:
        """Draws of any function of an effect: ``fn`` applied to each draw.

        Parameters
        ----------
        fn : callable
            A vectorized function of an array of effect draws, such as ``np.exp``.
        effect : Effect or CustomEffect
            The effect.

        Returns
        -------
        numpy.ndarray
        """
        return np.asarray(fn(self.values(effect)))

    def effects(self, *effects: AnyEffect, at: Mapping[str, Any] | None = None) -> Table:
        """Summaries of each effect's draws: mean, standard deviation, quantiles.

        Parameters
        ----------
        *effects : Effect or CustomEffect
            Defaults to the difference and the relative effect of every metric.
        at : mapping, optional
            Read every effect at a segment.
        """
        chosen: Sequence[AnyEffect] = effects or default_effects(self.design.metrics)
        rows = []
        for e in chosen:
            v = self.values(e, at)
            q = np.quantile(v, [0.025, 0.5, 0.975])
            rows.append([e.name, e.contrast, float(v.mean()), float(v.std(ddof=1)), *map(float, q)])
        footer = f"{self.n_failed} degenerate draws dropped" if self.n_failed else None
        return Table(
            ["metric", "contrast", "mean", "sd", "2.5%", "50%", "97.5%"], rows, title=f"Draws: {self.label}",
            footer=footer,
        )  # fmt: skip

    def __repr__(self) -> str:
        return repr(self.effects())

    def _repr_html_(self) -> str:
        return self.effects()._repr_html_()
