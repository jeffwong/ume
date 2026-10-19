"""The linearized standard error: the bootstrap's first-order expansion.

A bootstrap draw of an effect is a smooth function of the per-stratum weighted
sums, $\\tau(\\theta)$ with $\\theta_h = \\sum_{a \\in h} \\xi_a v_a$ and $v_a = (1, u_a)$. At
$\\xi = 1$ the sums are $\\hat\\theta_h = (G_h, \\sum u_a)$. To first order,

$$\\tau(\\theta(\\xi)) - \\hat\\tau \\approx \\sum_h g_h^\\top (\\theta_h(\\xi) - \\hat\\theta_h)
= \\sum_a (\\xi_a - 1)\\, g_{h(a)}^\\top v_a, \\qquad g_h = \\partial\\tau / \\partial\\theta_h,$$

so with independent weights of mean 1 and variance 1,

$$\\text{Var} \\approx \\sum_a \\big(g_{h(a)}^\\top v_a\\big)^2
= \\sum_h g_h^\\top \\Big(\\sum_{a \\in h} v_a v_a^\\top\\Big) g_h.$$

Unlike the sandwich, $\\tau(\\theta)$ recomputes the propensities and $\\bar x_C$ from
$\\theta$, so this standard error includes their uncertainty (handoff, section 8).
The gradient $g_h$ is taken by central differences through the engine's own
per-draw fit (``fit_draws``): there is no second implementation of the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ume import _core
from ume._config import get_threads
from ume._design import Design
from ume._draws import Draws
from ume._effects import AnyEffect

#: Relative step of the central differences.
STEP = 1e-5


def stratum_gradients(
    effects: Sequence[AnyEffect],
    design: Design,
    sums: NDArray[np.float64],
    z: NDArray[np.float64],
    cohort: NDArray[np.uint32],
    n_cohorts: int,
    encouraged: bool,
    at: Mapping[str, Any] | None = None,
) -> NDArray[np.float64]:
    """$g_h = \\partial \\tau / \\partial \\theta_h$ of each effect, shape ``(E, H, 1 + q)``.

    Parameters
    ----------
    effects : sequence of Effect or CustomEffect
        The effects, each read at its own population (compliers or segment).
    design : Design
        The resolved design.
    sums : numpy.ndarray
        $\\hat\\theta_h = (G_h, \\sum u_a)$ per stratum, shape ``(H, 1 + q)``.
    z, cohort, n_cohorts, encouraged
        The strata's instrument and cohort index, the number of cohorts, and
        whether the design has encouragement.
    at : mapping, optional
        A segment for effects that do not set their own.
    """
    h, m = sums.shape
    active = [(i, k) for i in range(h) for k in range(m)]
    steps = np.array([STEP * max(abs(sums[i, k]), 1.0) for i, k in active])
    draws = np.repeat(sums[:, None, :], 2 * len(active), axis=1)
    for j, ((i, k), s) in enumerate(zip(active, steps, strict=True)):
        draws[i, 2 * j, k] += s
        draws[i, 2 * j + 1, k] -= s
    d, o = design.covariates.k + 1, design.layout.o
    weighted = design.cohort is not None
    raw = _core.fit_draws(d, o, np.ascontiguousarray(draws), z, cohort, n_cohorts, weighted, encouraged, get_threads())
    if not np.all(raw["ok"]):
        raise ValueError("the linearized standard error is undefined: the model is degenerate near the estimate")
    perturbed = Draws(raw, design, "linearization", "perturbation", 0)
    out = np.zeros((len(effects), h, m))
    for e, effect in enumerate(effects):
        v = perturbed.values(effect, at)
        for j, (i, k) in enumerate(active):
            out[e, i, k] = (v[2 * j] - v[2 * j + 1]) / (2 * steps[j])
    return out


def linearized_variance(
    gradients: NDArray[np.float64], g: NDArray[np.float64], first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """$\\sum_h g_h^\\top (\\sum_{a \\in h} v_a v_a^\\top) g_h$ for each effect, from compressed rows.

    `gradients` has shape ``(E, H, 1 + q)``; `g`, `first`, `second` are the
    compressed rows ($G_h$, $\\sum u_a$, $\\sum u_a u_a^\\top$).
    """
    out = np.zeros(gradients.shape[0])
    for i in range(g.size):
        vv = np.block([[np.array([[g[i]]]), first[i][None, :]], [first[i][:, None], second[i]]])
        out += np.einsum("ej,jk,ek->e", gradients[:, i, :], vv, gradients[:, i, :])
    return out
