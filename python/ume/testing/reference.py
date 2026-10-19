"""A reference implementation of the universal model in plain numpy.

The test oracle. It follows the CSS post's ``fit_rows`` and ``complier_mean``
line by line, with explicit $M_1$ and $M_2$ matrices, and shares no code with
the Rust engine: the engine must match it to rounding error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReferenceFit:
    """Coefficients $\\hat\\beta$ ($2d \\times o$) and the CR0 covariance of $\\text{vec}(\\hat\\beta)$."""

    beta: NDArray[np.float64]
    cov: NDArray[np.float64]


def fit_rows(
    r: NDArray[np.float64],
    z: NDArray[np.float64],
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    cluster: NDArray[np.integer],
) -> ReferenceFit:
    """Weighted 2SLS with cluster-robust (CR0) covariance, from the formulas.

    $\\hat\\beta = (M_1^\\top W M_2)^{-1} M_1^\\top W Y$ with $M_1 = (R, ZR)$ and
    $M_2 = (R, TR)$; the covariance is
    $(I_o \\otimes A^{-1}) [\\sum_a s_a s_a^\\top] (I_o \\otimes A^{-1})^\\top$.
    """
    m1 = np.hstack([r, z[:, None] * r])
    m2 = np.hstack([r, t[:, None] * r])
    a = (m1 * w[:, None]).T @ m2
    beta = np.linalg.solve(a, (m1 * w[:, None]).T @ y)
    eps = y - m2 @ beta
    groups = np.unique(cluster, return_inverse=True)[1]
    o = y.shape[1]
    inner = np.zeros((groups.max() + 1, m1.shape[1] * o))
    for col in range(o):
        block = slice(col * m1.shape[1], (col + 1) * m1.shape[1])
        np.add.at(inner[:, block], groups, m1 * (w * eps[:, col])[:, None])
    bread = np.kron(np.eye(o), np.linalg.inv(a))
    return ReferenceFit(beta, bread @ inner.T @ inner @ bread.T)


def complier_mean(
    x: NDArray[np.float64], z: NDArray[np.float64], t: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The compliers' mean of the covariates `x` (rows × k), as in the CSS post.

    Without encouragement ($T = Z$), the weighted mean of `x`. Otherwise the Wald
    ratio of weighted arm means of $T x$ over those of $T$.
    """
    if np.array_equal(t, z):
        mean: NDArray[np.float64] = (w[:, None] * x).sum(0) / w.sum()
        return mean

    def arm(value: float) -> tuple[NDArray[np.float64], float]:
        m = z == value
        return (w[m, None] * t[m, None] * x[m]).sum(0) / w[m].sum(), float((w[m] * t[m]).sum() / w[m].sum())

    (tx1, t1), (tx0, t0) = arm(1.0), arm(0.0)
    wald: NDArray[np.float64] = (tx1 - tx0) / (t1 - t0)
    return wald


def inverse_propensity_weights(
    z: NDArray[np.float64], cohort: NDArray[np.integer], cluster: NDArray[np.integer]
) -> NDArray[np.float64]:
    """Row weights $1/\\hat e$ or $1/(1-\\hat e)$, with $\\hat e$ the share of a cohort's *clusters* with $Z = 1$."""
    _, first = np.unique(cluster, return_index=True)
    e = {c: z[first][cohort[first] == c].mean() for c in np.unique(cohort)}
    e_row = np.array([e[c] for c in cohort])
    return np.where(z == 1, 1 / e_row, 1 / (1 - e_row))


def draw_rows(
    r: NDArray[np.float64],
    z: NDArray[np.float64],
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    cluster: NDArray[np.integer],
    cohort: NDArray[np.integer] | None,
    xi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One bootstrap draw on the rows, from the formulas (BUM post, ``draw_rows``).

    `xi` holds one weight per cluster, indexed by the cluster codes in `cluster`.
    The draw recomputes each cohort's propensity as the `xi`-weighted share of its
    clusters with $Z = 1$, refits the model with row weights $\\xi_a w_a(\\xi)$, and
    recomputes the compliers' mean. Returns $\\hat\\beta^{(b)}$ and $\\bar x_C^{(b)}$
    (centered).
    """
    groups = np.unique(cluster, return_inverse=True)[1]
    if cohort is None:
        w = xi[groups]
    else:
        _, first = np.unique(groups, return_index=True)
        zc, cc = z[first], cohort[first]
        e = {c: (xi * zc)[cc == c].sum() / xi[cc == c].sum() for c in np.unique(cc)}
        e_row = np.array([e[c] for c in cohort])
        w = xi[groups] * np.where(z == 1, 1 / e_row, 1 / (1 - e_row))
    m1 = np.hstack([r, z[:, None] * r])
    m2 = np.hstack([r, t[:, None] * r])
    beta = np.linalg.solve((m1 * w[:, None]).T @ m2, (m1 * w[:, None]).T @ y)
    return beta, complier_mean(r[:, 1:], z, t, w)
