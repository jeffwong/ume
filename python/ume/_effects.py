"""Effects: a grammar of three choices.

Every effect of the universal model is **contrast ∘ statistic ∘ population**.

- **population**: where the lines are read, at the compliers' mean $\\bar x_C$
  (the default) or at a segment $x_0$;
- **statistic**: the metric $m_t = \\mu_t(N) / \\mu_t(D)$, where a mean metric has
  $D \\equiv 1$;
- **contrast**: ``"difference"`` $m_1 - m_0$, or ``"relative"`` $m_1 / m_0 - 1$.

Here $\\mu_t(Y; x_0)$ is the compliers' level of column $Y$ under treatment $t$,
read on the fitted lines:

$$\\mu_1(Y) = \\delta^\\top\\hat\\beta_{T\\cdot Y}, \\qquad
\\mu_0(Y) = \\delta^\\top(\\hat\\beta_{T\\cdot Y} - \\hat\\beta_Y), \\qquad
\\delta = (0_d, 1, x_0 - x_{ref}).$$

The posts' effects are special cases: mean and difference at $\\bar x_C$ is
$\\tau_{LATE} = \\delta^\\top\\hat\\beta_y$; ratio and difference is $\\tau_R$; ratio
and relative is $\\rho_R$. Without encouragement every row is a complier, and
the same effects are average treatment effects.

Standard errors use the delta method, with gradients from complex-step
differentiation, which is exact to machine precision: functions of
:class:`Levels` must use only operations that accept complex numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ume._metrics import Metric, OutcomeLayout
from ume._roles import Categorical, Covariates, Numeric

Contrast = Literal["difference", "relative"]
#: Step of the complex-step derivative.
KAPPA = 1e-20


class Levels:
    """The compliers' levels of each outcome column, read on the lines at one point.

    Everything broadcasts over leading axes, so the same effect function
    evaluates one estimate, a complex-step gradient, or $B$ bootstrap draws.

    Parameters
    ----------
    beta : numpy.ndarray
        Coefficients $\\hat\\beta$, shape ``(..., 2d, o)``, real or complex.
    x0 : numpy.ndarray
        The point in centered units $x_0 - x_{ref}$, shape ``(..., k)``.
    layout : OutcomeLayout
        Which column of $\\hat\\beta$ belongs to which outcome.
    """

    def __init__(self, beta: NDArray[Any], x0: NDArray[np.float64], layout: OutcomeLayout) -> None:
        d = beta.shape[-2] // 2
        lead = np.broadcast_shapes(beta.shape[:-2], x0.shape[:-1])
        x0 = np.broadcast_to(x0, (*lead, x0.shape[-1]))
        self.delta = np.concatenate([np.zeros((*lead, d)), np.ones((*lead, 1)), x0], axis=-1)
        self.beta = beta
        self.layout = layout

    def line(self, column: str, treated: bool = False) -> Any:
        """The treatment line of `column` (or $T \\cdot$ `column`) at the point, $\\delta^\\top\\hat\\beta$."""
        return np.sum(self.delta * self.beta[..., :, self.layout.index(column, treated)], axis=-1)

    def mu1(self, column: str | None) -> Any:
        """$\\mu_1$: the compliers' mean of `column` with treatment. 1 for the constant (`None`)."""
        return 1.0 if column is None else self.line(column, treated=True)

    def mu0(self, column: str | None) -> Any:
        """$\\mu_0$: the compliers' mean of `column` without treatment. 1 for the constant (`None`)."""
        return 1.0 if column is None else self.line(column, treated=True) - self.line(column)


@dataclass(frozen=True)
class Effect:
    """One effect: a metric, a contrast and a population.

    Parameters
    ----------
    metric : Mean or Ratio
        The statistic.
    contrast : {"difference", "relative"}
        $m_1 - m_0$ or $m_1 / m_0 - 1$.
    at : mapping, optional
        A segment: covariate values to read the lines at. Covariates not listed
        stay at the compliers' mean. ``None`` means the compliers.
    """

    metric: Metric
    contrast: Contrast = "difference"
    at: Mapping[str, Any] | None = field(default=None, hash=False)

    @property
    def name(self) -> str:
        """The metric's name."""
        return self.metric.name

    def statistic(self, levels: Levels, t: int) -> Any:
        """$m_t = \\mu_t(N) / \\mu_t(D)$."""
        mu = levels.mu1 if t == 1 else levels.mu0
        return mu(self.metric.numerator) / mu(self.metric.denominator)

    def __call__(self, levels: Levels) -> Any:
        m1, m0 = self.statistic(levels, 1), self.statistic(levels, 0)
        return m1 - m0 if self.contrast == "difference" else m1 / m0 - 1


@dataclass(frozen=True)
class CustomEffect:
    """A user-defined effect: any function of :class:`Levels`.

    Parameters
    ----------
    name : str
        Display name.
    fn : callable
        ``fn(levels) -> value``. Use only operations that accept complex numbers
        (arithmetic, numpy functions), so that complex-step gradients work.
    at : mapping, optional
        A segment, as in :class:`Effect`.
    """

    name: str
    fn: Callable[[Levels], Any]
    at: Mapping[str, Any] | None = field(default=None, hash=False)
    contrast: str = "custom"

    def __call__(self, levels: Levels) -> Any:
        return self.fn(levels)


AnyEffect = Effect | CustomEffect


def effect(name: str, fn: Callable[[Levels], Any], at: Mapping[str, Any] | None = None) -> CustomEffect:
    """Define a custom effect from the compliers' levels.

    Parameters
    ----------
    name : str
        Display name.
    fn : callable
        ``fn(levels) -> value``, where ``levels.mu1(col)`` and ``levels.mu0(col)``
        are the compliers' means of column `col` with and without treatment.
    at : mapping, optional
        A segment to read the lines at.

    Returns
    -------
    CustomEffect

    Examples
    --------
    >>> import ume
    >>> plays_per_hour = ume.effect(
    ...     "plays_per_hour", lambda m: m.mu1("D") / m.mu1("y") - m.mu0("D") / m.mu0("y"))
    """
    return CustomEffect(name, fn, at)


def point(covariates: Covariates, base: NDArray[np.float64], at: Mapping[str, Any] | None) -> NDArray[np.float64]:
    """The centered point $x_0 - x_{ref}$: `base` with the covariates in `at` replaced.

    Numeric covariates take ``value - x_ref``. A categorical covariate takes the
    indicator of the given level (all zeros for the reference level). `base` may
    have leading axes (one point per draw); the replacements apply to every one.
    """
    x0 = np.array(base, dtype=np.float64)
    if not at:
        return x0
    unknown = set(at) - set(covariates.names)
    if unknown:
        raise ValueError(f"unknown covariates in `at`: {sorted(unknown)}")
    j = 0
    for name, spec in covariates.specs:
        if isinstance(spec, Numeric):
            if name in at:
                assert spec.x_ref is not None
                x0[..., j] = float(at[name]) - spec.x_ref
            j += 1
        else:
            assert isinstance(spec, Categorical)
            levels = spec.indicator_levels()
            if name in at:
                if at[name] != spec.reference and at[name] not in levels:
                    raise ValueError(f"{at[name]!r} is not a level of {name!r}")
                x0[..., j : j + len(levels)] = [1.0 if at[name] == level else 0.0 for level in levels]
            j += len(levels)
    return x0


def value_and_gradient(
    e: AnyEffect, beta: NDArray[np.float64], x0: NDArray[np.float64], layout: OutcomeLayout
) -> tuple[float, NDArray[np.float64]]:
    """The effect at $\\hat\\beta$ and its gradient with respect to $\\text{vec}(\\hat\\beta)$, column-major.

    The gradient uses complex steps,
    $\\partial f / \\partial\\theta_j = \\text{Im}\\,f(\\theta + i\\kappa e_j) / \\kappa$.
    """
    value = complex(e(Levels(beta, x0, layout))).real
    flat = beta.reshape(-1, order="F")
    grad = np.empty(flat.size)
    for j in range(flat.size):
        stepped = flat.astype(np.complex128)
        stepped[j] += 1j * KAPPA
        grad[j] = complex(e(Levels(stepped.reshape(beta.shape, order="F"), x0, layout))).imag / KAPPA
    return value, grad


def default_effects(metrics: tuple[Metric, ...]) -> list[Effect]:
    """The difference and the relative effect of every metric."""
    return [Effect(m, c) for m in metrics for c in ("difference", "relative")]
