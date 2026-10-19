"""The fitted universal model, its regression report, and its effects."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ume._design import Design, DesignFeatures
from ume._draws import Draws
from ume._effects import AnyEffect, default_effects, point, value_and_gradient
from ume._table import Table, format_value

if TYPE_CHECKING:
    from ume._compress import CompressedData

#: Two-sided 95% normal quantile.
Z_975 = 1.959963984540054


def _p_value(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


class FitSource(Protocol):
    """What a fit needs from its data: rows (``ModelData``) or compressed data (``CompressedData``)."""

    @property
    def design(self) -> Design:
        """The resolved design."""
        ...

    @property
    def n(self) -> int:
        """Number of rows."""
        ...

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        ...

    def features(self) -> DesignFeatures:
        """The special case of the universal model the data is."""
        ...

    def fit_raw(self) -> dict[str, Any]:
        """The Rust engine's output for this data."""
        ...

    def itt(self) -> FitSource:
        """The same data for the intent to treat model ($T := Z$)."""
        ...

    def compressed(self) -> CompressedData:
        """The data as compressed rows (itself, or compressed on demand)."""
        ...


class UniversalFit:
    """The universal model fitted to an experiment: weighted 2SLS with every covariate
    interacted with treatment, cluster-robust covariance and several outcome columns.

    Made by :meth:`ume.Experiment.fit` (from rows) or :meth:`ume.CompressedData.fit`
    (from compressed data); both give the same fit. Printing it shows a 2SLS
    regression report.

    Attributes
    ----------
    beta : numpy.ndarray
        Coefficients $\\hat\\beta$, $2d \\times o$: rows are the regressors
        $(1, x, T, Tx)$, columns the outcome columns.
    cov : numpy.ndarray
        Cluster-robust covariance of $\\text{vec}(\\hat\\beta)$ (column-major), $2do \\times 2do$.
    features : DesignFeatures
        Which special case of the model the experiment is.
    """

    def __init__(self, raw: Mapping[str, Any], source: FitSource) -> None:
        self._raw = raw
        self.source = source
        self.design = source.design
        self.features = source.features()
        self.beta: NDArray[np.float64] = raw["second"]["coef"]
        self.cov: NDArray[np.float64] = raw["second"]["cov"]
        self.a: NDArray[np.float64] = raw["second"]["a"]
        self._compliers: Mapping[str, Any] = raw["compliers"]

    # ---- labels -------------------------------------------------------------------------------

    @property
    def x_labels(self) -> list[str]:
        """Labels of the $k$ covariate columns."""
        return self.design.covariates.labels()

    @property
    def regressor_labels(self) -> list[str]:
        """Labels of the $2d$ regressors: ``Intercept``, covariates, ``T``, ``T:covariate``."""
        base = ["Intercept", *self.x_labels]
        return base + ["T"] + [f"T:{x}" for x in self.x_labels]

    @property
    def outcome_labels(self) -> list[str]:
        """Labels of the $o$ outcome columns."""
        return self.design.layout.labels

    # ---- compliers ----------------------------------------------------------------------------

    @property
    def x_ref(self) -> NDArray[np.float64]:
        """The reference point $x_{ref}$ of the $k$ covariate columns."""
        return np.asarray(self.design.covariates.x_ref())

    @property
    def x_C(self) -> NDArray[np.float64]:
        """The compliers' mean of the covariates, $\\bar x_C$ (original units)."""
        return np.asarray(self._compliers["x_c"]) + self.x_ref

    @property
    def pi_C(self) -> float:
        """Compliance rate $\\pi_C$: the share of rows whose treatment the instrument changes."""
        return float(self._compliers["pi_c"])

    def compliers(self) -> Table:
        """Who the compliers are: take-up by arm, compliance rate, and $\\bar x_C$ against the mean.

        Returns
        -------
        Table
            Shares are of rows, weighted by the inverse propensity weights.
        """
        c = self._compliers
        x_mean = np.asarray(c["x_mean"]) + self.x_ref
        rows: list[list[Any]] = [
            ["take-up, Z = 1", float(c["take_up_z1"]), None],
            ["take-up, Z = 0 (always-takers)", float(c["take_up_z0"]), None],
            ["compliance rate π_C", self.pi_C, None],
            ["never-takers (1 − take-up, Z = 1)", 1 - float(c["take_up_z1"]), None],
        ]
        rows += [
            [f"mean {label}", float(xc), float(xm)]
            for label, xc, xm in zip(self.x_labels, self.x_C, x_mean, strict=True)
        ]
        title = "Compliers (shares of rows, weighted)"
        return Table(["quantity", "compliers", "all rows"], rows, title=title)

    # ---- coefficients -------------------------------------------------------------------------

    def coef(self, outcome: str | None = None) -> Table:
        """Coefficient table of one outcome column: estimate, standard error, z, p, 95% interval.

        Parameters
        ----------
        outcome : str, optional
            An outcome label (see :attr:`outcome_labels`); defaults to the first.
        """
        label = outcome or self.outcome_labels[0]
        if label not in self.outcome_labels:
            raise ValueError(f"unknown outcome {label!r}; choose from {self.outcome_labels}")
        col = self.outcome_labels.index(label)
        p = self.beta.shape[0]
        se = np.sqrt(np.diag(self.cov)[col * p : (col + 1) * p])
        rows = []
        for name, b, s in zip(self.regressor_labels, self.beta[:, col], se, strict=True):
            z = b / s if s > 0 else float("nan")
            rows.append([name, float(b), float(s), float(z), _p_value(z), b - Z_975 * s, b + Z_975 * s])
        return Table(["", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"], rows, title=f"Outcome: {label}")

    def first_stage(self) -> Table | None:
        """First-stage strength: the cluster-robust Wald F of the excluded instruments
        $(Z, Zx)$ in each first-stage equation. ``None`` without encouragement."""
        first = self._raw["first"]
        if first is None:
            return None
        d = self.beta.shape[0] // 2
        names = ["T"] + [f"T:{x}" for x in self.x_labels]
        rows = [[n, float(f), d, self.source.n_clusters - 1] for n, f in zip(names, first["f"], strict=True)]
        return Table(["endogenous", "robust F", "df1", "df2"], rows, title="First stage (excluded instruments Z, Z·x)")

    # ---- effects ------------------------------------------------------------------------------

    def effects(
        self, *effects: AnyEffect, at: Mapping[str, Any] | None = None, se: str | Draws = "sandwich"
    ) -> EffectTable:
        """Estimate effects, with standard errors.

        Parameters
        ----------
        *effects : Effect or CustomEffect
            The effects. By default, the difference and the relative effect of every metric.
        at : mapping, optional
            Read every effect at a segment instead of at the compliers' mean.
        se : "sandwich", "linearized" or Draws
            ``"sandwich"``: the delta method on the cluster-robust covariance,
            holding $\\hat e$ and $\\bar x_C$ fixed (decision 0012).
            ``"linearized"``: the first-order expansion of the bootstrap in the
            per-stratum sums, which also moves $\\hat e$ and $\\bar x_C$
            (decision 0020); it needs the compressed rows, and compresses the
            rows on demand. A :class:`~ume.Draws` object: the standard deviation
            of each effect over the draws.

        Returns
        -------
        EffectTable
        """
        if isinstance(se, str) and se not in ("sandwich", "linearized"):
            raise ValueError("se must be 'sandwich', 'linearized' or a Draws object")
        draws = None if isinstance(se, str) else se
        rows = self._effect_rows(effects, at, "compliers", draws, linearized=se == "linearized")
        if draws is not None:
            method = draws.label
        elif se == "linearized":
            method = "linearized (moves e_hat and x_C)"
        else:
            method = "sandwich (" + self.features.covariance + ")"
        return EffectTable(rows, self, self.features.estimand, method)

    def _effect_rows(
        self,
        effects: Sequence[AnyEffect],
        at: Mapping[str, Any] | None,
        population: str,
        draws: Draws | None = None,
        linearized: bool = False,
    ) -> list[list[Any]]:
        chosen: Sequence[AnyEffect] = effects or default_effects(self.design.metrics)
        base = np.asarray(self._compliers["x_c"])
        lin_sd = self._linearized_sd(chosen, at) if linearized else None
        rows = []
        for i, e in enumerate(chosen):
            where = e.at if e.at is not None else at
            x0 = point(self.design.covariates, base, where)
            value, grad = value_and_gradient(e, self.beta, x0, self.design.layout)
            if lin_sd is not None:
                std = float(lin_sd[i])
            elif draws is None:
                std = float(np.sqrt(grad @ self.cov @ grad))
            else:
                std = float(np.std(draws.values(e, where), ddof=1))
            label = population if not where else ", ".join(f"{k}={v}" for k, v in where.items())
            rows.append([e.name, e.contrast, label, value, std, value - Z_975 * std, value + Z_975 * std])
        return rows

    def _linearized_sd(self, effects: Sequence[AnyEffect], at: Mapping[str, Any] | None) -> NDArray[np.float64]:
        """The linearized standard error of each effect, from the compressed rows."""
        from ume._linearize import linearized_variance, stratum_gradients

        cd = self.source.compressed()
        sums = np.concatenate([cd.g[:, None], cd.first], axis=1)
        z, cohort, n_cohorts = cd._cohorts()
        grads = stratum_gradients(effects, cd.design, sums, z, cohort, n_cohorts, self.features.encouraged, at)
        return np.sqrt(linearized_variance(grads, cd.g, cd.first, cd.second))

    def itt(self, *effects: AnyEffect) -> EffectTable:
        """Intent to treat effects: the effect of the assignment $Z$ itself.

        The ITT model is the universal model with the treatment replaced by the
        instrument, $T := Z$, which makes every row a complier. Its effects are
        the covariate-adjusted ITT of each metric, read at the weighted mean of
        the covariates.
        """
        source = self.source.itt()
        itt_fit = UniversalFit(source.fit_raw(), source)
        rows = itt_fit._effect_rows(effects, None, "all rows")
        return EffectTable(rows, self, "ITT", "sandwich (" + self.features.covariance + ")")

    # ---- printing -----------------------------------------------------------------------------

    def summary(self, outcomes: Sequence[str] | None = None) -> str:
        """The regression report: header, first stage, and one coefficient table per outcome.

        Parameters
        ----------
        outcomes : sequence of str, optional
            Outcome labels to show. Defaults to the metrics' own columns (not the
            ``T·`` columns).
        """
        f = self.features
        shown = outcomes or [c.label for c in self.design.layout.columns if not c.treated]
        width = 78
        weights = f"inverse propensity by {self.design.cohort}" if f.weighted else "none (all 1)"
        cov = f"{f.covariance}, clustered by {self.design.cluster}" if f.clustered else "HC0 (one row per cluster)"
        k = f.k
        header = [
            "Universal model: weighted 2SLS".center(width),
            "=" * width,
            _pair("Design:", f.describe()),
            _pair("Outcomes:", ", ".join(self.outcome_labels)),
            _pair("Instruments:", f"1, x, Z, Z·x   (2d = {2 * (k + 1)})", "Rows:", format_value(self.source.n)),
            _pair(
                "Regressors:", f"1, x, T, T·x   (2d = {2 * (k + 1)})", "Clusters:", format_value(self.source.n_clusters)
            ),
            _pair("Weights:", weights, "Covariance:", cov),
            _pair("Estimand:", f.estimand, "Compliance:", f"{self.pi_C:.4f}"),
        ]
        parts = ["\n".join(header)]
        first = self.first_stage()
        parts.append(repr(first) if first is not None else "First stage: none (T = Z, every row complies)")
        parts += [repr(self.coef(label)) for label in shown]
        hidden = [label for label in self.outcome_labels if label not in shown]
        footer = "=" * width
        if hidden:
            footer += f"\nMore outcomes: {', '.join(hidden)}  (fit.coef(label))"
        parts.append(footer)
        return ("\n" + "-" * width + "\n").join(parts[:-1]) + "\n" + parts[-1]

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(self.summary())}</pre>"


def _pair(label: str, value: str, label2: str = "", value2: str = "") -> str:
    left = f"{label:<13}{value}"
    if not label2:
        return left
    return f"{left:<46}{label2:<13}{value2}"


class EffectTable(Table):
    """Effects with standard errors and 95% intervals, with the compliers' summary on top."""

    def __init__(self, rows: list[list[Any]], fit: UniversalFit, estimand: str, method: str) -> None:
        f = fit.features
        lines = [f"Effects: {estimand}   se: {method}", f"  design: {f.describe()}"]
        if estimand == "ITT":
            lines.append("  the effect of the assignment Z, over all rows")
        else:
            x_c = ", ".join(
                f"{label} = {format_value(float(v))}" for label, v in zip(fit.x_labels, fit.x_C, strict=True)
            )
            lines.append(f"  compliance rate π_C = {fit.pi_C:.4f}")
            if x_c:
                lines.append(f"  compliers' mean x̄_C: {x_c}")
        super().__init__(
            ["metric", "contrast", "population", "estimate", "std err", "[0.025", "0.975]"],
            rows,
            title="\n".join(lines),
        )
        self.estimand = estimand

    def estimate(self, metric: str, contrast: str = "difference") -> float:
        """The estimate of one effect."""
        return float(self._find(metric, contrast)[3])

    def std_err(self, metric: str, contrast: str = "difference") -> float:
        """The standard error of one effect."""
        return float(self._find(metric, contrast)[4])

    def _find(self, metric: str, contrast: str) -> list[Any]:
        for r in self.rows:
            if r[0] == metric and r[1] == contrast:
                return r
        raise KeyError((metric, contrast))
