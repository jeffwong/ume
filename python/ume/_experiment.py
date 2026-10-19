"""An annotated dataframe: the native frame plus its :class:`~ume.Design`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import narwhals as nw

from ume._data import SparseOption, model_data
from ume._design import Design, DesignFeatures
from ume._effects import AnyEffect
from ume._fit import EffectTable, UniversalFit
from ume._metrics import Metric
from ume._roles import Covariates
from ume._source import is_lazy, lazy_columns
from ume._table import Table

if TYPE_CHECKING:
    from ume._compress import CompressedData
    from ume._draws import Draws


class Experiment:
    """A pandas or polars frame of rows, annotated with the roles of its columns.

    An ``Experiment`` behaves like the frame it wraps: attribute access and
    methods pass through to the native frame. A result that is a frame with
    every annotated column is wrapped again with the same design. Anything else
    (aggregates, group-bys, frames that dropped a role column) is returned as is.

    Build with :func:`ume.annotate`.

    Parameters
    ----------
    frame : pandas.DataFrame or polars.DataFrame
        One row per unit of analysis.
    design : Design
        The roles of the columns.
    """

    __slots__ = ("_native", "design")

    def __init__(self, frame: Any, design: Design) -> None:
        native = nw.to_native(nw.from_native(frame))
        missing = [c for c in design.columns if c not in _columns(native)]
        if missing:
            raise ValueError(f"columns not found: {missing}")
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "design", design)

    @property
    def lazy(self) -> bool:
        """Whether the rows are a lazy frame: compression, bootstraps and two-pass
        effects then stream it in batches; :meth:`fit` collects it."""
        return is_lazy(self._native)

    def _eager(self) -> Any:
        """The rows as an eager frame (collects a lazy frame)."""
        return nw.to_native(nw.from_native(self._native).collect()) if self.lazy else self._native

    @property
    def frame(self) -> Any:
        """The native frame, unwrapped."""
        return self._native

    def _wrap(self, obj: Any) -> Any:
        if isinstance(obj, type(self._native)) and all(c in _columns(obj) for c in self.design.columns):
            return Experiment(obj, self.design)
        return obj

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._native, name)
        if not callable(attr):
            return self._wrap(attr)

        def method(*args: Any, **kwargs: Any) -> Any:
            return self._wrap(attr(*args, **kwargs))

        method.__doc__ = attr.__doc__
        method.__name__ = name
        return method

    def __getitem__(self, key: Any) -> Any:
        return self._wrap(self._native[key])

    def __len__(self) -> int:
        return len(self._native)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("an Experiment is immutable; derive a new one with frame operations")

    # ---- the grammar of an analysis ---------------------------------------------------------------

    def metrics(self, *metrics: Metric) -> Experiment:
        """Declare the metrics to analyze, replacing any declared before.

        Parameters
        ----------
        *metrics : Mean or Ratio
            From :func:`ume.mean` and :func:`ume.ratio`.

        Returns
        -------
        Experiment
        """
        return Experiment(self._native, self.design.with_metrics(*metrics))

    @property
    def outcome_columns(self) -> Table:
        """The outcome columns $Y$ the metrics need, including the $T \\cdot Y$ columns."""
        rows = []
        for c in self.design.layout.columns:
            users = [m.name for m in self.design.metrics if c.source in (m.numerator, m.denominator)]
            role = "T × column: compliers' own levels" if c.treated else "metric column"
            rows.append([c.label, ", ".join(users), role])
        return Table(["column", "metrics", "role"], rows, title=f"outcome columns (o = {len(rows)})")

    def features(self) -> DesignFeatures:
        """Which special case of the universal model this experiment is, read from the data."""
        return model_data(self._eager(), self.design).features()

    def compress(
        self, sparse: SparseOption = "auto", *, batch_size: int | None = None, presorted: bool = False
    ) -> CompressedData:
        """Compress the rows into one compressed row per stratum, in one pass.

        Rows are grouped by cluster first if they are not already. The result
        fits to exactly the same model as :meth:`fit`, and its size does not
        depend on the number of rows.

        Parameters
        ----------
        sparse : bool or "auto"
            Build the design matrix dense or sparse during the pass.
        batch_size : int, optional
            Rows per batch when the rows are a lazy frame (default 500,000).
        presorted : bool
            The lazy frame's rows are already grouped by cluster: skip the sort.

        Returns
        -------
        CompressedData
        """
        from ume._compress import compress

        return compress(self._native, self.design, sparse, batch_size=batch_size, presorted=presorted)

    def bootstrap(
        self,
        B: int = 1000,
        weights: str = "bayesian",
        seed: int = 0,
        sparse: SparseOption = "auto",
        *,
        batch_size: int | None = None,
        presorted: bool = False,
    ) -> Draws:
        """Bootstrap the universal model exactly, in one new pass over the rows.

        Each draw refits the model with one weight per cluster, recomputing the
        propensities, $\\hat\\beta$ and $\\bar x_C$. The weights come from Philox
        streams keyed by (seed, cluster, draw), so they do not depend on row order.

        Parameters
        ----------
        B : int
            Number of draws.
        weights : {"bayesian", "poisson", "multinomial"}
            Exp(1) weights give the Bayesian bootstrap (a posterior); Poisson(1)
            the Poisson bootstrap; multinomial counts over the clusters the
            classical bootstrap.
        seed : int
            Seed of the weight streams.
        sparse : bool or "auto"
            Dense or sparse design matrix during the pass.
        batch_size, presorted
            As in :meth:`compress`.

        Returns
        -------
        Draws
        """
        from ume._compress import bootstrap

        return bootstrap(
            self._native, self.design, B, weights, seed, sparse, batch_size=batch_size, presorted=presorted
        )

    def effects(
        self,
        *effects: AnyEffect,
        at: Mapping[str, Any] | None = None,
        se: str = "sandwich",
        sparse: SparseOption = "auto",
        batch_size: int | None = None,
        presorted: bool = False,
    ) -> EffectTable:
        """Effects and standard errors in two passes, without second moments (delta vectors early).

        Pass 1 builds the first moments, enough for the point estimate and for
        each effect's projection $c_h$; pass 2 rebuilds each cluster summary and
        adds $(c_h^\\top v_a)^2$. The storage is $Hq$ numbers and one per effect,
        instead of the $Hq^2$ of the compressed dataset, which matters with many
        covariates. The results equal ``exp.fit().effects(se=se)``; the price is
        that the effects are fixed before the second pass.

        Parameters
        ----------
        *effects : Effect or CustomEffect
            Defaults to the difference and the relative effect of every metric.
        at : mapping, optional
            A segment to read the effects at.
        se : {"sandwich", "linearized"}
            The standard error.
        sparse : bool or "auto"
            Dense or sparse design matrix during the passes.
        batch_size, presorted
            As in :meth:`compress`.

        Returns
        -------
        EffectTable
        """
        from ume._compress import early_effects

        return early_effects(
            self._native, self.design, effects, at, se, sparse, batch_size=batch_size, presorted=presorted
        )

    def fit(self, sparse: SparseOption = "auto") -> UniversalFit:
        """Fit the universal model on the rows.

        Parameters
        ----------
        sparse : bool or "auto"
            Use a sparse design matrix. ``"auto"`` chooses sparse when at most
            30% of the design matrix is nonzero, as with many-level categoricals.

        Returns
        -------
        UniversalFit
        """
        data = model_data(self._eager(), self.design, sparse)
        return UniversalFit(data.fit_raw(), data)

    def __repr__(self) -> str:
        d = self.design
        frame = nw.from_native(self._native)
        backend = nw.get_native_namespace(frame).__name__.split(".")[0]
        rows = "lazy" if self.lazy else f"{len(frame):,} rows"
        lines = [f"<ume.Experiment: {rows} × {len(_columns(self._native))} columns, {backend}>"]
        lines.append(f"  cluster     {d.cluster}")
        if d.treatment == d.instrument:
            lines.append(f"  treatment   {d.treatment}   (randomized; no encouragement)")
        else:
            lines.append(f"  instrument  {d.instrument}   (randomized assignment)")
            lines.append(f"  treatment   {d.treatment}   (received)")
        lines.append(f"  cohort      {d.cohort or '— (no inverse propensity weights)'}")
        covs = ", ".join(_describe(n, s) for n, s in d.covariates.specs) or "none"
        lines.append(f"  covariates  {covs}")
        mets = ", ".join(f"{m.name} ({m.kind})" for m in d.metrics) or "none yet; add with .metrics(...)"
        lines.append(f"  metrics     {mets}")
        return "\n".join(lines)


def _columns(native: Any) -> list[str]:
    """Column names of an eager or lazy frame."""
    return lazy_columns(native) if is_lazy(native) else list(nw.from_native(native, eager_only=True).columns)


def _describe(name: str, spec: Any) -> str:
    from ume._roles import Numeric

    if isinstance(spec, Numeric):
        return name if spec.x_ref is None else f"{name} (x_ref={spec.x_ref:g})"
    return f"{name} (categorical, ref={spec.reference!r})"


def annotate(
    frame: Any,
    *,
    cluster: str,
    instrument: str | None = None,
    treatment: str | None = None,
    cohort: str | None = None,
    covariates: Covariates | list[str] | None = None,
) -> Experiment:
    """Annotate a frame of rows with the roles of its columns.

    Give `treatment` alone for a randomized experiment (the assignment is the
    treatment), `instrument` and `treatment` for an encouragement design, or
    `instrument` alone when everyone assigned is treated.

    Parameters
    ----------
    frame : pandas.DataFrame, polars.DataFrame or polars.LazyFrame
        One row per unit of analysis (for example, one row per profile). With a
        lazy frame (``pl.scan_parquet(...)``), compression, bootstraps and
        two-pass effects stream it in batches.
    cluster : str
        The unit of randomization and of independence (for example, the account).
    instrument : str, optional
        The randomized assignment $Z$ (0/1, constant within a cluster).
    treatment : str, optional
        The treatment received $T$ (0/1).
    cohort : str, optional
        Cells within which $Z$ is randomized with one propensity, such as the day
        a cluster entered a ramped experiment. Each cluster then gets an inverse
        propensity weight.
    covariates : Covariates or list of str, optional
        From :func:`ume.covariates`. A list of names means numeric covariates
        centered at their means.

    Returns
    -------
    Experiment

    Examples
    --------
    >>> import ume
    >>> from ume.testing import simulate_accounts
    >>> df = simulate_accounts(2000, seed=1)
    >>> exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T", cohort="day",
    ...                    covariates=ume.covariates(x1=2.0, x2=2.0))
    >>> exp.design.treatment
    'T'
    """
    if instrument is None and treatment is None:
        raise ValueError("give the treatment, the instrument, or both")
    z = instrument if instrument is not None else treatment
    t = treatment if treatment is not None else instrument
    assert z is not None and t is not None
    if covariates is None:
        cov = Covariates()
    elif isinstance(covariates, Covariates):
        cov = covariates
    else:
        from ume._roles import covariates as make_covariates

        cov = make_covariates(*covariates)
    return Experiment(frame, Design(cluster=cluster, instrument=z, treatment=t, cohort=cohort, covariates=cov))


def design(
    *,
    cluster: str,
    instrument: str | None = None,
    treatment: str | None = None,
    cohort: str | None = None,
    covariates: Covariates | list[str] | None = None,
    metrics: tuple[Metric, ...] | list[Metric] = (),
) -> Design:
    """Declare a design without data, for streaming batches through
    :func:`ume.compress`, :func:`ume.bootstrap` or :func:`ume.effects`.

    The roles are as in :func:`annotate`. For a stream of batches, declare every
    reference point and categorical level, because they must be fixed before the
    pass.

    Parameters
    ----------
    cluster : str
        The unit of randomization and of independence.
    instrument : str, optional
        The randomized assignment $Z$ (0/1, constant within a cluster).
    treatment : str, optional
        The treatment received $T$ (0/1). Give the treatment, the instrument, or both.
    cohort : str, optional
        Cells within which $Z$ is randomized with one propensity.
    covariates : Covariates or list of str, optional
        From :func:`ume.covariates`; a list of names means numeric covariates
        whose reference points are resolved from data.
    metrics : sequence of Mean or Ratio
        The metrics to analyze.

    Returns
    -------
    Design

    Examples
    --------
    >>> import ume
    >>> d = ume.design(cluster="account", instrument="Z", treatment="T",
    ...                covariates=ume.covariates(x1=2.0), metrics=[ume.mean("y")])
    >>> d.covariates.resolved()
    True
    """
    if instrument is None and treatment is None:
        raise ValueError("give the treatment, the instrument, or both")
    z = instrument if instrument is not None else treatment
    t = treatment if treatment is not None else instrument
    assert z is not None and t is not None
    if covariates is None:
        cov = Covariates()
    elif isinstance(covariates, Covariates):
        cov = covariates
    else:
        from ume._roles import covariates as make_covariates

        cov = make_covariates(*covariates)
    return Design(cluster=cluster, instrument=z, treatment=t, cohort=cohort, covariates=cov).with_metrics(*metrics)
