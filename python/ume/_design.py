"""The design: which column plays which role, and what the data says about it.

A :class:`Design` names the columns. :class:`DesignFeatures` is the taxonomy:
which special case of the universal model the data is. Every feature is
detected from the data, never switched by hand, because each one is a property
of the same model:

=================  ==========================  ===============================
Feature            Detected when               Part of the model it touches
=================  ==========================  ===============================
encouragement      $T \\ne Z$ in some row       the first stage; the estimand
clusters           more rows than clusters     the meat (CR0, else HC0)
adaptive           propensities differ         the inverse propensity weights
ratio metrics      a ``ume.ratio`` metric      the outcome columns
covariates         $k > 0$                     the lines
=================  ==========================  ===============================
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ume._metrics import Metric, OutcomeLayout
from ume._roles import Covariates

if TYPE_CHECKING:
    from ume._experiment import Experiment


@dataclass(frozen=True)
class Design:
    """The roles of the columns of an experiment's rows.

    Build with :func:`ume.annotate`, or directly and then :meth:`bind` to data.

    Parameters
    ----------
    cluster : str
        The unit of randomization and of independence. Rows of a cluster may be
        correlated in any way; clusters are independent.
    instrument : str
        The randomized assignment $Z$, 0 or 1, constant within a cluster.
    treatment : str
        The treatment received $T$, 0 or 1. Equal to `instrument` for a design
        without encouragement.
    cohort : str, optional
        The cells within which $Z$ is randomized with one propensity, constant
        within a cluster. When given, each cluster gets an inverse propensity
        weight. When omitted, all weights are 1.
    covariates : Covariates
        Pre-treatment covariates and their reference points.
    metrics : tuple of Metric
        The metrics to analyze.
    """

    cluster: str
    instrument: str
    treatment: str
    cohort: str | None = None
    covariates: Covariates = field(default_factory=Covariates)
    metrics: tuple[Metric, ...] = ()

    @property
    def columns(self) -> list[str]:
        """Every data column the design uses."""
        roles = [self.cluster, self.instrument, self.treatment, self.cohort, *self.covariates.names]
        for m in self.metrics:
            roles += [m.numerator, m.denominator]
        out: list[str] = []
        for c in roles:
            if c is not None and c not in out:
                out.append(c)
        return out

    @property
    def layout(self) -> OutcomeLayout:
        """The outcome columns $Y$ derived from the metrics."""
        return OutcomeLayout.from_metrics(self.metrics)

    def with_metrics(self, *metrics: Metric) -> Design:
        """Return a copy that analyzes `metrics`, replacing any metrics already declared."""
        names = [m.name for m in metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"metric names must be unique, got {names}")
        return replace(self, metrics=tuple(metrics))

    def bind(self, frame: object) -> Experiment:
        """Attach this design to a frame of rows.

        Parameters
        ----------
        frame : pandas.DataFrame, polars.DataFrame or polars.LazyFrame
            The rows; every column the design names must be present.

        Returns
        -------
        Experiment
        """
        from ume._experiment import Experiment

        return Experiment(frame, self)


@dataclass(frozen=True)
class DesignFeatures:
    """Which special case of the universal model an experiment is.

    Made by :meth:`Experiment.features`, :meth:`CompressedData.features` and
    every fit, from the data, rather than built directly.

    Parameters
    ----------
    encouraged : bool
        $T \\ne Z$ in some row: an encouragement design, estimand LATE.
    clustered : bool
        Some cluster has more than one row: cluster-robust covariance (CR0).
    adaptive : bool
        The propensity of $Z = 1$ differs across cohorts.
    n_cohorts : int
        Number of cohorts (1 when no cohort column is given).
    weighted : bool
        Inverse propensity weights are used (a cohort column is given).
    ratio : bool
        Some metric is a ratio.
    k : int
        Number of covariate columns.
    """

    encouraged: bool
    clustered: bool
    adaptive: bool
    n_cohorts: int
    weighted: bool
    ratio: bool
    k: int

    @property
    def estimand(self) -> str:
        """``"LATE"`` with encouragement (the compliers' effect), else ``"ATE"``."""
        return "LATE" if self.encouraged else "ATE"

    @property
    def covariance(self) -> str:
        """``"CR0"`` when clustered, else ``"HC0"``: the same formula."""
        return "CR0" if self.clustered else "HC0"

    def describe(self) -> str:
        """A one-line name of the design, such as
        ``"clustered encouragement design with adaptive assignment (4 cohorts)"``."""
        parts = ["clustered" if self.clustered else "unit-randomized"]
        parts.append("encouragement design" if self.encouraged else "randomized experiment")
        text = " ".join(parts)
        if self.adaptive:
            text += f" with adaptive assignment ({self.n_cohorts} cohorts)"
        if self.k:
            text += f", {self.k} covariate column{'s' if self.k > 1 else ''}"
        return text


def design_to_dict(design: Design) -> dict[str, object]:
    """A JSON-safe description of a design: roles, covariates with $x_{ref}$ and levels, metrics."""
    from ume._roles import Categorical, Numeric

    covs: list[dict[str, object]] = []
    for name, spec in design.covariates.specs:
        if isinstance(spec, Numeric):
            covs.append({"name": name, "kind": "numeric", "x_ref": spec.x_ref})
        else:
            assert isinstance(spec, Categorical)
            levels = list(spec.levels) if spec.levels is not None else None
            covs.append({"name": name, "kind": "categorical", "reference": spec.reference, "levels": levels})
    metrics = [
        {"kind": m.kind, "numerator": m.numerator, "denominator": m.denominator, "name": m.name} for m in design.metrics
    ]
    return {
        "cluster": design.cluster,
        "instrument": design.instrument,
        "treatment": design.treatment,
        "cohort": design.cohort,
        "covariates": covs,
        "metrics": metrics,
    }


def design_from_dict(data: dict[str, Any]) -> Design:
    """The inverse of :func:`design_to_dict`."""
    from ume._metrics import Mean, Ratio
    from ume._roles import Categorical, Covariates, Numeric

    specs: list[tuple[str, Numeric | Categorical]] = []
    for c in data["covariates"]:
        if c["kind"] == "numeric":
            specs.append((c["name"], Numeric(c["x_ref"])))
        else:
            levels = tuple(c["levels"]) if c["levels"] is not None else None
            specs.append((c["name"], Categorical(c["reference"], levels)))
    metrics: list[Mean | Ratio] = [
        Mean(m["numerator"], m["name"]) if m["kind"] == "mean" else Ratio(m["numerator"], m["denominator"], m["name"])
        for m in data["metrics"]
    ]
    return Design(
        cluster=data["cluster"],
        instrument=data["instrument"],
        treatment=data["treatment"],
        cohort=data["cohort"],
        covariates=Covariates(tuple(specs)),
        metrics=tuple(metrics),
    )
