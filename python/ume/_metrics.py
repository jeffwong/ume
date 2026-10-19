"""Metrics and the outcome columns they need.

The general metric is a ratio $N / D$. A mean metric is the special case
$D \\equiv 1$. Each metric column $Y$ enters the model twice: as $Y$ and as
$T \\cdot Y$. The treatment line of $T \\cdot Y$ read at the compliers' mean is the
compliers' level with treatment, $E[Y(1) \\mid C]$, and subtracting the line of
$Y$ gives $E[Y(0) \\mid C]$ (CSS post, "The ratio needs the compliers' own
levels"). The constant $D \\equiv 1$ needs no columns: its treatment line is 0
and the line of $T \\cdot 1 = T$ is exactly 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mean:
    """A mean metric: the average of one column per row.

    Parameters
    ----------
    column : str
        The outcome column.
    name : str, optional
        Display name; defaults to `column`.
    """

    column: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.column)

    @property
    def numerator(self) -> str:
        """The column $N$ of the metric viewed as a ratio $N / 1$."""
        return self.column

    @property
    def denominator(self) -> None:
        """None: the denominator of a mean is the constant 1."""
        return None

    @property
    def kind(self) -> str:
        """``"mean"``."""
        return "mean"


@dataclass(frozen=True)
class Ratio:
    """A ratio metric: the ratio of the totals of two columns, such as rebuffers per play.

    Parameters
    ----------
    numerator : str
        Column $N$.
    denominator : str
        Column $D$.
    name : str, optional
        Display name; defaults to ``"numerator/denominator"``.
    """

    numerator: str
    denominator: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"{self.numerator}/{self.denominator}")

    @property
    def kind(self) -> str:
        """``"ratio"``."""
        return "ratio"


Metric = Mean | Ratio


def mean(column: str, name: str | None = None) -> Mean:
    """Declare a mean metric.

    Parameters
    ----------
    column : str
        The outcome column.
    name : str, optional
        Display name.

    Returns
    -------
    Mean

    Examples
    --------
    >>> import ume
    >>> ume.mean("y", name="hours")
    Mean(column='y', name='hours')
    """
    return Mean(column, name or "")


def ratio(numerator: str, denominator: str, name: str | None = None) -> Ratio:
    """Declare a ratio metric $N / D$.

    Parameters
    ----------
    numerator, denominator : str
        The columns $N$ and $D$.
    name : str, optional
        Display name.

    Returns
    -------
    Ratio

    Examples
    --------
    >>> import ume
    >>> ume.ratio("N", "D", name="rebuffers_per_play").name
    'rebuffers_per_play'
    """
    return Ratio(numerator, denominator, name or "")


@dataclass(frozen=True)
class OutcomeColumn:
    """One column of the outcome matrix $Y$: a data column, optionally times $T$."""

    source: str
    treated: bool

    @property
    def label(self) -> str:
        """``"y"`` or ``"T·y"``."""
        return f"T·{self.source}" if self.treated else self.source


@dataclass(frozen=True)
class OutcomeLayout:
    """The outcome columns of the model, derived from the metrics.

    Each distinct metric column $Y$ contributes $Y$ and $T \\cdot Y$, once, however
    many metrics use it. Columns are ordered: every raw column, then every
    treated column.
    """

    columns: tuple[OutcomeColumn, ...]

    @classmethod
    def from_metrics(cls, metrics: tuple[Metric, ...]) -> OutcomeLayout:
        """Build the layout for `metrics`."""
        sources: list[str] = []
        for m in metrics:
            for col in (m.numerator, m.denominator):
                if col is not None and col not in sources:
                    sources.append(col)
        cols = [OutcomeColumn(s, False) for s in sources] + [OutcomeColumn(s, True) for s in sources]
        return cls(tuple(cols))

    @property
    def o(self) -> int:
        """Number of outcome columns $o$."""
        return len(self.columns)

    @property
    def labels(self) -> list[str]:
        """Labels of the outcome columns, in order."""
        return [c.label for c in self.columns]

    def index(self, source: str, treated: bool = False) -> int:
        """Position of the column `source` (or $T \\cdot$ `source`) in $Y$."""
        return self.columns.index(OutcomeColumn(source, treated))
