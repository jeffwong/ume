"""Covariates and their reference points.

Each covariate enters the model centered at a fixed reference point
$x_{ref}$. A numeric covariate contributes one column, $x - x_{ref}$. A
categorical covariate contributes one indicator per level other than its
reference level, centered at 0, so its columns stay sparse.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Numeric:
    """A numeric covariate.

    Parameters
    ----------
    x_ref : float, optional
        The reference point. When omitted, the column mean is used and recorded
        when the design is bound to data.
    """

    x_ref: float | None = None


@dataclass(frozen=True)
class Categorical:
    """A categorical covariate, entered as one indicator per non-reference level.

    Parameters
    ----------
    reference : object
        The reference level. Its rows have all indicators equal to 0.
    levels : tuple, optional
        All levels, in order. When omitted, they are the sorted distinct values
        found when the design is bound to data.
    """

    reference: Any
    levels: tuple[Any, ...] | None = None

    def indicator_levels(self) -> tuple[Any, ...]:
        """The levels that get an indicator column: every level except the reference."""
        if self.levels is None:
            raise ValueError("levels are resolved when the design is bound to data")
        return tuple(level for level in self.levels if level != self.reference)


Spec = Numeric | Categorical


@dataclass(frozen=True)
class Covariates:
    """The covariates of a design, in order, each with its specification.

    Build with :func:`ume.covariates`.
    """

    specs: tuple[tuple[str, Spec], ...] = ()

    @property
    def names(self) -> list[str]:
        """Names of the covariate columns in the data."""
        return [name for name, _ in self.specs]

    def labels(self) -> list[str]:
        """Labels of the $k$ columns of $x$: ``name`` or ``name[level]``."""
        out: list[str] = []
        for name, spec in self.specs:
            if isinstance(spec, Numeric):
                out.append(name)
            else:
                out.extend(f"{name}[{level}]" for level in spec.indicator_levels())
        return out

    @property
    def k(self) -> int:
        """Number of columns of $x$."""
        return len(self.labels())

    def x_ref(self) -> list[float]:
        """The reference point of each of the $k$ columns of $x$ (0 for indicators)."""
        out: list[float] = []
        for _, spec in self.specs:
            if isinstance(spec, Numeric):
                if spec.x_ref is None:
                    raise ValueError("x_ref is resolved when the design is bound to data")
                out.append(spec.x_ref)
            else:
                out.extend(0.0 for _ in spec.indicator_levels())
        return out

    def resolved(self) -> bool:
        """Whether every reference point and level set is known."""
        return all((s.x_ref is not None) if isinstance(s, Numeric) else (s.levels is not None) for _, s in self.specs)

    def replace_spec(self, name: str, spec: Spec) -> Covariates:
        """Return a copy with the specification of `name` replaced."""
        return replace(self, specs=tuple((n, spec if n == name else s) for n, s in self.specs))


def covariates(*names: str, **specs: float | Spec | None) -> Covariates:
    """Declare the covariates of a design and their reference points.

    Parameters
    ----------
    *names : str
        Numeric covariates whose reference point is their column mean.
    **specs : float, Numeric, Categorical or None
        ``name=value`` sets a numeric covariate's reference point.
        ``name=ume.categorical(...)`` declares a categorical covariate, and
        ``name=None`` is the same as listing the name in `names`.

    Returns
    -------
    Covariates

    Examples
    --------
    >>> import ume
    >>> cov = ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv"))
    >>> cov.names
    ['x1', 'x2', 'kids', 'device']
    """
    out: list[tuple[str, Spec]] = [(n, Numeric()) for n in names]
    for name, value in specs.items():
        if isinstance(value, Numeric | Categorical):
            out.append((name, value))
        elif value is None:
            out.append((name, Numeric()))
        else:
            out.append((name, Numeric(float(value))))
    seen = [n for n, _ in out]
    if len(seen) != len(set(seen)):
        raise ValueError(f"duplicate covariates: {seen}")
    return Covariates(tuple(out))


def categorical(reference: Any, levels: list[Any] | tuple[Any, ...] | None = None) -> Categorical:
    """Declare a categorical covariate.

    Parameters
    ----------
    reference : object
        The reference level (all indicators 0).
    levels : list, optional
        All levels. Inferred from the data when omitted. Declare them when the
        data arrives in shards, so that every shard has the same columns.

    Returns
    -------
    Categorical

    Examples
    --------
    >>> import ume
    >>> ume.categorical(reference="tv", levels=["tv", "mobile", "web"]).indicator_levels()
    ('mobile', 'web')
    """
    if levels is not None and reference not in levels:
        raise ValueError(f"reference level {reference!r} is not in levels {list(levels)}")
    return Categorical(reference, tuple(levels) if levels is not None else None)
