"""Where rows come from: a frame in memory, a lazy frame, or a stream of batches.

Every pass (compression, the exact bootstrap, the two passes of delta vectors
early) reads its rows from a :class:`Source`, which guarantees two things:

1. the design is **resolved**: every reference point $x_{ref}$ and every
   categorical level is fixed before the first row is read;
2. the rows arrive **grouped by cluster**, in batches.

==========================  ===============================  ==============================
Source                      Resolving                        Grouping
==========================  ===============================  ==============================
pandas / polars DataFrame   column means and distinct values  sorted in memory if needed
polars LazyFrame            one lazy aggregation              a lazy sort by cluster
                                                              (skip with ``presorted=True``)
iterable of frames          must be declared                  must already be grouped
==========================  ===============================  ==============================
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np

from ume._data import _column, cluster_keys, resolve
from ume._design import Design
from ume._roles import Categorical, Numeric

#: Rows per batch when streaming a lazy frame.
BATCH_SIZE = 500_000


@dataclass
class Source:
    """Rows grouped by cluster, with a resolved design.

    Attributes
    ----------
    design : Design
        The resolved design.
    batches : callable
        Returns a fresh iterator of native frames, each a batch of rows.
    n_clusters : callable or None
        Counts the clusters (one aggregation), when the source allows it.
    check_levels : bool
        Whether batches must be checked for undeclared categorical levels.
    reiterable : bool
        Whether ``batches`` can be called more than once (two-pass methods need it).
    """

    design: Design
    batches: Callable[[], Iterator[Any]]
    n_clusters: Callable[[], int] | None
    check_levels: bool
    reiterable: bool


def is_lazy(obj: Any) -> bool:
    """Whether `obj` is a lazy frame narwhals understands (polars LazyFrame, and others)."""
    try:
        return isinstance(nw.from_native(obj), nw.LazyFrame)
    except TypeError:
        return False


def is_eager(obj: Any) -> bool:
    """Whether `obj` is an eager frame narwhals understands (pandas, polars DataFrame)."""
    try:
        nw.from_native(obj, eager_only=True)
        return True
    except TypeError:
        return False


def lazy_columns(obj: Any) -> list[str]:
    """Column names of a lazy frame, from its schema."""
    return list(nw.from_native(obj).collect_schema().names())


def resolve_lazy(native: Any, design: Design) -> Design:
    """Resolve the reference points and levels of a lazy frame in one aggregation per kind."""
    lf = nw.from_native(native)
    missing = [c for c in design.columns if c not in lazy_columns(native)]
    if missing:
        raise ValueError(f"columns not found: {missing}")
    if not design.metrics:
        raise ValueError("declare at least one metric first, for example exp.metrics(ume.mean('y'))")
    cov = design.covariates
    means = [name for name, spec in cov.specs if isinstance(spec, Numeric) and spec.x_ref is None]
    if means:
        row = lf.select(*[nw.col(n).cast(nw.Float64).mean() for n in means]).collect().row(0)
        for name, value in zip(means, row, strict=True):
            cov = cov.replace_spec(name, Numeric(float(value)))
    for name, spec in cov.specs:
        if isinstance(spec, Categorical) and spec.levels is None:
            seen = sorted(lf.select(name).unique().collect()[name].to_list())
            if spec.reference not in seen:
                raise ValueError(f"reference level {spec.reference!r} of {name!r} does not occur in the data")
            cov = cov.replace_spec(name, Categorical(spec.reference, tuple(seen)))
    return dataclasses.replace(design, covariates=cov)


def _contiguous(keys: np.ndarray) -> bool:
    """Whether every cluster's rows are contiguous. Sorted keys are, which is checked in O(n) first."""
    if bool(np.all(keys[1:] >= keys[:-1])):
        return True
    starts = np.concatenate([[True], keys[1:] != keys[:-1]])
    return int(starts.sum()) == int(np.unique(keys).size)


def open_source(rows: Any, design: Design, *, batch_size: int | None = None, presorted: bool = False) -> Source:
    """Wrap `rows` (a frame, a lazy frame, or an iterable of frames) as a :class:`Source`.

    Raises
    ------
    ValueError
        If the source is an iterable of batches and the design is not fully
        declared (reference points and levels must be fixed before the pass).
    """
    if is_eager(rows):
        frame, resolved = resolve(rows, design)
        keys = cluster_keys(_column(frame, resolved.cluster))
        if not _contiguous(keys):
            frame = frame[np.argsort(keys, kind="stable").tolist()]
        native = nw.to_native(frame)
        return Source(
            resolved, lambda: iter([native]), lambda: int(np.unique(keys).size), check_levels=False, reiterable=True
        )
    if is_lazy(rows):
        resolved = resolve_lazy(rows, design)
        lf = nw.from_native(rows)
        ordered = nw.to_native(lf if presorted else lf.sort(resolved.cluster))
        size = batch_size or BATCH_SIZE

        def batches() -> Iterator[Any]:
            if hasattr(ordered, "collect_batches"):  # polars streams; other engines collect
                return iter(ordered.collect_batches(chunk_size=size))
            return iter([nw.to_native(nw.from_native(ordered).collect())])

        def n_clusters() -> int:
            return int(lf.select(nw.col(resolved.cluster).n_unique()).collect().item())

        return Source(resolved, batches, n_clusters, check_levels=False, reiterable=True)
    if not design.covariates.resolved():
        raise ValueError(
            "streaming batches needs every x_ref and categorical level declared before the pass "
            "(ume.covariates(x1=2.0, ...), ume.categorical(reference=..., levels=[...]))"
        )
    if callable(rows):
        return Source(design, rows, None, check_levels=True, reiterable=True)
    if isinstance(rows, Iterable):
        consumed = [False]

        def once() -> Iterator[Any]:
            if consumed[0]:
                raise ValueError("this iterable of batches was already read; pass a function that returns batches")
            consumed[0] = True
            return iter(rows)

        return Source(design, once, None, check_levels=True, reiterable=False)
    raise TypeError(f"cannot read rows from {type(rows).__name__}")
