"""From a frame of rows to the arrays of the engine.

This is the only module that reads data columns. It validates the roles,
resolves reference points and levels, computes the inverse propensity weights,
and builds the design matrix $R = (1, x - x_{ref})$, dense or sparse.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
import numpy as np
from numpy.typing import NDArray

from ume import _core
from ume._design import Design, DesignFeatures
from ume._roles import Categorical, Covariates, Numeric

if TYPE_CHECKING:
    from ume._compress import CompressedData

SparseOption = bool | Literal["auto"]
#: Use a sparse design matrix when at most this share of its entries is nonzero.
SPARSE_DENSITY = 0.3


@dataclass(frozen=True)
class CSR:
    """A design matrix in compressed sparse row form."""

    indptr: NDArray[np.uintp]
    indices: NDArray[np.uintp]
    data: NDArray[np.float64]
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        """Number of stored entries."""
        return int(self.data.size)

    def toarray(self) -> NDArray[np.float64]:
        """The dense matrix (for tests and small data)."""
        out = np.zeros(self.shape)
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr).astype(np.int64))
        out[rows, self.indices.astype(np.int64)] = self.data
        return out


@dataclass(frozen=True)
class ModelData:
    """The engine's view of the rows.

    Attributes
    ----------
    r : numpy.ndarray or CSR
        Design matrix $R = (1, x - x_{ref})$, $n \\times d$.
    z, t : numpy.ndarray
        Instrument and treatment received per row.
    y : numpy.ndarray
        Outcome matrix $Y$, $n \\times o$, in the order of the design's layout.
    w : numpy.ndarray
        Inverse propensity weight per row (1 without a cohort column).
    cluster : numpy.ndarray
        Cluster index per row, ``uint32`` in ``0..n_clusters``.
    n_clusters : int
        Number of clusters $G$.
    e_hat : numpy.ndarray
        Propensity per cohort (empty without a cohort column).
    cohorts : list
        Cohort labels, in the order of `e_hat`.
    design : Design
        The design, with reference points and levels resolved.
    """

    r: NDArray[np.float64] | CSR
    z: NDArray[np.float64]
    t: NDArray[np.float64]
    y: NDArray[np.float64]
    w: NDArray[np.float64]
    cluster: NDArray[np.uint32]
    n_clusters: int
    e_hat: NDArray[np.float64]
    cohorts: list[Any]
    design: Design
    native: Any = None

    @property
    def n(self) -> int:
        """Number of rows."""
        return int(self.z.size)

    @property
    def sparse(self) -> bool:
        """Whether the design matrix is sparse."""
        return isinstance(self.r, CSR)

    def features(self) -> DesignFeatures:
        """The special case of the universal model these rows are."""
        e = self.e_hat
        return DesignFeatures(
            encouraged=bool(np.any(self.t != self.z)),
            clustered=self.n > self.n_clusters,
            adaptive=bool(e.size > 1 and np.ptp(e) > 1e-12),
            n_cohorts=max(int(e.size), 1),
            weighted=self.design.cohort is not None,
            ratio=any(m.kind == "ratio" for m in self.design.metrics),
            k=self.design.covariates.k,
        )

    def itt(self) -> ModelData:
        """The rows of the intent to treat model: the treatment replaced by the instrument, $T := Z$.

        Every $T \\cdot Y$ outcome column becomes $Z \\cdot Y$.
        """
        layout = self.design.layout
        y = self.y.copy()
        for j, c in enumerate(layout.columns):
            if c.treated:
                y[:, j] = self.z * self.y[:, layout.index(c.source)]
        return dataclasses.replace(self, t=self.z, y=np.ascontiguousarray(y))

    def compressed(self) -> CompressedData:
        """The rows, compressed (one pass)."""
        from ume._compress import compress

        if self.native is None:
            raise ValueError("these rows were built without their frame; compress the Experiment instead")
        return compress(self.native, self.design, self.sparse)

    def fit_raw(self) -> dict[str, Any]:
        """Fit the universal model with the Rust engine and return its raw output."""
        args = (self.z, self.t, self.y, self.w, self.cluster, self.n_clusters)
        if isinstance(self.r, CSR):
            return _core.fit_rows_sparse(self.r.indptr, self.r.indices, self.r.data, self.r.shape[1], *args)
        return _core.fit_rows_dense(self.r, *args)


def _check_nulls(frame: Any, name: str) -> None:
    if frame[name].null_count():
        raise ValueError(f"column {name!r} has missing values; ume does not handle missing values yet")


def _column(frame: Any, name: str) -> NDArray[Any]:
    _check_nulls(frame, name)
    values: NDArray[Any] = frame[name].to_numpy()
    return values


def _binary(values: NDArray[Any], name: str) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64)
    if not np.isin(out, (0.0, 1.0)).all():
        raise ValueError(f"column {name!r} must be 0 or 1")
    return out


def _constant_within(values: NDArray[Any], cluster: NDArray[np.intp], n_clusters: int, name: str) -> NDArray[Any]:
    """The value of each cluster, checking that `values` is constant within clusters."""
    first = np.empty(n_clusters, dtype=values.dtype)
    first[cluster[::-1]] = values[::-1]
    if not np.array_equal(first[cluster], values):
        raise ValueError(f"column {name!r} must be constant within each cluster")
    return first


def resolve_covariates(frame: Any, covariates: Covariates) -> Covariates:
    """Fill in missing reference points (column means) and levels (sorted distinct values)."""
    out = covariates
    for name, spec in covariates.specs:
        if isinstance(spec, Numeric) and spec.x_ref is None:
            out = out.replace_spec(name, Numeric(float(np.mean(_column(frame, name).astype(np.float64)))))
        elif isinstance(spec, Categorical):
            seen = sorted(frame[name].drop_nulls().unique().to_list())  # hashed in the frame's engine
            if spec.levels is None:
                if spec.reference not in seen:
                    raise ValueError(f"reference level {spec.reference!r} of {name!r} does not occur in the data")
                out = out.replace_spec(name, Categorical(spec.reference, tuple(seen)))
            elif unknown := set(seen) - set(spec.levels):
                raise ValueError(f"column {name!r} has levels not declared: {sorted(unknown)}")
    return out


def design_matrix(frame: Any, covariates: Covariates, sparse: SparseOption = "auto") -> NDArray[np.float64] | CSR:
    """Build $R = (1, x - x_{ref})$ for resolved `covariates`, dense or sparse.

    With ``sparse="auto"``, the matrix is sparse when at most 30% of its entries
    are nonzero.
    """
    n = len(frame)
    cols: list[tuple[NDArray[np.intp] | None, NDArray[np.float64]]] = [(None, np.ones(n))]
    for name, spec in covariates.specs:
        if isinstance(spec, Numeric):
            values = _column(frame, name)  # also rejects missing values
            assert spec.x_ref is not None, "resolve_covariates first"
            cols.append((None, values.astype(np.float64) - spec.x_ref))
        else:
            _check_nulls(frame, name)  # no conversion to numpy: strings are slow to convert
            levels = spec.indicator_levels()
            # Compare in the frame's engine (vectorized, and fast for strings).
            hits = frame.select(*[(nw.col(name) == level).alias(f"l{i}") for i, level in enumerate(levels)])
            for i in range(len(levels)):
                idx = np.flatnonzero(hits[f"l{i}"].to_numpy())
                cols.append((idx, np.ones(idx.size)))
    d = len(cols)
    nnz = sum(n if rows is None else rows.size for rows, _ in cols)
    use_sparse = sparse if isinstance(sparse, bool) else nnz <= SPARSE_DENSITY * n * d
    if not use_sparse:
        dense = np.zeros((n, d))
        for j, (rows, vals) in enumerate(cols):
            if rows is None:
                dense[:, j] = vals
            else:
                dense[rows, j] = vals
        return dense
    row_idx = np.concatenate([np.arange(n) if rows is None else rows for rows, _ in cols])
    col_idx = np.concatenate([np.full(n if rows is None else rows.size, j) for j, (rows, _) in enumerate(cols)])
    data = np.concatenate([vals for _, vals in cols])
    keep = data != 0
    row_idx, col_idx, data = row_idx[keep], col_idx[keep], data[keep]
    order = np.lexsort((col_idx, row_idx))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(row_idx, minlength=n))]).astype(np.uintp)
    return CSR(indptr, col_idx[order].astype(np.uintp), np.ascontiguousarray(data[order]), (n, d))


def resolve(native: Any, design: Design) -> tuple[Any, Design]:
    """Check `native` against `design`, and resolve the design's reference points and levels.

    Returns the narwhals frame and the resolved design.

    Raises
    ------
    ValueError
        If no metrics are declared, or a role column is missing.
    """
    if not design.metrics:
        raise ValueError("declare at least one metric first, for example exp.metrics(ume.mean('y'))")
    frame = nw.from_native(native, eager_only=True)
    missing = [c for c in design.columns if c not in frame.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}")
    return frame, dataclasses.replace(design, covariates=resolve_covariates(frame, design.covariates))


def batch_arrays(frame: Any, design: Design) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The instrument $Z$, treatment $T$ and outcome matrix $Y$ of a batch of rows."""
    z = _binary(_column(frame, design.instrument), design.instrument)
    t = z if design.treatment == design.instrument else _binary(_column(frame, design.treatment), design.treatment)
    layout = design.layout
    columns = {c.source: _column(frame, c.source).astype(np.float64) for c in layout.columns if not c.treated}
    y = np.column_stack([t * columns[c.source] if c.treated else columns[c.source] for c in layout.columns])
    return np.ascontiguousarray(z), np.ascontiguousarray(t), np.ascontiguousarray(y)


def cluster_keys(values: NDArray[Any]) -> NDArray[np.uint64]:
    """A stable 64-bit key per row from its cluster id.

    Integer ids are their own key. Other ids are hashed (BLAKE2b, 8 bytes, of
    ``str(id)``), so every process and shard computes the same key for a cluster.
    """
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64).view(np.uint64)
    labels, inverse = np.unique(values, return_inverse=True)
    hashed = np.array(
        [int.from_bytes(hashlib.blake2b(str(v).encode(), digest_size=8).digest(), "little") for v in labels],
        dtype=np.uint64,
    )
    return hashed[inverse]


def model_data(native: Any, design: Design, sparse: SparseOption = "auto") -> ModelData:
    """Validate `native` against `design` and build the engine's arrays.

    Raises
    ------
    ValueError
        If a role column is missing, has missing values, is not 0/1 where it
        must be, or varies within a cluster where it must be constant; or if no
        metrics are declared.
    """
    frame, design = resolve(native, design)
    labels, cluster = np.unique(_column(frame, design.cluster), return_inverse=True)
    cluster = cluster.astype(np.intp)
    n_clusters = int(labels.size)
    z, t, y = batch_arrays(frame, design)
    cluster_z = _constant_within(z, cluster, n_clusters, design.instrument)
    if design.cohort is None:
        w = np.ones(z.size)
        e_hat, cohorts = np.empty(0), []
    else:
        cohort_labels, cohort = np.unique(_column(frame, design.cohort), return_inverse=True)
        cluster_cohort = _constant_within(cohort.astype(np.intp), cluster, n_clusters, design.cohort)
        e_hat, w_cluster = _core.inverse_propensity(
            cluster_z, cluster_cohort.astype(np.uint32), np.ones(n_clusters), int(cohort_labels.size)
        )
        w = w_cluster[cluster]
        cohorts = cohort_labels.tolist()
    return ModelData(
        r=design_matrix(frame, design.covariates, sparse),
        z=z,
        t=t,
        y=y,
        w=np.ascontiguousarray(w),
        cluster=cluster.astype(np.uint32),
        n_clusters=n_clusters,
        e_hat=np.asarray(e_hat),
        cohorts=cohorts,
        design=design,
        native=native,
    )
