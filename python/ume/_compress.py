"""The compressed dataset: per-stratum sufficient statistics, built in one pass.

A **stratum** is an arm × cohort, $h = (Z, \\text{cohort})$: every cluster in it has
the same instrument and the same weight. Its **compressed row** holds three sums
over its clusters:

- $G_h$, the number of clusters;
- the **first moments** $\\sum_{a \\in h} u_a$, which give the estimate;
- the **second moments** $\\sum_{a \\in h} u_a u_a^\\top$, which give the covariance.

Here $u_a$ is the **cluster summary**: the blocks $(S^Y, S, S^T)$ of the
cluster's rows, built in a buffer during the pass and never stored. Compressed
rows add, so shards can be compressed separately and merged with ``+``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import warnings
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple

import narwhals as nw
import numpy as np
from numpy.typing import NDArray

from ume import _core
from ume._config import get_threads
from ume._data import CSR, SparseOption, _column, batch_arrays, cluster_keys, design_matrix, resolve_covariates
from ume._design import Design, DesignFeatures, design_from_dict, design_to_dict
from ume._draws import Draws
from ume._drawsums import DrawSums
from ume._effects import AnyEffect, default_effects, point, value_and_gradient
from ume._fit import Z_975, EffectTable, UniversalFit
from ume._linearize import stratum_gradients
from ume._source import Source, open_source
from ume._table import Table

#: Strata smaller than this make the Gaussian posterior unreliable (decision 0019).
SMALL_STRATUM = 200


class Stratum(NamedTuple):
    """A stratum: an instrument arm within a cohort. Every cluster in it has the same weight.

    Parameters
    ----------
    Z : int
        The instrument arm, 0 or 1.
    cohort : object, optional
        The cohort label, a value of the design's cohort column; None when the
        design has no cohort column.
    """

    Z: int
    cohort: Any = None


def _row_labels(design: Design) -> list[str]:
    return ["1", *design.covariates.labels()]


class CompressedRow:
    """The compressed row of one stratum.

    Made by indexing :class:`CompressedData` with a :class:`Stratum`; rarely
    built directly.

    Parameters
    ----------
    stratum : Stratum
        The stratum the row belongs to.
    g : float
        Its number of clusters $G_h$.
    first : numpy.ndarray
        $\\sum_{a \\in h} u_a$, length $q$, in the distinct layout.
    second : numpy.ndarray
        $\\sum_{a \\in h} u_a u_a^\\top$, shape $(q, q)$.
    design : Design
        The resolved design, which labels the entries.

    Attributes
    ----------
    G : int
        Number of clusters $G_h$.
    first_moments : numpy.ndarray
        $\\sum_{a \\in h} u_a$, length $q$, in the order of :attr:`entries`.
    second_moments : numpy.ndarray
        $\\sum_{a \\in h} u_a u_a^\\top$, $q \\times q$.
    """

    def __init__(self, stratum: Stratum, g: float, first: NDArray[np.float64], second: NDArray[np.float64],
                 design: Design) -> None:  # fmt: skip
        self.stratum = stratum
        self.G = int(g)
        self.first_moments = first
        self.second_moments = second
        self._design = design

    @property
    def entries(self) -> list[tuple[str, str, str]]:
        """Label of each entry of $u_a$: ``(block, row, column)``, with block ``S^Y``, ``S`` or ``S^T``."""
        rows = _row_labels(self._design)
        outcomes = self._design.layout.labels
        d, o = len(rows), len(outcomes)
        return [
            (block, rows[i], outcomes[j] if block == "S^Y" else rows[j]) for block, i, j in _core.summary_layout(d, o)
        ]

    def blocks(self) -> dict[str, Table]:
        """The first moments as the three blocks $S^Y$ ($d \\times o$), $S$ and $S^T$ ($d \\times d$)."""
        rows = _row_labels(self._design)
        outcomes = self._design.layout.labels
        d = len(rows)
        mats = {"S^Y": np.zeros((d, len(outcomes))), "S": np.zeros((d, d)), "S^T": np.zeros((d, d))}
        for value, (block, i, j) in zip(self.first_moments, _core.summary_layout(d, len(outcomes)), strict=True):
            mats[block][i, j] = value
            if block != "S^Y":
                mats[block][j, i] = value
        heads = {"S^Y": outcomes, "S": rows, "S^T": rows}
        return {
            name: Table(["", *heads[name]], [[r, *m[k]] for k, r in enumerate(rows)], title=f"{name}, {self.stratum}")
            for name, m in mats.items()
        }

    def __repr__(self) -> str:
        q = self.first_moments.size
        return f"<CompressedRow {self.stratum}: G = {self.G:,}, q = {q}, second moments {q} × {q}>"


class CompressedData(Mapping[Stratum, CompressedRow]):
    """The compressed dataset: one :class:`CompressedRow` per stratum.

    It is a mapping from :class:`Stratum` to :class:`CompressedRow`, so it reads
    like a dictionary of sufficient statistics. Its size does not depend on the
    number of rows. Fitting it gives exactly the fit on the rows.

    Made by :meth:`ume.Experiment.compress`, :func:`ume.compress`,
    :meth:`ume.Compressor.finish` or :func:`ume.load`, rather than built directly.
    Merge shards with ``+``.

    Parameters
    ----------
    design : Design
        The resolved design: roles, reference points, levels and metrics.
    strata : list of Stratum
        The strata, one per compressed row.
    g : numpy.ndarray
        Clusters per stratum $G_h$, shape $(H,)$.
    first : numpy.ndarray
        First moments $\\sum_{a \\in h} u_a$ per stratum, shape $(H, q)$.
    second : numpy.ndarray
        Second moments $\\sum_{a \\in h} u_a u_a^\\top$ per stratum, shape $(H, q, q)$.
    n_rows : int
        Number of rows compressed.
    cluster_keys : numpy.ndarray, optional
        Sorted 64-bit keys of the clusters compressed. With them, ``+`` refuses
        shards that share a cluster; without them it cannot check.
    """

    def __init__(
        self,
        design: Design,
        strata: list[Stratum],
        g: NDArray[np.float64],
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        n_rows: int,
        cluster_keys: NDArray[np.uint64] | None = None,
    ) -> None:
        order = sorted(range(len(strata)), key=lambda i: (str(strata[i].cohort), strata[i].Z))
        self.design = design
        self.strata = [strata[i] for i in order]
        self.g = np.ascontiguousarray(g[order])
        self.first = np.ascontiguousarray(first[order])
        self.second = np.ascontiguousarray(second[order])
        self._n_rows = int(n_rows)
        #: Sorted keys of the clusters compressed, to reject merges of shards that share a cluster.
        self.cluster_keys = cluster_keys

    # ---- the mapping ------------------------------------------------------------------------------

    def __getitem__(self, key: Stratum) -> CompressedRow:
        i = self.strata.index(Stratum(*key))
        return CompressedRow(self.strata[i], self.g[i], self.first[i], self.second[i], self.design)

    def __iter__(self) -> Iterator[Stratum]:
        return iter(self.strata)

    def __len__(self) -> int:
        return len(self.strata)

    # ---- sizes ------------------------------------------------------------------------------------

    @property
    def d(self) -> int:
        """Columns of the design matrix."""
        return self.design.covariates.k + 1

    @property
    def o(self) -> int:
        """Outcome columns."""
        return self.design.layout.o

    @property
    def q(self) -> int:
        """Length of a cluster summary, $q = do + d(d+1)$."""
        return int(self.first.shape[1])

    @property
    def n(self) -> int:
        """Number of rows compressed."""
        return self._n_rows

    @property
    def n_clusters(self) -> int:
        """Number of clusters, $\\sum_h G_h$."""
        return int(self.g.sum())

    # ---- the model --------------------------------------------------------------------------------

    def _cohorts(self) -> tuple[NDArray[np.float64], NDArray[np.uint32], int]:
        z = np.array([float(h.Z) for h in self.strata])
        labels = sorted({h.cohort for h in self.strata}, key=str)
        cohort = np.array([labels.index(h.cohort) for h in self.strata], dtype=np.uint32)
        return z, cohort, len(labels)

    def e_hat(self) -> dict[Any, float]:
        """Propensity of $Z = 1$ per cohort: the share of the cohort's clusters in the arm $Z = 1$."""
        if self.design.cohort is None:
            return {}
        z, cohort, n_cohorts = self._cohorts()
        e, _ = _core.inverse_propensity(z, cohort, self.g, n_cohorts)
        labels = sorted({h.cohort for h in self.strata}, key=str)
        return dict(zip(labels, (float(v) for v in e), strict=True))

    def features(self) -> DesignFeatures:
        """The special case of the universal model, read from the compressed rows."""
        entries = _core.summary_layout(self.d, self.o)
        s = {(i, j): k for k, (b, i, j) in enumerate(entries) if b == "S"}
        st = {(i, j): k for k, (b, i, j) in enumerate(entries) if b == "S^T"}
        z = np.array([float(h.Z) for h in self.strata])
        s_idx, st_idx = [s[p] for p in s], [st[p] for p in s]
        encouraged = bool(np.any(self.first[:, st_idx] != z[:, None] * self.first[:, s_idx]))
        e = np.array(list(self.e_hat().values()))
        return DesignFeatures(
            encouraged=encouraged,
            clustered=self.n > self.n_clusters,
            adaptive=bool(e.size > 1 and np.ptp(e) > 1e-12),
            n_cohorts=max(int(e.size), 1),
            weighted=self.design.cohort is not None,
            ratio=any(m.kind == "ratio" for m in self.design.metrics),
            k=self.design.covariates.k,
        )

    def fit_raw(self) -> dict[str, Any]:
        """Fit the universal model with the Rust engine and return its raw output."""
        z, cohort, n_cohorts = self._cohorts()
        weighted = self.design.cohort is not None
        return _core.fit_compressed(self.d, self.o, self.g, self.first, self.second, z, cohort, n_cohorts, weighted)

    def fit(self) -> UniversalFit:
        """Fit the universal model from the compressed rows.

        Returns
        -------
        UniversalFit
            The same fit as :meth:`ume.Experiment.fit` on the rows.
        """
        return UniversalFit(self.fit_raw(), self)

    def compressed(self) -> CompressedData:
        """Itself: already compressed."""
        return self

    def posterior(self, B: int = 1000, seed: int = 0) -> Draws:
        """Posterior draws from the compressed rows alone, with no pass over the rows.

        Given the data, a stratum's bootstrap-weighted sum $\\sum_{a \\in h} \\xi_a v_a$
        has mean $\\sum v_a$ and covariance $\\sum v_a v_a^\\top$ when the weights
        have mean 1 and variance 1 (as Exp(1) and Poisson(1) do): the two fields of
        the compressed row. Each draw samples every stratum's sum from that normal
        distribution and refits the model exactly, recomputing the propensities,
        $\\hat\\beta$ and $\\bar x_C$. This is exactly a multiplier bootstrap with
        $N(1, 1)$ weights; it keeps the first two moments of the Bayesian bootstrap's
        weighted sums and loses their third (BUM post).

        Parameters
        ----------
        B : int
            Number of draws.
        seed : int
            Seed of the normal draws.

        A ``UserWarning`` is issued if a stratum has fewer than 200 clusters: the
        normal multipliers can be negative, and in small strata the drawn sums
        can be far from any population, which widens and skews the posterior.
        Use :meth:`ume.Experiment.bootstrap` there (decision 0019).

        Returns
        -------
        Draws
        """
        smallest = int(np.min(self.g))
        if smallest < SMALL_STRATUM:
            warnings.warn(
                f"a stratum has fewer than {SMALL_STRATUM} clusters ({smallest}); the Gaussian posterior "
                "can be too wide and skewed there. Prefer the exact engine, Experiment.bootstrap().",
                UserWarning,
                stacklevel=2,
            )
        sums = _core.gaussian_sums(self.d, self.o, self.g, self.first, self.second, B, seed)
        z, cohort, n_cohorts = self._cohorts()
        weighted = self.design.cohort is not None
        encouraged = self.features().encouraged
        raw = _core.fit_draws(self.d, self.o, sums, z, cohort, n_cohorts, weighted, encouraged, get_threads())
        return Draws(raw, self.design, "gaussian", "normal multiplier", seed)

    def itt(self) -> CompressedData:
        """The compressed rows of the intent to treat model ($T := Z$), by a linear map of each summary."""
        layout = self.design.layout
        source = np.array([layout.index(c.source) if c.treated else -1 for c in layout.columns], dtype=np.int64)
        z = np.array([float(h.Z) for h in self.strata])
        first, second = _core.compressed_intent_to_treat(self.d, self.o, self.g, self.first, self.second, z, source)
        return CompressedData(self.design, list(self.strata), self.g, first, second, self.n, None)

    # ---- storage ----------------------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str], *, keep_keys: bool = True) -> None:
        """Save the compressed dataset to a file (numpy ``.npz``, format ``ume.compressed`` version 1).

        The file holds $G_h$, the first and second moments, and JSON metadata: the
        format version, the ume version, the design (roles, $x_{ref}$, levels,
        metrics), the strata, and the number of rows. The layout is the distinct
        layout of decision 0017. Files saved from shards can be loaded and merged
        with ``+``.

        Parameters
        ----------
        path : str or path-like
            Where to write; ``.npz`` is appended if missing.
        keep_keys : bool
            Also store the cluster keys (8 bytes per cluster), so that merges of
            loaded shards can still refuse shards that share a cluster.
        """
        meta = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "ume": _core.__version__,
            "design": design_to_dict(self.design),
            "strata": [[h.Z, h.cohort] for h in self.strata],
            "n_rows": self.n,
            "layout": {"d": self.d, "o": self.o, "q": self.q},
        }
        arrays: dict[str, NDArray[Any]] = {"g": self.g, "first": self.first, "second": self.second}
        if keep_keys and self.cluster_keys is not None:
            arrays["cluster_keys"] = self.cluster_keys
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

    # ---- merging ----------------------------------------------------------------------------------

    def __add__(self, other: CompressedData) -> CompressedData:
        """Merge two shards: add the compressed rows of each stratum.

        Parameters
        ----------
        other : CompressedData
            The other shard, compressed with the same design.

        Returns
        -------
        CompressedData

        Raises
        ------
        ValueError
            If the designs differ (roles, reference points, levels, metrics), or
            if both shards contain rows of the same cluster.
        """
        if not isinstance(other, CompressedData):
            return NotImplemented
        if self.design != other.design:
            raise ValueError("cannot merge compressed data with different designs (roles, x_ref, levels or metrics)")
        keys = None
        if self.cluster_keys is not None and other.cluster_keys is not None:
            shared = np.intersect1d(self.cluster_keys, other.cluster_keys, assume_unique=True)
            if shared.size:
                raise ValueError(
                    f"{shared.size} clusters have rows in both shards; shuffle the rows by cluster before sharding"
                )
            keys = np.union1d(self.cluster_keys, other.cluster_keys)
        strata = list(dict.fromkeys([*self.strata, *other.strata]))
        g = np.zeros(len(strata))
        first = np.zeros((len(strata), self.q))
        second = np.zeros((len(strata), self.q, self.q))
        for part in (self, other):
            for i, h in enumerate(part.strata):
                j = strata.index(h)
                g[j] += part.g[i]
                first[j] += part.first[i]
                second[j] += part.second[i]
        return CompressedData(self.design, strata, g, first, second, self.n + other.n, keys)

    # ---- printing ---------------------------------------------------------------------------------

    def table(self) -> Table:
        """One line per stratum: instrument, cohort, clusters, and the cohort's propensity."""
        e = self.e_hat()
        rows = [[h.Z, h.cohort if h.cohort is not None else "", int(g), e.get(h.cohort, float("nan"))]
                for h, g in zip(self.strata, self.g, strict=True)]  # fmt: skip
        cohort = self.design.cohort or "cohort"
        return Table(["Z", cohort, "G_h", "e_hat"], rows)

    def __repr__(self) -> str:
        f = self.features()
        header = (
            f"<ume.CompressedData: {len(self)} strata from {self.n:,} rows in {self.n_clusters:,} clusters>\n"
            f"  layout   distinct, q = {self.q} (d = {self.d}, o = {self.o}); "
            f"{self.q * (self.q + 3) // 2 + 1:,} numbers per stratum\n"
            f"  design   {f.describe()}"
        )
        return header + "\n" + repr(self.table())

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(repr(self))}</pre>"


FORMAT = "ume.compressed"
FORMAT_VERSION = 1


def load(path: str | os.PathLike[str]) -> CompressedData:
    """Load a compressed dataset saved with :meth:`CompressedData.save`.

    Parameters
    ----------
    path : str or path-like
        The ``.npz`` file.

    Returns
    -------
    CompressedData

    Raises
    ------
    ValueError
        If the file is not a ume compressed dataset, or its format version is newer
        than this ume understands.
    """
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a ume compressed dataset")
        if meta["version"] > FORMAT_VERSION:
            raise ValueError(f"{path} has format version {meta['version']}; this ume reads up to {FORMAT_VERSION}")
        keys = f["cluster_keys"] if "cluster_keys" in f.files else None
        strata = [Stratum(int(z), cohort) for z, cohort in meta["strata"]]
        return CompressedData(
            design_from_dict(meta["design"]), strata, f["g"], f["first"], f["second"], meta["n_rows"], keys
        )


class _Pass:
    """Feeds row batches to a Rust pass: validates each batch, builds its arrays, assigns strata."""

    @staticmethod
    def _require_streamable(design: Design) -> None:
        if not design.metrics:
            raise ValueError("declare the metrics before compressing")
        if not design.covariates.resolved():
            raise ValueError(
                "a streaming pass needs every x_ref and categorical level declared before the pass; "
                "use Experiment.compress() to infer them from data in memory"
            )

    def __init__(self, design: Design, sparse: SparseOption, core: Any) -> None:
        self.design = design
        self.sparse = sparse
        self._check_levels = True
        self._strata: dict[Stratum, int] = {}
        self._encouraged = False
        self._core = core

    def update(self, frame: Any) -> Any:
        """Add a batch of rows (a pandas or polars frame) and return the pass."""
        nw_frame = nw.from_native(frame, eager_only=True)
        missing = [c for c in self.design.columns if c not in nw_frame.columns]
        if missing:
            raise ValueError(f"columns not found: {missing}")
        if self._check_levels:
            resolve_covariates(nw_frame, self.design.covariates)  # rejects undeclared levels
        z, t, y = batch_arrays(nw_frame, self.design)
        self._encouraged |= bool(np.any(t != z))
        keys = cluster_keys(_column(nw_frame, self.design.cluster))
        strata = self._stratum_index(nw_frame, z)
        r = design_matrix(nw_frame, self.design.covariates, self.sparse)
        if isinstance(r, CSR):
            self._core.update_sparse(r.indptr, r.indices, r.data, r.shape[1], t, y, keys, strata)
        else:
            self._core.update_dense(r, t, y, keys, strata)
        return self

    def _stratum_index(self, frame: Any, z: NDArray[np.float64]) -> NDArray[np.uint32]:
        """The stratum index of each row, registering strata the first time they appear."""
        if self.design.cohort is None:
            labels: list[Any] = [None]
            codes = np.zeros(z.size, dtype=np.int64)
        else:
            column = frame[self.design.cohort]
            labels = sorted(column.unique().to_list())  # hashed, then a small sort
            codes = np.searchsorted(np.asarray(labels), column.to_numpy())
        combo = z.astype(np.int64) * len(labels) + codes
        present = np.flatnonzero(np.bincount(combo, minlength=2 * len(labels)))  # O(n), no sort
        table = np.zeros(2 * len(labels), dtype=np.uint32)
        for c in present:
            h = Stratum(int(c) // len(labels), labels[int(c) % len(labels)])
            table[c] = self._strata.setdefault(h, len(self._strata))
        index: NDArray[np.uint32] = table[combo]
        return index

    def _strata_keys(self) -> tuple[list[Stratum], NDArray[np.float64], NDArray[np.uint32], int]:
        """Strata in index order, with their instrument, cohort index and the number of cohorts."""
        strata = sorted(self._strata, key=self._strata.__getitem__)
        labels = sorted({h.cohort for h in strata}, key=str)
        z = np.array([float(h.Z) for h in strata])
        cohort = np.array([labels.index(h.cohort) for h in strata], dtype=np.uint32)
        return strata, z, cohort, len(labels)


class _CorePass(_Pass):
    """A driver for any Rust pass object (first moments, scores)."""

    def __init__(self, design: Design, sparse: SparseOption, core: Any, strata: dict[Stratum, int] | None = None):
        self._require_streamable(design)
        super().__init__(design, sparse, core)
        if strata is not None:
            self._strata = dict(strata)


class Compressor(_Pass):
    """Compress rows in one pass, batch by batch.

    Use it when the rows do not fit in memory, or arrive in shards: feed each
    batch with :meth:`update`, then call :meth:`finish`. Rows of a cluster must
    be contiguous (possibly across batches). Compress shards separately and add
    the results with ``+``.

    Parameters
    ----------
    design : Design
        The design, with metrics, every numeric reference point and every
        categorical level declared (they must be fixed before the pass).
    sparse : bool or "auto"
        Build each batch's design matrix dense or sparse.

    Examples
    --------
    >>> import ume
    >>> from ume.testing import simulate_accounts
    >>> df = simulate_accounts(3000, seed=1)
    >>> design = ume.annotate(df, cluster="account", instrument="Z", treatment="T",
    ...     covariates=ume.covariates(x1=2.0)).metrics(ume.mean("y")).design
    >>> c = ume.Compressor(design)
    >>> for batch in (df[:2000], df[2000:]):
    ...     c = c.update(batch)
    >>> cd = c.finish()
    """

    def __init__(self, design: Design, sparse: SparseOption = "auto") -> None:
        self._require_streamable(design)
        super().__init__(design, sparse, _core.Compressor(design.covariates.k + 1, design.layout.o, get_threads()))

    def finish(self) -> CompressedData:
        """Flush the last cluster and return the compressed dataset."""
        out = self._core.finish()
        strata = sorted(self._strata, key=self._strata.__getitem__)
        d, o = self.design.covariates.k + 1, self.design.layout.o
        q = d * o + d * (d + 1)
        n, k = len(strata), out["g"].size
        g, first, second = np.zeros(n), np.zeros((n, q)), np.zeros((n, q, q))
        g[:k], first[:k], second[:k] = out["g"], out["first"], out["second"]
        return CompressedData(self.design, strata, g, first, second, int(out["n_rows"]), out["keys"])


class BootstrapPass(_Pass):
    """The exact bootstrap: one pass over the rows that builds the weighted sums of $B$ draws.

    Made by :meth:`ume.Experiment.bootstrap`.
    """

    def __init__(
        self, design: Design, B: int, weights: str, seed: int, n_clusters: int | None, sparse: SparseOption = "auto"
    ) -> None:
        self._require_streamable(design)
        d, o = design.covariates.k + 1, design.layout.o
        super().__init__(design, sparse, _core.BootstrapCompressor(d, o, B, seed, weights, n_clusters, get_threads()))
        self.B, self.weights, self.seed = B, weights, seed

    def finish(self) -> DrawSums:
        """Flush the last cluster and return the draws' weighted sums (fit them with ``.draws()``)."""
        out = self._core.finish()
        strata = sorted(self._strata, key=self._strata.__getitem__)
        return DrawSums(
            self.design, strata, out["sums"], int(out["n_rows"]), self.weights, self.seed,
            self._encouraged, out["keys"],
        )  # fmt: skip


@dataclasses.dataclass(frozen=True)
class _PointSummary:
    """What an effects table shows about a point estimate made from first moments."""

    design: Design
    features: DesignFeatures
    x_C: NDArray[np.float64]
    pi_C: float

    @property
    def x_labels(self) -> list[str]:
        return self.design.covariates.labels()


def _feed(driver: _Pass, source: Source) -> None:
    """Stream every batch of `source` through a pass driver."""
    driver._check_levels = source.check_levels
    for batch in source.batches():
        driver.update(batch)


def early_effects(
    rows: Any,
    design: Design,
    effects: tuple[AnyEffect, ...],
    at: Mapping[str, Any] | None,
    se: str,
    sparse: SparseOption = "auto",
    **source_options: Any,
) -> EffectTable:
    """Effects with standard errors in two passes and no second moments (delta vectors early)."""
    if se not in ("sandwich", "linearized"):
        raise ValueError("se must be 'sandwich' or 'linearized'")
    source = open_source(rows, design, **source_options)
    if not source.reiterable:
        raise ValueError("two passes need a source that can be read twice: a frame, a lazy frame, or a function")
    resolved = source.design
    d, o = resolved.covariates.k + 1, resolved.layout.o
    first = _CorePass(resolved, sparse, _core.FirstMomentsPass(d, o, get_threads()))
    _feed(first, source)
    out = first._core.finish()
    sums = np.ascontiguousarray(out["sums"])
    _, z, cohort, n_cohorts = first._strata_keys()
    weighted = resolved.cohort is not None
    fit = _core.fit_first_moments(d, o, sums, z, cohort, n_cohorts, weighted, first._encouraged)
    chosen = list(effects) or default_effects(resolved.metrics)
    x0s = [point(resolved.covariates, np.asarray(fit["x_c"]), e.at if e.at is not None else at) for e in chosen]
    values_grads = [value_and_gradient(e, fit["beta"], x0, resolved.layout) for e, x0 in zip(chosen, x0s, strict=True)]
    if se == "sandwich":
        grads = np.ascontiguousarray(np.array([g for _, g in values_grads]))
        projections = _core.sandwich_projections(d, o, fit["beta"], fit["a"], grads, fit["w"], z)
    else:
        grads_h = stratum_gradients(chosen, resolved, sums, z, cohort, n_cohorts, first._encouraged, at)
        projections = np.ascontiguousarray(np.transpose(grads_h, (1, 0, 2)))
    second = _CorePass(resolved, sparse, _core.ScorePass(d, o, projections, get_threads()), first._strata)
    _feed(second, source)
    sd = np.sqrt(second._core.finish())
    e_hat = np.asarray(fit["e_hat"])
    features = DesignFeatures(
        encouraged=first._encouraged,
        clustered=int(out["n_rows"]) > int(sums[:, 0].sum()),
        adaptive=bool(e_hat.size > 1 and np.ptp(e_hat) > 1e-12),
        n_cohorts=max(int(e_hat.size), 1),
        weighted=weighted,
        ratio=any(m.kind == "ratio" for m in resolved.metrics),
        k=resolved.covariates.k,
    )
    summary = _PointSummary(
        resolved, features, np.asarray(fit["x_c"]) + resolved.covariates.x_ref(), float(fit["pi_c"])
    )
    rows = []
    for e, (value, _), s in zip(chosen, values_grads, sd, strict=True):
        where = e.at if e.at is not None else at
        label = "compliers" if not where else ", ".join(f"{k}={v}" for k, v in where.items())
        rows.append([e.name, e.contrast, label, value, float(s), value - Z_975 * s, value + Z_975 * s])
    method = ("sandwich (" + features.covariance + ")" if se == "sandwich" else "linearized") + ", two passes"
    return EffectTable(rows, summary, features.estimand, method)  # type: ignore[arg-type]


def effects(
    rows: Any,
    design: Design,
    *effects: AnyEffect,
    at: Mapping[str, Any] | None = None,
    se: str = "sandwich",
    sparse: SparseOption = "auto",
    **source_options: Any,
) -> EffectTable:
    """Effects and standard errors of rows in two passes, without second moments.

    The streaming form of :meth:`ume.Experiment.effects` (delta vectors early).

    Parameters
    ----------
    rows : DataFrame, LazyFrame, or a function returning an iterable of frames
        The rows. They are read twice, so a plain iterator is not enough: pass a
        function that returns fresh batches.
    design : Design
        The roles and metrics.
    *effects : Effect or CustomEffect
        Defaults to the difference and the relative effect of every metric.
    at : mapping, optional
        A segment to read the effects at.
    se : {"sandwich", "linearized"}
        The standard error.
    sparse : bool or "auto"
        Dense or sparse design matrix in each batch.
    **source_options
        ``batch_size`` and ``presorted``, as in :func:`ume.compress`.

    Returns
    -------
    EffectTable
    """
    return early_effects(rows, design, effects, at, se, sparse, **source_options)


def compress(rows: Any, design: Design, sparse: SparseOption = "auto", **source_options: Any) -> CompressedData:
    """Compress rows (a frame, a lazy frame, or batches) in one pass.

    Parameters
    ----------
    rows : DataFrame, LazyFrame, iterable of frames, or a function returning one
        The rows. A frame in memory is sorted by cluster if needed; a lazy frame
        is sorted lazily unless ``presorted=True``; batches must already be grouped
        by cluster, and the design fully declared.
    design : Design
        The roles and metrics (from :func:`ume.design` or ``Experiment.design``).
    sparse : bool or "auto"
        Dense or sparse design matrix in each batch.
    **source_options
        ``batch_size`` (rows per batch of a lazy frame) and ``presorted``.

    Returns
    -------
    CompressedData
    """
    source = open_source(rows, design, **source_options)
    compressor = Compressor(source.design, sparse)
    _feed(compressor, source)
    return compressor.finish()


def bootstrap(
    rows: Any,
    design: Design,
    B: int = 1000,
    weights: str = "bayesian",
    seed: int = 0,
    sparse: SparseOption = "auto",
    **source_options: Any,
) -> Draws:
    """The exact bootstrap of rows (a frame, a lazy frame, or batches): one pass for all draws.

    The streaming form of :meth:`ume.Experiment.bootstrap`. The multinomial
    (classical) bootstrap needs the number of clusters before the pass, so it
    needs a frame or a lazy frame. For shards processed separately, use
    :func:`ume.bootstrap_sums`, merge, then fit.

    Parameters
    ----------
    rows : DataFrame, LazyFrame, iterable of frames, or a function returning one
        The rows, as in :func:`ume.compress`.
    design : Design
        The roles and metrics.
    B : int
        Number of draws.
    weights : {"bayesian", "poisson", "multinomial"}
        The weight scheme: Exp(1), Poisson(1), or multinomial counts over the clusters.
    seed : int
        Seed of the weight streams.
    sparse : bool or "auto"
        Dense or sparse design matrix in each batch.
    **source_options
        ``batch_size`` and ``presorted``, as in :func:`ume.compress`.

    Returns
    -------
    Draws
    """
    return bootstrap_sums(rows, design, B, weights, seed, sparse, **source_options).draws()


def bootstrap_sums(
    rows: Any,
    design: Design,
    B: int = 1000,
    weights: str = "bayesian",
    seed: int = 0,
    sparse: SparseOption = "auto",
    **source_options: Any,
) -> DrawSums:
    """The exact bootstrap's weighted sums of one shard of rows, before fitting.

    Process each shard with the same design, weights, seed and ``B``; add the
    results with ``+``; then call ``.draws()``. Each cluster's weights depend only
    on (seed, cluster, draw), so the merged sums equal one pass over all rows.
    Only independent weights (``"bayesian"``, ``"poisson"``) can be merged.

    Parameters
    ----------
    rows : DataFrame, LazyFrame, iterable of frames, or a function returning one
        One shard's rows, as in :func:`ume.compress`.
    design : Design
        The roles and metrics, the same for every shard.
    B : int
        Number of draws, the same for every shard.
    weights : {"bayesian", "poisson", "multinomial"}
        The weight scheme, the same for every shard.
    seed : int
        Seed of the weight streams, the same for every shard.
    sparse : bool or "auto"
        Dense or sparse design matrix in each batch.
    **source_options
        ``batch_size`` and ``presorted``, as in :func:`ume.compress`.

    Returns
    -------
    DrawSums
    """
    source = open_source(rows, design, **source_options)
    n_clusters = None
    if weights == "multinomial":
        if source.n_clusters is None:
            raise ValueError("the multinomial bootstrap needs the number of clusters: pass a frame or a lazy frame")
        n_clusters = source.n_clusters()
    boot = BootstrapPass(source.design, B, weights, seed, n_clusters, sparse)
    _feed(boot, source)
    return boot.finish()


__all__ = [
    "BootstrapPass",
    "CompressedData",
    "CompressedRow",
    "Compressor",
    "Stratum",
    "bootstrap",
    "bootstrap_sums",
    "compress",
    "early_effects",
    "effects",
    "load",
]
