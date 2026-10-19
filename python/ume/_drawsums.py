"""The exact bootstrap's weighted sums before fitting: mergeable across shards."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ume import _core
from ume._config import get_threads
from ume._design import Design
from ume._draws import Draws

#: Schemes whose sums add across shards: independent weights.
MERGEABLE = ("bayesian", "poisson")


class DrawSums:
    """The weighted sums of $B$ exact bootstrap draws, per stratum, before fitting.

    Row $b$ of stratum $h$ is $\\sum_{a \\in h} \\xi^{(b)}_a v_a$. With independent
    weights (Bayesian, Poisson), each cluster's weights depend only on (seed,
    cluster, draw), so shards processed separately produce sums that add: merge
    them with ``+``, then fit every draw once with :meth:`draws`. Fitting must come
    after merging, because a draw's propensities use the total weight of each
    whole cohort.

    Made by :func:`ume.bootstrap_sums`, rather than built directly.

    Parameters
    ----------
    design : Design
        The resolved design.
    strata : list of Stratum
        The strata, one per block of sums.
    sums : numpy.ndarray
        Shape $(H, B, 1 + q)$: row $b$ of stratum $h$ is $\\sum_{a \\in h} \\xi^{(b)}_a v_a$.
    n_rows : int
        Number of rows read.
    weights : str
        The weight scheme: ``"bayesian"``, ``"poisson"`` or ``"multinomial"``.
    seed : int
        Seed of the weight streams.
    encouraged : bool
        Whether some row had $T \\ne Z$ (it decides how $\\bar x_C$ is computed).
    cluster_keys : numpy.ndarray, optional
        Sorted keys of the clusters read, to refuse shards that share a cluster.
    """

    def __init__(
        self,
        design: Design,
        strata: list[Any],
        sums: NDArray[np.float64],
        n_rows: int,
        weights: str,
        seed: int,
        encouraged: bool,
        cluster_keys: NDArray[np.uint64] | None,
    ) -> None:
        self.design = design
        self.strata = list(strata)
        self.sums = np.ascontiguousarray(sums)
        self.n_rows = int(n_rows)
        self.weights = weights
        self.seed = seed
        self.encouraged = encouraged
        self.cluster_keys = cluster_keys

    @property
    def B(self) -> int:
        """Number of draws."""
        return int(self.sums.shape[1])

    def __add__(self, other: DrawSums) -> DrawSums:
        """Merge the sums of two shards.

        Parameters
        ----------
        other : DrawSums
            The other shard's sums.

        Returns
        -------
        DrawSums

        Raises
        ------
        ValueError
            If the designs, weights, seeds or numbers of draws differ, if the
            weights are multinomial (their counts are coupled across shards), or
            if both shards contain rows of the same cluster.
        """
        if not isinstance(other, DrawSums):
            return NotImplemented
        if self.weights not in MERGEABLE:
            raise ValueError("multinomial bootstrap sums cannot be merged: the counts are coupled across clusters")
        if (self.design, self.weights, self.seed, self.B) != (other.design, other.weights, other.seed, other.B):
            raise ValueError("cannot merge draw sums with different designs, weights, seeds or numbers of draws")
        keys = None
        if self.cluster_keys is not None and other.cluster_keys is not None:
            shared = np.intersect1d(self.cluster_keys, other.cluster_keys, assume_unique=True)
            if shared.size:
                raise ValueError(f"{shared.size} clusters have rows in both shards")
            keys = np.union1d(self.cluster_keys, other.cluster_keys)
        strata = list(dict.fromkeys([*self.strata, *other.strata]))
        sums = np.zeros((len(strata), self.B, self.sums.shape[2]))
        for part in (self, other):
            for i, h in enumerate(part.strata):
                sums[strata.index(h)] += part.sums[i]
        return DrawSums(
            self.design, strata, sums, self.n_rows + other.n_rows, self.weights, self.seed,
            self.encouraged or other.encouraged, keys,
        )  # fmt: skip

    def draws(self) -> Draws:
        """Fit every draw: recompute its propensities, $\\hat\\beta$ and $\\bar x_C$ from its sums."""
        labels = sorted({h.cohort for h in self.strata}, key=str)
        z = np.array([float(h.Z) for h in self.strata])
        cohort = np.array([labels.index(h.cohort) for h in self.strata], dtype=np.uint32)
        d, o = self.design.covariates.k + 1, self.design.layout.o
        weighted = self.design.cohort is not None
        raw = _core.fit_draws(d, o, self.sums, z, cohort, len(labels), weighted, self.encouraged, get_threads())
        return Draws(raw, self.design, "exact", self.weights, self.seed)

    def __repr__(self) -> str:
        return (
            f"<ume.DrawSums: {self.B:,} {self.weights} draws, {len(self.strata)} strata, "
            f"{self.n_rows:,} rows; merge shards with +, then .draws()>"
        )
