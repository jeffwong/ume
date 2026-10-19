# Type stubs for the Rust extension module ``ume._core``.
#
# Signatures only. Docstrings live in the Rust `///` comments
# (crates/ume-python/src/lib.rs), which PyO3 exposes as ``__doc__``;
# quartodoc reads them from there. Do not copy them here.

from typing import Any

import numpy as np
from numpy.typing import NDArray

__version__: str

def fit_rows_dense(
    r: NDArray[np.float64],
    z: NDArray[np.float64],
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    cluster: NDArray[np.uint32],
    n_clusters: int,
) -> dict[str, Any]: ...
def fit_rows_sparse(
    indptr: NDArray[np.uintp],
    indices: NDArray[np.uintp],
    data: NDArray[np.float64],
    ncols: int,
    z: NDArray[np.float64],
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    cluster: NDArray[np.uint32],
    n_clusters: int,
) -> dict[str, Any]: ...
def inverse_propensity(
    z: NDArray[np.float64],
    cohort: NDArray[np.uint32],
    xi: NDArray[np.float64],
    n_cohorts: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def fit_compressed(
    d: int,
    o: int,
    g: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    z: NDArray[np.float64],
    cohort: NDArray[np.uint32],
    n_cohorts: int,
    weighted: bool,
) -> dict[str, Any]: ...
def compressed_intent_to_treat(
    d: int,
    o: int,
    g: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    z: NDArray[np.float64],
    treated_source: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def summary_layout(d: int, o: int) -> list[tuple[str, int, int]]: ...

class Compressor:
    def __init__(self, d: int, o: int, threads: int = 1) -> None: ...
    def update_dense(
        self,
        r: NDArray[np.float64],
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        keys: NDArray[np.uint64],
        strata: NDArray[np.uint32],
    ) -> None: ...
    def update_sparse(
        self,
        indptr: NDArray[np.uintp],
        indices: NDArray[np.uintp],
        data: NDArray[np.float64],
        ncols: int,
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        keys: NDArray[np.uint64],
        strata: NDArray[np.uint32],
    ) -> None: ...
    def finish(self) -> dict[str, Any]: ...

def bootstrap_weights(keys: NDArray[np.uint64], b: int, seed: int, weights: str) -> NDArray[np.float64]: ...
def gaussian_sums(
    d: int,
    o: int,
    g: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    b: int,
    seed: int,
) -> NDArray[np.float64]: ...
def fit_draws(
    d: int,
    o: int,
    sums: NDArray[np.float64],
    z: NDArray[np.float64],
    cohort: NDArray[np.uint32],
    n_cohorts: int,
    weighted: bool,
    encouraged: bool,
    threads: int = 1,
) -> dict[str, Any]: ...

class BootstrapCompressor:
    def __init__(
        self, d: int, o: int, b: int, seed: int, weights: str, n_clusters: int | None = None, threads: int = 1
    ) -> None: ...
    def update_dense(
        self,
        r: NDArray[np.float64],
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        keys: NDArray[np.uint64],
        strata: NDArray[np.uint32],
    ) -> None: ...
    def update_sparse(
        self,
        indptr: NDArray[np.uintp],
        indices: NDArray[np.uintp],
        data: NDArray[np.float64],
        ncols: int,
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        keys: NDArray[np.uint64],
        strata: NDArray[np.uint32],
    ) -> None: ...
    def finish(self) -> dict[str, Any]: ...

def fit_first_moments(
    d: int,
    o: int,
    sums: NDArray[np.float64],
    z: NDArray[np.float64],
    cohort: NDArray[np.uint32],
    n_cohorts: int,
    weighted: bool,
    encouraged: bool,
) -> dict[str, Any]: ...
def sandwich_projections(
    d: int,
    o: int,
    beta: NDArray[np.float64],
    a: NDArray[np.float64],
    grads: NDArray[np.float64],
    w: NDArray[np.float64],
    z: NDArray[np.float64],
) -> NDArray[np.float64]: ...

class FirstMomentsPass:
    def __init__(self, d: int, o: int, threads: int = 1) -> None: ...
    def update_dense(self, r: Any, t: Any, y: Any, keys: Any, strata: Any) -> None: ...
    def update_sparse(
        self, indptr: Any, indices: Any, data: Any, ncols: int, t: Any, y: Any, keys: Any, strata: Any
    ) -> None: ...
    def finish(self) -> dict[str, Any]: ...

class ScorePass:
    def __init__(self, d: int, o: int, projections: NDArray[np.float64], threads: int = 1) -> None: ...
    def update_dense(self, r: Any, t: Any, y: Any, keys: Any, strata: Any) -> None: ...
    def update_sparse(
        self, indptr: Any, indices: Any, data: Any, ncols: int, t: Any, y: Any, keys: Any, strata: Any
    ) -> None: ...
    def finish(self) -> NDArray[np.float64]: ...
