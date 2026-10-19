//! Python bindings for `ume-core`, compiled into the `ume._core` extension module.
//!
//! Keep this layer thin: convert arrays, release the GIL, call into `ume-core`,
//! and map errors. The dense and sparse entry points call the same generic engine.

use faer::{Mat, MatRef};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use ume_core::bootstrap::{
    BootstrapCompressor, DrawsFold, StratumKey, fit_draws as core_fit_draws, gaussian_sums as core_gaussian_sums,
};
use ume_core::compress::{Compressed, Compressor, Fold, Merge, MomentsFold, StratumMoments};
use ume_core::early::{FirstMomentsFold, FirstMomentsPass, ScoreFold, ScorePass, sandwich_projection};
use ume_core::fit::{
    Fit, RowData, Stage, Stratum, first_stage_f, fit, fit_compressed as core_fit_compressed, intent_to_treat,
};
use ume_core::fit::{StratumSums, point, strata_blocks};
use ume_core::layout::{Block, Layout};
use ume_core::parallel::ParallelPass;
use ume_core::propensity::inverse_propensity as core_inverse_propensity;
use ume_core::rows::{Dense, RowMatrix, Sparse};
use ume_core::sandwich::Scoped;
use ume_core::weights::{Scheme, Weights};

fn to_py_err(e: ume_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn mat_to_numpy<'py>(py: Python<'py>, m: &Mat<f64>) -> Bound<'py, PyAny> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
        .into_pyarray(py)
        .into_any()
}

fn vec_to_numpy<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyAny> {
    Array1::from(v.to_vec()).into_pyarray(py).into_any()
}

fn stage_to_dict<'py>(py: Python<'py>, s: &Stage) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("coef", mat_to_numpy(py, &s.coef))?;
    out.set_item("a", mat_to_numpy(py, &s.a))?;
    match &s.cov {
        Scoped::Joint(m) => out.set_item("cov", mat_to_numpy(py, m))?,
        Scoped::PerColumn(blocks) => {
            let list: Vec<Bound<'py, PyAny>> = blocks.iter().map(|b| mat_to_numpy(py, b)).collect();
            out.set_item("cov_blocks", list)?;
        }
    }
    Ok(out)
}

fn fit_to_dict<'py>(py: Python<'py>, f: &Fit) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (name, b) in [("blocks_all", &f.blocks_all), ("blocks_z", &f.blocks_z)] {
        let blocks = PyDict::new(py);
        blocks.set_item("s_y", mat_to_numpy(py, &b.s_y))?;
        blocks.set_item("s", mat_to_numpy(py, &b.s))?;
        blocks.set_item("s_t", mat_to_numpy(py, &b.s_t))?;
        out.set_item(name, blocks)?;
    }
    out.set_item("second", stage_to_dict(py, &f.second)?)?;
    match &f.first {
        Some(first) => {
            let d = stage_to_dict(py, first)?;
            d.set_item("f", vec_to_numpy(py, &first_stage_f(first)))?;
            out.set_item("first", d)?;
        }
        None => out.set_item("first", py.None())?,
    }
    let c = &f.compliers;
    let compliers = PyDict::new(py);
    compliers.set_item("take_up_z1", c.take_up_z1)?;
    compliers.set_item("take_up_z0", c.take_up_z0)?;
    compliers.set_item("pi_c", c.pi_c)?;
    compliers.set_item("x_c", vec_to_numpy(py, &c.x_c))?;
    compliers.set_item("x_mean", vec_to_numpy(py, &c.x_mean))?;
    out.set_item("compliers", compliers)?;
    out.set_item("encouraged", f.encouraged)?;
    Ok(out)
}

/// Inputs shared by both entry points, borrowed from numpy.
struct Columns<'a> {
    z: &'a [f64],
    t: &'a [f64],
    y: MatRef<'a, f64>,
    w: &'a [f64],
    cluster: &'a [u32],
    n_clusters: usize,
}

fn run<R: RowMatrix + Sync>(py: Python<'_>, r: &R, c: &Columns<'_>) -> PyResult<Fit> {
    let n = r.nrows();
    let lengths = [c.z.len(), c.t.len(), c.y.nrows(), c.w.len(), c.cluster.len()];
    if lengths.iter().any(|&len| len != n) {
        return Err(PyValueError::new_err(format!(
            "every input must have {n} rows, got {lengths:?}"
        )));
    }
    if c.cluster.iter().any(|&g| g as usize >= c.n_clusters) {
        return Err(PyValueError::new_err("cluster index out of range"));
    }
    let data = RowData {
        r,
        z: c.z,
        t: c.t,
        y: c.y,
        w: c.w,
        cluster: c.cluster,
        n_clusters: c.n_clusters,
    };
    py.detach(|| fit(&data)).map_err(to_py_err)
}

/// Fits the universal model on rows, with a dense design matrix.
///
/// Low-level kernel behind ``Experiment.fit``. All arrays must be C-contiguous.
///
/// Parameters
/// ----------
/// r : numpy.ndarray
///     Design matrix ``(1, x - x_ref)``, float64, shape ``(n, d)``.
/// z, t : numpy.ndarray
///     Instrument and treatment received, float64, shape ``(n,)``.
/// y : numpy.ndarray
///     Outcome columns, float64, shape ``(n, o)``.
/// w : numpy.ndarray
///     Row weights (inverse propensity times bootstrap weight), float64, shape ``(n,)``.
/// cluster : numpy.ndarray
///     Cluster index of each row in ``0..n_clusters``, uint32, shape ``(n,)``.
/// n_clusters : int
///     Number of clusters.
///
/// Returns
/// -------
/// dict
///     ``blocks_all``, ``blocks_z``; ``second`` with ``coef``, ``a`` and the joint
///     ``cov``; ``first`` with ``coef``, ``a``, per-equation ``cov_blocks`` and the
///     robust ``f`` statistics, or None without encouragement; ``compliers`` and
///     ``encouraged``.
///
/// Raises
/// ------
/// ValueError
///     If the shapes disagree, or the normal equations are singular.
#[pyfunction]
#[pyo3(signature = (r, z, t, y, w, cluster, n_clusters))]
#[allow(clippy::too_many_arguments)]
fn fit_rows_dense<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    cluster: PyReadonlyArray1<'py, u32>,
    n_clusters: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (n, d) = (r.shape()[0], r.shape()[1]);
    let o = y.shape()[1];
    let rm = Dense::from_row_major(r.as_slice()?, n, d);
    let c = Columns {
        z: z.as_slice()?,
        t: t.as_slice()?,
        y: MatRef::from_row_major_slice(y.as_slice()?, y.shape()[0], o),
        w: w.as_slice()?,
        cluster: cluster.as_slice()?,
        n_clusters,
    };
    let f = run(py, &rm, &c)?;
    fit_to_dict(py, &f)
}

/// Fits the universal model on rows, with a sparse (CSR) design matrix.
///
/// Same as ``fit_rows_dense``, with ``r`` given as CSR arrays. Use it when
/// categorical covariates with many levels make ``r`` mostly zeros.
///
/// Parameters
/// ----------
/// indptr, indices : numpy.ndarray
///     CSR row pointers (length ``n + 1``) and column indices, uint64 (``usize``).
/// data : numpy.ndarray
///     CSR values, float64.
/// ncols : int
///     Number of columns ``d`` of the design matrix.
/// z, t, y, w, cluster, n_clusters
///     As in ``fit_rows_dense``.
///
/// Returns
/// -------
/// dict
///     As in ``fit_rows_dense``.
///
/// Raises
/// ------
/// ValueError
///     If the CSR arrays or the shapes are invalid, or the normal equations are singular.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, ncols, z, t, y, w, cluster, n_clusters))]
#[allow(clippy::too_many_arguments)]
fn fit_rows_sparse<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, usize>,
    indices: PyReadonlyArray1<'py, usize>,
    data: PyReadonlyArray1<'py, f64>,
    ncols: usize,
    z: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    cluster: PyReadonlyArray1<'py, u32>,
    n_clusters: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (row_ptr, col_idx, values) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let n = row_ptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| PyValueError::new_err("indptr must not be empty"))?;
    let valid = row_ptr[0] == 0
        && row_ptr.windows(2).all(|p| p[0] <= p[1])
        && row_ptr[n] == col_idx.len()
        && col_idx.len() == values.len()
        && col_idx.iter().all(|&j| j < ncols);
    if !valid {
        return Err(PyValueError::new_err("invalid CSR arrays"));
    }
    let rm = Sparse::from_csr(n, ncols, row_ptr, col_idx, values);
    let o = y.shape()[1];
    let c = Columns {
        z: z.as_slice()?,
        t: t.as_slice()?,
        y: MatRef::from_row_major_slice(y.as_slice()?, y.shape()[0], o),
        w: w.as_slice()?,
        cluster: cluster.as_slice()?,
        n_clusters,
    };
    let f = run(py, &rm, &c)?;
    fit_to_dict(py, &f)
}

/// Cohort propensities and inverse propensity weights, per cluster.
///
/// Parameters
/// ----------
/// z : numpy.ndarray
///     Instrument of each cluster, float64 (0 or 1).
/// cohort : numpy.ndarray
///     Cohort index of each cluster in ``0..n_cohorts``, uint32.
/// xi : numpy.ndarray
///     Weight of each cluster (1 for the point estimate), float64.
/// n_cohorts : int
///     Number of cohorts.
///
/// Returns
/// -------
/// tuple of numpy.ndarray
///     ``e_hat`` per cohort (the weighted share of its clusters with ``Z = 1``)
///     and ``w`` per cluster (``1 / e_hat`` if ``Z = 1``, ``1 / (1 - e_hat)`` if not).
///
/// Raises
/// ------
/// ValueError
///     If a cohort has a propensity of 0 or 1.
#[pyfunction]
#[pyo3(signature = (z, cohort, xi, n_cohorts))]
fn inverse_propensity<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    cohort: PyReadonlyArray1<'py, u32>,
    xi: PyReadonlyArray1<'py, f64>,
    n_cohorts: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (z, cohort, xi) = (z.as_slice()?, cohort.as_slice()?, xi.as_slice()?);
    if z.len() != cohort.len() || z.len() != xi.len() || cohort.iter().any(|&c| c as usize >= n_cohorts) {
        return Err(PyValueError::new_err(
            "z, cohort and xi must have one entry per cluster, cohorts in range",
        ));
    }
    let p = core_inverse_propensity(z, cohort, xi, n_cohorts).map_err(to_py_err)?;
    Ok((vec_to_numpy(py, &p.e_hat), vec_to_numpy(py, &p.w)))
}

// ---- compression --------------------------------------------------------------------------------

/// One pass over rows grouped by cluster, to compressed rows per stratum.
///
/// Feed batches with ``update_dense`` or ``update_sparse``; rows of a cluster
/// must be contiguous, possibly across batches. ``finish`` returns the
/// compressed rows. Low-level kernel behind ``Experiment.compress`` and
/// ``ume.Compressor``.
///
/// Parameters
/// ----------
/// d : int
///     Columns of the design matrix.
/// o : int
///     Outcome columns.
/// threads : int
///     Worker threads; clusters are routed to workers by key hash and merged at the end.
macro_rules! pass_class {
    ($ty:ty, { $($own:tt)* }) => {
        #[pymethods]
        impl $ty {
            $($own)*

            /// Adds a batch of rows with a dense design matrix (see ``Compressor.update_dense``).
            #[pyo3(signature = (r, t, y, keys, strata))]
            fn update_dense(
                &mut self,
                py: Python<'_>,
                r: PyReadonlyArray2<'_, f64>,
                t: PyReadonlyArray1<'_, f64>,
                y: PyReadonlyArray2<'_, f64>,
                keys: PyReadonlyArray1<'_, u64>,
                strata: PyReadonlyArray1<'_, u32>,
            ) -> PyResult<()> {
                let state = self
                    .inner
                    .as_mut()
                    .ok_or_else(|| PyValueError::new_err("this pass is finished"))?;
                pass_dense(py, state, r, t, y, keys, strata)
            }

            /// Adds a batch of rows with a sparse design matrix (see ``Compressor.update_sparse``).
            #[pyo3(signature = (indptr, indices, data, ncols, t, y, keys, strata))]
            #[allow(clippy::too_many_arguments)]
            fn update_sparse(
                &mut self,
                py: Python<'_>,
                indptr: PyReadonlyArray1<'_, usize>,
                indices: PyReadonlyArray1<'_, usize>,
                data: PyReadonlyArray1<'_, f64>,
                ncols: usize,
                t: PyReadonlyArray1<'_, f64>,
                y: PyReadonlyArray2<'_, f64>,
                keys: PyReadonlyArray1<'_, u64>,
                strata: PyReadonlyArray1<'_, u32>,
            ) -> PyResult<()> {
                let state = self
                    .inner
                    .as_mut()
                    .ok_or_else(|| PyValueError::new_err("this pass is finished"))?;
                pass_sparse(py, state, indptr, indices, data, ncols, t, y, keys, strata)
            }
        }
    };
}

#[pyclass(name = "Compressor", module = "ume._core")]
struct PyCompressor {
    inner: Option<ParallelPass<MomentsFold>>,
}

fn check_batch(n: usize, t: &[f64], y_rows: usize, keys: &[u64], strata: &[u32]) -> PyResult<()> {
    if t.len() != n || y_rows != n || keys.len() != n || strata.len() != n {
        return Err(PyValueError::new_err(format!(
            "every input of a batch must have {n} rows"
        )));
    }
    Ok(())
}

fn take<P>(inner: &mut Option<P>) -> PyResult<P> {
    inner
        .take()
        .ok_or_else(|| PyValueError::new_err("this pass is finished"))
}

pass_class!(PyCompressor, {
    #[new]
    #[pyo3(signature = (d, o, threads=1))]
    fn new(d: usize, o: usize, threads: usize) -> Self {
        let layout = Layout::new(d, o);
        let template = Compressor::new(layout).into_fold().0;
        Self {
            inner: Some(ParallelPass::new(layout, threads, || template.fresh())),
        }
    }

    /// Flushes the last cluster and returns the compressed rows.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``g`` (clusters per stratum, shape ``(H,)``), ``first`` (``(H, q)``),
    ///     ``second`` (``(H, q, q)``), ``n_rows``, and ``keys`` (the sorted
    ///     cluster keys seen, for checking that shards do not share clusters).
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let pass = take(&mut self.inner)?.merged();
        let mut keys: Vec<u64> = pass.flushed_keys().iter().copied().collect();
        keys.sort_unstable();
        let out = compressed_to_dict(py, &pass.finish())?;
        out.set_item("keys", Array1::from(keys).into_pyarray(py))?;
        Ok(out)
    }
});

fn csr_rows(row_ptr: &[usize], col_idx: &[usize], values: &[f64], ncols: usize) -> PyResult<usize> {
    let n = row_ptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| PyValueError::new_err("indptr must not be empty"))?;
    let valid = row_ptr[0] == 0
        && row_ptr.windows(2).all(|p| p[0] <= p[1])
        && row_ptr[n] == col_idx.len()
        && col_idx.len() == values.len()
        && col_idx.iter().all(|&j| j < ncols);
    if valid {
        Ok(n)
    } else {
        Err(PyValueError::new_err("invalid CSR arrays"))
    }
}

fn compressed_to_dict<'py>(py: Python<'py>, c: &Compressed) -> PyResult<Bound<'py, PyDict>> {
    let q = c.layout.q();
    let h = c.strata.len();
    let out = PyDict::new(py);
    out.set_item("g", vec_to_numpy(py, &c.strata.iter().map(|s| s.g).collect::<Vec<_>>()))?;
    let first = Array2::from_shape_fn((h, q), |(i, k)| c.strata[i].first[k]);
    out.set_item("first", first.into_pyarray(py))?;
    let second = Array3::from_shape_fn((h, q, q), |(i, a, b)| c.strata[i].second[(a, b)]);
    out.set_item("second", second.into_pyarray(py))?;
    out.set_item("n_rows", c.n_rows)?;
    Ok(out)
}

/// Strata moments borrowed from numpy, rebuilt as owned values for the engine.
fn moments_from_numpy(
    g: &[f64],
    first: &PyReadonlyArray2<'_, f64>,
    second: &PyReadonlyArray3<'_, f64>,
    q: usize,
) -> PyResult<Vec<StratumMoments>> {
    let h = g.len();
    if first.shape() != [h, q] || second.shape() != [h, q, q] {
        return Err(PyValueError::new_err(format!(
            "first must be ({h}, {q}) and second ({h}, {q}, {q})"
        )));
    }
    let (f, s) = (first.as_slice()?, second.as_slice()?);
    Ok((0..h)
        .map(|i| StratumMoments {
            g: g[i],
            first: f[i * q..(i + 1) * q].to_vec(),
            second: Mat::from_fn(q, q, |a, b| s[i * q * q + a * q + b]),
        })
        .collect())
}

/// Fits the universal model from compressed rows.
///
/// Low-level kernel behind ``CompressedData.fit``.
///
/// Parameters
/// ----------
/// d, o : int
///     The layout: columns of the design matrix and outcome columns.
/// g, first, second : numpy.ndarray
///     The compressed rows: clusters per stratum ``(H,)``, first moments
///     ``(H, q)``, second moments ``(H, q, q)``.
/// z : numpy.ndarray
///     Instrument of each stratum, float64 ``(H,)``.
/// cohort : numpy.ndarray
///     Cohort index of each stratum, uint32 ``(H,)``.
/// n_cohorts : int
///     Number of cohorts.
/// weighted : bool
///     Use inverse propensity weights; otherwise every weight is 1.
///
/// Returns
/// -------
/// dict
///     As ``fit_rows_dense``, plus ``e_hat`` (per cohort) and ``w`` (per stratum).
///
/// Raises
/// ------
/// ValueError
///     If the shapes disagree, a cohort has one arm, or the equations are singular.
#[pyfunction]
#[pyo3(signature = (d, o, g, first, second, z, cohort, n_cohorts, weighted))]
#[allow(clippy::too_many_arguments)]
fn fit_compressed<'py>(
    py: Python<'py>,
    d: usize,
    o: usize,
    g: PyReadonlyArray1<'py, f64>,
    first: PyReadonlyArray2<'py, f64>,
    second: PyReadonlyArray3<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    cohort: PyReadonlyArray1<'py, u32>,
    n_cohorts: usize,
    weighted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let layout = Layout::new(d, o);
    let moments = moments_from_numpy(g.as_slice()?, &first, &second, layout.q())?;
    let (z, cohort) = (z.as_slice()?, cohort.as_slice()?);
    if z.len() != moments.len() || cohort.len() != moments.len() || cohort.iter().any(|&c| c as usize >= n_cohorts) {
        return Err(PyValueError::new_err(
            "z and cohort need one entry per stratum, cohorts in range",
        ));
    }
    let strata: Vec<Stratum<'_>> = moments
        .iter()
        .zip(z)
        .zip(cohort)
        .map(|((m, &z), &c)| Stratum {
            z,
            cohort: c,
            moments: m,
        })
        .collect();
    let result = py
        .detach(|| core_fit_compressed(layout, &strata, n_cohorts, weighted))
        .map_err(to_py_err)?;
    let out = fit_to_dict(py, &result.fit)?;
    out.set_item("e_hat", vec_to_numpy(py, &result.e_hat))?;
    out.set_item("w", vec_to_numpy(py, &result.w))?;
    Ok(out)
}

/// Compressed rows of the intent to treat model (treatment := instrument).
///
/// Parameters
/// ----------
/// d, o : int
///     The layout.
/// g, first, second : numpy.ndarray
///     The compressed rows, as in ``fit_compressed``.
/// z : numpy.ndarray
///     Instrument of each stratum.
/// treated_source : numpy.ndarray
///     For each outcome column, the index of its raw column if it is a ``T·Y``
///     column, else -1; int64 ``(o,)``.
///
/// Returns
/// -------
/// tuple of numpy.ndarray
///     The transformed ``first`` ``(H, q)`` and ``second`` ``(H, q, q)``.
#[pyfunction]
#[pyo3(signature = (d, o, g, first, second, z, treated_source))]
#[allow(clippy::too_many_arguments)]
fn compressed_intent_to_treat<'py>(
    py: Python<'py>,
    d: usize,
    o: usize,
    g: PyReadonlyArray1<'py, f64>,
    first: PyReadonlyArray2<'py, f64>,
    second: PyReadonlyArray3<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    treated_source: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let layout = Layout::new(d, o);
    let moments = moments_from_numpy(g.as_slice()?, &first, &second, layout.q())?;
    let source: Vec<Option<usize>> = treated_source
        .as_slice()?
        .iter()
        .map(|&s| if s < 0 { None } else { Some(s as usize) })
        .collect();
    let z = z.as_slice()?;
    let itt: Vec<StratumMoments> = moments
        .iter()
        .zip(z)
        .map(|(m, &zh)| intent_to_treat(&layout, m, zh, &source))
        .collect();
    let c = Compressed {
        layout,
        strata: itt,
        n_rows: 0,
    };
    let dict = compressed_to_dict(py, &c)?;
    Ok((dict.get_item("first")?.unwrap(), dict.get_item("second")?.unwrap()))
}

/// The entries of a cluster summary in the distinct layout.
///
/// Parameters
/// ----------
/// d, o : int
///     Columns of the design matrix and outcome columns.
///
/// Returns
/// -------
/// list of tuple
///     ``(block, i, j)`` for each of the ``q`` entries, with ``block`` one of
///     ``"S^Y"`` (``i`` a column of R, ``j`` an outcome), ``"S"`` or ``"S^T"``
///     (``i <= j`` columns of R).
#[pyfunction]
#[pyo3(signature = (d, o))]
fn summary_layout(d: usize, o: usize) -> Vec<(&'static str, usize, usize)> {
    Layout::new(d, o)
        .entries()
        .into_iter()
        .map(|(b, i, j)| {
            let name = match b {
                Block::Sy => "S^Y",
                Block::S => "S",
                Block::St => "S^T",
            };
            (name, i, j)
        })
        .collect()
}

// ---- bootstrap ----------------------------------------------------------------------------------

fn parse_scheme(name: &str) -> PyResult<Scheme> {
    match name {
        "bayesian" => Ok(Scheme::Bayesian),
        "poisson" => Ok(Scheme::Poisson),
        "multinomial" => Ok(Scheme::Multinomial),
        _ => Err(PyValueError::new_err(format!(
            "unknown weights {name:?}: choose 'bayesian', 'poisson' or 'multinomial'"
        ))),
    }
}

/// Adds a dense batch to any pass.
fn pass_dense<F: Fold + Merge + Send>(
    py: Python<'_>,
    state: &mut ParallelPass<F>,
    r: PyReadonlyArray2<'_, f64>,
    t: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    keys: PyReadonlyArray1<'_, u64>,
    strata: PyReadonlyArray1<'_, u32>,
) -> PyResult<()> {
    let (n, d) = (r.shape()[0], r.shape()[1]);
    let (t, keys, strata) = (t.as_slice()?, keys.as_slice()?, strata.as_slice()?);
    check_batch(n, t, y.shape()[0], keys, strata)?;
    let rm = Dense::from_row_major(r.as_slice()?, n, d);
    let ym = MatRef::from_row_major_slice(y.as_slice()?, n, y.shape()[1]);
    py.detach(|| state.update(&rm, t, ym, keys, strata)).map_err(to_py_err)
}

/// Adds a sparse (CSR) batch to any pass.
#[allow(clippy::too_many_arguments)]
fn pass_sparse<F: Fold + Merge + Send>(
    py: Python<'_>,
    state: &mut ParallelPass<F>,
    indptr: PyReadonlyArray1<'_, usize>,
    indices: PyReadonlyArray1<'_, usize>,
    data: PyReadonlyArray1<'_, f64>,
    ncols: usize,
    t: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    keys: PyReadonlyArray1<'_, u64>,
    strata: PyReadonlyArray1<'_, u32>,
) -> PyResult<()> {
    let (row_ptr, col_idx, values) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let n = csr_rows(row_ptr, col_idx, values, ncols)?;
    let (t, keys, strata) = (t.as_slice()?, keys.as_slice()?, strata.as_slice()?);
    check_batch(n, t, y.shape()[0], keys, strata)?;
    let rm = Sparse::from_csr(n, ncols, row_ptr, col_idx, values);
    let ym = MatRef::from_row_major_slice(y.as_slice()?, n, y.shape()[1]);
    py.detach(|| state.update(&rm, t, ym, keys, strata)).map_err(to_py_err)
}

fn sums_to_numpy<'py>(py: Python<'py>, strata: &[Mat<f64>], b: usize, m: usize) -> Bound<'py, PyAny> {
    Array3::from_shape_fn((strata.len(), b, m), |(h, draw, k)| strata[h][(draw, k)])
        .into_pyarray(py)
        .into_any()
}

/// The bootstrap weights of clusters, in pass order.
///
/// Each weight is a pure function of (seed, cluster key, draw), from a
/// Philox4x64-10 stream, except the multinomial counts, which depend on the
/// order of the clusters.
///
/// Parameters
/// ----------
/// keys : numpy.ndarray
///     Cluster keys, uint64, in pass order.
/// b : int
///     Number of draws.
/// seed : int
///     Seed (a non-negative 64-bit integer).
/// weights : {"bayesian", "poisson", "multinomial"}
///     Exp(1), Poisson(1), or Multinomial(G; 1/G, ...) counts over the given clusters.
///
/// Returns
/// -------
/// numpy.ndarray
///     The weights, float64, shape ``(len(keys), b)``.
#[pyfunction]
#[pyo3(signature = (keys, b, seed, weights))]
fn bootstrap_weights<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, u64>,
    b: usize,
    seed: u64,
    weights: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let keys = keys.as_slice()?;
    let mut w = Weights::new(parse_scheme(weights)?, seed, b, Some(keys.len() as u64));
    let mut out = Array2::<f64>::zeros((keys.len(), b));
    let mut row = vec![0.0; b];
    for (i, &key) in keys.iter().enumerate() {
        w.next(key, &mut row);
        out.row_mut(i).assign(&Array1::from(row.clone()));
    }
    Ok(out.into_pyarray(py).into_any())
}

/// One pass over rows grouped by cluster, to the weighted sums of B bootstrap draws.
///
/// The exact engine behind ``Experiment.bootstrap``: at each cluster's flush,
/// adds ``xi[b] * (1, u_a)`` into its stratum for every draw ``b``.
///
/// Parameters
/// ----------
/// d, o : int
///     Columns of the design matrix and outcome columns.
/// b : int
///     Number of draws.
/// seed : int
///     Seed of the weight streams.
/// weights : {"bayesian", "poisson", "multinomial"}
///     The weight scheme.
/// n_clusters : int, optional
///     Number of clusters G, required for ``"multinomial"``.
/// threads : int
///     Worker threads (the multinomial bootstrap always uses one).
#[pyclass(name = "BootstrapCompressor", module = "ume._core")]
struct PyBootstrapCompressor {
    inner: Option<ParallelPass<DrawsFold>>,
    b: usize,
}

pass_class!(PyBootstrapCompressor, {
    #[new]
    #[pyo3(signature = (d, o, b, seed, weights, n_clusters=None, threads=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        d: usize,
        o: usize,
        b: usize,
        seed: u64,
        weights: &str,
        n_clusters: Option<u64>,
        threads: usize,
    ) -> PyResult<Self> {
        let scheme = parse_scheme(weights)?;
        if scheme == Scheme::Multinomial && n_clusters.is_none() {
            return Err(PyValueError::new_err("the multinomial bootstrap needs n_clusters"));
        }
        let layout = Layout::new(d, o);
        let template = BootstrapCompressor::bootstrap(layout, scheme, seed, b, n_clusters)
            .into_fold()
            .0;
        let threads = if template.mergeable() { threads } else { 1 };
        Ok(Self {
            inner: Some(ParallelPass::new(layout, threads, || template.fresh())),
            b,
        })
    }

    /// Flushes the last cluster and returns the draws' weighted sums.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``sums`` (shape ``(H, b, 1 + q)``: row ``b`` of stratum ``h`` is
    ///     ``sum_a xi[b] (1, u_a)``), ``n_rows``, and ``keys`` (the sorted cluster keys seen).
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let pass = take(&mut self.inner)?.merged();
        let m = pass.layout().q() + 1;
        let mut keys: Vec<u64> = pass.flushed_keys().iter().copied().collect();
        keys.sort_unstable();
        let draws = pass.finish_draws();
        let out = PyDict::new(py);
        out.set_item("sums", sums_to_numpy(py, &draws.strata, self.b, m))?;
        out.set_item("n_rows", draws.n_rows)?;
        out.set_item("keys", Array1::from(keys).into_pyarray(py))?;
        Ok(out)
    }
});

/// Draws of each stratum's weighted sum from its compressed row: the Gaussian engine.
///
/// Given the data, ``sum_a xi_a v_a`` has mean ``sum v_a`` and covariance
/// ``sum v_a v_a^T`` for independent weights with mean 1 and variance 1, so each
/// stratum's sum is drawn from that normal distribution.
///
/// Parameters
/// ----------
/// d, o : int
///     The layout.
/// g, first, second : numpy.ndarray
///     The compressed rows, as in ``fit_compressed``.
/// b : int
///     Number of draws.
/// seed : int
///     Seed of the normal streams (keyed by stratum position).
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, b, 1 + q)``.
#[pyfunction]
#[pyo3(signature = (d, o, g, first, second, b, seed))]
#[allow(clippy::too_many_arguments)]
fn gaussian_sums<'py>(
    py: Python<'py>,
    d: usize,
    o: usize,
    g: PyReadonlyArray1<'py, f64>,
    first: PyReadonlyArray2<'py, f64>,
    second: PyReadonlyArray3<'py, f64>,
    b: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let layout = Layout::new(d, o);
    let moments = moments_from_numpy(g.as_slice()?, &first, &second, layout.q())?;
    let strata = py
        .detach(|| {
            moments
                .iter()
                .enumerate()
                .map(|(h, m)| core_gaussian_sums(m, h as u64, seed, b))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(to_py_err)?;
    Ok(sums_to_numpy(py, &strata, b, layout.q() + 1))
}

/// Fits every bootstrap draw from its per-stratum weighted sums.
///
/// Each draw recomputes the propensities from its stratum totals, then fits the
/// model's coefficients and compliers exactly as the point estimate does.
///
/// Parameters
/// ----------
/// d, o : int
///     The layout.
/// sums : numpy.ndarray
///     Shape ``(H, b, 1 + q)``, from ``BootstrapCompressor`` or ``gaussian_sums``.
/// z, cohort : numpy.ndarray
///     Instrument (float64) and cohort index (uint32) of each stratum.
/// n_cohorts : int
///     Number of cohorts.
/// weighted : bool
///     Use inverse propensity weights.
/// encouraged : bool
///     Whether the design has encouragement (it decides how x̄_C is computed).
/// threads : int
///     Worker threads; the draws are independent.
///
/// Returns
/// -------
/// dict
///     ``beta`` ``(b, 2d, o)``, ``x_c`` ``(b, k)``, ``pi_c`` ``(b,)``, ``e_hat``
///     ``(b, n_cohorts)``, and ``ok`` ``(b,)``: False for draws whose model is
///     degenerate, whose other entries are NaN.
#[pyfunction]
#[pyo3(signature = (d, o, sums, z, cohort, n_cohorts, weighted, encouraged, threads=1))]
#[allow(clippy::too_many_arguments)]
fn fit_draws<'py>(
    py: Python<'py>,
    d: usize,
    o: usize,
    sums: PyReadonlyArray3<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    cohort: PyReadonlyArray1<'py, u32>,
    n_cohorts: usize,
    weighted: bool,
    encouraged: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let layout = Layout::new(d, o);
    let shape = sums.shape().to_vec();
    let (h, b, m) = (shape[0], shape[1], shape[2]);
    if m != layout.q() + 1 || z.len() != h || cohort.len() != h {
        return Err(PyValueError::new_err(
            "sums must be (H, b, 1 + q), with z and cohort of length H",
        ));
    }
    let flat = sums.as_slice()?;
    let mats: Vec<Mat<f64>> = (0..h)
        .map(|s| Mat::from_fn(b, m, |draw, k| flat[s * b * m + draw * m + k]))
        .collect();
    let keys: Vec<StratumKey> = z
        .as_slice()?
        .iter()
        .zip(cohort.as_slice()?)
        .map(|(&z, &c)| StratumKey { z, cohort: c })
        .collect();
    let fits = py.detach(|| core_fit_draws(&layout, &keys, &mats, n_cohorts, weighted, encouraged, threads));
    let k = d - 1;
    let mut beta = Array3::<f64>::from_elem((b, 2 * d, o), f64::NAN);
    let mut x_c = Array2::<f64>::from_elem((b, k), f64::NAN);
    let mut pi_c = Array1::<f64>::from_elem(b, f64::NAN);
    let mut e_hat = Array2::<f64>::from_elem((b, n_cohorts), f64::NAN);
    let mut ok = Array1::<bool>::from_elem(b, false);
    for (draw, fit) in fits.iter().enumerate() {
        if let Ok(f) = fit {
            ok[draw] = true;
            for i in 0..2 * d {
                for l in 0..o {
                    beta[(draw, i, l)] = f.point.beta[(i, l)];
                }
            }
            for (j, v) in f.point.compliers.x_c.iter().enumerate() {
                x_c[(draw, j)] = *v;
            }
            pi_c[draw] = f.point.compliers.pi_c;
            for (c, v) in f.e_hat.iter().enumerate() {
                e_hat[(draw, c)] = *v;
            }
        }
    }
    let out = PyDict::new(py);
    out.set_item("beta", beta.into_pyarray(py))?;
    out.set_item("x_c", x_c.into_pyarray(py))?;
    out.set_item("pi_c", pi_c.into_pyarray(py))?;
    out.set_item("e_hat", e_hat.into_pyarray(py))?;
    out.set_item("ok", ok.into_pyarray(py))?;
    Ok(out)
}

// ---- delta vectors early ----------------------------------------------------------------------

/// Pass 1 of delta vectors early: per stratum, ``(G_h, sum u_a)``.
///
/// Parameters
/// ----------
/// d, o : int
///     Columns of the design matrix and outcome columns.
#[pyclass(name = "FirstMomentsPass", module = "ume._core")]
struct PyFirstMomentsPass {
    inner: Option<ParallelPass<FirstMomentsFold>>,
}

pass_class!(PyFirstMomentsPass, {
    #[new]
    #[pyo3(signature = (d, o, threads=1))]
    fn new(d: usize, o: usize, threads: usize) -> Self {
        let layout = Layout::new(d, o);
        let template = FirstMomentsPass::first_moments(layout).into_fold().0;
        Self {
            inner: Some(ParallelPass::new(layout, threads, || template.fresh())),
        }
    }

    /// Flushes the last cluster.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``sums`` (shape ``(H, 1 + q)``: ``(G_h, sum u_a)``) and ``n_rows``.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("this pass is finished"))?;
        let (strata, n_rows) = state.merged().finish_first();
        let m = strata.first().map_or(0, Vec::len);
        let sums = Array2::from_shape_fn((strata.len(), m), |(h, k)| strata[h][k]);
        let out = PyDict::new(py);
        out.set_item("sums", sums.into_pyarray(py))?;
        out.set_item("n_rows", n_rows)?;
        Ok(out)
    }
});

/// Pass 2 of delta vectors early: per effect, ``sum_a (c_h . v_a)^2``.
///
/// Parameters
/// ----------
/// d, o : int
///     Columns of the design matrix and outcome columns.
/// projections : numpy.ndarray
///     ``c_h`` of each effect, shape ``(H, E, 1 + q)``, indexed by the stratum
///     indices of pass 1.
#[pyclass(name = "ScorePass", module = "ume._core")]
struct PyScorePass {
    inner: Option<ParallelPass<ScoreFold>>,
}

pass_class!(PyScorePass, {
    #[new]
    #[pyo3(signature = (d, o, projections, threads=1))]
    fn new(d: usize, o: usize, projections: PyReadonlyArray3<'_, f64>, threads: usize) -> PyResult<Self> {
        let layout = Layout::new(d, o);
        let shape = projections.shape().to_vec();
        let (h, e, m) = (shape[0], shape[1], shape[2]);
        if m != layout.q() + 1 {
            return Err(PyValueError::new_err("projections must be (H, E, 1 + q)"));
        }
        let flat = projections.as_slice()?;
        let mats = (0..h)
            .map(|s| Mat::from_fn(e, m, |i, k| flat[s * e * m + i * m + k]))
            .collect();
        let template = ScorePass::scores(layout, mats).into_fold().0;
        Ok(Self {
            inner: Some(ParallelPass::new(layout, threads, || template.fresh())),
        })
    }

    /// Flushes the last cluster and returns ``sum_a psi_a^2`` per effect, shape ``(E,)``.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("this pass is finished"))?;
        Ok(vec_to_numpy(py, &state.merged().finish_scores()))
    }
});

/// The point estimate from first moments: ``beta``, ``A``, the compliers, the weights.
///
/// Parameters
/// ----------
/// d, o : int
///     The layout.
/// sums : numpy.ndarray
///     ``(G_h, sum u_a)`` per stratum, shape ``(H, 1 + q)``.
/// z, cohort : numpy.ndarray
///     Instrument (float64) and cohort index (uint32) of each stratum.
/// n_cohorts : int
///     Number of cohorts.
/// weighted, encouraged : bool
///     Use inverse propensity weights; whether the design has encouragement.
///
/// Returns
/// -------
/// dict
///     ``beta``, ``a``, ``x_c``, ``pi_c``, ``e_hat`` and ``w`` (per stratum).
#[pyfunction]
#[pyo3(signature = (d, o, sums, z, cohort, n_cohorts, weighted, encouraged))]
#[allow(clippy::too_many_arguments)]
fn fit_first_moments<'py>(
    py: Python<'py>,
    d: usize,
    o: usize,
    sums: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    cohort: PyReadonlyArray1<'py, u32>,
    n_cohorts: usize,
    weighted: bool,
    encouraged: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let layout = Layout::new(d, o);
    let m = layout.q() + 1;
    let (h, flat) = (sums.shape()[0], sums.as_slice()?);
    if sums.shape()[1] != m || z.len() != h || cohort.len() != h {
        return Err(PyValueError::new_err(
            "sums must be (H, 1 + q), with z and cohort of length H",
        ));
    }
    let strata: Vec<StratumSums<'_>> = (0..h)
        .map(|i| StratumSums {
            z: z.as_slice().unwrap()[i],
            cohort: cohort.as_slice().unwrap()[i],
            g: flat[i * m],
            first: &flat[i * m + 1..(i + 1) * m],
        })
        .collect();
    let (all, zb, e_hat, w) = strata_blocks(&layout, &strata, n_cohorts, weighted).map_err(to_py_err)?;
    let p = point(&all, &zb, encouraged).map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("beta", mat_to_numpy(py, &p.beta))?;
    out.set_item("a", mat_to_numpy(py, &p.a))?;
    out.set_item("x_c", vec_to_numpy(py, &p.compliers.x_c))?;
    out.set_item("pi_c", p.compliers.pi_c)?;
    out.set_item("e_hat", vec_to_numpy(py, &e_hat))?;
    out.set_item("w", vec_to_numpy(py, &w))?;
    Ok(out)
}

/// Sandwich projections ``c_h`` of effects, for pass 2 of delta vectors early.
///
/// Parameters
/// ----------
/// d, o : int
///     The layout.
/// beta, a : numpy.ndarray
///     The point estimate and its ``A``.
/// grads : numpy.ndarray
///     Gradient of each effect with respect to ``vec(beta)`` (column-major), shape ``(E, 2do)``.
/// w, z : numpy.ndarray
///     Weight and instrument of each stratum, shape ``(H,)``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, E, 1 + q)``.
#[pyfunction]
#[pyo3(signature = (d, o, beta, a, grads, w, z))]
#[allow(clippy::too_many_arguments)]
fn sandwich_projections<'py>(
    py: Python<'py>,
    d: usize,
    o: usize,
    beta: PyReadonlyArray2<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    grads: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let layout = Layout::new(d, o);
    let bm = MatRef::from_row_major_slice(beta.as_slice()?, 2 * d, o);
    let am = MatRef::from_row_major_slice(a.as_slice()?, 2 * d, 2 * d);
    let (g, e) = (grads.as_slice()?, grads.shape()[0]);
    let (w, z) = (w.as_slice()?, z.as_slice()?);
    let m = layout.q() + 1;
    let p = 2 * d * o;
    let mut out = Array3::<f64>::zeros((w.len(), e, m));
    for h in 0..w.len() {
        for i in 0..e {
            let c = sandwich_projection(&layout, bm, am, &g[i * p..(i + 1) * p], w[h], z[h]);
            for (k, v) in c.into_iter().enumerate() {
                out[(h, i, k)] = v;
            }
        }
    }
    Ok(out.into_pyarray(py).into_any())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(fit_rows_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rows_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_propensity, m)?)?;
    m.add_function(wrap_pyfunction!(fit_compressed, m)?)?;
    m.add_function(wrap_pyfunction!(compressed_intent_to_treat, m)?)?;
    m.add_function(wrap_pyfunction!(summary_layout, m)?)?;
    m.add_class::<PyCompressor>()?;
    m.add_class::<PyBootstrapCompressor>()?;
    m.add_class::<PyFirstMomentsPass>()?;
    m.add_class::<PyScorePass>()?;
    m.add_function(wrap_pyfunction!(fit_first_moments, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_projections, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_weights, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_sums, m)?)?;
    m.add_function(wrap_pyfunction!(fit_draws, m)?)?;
    Ok(())
}
