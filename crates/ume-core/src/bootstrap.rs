//! Bootstrap draws of the universal model, from rows or from compressed data.
//!
//! A draw is the universal model refit with one more weight per cluster,
//! $\xi_a^{(b)}$ (BUM post). Everything a draw needs is one weighted sum per
//! stratum, $\sum_{a \in h} \xi_a v_a$ with $v_a = (1, u_a)$. Its first entry is the
//! stratum's total weight, which gives the draw's propensities, and the rest gives
//! its blocks. [`fit_draws`] turns those sums into each draw's $\hat\beta$ and
//! compliers with the same [`strata_blocks`] and [`point`] as the point estimate.
//! The engines differ only in where the sums come from:
//!
//! - **exact** ([`BootstrapCompressor`]): one pass over the rows, adding
//!   $\xi^{(b)}_a v_a$ for every draw $b$ at each cluster's flush;
//! - **Gaussian** ([`gaussian_sums`]): no pass. Given the data, the weighted sums of
//!   a stratum have mean $\sum v_a$ and covariance $\sum v_a v_a^\top$ when the
//!   weights have mean 1 and variance 1, so each stratum's sum is drawn from that
//!   normal distribution. The compressed row is the posterior's first two moments.

use faer::{Mat, Side};

use crate::Error;
use crate::compress::{Fold, Pass, StratumMoments};
use crate::fit::{Point, StratumSums, point, strata_blocks};
use crate::layout::Layout;
use crate::rng::Stream;
use crate::weights::{Scheme, Weights};

/// Stream tag of the normal draws of the Gaussian engine (decision 0018).
const NORMAL_TAG: u64 = 3;

/// The fold of the exact bootstrap: per stratum, a $B \times (1 + q)$ matrix whose
/// row $b$ is $\sum_{a \in h} \xi^{(b)}_a v_a$.
///
/// Column $k$ is contiguous over $b$ (faer is column-major), so the flush is one
/// contiguous `axpy` per nonzero entry of $u_a$: about 2× faster than looping over
/// draws first (handoff 5.5).
pub struct DrawsFold {
    q: usize,
    weights: Weights,
    xi: Vec<f64>,
    strata: Vec<Mat<f64>>,
}

impl Fold for DrawsFold {
    fn fold(&mut self, key: u64, stratum: u32, u: &[f64], nz: &[usize]) {
        let h = stratum as usize;
        let b = self.weights.draws();
        while self.strata.len() <= h {
            self.strata.push(Mat::zeros(b, 1 + self.q));
        }
        self.weights.next(key, &mut self.xi);
        let state = &mut self.strata[h];
        for (s, xi) in state.col_as_slice_mut(0).iter_mut().zip(&self.xi) {
            *s += xi;
        }
        for &k in nz {
            let uk = u[k];
            for (s, xi) in state.col_as_slice_mut(k + 1).iter_mut().zip(&self.xi) {
                *s += xi * uk;
            }
        }
    }
}

impl crate::compress::Merge for DrawsFold {
    fn merge(&mut self, other: Self) {
        for (h, theirs) in other.strata.into_iter().enumerate() {
            if self.strata.len() <= h {
                self.strata.push(Mat::zeros(theirs.nrows(), theirs.ncols()));
            }
            self.strata[h] += &theirs;
        }
    }
}

impl DrawsFold {
    /// Whether states over disjoint clusters can be merged: true for independent
    /// weights (Bayesian, Poisson), false for the coupled multinomial counts.
    pub fn mergeable(&self) -> bool {
        self.weights.scheme() != Scheme::Multinomial
    }

    /// A fresh fold with the same scheme, seed and draws, for another worker.
    pub fn fresh(&self) -> Self {
        Self {
            q: self.q,
            weights: self.weights.fresh(),
            xi: vec![0.0; self.xi.len()],
            strata: Vec::new(),
        }
    }
}

/// The exact bootstrap pass: rows to $B$ weighted sums per stratum.
pub type BootstrapCompressor = Pass<DrawsFold>;

/// The output of a draw engine: per stratum, a $B \times (1 + q)$ matrix of weighted sums.
#[derive(Clone, Debug)]
pub struct DrawSums {
    /// The layout of the summaries.
    pub layout: Layout,
    /// One matrix per stratum index; row $b$ is $\sum_{a \in h} \xi^{(b)}_a v_a$.
    pub strata: Vec<Mat<f64>>,
    /// Number of rows read (0 for the Gaussian engine).
    pub n_rows: u64,
}

impl Pass<DrawsFold> {
    /// An exact bootstrap pass with $B$ draws of `scheme`. `n_clusters` ($G$) is
    /// required by [`Scheme::Multinomial`].
    pub fn bootstrap(layout: Layout, scheme: Scheme, seed: u64, b: usize, n_clusters: Option<u64>) -> Self {
        let fold = DrawsFold {
            q: layout.q(),
            weights: Weights::new(scheme, seed, b, n_clusters),
            xi: vec![0.0; b],
            strata: Vec::new(),
        };
        Self::with_fold(layout, fold)
    }

    /// Flushes the last cluster and returns the draws' weighted sums.
    pub fn finish_draws(self) -> DrawSums {
        let layout = self.layout();
        let (fold, n_rows) = self.into_fold();
        DrawSums {
            layout,
            strata: fold.strata,
            n_rows,
        }
    }
}

/// $B$ draws of one stratum's weighted sum from $N(\sum v_a, \sum v_a v_a^\top)$,
/// as a $B \times (1 + q)$ matrix.
///
/// The covariance is singular (unprompted strata have $S^T = 0$, for example), so
/// its square root comes from a symmetric eigendecomposition with negative
/// eigenvalues clamped to 0, not from a Cholesky factorization. The normal draws
/// come from the Philox stream keyed by (seed, `stratum`), with Box-Muller.
///
/// # Errors
/// [`Error::Singular`] if the eigendecomposition fails.
pub fn gaussian_sums(moments: &StratumMoments, stratum: u64, seed: u64, b: usize) -> Result<Mat<f64>, Error> {
    let q = moments.first.len();
    let m = q + 1;
    let mean: Vec<f64> = std::iter::once(moments.g)
        .chain(moments.first.iter().copied())
        .collect();
    let cov = Mat::from_fn(m, m, |i, j| match (i, j) {
        (0, 0) => moments.g,
        (0, j) => moments.first[j - 1],
        (i, 0) => moments.first[i - 1],
        (i, j) => moments.second[(i - 1, j - 1)],
    });
    let evd = cov.self_adjoint_eigen(Side::Lower).map_err(|_| Error::Singular)?;
    let (u, s) = (evd.U(), evd.S().column_vector());
    let root = Mat::from_fn(m, m, |i, k| u[(i, k)] * s[k].max(0.0).sqrt());
    let normals = standard_normals(Stream::new(seed, stratum, NORMAL_TAG), b * m);
    let z = Mat::from_fn(m, b, |k, draw| normals[draw * m + k]);
    let draws = &root * &z;
    Ok(Mat::from_fn(b, m, |draw, k| mean[k] + draws[(k, draw)]))
}

/// `n` standard normal numbers from a uniform stream, by Box-Muller.
fn standard_normals(stream: Stream, n: usize) -> Vec<f64> {
    let mut uniforms = vec![0.0; n + n % 2];
    stream.uniforms(&mut uniforms);
    let mut out = Vec::with_capacity(uniforms.len());
    for pair in uniforms.chunks(2) {
        let radius = (-2.0 * pair[0].ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * pair[1];
        out.push(radius * angle.cos());
        out.push(radius * angle.sin());
    }
    out.truncate(n);
    out
}

/// The instrument and cohort of a stratum, for [`fit_draws`].
#[derive(Clone, Copy, Debug)]
pub struct StratumKey {
    /// $Z_h$, 0 or 1.
    pub z: f64,
    /// Cohort index.
    pub cohort: u32,
}

/// One draw's fit: its point estimate and propensities.
#[derive(Clone, Debug)]
pub struct DrawFit {
    /// $\hat\beta^{(b)}$ and the compliers of the draw.
    pub point: Point,
    /// $\hat e_c^{(b)}$ per cohort.
    pub e_hat: Vec<f64>,
}

/// Fits every draw: the draw's propensities from the stratum totals, then the
/// same blocks and point estimate as the data. A draw whose model is degenerate
/// (for example, a Poisson draw that leaves a cohort with weight in one arm only)
/// is an `Err`. The draws are independent, so they are fit on `threads` threads.
pub fn fit_draws(
    layout: &Layout,
    keys: &[StratumKey],
    sums: &[Mat<f64>],
    n_cohorts: usize,
    weighted: bool,
    encouraged: bool,
    threads: usize,
) -> Vec<Result<DrawFit, Error>> {
    let b = sums.first().map_or(0, |m| m.nrows());
    let fit_one = |draw: usize| -> Result<DrawFit, Error> {
        let firsts: Vec<Vec<f64>> = sums
            .iter()
            .map(|m| (1..m.ncols()).map(|k| m[(draw, k)]).collect())
            .collect();
        let strata: Vec<StratumSums<'_>> = keys
            .iter()
            .zip(sums)
            .zip(&firsts)
            .map(|((key, m), first)| StratumSums {
                z: key.z,
                cohort: key.cohort,
                g: m[(draw, 0)],
                first,
            })
            .collect();
        let (all, z, e_hat, _) = strata_blocks(layout, &strata, n_cohorts, weighted)?;
        Ok(DrawFit {
            point: point(&all, &z, encouraged)?,
            e_hat,
        })
    };
    let threads = threads.clamp(1, b.max(1));
    let chunk = b.div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                let fit_one = &fit_one;
                scope.spawn(move || ((w * chunk)..((w + 1) * chunk).min(b)).map(fit_one).collect::<Vec<_>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a draw worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::Compressor;
    use crate::rows::Dense;

    /// Design matrix values, treatment, outcomes, cluster keys, strata.
    type PassInputs = (Vec<f64>, Vec<f64>, Mat<f64>, Vec<u64>, Vec<u32>);

    /// Six rows in four clusters, two strata, one covariate, one outcome.
    fn pass_inputs() -> PassInputs {
        let r = vec![1.0, 0.5, 1.0, -1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.5, 1.0, -0.5];
        let t = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let y = Mat::from_fn(6, 1, |i, _| i as f64 + 1.0);
        (r, t, y, vec![1, 1, 2, 3, 3, 4], vec![1, 1, 0, 1, 1, 0])
    }

    #[test]
    fn multinomial_draws_keep_the_number_of_clusters() {
        let (r, t, y, keys, strata) = pass_inputs();
        let dense = Dense::from_row_major(&r, 6, 2);
        let layout = Layout::new(2, 1);
        let mut c = Compressor::new(layout);
        c.update(&dense, &t, y.as_ref(), &keys, &strata).unwrap();
        let compressed = c.finish();
        let mut boot = BootstrapCompressor::bootstrap(layout, Scheme::Multinomial, 1, 3, Some(4));
        boot.update(&dense, &t, y.as_ref(), &keys, &strata).unwrap();
        let draws = boot.finish_draws();
        // The multinomial draws sum to G = 4 clusters in total.
        for draw in 0..3 {
            let total: f64 = draws.strata.iter().map(|m| m[(draw, 0)]).sum();
            assert_eq!(total, 4.0);
        }
        assert_eq!(compressed.strata.len(), draws.strata.len());
    }

    #[test]
    fn gaussian_draws_have_the_compressed_moments() {
        let (r, t, y, keys, strata) = pass_inputs();
        let dense = Dense::from_row_major(&r, 6, 2);
        let mut c = Compressor::new(Layout::new(2, 1));
        c.update(&dense, &t, y.as_ref(), &keys, &strata).unwrap();
        let moments = &c.finish().strata[1];
        let b = 40_000;
        let draws = gaussian_sums(moments, 1, 9, b).unwrap();
        let mean0: f64 = (0..b).map(|i| draws[(i, 0)]).sum::<f64>() / b as f64;
        let var0: f64 = (0..b).map(|i| (draws[(i, 0)] - mean0).powi(2)).sum::<f64>() / b as f64;
        assert!((mean0 - moments.g).abs() < 0.05, "{mean0}");
        assert!((var0 - moments.g).abs() < 0.1, "{var0}");
    }
}
