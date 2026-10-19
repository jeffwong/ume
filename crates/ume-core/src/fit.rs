//! The universal model: one assembly, two sources of data.
//!
//! One model covers every design: weighted two stage least squares with every
//! covariate interacted with treatment, cluster-robust covariance, and several
//! outcome columns. The design features are properties of the data, not
//! switches:
//!
//! - no encouragement ($T = Z$): the first stage is exact, and the model is WLS;
//! - no clusters (one row per cluster): the covariance is HC0;
//! - no adaptive assignment: the weights are constant;
//! - mean or ratio metrics: just outcome columns.
//!
//! [`assemble`] is the model: [`Blocks`] → [`NormalEquations`] → solve → meat →
//! [`sandwich`], for both stages, plus the [`Compliers`]. Its inputs are two block
//! sets and a [`MeatSource`]. Rows and compressed data differ only in how they
//! provide those:
//!
//! | | blocks | meat |
//! |---|---|---|
//! | rows ([`fit`]) | weighted products of $R$ | cluster scores of the residuals |
//! | compressed ([`fit_compressed`]) | $\sum_h w_h \sum_{a \in h} u_a$ | $\sum_h C_h (\sum u_a u_a^\top) C_h^\top$ |

use faer::prelude::*;
use faer::{Mat, MatRef};

use crate::Error;
use crate::blocks::{Blocks, Compliers, NormalEquations};
use crate::compress::StratumMoments;
use crate::layout::Layout;
use crate::propensity::inverse_propensity;
use crate::rows::RowMatrix;
use crate::sandwich::{
    Scope, Scoped, StageKind, StratumMeat, cluster_scores, meat_from_moments, meat_from_scores, sandwich,
};

/// The coefficients of one stage and their cluster-robust covariance.
#[derive(Clone, Debug)]
pub struct Stage {
    /// Coefficients $\hat\theta$, $2d \times m$.
    pub coef: Mat<f64>,
    /// $A$ of the normal equations, $2d \times 2d$.
    pub a: Mat<f64>,
    /// Covariance of the coefficients: joint for the second stage, per column for the first.
    pub cov: Scoped,
}

/// The fitted universal model.
#[derive(Clone, Debug)]
pub struct Fit {
    /// Blocks over all rows, weights $w$.
    pub blocks_all: Blocks,
    /// Blocks over the arm $Z = 1$, weights $w Z$.
    pub blocks_z: Blocks,
    /// Second stage: $\hat\beta$ ($2d \times o$) and its joint covariance.
    pub second: Stage,
    /// First stage: $\hat\Gamma$ ($2d \times d$) and per-equation covariances; `None` without encouragement.
    pub first: Option<Stage>,
    /// Take-up, compliance rate and compliers' mean.
    pub compliers: Compliers,
    /// Whether $T \ne Z$ in some row.
    pub encouraged: bool,
}

/// Provides the meat $\sum_a s_a s_a^\top$ of a stage, given its coefficients.
pub trait MeatSource {
    /// The meat of stage `kind` at coefficients `coef`, kept at `scope`.
    fn meat(&self, kind: StageKind, coef: MatRef<'_, f64>, scope: Scope) -> Scoped;
}

/// The point estimate: $\hat\beta$ and the compliers, without a covariance.
///
/// This is all a bootstrap draw needs.
#[derive(Clone, Debug)]
pub struct Point {
    /// $\hat\beta$, $2d \times o$.
    pub beta: Mat<f64>,
    /// $A = M_1^\top W M_2$ of the second stage, $2d \times 2d$.
    pub a: Mat<f64>,
    /// Take-up, compliance rate and compliers' mean.
    pub compliers: Compliers,
}

/// The point estimate from two block sets.
///
/// # Errors
/// [`Error::Singular`] if the second stage is singular.
pub fn point(blocks_all: &Blocks, blocks_z: &Blocks, encouraged: bool) -> Result<Point, Error> {
    let eq = NormalEquations::second_stage(blocks_all, blocks_z);
    let beta = solve(&eq)?;
    Ok(Point {
        beta,
        a: eq.a,
        compliers: Compliers::from_blocks(blocks_all, blocks_z, encouraged),
    })
}

/// The universal model from its two block sets and a source of meat.
///
/// # Errors
/// [`Error::Singular`] if a stage's $A$ is singular.
pub fn assemble<S: MeatSource>(
    blocks_all: Blocks,
    blocks_z: Blocks,
    encouraged: bool,
    source: &S,
) -> Result<Fit, Error> {
    let Point { beta, a, compliers } = point(&blocks_all, &blocks_z, encouraged)?;
    let meat = source.meat(StageKind::Second, beta.as_ref(), Scope::Joint);
    let cov = sandwich(a.as_ref(), &meat);
    let second = Stage { coef: beta, a, cov };

    let first = if encouraged {
        let first_eq = NormalEquations::first_stage(&blocks_all, &blocks_z);
        let gamma = solve(&first_eq)?;
        let meat = source.meat(StageKind::First, gamma.as_ref(), Scope::PerColumn);
        let cov = sandwich(first_eq.a.as_ref(), &meat);
        Some(Stage {
            coef: gamma,
            a: first_eq.a,
            cov,
        })
    } else {
        None
    };
    Ok(Fit {
        blocks_all,
        blocks_z,
        second,
        first,
        compliers,
        encouraged,
    })
}

/// Solves $A\,\theta = b$, failing if $A$ is numerically singular.
fn solve(eq: &NormalEquations) -> Result<Mat<f64>, Error> {
    let theta = eq.a.partial_piv_lu().solve(&eq.b);
    if theta.as_ref().is_all_finite() {
        Ok(theta)
    } else {
        Err(Error::Singular)
    }
}

// ---- rows -----------------------------------------------------------------------------------

/// The rows of an experiment, ready for the engine.
pub struct RowData<'a, R: RowMatrix> {
    /// Design matrix $R = (1, x - x_{ref})$, $n \times d$.
    pub r: &'a R,
    /// Instrument $Z$ of each row (constant within a cluster).
    pub z: &'a [f64],
    /// Treatment received $T$ of each row.
    pub t: &'a [f64],
    /// Outcome columns $Y$, $n \times o$.
    pub y: MatRef<'a, f64>,
    /// Weight of each row: the cluster's inverse propensity weight times its bootstrap weight.
    pub w: &'a [f64],
    /// Cluster index of each row, in `0..n_clusters`.
    pub cluster: &'a [u32],
    /// Number of clusters $G$.
    pub n_clusters: usize,
}

impl<R: RowMatrix> MeatSource for RowData<'_, R> {
    fn meat(&self, kind: StageKind, coef: MatRef<'_, f64>, scope: Scope) -> Scoped {
        let resid = match kind {
            StageKind::Second => residuals(self.r, self.y, self.t, coef),
            StageKind::First => residuals(self.r, endogenous_regressors(self.r, self.t).as_ref(), self.z, coef),
        };
        let scores_of = |r: MatRef<'_, f64>| cluster_scores(self.r, self.w, self.z, r, self.cluster, self.n_clusters);
        match scope {
            Scope::Joint => meat_from_scores(scores_of(resid.as_ref()).as_ref(), 2 * self.r.ncols(), Scope::Joint),
            Scope::PerColumn => Scoped::PerColumn(
                (0..resid.ncols())
                    .map(|l| {
                        let scores = scores_of(resid.as_ref().subcols(l, 1));
                        scores.transpose() * &scores
                    })
                    .collect(),
            ),
        }
    }
}

/// Fits the universal model on rows.
///
/// # Errors
/// [`Error::Singular`] if a stage's $A$ is singular.
pub fn fit<R: RowMatrix>(data: &RowData<'_, R>) -> Result<Fit, Error> {
    let encouraged = data.t.iter().zip(data.z).any(|(t, z)| t != z);
    let wz: Vec<f64> = data.w.iter().zip(data.z).map(|(w, z)| w * z).collect();
    let blocks_all = Blocks::from_rows(data.r, data.w, data.t, data.y);
    let blocks_z = Blocks::from_rows(data.r, &wz, data.t, data.y);
    assemble(blocks_all, blocks_z, encouraged, data)
}

/// Residuals $Y - R\,\theta_0 - s \circ (R\,\theta_1)$ of a stage whose regressors are
/// $(R, sR)$, with $\theta = (\theta_0, \theta_1)$ split into its two $d$-row halves.
/// The second stage has $s = T$; the first stage has $s = Z$.
fn residuals<R: RowMatrix>(r: &R, y: MatRef<'_, f64>, s: &[f64], theta: MatRef<'_, f64>) -> Mat<f64> {
    let d = r.ncols();
    let fitted_0 = r.mul(theta.subrows(0, d));
    let fitted_1 = r.mul(theta.subrows(d, d));
    Mat::from_fn(y.nrows(), y.ncols(), |i, l| {
        y[(i, l)] - fitted_0[(i, l)] - s[i] * fitted_1[(i, l)]
    })
}

/// The endogenous regressors $T \circ R$, $n \times d$: the outcomes of the first stage.
fn endogenous_regressors<R: RowMatrix>(r: &R, t: &[f64]) -> Mat<f64> {
    let mut out = Mat::zeros(r.nrows(), r.ncols());
    for j in 0..r.ncols() {
        for (i, r_ij) in r.column(j).into_iter().enumerate() {
            out[(i, j)] = t[i] * r_ij;
        }
    }
    out
}

// ---- compressed data ------------------------------------------------------------------------

/// One compressed row with its stratum's instrument and cohort.
pub struct Stratum<'a> {
    /// The stratum's instrument $Z_h$, 0 or 1.
    pub z: f64,
    /// The stratum's cohort index.
    pub cohort: u32,
    /// Its compressed row.
    pub moments: &'a StratumMoments,
}

struct MomentsMeat<'a> {
    layout: Layout,
    strata: Vec<StratumMeat<'a>>,
}

impl MeatSource for MomentsMeat<'_> {
    fn meat(&self, kind: StageKind, coef: MatRef<'_, f64>, scope: Scope) -> Scoped {
        meat_from_moments(&self.layout, &self.strata, kind, coef, scope)
    }
}

/// A fit from compressed data, with the propensities it used.
#[derive(Clone, Debug)]
pub struct CompressedFit {
    /// The fitted model.
    pub fit: Fit,
    /// $\hat e_c$ per cohort (empty when unweighted).
    pub e_hat: Vec<f64>,
    /// $w_h$ per stratum.
    pub w: Vec<f64>,
}

/// Per-stratum totals: the input of [`strata_blocks`].
#[derive(Clone, Copy, Debug)]
pub struct StratumSums<'a> {
    /// The stratum's instrument $Z_h$, 0 or 1.
    pub z: f64,
    /// The stratum's cohort index.
    pub cohort: u32,
    /// Its total cluster weight: $G_h$ for the data, $\sum_{a \in h} \xi_a$ for a draw.
    pub g: f64,
    /// Its weighted sum of cluster summaries, $\sum_{a \in h} \xi_a u_a$, length $q$.
    pub first: &'a [f64],
}

/// The model's inputs from per-stratum totals: the propensities, the stratum
/// weights $w_h$, and the two block sets.
///
/// The propensity of cohort $c$ is the weighted share of its clusters with
/// $Z = 1$, $\hat e_c = \sum_{h \in c} Z_h g_h / \sum_{h \in c} g_h$: the same
/// [`inverse_propensity`] as on rows, with strata as units weighted by their
/// totals. The point estimate uses $g_h = G_h$; a bootstrap draw uses
/// $g_h = \sum_{a \in h} \xi_a$, so every draw recomputes the propensities.
/// With `weighted = false`, every $w_h$ is 1.
///
/// # Errors
/// [`Error::DegenerateCohort`] if a cohort has weight in only one arm.
pub fn strata_blocks(
    layout: &Layout,
    strata: &[StratumSums<'_>],
    n_cohorts: usize,
    weighted: bool,
) -> Result<(Blocks, Blocks, Vec<f64>, Vec<f64>), Error> {
    let (e_hat, w) = if weighted {
        let z: Vec<f64> = strata.iter().map(|h| h.z).collect();
        let cohort: Vec<u32> = strata.iter().map(|h| h.cohort).collect();
        let g: Vec<f64> = strata.iter().map(|h| h.g).collect();
        let p = inverse_propensity(&z, &cohort, &g, n_cohorts)?;
        (p.e_hat, p.w)
    } else {
        (Vec::new(), vec![1.0; strata.len()])
    };
    let mut blocks_all = Blocks::zeros(layout.d, layout.o);
    let mut blocks_z = Blocks::zeros(layout.d, layout.o);
    for (h, &w_h) in strata.iter().zip(&w) {
        let b = layout.blocks(h.first, w_h);
        if h.z == 1.0 {
            blocks_z.add_assign(&b);
        }
        blocks_all.add_assign(&b);
    }
    Ok((blocks_all, blocks_z, e_hat, w))
}

/// Whether $T \ne Z$ in some row, read from first moments: $T = Z$ exactly when
/// every $S^T$ entry equals $Z$ times the $S$ entry, in every stratum.
pub fn encouraged(layout: &Layout, strata: &[StratumSums<'_>]) -> bool {
    strata.iter().any(|h| {
        (0..layout.d).any(|j| (j..layout.d).any(|m| h.first[layout.st(j, m)] != h.z * h.first[layout.s(j, m)]))
    })
}

/// Fits the universal model from compressed rows.
///
/// # Errors
/// [`Error::Singular`] if a stage is singular, [`Error::DegenerateCohort`] if a
/// cohort has only one arm.
pub fn fit_compressed(
    layout: Layout,
    strata: &[Stratum<'_>],
    n_cohorts: usize,
    weighted: bool,
) -> Result<CompressedFit, Error> {
    let sums: Vec<StratumSums<'_>> = strata
        .iter()
        .map(|h| StratumSums {
            z: h.z,
            cohort: h.cohort,
            g: h.moments.g,
            first: &h.moments.first,
        })
        .collect();
    let (blocks_all, blocks_z, e_hat, w) = strata_blocks(&layout, &sums, n_cohorts, weighted)?;
    let encouraged = encouraged(&layout, &sums);
    let source = MomentsMeat {
        layout,
        strata: strata
            .iter()
            .zip(&w)
            .map(|(h, &w_h)| StratumMeat {
                w: w_h,
                z: h.z,
                moments: h.moments,
            })
            .collect(),
    };
    let fit = assemble(blocks_all, blocks_z, encouraged, &source)?;
    Ok(CompressedFit { fit, e_hat, w })
}

/// Compressed rows of the intent to treat model, where the treatment is the
/// instrument ($T := Z$).
///
/// Each cluster summary changes linearly: $S^T$ becomes $Z S$, and the $S^Y$
/// columns of each $T \cdot Y$ outcome become $Z$ times the columns of $Y$.
/// `treated_source[l]` is the outcome index of the raw column behind outcome
/// $l$ if $l$ is a $T \cdot Y$ column, and `None` otherwise.
pub fn intent_to_treat(
    layout: &Layout,
    moments: &StratumMoments,
    z: f64,
    treated_source: &[Option<usize>],
) -> StratumMoments {
    let q = layout.q();
    let mut src: Vec<usize> = (0..q).collect();
    let mut scale = vec![1.0; q];
    for (l, source) in treated_source.iter().enumerate() {
        if let Some(raw) = source {
            for j in 0..layout.d {
                src[layout.sy(j, l)] = layout.sy(j, *raw);
                scale[layout.sy(j, l)] = z;
            }
        }
    }
    for j in 0..layout.d {
        for m in j..layout.d {
            src[layout.st(j, m)] = layout.s(j, m);
            scale[layout.st(j, m)] = z;
        }
    }
    StratumMoments {
        g: moments.g,
        first: (0..q).map(|k| scale[k] * moments.first[src[k]]).collect(),
        second: Mat::from_fn(q, q, |a, b| scale[a] * scale[b] * moments.second[(src[a], src[b])]),
    }
}

/// Cluster-robust Wald F statistic of the excluded instruments $(Z, Zx)$ in each
/// first-stage equation: $F_j = \hat\gamma_{Z,j}^\top \hat V_{ZZ,j}^{-1} \hat\gamma_{Z,j} / d$.
pub fn first_stage_f(first: &Stage) -> Vec<f64> {
    let p = first.a.nrows();
    let d = p / 2;
    (0..first.coef.ncols())
        .map(|j| {
            let gamma_z = first.coef.as_ref().subrows(d, d).col(j).to_owned();
            let v_zz = first.cov.block(j, p).as_ref().submatrix(d, d, d, d).to_owned();
            let solved = v_zz.partial_piv_lu().solve(gamma_z.as_mat());
            let quad: f64 = (0..d).map(|i| gamma_z[i] * solved[(i, 0)]).sum();
            quad / d as f64
        })
        .collect()
}
