//! Cluster-robust covariance of a just-identified stage.
//!
//! The covariance of $\text{vec}(\hat\theta)$ is the sandwich (CSS post)
//!
//! $$(I_m \otimes A^{-1})\ \underbrace{\Big[\sum_a s_a s_a^\top\Big]}_{\text{meat}}\ (I_m \otimes A^{-1})^\top, \qquad s_a = \text{vec}\Big(\sum_{p \in a} w_{ap}\, m_{1,ap}\, \hat\varepsilon_{ap}^\top\Big),$$
//!
//! with $m_1 = (1, x, Z, Zx)$ the instruments and $\hat\varepsilon$ the $m$ residual
//! columns of the stage. Summing within a cluster before the outer product makes
//! it cluster robust (CR0); with one row per cluster it is HC0, the same formula.
//!
//! The meat has two sources, and the bread is the same for both:
//!
//! - rows: the cluster scores $s_a$ directly ([`cluster_scores`], [`meat_from_scores`]);
//! - compressed data: $s_a = C_h u_a$ is linear in the cluster summary, with
//!   $C_h = w_h P_{Z_h} L(\hat\theta)$, so the meat is
//!   $\sum_h C_h \big(\sum_{a \in h} u_a u_a^\top\big) C_h^\top$ ([`meat_from_moments`]).

use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, MatRef};

use crate::compress::StratumMoments;
use crate::layout::Layout;
use crate::rows::RowMatrix;

/// Which stage of the model: what its regressors and residual columns are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    /// Regressors $(R, TR)$; residual columns $Y - R\beta_0 - T R \beta_1$, one per outcome.
    Second,
    /// Regressors $(R, ZR)$; residual columns $TR - R\Gamma_0 - Z R \Gamma_1$, one per
    /// endogenous regressor.
    First,
}

/// How much of a $2dm \times 2dm$ matrix (meat or covariance) is kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every entry, column-major over the $m$ residual columns. Needed when an
    /// effect combines columns, as a ratio metric does.
    Joint,
    /// One $2d \times 2d$ block per column. Enough for per-equation tests.
    PerColumn,
}

/// A matrix kept at a [`Scope`].
#[derive(Clone, Debug)]
pub enum Scoped {
    /// The full $2dm \times 2dm$ matrix.
    Joint(Mat<f64>),
    /// One $2d \times 2d$ block per column.
    PerColumn(Vec<Mat<f64>>),
}

impl Scoped {
    /// The $2d \times 2d$ block of column `j`, for blocks of size `p`.
    pub fn block(&self, j: usize, p: usize) -> Mat<f64> {
        match self {
            Scoped::Joint(m) => m.as_ref().submatrix(j * p, j * p, p, p).to_owned(),
            Scoped::PerColumn(blocks) => blocks[j].clone(),
        }
    }
}

/// The $G \times 2dm$ matrix whose row $a$ is $s_a$, read column-major: for residual
/// column $l$, entries $l \cdot 2d .. (l + 1) \cdot 2d$ hold
/// $\big(\sum_p w\, R\, \hat\varepsilon_l,\ \sum_p w\, Z R\, \hat\varepsilon_l\big)$.
pub fn cluster_scores<R: RowMatrix>(
    r: &R,
    w: &[f64],
    z: &[f64],
    resid: MatRef<'_, f64>,
    cluster: &[u32],
    n_clusters: usize,
) -> Mat<f64> {
    let d = r.ncols();
    let m = resid.ncols();
    let mut scores = Mat::zeros(n_clusters, 2 * d * m);
    for l in 0..m {
        let v: Vec<f64> = (0..r.nrows()).map(|i| w[i] * resid[(i, l)]).collect();
        let vz: Vec<f64> = v.iter().zip(z).map(|(a, b)| a * b).collect();
        let base = r.group_sum(&v, cluster, n_clusters);
        let instrumented = r.group_sum(&vz, cluster, n_clusters);
        for g in 0..n_clusters {
            for j in 0..d {
                scores[(g, l * 2 * d + j)] = base[(g, j)];
                scores[(g, l * 2 * d + d + j)] = instrumented[(g, j)];
            }
        }
    }
    scores
}

/// The meat $\sum_a s_a s_a^\top$ from cluster scores, at `scope`, for blocks of size `p = 2d`.
pub fn meat_from_scores(scores: MatRef<'_, f64>, p: usize, scope: Scope) -> Scoped {
    match scope {
        Scope::Joint => Scoped::Joint(scores.transpose() * scores),
        Scope::PerColumn => Scoped::PerColumn(
            (0..scores.ncols() / p)
                .map(|l| {
                    let s = scores.subcols(l * p, p);
                    s.transpose() * s
                })
                .collect(),
        ),
    }
}

/// One stratum's inputs to [`meat_from_moments`].
pub struct StratumMeat<'a> {
    /// The stratum's weight $w_h$.
    pub w: f64,
    /// The stratum's instrument $Z_h$.
    pub z: f64,
    /// Its compressed row.
    pub moments: &'a StratumMoments,
}

/// The matrix $C_h$ ($2dm \times q$) with $s_a = C_h u_a$ for every cluster of a
/// stratum with weight `w` and instrument `z`.
///
/// Row $l \cdot 2d + j$ is $w$ times the coefficients of $E_{jl}$ in $u_a$, and row
/// $l \cdot 2d + d + j$ is $w z$ times the same, where $E$ is the cluster's sum of
/// $R^\top \hat\varepsilon$:
///
/// - second stage: $E_{jl} = S^Y_{jl} - \sum_m S_{jm} \beta_{0,ml} - \sum_m S^T_{jm} \beta_{1,ml}$;
/// - first stage: $E_{je} = S^T_{je} - \sum_m S_{jm} \Gamma_{0,me} - z \sum_m S_{jm} \Gamma_{1,me}$.
pub fn score_map(layout: &Layout, kind: StageKind, coef: MatRef<'_, f64>, w: f64, z: f64) -> Mat<f64> {
    let d = layout.d;
    let m = coef.ncols();
    let mut c = Mat::zeros(2 * d * m, layout.q());
    for l in 0..m {
        for j in 0..d {
            let mut row = vec![0.0; layout.q()];
            match kind {
                StageKind::Second => {
                    row[layout.sy(j, l)] += 1.0;
                    for k in 0..d {
                        row[layout.s(j, k)] -= coef[(k, l)];
                        row[layout.st(j, k)] -= coef[(d + k, l)];
                    }
                }
                StageKind::First => {
                    row[layout.st(j, l)] += 1.0;
                    for k in 0..d {
                        row[layout.s(j, k)] -= coef[(k, l)] + z * coef[(d + k, l)];
                    }
                }
            }
            for (col, value) in row.into_iter().enumerate() {
                c[(l * 2 * d + j, col)] = w * value;
                c[(l * 2 * d + d + j, col)] = w * z * value;
            }
        }
    }
    c
}

/// The meat from compressed rows: $\sum_h C_h \big(\sum_{a \in h} u_a u_a^\top\big) C_h^\top$.
pub fn meat_from_moments(
    layout: &Layout,
    strata: &[StratumMeat<'_>],
    kind: StageKind,
    coef: MatRef<'_, f64>,
    scope: Scope,
) -> Scoped {
    let p = 2 * layout.d;
    let m = coef.ncols();
    let mut joint = Mat::zeros(
        if scope == Scope::Joint { p * m } else { 0 },
        if scope == Scope::Joint { p * m } else { 0 },
    );
    let mut blocks = vec![Mat::zeros(p, p); if scope == Scope::PerColumn { m } else { 0 }];
    for h in strata {
        if h.moments.g == 0.0 {
            continue;
        }
        let c = score_map(layout, kind, coef, h.w, h.z);
        let second = h.moments.second.as_ref();
        match scope {
            Scope::Joint => joint += &c * second * c.transpose(),
            Scope::PerColumn => {
                for (l, block) in blocks.iter_mut().enumerate() {
                    let cl = c.as_ref().subrows(l * p, p);
                    *block += cl * second * cl.transpose();
                }
            }
        }
    }
    match scope {
        Scope::Joint => Scoped::Joint(joint),
        Scope::PerColumn => Scoped::PerColumn(blocks),
    }
}

/// The covariance $(I_m \otimes A^{-1})\, \text{meat}\, (I_m \otimes A^{-1})^\top$, at the
/// meat's scope, for the $2d \times 2d$ matrix $A$ of the stage.
pub fn sandwich(a: MatRef<'_, f64>, meat: &Scoped) -> Scoped {
    let p = a.nrows();
    let a_inv = a.partial_piv_lu().inverse();
    match meat {
        Scoped::Joint(meat) => {
            let m = meat.nrows() / p;
            let mut cov = Mat::zeros(p * m, p * m);
            for l in 0..m {
                for k in 0..m {
                    let block = meat.as_ref().submatrix(l * p, k * p, p, p);
                    let bread = &a_inv * block * a_inv.transpose();
                    cov.as_mut().submatrix_mut(l * p, k * p, p, p).copy_from(&bread);
                }
            }
            Scoped::Joint(cov)
        }
        Scoped::PerColumn(blocks) => Scoped::PerColumn(blocks.iter().map(|b| &a_inv * b * a_inv.transpose()).collect()),
    }
}
