//! Standard errors in two passes, without second moments: delta vectors early.
//!
//! The variance of an effect is a sum of squared influences over clusters,
//! $\sum_a \psi_a^2$, with $\psi_a = c_h^\top v_a$ linear in the cluster summary
//! $v_a = (1, u_a)$ and $c_h$ the same for every cluster of a stratum (CSS post,
//! "Delta vectors earlier"). So:
//!
//! 1. **pass 1** ([`FirstMomentsFold`]): $G_h$ and $\sum_{a \in h} u_a$, enough for
//!    the point estimate and every $c_h$;
//! 2. **pass 2** ([`ScoreFold`]): rebuild each $u_a$ and add $(c_h^\top v_a)^2$, one
//!    total per effect.
//!
//! Storage is $Hq$ numbers plus one per effect, instead of $Hq^2$, at the price of
//! choosing the effects before the second pass. The same fold serves both
//! standard errors:
//!
//! - the **sandwich**: $c_h = \big(0,\ C_h^\top (I_o \otimes A^{-1})^\top g\big)$, with $g$ the
//!   gradient of the effect with respect to $\text{vec}(\hat\beta)$ ([`sandwich_projection`]);
//! - the **linearized** SE: $c_h$ is the gradient of the effect with respect to the
//!   stratum's sums $\sum_a \xi_a v_a$ at $\xi = 1$, which also moves $\hat e$ and $\bar x_C$.

use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, MatRef};

use crate::compress::{Fold, Pass};
use crate::layout::Layout;
use crate::sandwich::{StageKind, score_map};

/// Pass 1: per stratum, $G_h$ and $\sum_{a \in h} u_a$.
pub struct FirstMomentsFold {
    q: usize,
    /// $(G_h, \sum u_a)$ per stratum index, length $1 + q$ each.
    pub strata: Vec<Vec<f64>>,
}

impl Fold for FirstMomentsFold {
    fn fold(&mut self, _key: u64, stratum: u32, u: &[f64], nz: &[usize]) {
        let h = stratum as usize;
        while self.strata.len() <= h {
            self.strata.push(vec![0.0; 1 + self.q]);
        }
        let s = &mut self.strata[h];
        s[0] += 1.0;
        for &k in nz {
            s[k + 1] += u[k];
        }
    }
}

impl crate::compress::Merge for FirstMomentsFold {
    fn merge(&mut self, other: Self) {
        for (h, theirs) in other.strata.into_iter().enumerate() {
            if self.strata.len() <= h {
                self.strata.push(vec![0.0; 1 + self.q]);
            }
            for (a, b) in self.strata[h].iter_mut().zip(theirs) {
                *a += b;
            }
        }
    }
}

impl FirstMomentsFold {
    /// A fresh fold for another worker.
    pub fn fresh(&self) -> Self {
        Self {
            q: self.q,
            strata: Vec::new(),
        }
    }
}

/// The first pass of delta vectors early.
pub type FirstMomentsPass = Pass<FirstMomentsFold>;

impl Pass<FirstMomentsFold> {
    /// A first-moments pass for summaries with the given layout.
    pub fn first_moments(layout: Layout) -> Self {
        Self::with_fold(
            layout,
            FirstMomentsFold {
                q: layout.q(),
                strata: Vec::new(),
            },
        )
    }

    /// Flushes the last cluster; returns $(G_h, \sum u_a)$ per stratum and the rows read.
    pub fn finish_first(self) -> (Vec<Vec<f64>>, u64) {
        let (fold, n_rows) = self.into_fold();
        (fold.strata, n_rows)
    }
}

/// Pass 2: per effect, $\sum_a (c_{h(a)}^\top v_a)^2$.
pub struct ScoreFold {
    /// Per stratum, an $E \times (1 + q)$ matrix: row $e$ is $c_h$ of effect $e$.
    projections: Vec<Mat<f64>>,
    /// $\sum_a \psi_a^2$ per effect.
    pub totals: Vec<f64>,
}

impl Fold for ScoreFold {
    fn fold(&mut self, _key: u64, stratum: u32, u: &[f64], nz: &[usize]) {
        let c = &self.projections[stratum as usize];
        for (e, total) in self.totals.iter_mut().enumerate() {
            let psi = c[(e, 0)] + nz.iter().map(|&k| c[(e, k + 1)] * u[k]).sum::<f64>();
            *total += psi * psi;
        }
    }
}

impl crate::compress::Merge for ScoreFold {
    fn merge(&mut self, other: Self) {
        for (a, b) in self.totals.iter_mut().zip(other.totals) {
            *a += b;
        }
    }
}

impl ScoreFold {
    /// A fresh fold with the same projections, for another worker.
    pub fn fresh(&self) -> Self {
        Self {
            projections: self.projections.clone(),
            totals: vec![0.0; self.totals.len()],
        }
    }
}

/// The second pass of delta vectors early.
pub type ScorePass = Pass<ScoreFold>;

impl Pass<ScoreFold> {
    /// A second pass with the projections $c_h$ of $E$ effects: one $E \times (1 + q)$
    /// matrix per stratum index.
    pub fn scores(layout: Layout, projections: Vec<Mat<f64>>) -> Self {
        let n_effects = projections.first().map_or(0, |m| m.nrows());
        Self::with_fold(
            layout,
            ScoreFold {
                projections,
                totals: vec![0.0; n_effects],
            },
        )
    }

    /// Flushes the last cluster; returns the variance $\sum_a \psi_a^2$ of each effect.
    pub fn finish_scores(self) -> Vec<f64> {
        self.into_fold().0.totals
    }
}

/// The sandwich projection of one stratum: $c_h$ with $\psi_a = c_h^\top v_a$ and
/// $\sum_a \psi_a^2 = g^\top \widehat{\text{Cov}}(\text{vec}\,\hat\beta)\, g$.
///
/// $c_h = \big(0,\ C_h^\top (I_o \otimes A^{-1})^\top g\big)$, where $C_h$ is the stratum's
/// [`score_map`] (weight `w`, instrument `z`). Length $1 + q$; entry 0 is 0 because
/// the sandwich does not move with the stratum's total weight.
pub fn sandwich_projection(
    layout: &Layout,
    beta: MatRef<'_, f64>,
    a: MatRef<'_, f64>,
    grad: &[f64],
    w: f64,
    z: f64,
) -> Vec<f64> {
    let p = a.nrows();
    let o = beta.ncols();
    let a_inv = a.partial_piv_lu().inverse();
    // v = (I_o ⊗ A^{-1})^T g, block by block.
    let mut v = vec![0.0; p * o];
    for l in 0..o {
        for i in 0..p {
            v[l * p + i] = (0..p).map(|j| a_inv[(j, i)] * grad[l * p + j]).sum();
        }
    }
    let c = score_map(layout, StageKind::Second, beta, w, z);
    let mut out = vec![0.0; 1 + layout.q()];
    for k in 0..layout.q() {
        out[k + 1] = (0..p * o).map(|r| c[(r, k)] * v[r]).sum();
    }
    out
}
