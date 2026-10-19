//! Numeric engine of ume: the universal model.
//!
//! One model covers the experiment designs that ume analyzes: weighted two
//! stage least squares with every covariate interacted with treatment,
//! cluster-robust covariance, and several outcome columns. AB tests,
//! cluster-randomized tests, encouragement designs, adaptive (ramped)
//! assignment, and mean or ratio metrics are special cases. The crate is
//! organized as primitives of that one model, never as one code path per design:
//!
//! | Module | Primitive |
//! |---|---|
//! | [`rows`] | [`RowMatrix`](rows::RowMatrix): the design matrix $R = (1, x - x_{ref})$, dense or sparse |
//! | [`blocks`] | [`Blocks`](blocks::Blocks) $(S^Y, S, S^T)$, the additive sufficient statistic; the normal equations of both stages; the compliers |
//! | [`layout`] | [`Layout`](layout::Layout) of a cluster summary $u_a$: the per-cluster blocks |
//! | [`compress`] | one pass over rows to per-stratum $G_h$, $\sum u_a$, $\sum u_a u_a^\top$; mergeable |
//! | [`propensity`] | cohort propensities and inverse propensity weights |
//! | [`sandwich`] | the meat (from cluster scores or from second moments) and the sandwich |
//! | [`fit`] | the universal model, assembled once, fed by rows or by compressed data |
//! | [`rng`], [`weights`] | Philox4x64-10 streams keyed by (seed, cluster, draw); Bayesian, Poisson, multinomial weights |
//! | [`parallel`] | [`ParallelPass`](parallel::ParallelPass): any pass on several threads, clusters routed by key hash, merged exactly |
//! | [`early`] | delta vectors early: two passes ([`FirstMomentsFold`](early::FirstMomentsFold), [`ScoreFold`](early::ScoreFold)) give $\sum_a \psi_a^2$ without second moments |
//! | [`bootstrap`] | draws: exact (a [`Pass`](compress::Pass) whose fold adds $\xi^{(b)} v_a$) or Gaussian (from the compressed moments), fit by one `fit_draws` |
//!
//! The crate is pure Rust with no Python dependency. Bindings live in the
//! `ume-python` crate.

pub mod blocks;
pub mod bootstrap;
pub mod compress;
pub mod early;
pub mod fit;
pub mod layout;
pub mod parallel;
pub mod propensity;
pub mod rng;
pub mod rows;
pub mod sandwich;
pub mod weights;

use std::fmt;

/// Errors of the engine.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The normal equations of a stage are singular: a covariate is constant or
    /// collinear, an arm has no rows, or no row takes the treatment.
    Singular,
    /// A cohort has a propensity of 0 or 1, so one arm is empty and its inverse
    /// propensity weight is undefined. Holds the cohort index.
    DegenerateCohort(usize),
    /// Rows of a cluster are not contiguous: the cluster (by key) reappeared
    /// after its rows ended. Sort or shuffle the rows by cluster first.
    NotContiguous(u64),
    /// Rows of one cluster are in different strata (instrument or cohort varies).
    StratumChanges(u64),
    /// Compressed data with different layouts cannot be merged.
    LayoutMismatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Singular => write!(
                f,
                "the model's normal equations are singular: check for constant or collinear covariates, \
                 an empty arm, or no treated rows"
            ),
            Error::DegenerateCohort(c) => {
                write!(f, "cohort {c} has a propensity of 0 or 1: one arm is empty")
            }
            Error::NotContiguous(key) => write!(
                f,
                "the rows of cluster {key} are not contiguous: sort or shuffle the rows by cluster before compressing"
            ),
            Error::StratumChanges(key) => {
                write!(
                    f,
                    "the rows of cluster {key} are in different strata: the instrument and cohort must be constant"
                )
            }
            Error::LayoutMismatch => write!(f, "compressed data with different layouts cannot be merged"),
        }
    }
}

impl std::error::Error for Error {}
