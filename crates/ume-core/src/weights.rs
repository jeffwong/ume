//! Bootstrap weights: one weight per cluster and draw.
//!
//! The classical, Poisson and Bayesian bootstraps are the same computation with
//! differently distributed weights (bootstrap post): each draw refits the model
//! with every cluster counted $\xi_a^{(b)}$ times. The weight goes on the
//! cluster, never on the row (BUM post). Every scheme has mean 1 and variance
//! 1 (the multinomial $1 - 1/G$), which is why all of them linearize to the
//! cluster sandwich.

use crate::rng::Stream;

/// How the weights of a draw are distributed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// $\xi_a \sim$ Exp(1), independent: the Bayesian bootstrap, a posterior.
    Bayesian,
    /// $\xi_a \sim$ Poisson(1), independent: the Poisson bootstrap.
    Poisson,
    /// $(\xi_1, \dots, \xi_G) \sim$ Multinomial$(G; 1/G, \dots, 1/G)$: the classical bootstrap
    /// of clusters. The counts are coupled, so they are drawn in pass order.
    Multinomial,
}

impl Scheme {
    /// The stream tag of the scheme (decision 0018).
    pub fn tag(self) -> u64 {
        match self {
            Scheme::Bayesian => 0,
            Scheme::Poisson => 1,
            Scheme::Multinomial => 2,
        }
    }
}

/// $P(X \le k)$ inversion for Poisson(1): the smallest $k$ with $F(k) \ge u$.
fn poisson_one(u: f64) -> f64 {
    let mut p = (-1.0f64).exp();
    let (mut k, mut cdf) = (0u32, p);
    while u > cdf && k < 64 {
        k += 1;
        p /= k as f64;
        cdf += p;
    }
    k as f64
}

/// Inversion for Binomial$(n, p)$: the smallest $k$ with $F(k) \ge u$.
fn binomial(u: f64, n: u64, p: f64) -> u64 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    let ratio = p / (1.0 - p);
    let mut pk = (1.0 - p).powf(n as f64);
    let (mut k, mut cdf) = (0u64, pk);
    while u > cdf && k < n {
        pk *= (n - k) as f64 / (k + 1) as f64 * ratio;
        k += 1;
        cdf += pk;
    }
    k
}

/// Draws the $B$ weights of each cluster, in pass order.
#[derive(Clone, Debug)]
pub struct Weights {
    scheme: Scheme,
    seed: u64,
    uniforms: Vec<f64>,
    /// Multinomial only: clusters not yet drawn, and draws left per draw $b$.
    clusters_left: u64,
    left: Vec<u64>,
}

impl Weights {
    /// Weights for `b` draws. `n_clusters` ($G$) is required by the multinomial
    /// scheme and ignored by the others.
    pub fn new(scheme: Scheme, seed: u64, b: usize, n_clusters: Option<u64>) -> Self {
        let g = n_clusters.unwrap_or(0);
        Self {
            scheme,
            seed,
            uniforms: vec![0.0; b],
            clusters_left: g,
            left: vec![g; b],
        }
    }

    /// Number of draws $B$.
    pub fn draws(&self) -> usize {
        self.uniforms.len()
    }

    /// The scheme.
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// Weights with the same scheme, seed and draws, starting fresh (for another
    /// worker). Independent schemes do not depend on state, so the weights of any
    /// cluster are the same in every worker.
    pub fn fresh(&self) -> Self {
        Self::new(self.scheme, self.seed, self.uniforms.len(), Some(self.clusters_left))
    }

    /// Writes the next cluster's weights $\xi^{(0..B)}$ into `out`.
    ///
    /// # Panics
    /// For the multinomial scheme, if more clusters arrive than were declared.
    pub fn next(&mut self, cluster_key: u64, out: &mut [f64]) {
        let stream = Stream::new(self.seed, cluster_key, self.scheme.tag());
        stream.uniforms(&mut self.uniforms);
        match self.scheme {
            Scheme::Bayesian => {
                for (xi, u) in out.iter_mut().zip(&self.uniforms) {
                    *xi = -u.ln();
                }
            }
            Scheme::Poisson => {
                for (xi, &u) in out.iter_mut().zip(&self.uniforms) {
                    *xi = poisson_one(u);
                }
            }
            Scheme::Multinomial => {
                assert!(
                    self.clusters_left > 0,
                    "more clusters than declared for the multinomial bootstrap"
                );
                let p = 1.0 / self.clusters_left as f64;
                for ((xi, &u), left) in out.iter_mut().zip(&self.uniforms).zip(&mut self.left) {
                    let c = binomial(u, *left, p);
                    *left -= c;
                    *xi = c as f64;
                }
                self.clusters_left -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Bootstrap weights have the moments of their distributions, multinomial counts sum to G, and a weight depends only on (seed, cluster, draw).

    use super::*;

    fn moments(scheme: Scheme, n: usize) -> (f64, f64, f64) {
        let mut w = Weights::new(scheme, 7, 1, Some(n as u64));
        let mut xs = Vec::with_capacity(n);
        let mut out = [0.0];
        for a in 0..n {
            w.next(a as u64, &mut out);
            xs.push(out[0]);
        }
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let third = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;
        (mean, var, third)
    }

    /// Exp(1) weights: mean 1, variance 1, third central moment 2.
    #[test]
    fn bayesian_weights_have_the_moments_of_exp_1() {
        let (m, v, t) = moments(Scheme::Bayesian, 200_000);
        assert!(
            (m - 1.0).abs() < 0.01 && (v - 1.0).abs() < 0.02 && (t - 2.0).abs() < 0.1,
            "{m} {v} {t}"
        );
    }

    /// Poisson(1) weights: mean 1, variance 1, third central moment 1.
    #[test]
    fn poisson_weights_have_the_moments_of_poisson_1() {
        let (m, v, t) = moments(Scheme::Poisson, 200_000);
        assert!(
            (m - 1.0).abs() < 0.01 && (v - 1.0).abs() < 0.02 && (t - 1.0).abs() < 0.05,
            "{m} {v} {t}"
        );
    }

    /// In every draw, the multinomial counts over G clusters sum to G.
    #[test]
    fn multinomial_counts_sum_to_the_number_of_clusters() {
        let g = 1000;
        let mut w = Weights::new(Scheme::Multinomial, 3, 50, Some(g));
        let mut totals = vec![0.0; 50];
        let mut out = vec![0.0; 50];
        for a in 0..g {
            w.next(a * 13, &mut out);
            for (t, x) in totals.iter_mut().zip(&out) {
                *t += x;
            }
        }
        assert!(totals.iter().all(|&t| t == g as f64));
    }

    /// A cluster's weights are the same whatever clusters came before it.
    #[test]
    fn weights_depend_only_on_seed_cluster_and_draw() {
        let (mut a, mut b) = (
            Weights::new(Scheme::Bayesian, 5, 8, None),
            Weights::new(Scheme::Bayesian, 5, 8, None),
        );
        let (mut x, mut y, mut skip) = (vec![0.0; 8], vec![0.0; 8], vec![0.0; 8]);
        a.next(42, &mut x);
        b.next(41, &mut skip);
        b.next(42, &mut y);
        assert_eq!(x, y);
    }
}
