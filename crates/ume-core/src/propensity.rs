//! Cohort propensities and inverse propensity weights, per cluster.
//!
//! Within a cohort $c$ the instrument is randomized with probability $e_c$ (A2).
//! The estimate is the weighted share of the cohort's **clusters** with $Z = 1$,
//!
//! $$\hat e_c = \frac{\sum_{a \in c} \xi_a Z_a}{\sum_{a \in c} \xi_a},$$
//!
//! and each cluster gets $w_a = 1/\hat e_c$ if $Z_a = 1$ and $1/(1 - \hat e_c)$ if not.
//! The cluster weights $\xi_a$ are 1 for the point estimate and a bootstrap draw
//! otherwise (BUM post: every draw recomputes the propensities).

use crate::Error;

/// Propensities per cohort and inverse propensity weights per cluster.
#[derive(Clone, Debug)]
pub struct Propensity {
    /// $\hat e_c$ for each cohort.
    pub e_hat: Vec<f64>,
    /// $w_a$ for each cluster.
    pub w: Vec<f64>,
}

/// Computes $\hat e_c$ and $w_a$ from each cluster's instrument `z`, cohort index
/// `cohort` (in `0..n_cohorts`) and weight `xi`.
///
/// # Errors
/// [`Error::DegenerateCohort`] if a cohort has $\hat e_c$ equal to 0 or 1, or no weight.
pub fn inverse_propensity(z: &[f64], cohort: &[u32], xi: &[f64], n_cohorts: usize) -> Result<Propensity, Error> {
    let (mut treated, mut total) = (vec![0.0; n_cohorts], vec![0.0; n_cohorts]);
    for ((&z_a, &c), &xi_a) in z.iter().zip(cohort).zip(xi) {
        treated[c as usize] += xi_a * z_a;
        total[c as usize] += xi_a;
    }
    let e_hat: Vec<f64> = treated.iter().zip(&total).map(|(t, n)| t / n).collect();
    if let Some(c) = e_hat.iter().position(|e| !(*e > 0.0 && *e < 1.0)) {
        return Err(Error::DegenerateCohort(c));
    }
    let w = z
        .iter()
        .zip(cohort)
        .map(|(&z_a, &c)| {
            let e = e_hat[c as usize];
            if z_a == 1.0 { 1.0 / e } else { 1.0 / (1.0 - e) }
        })
        .collect();
    Ok(Propensity { e_hat, w })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propensity_is_the_share_of_clusters() {
        // Cohort 0 has Z = (1, 0, 0), cohort 1 has Z = (0, 1, 1).
        let z = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let cohort = [0, 0, 0, 1, 1, 1];
        let p = inverse_propensity(&z, &cohort, &[1.0; 6], 2).unwrap();
        assert!((p.e_hat[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((p.e_hat[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((p.w[0] - 3.0).abs() < 1e-12);
        assert!((p.w[2] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn a_cohort_without_controls_is_an_error() {
        let p = inverse_propensity(&[1.0, 1.0], &[0, 0], &[1.0, 1.0], 1);
        assert!(matches!(p, Err(Error::DegenerateCohort(0))));
    }
}
