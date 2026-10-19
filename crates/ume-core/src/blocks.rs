//! The sufficient statistic of the universal model, and what it determines.
//!
//! For any set of rows with weights $v$, three blocks summarize everything the
//! model can learn from them (CSS post, "What the model needs"):
//!
//! $$S^Y = R^\top V Y \ (d \times o), \qquad S = R^\top V R \ (d \times d), \qquad S^T = R^\top V\,\mathrm{diag}(T)\,R \ (d \times d).$$
//!
//! [`Blocks`] is additive, so it is the same object whether it comes from the rows
//! (one weighted product) or from compressed data (a weighted sum of stratum
//! sums). Two block sets determine the fit: one over all rows with weights $w$,
//! and one over the instrument arm $Z = 1$, with weights $w Z$. From them:
//!
//! - [`NormalEquations::second_stage`] gives $A = M_1^\top W M_2$ and $M_1^\top W Y$;
//! - [`NormalEquations::first_stage`] gives $M_1^\top W M_1$ and $M_1^\top W (T R)$,
//!   the special case of the second stage with $T$ replaced by $Z$;
//! - [`Compliers::from_blocks`] gives the take-up in each arm, the compliance
//!   rate $\pi_C$ and the compliers' mean $\bar x_C$.

use faer::{Mat, MatRef};

use crate::rows::RowMatrix;

/// The blocks $(S^Y, S, S^T)$ of a weighted set of rows.
#[derive(Clone, Debug)]
pub struct Blocks {
    /// $S^Y = R^\top V Y$, $d \times o$.
    pub s_y: Mat<f64>,
    /// $S = R^\top V R$, $d \times d$.
    pub s: Mat<f64>,
    /// $S^T = R^\top V\,\mathrm{diag}(T)\,R$, $d \times d$.
    pub s_t: Mat<f64>,
}

impl Blocks {
    /// Zero blocks for $d$ columns of $R$ and $o$ outcome columns.
    pub fn zeros(d: usize, o: usize) -> Self {
        Self {
            s_y: Mat::zeros(d, o),
            s: Mat::zeros(d, d),
            s_t: Mat::zeros(d, d),
        }
    }

    /// Blocks of the rows of `r` with weights `v`, treatment `t` and outcomes `y`.
    pub fn from_rows<R: RowMatrix>(r: &R, v: &[f64], t: &[f64], y: MatRef<'_, f64>) -> Self {
        let vt: Vec<f64> = v.iter().zip(t).map(|(vi, ti)| vi * ti).collect();
        Self {
            s_y: r.weighted_cross(v, y),
            s: r.weighted_gram(v),
            s_t: r.weighted_gram(&vt),
        }
    }

    /// Number of columns of $R$, $d$.
    pub fn d(&self) -> usize {
        self.s.nrows()
    }

    /// Number of outcome columns, $o$.
    pub fn o(&self) -> usize {
        self.s_y.ncols()
    }

    /// `self - other`, block by block. The blocks of the arm $Z = 0$ are `all - z`.
    pub fn minus(&self, other: &Blocks) -> Blocks {
        Blocks {
            s_y: &self.s_y - &other.s_y,
            s: &self.s - &other.s,
            s_t: &self.s_t - &other.s_t,
        }
    }

    /// Adds `other` into `self`. Blocks form a commutative monoid under this operation.
    pub fn add_assign(&mut self, other: &Blocks) {
        self.s_y += &other.s_y;
        self.s += &other.s;
        self.s_t += &other.s_t;
    }

    /// Weighted share of each entry of $T (1, x)$ in these rows: column 0 of $S^T$
    /// divided by the total weight $S_{00}$. Entry 0 is the take-up of the treatment.
    fn treated_shares(&self) -> Vec<f64> {
        let total = self.s[(0, 0)];
        (0..self.d()).map(|j| self.s_t[(j, 0)] / total).collect()
    }
}

/// The just-identified normal equations $A\,\theta = b$ of one stage.
#[derive(Clone, Debug)]
pub struct NormalEquations {
    /// $A$, square of size $2d$.
    pub a: Mat<f64>,
    /// $b$, $2d \times m$, one column per outcome of the stage.
    pub b: Mat<f64>,
}

impl NormalEquations {
    /// The second stage: instruments $(1, x, Z, Zx)$ against regressors $(1, x, T, Tx)$,
    ///
    /// $$A = \begin{pmatrix} S & S^T \\ S_Z & S^T_Z \end{pmatrix}, \qquad b = \begin{pmatrix} S^Y \\ S^Y_Z \end{pmatrix},$$
    ///
    /// where unsubscripted blocks are over all rows and $Z$-subscripted blocks are
    /// over the arm $Z = 1$.
    pub fn second_stage(all: &Blocks, z: &Blocks) -> Self {
        Self {
            a: stack_2x2(&all.s, &all.s_t, &z.s, &z.s_t),
            b: stack_rows(&all.s_y, &z.s_y),
        }
    }

    /// The first stage: the endogenous regressors $T (1, x)$ on the instruments
    /// $(1, x, Z, Zx)$. It is the second stage with $T$ replaced by $Z$ in the
    /// regressors (so $M_2 = M_1$, and $S^T$ becomes $S_Z$ because $Z^2 = Z$), and
    /// with the outcomes $T (1, x)$, whose cross products are $S^T$:
    ///
    /// $$A = \begin{pmatrix} S & S_Z \\ S_Z & S_Z \end{pmatrix}, \qquad b = \begin{pmatrix} S^T \\ S^T_Z \end{pmatrix}.$$
    pub fn first_stage(all: &Blocks, z: &Blocks) -> Self {
        Self {
            a: stack_2x2(&all.s, &z.s, &z.s, &z.s),
            b: stack_rows(&all.s_t, &z.s_t),
        }
    }
}

/// $\begin{pmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{pmatrix}$ for four $d \times d$ blocks.
fn stack_2x2(a00: &Mat<f64>, a01: &Mat<f64>, a10: &Mat<f64>, a11: &Mat<f64>) -> Mat<f64> {
    let d = a00.nrows();
    Mat::from_fn(2 * d, 2 * d, |i, j| match (i < d, j < d) {
        (true, true) => a00[(i, j)],
        (true, false) => a01[(i, j - d)],
        (false, true) => a10[(i - d, j)],
        (false, false) => a11[(i - d, j - d)],
    })
}

/// $\begin{pmatrix} top \\ bottom \end{pmatrix}$ for two blocks with the same number of columns.
fn stack_rows(top: &Mat<f64>, bottom: &Mat<f64>) -> Mat<f64> {
    let d = top.nrows();
    Mat::from_fn(
        2 * d,
        top.ncols(),
        |i, l| if i < d { top[(i, l)] } else { bottom[(i - d, l)] },
    )
}

/// Who the compliers are, from the blocks of the two arms.
#[derive(Clone, Debug)]
pub struct Compliers {
    /// Weighted share of rows with $T = 1$ in the arm $Z = 1$.
    pub take_up_z1: f64,
    /// Weighted share of rows with $T = 1$ in the arm $Z = 0$.
    pub take_up_z0: f64,
    /// Compliance rate $\pi_C$ = `take_up_z1 - take_up_z0` (a share of rows).
    pub pi_c: f64,
    /// Compliers' mean of the centered covariates, $\bar x_C - x_{ref}$, length $k$.
    pub x_c: Vec<f64>,
    /// Weighted mean of the centered covariates over all rows, length $k$.
    pub x_mean: Vec<f64>,
}

impl Compliers {
    /// Compliers from the blocks over all rows and over the arm $Z = 1$.
    ///
    /// With encouragement, $\bar x_C$ is the Wald ratio of intent to treat effects
    /// on $T x$ and on $T$ (CSS post):
    /// $\bar x_C = \frac{E_w[Tx \mid Z=1] - E_w[Tx \mid Z=0]}{E_w[T \mid Z=1] - E_w[T \mid Z=0]}$.
    /// Without encouragement ($T = Z$) every row is a complier, and $\bar x_C$ is the
    /// weighted mean of $x$ over all rows, as in Lin's regression. This is the one
    /// place where the engine distinguishes the two designs: the Wald ratio would
    /// still be a valid estimate when $T = Z$, but it would use only the arm $Z = 1$,
    /// and it would no longer match the textbook analysis.
    pub fn from_blocks(all: &Blocks, z: &Blocks, encouraged: bool) -> Self {
        let z0 = all.minus(z);
        let (arm1, arm0) = (z.treated_shares(), z0.treated_shares());
        let diff: Vec<f64> = arm1.iter().zip(&arm0).map(|(a, b)| a - b).collect();
        let total = all.s[(0, 0)];
        let x_mean: Vec<f64> = (1..all.d()).map(|j| all.s[(j, 0)] / total).collect();
        let x_c = if encouraged {
            diff[1..].iter().map(|v| v / diff[0]).collect()
        } else {
            x_mean.clone()
        };
        Self {
            take_up_z1: arm1[0],
            take_up_z0: arm0[0],
            pi_c: diff[0],
            x_c,
            x_mean,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rows::Dense;

    /// Four rows, one covariate: Z = (1, 1, 0, 0), T = (1, 0, 0, 0).
    fn example() -> (Blocks, Blocks) {
        let values = [1.0, 2.0, 1.0, 4.0, 1.0, 1.0, 1.0, 3.0];
        let r = Dense::from_row_major(&values, 4, 2);
        let (w, z, t) = ([1.0; 4], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        let y = Mat::from_fn(4, 1, |i, _| i as f64);
        let wz: Vec<f64> = w.iter().zip(&z).map(|(a, b)| a * b).collect();
        (
            Blocks::from_rows(&r, &w, &t, y.as_ref()),
            Blocks::from_rows(&r, &wz, &t, y.as_ref()),
        )
    }

    #[test]
    fn blocks_are_additive() {
        let (all, z) = example();
        let mut sum = z.clone();
        sum.add_assign(&all.minus(&z));
        for (a, b) in [(&sum.s, &all.s), (&sum.s_t, &all.s_t)] {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((a[(i, j)] - b[(i, j)]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn compliers_follow_the_wald_ratio() {
        let (all, z) = example();
        let c = Compliers::from_blocks(&all, &z, true);
        assert!((c.take_up_z1 - 0.5).abs() < 1e-12);
        assert!(c.take_up_z0.abs() < 1e-12);
        assert!((c.pi_c - 0.5).abs() < 1e-12);
        // E[Tx | Z=1] = 2 / 2 = 1, E[Tx | Z=0] = 0, so x_C = 1 / 0.5 = 2: the one treated row.
        assert!((c.x_c[0] - 2.0).abs() < 1e-12);
        assert!((c.x_mean[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn first_stage_is_the_second_stage_with_t_equal_to_z() {
        let (all, z) = example();
        let first = NormalEquations::first_stage(&all, &z);
        // With T replaced by Z, S^T over all rows is S over the arm Z = 1.
        let pretend = Blocks {
            s_y: all.s_y.clone(),
            s: all.s.clone(),
            s_t: z.s.clone(),
        };
        let pretend_z = Blocks {
            s_y: z.s_y.clone(),
            s: z.s.clone(),
            s_t: z.s.clone(),
        };
        let second = NormalEquations::second_stage(&pretend, &pretend_z);
        assert_eq!(first.a, second.a);
    }
}
