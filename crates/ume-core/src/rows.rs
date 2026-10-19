//! The design matrix of the rows, $R = (1, x - x_{ref})$, dense or sparse.
//!
//! Every quantity of the universal model touches the rows only through four
//! products with $R$: a weighted Gram matrix $R^\top \mathrm{diag}(v) R$, a weighted
//! cross product $R^\top \mathrm{diag}(v) Y$, the product $R\,b$, and per-cluster sums
//! of weighted rows. [`RowMatrix`] is that interface. The engine is generic over
//! it, so a dense or a sparse design matrix is a choice of type, not a fork in
//! the engine. Sparse matters when categorical covariates have many levels.
//!
//! The small matrices these products return ($d \times d$, $d \times o$, $G \times d$)
//! are dense.

use faer::sparse::{SparseRowMatRef, SymbolicSparseRowMatRef};
use faer::{Mat, MatRef};

/// A row-oriented $n \times d$ design matrix: one row per input row, one column per
/// entry of $(1, x - x_{ref})$.
pub trait RowMatrix {
    /// Number of rows $n$.
    fn nrows(&self) -> usize;

    /// Number of columns $d = k + 1$.
    fn ncols(&self) -> usize;

    /// $R^\top \mathrm{diag}(v) R$, a symmetric $d \times d$ matrix.
    fn weighted_gram(&self, v: &[f64]) -> Mat<f64>;

    /// $R^\top \mathrm{diag}(v) Y$ for a dense $n \times o$ matrix $Y$; returns $d \times o$.
    fn weighted_cross(&self, v: &[f64], y: MatRef<'_, f64>) -> Mat<f64>;

    /// $R\,b$ for a dense $d \times m$ matrix $b$; returns $n \times m$.
    fn mul(&self, b: MatRef<'_, f64>) -> Mat<f64>;

    /// Per-group sums of weighted rows: row $g$ of the $G \times d$ result is
    /// $\sum_{i : \text{group}_i = g} v_i R_i$.
    fn group_sum(&self, v: &[f64], group: &[u32], n_groups: usize) -> Mat<f64>;

    /// Column $j$ of $R$, as a dense vector of length $n$.
    fn column(&self, j: usize) -> Vec<f64>;

    /// Calls `f(j, R_ij)` for each nonzero entry of row `i`, in increasing `j`.
    fn for_each_in_row(&self, i: usize, f: impl FnMut(usize, f64));
}

/// A dense design matrix, a view of an $n \times d$ faer matrix.
#[derive(Clone, Copy)]
pub struct Dense<'a> {
    r: MatRef<'a, f64>,
}

impl<'a> Dense<'a> {
    /// Wraps a dense $n \times d$ matrix.
    pub fn new(r: MatRef<'a, f64>) -> Self {
        Self { r }
    }

    /// Views a row-major (C-order) buffer of length `nrows * ncols`, as numpy stores it.
    ///
    /// # Panics
    /// If `values.len() != nrows * ncols`.
    pub fn from_row_major(values: &'a [f64], nrows: usize, ncols: usize) -> Self {
        Self {
            r: MatRef::from_row_major_slice(values, nrows, ncols),
        }
    }
}

impl RowMatrix for Dense<'_> {
    fn nrows(&self) -> usize {
        self.r.nrows()
    }

    fn ncols(&self) -> usize {
        self.r.ncols()
    }

    fn weighted_gram(&self, v: &[f64]) -> Mat<f64> {
        let scaled = Mat::from_fn(self.r.nrows(), self.r.ncols(), |i, j| v[i] * self.r[(i, j)]);
        self.r.transpose() * scaled
    }

    fn weighted_cross(&self, v: &[f64], y: MatRef<'_, f64>) -> Mat<f64> {
        let scaled = Mat::from_fn(y.nrows(), y.ncols(), |i, l| v[i] * y[(i, l)]);
        self.r.transpose() * scaled
    }

    fn mul(&self, b: MatRef<'_, f64>) -> Mat<f64> {
        self.r * b
    }

    fn group_sum(&self, v: &[f64], group: &[u32], n_groups: usize) -> Mat<f64> {
        let mut out = Mat::zeros(n_groups, self.r.ncols());
        for j in 0..self.r.ncols() {
            for i in 0..self.r.nrows() {
                out[(group[i] as usize, j)] += v[i] * self.r[(i, j)];
            }
        }
        out
    }

    fn column(&self, j: usize) -> Vec<f64> {
        (0..self.r.nrows()).map(|i| self.r[(i, j)]).collect()
    }

    fn for_each_in_row(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        for j in 0..self.r.ncols() {
            let x = self.r[(i, j)];
            if x != 0.0 {
                f(j, x);
            }
        }
    }
}

/// A sparse design matrix in compressed sparse row (CSR) form, a faer view.
#[derive(Clone, Copy)]
pub struct Sparse<'a> {
    r: SparseRowMatRef<'a, usize, f64>,
}

impl<'a> Sparse<'a> {
    /// Views CSR arrays: row `i` holds `values[row_ptr[i]..row_ptr[i + 1]]` in the
    /// columns `col_idx[row_ptr[i]..row_ptr[i + 1]]`.
    ///
    /// # Panics
    /// If the arrays are not a valid CSR matrix (faer checks the invariants), or if
    /// `values` and `col_idx` differ in length.
    pub fn from_csr(nrows: usize, ncols: usize, row_ptr: &'a [usize], col_idx: &'a [usize], values: &'a [f64]) -> Self {
        let symbolic = SymbolicSparseRowMatRef::new_unsorted_checked(nrows, ncols, row_ptr, None, col_idx);
        Self {
            r: SparseRowMatRef::new(symbolic, values),
        }
    }

    /// Iterates over the stored entries of row `i` as `(column, value)` pairs.
    fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let ptr = self.r.symbolic().row_ptr();
        let (start, end) = (ptr[i], ptr[i + 1]);
        let cols = &self.r.symbolic().col_idx()[start..end];
        let vals = &self.r.val()[start..end];
        cols.iter().copied().zip(vals.iter().copied())
    }
}

impl RowMatrix for Sparse<'_> {
    fn nrows(&self) -> usize {
        self.r.nrows()
    }

    fn ncols(&self) -> usize {
        self.r.ncols()
    }

    fn weighted_gram(&self, v: &[f64]) -> Mat<f64> {
        let mut out = Mat::zeros(self.r.ncols(), self.r.ncols());
        for (i, &v_i) in v.iter().enumerate().take(self.r.nrows()) {
            if v_i == 0.0 {
                continue;
            }
            for (j, r_ij) in self.row(i) {
                let vr = v_i * r_ij;
                for (m, r_im) in self.row(i) {
                    out[(j, m)] += vr * r_im;
                }
            }
        }
        out
    }

    fn weighted_cross(&self, v: &[f64], y: MatRef<'_, f64>) -> Mat<f64> {
        let mut out = Mat::zeros(self.r.ncols(), y.ncols());
        for i in 0..self.r.nrows() {
            for (j, r_ij) in self.row(i) {
                let vr = v[i] * r_ij;
                for l in 0..y.ncols() {
                    out[(j, l)] += vr * y[(i, l)];
                }
            }
        }
        out
    }

    fn mul(&self, b: MatRef<'_, f64>) -> Mat<f64> {
        let mut out = Mat::zeros(self.r.nrows(), b.ncols());
        for i in 0..self.r.nrows() {
            for (j, r_ij) in self.row(i) {
                for l in 0..b.ncols() {
                    out[(i, l)] += r_ij * b[(j, l)];
                }
            }
        }
        out
    }

    fn group_sum(&self, v: &[f64], group: &[u32], n_groups: usize) -> Mat<f64> {
        let mut out = Mat::zeros(n_groups, self.r.ncols());
        for i in 0..self.r.nrows() {
            let g = group[i] as usize;
            for (j, r_ij) in self.row(i) {
                out[(g, j)] += v[i] * r_ij;
            }
        }
        out
    }

    fn column(&self, j: usize) -> Vec<f64> {
        (0..self.r.nrows())
            .map(|i| self.row(i).filter(|&(c, _)| c == j).map(|(_, x)| x).sum())
            .collect()
    }

    fn for_each_in_row(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        let mut entries: Vec<(usize, f64)> = self.row(i).filter(|&(_, x)| x != 0.0).collect();
        entries.sort_unstable_by_key(|&(j, _)| j);
        for (j, x) in entries {
            f(j, x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4 × 3 matrix with an intercept, a numeric column and an indicator.
    const VALUES: [f64; 12] = [1.0, 0.5, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0];

    fn sparse_parts() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let (mut ptr, mut idx, mut val) = (vec![0], vec![], vec![]);
        for i in 0..4 {
            for j in 0..3 {
                let x = VALUES[i * 3 + j];
                if x != 0.0 {
                    idx.push(j);
                    val.push(x);
                }
            }
            ptr.push(idx.len());
        }
        (ptr, idx, val)
    }

    fn assert_close(a: &Mat<f64>, b: &Mat<f64>) {
        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() < 1e-12,
                    "({i}, {j}): {} vs {}",
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    #[test]
    fn dense_and_sparse_agree_on_every_product() {
        let dense = Dense::from_row_major(&VALUES, 4, 3);
        let (ptr, idx, val) = sparse_parts();
        let sparse = Sparse::from_csr(4, 3, &ptr, &idx, &val);
        let v = [1.0, 2.0, 0.5, 3.0];
        let y = Mat::from_fn(4, 2, |i, l| (i + 2 * l) as f64);
        let b = Mat::from_fn(3, 2, |j, l| (j as f64) - (l as f64));
        let group = [0, 0, 1, 2];
        assert_close(&dense.weighted_gram(&v), &sparse.weighted_gram(&v));
        assert_close(
            &dense.weighted_cross(&v, y.as_ref()),
            &sparse.weighted_cross(&v, y.as_ref()),
        );
        assert_close(&dense.mul(b.as_ref()), &sparse.mul(b.as_ref()));
        assert_close(&dense.group_sum(&v, &group, 3), &sparse.group_sum(&v, &group, 3));
        assert_eq!(dense.column(1), sparse.column(1));
    }

    #[test]
    fn weighted_gram_matches_definition() {
        let dense = Dense::from_row_major(&VALUES, 4, 3);
        let v = [1.0, 2.0, 0.5, 3.0];
        let gram = dense.weighted_gram(&v);
        // Entry (0, 0) is the total weight, entry (0, 1) the weighted sum of the numeric column.
        assert!((gram[(0, 0)] - 6.5).abs() < 1e-12);
        assert!((gram[(0, 1)] - (0.5 - 2.0 + 1.0)).abs() < 1e-12);
    }
}
