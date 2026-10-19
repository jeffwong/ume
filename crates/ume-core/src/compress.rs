//! One pass over the rows to the compressed dataset.
//!
//! Rows arrive grouped by cluster. A buffer accumulates the current cluster's
//! summary $u_a$ ([`Layout`]); when the cluster changes, the summary is
//! **flushed** into its stratum: the count $G_h$, the first moments
//! $\sum_{a \in h} u_a$ and the second moments $\sum_{a \in h} u_a u_a^\top$. Then the
//! buffer is cleared. Cluster summaries are never stored.
//!
//! The buffer tracks which entries were touched, so the flush and the clear
//! skip entries that are zero for this cluster, such as all of $S^T_a$ for a
//! cluster that is not treated, or the indicators of levels it does not have.
//!
//! The compressed rows of a stratum add, so compressing shards separately and
//! adding them ([`Compressed::merge`]) equals compressing everything at once.
//!
//! The pass ([`Pass`]) is generic over its [`Fold`], the step that consumes a
//! cluster summary: [`MomentsFold`] builds the compressed dataset, and the exact
//! bootstrap's fold (in `bootstrap`) adds $\xi_a^{(b)} v_a$ for every draw.

use std::collections::HashSet;

use faer::{Mat, MatRef};

use crate::Error;
use crate::layout::Layout;
use crate::rows::RowMatrix;

/// The compressed row of one stratum: sums over its clusters.
#[derive(Clone, Debug)]
pub struct StratumMoments {
    /// Number of clusters $G_h$.
    pub g: f64,
    /// $\sum_{a \in h} u_a$, length $q$.
    pub first: Vec<f64>,
    /// $\sum_{a \in h} u_a u_a^\top$, $q \times q$, symmetric.
    pub second: Mat<f64>,
}

impl StratumMoments {
    /// An empty stratum for summaries of length `q`.
    pub fn zeros(q: usize) -> Self {
        Self {
            g: 0.0,
            first: vec![0.0; q],
            second: Mat::zeros(q, q),
        }
    }

    /// Adds `other` into `self`.
    pub fn add_assign(&mut self, other: &StratumMoments) {
        self.g += other.g;
        for (a, b) in self.first.iter_mut().zip(&other.first) {
            *a += b;
        }
        self.second += &other.second;
    }
}

/// The result of a compression pass: one [`StratumMoments`] per stratum index.
#[derive(Clone, Debug)]
pub struct Compressed {
    /// The layout of the summaries.
    pub layout: Layout,
    /// Compressed rows, indexed by the stratum index given to [`Compressor::update`].
    pub strata: Vec<StratumMoments>,
    /// Number of rows read.
    pub n_rows: u64,
}

impl Compressed {
    /// Adds the compressed rows of `other`, stratum by stratum.
    ///
    /// # Errors
    /// [`Error::LayoutMismatch`] if the layouts differ.
    pub fn merge(&mut self, other: &Compressed) -> Result<(), Error> {
        if self.layout != other.layout {
            return Err(Error::LayoutMismatch);
        }
        let q = self.layout.q();
        while self.strata.len() < other.strata.len() {
            self.strata.push(StratumMoments::zeros(q));
        }
        for (mine, theirs) in self.strata.iter_mut().zip(&other.strata) {
            mine.add_assign(theirs);
        }
        self.n_rows += other.n_rows;
        Ok(())
    }
}

/// What happens to a cluster summary when its rows end: the only part in which
/// the one-pass algorithms differ (handoff, section 5.3).
pub trait Fold {
    /// Folds the summary `u` of cluster `key` (in stratum `stratum`) into the state.
    /// Only the entries listed in `nz` can be nonzero.
    fn fold(&mut self, key: u64, stratum: u32, u: &[f64], nz: &[usize]);
}

/// A fold whose states over disjoint sets of clusters add: the monoid property that
/// lets passes over shards (or threads) be merged exactly.
pub trait Merge {
    /// Adds the state of `other`, built over other clusters, into `self`.
    fn merge(&mut self, other: Self);
}

impl Merge for MomentsFold {
    fn merge(&mut self, other: Self) {
        for (h, theirs) in other.strata.into_iter().enumerate() {
            if self.strata.len() <= h {
                self.strata.push(StratumMoments::zeros(self.q));
            }
            self.strata[h].add_assign(&theirs);
        }
    }
}

/// One pass over rows grouped by cluster: a buffer for the open cluster's summary
/// $u_a$, flushed into a [`Fold`] when the cluster changes. Feed row batches with
/// [`update`](Self::update).
pub struct Pass<F: Fold> {
    layout: Layout,
    fold: F,
    u: Vec<f64>,
    touched: Vec<bool>,
    nz: Vec<usize>,
    open: Option<(u64, u32)>,
    flushed: HashSet<u64>,
    n_rows: u64,
    /// Scratch space for the nonzero entries of one row, reused across rows.
    row: Vec<(usize, f64)>,
}

impl<F: Fold> Pass<F> {
    /// A pass with summaries of the given layout, folding into `fold`.
    pub fn with_fold(layout: Layout, fold: F) -> Self {
        let q = layout.q();
        Self {
            layout,
            fold,
            u: vec![0.0; q],
            touched: vec![false; q],
            nz: Vec::new(),
            open: None,
            flushed: HashSet::new(),
            n_rows: 0,
            row: Vec::new(),
        }
    }

    /// The layout of the summaries.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Flushes the open cluster, if any. After this, [`flushed_keys`](Self::flushed_keys)
    /// holds every cluster seen.
    pub fn close_open_cluster(&mut self) {
        self.flush();
    }

    /// The cluster keys flushed so far (for checking that shards do not share clusters).
    pub fn flushed_keys(&self) -> &HashSet<u64> {
        &self.flushed
    }

    /// Adds a batch of rows. Rows of a cluster must be contiguous, possibly across
    /// batches; each row carries its cluster's key and stratum index.
    ///
    /// # Errors
    /// [`Error::NotContiguous`] if a cluster reappears after its rows ended, and
    /// [`Error::StratumChanges`] if a cluster's rows have different strata.
    pub fn update<R: RowMatrix>(
        &mut self,
        r: &R,
        t: &[f64],
        y: MatRef<'_, f64>,
        keys: &[u64],
        strata: &[u32],
    ) -> Result<(), Error> {
        self.update_rows(r, t, y, keys, strata, 0..r.nrows())
    }

    /// Adds the given rows of a batch, in the order given. [`ParallelPass`](crate::parallel::ParallelPass)
    /// uses this to give each worker its clusters without copying the batch.
    ///
    /// # Errors
    /// As [`update`](Self::update).
    pub fn update_rows<R: RowMatrix>(
        &mut self,
        r: &R,
        t: &[f64],
        y: MatRef<'_, f64>,
        keys: &[u64],
        strata: &[u32],
        rows: impl IntoIterator<Item = usize>,
    ) -> Result<(), Error> {
        for i in rows {
            match self.open {
                Some((key, stratum)) if key == keys[i] => {
                    if stratum != strata[i] {
                        return Err(Error::StratumChanges(key));
                    }
                }
                _ => {
                    self.flush();
                    if self.flushed.contains(&keys[i]) {
                        return Err(Error::NotContiguous(keys[i]));
                    }
                    self.open = Some((keys[i], strata[i]));
                }
            }
            self.add_row(r, t[i], y, i);
            self.n_rows += 1;
        }
        Ok(())
    }

    /// Adds row `i` into the open cluster's summary.
    fn add_row<R: RowMatrix>(&mut self, r: &R, t: f64, y: MatRef<'_, f64>, i: usize) {
        let mut entries = std::mem::take(&mut self.row);
        entries.clear();
        r.for_each_in_row(i, |j, x| entries.push((j, x)));
        let layout = self.layout;
        for &(j, x) in &entries {
            for l in 0..layout.o {
                self.add(layout.sy(j, l), x * y[(i, l)]);
            }
        }
        for (a, &(j, xj)) in entries.iter().enumerate() {
            for &(m, xm) in &entries[a..] {
                let rr = xj * xm;
                self.add(layout.s(j, m), rr);
                if t != 0.0 {
                    self.add(layout.st(j, m), t * rr);
                }
            }
        }
        self.row = entries;
    }

    fn add(&mut self, k: usize, value: f64) {
        if !self.touched[k] {
            self.touched[k] = true;
            self.nz.push(k);
        }
        self.u[k] += value;
    }

    /// Folds the open cluster's summary and clears the buffer.
    fn flush(&mut self) {
        let Some((key, stratum)) = self.open.take() else { return };
        self.fold.fold(key, stratum, &self.u, &self.nz);
        for &k in &self.nz {
            self.u[k] = 0.0;
            self.touched[k] = false;
        }
        self.nz.clear();
        self.flushed.insert(key);
    }

    /// Flushes the last cluster and returns the fold and the number of rows read.
    pub fn into_fold(mut self) -> (F, u64) {
        self.flush();
        (self.fold, self.n_rows)
    }

    /// Flushes the last cluster and returns the fold, the rows read, and the cluster keys seen.
    pub fn into_parts(mut self) -> (F, u64, HashSet<u64>) {
        self.flush();
        (self.fold, self.n_rows, self.flushed)
    }

    /// A pass that continues from parts: used to merge parallel workers.
    pub fn from_parts(layout: Layout, fold: F, n_rows: u64, flushed: HashSet<u64>) -> Self {
        let mut pass = Self::with_fold(layout, fold);
        pass.n_rows = n_rows;
        pass.flushed = flushed;
        pass
    }
}

/// The fold of the compressed dataset: per stratum, $G_h$, $\sum u_a$, $\sum u_a u_a^\top$.
pub struct MomentsFold {
    q: usize,
    strata: Vec<StratumMoments>,
}

impl Fold for MomentsFold {
    fn fold(&mut self, _key: u64, stratum: u32, u: &[f64], nz: &[usize]) {
        let h = stratum as usize;
        while self.strata.len() <= h {
            self.strata.push(StratumMoments::zeros(self.q));
        }
        let state = &mut self.strata[h];
        state.g += 1.0;
        for (ia, &a) in nz.iter().enumerate() {
            let ua = u[a];
            state.first[a] += ua;
            for &b in &nz[ia..] {
                let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
                state.second[(hi, lo)] += ua * u[b];
            }
        }
    }
}

impl MomentsFold {
    /// A fresh fold for another worker.
    pub fn fresh(&self) -> Self {
        Self {
            q: self.q,
            strata: Vec::new(),
        }
    }
}

/// The compression pass: rows to the compressed dataset.
pub type Compressor = Pass<MomentsFold>;

impl Pass<MomentsFold> {
    /// A compressor for summaries with the given layout.
    pub fn new(layout: Layout) -> Self {
        Self::with_fold(
            layout,
            MomentsFold {
                q: layout.q(),
                strata: Vec::new(),
            },
        )
    }

    /// Flushes the last cluster and returns the compressed rows.
    pub fn finish(self) -> Compressed {
        let layout = self.layout;
        let (mut fold, n_rows) = self.into_fold();
        for state in &mut fold.strata {
            let q = state.second.nrows();
            for hi in 0..q {
                for lo in 0..hi {
                    state.second[(lo, hi)] = state.second[(hi, lo)];
                }
            }
        }
        Compressed {
            layout,
            strata: fold.strata,
            n_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    //! The one-pass compressor: its first moments are the blocks of each stratum's rows, batches and shards add up to one pass, and non-contiguous clusters are rejected.

    use super::*;
    use crate::blocks::Blocks;
    use crate::rows::Dense;

    /// Design matrix values, treatment, outcomes, cluster keys, strata.
    type TestRows = (Vec<f64>, Vec<f64>, Mat<f64>, Vec<u64>, Vec<u32>);

    /// Five rows in three clusters, with one covariate and one outcome.
    fn rows() -> TestRows {
        let r = vec![1.0, 0.5, 1.0, -1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.5];
        let t = vec![1.0, 0.0, 0.0, 1.0, 0.0];
        let y = Mat::from_fn(5, 1, |i, _| i as f64 + 1.0);
        (r, t, y, vec![7, 7, 8, 9, 9], vec![1, 1, 0, 1, 1])
    }

    /// A stratum's first moments equal the blocks of its rows computed directly.
    #[test]
    fn first_moments_are_the_blocks_of_the_stratum() {
        let (r, t, y, keys, strata) = rows();
        let dense = Dense::from_row_major(&r, 5, 2);
        let mut c = Compressor::new(Layout::new(2, 1));
        c.update(&dense, &t, y.as_ref(), &keys, &strata).unwrap();
        let out = c.finish();
        assert_eq!(out.strata[1].g, 2.0);
        assert_eq!(out.n_rows, 5);
        // Stratum 1 holds rows 0, 1, 3, 4.
        let w = [1.0, 1.0, 0.0, 1.0, 1.0];
        let expected = Blocks::from_rows(&dense, &w, &t, y.as_ref());
        let got = out.layout.blocks(&out.strata[1].first, 1.0);
        for (a, b) in [
            (&got.s, &expected.s),
            (&got.s_t, &expected.s_t),
            (&got.s_y, &expected.s_y),
        ] {
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    assert!((a[(i, j)] - b[(i, j)]).abs() < 1e-12);
                }
            }
        }
    }

    /// Batches that split a cluster, and shards merged afterwards, give the same compressed rows as one pass.
    #[test]
    fn batches_and_shards_add_up() {
        let (r, t, y, keys, strata) = rows();
        let dense = Dense::from_row_major(&r, 5, 2);
        let mut whole = Compressor::new(Layout::new(2, 1));
        whole.update(&dense, &t, y.as_ref(), &keys, &strata).unwrap();
        let whole = whole.finish();
        // Batches split cluster 7 across two calls; shards split between clusters 8 and 9.
        let mut a = Compressor::new(Layout::new(2, 1));
        let first = Dense::from_row_major(&r[..2], 1, 2);
        let second = Dense::from_row_major(&r[2..6], 2, 2);
        a.update(&first, &t[..1], y.as_ref().subrows(0, 1), &keys[..1], &strata[..1])
            .unwrap();
        a.update(&second, &t[1..3], y.as_ref().subrows(1, 2), &keys[1..3], &strata[1..3])
            .unwrap();
        let mut b = Compressor::new(Layout::new(2, 1));
        let rest = Dense::from_row_major(&r[6..], 2, 2);
        b.update(&rest, &t[3..], y.as_ref().subrows(3, 2), &keys[3..], &strata[3..])
            .unwrap();
        let mut merged = a.finish();
        merged.merge(&b.finish()).unwrap();
        for h in 0..2 {
            assert_eq!(merged.strata[h].g, whole.strata[h].g);
            for (x, z) in merged.strata[h].first.iter().zip(&whole.strata[h].first) {
                assert!((x - z).abs() < 1e-12);
            }
        }
    }

    /// A cluster whose rows reappear after it ended is rejected as not contiguous.
    #[test]
    fn a_cluster_must_be_contiguous() {
        let (r, t, y, mut keys, strata) = rows();
        keys[4] = 7;
        let dense = Dense::from_row_major(&r, 5, 2);
        let mut c = Compressor::new(Layout::new(2, 1));
        assert_eq!(
            c.update(&dense, &t, y.as_ref(), &keys, &strata),
            Err(Error::NotContiguous(7))
        );
    }
}
