//! Passes on several threads, by sharding clusters in-process.
//!
//! The states of every fold add over disjoint sets of clusters ([`Merge`]), the
//! same property that lets shards be compressed separately. A [`ParallelPass`]
//! uses it inside one process: each row goes to the worker of its cluster
//! (a hash of the cluster key), so every cluster's rows stay together and in
//! order within one worker, across batches. The workers' states are merged at
//! the end.
//!
//! Results equal the sequential pass up to the order of floating-point
//! additions (about 1e-15 relative). The coupled multinomial bootstrap cannot be
//! split, so it always runs on one worker.

use std::collections::HashSet;

use faer::MatRef;

use crate::Error;
use crate::compress::{Fold, Merge, Pass};
use crate::layout::Layout;
use crate::rows::RowMatrix;

/// The worker of a cluster: a mixed hash of its key, modulo the number of workers.
#[inline]
fn worker_of(key: u64, n: usize) -> usize {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) % n as u64) as usize
}

/// A pass whose clusters are spread over worker threads.
pub struct ParallelPass<F: Fold + Merge + Send> {
    layout: Layout,
    workers: Vec<Pass<F>>,
}

impl<F: Fold + Merge + Send> ParallelPass<F> {
    /// A pass on `threads` workers (at least 1), each folding into its own `make()`.
    pub fn new(layout: Layout, threads: usize, make: impl Fn() -> F) -> Self {
        let workers = (0..threads.max(1)).map(|_| Pass::with_fold(layout, make())).collect();
        Self { layout, workers }
    }

    /// Number of workers.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Adds a batch: each worker takes the rows of its clusters, in order, on its own thread.
    ///
    /// # Errors
    /// As [`Pass::update`], from any worker.
    pub fn update<R: RowMatrix + Sync>(
        &mut self,
        r: &R,
        t: &[f64],
        y: MatRef<'_, f64>,
        keys: &[u64],
        strata: &[u32],
    ) -> Result<(), Error> {
        let n = self.workers.len();
        if n == 1 {
            return self.workers[0].update(r, t, y, keys, strata);
        }
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &key) in keys.iter().enumerate() {
            rows[worker_of(key, n)].push(i);
        }
        let results: Vec<Result<(), Error>> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .workers
                .iter_mut()
                .zip(rows)
                .map(|(worker, mine)| scope.spawn(move || worker.update_rows(r, t, y, keys, strata, mine)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a pass worker panicked"))
                .collect()
        });
        results.into_iter().collect()
    }

    /// Flushes every worker and merges them into one sequential [`Pass`].
    pub fn merged(self) -> Pass<F> {
        let mut parts = self.workers.into_iter().map(Pass::into_parts);
        let (mut fold, mut n_rows, mut keys) = parts.next().expect("at least one worker");
        for (f, n, k) in parts {
            fold.merge(f);
            n_rows += n;
            keys.extend(k);
        }
        Pass::from_parts(self.layout, fold, n_rows, keys)
    }

    /// The keys seen so far by every worker (open clusters excluded).
    pub fn flushed_keys(&self) -> HashSet<u64> {
        self.workers
            .iter()
            .flat_map(|w| w.flushed_keys().iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    //! A pass on several threads equals the sequential pass, including batches whose boundaries fall inside a cluster.

    use super::*;
    use crate::compress::Compressor;
    use crate::rows::Dense;
    use faer::Mat;

    /// Four workers, over two batches with a cluster split between them, give the sequential pass's compressed rows.
    #[test]
    fn a_parallel_pass_equals_the_sequential_one() {
        let n = 400;
        let r: Vec<f64> = (0..n).flat_map(|i| [1.0, (i % 17) as f64 / 7.0 - 1.0]).collect();
        let t: Vec<f64> = (0..n).map(|i| ((i / 3) % 2 * (i % 2)) as f64).collect();
        let y = Mat::from_fn(n, 2, |i, l| (i * (l + 1) % 11) as f64);
        let keys: Vec<u64> = (0..n).map(|i| (i / 3) as u64).collect();
        let strata: Vec<u32> = keys.iter().map(|k| (k % 2) as u32).collect();
        let dense = Dense::from_row_major(&r, n, 2);
        let layout = Layout::new(2, 2);
        let mut seq = Compressor::new(layout);
        seq.update(&dense, &t, y.as_ref(), &keys, &strata).unwrap();
        let seq = seq.finish();
        let mut par = ParallelPass::new(layout, 4, || Compressor::new(layout).into_fold().0);
        // Two batches, the boundary inside a cluster.
        par.update(
            &Dense::from_row_major(&r[..2 * 200], 200, 2),
            &t[..200],
            y.as_ref().subrows(0, 200),
            &keys[..200],
            &strata[..200],
        )
        .unwrap();
        par.update(
            &Dense::from_row_major(&r[2 * 200..], 200, 2),
            &t[200..],
            y.as_ref().subrows(200, 200),
            &keys[200..],
            &strata[200..],
        )
        .unwrap();
        let par = par.merged().finish();
        assert_eq!(par.n_rows, seq.n_rows);
        for (a, b) in par.strata.iter().zip(&seq.strata) {
            assert_eq!(a.g, b.g);
            for (x, z) in a.first.iter().zip(&b.first) {
                assert!((x - z).abs() < 1e-9);
            }
        }
    }
}
