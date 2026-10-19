//! Criterion benchmarks for `ume-core`. Run with `make bench-rust`.
//!
//! The same fit is timed with a dense and a sparse design matrix, on synthetic
//! rows with a many-level categorical covariate, where sparse should win.

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Mat;
use std::hint::black_box;
use ume_core::compress::Compressor;
use ume_core::fit::{RowData, fit};
use ume_core::layout::Layout;
use ume_core::rows::{Dense, Sparse};

/// Synthetic rows: an intercept, one numeric covariate, and `levels - 1` indicators.
struct Rows {
    n: usize,
    d: usize,
    dense: Vec<f64>,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
    z: Vec<f64>,
    t: Vec<f64>,
    y: Mat<f64>,
    w: Vec<f64>,
    cluster: Vec<u32>,
    n_clusters: usize,
}

fn rows(n: usize, levels: usize) -> Rows {
    let d = 2 + (levels - 1);
    let (mut dense, mut row_ptr, mut col_idx, mut values) = (vec![0.0; n * d], vec![0], vec![], vec![]);
    let (mut z, mut t, mut cluster) = (vec![0.0; n], vec![0.0; n], vec![0u32; n]);
    for i in 0..n {
        let x = ((i * 7919) % 1000) as f64 / 500.0 - 1.0;
        let level = (i * 31) % levels;
        let mut push = |j: usize, v: f64| {
            dense[i * d + j] = v;
            col_idx.push(j);
            values.push(v);
        };
        push(0, 1.0);
        push(1, x);
        if level > 0 {
            push(1 + level, 1.0);
        }
        row_ptr.push(col_idx.len());
        cluster[i] = (i / 2) as u32;
        z[i] = ((i / 2) % 2) as f64;
        t[i] = if z[i] == 1.0 && i % 3 != 0 { 1.0 } else { 0.0 };
    }
    let y = Mat::from_fn(n, 2, |i, l| (i % 13) as f64 * 0.1 + l as f64 + t[i]);
    Rows {
        n,
        d,
        dense,
        row_ptr,
        col_idx,
        values,
        z,
        t,
        y,
        w: vec![1.0; n],
        cluster,
        n_clusters: n / 2,
    }
}

fn bench_fit(c: &mut Criterion) {
    let r = rows(20_000, 50);
    let dense = Dense::from_row_major(&r.dense, r.n, r.d);
    let sparse = Sparse::from_csr(r.n, r.d, &r.row_ptr, &r.col_idx, &r.values);
    let mut group = c.benchmark_group("fit/20k rows, 50-level categorical");
    group.sample_size(20);
    group.bench_function("dense", |b| {
        b.iter(|| {
            let data = RowData {
                r: &dense,
                z: &r.z,
                t: &r.t,
                y: r.y.as_ref(),
                w: &r.w,
                cluster: &r.cluster,
                n_clusters: r.n_clusters,
            };
            black_box(fit(&data).unwrap())
        })
    });
    group.bench_function("sparse", |b| {
        b.iter(|| {
            let data = RowData {
                r: &sparse,
                z: &r.z,
                t: &r.t,
                y: r.y.as_ref(),
                w: &r.w,
                cluster: &r.cluster,
                n_clusters: r.n_clusters,
            };
            black_box(fit(&data).unwrap())
        })
    });
    group.finish();
}

/// One compression pass over the same rows, dense and sparse.
fn bench_compress(c: &mut Criterion) {
    let r = rows(20_000, 50);
    let dense = Dense::from_row_major(&r.dense, r.n, r.d);
    let sparse = Sparse::from_csr(r.n, r.d, &r.row_ptr, &r.col_idx, &r.values);
    let keys: Vec<u64> = r.cluster.iter().map(|&g| g as u64).collect();
    let strata: Vec<u32> = r.z.iter().map(|&z| z as u32).collect();
    let mut group = c.benchmark_group("compress/20k rows, 50-level categorical");
    group.sample_size(20);
    group.bench_function("dense", |b| {
        b.iter(|| {
            let mut c = Compressor::new(Layout::new(r.d, 2));
            c.update(&dense, &r.t, r.y.as_ref(), &keys, &strata).unwrap();
            black_box(c.finish())
        })
    });
    group.bench_function("sparse", |b| {
        b.iter(|| {
            let mut c = Compressor::new(Layout::new(r.d, 2));
            c.update(&sparse, &r.t, r.y.as_ref(), &keys, &strata).unwrap();
            black_box(c.finish())
        })
    });
    group.finish();
}

criterion_group!(benches, bench_fit, bench_compress);
criterion_main!(benches);
