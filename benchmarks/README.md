# Benchmarks

Performance and memory are features of this library. Benchmark before and after any optimization, and put the numbers in the CHANGELOG's Performance section.

| Suite | Measures | Run |
|---|---|---|
| `rust/` | Wall time of `ume-core` kernels, without Python overhead (criterion). | `make bench-rust` |
| `python/` | Wall time through the public Python API (pytest-benchmark). | `make bench-python` |
| `python/` | Peak memory, including allocations made in Rust (memray). | `make bench-memory` |

Always benchmark a **release** build. `make bench-python` and `make bench-memory` rebuild the extension with `--release` first.

## The Benchmarks page

`make bench-results` runs all three suites plus `scale.py` (every step at 1M accounts, 1 and 4 threads, with peak memory per step) and collects them with `report.py` into `results/benchmarks.json`. The docs' Benchmarks page renders that file; `make bench-page` refreshes it and re-renders the page. Commit `results/benchmarks.json` to publish the numbers with the docs.

## Tracking changes

Before an optimization, save a baseline; after it, compare:

```sh
make bench-save                 # criterion baseline "main" in target/criterion, pytest-benchmark run in .benchmarks/
make bench-compare              # criterion reports the change per benchmark; pytest fails on a >10% slowdown
make bench-save BASELINE=before-sparse-gram
```

Baselines are machine-specific and stay local (both folders are git-ignored). Put the numbers that matter in the CHANGELOG's Performance section.

## Conventions

- Rust benches live in `rust/` and are registered as `[[bench]]` targets of `crates/ume-core`. criterion writes reports to `target/criterion/`.
- Python benchmark files are named `bench_*.py`, so the unit test run never collects them.
- Use problem sizes that match the handoff's performance reference (1M accounts, 2.19M profile rows, $k=5$, $o=5$, $B=1000$) for headline numbers, and smaller sizes for quick checks.
- Memory budgets go in `@pytest.mark.limit_memory("…")`, so a regression fails the run.
- Do not measure memory with `tracemalloc`: it cannot see Rust allocations (see decision 0006).
