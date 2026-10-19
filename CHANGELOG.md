# Changelog

All notable changes to `ume` are recorded here. The project follows [Semantic Versioning](https://semver.org). Before 1.0.0, breaking changes bump the minor version.

Each release lists its changes under these headings, in this order, leaving out empty ones:

1. **Breaking changes**: anything that can break existing user code or change existing results.
2. **New features**
3. **Bug fixes**
4. **Performance**: significant optimizations, with measured before and after numbers and the benchmark that shows them.

## Unreleased

### New features

- Project skeleton: Python package `ume` over a Rust core (`ume-core` kernels, `ume-python` PyO3 bindings), built with maturin and uv.
- Documentation site (Quarto + quartodoc) with an API reference generated from Python and Rust docstrings, articles, rustdoc, design docs and decision records.
- Benchmark suites: criterion for Rust kernels, pytest-benchmark and memray for the Python API.
- The universal model on rows, in Rust: `ume.annotate` → `.metrics(ume.mean(...), ume.ratio(...))` → `.fit()` → `.effects()`, `.compliers()`, `.itt()`, with a 2SLS regression report (first-stage robust F per endogenous regressor, coefficient tables per outcome). Every design of the CSS post (encouragement × clusters × ramp × metric) matches a numpy reference to 1e-10 and linearmodels.
- One engine generic over dense and sparse design matrices (faer); `sparse="auto"` for many-level categoricals.
- Effect grammar: contrast (difference, relative) ∘ statistic (mean, ratio) ∘ population (compliers, segment), plus custom effects of the compliers' levels, with complex-step delta-method standard errors.
- `ume.testing`: the posts' simulator and truth, and a numpy reference implementation.
- Compression: `exp.compress()` and the streaming `ume.Compressor` build, in one pass, one compressed row per stratum ($G_h$, $\sum u_a$, $\sum u_a u_a^\top$). `CompressedData` is a mapping from `Stratum` to `CompressedRow` (with `.blocks()`), merges shards with `+` (refusing shards that share a cluster), and `.fit()` gives exactly the fit on rows, including the first stage and the ITT.
- Two-stage release: `make release-check` / `make release-publish` locally, and a GitHub Release workflow with a reviewer-gated publish and wheels for every platform.

- Bootstraps and posteriors: `exp.bootstrap(B, weights="bayesian" | "poisson" | "multinomial", seed)` (exact, one pass over the rows), and `cd.posterior(B, seed)` (Gaussian, from the compressed rows alone; warns below 200 clusters per stratum). Both return `ume.Draws` (`.effects()`, `.values()`, `.prob()`), and `fit.effects(se=draws)` uses draw standard deviations. Weights come from Philox4x64-10 streams keyed by (seed, cluster, draw), bit-for-bit equal to numpy's Philox.

- The linearized standard error, `fit.effects(se="linearized")`: the bootstrap's first-order expansion in the per-stratum sums, which includes the uncertainty in $\hat e$ and $\bar x_C$. Its gradient is taken through the engine's own per-draw fit.
- Delta vectors early, `exp.effects(se="sandwich" | "linearized")`: the same estimates and standard errors in two passes over the rows, keeping $H(1+q)$ numbers instead of $H(1+q)^2$.
- `benchmarks/calibration.py`: a calibration Monte Carlo of every method against the simulated truth. Over 1,000 replications, the sandwich, linearized, exact Bayesian and Gaussian intervals all cover 94.3%–95.1% with SE/SD 0.999–1.015.

- Scale: `ume.annotate` accepts polars LazyFrames (resolved with one aggregation, sorted lazily, streamed in batches); `ume.design`, `ume.compress`, `ume.bootstrap` and `ume.effects` stream any iterable of batches; `cd.save()` and `ume.load()` store compressed data (format `ume.compressed` v1); every pass and draw fit runs on `ume.get_threads()` workers (`min(4, cores)` by default).

- `ume.bootstrap_sums` and `ume.DrawSums`: the exact bootstrap's sums of separately processed shards merge with `+` and are fit once with `.draws()` (Bayesian and Poisson weights).
- One abi3 wheel per platform for every Python >= 3.10.
- `make bench-save` / `make bench-compare`: local benchmark baselines (criterion and pytest-benchmark).
- `ume.testing.simulate_accounts(heavy_tail=...)`: heavy-tailed clusters, and calibration scenarios `running`, `small-strata` and `heavy-tails`.

- Every public input is documented (name, type, purpose); `scripts/check_docstrings.py` enforces it in `make lint`. Every test file and test states its purpose and oracle.

### Performance

- At 1M clusters (2.19M rows) and $B = 1000$: the exact bootstrap takes 3.8 s on 4 threads (12.2 s on 1), compression plus the Gaussian posterior 0.40 s.
- Batch preparation computes categorical levels, indicators and cohort codes in the frame's engine and skips converting string columns to numpy: compression of 2.19M rows went from 0.94 s to 0.38 s.
- Fitting from compressed data takes about 1 ms whatever the number of rows; compressing 2.19M rows takes 1.13 s (the Rust pass about 0.66 s).

- First-stage covariance is computed per equation (`Scope::PerColumn`) instead of jointly: on 20,000 rows with $d = 51$, the fit went from 3.12 s to 449 ms (dense) and from 2.98 s to 285 ms (sparse) (`make bench-rust`).
