# ume
Universal Model for Experimentation

The public API is Python (`python/ume`). The numeric kernels are Rust (`crates/ume-core`), exposed through PyO3 bindings (`crates/ume-python`) and built with [maturin](https://www.maturin.rs). The documentation site lives in `docs/` and is built with [Quarto](https://quarto.org) and [quartodoc](https://machow.github.io/quartodoc/).

## Setup

You need [rustup](https://rustup.rs), [uv](https://docs.astral.sh/uv/) and [Quarto](https://quarto.org/docs/get-started/) (only for the docs). Releasing from your machine also needs the [GitHub CLI](https://cli.github.com) (`gh`).

```sh
make dev       # create .venv with every dependency group and build the extension (uv sync)
```

`uv sync` and `uv run` rebuild the Rust extension automatically whenever a `.rs` or `Cargo.toml` file changes. `make build` forces an optimized (release) rebuild.

## Test

```sh
make lint          # ruff (lint + format check), mypy, rustfmt, clippy
make fmt           # auto-fix formatting and lint
make test          # Rust unit tests and doc tests, then pytest against a release build
make test-rust     # only the Rust tests:   cargo test -p ume-core
make test-python   # only the Python tests: uv run pytest
uv run pytest tests/test_api.py -k sparse   # one file or test
uv run pytest -m "not slow"                 # skip Monte Carlo tests
```

## Benchmark

Benchmarks run against a release build. See `benchmarks/README.md`.

```sh
make bench-rust     # criterion; HTML reports in target/criterion/
make bench-python   # pytest-benchmark wall times
make bench-memory   # memray peak memory, including allocations in Rust
```

## Documentation

The site has four parts: **Reference** (generated from docstrings), **Articles** (executed notebooks), **Rust** (rustdoc) and **Design** (design docs and decision records).

| Task | Command | Quarto equivalent |
|---|---|---|
| Build the whole site with the latest docstrings | `make docs` | `quarto render` |
| Re-execute every article, then rebuild | `make docs-articles` | `quarto render articles` |
| Re-execute one article | `cd docs && uv run quarto render articles/getting-started.qmd` | `quarto render <file>` |
| Serve the site in a local browser, with live reload | `make docs-preview` | `quarto preview` |

What each step does:

- **Docstrings → reference pages.** `quartodoc build` (run by `make docs`) reads the docstrings in `python/ume` and the Rust `///` comments compiled into `ume._core`, and writes `docs/reference/*.qmd`. Because it imports the compiled module, `make docs` first rebuilds the extension. After you edit a docstring, run `make docs` again. Or, while `make docs-preview` is running, run `cd docs && uv run quartodoc build --watch` in a second terminal, and the browser refreshes when you save.
- **Articles.** Articles in `docs/articles/` are executed and their results are frozen in `docs/_freeze/` (committed). A full `make docs` re-executes only articles whose source changed. To re-run them after a *code* change, use `make docs-articles`, which renders the `articles/` folder (Quarto always executes when you render a single file or folder), then rebuilds the site. Commit the updated `docs/_freeze/`.
- **Preview.** `make docs-preview` builds everything, then serves `docs/_site` at a local URL and opens your browser. Editing a `.qmd` file re-renders that page on save. Stop it with Ctrl-C.
- **Rust docs.** `make docs` also runs `cargo doc` and copies the result to `docs/_site/rust/`. To browse rustdoc alone: `cargo doc -p ume-core --open`.

## Release

A release is a deliberate **two-stage** process. Stage 1 verifies and builds, and publishes nothing. Stage 2 publishes exactly what stage 1 built, after you have looked at it. The steps are the same locally and on GitHub.

### Before a release

1. Choose the version, following [SemVer](https://semver.org) (before 1.0, a breaking change bumps the minor version). It lives in **one** place, `version` under `[workspace.package]` in the root `Cargo.toml`; `ume.__version__` and the wheel both read it.
2. `make release-bump VERSION=X.Y.Z` sets it, renames `## Unreleased` in `CHANGELOG.md` to `## X.Y.Z (YYYY-MM-DD)` (opening an empty `## Unreleased`), and updates the lock files. Check the changelog's entries.
3. Commit the changes.
4. Optionally, `make release-tag` tags the commit `vX.Y.Z` (annotated, after the version, changelog and tag checks). It does not push: `git push origin vX.Y.Z`. Publishing later reuses this tag if it points at the commit being released.

```sh
make release-version                            # print the current version
make release-bump VERSION=0.3.0                 # set it and date the changelog
make release-tag                                # tag HEAD v0.3.0 (local)
```

### On GitHub (recommended: builds wheels for every platform)

One-time setup:
- On [PyPI](https://pypi.org/manage/account/publishing/) and [TestPyPI](https://test.pypi.org/manage/account/publishing/), add a *trusted publisher*: this repository, workflow `release.yml`, environment `pypi` or `testpypi`. No API tokens are needed.
- Under Settings → Environments, create `testpypi` and `pypi` and add yourself as a **required reviewer**. This is what pauses the run between the two stages.
- Under Settings → Pages, set Source to **GitHub Actions**.

Then:
1. **Stage 1.** Actions → Release → Run workflow, and choose `testpypi` or `pypi`. The workflow checks the version, changelog and tag, then runs the linters, the tests and the docs build. It builds wheels (Linux x86_64 and aarch64, macOS Intel and Apple Silicon, Windows; Python 3.10–3.13) and the sdist, installs the built wheels on three operating systems and runs the tests against them, and checks the package metadata.
2. **Inspect.** The run pauses at the `publish` job. The run summary lists every file with its SHA-256, and the wheels, the sdist and the docs build are downloadable as artifacts.
3. **Stage 2.** Approve the `publish` job. It uploads to the chosen index. For `pypi` it also tags `vX.Y.Z` and deploys the docs to GitHub Pages.

Release to `testpypi` first, install from it, and then rerun with `pypi`.

### From your machine

```sh
make release-check                              # stage 1 (scripts/release_check.sh)
```

Stage 1 refuses to run on uncommitted changes (`ALLOW_DIRTY=1 make release-check` does a dry run that stage 2 will refuse). It checks the version, changelog and tag, then lints, tests and builds the docs. It then builds the sdist and a wheel for this machine into `dist/`, runs the tests against the installed wheel in a clean environment, and writes `dist/release-manifest.txt` with the git commit and checksums. Then inspect:

```sh
ls -l dist/ && cat dist/release-manifest.txt
unzip -l dist/*.whl
tar -tzf dist/*.tar.gz
make docs-preview
```

```sh
export UV_PUBLISH_TOKEN=pypi-...                # API token for the chosen index
make release-publish                            # stage 2 to TestPyPI (default)
make release-publish REPOSITORY=pypi            # stage 2 to PyPI, then tag + docs deploy
```

Stage 2 refuses to run if `HEAD`, the working tree or any file in `dist/` has changed since stage 1, and it asks you to type the version before uploading. A local release uploads only this machine's wheel and the sdist, so prefer the GitHub path for real releases.

### Refreshing the published docs without a release

Actions → Docs → Run workflow (or `gh workflow run docs.yml`) rebuilds the site from the chosen ref and deploys it.
