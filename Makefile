.PHONY: bench-save bench-compare release-check release-publish docs-articles dev build test test-rust test-python lint fmt docs docs-preview bench bench-rust bench-python bench-memory

dev:           ## Create .venv, install all dependency groups, build the extension
	uv sync

build:         ## Rebuild the Rust extension in release mode into .venv
	uv run maturin develop --release

test: test-rust test-python

test-rust:     ## Unit tests and doc tests of the Rust core
	cargo test -p ume-core

test-python: build
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings

fmt:
	uv run ruff check --fix .
	uv run ruff format .
	cargo fmt --all

docs: build    ## Build the site into docs/_site: reference pages, articles, rustdoc
	rm -rf docs/_site docs/reference
	cd docs && uv run quartodoc build
	cd docs && uv run quarto render
	cargo doc --no-deps -p ume-core
	rm -rf docs/_site/rust && cp -R target/doc docs/_site/rust

docs-articles: build  ## Re-execute every article (ignores the freeze), then rebuild the site
	cd docs && uv run quarto render articles
	$(MAKE) docs

docs-preview: docs
	cd docs && uv run quarto preview --no-render

bench: bench-rust bench-python bench-memory

bench-rust:
	cargo bench -p ume-core

bench-python: build
	uv run pytest benchmarks/python -o python_files='bench_*.py' --benchmark-only

bench-save: build    ## Save a benchmark baseline (BASELINE=name, default "main")
	cargo bench -p ume-core -- --save-baseline $(or $(BASELINE),main)
	uv run pytest benchmarks/python -o python_files='bench_*.py' --benchmark-only --benchmark-save=$(or $(BASELINE),main)

bench-compare: build ## Compare against a saved baseline; fails on a >10% mean slowdown in Python benchmarks
	cargo bench -p ume-core -- --baseline $(or $(BASELINE),main)
	uv run pytest benchmarks/python -o python_files='bench_*.py' --benchmark-only \
		--benchmark-compare --benchmark-compare-fail=mean:10%

bench-memory: build
	uv run pytest benchmarks/python -o python_files='bench_*.py' --memray --native --benchmark-disable

release-check: ## Release stage 1: verify and build into dist/. Publishes nothing.
	scripts/release_check.sh

release-publish: ## Release stage 2: publish dist/. REPOSITORY=testpypi|pypi
	scripts/release_publish.sh --repository $(or $(REPOSITORY),testpypi)
