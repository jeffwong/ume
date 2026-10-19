#!/usr/bin/env bash
# Release stage 1 of 2: verify and build. Publishes nothing.
#
# Runs every check, builds the sdist and a wheel into dist/, tests the built
# wheel in a clean environment, and writes dist/release-manifest.txt (git commit
# + checksums). Inspect dist/ and docs/_site, then run scripts/release_publish.sh.
#
# Usage: scripts/release_check.sh
# Env:   ALLOW_DIRTY=1  run on an uncommitted tree (dry run; publish will refuse it)
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

step "1/8 Working tree"
if [[ -n "$(git status --porcelain)" ]]; then
  [[ "${ALLOW_DIRTY:-0}" == 1 ]] || fail "uncommitted changes. Commit them, or set ALLOW_DIRTY=1 for a dry run."
  echo "warning: tree is dirty (ALLOW_DIRTY=1); the publish step will refuse this build."
fi
uv sync --locked

step "2/8 Version, changelog, tag"
uv run python scripts/release_version.py check
NAME=$(uv run python scripts/release_version.py name)
VERSION=$(uv run python scripts/release_version.py version)

step "3/8 Lint"
make lint

step "4/8 Unit tests (Rust, then Python against a release build)"
make test

step "5/8 Documentation"
make docs

step "6/8 Build sdist and wheel"
rm -rf dist
uv run maturin sdist --out dist
uv run maturin build --release --out dist

step "7/8 Package metadata and wheel smoke test"
uvx twine check --strict dist/*
# Test the built wheel, not the editable install: a clean env with only the wheel
# and the test dependencies. tests/ imports `ume` from site-packages.
uv run --isolated --no-project --python "$(cat .python-version)" \
  --with "$(ls dist/*.whl | head -1)" --with pytest --with hypothesis \
  pytest tests -q -p no:cacheprovider

step "8/8 Manifest"
{
  echo "name=$NAME"
  echo "version=$VERSION"
  echo "commit=$(git rev-parse HEAD)"
  echo "dirty=$([[ -n "$(git status --porcelain)" ]] && echo 1 || echo 0)"
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "--- sha256"
  (cd dist && shasum -a 256 ./*.whl ./*.tar.gz)
} > dist/release-manifest.txt
cat dist/release-manifest.txt

cat <<MSG

Stage 1 passed for $NAME $VERSION. Nothing has been published.
Inspect before publishing:
  ls -l dist/                          # the files that will be uploaded
  unzip -l dist/*.whl                  # wheel contents
  tar -tzf dist/*.tar.gz               # sdist contents
  make docs-preview                    # the docs site that will be deployed
Then:
  scripts/release_publish.sh --repository testpypi
  scripts/release_publish.sh --repository pypi
MSG
