#!/usr/bin/env bash
# Release stage 2 of 2: publish what stage 1 built and you inspected.
#
# Refuses to run unless dist/release-manifest.txt matches the current commit and
# the files in dist/. Asks you to type the version before uploading.
#
# Usage: scripts/release_publish.sh [--repository testpypi|pypi] [--no-docs]
#   --repository testpypi  (default) upload to test.pypi.org only; no tag, no docs.
#   --repository pypi      upload to pypi.org, tag v<version>, push the tag, deploy docs.
#   --no-docs              with pypi: skip the GitHub Pages deploy (needs the gh CLI).
# Env: UV_PUBLISH_TOKEN    API token for the chosen index (pypi-...).
#
# Local publishing uploads the sdist and the one wheel built on this machine.
# For wheels on every platform, use the GitHub Release workflow instead.
set -euo pipefail
cd "$(dirname "$0")/.."

REPOSITORY=testpypi
DOCS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository) REPOSITORY="$2"; shift 2 ;;
    --no-docs) DOCS=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

fail() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }
field() { sed -n "s/^$1=//p" dist/release-manifest.txt; }

case "$REPOSITORY" in
  testpypi) PUBLISH_URL=https://test.pypi.org/legacy/; CHECK_URL=https://test.pypi.org/simple/ ;;
  pypi)     PUBLISH_URL=https://upload.pypi.org/legacy/; CHECK_URL=https://pypi.org/simple/ ;;
  *) fail "--repository must be testpypi or pypi" ;;
esac

[[ -f dist/release-manifest.txt ]] || fail "no dist/release-manifest.txt. Run scripts/release_check.sh first."
NAME=$(field name); VERSION=$(field version)
[[ "$(field dirty)" == 0 ]] || fail "stage 1 ran on a dirty tree (ALLOW_DIRTY=1). Commit and rerun it."
[[ -z "$(git status --porcelain)" ]] || fail "the working tree has changed since stage 1."
[[ "$(field commit)" == "$(git rev-parse HEAD)" ]] || fail "HEAD is not the commit stage 1 built. Rerun stage 1."
(cd dist && sed '1,/^--- sha256$/d' release-manifest.txt | shasum -a 256 --check --quiet) \
  || fail "files in dist/ do not match the manifest."
[[ -n "${UV_PUBLISH_TOKEN:-}" ]] || fail "set UV_PUBLISH_TOKEN to an API token for $REPOSITORY."

echo "About to publish $NAME $VERSION to $REPOSITORY ($PUBLISH_URL):"
(cd dist && ls -l ./*.whl ./*.tar.gz)
[[ "$REPOSITORY" == pypi ]] && echo "Then: tag v$VERSION, push it to origin$([[ $DOCS == 1 ]] && echo ', deploy docs to GitHub Pages')."
read -r -p "Type the version ($VERSION) to continue: " CONFIRM
[[ "$CONFIRM" == "$VERSION" ]] || fail "confirmation did not match; nothing was published."

uv publish --publish-url "$PUBLISH_URL" --check-url "$CHECK_URL" dist/*.whl dist/*.tar.gz

if [[ "$REPOSITORY" == testpypi ]]; then
  cat <<MSG

Published to TestPyPI. Try it in a clean environment:
  uv run --isolated --no-project --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match \\
    --with "$NAME==$VERSION" python -c "import ume; print(ume.__version__)"
MSG
  exit 0
fi

# A source release may already be tagged at this commit (make release-tag, decision 0024).
git rev-parse --verify --quiet "refs/tags/v$VERSION" >/dev/null || git tag -a "v$VERSION" -m "$NAME $VERSION"
git push origin "v$VERSION"

if [[ $DOCS == 1 ]]; then
  # One deploy path for the docs: the GitHub Pages workflow, run on the new tag.
  gh workflow run docs.yml --ref "v$VERSION"
  echo "Docs deploy started: gh run list --workflow docs.yml"
fi
echo "Released $NAME $VERSION."
