"""Release metadata checks shared by the local scripts and the GitHub workflow.

Usage
-----
python scripts/release_version.py version   # print the package version
python scripts/release_version.py name      # print the distribution name
python scripts/release_version.py check     # fail unless the release is consistent
python scripts/release_version.py bump X.Y.Z  # set the version, date the changelog section
python scripts/release_version.py tag       # tag HEAD v<version> (annotated, local only)

``check`` verifies that

1. the version in ``Cargo.toml`` (the single source, decision 0003) is valid SemVer,
2. ``CHANGELOG.md`` has a ``## <version>`` section and no leftover ``## Unreleased`` content,
3. the tag ``v<version>`` does not exist yet, locally or on ``origin``, or already points at
   ``HEAD`` (a source release tagged before publishing, decision 0024).

``bump`` writes the version into ``Cargo.toml``, renames ``## Unreleased`` to
``## X.Y.Z (YYYY-MM-DD)`` and opens an empty ``## Unreleased`` above it. ``tag``
refuses a dirty tree or a failing ``check``, and never pushes.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")


def version() -> str:
    """Return the workspace version from the root ``Cargo.toml``."""
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    return str(cargo["workspace"]["package"]["version"])


def name() -> str:
    """Return the distribution name from ``pyproject.toml``."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return str(pyproject["project"]["name"])


def changelog_errors(v: str) -> list[str]:
    """Return problems with the CHANGELOG for release ``v``; empty if none."""
    text = (ROOT / "CHANGELOG.md").read_text()
    errors = []
    if not re.search(rf"^## \[?{re.escape(v)}\]?(\s|$)", text, flags=re.MULTILINE):
        errors.append(f"CHANGELOG.md has no '## {v}' section. Rename '## Unreleased' to '## {v} (YYYY-MM-DD)'.")
    unreleased = re.search(r"^## Unreleased\s*\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if unreleased and unreleased.group(1).strip():
        errors.append("CHANGELOG.md still has entries under '## Unreleased'.")
    return errors


def git(*args: str) -> str:
    """Run git in the repository root and return its stripped standard output ("" on failure)."""
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def tag_errors(v: str) -> list[str]:
    """Return an error if tag ``v<version>`` exists, locally or on origin, at a commit other than HEAD."""
    tag = f"v{v}"
    head = git("rev-parse", "HEAD")
    local = git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    remote = git("ls-remote", "--tags", "origin", f"refs/tags/{tag}^{{}}", f"refs/tags/{tag}").split()
    remote_commit = remote[-2] if remote else ""
    moved = [c for c in (local, remote_commit) if c and c != head]
    return [f"Tag {tag} already exists at another commit. Bump the version in Cargo.toml."] if moved else []


def bump(v: str) -> int:
    """Set the version in ``Cargo.toml`` and date the changelog's Unreleased section as ``v``."""
    if not SEMVER.match(v):
        print(f"error: {v!r} is not SemVer (X.Y.Z).", file=sys.stderr)
        return 1
    cargo = ROOT / "Cargo.toml"
    text, n = re.subn(
        r'(\[workspace\.package\][^\[]*?^version\s*=\s*)"[^"]*"', rf'\g<1>"{v}"', cargo.read_text(),
        count=1, flags=re.MULTILINE | re.DOTALL,
    )  # fmt: skip
    if n != 1:
        print("error: no version under [workspace.package] in Cargo.toml.", file=sys.stderr)
        return 1
    changelog = ROOT / "CHANGELOG.md"
    log = changelog.read_text()
    if not re.search(r"^## Unreleased[ \t]*$", log, flags=re.MULTILINE):
        print("error: CHANGELOG.md has no '## Unreleased' section.", file=sys.stderr)
        return 1
    log = re.sub(
        r"^## Unreleased[ \t]*$", f"## Unreleased\n\n## {v} ({time.strftime('%Y-%m-%d')})", log, count=1,
        flags=re.MULTILINE,
    )  # fmt: skip
    cargo.write_text(text)
    changelog.write_text(log)
    print(f"Version {v} in Cargo.toml; CHANGELOG.md section '## {v}' dated today. Review, rebuild, commit.")
    return 0


def tag() -> int:
    """Create the annotated tag ``v<version>`` at HEAD after the release checks; never push."""
    v = version()
    if git("status", "--porcelain"):
        print("error: uncommitted changes. Commit the release first.", file=sys.stderr)
        return 1
    if check():
        return 1
    if git("tag", "--list", f"v{v}"):
        print(f"Tag v{v} already points at HEAD.")
        return 0
    subprocess.run(["git", "tag", "-a", f"v{v}", "-m", f"{name()} {v}"], cwd=ROOT, check=True)
    print(f"Tagged v{v} at {git('rev-parse', '--short', 'HEAD')}. Push it with: git push origin v{v}")
    return 0


def check() -> int:
    """Run every release check; print problems and return a process exit code."""
    v = version()
    errors = [] if SEMVER.match(v) else [f"Version {v!r} in Cargo.toml is not SemVer."]
    errors += changelog_errors(v) + tag_errors(v)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if not errors:
        print(f"{name()} {v}: version, changelog and tag are consistent.")
    return 1 if errors else 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "version":
        print(version())
    elif command == "name":
        print(name())
    elif command == "check":
        sys.exit(check())
    elif command == "bump" and len(sys.argv) == 3:
        sys.exit(bump(sys.argv[2]))
    elif command == "tag":
        sys.exit(tag())
    else:
        sys.exit(f"unknown command {command!r}; use version, name, check, bump X.Y.Z or tag")
