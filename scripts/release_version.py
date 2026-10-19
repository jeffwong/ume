"""Release metadata checks shared by the local scripts and the GitHub workflow.

Usage
-----
python scripts/release_version.py version   # print the package version
python scripts/release_version.py name      # print the distribution name
python scripts/release_version.py check     # fail unless the release is consistent

``check`` verifies that

1. the version in ``Cargo.toml`` (the single source, decision 0003) is valid SemVer,
2. ``CHANGELOG.md`` has a ``## <version>`` section and no leftover ``## Unreleased`` content,
3. the tag ``v<version>`` does not exist yet, locally or on ``origin``.
"""

from __future__ import annotations

import re
import subprocess
import sys
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


def tag_errors(v: str) -> list[str]:
    """Return an error if tag ``v<version>`` already exists locally or on origin."""
    tag = f"v{v}"
    local = subprocess.run(["git", "tag", "--list", tag], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    remote = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    return [f"Tag {tag} already exists. Bump the version in Cargo.toml."] if local or remote else []


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
    else:
        sys.exit(f"unknown command {command!r}; use version, name or check")
