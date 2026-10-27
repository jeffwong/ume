"""Collect every benchmark's results into one file for the docs' Benchmarks page.

Reads:

- criterion's results in ``target/criterion`` (the Rust kernels), keeping only
  benchmarks written since ``--since`` (a Unix time; the Makefile records it
  before ``cargo bench``), so stale results of removed benchmarks are left out;
- pytest-benchmark's JSON (the Python API), ``benchmarks/results/python.json``;
- the scale benchmarks, ``benchmarks/results/scale.json``;

and writes ``benchmarks/results/benchmarks.json`` with the machine, the ume
version and the git commit they were measured on.

Run: ``make bench-results``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"


def criterion(since: float) -> list[dict[str, Any]]:
    """The criterion benchmarks run since `since`: mean time and its 95% interval, in seconds."""
    out = []
    for meta in sorted((ROOT / "target" / "criterion").glob("**/new/benchmark.json")):
        estimates = meta.parent / "estimates.json"
        if not estimates.exists() or estimates.stat().st_mtime < since:
            continue
        info, est = json.loads(meta.read_text()), json.loads(estimates.read_text())
        mean = est["mean"]
        out.append({
            "group": info["group_id"],
            "function": info["function_id"] or "",
            "mean_s": mean["point_estimate"] / 1e9,
            "low_s": mean["confidence_interval"]["lower_bound"] / 1e9,
            "high_s": mean["confidence_interval"]["upper_bound"] / 1e9,
        })  # fmt: skip
    return out


def pytest_benchmarks(path: Path) -> list[dict[str, Any]]:
    """pytest-benchmark's results: mean, standard deviation and rounds, in seconds."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [
        {"name": b["name"], "mean_s": st["mean"], "stddev_s": st["stddev"], "rounds": st["rounds"]}
        for b in data["benchmarks"]
        for st in [b["stats"]]
    ]


def processor() -> str:
    """The CPU's name: the chip on macOS (``sysctl``), else what ``platform`` reports."""
    try:
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        if name.strip():
            return name.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", type=float, default=0.0, help="ignore criterion results older than this Unix time")
    parser.add_argument("--out", type=Path, default=RESULTS / "benchmarks.json")
    a = parser.parse_args()

    import ume

    scale_path = RESULTS / "scale.json"
    report = {
        "measured": time.strftime("%Y-%m-%d %H:%M", time.localtime()),
        "machine": {
            "platform": platform.platform(),
            "processor": processor(),
            "cores": os.cpu_count(),
            "python": platform.python_version(),
        },
        "ume": ume.__version__,
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "rust": criterion(a.since),
        "python": pytest_benchmarks(RESULTS / "python.json"),
        "scale": json.loads(scale_path.read_text()) if scale_path.exists() else None,
    }
    a.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {a.out.relative_to(ROOT)}: {len(report['rust'])} Rust, {len(report['python'])} Python benchmarks")


if __name__ == "__main__":
    main()
