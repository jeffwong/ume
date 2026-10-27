"""End-to-end benchmarks at production scale: time and peak memory of every step.

Each step runs in a fresh subprocess on the same simulated data (the running
example at ``--accounts`` accounts, 1M by default: about 2.19M rows, $k = 5$,
$o = 6$), once per thread count. A subprocess per measurement means the peak
resident memory (RSS, which includes Rust's allocations) belongs to that step
alone. A ``baseline`` step that only simulates and annotates the data gives the
memory every step starts from, so ``peak - baseline`` is the step's own cost.

Run: ``uv run python benchmarks/scale.py`` (several minutes), or through
``make bench-results``.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STEPS = {
    "baseline": "simulate and annotate the rows (not timed)",
    "fit_rows": "fit the universal model on the rows",
    "compress": "one compression pass",
    "fit_compressed": "fit from the compressed data (memory includes the compression before it)",
    "gaussian_posterior": "compress, then B Gaussian posterior draws",
    "exact_bootstrap": "exact Bayesian bootstrap, B draws, one pass",
    "two_pass_effects": "influence standard errors in two passes",
}


def peak_rss_bytes() -> int:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def child(step: str, accounts: int, threads: int, draws: int) -> dict[str, Any]:
    """Run one step in this process and return its wall time and peak memory."""
    import ume
    from ume.testing import simulate_accounts

    ume.set_threads(threads)
    df = simulate_accounts(accounts, seed=1)
    exp = ume.annotate(
        df, cluster="account", instrument="Z", treatment="T", cohort="day",
        covariates=ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv")),
    ).metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp"))  # fmt: skip
    cd = exp.compress() if step == "fit_compressed" else None
    run = {
        "baseline": lambda: None,
        "fit_rows": exp.fit,
        "compress": exp.compress,
        "fit_compressed": lambda: cd.fit(),  # type: ignore[union-attr]
        "gaussian_posterior": lambda: exp.compress().posterior(draws, seed=1),
        "exact_bootstrap": lambda: exp.bootstrap(draws, seed=1),
        "two_pass_effects": lambda: exp.effects(se="influence"),
    }[step]
    start = time.perf_counter()
    run()
    seconds = time.perf_counter() - start
    return {"step": step, "threads": threads, "seconds": seconds, "peak_rss_bytes": peak_rss_bytes(), "rows": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accounts", type=int, default=1_000_000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--threads", default="1,4", help="comma-separated thread counts")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "scale.json")
    parser.add_argument("--child", nargs=2, metavar=("STEP", "THREADS"), help=argparse.SUPPRESS)
    a = parser.parse_args()
    if a.child:
        print(json.dumps(child(a.child[0], a.accounts, int(a.child[1]), a.draws)))
        return
    results = []
    for threads in [int(t) for t in a.threads.split(",")]:
        for step in STEPS:
            cmd = [sys.executable, __file__, "--accounts", str(a.accounts), "--draws", str(a.draws),
                   "--child", step, str(threads)]  # fmt: skip
            out = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()[-1])
            out["description"] = STEPS[step]
            results.append(out)
            peak = out["peak_rss_bytes"] / 2**20
            print(f"threads={threads} {step:<20} {out['seconds']:8.3f} s  peak {peak:8.0f} MiB")
    meta = {"accounts": a.accounts, "draws": a.draws, "rows": results[0]["rows"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))


if __name__ == "__main__":
    main()
