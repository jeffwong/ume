"""Calibration Monte Carlo: are ume's intervals calibrated against a known truth?

Each replication simulates the running example with a new seed, fits it, and
records every method's estimate and uncertainty for the LATE and the ratio
effect. Across replications we report, per method and effect:

- **bias** of the estimate (the mean of the draws, for draws), with its Monte Carlo SE;
- **SE/SD**: the mean standard error (or SD of the draws) divided by the standard
  deviation of the point estimates across replications; 1 means the scale is right;
- **coverage** of the 95% interval: normal for the standard errors, the 2.5%-97.5%
  quantiles for the draws.

With R replications the Monte Carlo noise is about $1/\\sqrt{2(R-1)}$ for SE/SD
and $\\sqrt{0.95 \\cdot 0.05 / R}$ for coverage. ``per row (wrong)`` puts the
Bayesian bootstrap weight on each profile instead of each account: a negative
control that must under-cover (BUM post).

Scenarios:

- ``running``: the running example, 20,000 accounts;
- ``small-strata``: 3,000 accounts, so the smallest stratum has about 60 clusters;
- ``heavy-tails``: 20,000 accounts with an account-level Student-t(2.5) shock to hours.

Run: ``uv run python benchmarks/calibration.py --scenario running --reps 1000``
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import ume
from ume.testing import simulate_accounts, truth

COVARIATES = ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv"))
EFFECTS = {"LATE": ("hours", "difference"), "tau_R": ("rpp", "difference")}
SCENARIOS = {
    "running": {"accounts": 20_000, "heavy_tail": 0.0},
    "small-strata": {"accounts": 3_000, "heavy_tail": 0.0},
    "heavy-tails": {"accounts": 20_000, "heavy_tail": 2.0},
}
EXACT = {"bayesian, exact": "bayesian", "poisson, exact": "poisson", "multinomial, exact": "multinomial"}
METHODS = ["sandwich", "linearized", *EXACT, "gaussian posterior", "per row (wrong)"]


def annotate(df: object, cluster: str) -> ume.Experiment:
    exp = ume.annotate(df, cluster=cluster, instrument="Z", treatment="T", cohort="day", covariates=COVARIATES)
    return exp.metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp"))


def replication(args: tuple[int, int, float, int]) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """One replication: per method and effect, (estimate, scale, interval low, interval high)."""
    seed, n_accounts, heavy_tail, draws = args
    ume.set_threads(1)  # parallelism comes from the replications
    df = simulate_accounts(n_accounts, seed=seed, heavy_tail=heavy_tail)
    exp = annotate(df, "account")
    cd = exp.compress()
    fit = cd.fit()
    effects = {name: ume.Effect(next(m for m in exp.design.metrics if m.name == metric), contrast)
               for name, (metric, contrast) in EFFECTS.items()}  # fmt: skip
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for method, se in (("sandwich", "sandwich"), ("linearized", "linearized")):
        table = fit.effects(*effects.values(), se=se)
        out[method] = {}
        for name, e in effects.items():
            est, s = table.estimate(e.name, e.contrast), table.std_err(e.name, e.contrast)
            out[method][name] = (est, s, est - 1.959964 * s, est + 1.959964 * s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the small-strata warning is expected there
        all_draws = {method: exp.bootstrap(draws, weights=w, seed=seed) for method, w in EXACT.items()}
        all_draws["gaussian posterior"] = cd.posterior(draws, seed=seed)
        all_draws["per row (wrong)"] = annotate(df.with_row_index("row"), "row").bootstrap(draws, seed=seed)
    for method, d in all_draws.items():
        out[method] = {}
        for name, e in effects.items():
            v = d.values(e)
            lo, hi = np.quantile(v, [0.025, 0.975])
            out[method][name] = (float(v.mean()), float(v.std(ddof=1)), float(lo), float(hi))
    return out


def summarize(results: list[dict], true: dict[str, float]) -> list[dict]:
    rows = []
    r = len(results)
    for method in METHODS:
        for name in EFFECTS:
            est = np.array([res[method][name][0] for res in results])
            scale = np.array([res[method][name][1] for res in results])
            lo = np.array([res[method][name][2] for res in results])
            hi = np.array([res[method][name][3] for res in results])
            sd = np.array([res["sandwich"][name][0] for res in results]).std(ddof=1)  # SD of the estimates
            rows.append({
                "method": method,
                "effect": name,
                "bias": float(est.mean() - true[name]),
                "bias_mc_se": float(est.std(ddof=1) / np.sqrt(r)),
                "se_over_sd": float(scale.mean() / sd),
                "coverage": float(np.mean((lo <= true[name]) & (true[name] <= hi))),
            })  # fmt: skip
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="running")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=None)
    a = parser.parse_args()
    scenario = SCENARIOS[a.scenario]
    out = a.out or Path(__file__).parent / "results" / f"calibration-{a.scenario}.json"

    t0 = time.perf_counter()
    pairs = [truth(seed=s, n_accounts=1_000_000) for s in (1, 2)]
    true = {
        "LATE": float(np.mean([p["tau_LATE"] for p in pairs])),
        "tau_R": float(np.mean([p["tau_R"] for p in pairs])),
    }
    jobs = [(100_000 + i, scenario["accounts"], scenario["heavy_tail"], a.draws) for i in range(a.reps)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(replication, jobs, chunksize=8))
    rows = summarize(results, true)
    elapsed = time.perf_counter() - t0

    r = a.reps
    size = f"{scenario['accounts']:,} accounts, {a.draws} draws"
    print(f"scenario {a.scenario}: {r} replications, {size}, {elapsed:.0f} s")
    print(f"truth: LATE = {true['LATE']:.4f}, tau_R = {true['tau_R']:.5f}")
    print(f"Monte Carlo noise: SE/SD ±{1 / np.sqrt(2 * (r - 1)):.3f}, coverage ±{np.sqrt(0.95 * 0.05 / r):.3f}")
    print(f"{'method':<20}{'effect':<8}{'bias':>11}{'± MC SE':>10}{'SE/SD':>8}{'coverage':>10}")
    for row in rows:
        print(f"{row['method']:<20}{row['effect']:<8}{row['bias']:>11.5f}{row['bias_mc_se']:>10.5f}"
              f"{row['se_over_sd']:>8.3f}{row['coverage']:>10.1%}")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"scenario": a.scenario, "reps": r, "draws": a.draws, "truth": true, "seconds": elapsed, **scenario}
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
