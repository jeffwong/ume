"""The running example of the posts: accounts with profiles, a prompt, a ramp.

A port of ``simulate_accounts`` and ``truth_pair`` from the conditionally
sufficient statistics post. It uses numpy's random generator, so its draws
differ from the R original, and the posts' printed numbers are not reproduced
(decision 0013). The data generating process is the same.
"""

from __future__ import annotations

from typing import Any, Literal

import narwhals as nw
import numpy as np
from numpy.typing import NDArray

PROPENSITY = (0.05, 0.10, 0.25, 0.50)
Backend = Literal["pandas", "polars"]


def _logistic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_columns(
    n_accounts: int = 20_000,
    propensity: tuple[float, ...] = PROPENSITY,
    *,
    seed: int | np.random.Generator = 2026,
    force_z: int | None = None,
    full_compliance: bool = False,
    one_profile: bool = False,
    heavy_tail: float = 0.0,
) -> dict[str, NDArray[Any]]:
    """Simulate profile rows as a dict of numpy columns. See :func:`simulate_accounts`."""
    rng = np.random.default_rng(seed)
    n_days = len(propensity)
    day_prob = np.arange(n_days, 0, -1, dtype=np.float64)
    day = rng.choice(np.arange(1, n_days + 1), size=n_accounts, p=day_prob / day_prob.sum())
    n_profiles = np.ones(n_accounts, dtype=np.int64) if one_profile else np.minimum(1 + rng.poisson(1.2, n_accounts), 5)
    quality = rng.normal(0, 0.5, n_accounts)  # shared by the account's profiles
    engagement = rng.normal(0, 0.5, n_accounts)  # account part of engagement
    z_account = rng.binomial(1, np.asarray(propensity)[day - 1])
    if force_z is not None:
        z_account = np.full(n_accounts, force_z)

    a = np.repeat(np.arange(n_accounts), n_profiles)
    n = a.size
    profile = np.concatenate([np.arange(1, k + 1) for k in n_profiles])
    kids = rng.binomial(1, np.where(profile == 1, 0.05, 0.3))
    device = rng.choice(np.array(["tv", "mobile", "web"]), size=n, p=[0.5, 0.35, 0.15])
    engagement_p = engagement[a] + rng.normal(0, 0.8, n)  # the profile's engagement, not observed
    d = day[a]
    x1 = rng.lognormal(0.7 - 0.12 * d - 0.6 * kids + 0.3 * engagement_p, 0.6)
    x2 = rng.gamma(2.0, 1.0, n) * (1.2 - 0.15 * d)
    complier = rng.binomial(1, _logistic(-0.3 + 0.5 * np.log(x1) + 0.6 * engagement_p - 0.8 * kids))
    z = z_account[a]
    t = z if full_compliance else z * complier
    y = (
        0.5 + 0.8 * x1 + 0.1 * x2 + 0.5 * engagement_p + quality[a]
        + t * (0.3 + 0.15 * x1 - 0.2 * kids)
        + rng.normal(0, 0.5 + 0.3 * x1)
    )  # fmt: skip
    tv, mobile = (device == "tv"), (device == "mobile")
    plays = 1 + rng.poisson(np.exp(1 + 0.4 * np.log(x1) + 0.3 * engagement_p + 0.2 * tv) * (1 + 0.15 * t))
    rebuffers = rng.poisson(
        plays * 0.08 * np.exp(-0.4 * t * (1 + 0.5 * mobile) + 0.4 * quality[a] - 0.2 * engagement_p)
    )
    if heavy_tail:
        # Drawn last, so the other columns are the same as without it. The shock does
        # not depend on T, so the true effects do not change.
        y = y + heavy_tail * rng.standard_t(2.5, n_accounts)[a]
    return {
        "account": a + 1,
        "profile": profile,
        "day": d,
        "Z": z.astype(np.int64),
        "T": t.astype(np.int64),
        "kids": kids.astype(np.int64),
        "device": device,
        "x1": x1,
        "x2": x2,
        "y": y,
        "D": plays.astype(np.float64),
        "N": rebuffers.astype(np.float64),
        "complier": complier.astype(np.int64),
    }


def simulate_accounts(
    n_accounts: int = 20_000,
    propensity: tuple[float, ...] = PROPENSITY,
    *,
    seed: int | np.random.Generator = 2026,
    force_z: int | None = None,
    full_compliance: bool = False,
    one_profile: bool = False,
    heavy_tail: float = 0.0,
    backend: Backend = "polars",
) -> Any:
    """Simulate the running example: profiles of accounts in a ramped encouragement experiment.

    Each account enters on day 1 to 4 and is prompted ($Z = 1$) with that day's
    propensity. Each of its 1 to 5 profiles has covariates (hours per day `x1`,
    tenure `x2`, `kids`, `device`), and a prompted profile turns the setting on
    ($T = 1$) if it is a complier. Outcomes: hours `y`, plays `D`, rebuffers `N`.
    An unobserved engagement drives both take-up and outcomes, and profiles of
    an account share a network quality, so clusters matter.

    Parameters
    ----------
    n_accounts : int
        Number of accounts (clusters).
    propensity : tuple of float
        Propensity of $Z = 1$ for each day.
    seed : int or numpy.random.Generator
        Random seed.
    force_z : {0, 1}, optional
        Assign every account to this arm (for computing the truth).
    full_compliance : bool
        $T = Z$: no encouragement.
    one_profile : bool
        One profile per account: no clusters.
    heavy_tail : float
        Scale of an account-level Student-t (2.5 degrees of freedom) shock added
        to hours: heavy-tailed clusters, the "whales" of the BUM post. 0 adds none.
    backend : {"polars", "pandas"}
        The kind of frame to return.

    Returns
    -------
    polars.DataFrame or pandas.DataFrame
        One row per profile, grouped by account. The column `complier` is known
        only because the data is simulated.
    """
    columns = simulate_columns(
        n_accounts,
        propensity,
        seed=seed,
        force_z=force_z,
        full_compliance=full_compliance,
        one_profile=one_profile,
        heavy_tail=heavy_tail,
    )
    return nw.to_native(nw.from_dict(columns, backend=backend))


def truth(seed: int = 1, n_accounts: int = 1_000_000) -> dict[str, float]:
    """The true complier effects, from simulating the same accounts twice.

    Every account is prompted in one world and none in the other; the same
    profiles are compliers in both. Returns the effect on compliers' hours
    $\\tau_{LATE}$, on their rebuffers per play $\\tau_R$, and the relative effect
    $\\rho_R$.
    """
    p1 = simulate_columns(n_accounts, seed=seed, force_z=1)
    p0 = simulate_columns(n_accounts, seed=seed, force_z=0)
    c = p1["complier"] == 1
    rate_1 = p1["N"][c].sum() / p1["D"][c].sum()
    rate_0 = p0["N"][c].sum() / p0["D"][c].sum()
    return {
        "tau_LATE": float(np.mean(p1["y"][c] - p0["y"][c])),
        "tau_R": float(rate_1 - rate_0),
        "rho_R": float(rate_1 / rate_0 - 1),
    }
