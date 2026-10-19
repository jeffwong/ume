"""Invariants 3 to 7: identity draw, exact equals rows, weight streams, moments identity, linearization."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ume
from designs import CASES, COVARIATES, Case
from ume import _core
from ume._data import cluster_keys, model_data
from ume.testing import draw_rows, simulate_accounts


@pytest.fixture(scope="module")
def exp() -> ume.Experiment:
    df = simulate_accounts(2000, seed=21)
    ann = ume.annotate(df, cluster="account", instrument="Z", treatment="T", cohort="day", covariates=COVARIATES)
    return ann.metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp"))


# ---- invariant 5: weight streams ----------------------------------------------------------------


def test_bayesian_weights_match_numpy_philox_bit_for_bit() -> None:
    seed, keys, b = 2026, np.array([1, 7, 123456789, 2**63 + 5], dtype=np.uint64), 12
    ours = _core.bootstrap_weights(keys, b, seed, "bayesian")
    for i, key in enumerate(keys):
        for draw in range(4, b):  # blocks >= 1, so numpy's pre-increment needs no carry
            block, word = divmod(draw, 4)
            # A uint64 array: numpy rounds Python ints >= 2**63 in a key list through a float.
            bg = np.random.Philox(key=np.array([seed, key], dtype=np.uint64), counter=[block - 1, 0, 0, 0])
            x = int(bg.random_raw(4)[word])
            u = ((x >> 12) + 0.5) * 2.0**-52
            assert ours[i, draw] == -np.log(u)


def test_weights_have_the_right_moments() -> None:
    keys = np.arange(200_000, dtype=np.uint64)
    for scheme, third in [("bayesian", 2.0), ("poisson", 1.0)]:
        w = _core.bootstrap_weights(keys, 1, 3, scheme)[:, 0]
        assert w.mean() == pytest.approx(1, abs=0.01)
        assert w.var() == pytest.approx(1, abs=0.02)
        assert ((w - w.mean()) ** 3).mean() == pytest.approx(third, abs=0.1)
    counts = _core.bootstrap_weights(np.arange(500, dtype=np.uint64), 40, 3, "multinomial")
    assert np.all(counts.sum(axis=0) == 500)


def test_weights_depend_only_on_seed_cluster_and_draw(exp: ume.Experiment) -> None:
    shuffled = ume.Experiment(exp.frame.sample(fraction=1.0, shuffle=True, seed=4), exp.design)
    a, b = exp.bootstrap(50, seed=9), shuffled.bootstrap(50, seed=9)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9, atol=1e-12)
    assert not np.allclose(a.beta, exp.bootstrap(50, seed=10).beta)


# ---- invariants 3 and 4: identity draw, exact one pass equals rows -------------------------------


def test_identity_draw_reproduces_the_point_estimate(exp: ume.Experiment) -> None:
    cd = exp.compress()
    fit = cd.fit()
    sums = np.concatenate([cd.g[:, None], cd.first], axis=1)[:, None, :]  # every xi = 1
    z, cohort, n_cohorts = cd._cohorts()
    raw = _core.fit_draws(cd.d, cd.o, np.ascontiguousarray(sums), z, cohort, n_cohorts, True, True)
    draw = ume.Draws(raw, cd.design, "exact", "unit", 0)
    np.testing.assert_allclose(draw.beta[0], fit.beta, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(draw.x_C[0], fit.x_C, rtol=1e-12)


@pytest.mark.parametrize("weights", ["bayesian", "poisson", "multinomial"])
def test_exact_bootstrap_equals_refits_on_rows(exp: ume.Experiment, weights: str) -> None:
    draws = exp.bootstrap(6, weights=weights, seed=5)
    data = model_data(exp.frame, exp.design)
    keys = cluster_keys(exp.frame["account"].to_numpy())
    order = np.unique(keys, return_index=True)[1]
    pass_keys = keys[np.sort(order)]  # the frame is sorted by account: pass order = first appearance
    xi = _core.bootstrap_weights(pass_keys, 6, 5, weights)
    codes = np.unique(keys, return_inverse=True)[1]
    # cluster codes are sorted by key; map pass order to codes
    xi_by_code = np.empty_like(xi)
    xi_by_code[np.unique(keys[np.sort(order)], return_inverse=True)[1]] = xi
    cohort = exp.frame["day"].to_numpy()
    for b in range(6):
        beta, x_c = draw_rows(dense(data.r), data.z, data.t, data.y, codes, cohort, xi_by_code[:, b])
        np.testing.assert_allclose(draws.beta[b], beta, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(draws._x_c[b], x_c, rtol=1e-9, atol=1e-11)


def dense(r: object) -> np.ndarray:
    from ume._data import CSR

    return r.toarray() if isinstance(r, CSR) else np.asarray(r)


# ---- invariant 6: the compressed row is the posterior's first two moments ------------------------


def test_moments_identity() -> None:
    df = simulate_accounts(400, seed=8)
    exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T").metrics(ume.mean("y"))
    cd = exp.compress()
    from ume._compress import BootstrapPass, _feed
    from ume._source import open_source

    source = open_source(exp.frame, exp.design)
    boot = BootstrapPass(source.design, 40_000, "bayesian", 1, None)
    _feed(boot, source)
    drawsums = boot.finish()
    sums = drawsums.sums  # (H, B, 1 + q)
    for h, stratum in enumerate(drawsums.strata):
        i = cd.strata.index(stratum)
        mean = np.concatenate([[cd.g[i]], cd.first[i]])
        second = np.block([[cd.g[i], cd.first[i][None, :]], [cd.first[i][:, None], cd.second[i]]])
        np.testing.assert_allclose(sums[h].mean(axis=0), mean, rtol=0.03, atol=0.5)
        cov = np.cov(sums[h], rowvar=False)
        np.testing.assert_allclose(cov, second, rtol=0.06, atol=0.05 * np.abs(second).max())


# ---- invariant 7: the posterior agrees with the sandwich -----------------------------------------


@pytest.mark.parametrize("case", [c for c in CASES if c.metric == "mean"], ids=lambda c: c.id)
def test_exact_posterior_sd_is_close_to_the_sandwich(case: Case) -> None:
    exp = case.experiment(n_accounts=3000)
    fit = exp.fit()
    sd = fit.effects(se=exp.bootstrap(400, seed=1)).std_err("hours")
    assert sd == pytest.approx(fit.effects().std_err("hours"), rel=0.15)


@pytest.mark.parametrize("case", [c for c in CASES if c.metric == "mean"], ids=lambda c: c.id)
def test_gaussian_posterior_sd_is_close_to_the_sandwich(case: Case) -> None:
    # The Gaussian engine needs large strata: every stratum here has 200+ clusters (decision 0019).
    exp = case.experiment(n_accounts=12_000)
    cd = exp.compress()
    assert cd.g.min() >= 200
    fit = cd.fit()
    sd = fit.effects(se=cd.posterior(400, seed=1)).std_err("hours")
    assert sd == pytest.approx(fit.effects().std_err("hours"), rel=0.15)


def test_gaussian_posterior_warns_on_small_strata() -> None:
    exp = Case(False, False, True, "mean").experiment(n_accounts=3000)  # smallest stratum: 60 clusters
    with pytest.warns(UserWarning, match="fewer than 200 clusters"):
        exp.compress().posterior(50, seed=1)


# ---- the Draws object ----------------------------------------------------------------------------


def test_draws_answer_probability_questions(exp: ume.Experiment) -> None:
    post = exp.bootstrap(500, seed=3)
    rho = ume.Effect(exp.design.metrics[1], "relative")
    values = post.values(rho)
    assert values.shape == (post.B,)
    assert post.prob(rho, below=-0.1) == pytest.approx(float(np.mean(values < -0.1)))
    seg = post.values(ume.Effect(exp.design.metrics[0]), at={"kids": 1})
    assert seg.shape == (post.B,) and not np.allclose(seg, post.values(ume.Effect(exp.design.metrics[0])))
    assert "bayesian bootstrap" in repr(post)
    assert "bayesian bootstrap" in repr(exp.fit().effects(se=exp.bootstrap(50)))


def test_degenerate_poisson_draws_are_dropped() -> None:
    # A tiny cohort makes some Poisson draws leave one arm with no weight.
    df = simulate_accounts(300, propensity=(0.02, 0.5, 0.5, 0.5), seed=6)
    exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T", cohort="day").metrics(ume.mean("y"))
    draws = exp.bootstrap(300, weights="poisson", seed=1)
    assert draws.n_failed > 0 and 300 - draws.n_failed == draws.B
    assert np.isfinite(draws.beta).all()
    assert "degenerate draws dropped" in repr(draws)


def test_unknown_weights_are_rejected(exp: ume.Experiment) -> None:
    with pytest.raises(ValueError, match="unknown weights"):
        exp.bootstrap(10, weights="jackknife")
    _ = pl  # polars is the frame type of the fixtures
