"""The effect grammar reproduces the posts' effects, and its gradients are exact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS

import ume
from designs import Case
from ume._data import model_data
from ume._effects import Levels, value_and_gradient


@pytest.fixture(scope="module")
def fit() -> ume.UniversalFit:
    exp = Case(True, True, True, "ratio").experiment()
    return exp.metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp")).fit()


def lines(fit: ume.UniversalFit) -> dict[str, float]:
    """Each outcome's treatment line at the compliers' mean, as in the posts."""
    x_c = fit.x_C - fit.x_ref
    delta = np.concatenate([np.zeros(fit.beta.shape[0] // 2), [1.0], x_c])
    return {label: float(delta @ fit.beta[:, j]) for j, label in enumerate(fit.outcome_labels)}


def test_grammar_reproduces_the_posts_effects(fit: ume.UniversalFit) -> None:
    eff, ell = fit.effects(), lines(fit)
    assert eff.estimate("hours") == pytest.approx(ell["y"], rel=1e-12)
    tau_r = ell["T·N"] / ell["T·D"] - (ell["T·N"] - ell["N"]) / (ell["T·D"] - ell["D"])
    rho_r = (ell["T·N"] / ell["T·D"]) / ((ell["T·N"] - ell["N"]) / (ell["T·D"] - ell["D"])) - 1
    assert eff.estimate("rpp") == pytest.approx(tau_r, rel=1e-12)
    assert eff.estimate("rpp", "relative") == pytest.approx(rho_r, rel=1e-12)
    assert eff.estimate("hours", "relative") == pytest.approx(ell["T·y"] / (ell["T·y"] - ell["y"]) - 1, rel=1e-12)


def test_a_mean_is_a_ratio_with_denominator_one(fit: ume.UniversalFit) -> None:
    # The line of the outcome T is exactly 1, so mu_1(T)/... reduces: check the closed form directly.
    levels = Levels(fit.beta, fit.x_C - fit.x_ref, fit.design.layout)
    assert levels.mu1(None) == 1.0 and levels.mu0(None) == 1.0
    mean_effect = ume.Effect(ume.mean("y"))
    assert mean_effect(levels) == pytest.approx(levels.line("y"), rel=1e-12)


def test_complex_step_gradient_matches_finite_differences(fit: ume.UniversalFit) -> None:
    effect = ume.Effect(fit.design.metrics[1], "relative")
    x0 = fit.x_C - fit.x_ref
    _, grad = value_and_gradient(effect, fit.beta, x0, fit.design.layout)
    flat, h = fit.beta.reshape(-1, order="F"), 1e-6
    for j in range(flat.size):
        up, down = flat.copy(), flat.copy()
        up[j] += h
        down[j] -= h
        f = [effect(Levels(v.reshape(fit.beta.shape, order="F"), x0, fit.design.layout)) for v in (up, down)]
        assert grad[j] == pytest.approx((f[0] - f[1]) / (2 * h), rel=1e-5, abs=1e-8)


def test_custom_effect_equals_builtin(fit: ume.UniversalFit) -> None:
    custom = ume.effect("rpp", lambda m: m.mu1("N") / m.mu1("D") - m.mu0("N") / m.mu0("D"))
    out = fit.effects(custom)
    assert out.estimate("rpp", "custom") == pytest.approx(fit.effects().estimate("rpp"), rel=1e-12)
    assert out.std_err("rpp", "custom") == pytest.approx(fit.effects().std_err("rpp"), rel=1e-10)


def test_segment_reads_the_line_at_x0(fit: ume.UniversalFit) -> None:
    eff = fit.effects(ume.Effect(ume.mean("y", name="hours")), at={"x1": 1.0, "kids": 1, "device": "mobile"})
    labels = fit.x_labels
    x0 = fit.x_C - fit.x_ref
    x0[labels.index("x1")] = 1.0 - 2.0
    x0[labels.index("kids")] = 1.0
    x0[labels.index("device[mobile]")], x0[labels.index("device[web]")] = 1.0, 0.0
    delta = np.concatenate([np.zeros(fit.beta.shape[0] // 2), [1.0], x0])
    assert eff.estimate("hours") == pytest.approx(float(delta @ fit.beta[:, 0]), rel=1e-12)
    with pytest.raises(ValueError, match="unknown covariates"):
        fit.effects(at={"nope": 1})


def test_first_stage_f_matches_linearmodels() -> None:
    exp = Case(True, True, True, "mean").experiment()
    fit = exp.fit()
    data = model_data(exp.frame, exp.design)
    x = data.r[:, 1:]
    k = x.shape[1]
    df = pd.DataFrame({"one": 1.0, **{f"x{j}": x[:, j] for j in range(k)}})
    for j in range(k):
        df[f"Tx{j}"], df[f"Zx{j}"] = data.t * x[:, j], data.z * x[:, j]
    df["T"], df["Z"] = data.t, data.z
    res = IV2SLS(
        data.y[:, 0],
        df[["one"] + [f"x{j}" for j in range(k)]],
        df[["T"] + [f"Tx{j}" for j in range(k)]],
        df[["Z"] + [f"Zx{j}" for j in range(k)]],
        weights=data.w,
    ).fit(cov_type="clustered", clusters=pd.Series(data.cluster), debiased=False)
    wald = res.first_stage.diagnostics["f.stat"].to_numpy()  # chi2(d) Wald statistics
    table = fit.first_stage()
    assert table is not None
    np.testing.assert_allclose(np.array(table.to_dict()["robust F"]) * (k + 1), wald, rtol=1e-8)


def test_itt_without_covariates_is_the_difference_in_weighted_means() -> None:
    exp = Case(True, True, True, "mean").experiment()
    exp = ume.annotate(exp.frame, cluster="account", instrument="Z", treatment="T", cohort="day")
    exp = exp.metrics(ume.mean("y", name="hours"))
    itt = exp.fit().itt()
    data = model_data(exp.frame, exp.design)
    z, w, y = data.z, data.w, data.y[:, 0]
    expected = (w * z * y).sum() / (w * z).sum() - (w * (1 - z) * y).sum() / (w * (1 - z)).sum()
    assert itt.estimate("hours") == pytest.approx(expected, rel=1e-10)


@pytest.mark.slow
def test_estimates_are_near_the_truth() -> None:
    from ume.testing import simulate_accounts, truth

    true = truth(seed=1, n_accounts=400_000)
    df = simulate_accounts(20_000, seed=2026)
    exp = ume.annotate(
        df,
        cluster="account",
        instrument="Z",
        treatment="T",
        cohort="day",
        covariates=ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv")),
    )
    eff = exp.metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp")).fit().effects()
    assert abs(eff.estimate("hours") - true["tau_LATE"]) < 4 * eff.std_err("hours")
    assert abs(eff.estimate("rpp") - true["tau_R"]) < 4 * eff.std_err("rpp")
