"""Invariant 1 on rows: the Rust engine equals the numpy reference, which equals the textbook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS

from designs import CASES, Case
from ume._data import CSR, model_data
from ume.testing import complier_mean, fit_rows, inverse_propensity_weights

# linearmodels takes sqrt of a tiny negative variance for outcome columns that are fitted exactly.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")


@pytest.fixture(scope="module", params=CASES, ids=lambda c: c.id)
def case(request: pytest.FixtureRequest) -> Case:
    return request.param


def dense(r: np.ndarray | CSR) -> np.ndarray:
    return r.toarray() if isinstance(r, CSR) else r


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_engine_matches_reference(case: Case, sparse: bool) -> None:
    exp = case.experiment()
    data = model_data(exp.frame, exp.design, sparse=sparse)
    assert data.sparse is sparse
    fit = exp.fit(sparse=sparse)
    ref = fit_rows(dense(data.r), data.z, data.t, data.y, data.w, data.cluster)
    np.testing.assert_allclose(fit.beta, ref.beta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit.cov, ref.cov, rtol=1e-8, atol=1e-14)
    x_c = complier_mean(dense(data.r)[:, 1:], data.z, data.t, data.w)
    np.testing.assert_allclose(fit.x_C - fit.x_ref, x_c, rtol=1e-10, atol=1e-12)


def test_features_are_detected(case: Case) -> None:
    f = case.experiment().features()
    assert (f.encouraged, f.clustered, f.adaptive) == (case.encouraged, case.clustered, case.ramp)
    assert f.estimand == ("LATE" if case.encouraged else "ATE")
    assert f.covariance == ("CR0" if case.clustered else "HC0")


def test_weights_match_reference(case: Case) -> None:
    exp = case.experiment()
    data = model_data(exp.frame, exp.design)
    if not case.ramp:
        assert np.all(data.w == 1)
        return
    frame = exp.frame
    expected = inverse_propensity_weights(data.z, frame["day"].to_numpy(), data.cluster)
    np.testing.assert_allclose(data.w, expected, rtol=1e-12)


def textbook(case: Case, outcome: np.ndarray, data, x_c: np.ndarray) -> tuple[float, float]:
    """linearmodels on the rows, covariates centered at x̄_C so that the coefficient on T is the effect."""
    x = dense(data.r)[:, 1:] + np.asarray(data.design.covariates.x_ref()) - x_c
    cols = {f"x{j}": x[:, j] for j in range(x.shape[1])}
    df = pd.DataFrame({"one": 1.0, **cols})
    for j in range(x.shape[1]):
        df[f"Tx{j}"] = data.t * x[:, j]
        df[f"Zx{j}"] = data.z * x[:, j]
    df["T"], df["Z"] = data.t, data.z
    xs = [f"x{j}" for j in range(x.shape[1])]
    cov = dict(cov_type="clustered", clusters=pd.Series(data.cluster)) if case.clustered else dict(cov_type="robust")
    if case.encouraged:
        endog = ["T"] + [f"Tx{j}" for j in range(x.shape[1])]
        instr = ["Z"] + [f"Zx{j}" for j in range(x.shape[1])]
        model = IV2SLS(outcome, df[["one", *xs]], df[endog], df[instr], weights=data.w)
    else:
        model = IV2SLS(
            outcome, df[["one", *xs, "T"] + [f"Tx{j}" for j in range(x.shape[1])]], None, None, weights=data.w
        )
    res = model.fit(debiased=False, **cov)
    return float(res.params["T"]), float(res.std_errors["T"])


def test_effects_match_textbook(case: Case) -> None:
    exp = case.experiment()
    fit = exp.fit()
    data = model_data(exp.frame, exp.design)
    eff = fit.effects()
    name = exp.design.metrics[0].name
    if case.metric == "mean":
        est, se = textbook(case, data.y[:, 0], data, fit.x_C)
        assert eff.estimate(name) == pytest.approx(est, rel=1e-9, abs=1e-12)
        assert eff.std_err(name) == pytest.approx(se, rel=1e-8)
    else:
        layout = exp.design.layout
        line = {c.label: textbook(case, data.y[:, i], data, fit.x_C)[0] for i, c in enumerate(layout.columns)}
        tau_r = line["T·N"] / line["T·D"] - (line["T·N"] - line["N"]) / (line["T·D"] - line["D"])
        assert eff.estimate(name) == pytest.approx(tau_r, rel=1e-9, abs=1e-12)
