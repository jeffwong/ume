"""Invariant 8 (delta vectors early equal at the end) and the linearized standard error."""

from __future__ import annotations

import numpy as np
import pytest

import ume
from designs import CASES, Case
from ume.testing import simulate_accounts


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("se", ["sandwich", "linearized"])
def test_two_passes_equal_the_compressed_fit(case: Case, se: str) -> None:
    exp = case.experiment()
    end, early = exp.fit().effects(se=se), exp.effects(se=se)
    np.testing.assert_allclose([r[3] for r in early.rows], [r[3] for r in end.rows], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose([r[4] for r in early.rows], [r[4] for r in end.rows], rtol=1e-7)


@pytest.mark.parametrize(("encouraged", "clustered"), [(False, False), (False, True), (True, False), (True, True)])
def test_linearized_equals_sandwich_when_nothing_else_moves(encouraged: bool, clustered: bool) -> None:
    # No covariates (x_C plays no role) and no cohort (no propensity): the two must agree.
    df = simulate_accounts(3000, seed=4, full_compliance=not encouraged, one_profile=not clustered)
    exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T" if encouraged else None)
    fit = exp.metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp")).fit()
    sandwich, linearized = fit.effects(), fit.effects(se="linearized")
    np.testing.assert_allclose([r[4] for r in linearized.rows], [r[4] for r in sandwich.rows], rtol=1e-6)


def test_linearized_tracks_the_bayesian_bootstrap() -> None:
    df = simulate_accounts(6000, seed=12)
    cov = ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv"))
    exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T", cohort="day", covariates=cov)
    exp = exp.metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp"))
    fit = exp.fit()
    post = exp.bootstrap(3000, seed=2)
    for e in [ume.Effect(m, c) for m in exp.design.metrics for c in ("difference", "relative")]:
        sd = float(post.values(e).std(ddof=1))
        assert fit.effects(e, se="linearized").std_err(e.name, e.contrast) == pytest.approx(sd, rel=0.06)


def test_linearized_at_a_segment_and_from_compressed_data() -> None:
    exp = Case(True, True, True, "mean").experiment()
    a = exp.fit().effects(at={"kids": 1}, se="linearized")
    b = exp.compress().fit().effects(at={"kids": 1}, se="linearized")
    np.testing.assert_allclose([r[4] for r in a.rows], [r[4] for r in b.rows], rtol=1e-8)
    assert "linearized" in repr(a)


def test_unknown_standard_errors_are_rejected() -> None:
    exp = Case(True, True, True, "mean").experiment()
    with pytest.raises(ValueError, match="se must be"):
        exp.effects(se="jackknife")
    with pytest.raises(ValueError, match="se must be"):
        exp.fit().effects(se="jackknife")
