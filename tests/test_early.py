r"""Invariant 8 (delta vectors early) and the linearized standard error.

Delta vectors early compute an effect's standard error in two passes over the
rows (first moments, then $\sum_a \psi_a^2$) without second moments. These tests
check that:

- the two passes equal the compressed fit in all 16 designs, for the sandwich
  and for the linearized standard error;
- the linearized standard error equals the sandwich exactly when nothing but
  $\hat\beta$ can move (no covariates, no cohort);
- it tracks the Bayesian bootstrap's standard deviation;
- it works at segments and from compressed data;
- unknown methods are rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

import ume
from designs import CASES, Case
from ume.testing import simulate_accounts


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("se", ["sandwich", "linearized"])
def test_two_passes_equal_the_compressed_fit(case: Case, se: str) -> None:
    """Invariant 8: two passes give the same estimates and standard errors as the compressed fit, for both methods."""
    exp = case.experiment()
    end, early = exp.fit().effects(se=se), exp.effects(se=se)
    np.testing.assert_allclose([r[3] for r in early.rows], [r[3] for r in end.rows], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose([r[4] for r in early.rows], [r[4] for r in end.rows], rtol=1e-7)


@pytest.mark.parametrize(("encouraged", "clustered"), [(False, False), (False, True), (True, False), (True, True)])
def test_linearized_equals_sandwich_when_nothing_else_moves(encouraged: bool, clustered: bool) -> None:
    r"""Without covariates ($\bar x_C$ plays no role) and without a cohort (no propensity), the linearized standard
    error must equal the sandwich.
    """
    df = simulate_accounts(3000, seed=4, full_compliance=not encouraged, one_profile=not clustered)
    exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T" if encouraged else None)
    fit = exp.metrics(ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp")).fit()
    sandwich, linearized = fit.effects(), fit.effects(se="linearized")
    np.testing.assert_allclose([r[4] for r in linearized.rows], [r[4] for r in sandwich.rows], rtol=1e-6)


def test_linearized_tracks_the_bayesian_bootstrap() -> None:
    r"""The linearized standard error, which moves $\hat e$ and $\bar x_C$, is within 6% of the Bayesian bootstrap's
    SD for every default effect.
    """
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
    """The linearized standard error at a segment is the same from rows and from compressed data."""
    exp = Case(True, True, True, "mean").experiment()
    a = exp.fit().effects(at={"kids": 1}, se="linearized")
    b = exp.compress().fit().effects(at={"kids": 1}, se="linearized")
    np.testing.assert_allclose([r[4] for r in a.rows], [r[4] for r in b.rows], rtol=1e-8)
    assert "linearized" in repr(a)


def test_unknown_standard_errors_are_rejected() -> None:
    """Unknown standard-error methods raise a ValueError, in two passes and from a fit."""
    exp = Case(True, True, True, "mean").experiment()
    with pytest.raises(ValueError, match="se must be"):
        exp.effects(se="jackknife")
    with pytest.raises(ValueError, match="se must be"):
        exp.fit().effects(se="jackknife")
