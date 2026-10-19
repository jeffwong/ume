r"""Invariants 1 and 2 on compressed data: compression loses nothing, and shards merge exactly.

Compression reduces the rows to one compressed row per stratum: $G_h$,
$\sum u_a$ and $\sum u_a u_a^\top$. These tests check that:

- fitting the compressed data reproduces the row-level fit exactly (coefficients,
  CR0 covariance, compliers' mean, first-stage F, features, ITT), in all 16
  designs, with dense and sparse design matrices;
- compressing shards separately and adding them equals one pass, for random
  shard splits and for batches that split a cluster;
- misuse is refused: a cluster split across shards, rows not grouped by cluster
  in a streaming pass, a design not fixed before the pass;
- the compressed rows hold the documented blocks, and their size does not grow
  with the number of rows.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

import ume
from designs import CASES, COVARIATES, Case
from ume.testing import simulate_accounts


@pytest.fixture(scope="module", params=CASES, ids=lambda c: c.id)
def case(request: pytest.FixtureRequest) -> Case:
    return request.param


def assert_same_fit(a: ume.UniversalFit, b: ume.UniversalFit) -> None:
    """Assert that two fits agree: coefficients, covariance, compliers, features and first-stage F."""
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(a.cov, b.cov, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(a.x_C, b.x_C, rtol=1e-10, atol=1e-12)
    assert a.pi_C == pytest.approx(b.pi_C, rel=1e-12)
    assert a.features == b.features
    fa, fb = a.first_stage(), b.first_stage()
    assert (fa is None) == (fb is None)
    if fa is not None and fb is not None:
        np.testing.assert_allclose(fa.to_dict()["robust F"], fb.to_dict()["robust F"], rtol=1e-8)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_compressed_equals_rows(case: Case, sparse: bool) -> None:
    """Fitting the compressed data reproduces the row-level fit exactly, with dense and with sparse design
    matrices.
    """
    exp = case.experiment()
    assert_same_fit(exp.fit(), exp.compress(sparse=sparse).fit())


def test_compressed_itt_equals_rows(case: Case) -> None:
    """The ITT from compressed rows (a linear map of each cluster summary) equals the ITT refit on the rows."""
    exp = case.experiment()
    a, b = exp.fit().itt(), exp.compress().fit().itt()
    np.testing.assert_allclose([r[3] for r in a.rows], [r[3] for r in b.rows], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose([r[4] for r in a.rows], [r[4] for r in b.rows], rtol=1e-8)


@pytest.fixture(scope="module")
def exp() -> ume.Experiment:
    df = simulate_accounts(3000, seed=11)
    ann = ume.annotate(df, cluster="account", instrument="Z", treatment="T", cohort="day", covariates=COVARIATES)
    return ann.metrics(ume.mean("y"), ume.ratio("N", "D"))


def resolved(exp: ume.Experiment) -> ume.Design:
    """The design with x_ref and levels fixed, as a streaming pass needs."""
    from ume._data import resolve

    return resolve(exp.frame, exp.design)[1]


@pytest.mark.parametrize("seed", range(4))
def test_shards_merge_exactly(exp: ume.Experiment, seed: int) -> None:
    """Compressing random contiguous shards of clusters and adding them equals compressing all the rows at once."""
    rng = np.random.default_rng(seed)
    accounts = exp.frame["account"].unique().sort().to_numpy()
    cuts = np.sort(rng.choice(accounts[1:], size=3, replace=False))
    bounds = [accounts[0], *cuts, accounts[-1] + 1]
    design = resolved(exp)
    shards = [
        ume.Compressor(design).update(exp.frame.filter(pl.col("account").is_between(lo, hi, closed="left"))).finish()
        for lo, hi in itertools.pairwise(bounds)
    ]
    merged = shards[0] + shards[1] + shards[2] + shards[3]
    whole = exp.compress()
    assert merged.n == whole.n and merged.n_clusters == whole.n_clusters
    for h in whole:
        assert merged[h].G == whole[h].G
        np.testing.assert_allclose(merged[h].first_moments, whole[h].first_moments, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(merged[h].second_moments, whole[h].second_moments, rtol=1e-12, atol=1e-7)
    assert_same_fit(merged.fit(), exp.fit())


def test_batches_may_split_a_cluster(exp: ume.Experiment) -> None:
    """A streaming pass carries an open cluster across batch boundaries, so batches cut inside clusters give the
    same result.
    """
    c = ume.Compressor(resolved(exp))
    n = len(exp.frame)
    for lo in range(0, n, 777):  # batch boundaries fall inside clusters
        c.update(exp.frame[lo : lo + 777])
    assert_same_fit(c.finish().fit(), exp.fit())


def test_a_cluster_split_across_shards_is_rejected(exp: ume.Experiment) -> None:
    """Merging two shards that hold rows of the same cluster raises, because that cluster's summary would be split."""
    design = resolved(exp)
    multi = exp.frame.filter(
        pl.col("account") == exp.frame.group_by("account").len().filter(pl.col("len") > 1)["account"][0]
    )
    a = ume.Compressor(design).update(multi[:1]).finish()
    b = ume.Compressor(design).update(multi[1:]).finish()
    with pytest.raises(ValueError, match="clusters have rows in both shards"):
        a + b


def test_non_contiguous_rows_are_rejected_by_a_streaming_pass(exp: ume.Experiment) -> None:
    """A streaming pass refuses rows whose clusters are not grouped together."""
    shuffled = exp.frame.sample(fraction=1.0, shuffle=True, seed=1)
    with pytest.raises(ValueError, match="not contiguous"):
        ume.Compressor(resolved(exp)).update(shuffled)


def test_compress_groups_unsorted_rows(exp: ume.Experiment) -> None:
    """``Experiment.compress`` sorts unsorted rows by cluster first, with the same result."""
    shuffled = exp.frame.sample(fraction=1.0, shuffle=True, seed=1)
    assert_same_fit(ume.Experiment(shuffled, exp.design).compress().fit(), exp.fit())


def test_string_cluster_ids(exp: ume.Experiment) -> None:
    """String cluster ids (hashed to 64-bit keys) give the same result as integer ids."""
    renamed = exp.frame.with_columns(pl.format("acct-{}", pl.col("account")).alias("account"))
    assert_same_fit(ume.Experiment(renamed, exp.design).compress().fit(), exp.fit())


def test_streaming_needs_a_resolved_design(exp: ume.Experiment) -> None:
    """A streaming Compressor refuses a design whose reference points are not fixed before the pass."""
    unresolved = exp.design.covariates.replace_spec("x1", ume.Numeric())
    with pytest.raises(ValueError, match="declared before the pass"):
        ume.Compressor(ume.Design(**{**exp.design.__dict__, "covariates": unresolved}))


def test_compressed_rows_hold_the_blocks(exp: ume.Experiment) -> None:
    """A compressed row holds what the docs say: $S_{00}$ counts the stratum's rows, $G_h$ its clusters, and the
    layout has $q = do + d(d+1)$ entries.
    """
    cd = exp.compress()
    h = ume.Stratum(1, 4)
    row = cd[h]
    rows = exp.frame.filter((pl.col("Z") == 1) & (pl.col("day") == 4))
    s = row.blocks()["S"].to_dict()
    assert s["1"][0] == len(rows)  # S_00 counts the stratum's rows
    assert rows["account"].n_unique() == row.G
    assert len(row.entries) == cd.q == cd.d * cd.o + cd.d * (cd.d + 1)
    assert "8 strata" in repr(cd) and "Z" in repr(cd.table())


def test_size_does_not_grow_with_rows() -> None:
    """The compressed dataset has the same shape for 2,000 and for 8,000 accounts."""

    def size(n: int) -> tuple[int, ...]:
        df = simulate_accounts(n, seed=2)
        cd = ume.annotate(df, cluster="account", treatment="T", instrument="Z", cohort="day",
                          covariates=COVARIATES).metrics(ume.mean("y")).compress()  # fmt: skip
        return (len(cd), *cd.second.shape)

    assert size(2000) == size(8000)
