"""Milestone 5: threads, lazy frames, streams of batches, and storage."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ume
from ume.testing import simulate_accounts

COV = ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv", levels=["tv", "mobile", "web"]))
METRICS = (ume.mean("y", name="hours"), ume.ratio("N", "D", name="rpp"))


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return simulate_accounts(4000, seed=31)


@pytest.fixture(scope="module")
def design() -> ume.Design:
    return ume.design(cluster="account", instrument="Z", treatment="T", cohort="day", covariates=COV, metrics=METRICS)


@pytest.fixture(scope="module")
def parquet(df: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("rows") / "rows.parquet"
    df.sample(fraction=1.0, shuffle=True, seed=1).write_parquet(path)  # not grouped by account
    return path


def assert_same(a: ume.UniversalFit, b: ume.UniversalFit) -> None:
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.cov, b.cov, rtol=1e-9, atol=1e-15)


def test_design_without_data_equals_the_annotated_design(df: pl.DataFrame, design: ume.Design) -> None:
    exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T", cohort="day", covariates=COV)
    assert exp.metrics(*METRICS).design == design


# ---- threads ------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [2, 3, 4])
def test_results_do_not_depend_on_threads(df: pl.DataFrame, design: ume.Design, threads: int) -> None:
    exp = design.bind(df)
    try:
        ume.set_threads(1)
        cd1, b1, m1, e1 = exp.compress(), exp.bootstrap(30, seed=2), exp.bootstrap(5, "multinomial", 2), exp.effects()
        ume.set_threads(threads)
        cd, b, m, e = exp.compress(), exp.bootstrap(30, seed=2), exp.bootstrap(5, "multinomial", 2), exp.effects()
    finally:
        ume.set_threads(None)
    np.testing.assert_allclose(cd.second, cd1.second, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(b.beta, b1.beta, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(m.beta, m1.beta)  # the multinomial bootstrap always runs on one worker
    np.testing.assert_allclose([r[4] for r in e.rows], [r[4] for r in e1.rows], rtol=1e-10)


def test_thread_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UME_THREADS", "3")
    assert ume.get_threads() == 3
    with pytest.raises(ValueError, match="at least 1"):
        ume.set_threads(0)


# ---- lazy frames --------------------------------------------------------------------------------


def test_a_lazy_frame_streams_to_the_same_results(df: pl.DataFrame, design: ume.Design, parquet: Path) -> None:
    eager, lazy = design.bind(df), design.bind(pl.scan_parquet(parquet))
    assert lazy.lazy and not eager.lazy
    assert_same(lazy.compress(batch_size=1500).fit(), eager.compress().fit())
    np.testing.assert_allclose(
        lazy.bootstrap(20, seed=4).beta, eager.bootstrap(20, seed=4).beta, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        [r[4] for r in lazy.effects(se="linearized", batch_size=1500).rows],
        [r[4] for r in eager.effects(se="linearized").rows],
        rtol=1e-8,
    )
    assert_same(lazy.fit(), eager.fit())  # fit on rows collects the lazy frame
    assert isinstance(lazy.filter(pl.col("day") > 1), ume.Experiment)
    assert "lazy" in repr(lazy)


def test_lazy_frames_resolve_reference_points(df: pl.DataFrame, parquet: Path) -> None:
    cov = ume.covariates("x1", device=ume.categorical(reference="tv"))
    exp = ume.annotate(pl.scan_parquet(parquet), cluster="account", treatment="T", instrument="Z", covariates=cov)
    cd = exp.metrics(ume.mean("y")).compress()
    x_ref = dict(cd.design.covariates.specs)["x1"].x_ref
    assert x_ref == pytest.approx(float(df["x1"].mean()), rel=1e-12)
    assert dict(cd.design.covariates.specs)["device"].levels == ("mobile", "tv", "web")


def test_presorted_skips_the_sort_and_is_checked(df: pl.DataFrame, design: ume.Design, parquet: Path) -> None:
    sorted_lf = pl.scan_parquet(parquet).sort("account")
    assert_same(design.bind(sorted_lf).compress(presorted=True).fit(), design.bind(df).compress().fit())
    with pytest.raises(ValueError, match="not contiguous"):
        design.bind(pl.scan_parquet(parquet)).compress(presorted=True, batch_size=500)


# ---- streams of batches -------------------------------------------------------------------------


def batches(df: pl.DataFrame, size: int = 1234) -> Iterator[pl.DataFrame]:
    """The rows in batches, grouped by account (batch boundaries split clusters)."""
    ordered = df.sort("account")
    for lo in range(0, len(ordered), size):
        yield ordered[lo : lo + size]


def test_a_stream_of_batches(df: pl.DataFrame, design: ume.Design) -> None:
    assert_same(ume.compress(batches(df), design).fit(), design.bind(df).compress().fit())
    draws = ume.bootstrap(batches(df), design, B=10, seed=1)
    np.testing.assert_allclose(draws.beta, design.bind(df).bootstrap(10, seed=1).beta, rtol=1e-10, atol=1e-12)
    two = ume.effects(lambda: batches(df), design, se="linearized")  # a function: read twice
    np.testing.assert_allclose(
        [r[4] for r in two.rows], [r[4] for r in design.bind(df).effects(se="linearized").rows], rtol=1e-8
    )


def test_streams_need_a_declared_design_and_reiterability(df: pl.DataFrame, design: ume.Design) -> None:
    undeclared = ume.design(cluster="account", instrument="Z", treatment="T", covariates=["x1"], metrics=METRICS)
    with pytest.raises(ValueError, match="declared before the pass"):
        ume.compress(batches(df), undeclared)
    with pytest.raises(ValueError, match="read twice"):
        ume.effects(batches(df), design)
    with pytest.raises(ValueError, match="number of clusters"):
        ume.bootstrap(batches(df), design, B=5, weights="multinomial")


# ---- storage ------------------------------------------------------------------------------------


def test_save_and_load(df: pl.DataFrame, design: ume.Design, tmp_path: Path) -> None:
    cd = design.bind(df).compress()
    cd.save(tmp_path / "cd.npz")
    back = ume.load(tmp_path / "cd.npz")
    assert back.design == cd.design and back.strata == cd.strata and back.n == cd.n
    assert_same(back.fit(), cd.fit())
    np.testing.assert_array_equal(back.cluster_keys, cd.cluster_keys)


def test_saved_shards_merge(df: pl.DataFrame, design: ume.Design, tmp_path: Path) -> None:
    for k in range(3):
        ume.compress(df.filter(pl.col("account") % 3 == k), design).save(tmp_path / f"shard{k}.npz")
    shards = [ume.load(tmp_path / f"shard{k}.npz") for k in range(3)]
    assert_same((shards[0] + shards[1] + shards[2]).fit(), design.bind(df).compress().fit())
    with pytest.raises(ValueError, match="both shards"):
        shards[0] + shards[0]


def test_load_rejects_other_files(tmp_path: Path) -> None:
    np.savez(tmp_path / "other.npz", meta=np.array('{"format": "something else"}'))
    with pytest.raises(ValueError, match="not a ume compressed dataset"):
        ume.load(tmp_path / "other.npz")


# ---- exact bootstrap over shards ------------------------------------------------------------------


@pytest.mark.parametrize("weights", ["bayesian", "poisson"])
def test_bootstrap_sums_of_shards_merge(df: pl.DataFrame, design: ume.Design, weights: str) -> None:
    shards = [ume.bootstrap_sums(df.filter(pl.col("account") % 3 == k), design, B=25, weights=weights, seed=8)
              for k in range(3)]  # fmt: skip
    merged = (shards[0] + shards[1] + shards[2]).draws()
    whole = design.bind(df).bootstrap(25, weights=weights, seed=8)
    np.testing.assert_allclose(merged.beta, whole.beta, rtol=1e-10, atol=1e-12)
    assert "DrawSums" in repr(shards[0])


def test_bootstrap_sums_refuse_bad_merges(df: pl.DataFrame, design: ume.Design) -> None:
    a = ume.bootstrap_sums(df.filter(pl.col("account") % 2 == 0), design, B=5, seed=1)
    b = ume.bootstrap_sums(df.filter(pl.col("account") % 2 == 1), design, B=5, seed=2)
    with pytest.raises(ValueError, match="seeds"):
        a + b
    with pytest.raises(ValueError, match="both shards"):
        a + a
    m = ume.bootstrap_sums(df, design, B=5, weights="multinomial", seed=1)
    with pytest.raises(ValueError, match="multinomial"):
        m + m
