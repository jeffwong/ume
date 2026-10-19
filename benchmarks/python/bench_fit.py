"""Benchmarks of the universal model fit on rows, through the Python API."""

import pytest

import ume
from ume.testing import simulate_accounts

COVARIATES = ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv"))


@pytest.fixture(scope="module")
def experiment() -> ume.Experiment:
    """The running example at 20,000 accounts (about 44,000 rows), built outside the timed region."""
    df = simulate_accounts(20_000, seed=2026)
    exp = ume.annotate(df, cluster="account", instrument="Z", treatment="T", cohort="day", covariates=COVARIATES)
    return exp.metrics(ume.mean("y"), ume.ratio("N", "D"))


def test_fit_time(benchmark, experiment):
    """Wall time of validation, design matrix and the Rust fit."""
    benchmark(experiment.fit)


@pytest.mark.limit_memory("60 MB")
def test_fit_memory(experiment):
    """Peak memory of one fit, including allocations in Rust."""
    experiment.fit()


def test_compress_time(benchmark, experiment):
    """Wall time of one compression pass (validation, design matrix, Rust pass)."""
    benchmark(experiment.compress)


def test_fit_compressed_time(benchmark, experiment):
    """Wall time of a fit from compressed data: independent of the number of rows."""
    cd = experiment.compress()
    benchmark(cd.fit)


@pytest.mark.limit_memory("40 MB")
def test_compress_memory(experiment):
    """Peak memory of one compression pass: cluster summaries are never stored."""
    experiment.compress()
