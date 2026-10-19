"""The annotation API: frames, delegation, validation, printing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import ume
from ume import _core
from ume.testing import simulate_accounts

COV = ume.covariates(x1=2.0, kids=0, device=ume.categorical(reference="tv"))


def annotate(df: object, **kwargs: object) -> ume.Experiment:
    args = dict(cluster="account", instrument="Z", treatment="T", cohort="day", covariates=COV) | kwargs
    return ume.annotate(df, **args).metrics(ume.mean("y", name="hours"))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return simulate_accounts(1500, seed=3)


def test_version_matches_extension() -> None:
    assert ume.__version__ == _core.__version__


def test_rust_docstrings_reach_python() -> None:
    assert _core.fit_rows_dense.__doc__ is not None
    assert "Parameters" in _core.fit_rows_dense.__doc__


def test_pandas_and_polars_give_the_same_fit(df: pl.DataFrame) -> None:
    fit_pl = annotate(df).fit()
    fit_pd = annotate(simulate_accounts(1500, seed=3, backend="pandas")).fit()
    np.testing.assert_allclose(fit_pl.beta, fit_pd.beta, rtol=1e-12)
    np.testing.assert_allclose(fit_pl.cov, fit_pd.cov, rtol=1e-12)


def test_frame_operations_keep_the_annotation(df: pl.DataFrame) -> None:
    exp = annotate(df)
    assert isinstance(exp.filter(pl.col("day") > 1), ume.Experiment)
    assert isinstance(exp.head(10), ume.Experiment)
    assert isinstance(exp.group_by("day").agg(pl.col("y").mean()), pl.DataFrame)
    assert isinstance(exp.select("y"), pl.DataFrame)
    assert exp.columns == df.columns
    assert len(exp) == len(df)
    exp_pd = annotate(simulate_accounts(1500, seed=3, backend="pandas"))
    assert isinstance(exp_pd[exp_pd.frame["day"] > 1], ume.Experiment)
    assert isinstance(exp_pd.groupby("day")["y"].mean(), pd.Series)


def test_randomized_experiment_needs_only_the_treatment(df: pl.DataFrame) -> None:
    ab = simulate_accounts(1500, seed=3, full_compliance=True)
    exp = ume.annotate(ab, cluster="account", treatment="T", covariates=["x1"]).metrics(ume.mean("y"))
    assert exp.design.instrument == exp.design.treatment == "T"
    fit = exp.fit()
    assert not fit.features.encouraged and fit.features.estimand == "ATE"
    assert fit.first_stage() is None
    assert fit.design.covariates.specs[0][1].x_ref == pytest.approx(float(ab["x1"].mean()))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.drop("x1"), "columns not found"),
        (
            lambda d: d.with_columns(Z=pl.when(pl.col("profile") == 2).then(1 - pl.col("Z")).otherwise(pl.col("Z"))),
            "constant within each cluster",
        ),
        (lambda d: d.with_columns(T=pl.col("T") * 2), "must be 0 or 1"),
        (
            lambda d: d.with_columns(y=pl.when(pl.col("profile") == 3).then(None).otherwise(pl.col("y"))),
            "missing values",
        ),
    ],
    ids=["missing-column", "z-varies-in-cluster", "t-not-binary", "nulls"],
)
def test_invalid_data_is_rejected(df: pl.DataFrame, change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        annotate(change(df)).fit()


def test_metrics_are_required(df: pl.DataFrame) -> None:
    exp = ume.annotate(df, cluster="account", treatment="T")
    with pytest.raises(ValueError, match="declare at least one metric"):
        exp.fit()


def test_outcome_columns_come_from_the_metrics(df: pl.DataFrame) -> None:
    exp = annotate(df).metrics(ume.mean("y"), ume.ratio("N", "D"), ume.ratio("y", "D", name="y_per_play"))
    assert exp.design.layout.labels == ["y", "N", "D", "T·y", "T·N", "T·D"]


def test_every_result_prints(df: pl.DataFrame) -> None:
    fit = annotate(df).fit()
    for obj, text in [
        (annotate(df), "ume.Experiment"),
        (fit, "Universal model: weighted 2SLS"),
        (fit.effects(), "compliance rate"),
        (fit.compliers(), "compliance rate"),
        (fit.itt(), "ITT"),
        (fit.coef("y"), "T:x1"),
    ]:
        assert text in repr(obj)
        assert "<" in obj._repr_html_() if hasattr(obj, "_repr_html_") else True


def test_many_level_categorical_uses_a_sparse_design(df: pl.DataFrame) -> None:
    rng = np.random.default_rng(0)
    base = simulate_accounts(8000, propensity=(0.5, 0.5, 0.5, 0.5), seed=4)
    wide = base.with_columns(region=pl.Series(rng.integers(0, 15, len(base)).astype(str)))
    cov = ume.covariates(x1=2.0, region=ume.categorical(reference="0"))
    exp = annotate(wide, covariates=cov)
    from ume._data import model_data

    assert model_data(exp.frame, exp.design).sparse
    np.testing.assert_allclose(exp.fit(sparse=True).beta, exp.fit(sparse=False).beta, rtol=1e-9, atol=1e-10)
