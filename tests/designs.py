"""The 16 designs of the CSS post: encouragement × clusters × ramp × metric."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import ume
from ume.testing import simulate_accounts

COVARIATES = ume.covariates(x1=2.0, x2=2.0, kids=0, device=ume.categorical(reference="tv"))


@dataclass(frozen=True)
class Case:
    """One design: which features are on."""

    encouraged: bool
    clustered: bool
    ramp: bool
    metric: str

    @property
    def id(self) -> str:
        on = [n for n, v in (("enc", self.encouraged), ("clu", self.clustered), ("ramp", self.ramp)) if v]
        return "-".join(on or ["ab"]) + f"-{self.metric}"

    def experiment(self, n_accounts: int = 3000, seed: int = 7) -> ume.Experiment:
        """Simulate the design and annotate it."""
        df = simulate_accounts(
            n_accounts,
            propensity=(0.05, 0.10, 0.25, 0.50) if self.ramp else (0.3, 0.3, 0.3, 0.3),
            seed=seed,
            full_compliance=not self.encouraged,
            one_profile=not self.clustered,
        )
        exp = ume.annotate(
            df,
            cluster="account",
            instrument="Z",
            treatment="T" if self.encouraged else None,
            cohort="day" if self.ramp else None,
            covariates=COVARIATES,
        )
        metric = ume.mean("y", name="hours") if self.metric == "mean" else ume.ratio("N", "D", name="rpp")
        return exp.metrics(metric)


CASES = [Case(*v) for v in itertools.product([False, True], [False, True], [False, True], ["mean", "ratio"])]
