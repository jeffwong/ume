"""Testing tools: the posts' simulator and a numpy reference implementation.

These are public so that users can check ume against known truth and against
textbook formulas, the way its own tests do.
"""

from ume.testing.reference import ReferenceFit, complier_mean, draw_rows, fit_rows, inverse_propensity_weights
from ume.testing.simulate import PROPENSITY, simulate_accounts, simulate_columns, truth

__all__ = [
    "PROPENSITY",
    "ReferenceFit",
    "complier_mean",
    "draw_rows",
    "fit_rows",
    "inverse_propensity_weights",
    "simulate_accounts",
    "simulate_columns",
    "truth",
]
