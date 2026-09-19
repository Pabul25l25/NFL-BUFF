"""Feature Engineering Agent — construcción de features con contrato as-of."""

from nflp.agents.features.agent import FeatureAgent
from nflp.agents.features.contracts import (
    DENYLIST,
    LeakageError,
    LeakageReport,
    assert_no_leakage,
    audit_features,
    feature_columns,
    shifted_ewm,
    shifted_rolling,
)
from nflp.agents.features.elo import EloConfig, compute_elo, elo_win_probability

__all__ = [
    "DENYLIST",
    "EloConfig",
    "FeatureAgent",
    "LeakageError",
    "LeakageReport",
    "assert_no_leakage",
    "audit_features",
    "compute_elo",
    "elo_win_probability",
    "feature_columns",
    "shifted_ewm",
    "shifted_rolling",
]
