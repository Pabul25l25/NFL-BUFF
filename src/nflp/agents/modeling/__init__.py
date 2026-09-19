"""Modeling Agent — entrenamiento, calibración, versionado y explicabilidad."""

from nflp.agents.modeling.agent import ModelingAgent, TrainingResult
from nflp.agents.modeling.calibration import (
    ProbabilityCalibrator,
    ResidualDistribution,
    confidence_level,
)
from nflp.agents.modeling.datasets import (
    TARGET_SPECS,
    Dataset,
    load_game_dataset,
    season_folds,
    temporal_split,
)
from nflp.agents.modeling.models import build_estimator, default_params

__all__ = [
    "TARGET_SPECS",
    "Dataset",
    "ModelingAgent",
    "ProbabilityCalibrator",
    "ResidualDistribution",
    "TrainingResult",
    "build_estimator",
    "confidence_level",
    "default_params",
    "load_game_dataset",
    "season_folds",
    "temporal_split",
]
