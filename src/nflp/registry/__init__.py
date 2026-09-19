"""Registro de modelos: MLflow para artefactos, DuckDB para estado consultable."""

from nflp.registry.store import (
    ModelRecord,
    ModelRegistry,
    PromotionDecision,
    load_model_artifact,
)

__all__ = ["ModelRecord", "ModelRegistry", "PromotionDecision", "load_model_artifact"]
