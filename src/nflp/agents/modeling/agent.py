"""Modeling Agent — entrenamiento, calibración, versionado y explicabilidad.

Flujo de un entrenamiento:

1. Cargar el dataset (con auditoría anti-leakage obligatoria).
2. Corte **temporal** train/validación (nunca aleatorio).
3. Opcionalmente, tuning con Optuna sobre validación cruzada **por temporada**.
4. Entrenar las familias solicitadas y quedarse con la que mejor valida.
5. Calibrar probabilidades (o estimar la distribución de residuos) sobre la
   validación, que es posterior en el tiempo al entrenamiento.
6. Aplicar el **gate de promoción**: si la brecha train/validación es excesiva, el
   modelo queda marcado como rechazado y no puede llegar a producción.
7. Registrar en MLflow y en la base, con toda la metadata para hacer rollback.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nflp.agents.backtesting.metrics import classification_metrics, regression_metrics
from nflp.agents.modeling.calibration import ProbabilityCalibrator, ResidualDistribution
from nflp.agents.modeling.datasets import (
    TARGET_SPECS,
    Dataset,
    load_game_dataset,
    season_folds,
    temporal_split_three_way,
)
from nflp.agents.modeling.models import (
    build_estimator,
    default_params,
    feature_importances,
    predict_proba,
    suggest_params,
)
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger
from nflp.registry import ModelRecord, ModelRegistry

log = get_logger(__name__)


@dataclass
class TrainingResult:
    """Resultado de entrenar un objetivo."""

    model_id: str
    target: str
    family: str
    version: int
    metric_name: str
    train_metric: float
    val_metric: float
    train_val_gap: float
    status: str
    blocked_reason: str | None
    metrics: dict[str, float]
    comparison: pd.DataFrame


class ModelingAgent:
    """Entrena, calibra y versiona los modelos de predicción."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.registry = ModelRegistry(self.settings, self.db)

    # ------------------------------------------------------------ evaluación

    def _score(self, dataset: Dataset, model: Any, kind: str) -> tuple[float, dict[str, float]]:
        """Métrica primaria y conjunto completo, según el tipo de tarea."""
        if kind == "classification":
            probability = predict_proba(model, dataset.X)
            metrics = classification_metrics(dataset.y, probability)
            return metrics.log_loss, metrics.as_dict()

        prediction = model.predict(dataset.X)
        metrics = regression_metrics(dataset.y, prediction)
        return metrics.mae, metrics.as_dict()

    # ---------------------------------------------------------------- tuning

    def _tune(
        self, dataset: Dataset, family: str, kind: str, n_trials: int
    ) -> dict[str, Any]:
        """Búsqueda de hiperparámetros con validación cruzada temporal.

        El criterio es la media de la métrica sobre folds **por temporada**, con
        ventana expansiva. Nunca se usa un k-fold aleatorio: mezclaría futuro y
        pasado y elegiría hiperparámetros que sobreajustan.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        folds = season_folds(dataset, n_splits=min(5, dataset.meta["season"].nunique() - 1))
        if not folds:
            log.warning("Sin folds suficientes para tuning; se usan los valores por defecto.")
            return default_params(family, kind, self.settings.modeling.random_seed)

        def objective(trial: Any) -> float:
            params = suggest_params(trial, family, kind, self.settings.modeling.random_seed)
            scores: list[float] = []
            for train_mask, val_mask in folds:
                train_fold = dataset.slice(train_mask)
                val_fold = dataset.slice(val_mask)
                model = build_estimator(family, kind, params)
                model.fit(train_fold.X, train_fold.y)
                score, _ = self._score(val_fold, model, kind)
                if not np.isnan(score):
                    scores.append(score)
            return float(np.mean(scores)) if scores else float("inf")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.settings.modeling.random_seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        log.info(
            "Optuna (%s): mejor métrica CV temporal = %.4f en %d ensayos.",
            family, study.best_value, n_trials,
        )
        best = suggest_params(study.best_trial, family, kind, self.settings.modeling.random_seed)
        return best

    # ------------------------------------------------------------ entrenar

    def train(
        self,
        target: str,
        *,
        families: list[str] | None = None,
        tune: bool = False,
        seasons: list[int] | None = None,
        val_seasons: int = 2,
        calib_seasons: int = 2,
        register: bool = True,
    ) -> TrainingResult:
        """Entrena un objetivo comparando familias y aplicando el gate."""
        if target not in TARGET_SPECS:
            raise ValueError(f"Objetivo desconocido: {target}")

        kind = TARGET_SPECS[target]["kind"]
        metric_name = TARGET_SPECS[target]["metric"]
        families = families or self.settings.modeling.families

        dataset = load_game_dataset(
            target,
            db=self.db,
            feature_version=self.settings.features.version,
            seasons=seasons,
            min_prior_games=self.settings.features.min_prior_games,
        )
        # Tres bloques consecutivos en el tiempo: el calibrador se ajusta sobre
        # `calib_set` y la métrica se mide sobre `val_set`, posterior a ambos.
        # Medir sobre el mismo conjunto en que se calibra infla el resultado.
        train_set, calib_set, val_set = temporal_split_three_way(
            dataset, calib_seasons=calib_seasons, val_seasons=val_seasons
        )

        log.info(
            "Entrenando '%s': %d filas entrenamiento · %d calibración · %d validación · %d features.",
            target, len(train_set), len(calib_set), len(val_set), len(dataset.feature_names),
        )

        # --- comparación de familias ---
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for family in families:
            params = (
                self._tune(train_set, family, kind, self.settings.modeling.optuna_trials)
                if tune
                else default_params(family, kind, self.settings.modeling.random_seed)
            )

            model = build_estimator(family, kind, params)
            model.fit(train_set.X, train_set.y)

            train_score, train_metrics = self._score(train_set, model, kind)
            val_score, val_metrics = self._score(val_set, model, kind)

            rows.append(
                {
                    "familia": family,
                    f"{metric_name}_train": round(train_score, 4),
                    f"{metric_name}_val": round(val_score, 4),
                    "brecha_%": round(100 * (val_score - train_score) / max(abs(train_score), 1e-9), 1),
                    **{f"val_{k}": round(v, 4) for k, v in val_metrics.items() if k != "n"},
                }
            )
            log.info(
                "  %-9s train=%.4f  val=%.4f  brecha=%+.1f%%",
                family, train_score, val_score,
                100 * (val_score - train_score) / max(abs(train_score), 1e-9),
            )

            if best is None or val_score < best["val_score"]:
                best = {
                    "family": family, "model": model, "params": params,
                    "train_score": train_score, "val_score": val_score,
                    "train_metrics": train_metrics, "val_metrics": val_metrics,
                }

        assert best is not None
        comparison = pd.DataFrame(rows)
        log.info("Familia ganadora: %s (%s val = %.4f).", best["family"], metric_name, best["val_score"])

        # --- calibración: se AJUSTA en `calib_set`, se MIDE en `val_set` ---
        calibrator: ProbabilityCalibrator | None = None
        residuals: ResidualDistribution | None = None

        if kind == "classification":
            calibrator = ProbabilityCalibrator(self.settings.modeling.calibration)
            calibrator.fit(
                predict_proba(best["model"], calib_set.X), calib_set.y.to_numpy()
            )

            raw_val_proba = predict_proba(best["model"], val_set.X)
            calibrated = calibrator.transform(raw_val_proba)
            calibrated_metrics = classification_metrics(val_set.y, calibrated)

            log.info(
                "Calibración (%s), medida fuera de muestra: ECE %.4f -> %.4f · log-loss %.4f -> %.4f",
                calibrator.method,
                best["val_metrics"]["ece"], calibrated_metrics.ece,
                best["val_metrics"]["log_loss"], calibrated_metrics.log_loss,
            )
            best["val_metrics"] = calibrated_metrics.as_dict()
            best["val_score"] = calibrated_metrics.log_loss
        else:
            # La dispersión de residuos también se estima en `calib_set`.
            residuals = ResidualDistribution().fit(
                calib_set.y.to_numpy(), best["model"].predict(calib_set.X)
            )

        # --- gate de promoción ---
        decision = self.registry.evaluate_promotion(
            best["train_score"], best["val_score"], lower_is_better=True
        )
        log.info(decision.describe())

        # --- registro ---
        version = self.registry.next_version(target)
        model_id = str(uuid.uuid4())

        record = ModelRecord(
            model_id=model_id,
            target=target,
            family=best["family"],
            version=version,
            feature_set_version=self.settings.features.version,
            feature_names=dataset.feature_names,
            hyperparameters=best["params"],
            dataset_hash=dataset.dataset_hash,
            train_seasons=f"{train_set.meta['season'].min()}-{train_set.meta['season'].max()}",
            n_train_rows=len(train_set),
            n_val_rows=len(val_set),
            metric_name=metric_name,
            train_metric=float(best["train_score"]),
            val_metric=float(best["val_score"]),
            train_val_gap=decision.train_val_gap,
            metrics={**{f"val_{k}": v for k, v in best["val_metrics"].items()}},
            status="shadow",
        )

        if register:
            artifact = {
                "model": best["model"],
                "calibrator": calibrator,
                "residuals": residuals,
                "feature_names": dataset.feature_names,
                "target": target,
                "kind": kind,
                "family": best["family"],
                "importances": feature_importances(best["model"], dataset.feature_names),
            }
            self.registry.register(record, artifact, decision=decision)

        return TrainingResult(
            model_id=model_id,
            target=target,
            family=best["family"],
            version=version,
            metric_name=metric_name,
            train_metric=float(best["train_score"]),
            val_metric=float(best["val_score"]),
            train_val_gap=decision.train_val_gap,
            status=record.status,
            blocked_reason=record.promotion_blocked_reason,
            metrics=best["val_metrics"],
            comparison=comparison,
        )

    def train_all(
        self, *, tune: bool = False, targets: list[str] | None = None
    ) -> list[TrainingResult]:
        """Entrena todos los objetivos configurados."""
        targets = targets or self.settings.modeling.targets
        return [self.train(target, tune=tune) for target in targets]
