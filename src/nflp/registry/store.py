"""Registro de modelos con gate de promoción.

Arquitectura de dos capas, deliberada:

* **MLflow** es la fuente de verdad de artefactos y métricas de experimento.
* **La tabla ``model_registry``** espeja el estado del ciclo de vida. Existe para
  que el dashboard y el gate de promoción consulten sin depender de que el
  servidor de MLflow esté levantado — y para poder hacer rollback con una
  sentencia SQL.

**El gate de promoción** (sección 5 del prompt) implementa la regla dura: si la
brecha entre rendimiento en entrenamiento y validación supera el umbral, el
modelo **no se promueve**, por buenas que sean sus métricas absolutas. Una brecha
grande significa que memorizó; sus métricas de validación son suerte, no señal.

Ciclo de vida: ``rejected`` → ``shadow`` → ``production`` → ``archived``.
Ningún modelo pasa directamente a producción: primero corre en paralelo.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from nflp.core.config import Settings, get_settings, project_root
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelRecord:
    """Metadatos completos de una versión de modelo."""

    model_id: str
    target: str
    family: str
    version: int
    feature_set_version: str
    feature_names: list[str]
    hyperparameters: dict[str, Any]
    dataset_hash: str
    train_seasons: str
    n_train_rows: int
    n_val_rows: int
    metric_name: str
    train_metric: float
    val_metric: float
    test_metric: float | None = None
    train_val_gap: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "shadow"
    promotion_blocked_reason: str | None = None
    mlflow_run_id: str | None = None
    artifact_path: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None))

    def to_row(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target": self.target,
            "family": self.family,
            "version": self.version,
            "feature_set_version": self.feature_set_version,
            "mlflow_run_id": self.mlflow_run_id,
            "artifact_path": self.artifact_path,
            "feature_names": json.dumps(self.feature_names, ensure_ascii=False),
            "hyperparameters": json.dumps(_jsonable(self.hyperparameters), ensure_ascii=False),
            "dataset_hash": self.dataset_hash,
            "train_seasons": self.train_seasons,
            "n_train_rows": self.n_train_rows,
            "n_val_rows": self.n_val_rows,
            "metric_name": self.metric_name,
            "train_metric": self.train_metric,
            "val_metric": self.val_metric,
            "test_metric": self.test_metric,
            "train_val_gap": self.train_val_gap,
            "metrics_json": json.dumps(_jsonable(self.metrics), ensure_ascii=False),
            "status": self.status,
            "promotion_blocked_reason": self.promotion_blocked_reason,
            "created_at": self.created_at,
            "promoted_at": None,
            "archived_at": None,
        }


@dataclass
class PromotionDecision:
    """Resultado de evaluar si un modelo puede promoverse."""

    allowed: bool
    reason: str
    train_val_gap: float
    threshold: float

    def describe(self) -> str:
        verdict = "PERMITIDA" if self.allowed else "BLOQUEADA"
        return f"Promoción {verdict}: {self.reason}"


class ModelRegistry:
    """Versionado, promoción y rollback de modelos."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.artifacts_dir = self.settings.paths.artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- versiones

    def next_version(self, target: str) -> int:
        current = self.db.scalar(
            "SELECT max(version) FROM model_registry WHERE target = ?", [target]
        )
        return int(current or 0) + 1

    # ---------------------------------------------------------------- gate

    def evaluate_promotion(
        self,
        train_metric: float,
        val_metric: float,
        *,
        lower_is_better: bool = True,
    ) -> PromotionDecision:
        """Aplica el gate de over/underfitting.

        La brecha se mide en términos **relativos**: cuánto empeora la métrica al
        pasar de entrenamiento a validación. Un modelo con log-loss 0,50 en train
        y 0,65 en validación tiene una brecha del 30 % y está memorizando.
        """
        threshold = self.settings.monitoring.max_train_val_gap

        if train_metric is None or val_metric is None or pd.isna(train_metric) or pd.isna(val_metric):
            return PromotionDecision(False, "métricas incompletas", float("nan"), threshold)

        denominator = abs(train_metric) if abs(train_metric) > 1e-9 else 1e-9
        gap = (
            (val_metric - train_metric) / denominator
            if lower_is_better
            else (train_metric - val_metric) / denominator
        )

        if gap > threshold:
            return PromotionDecision(
                allowed=False,
                reason=(
                    f"brecha train/validación {gap:.1%} supera el umbral {threshold:.1%} "
                    f"(train={train_metric:.4f}, val={val_metric:.4f}) — indicio de sobreajuste"
                ),
                train_val_gap=gap,
                threshold=threshold,
            )

        # Una brecha muy negativa (validación mucho mejor que entrenamiento)
        # también es sospechosa: suele indicar leakage o un corte mal hecho.
        if gap < -threshold:
            return PromotionDecision(
                allowed=False,
                reason=(
                    f"validación {abs(gap):.1%} mejor que entrenamiento — "
                    "sospecha de leakage o corte temporal incorrecto"
                ),
                train_val_gap=gap,
                threshold=threshold,
            )

        return PromotionDecision(
            allowed=True,
            reason=f"brecha train/validación {gap:.1%} dentro del umbral {threshold:.1%}",
            train_val_gap=gap,
            threshold=threshold,
        )

    # ------------------------------------------------------------ persistencia

    def register(
        self,
        record: ModelRecord,
        artifact: dict[str, Any],
        *,
        decision: PromotionDecision | None = None,
    ) -> ModelRecord:
        """Guarda el artefacto en disco, lo registra en MLflow y en la base."""
        artifact_path = self.artifacts_dir / f"{record.target}_v{record.version}_{record.model_id[:8]}.joblib"
        joblib.dump(artifact, artifact_path)
        record.artifact_path = str(artifact_path)

        record.mlflow_run_id = self._log_to_mlflow(record, artifact_path)

        if decision is not None:
            record.train_val_gap = decision.train_val_gap
            if not decision.allowed:
                record.status = "rejected"
                record.promotion_blocked_reason = decision.reason

        self.db.write_df(pd.DataFrame([record.to_row()]), "model_registry", mode="append")

        self._log_event(
            action="block" if record.status == "rejected" else "shadow_start",
            target=record.target,
            model_id=record.model_id,
            reason=record.promotion_blocked_reason or "modelo registrado en shadow",
        )

        log.info(
            "Modelo registrado: %s v%d (%s) estado=%s %s=%.4f",
            record.target, record.version, record.family, record.status,
            record.metric_name, record.val_metric,
        )
        return record

    def _log_to_mlflow(self, record: ModelRecord, artifact_path: Path) -> str | None:
        """Registra el experimento en MLflow (backend de fichero local)."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
            mlflow.set_experiment(self.settings.mlflow.experiment_name)

            with mlflow.start_run(run_name=f"{record.target}_v{record.version}") as run:
                mlflow.log_params(
                    {
                        "target": record.target,
                        "family": record.family,
                        "version": record.version,
                        "feature_set_version": record.feature_set_version,
                        "dataset_hash": record.dataset_hash,
                        "n_features": len(record.feature_names),
                        "n_train_rows": record.n_train_rows,
                        **{f"hp_{k}": v for k, v in list(record.hyperparameters.items())[:30]},
                    }
                )
                mlflow.log_metrics(
                    {
                        "train_metric": float(record.train_metric),
                        "val_metric": float(record.val_metric),
                        "train_val_gap": float(record.train_val_gap),
                        **{
                            k: float(v)
                            for k, v in record.metrics.items()
                            if isinstance(v, (int, float)) and not pd.isna(v)
                        },
                    }
                )
                mlflow.log_artifact(str(artifact_path))
                return run.info.run_id
        except Exception as exc:  # noqa: BLE001 - MLflow nunca debe tumbar el entrenamiento
            log.warning("No se pudo registrar en MLflow: %s", exc)
            return None

    # -------------------------------------------------------------- promoción

    def promote(self, model_id: str, *, reason: str = "promoción manual") -> bool:
        """Promueve un modelo a producción y archiva el anterior."""
        info = self.db.query(
            "SELECT target, status, version FROM model_registry WHERE model_id = ?",
            [model_id],
        )
        if info.empty:
            log.error("Modelo no encontrado: %s", model_id)
            return False

        target = info.iloc[0]["target"]
        status = info.iloc[0]["status"]

        if status == "rejected":
            log.error(
                "El modelo %s fue rechazado por el gate de promoción; no puede promoverse.",
                model_id,
            )
            return False

        previous = self.current_production(target)
        now = datetime.now(UTC).replace(tzinfo=None)

        if previous is not None:
            self.db.execute(
                "UPDATE model_registry SET status = 'archived', archived_at = ? "
                "WHERE model_id = ?",
                [now, previous],
            )

        self.db.execute(
            "UPDATE model_registry SET status = 'production', promoted_at = ? WHERE model_id = ?",
            [now, model_id],
        )

        self._log_event(
            action="promote", target=target, model_id=model_id,
            previous_model_id=previous, reason=reason,
        )
        log.info("Modelo %s promovido a producción para '%s'.", model_id, target)
        return True

    def rollback(self, target: str, *, reason: str = "rollback manual") -> str | None:
        """Vuelve a la versión anterior de producción."""
        candidates = self.db.query(
            """
            SELECT model_id, version FROM model_registry
            WHERE target = ? AND status = 'archived'
            ORDER BY version DESC LIMIT 1
            """,
            [target],
        )
        if candidates.empty:
            log.error("No hay versión archivada a la que volver para '%s'.", target)
            return None

        previous_id = candidates.iloc[0]["model_id"]
        current = self.current_production(target)
        now = datetime.now(UTC).replace(tzinfo=None)

        if current:
            self.db.execute(
                "UPDATE model_registry SET status = 'archived', archived_at = ? WHERE model_id = ?",
                [now, current],
            )
        self.db.execute(
            "UPDATE model_registry SET status = 'production', promoted_at = ? WHERE model_id = ?",
            [now, previous_id],
        )

        self._log_event(
            action="rollback", target=target, model_id=previous_id,
            previous_model_id=current, reason=reason,
        )
        log.warning("Rollback en '%s': vuelta al modelo %s.", target, previous_id)
        return str(previous_id)

    # ---------------------------------------------------------------- consulta

    def current_production(self, target: str) -> str | None:
        result = self.db.scalar(
            "SELECT model_id FROM model_registry "
            "WHERE target = ? AND status = 'production' "
            "ORDER BY version DESC LIMIT 1",
            [target],
        )
        return str(result) if result else None

    def current_shadow(self, target: str) -> str | None:
        result = self.db.scalar(
            "SELECT model_id FROM model_registry "
            "WHERE target = ? AND status = 'shadow' "
            "ORDER BY version DESC LIMIT 1",
            [target],
        )
        return str(result) if result else None

    def get(self, model_id: str) -> pd.DataFrame:
        return self.db.query("SELECT * FROM model_registry WHERE model_id = ?", [model_id])

    def list_models(self, target: str | None = None) -> pd.DataFrame:
        clause = "WHERE target = ?" if target else ""
        return self.db.query(
            f"""
            SELECT model_id, target, family, version, status, metric_name,
                   train_metric, val_metric, train_val_gap,
                   promotion_blocked_reason, created_at, promoted_at
            FROM model_registry {clause}
            ORDER BY target, version DESC
            """,
            [target] if target else None,
        )

    def load(self, model_id: str) -> dict[str, Any]:
        """Carga el artefacto serializado de un modelo."""
        path = self.db.scalar(
            "SELECT artifact_path FROM model_registry WHERE model_id = ?", [model_id]
        )
        if not path:
            raise FileNotFoundError(f"Sin artefacto registrado para {model_id}")
        return load_model_artifact(Path(str(path)), settings=self.settings)

    def load_production(self, target: str) -> tuple[str, dict[str, Any]] | None:
        model_id = self.current_production(target)
        if model_id is None:
            return None
        return model_id, self.load(model_id)

    # ------------------------------------------------------------- bitácora

    def log_event(
        self,
        *,
        action: str,
        target: str | None,
        model_id: str | None,
        previous_model_id: str | None = None,
        reason: str = "",
    ) -> None:
        """Escribe una fila en ``retraining_log`` (API pública para otros agentes)."""
        self._log_event(
            action=action, target=target, model_id=model_id,
            previous_model_id=previous_model_id, reason=reason,
        )

    def _log_event(
        self,
        *,
        action: str,
        target: str | None,
        model_id: str | None,
        previous_model_id: str | None = None,
        reason: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = pd.DataFrame(
            [
                {
                    "event_id": str(uuid.uuid4()),
                    "occurred_at": datetime.now(UTC).replace(tzinfo=None),
                    "target": target,
                    "action": action,
                    "model_id": model_id,
                    "previous_model_id": previous_model_id,
                    "reason": reason,
                    "details_json": json.dumps(_jsonable(details or {}), ensure_ascii=False),
                }
            ]
        )
        self.db.write_df(entry, "retraining_log", mode="append")


def allowed_artifact_roots(settings: Settings | None = None) -> list[Path]:
    """Directorios desde los que se permite deserializar artefactos.

    `joblib.load` ejecuta código arbitrario si el fichero es malicioso, así que
    la ruta guardada en ``model_registry.artifact_path`` nunca se carga a
    ciegas: tiene que caer dentro del directorio de artefactos configurado o
    del store de artefactos local de MLflow (``mlruns``).
    """
    settings = settings or get_settings()
    roots = [settings.paths.artifacts_dir, project_root() / "mlruns"]
    return [Path(r).resolve() for r in roots]


def load_model_artifact(path: Path, *, settings: Settings | None = None) -> dict[str, Any]:
    """Carga un artefacto joblib desde disco, solo si está en un directorio permitido."""
    resolved = Path(path).resolve()
    roots = allowed_artifact_roots(settings)
    if not any(resolved.is_relative_to(root) for root in roots):
        raise PermissionError(
            f"Artefacto fuera de los directorios permitidos ({', '.join(map(str, roots))}): "
            f"{resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Artefacto no encontrado: {resolved}")
    return joblib.load(resolved)


def _jsonable(value: Any) -> Any:
    """Convierte tipos de numpy/pandas a algo serializable en JSON."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
