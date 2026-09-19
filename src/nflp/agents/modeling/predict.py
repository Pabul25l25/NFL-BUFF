"""Generación de predicciones para partidos próximos, con justificación SHAP.

Cada predicción se persiste con **trazabilidad completa** (sección 7 del prompt):
qué modelo la generó, en qué versión, con qué corte as-of y cuándo. Eso permite
que el dashboard muestre el historial y que una predicción pasada pueda
auditarse aunque el modelo ya se haya reemplazado.

Los modelos en *shadow* también predicen, marcados con ``is_shadow = TRUE``. Así
se puede comparar su rendimiento real contra el de producción antes de
promoverlos, sin afectar a lo que ve el usuario.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nflp.agents.modeling.calibration import confidence_level, margin_to_win_probability
from nflp.agents.modeling.datasets import load_game_dataset
from nflp.agents.modeling.explain import Explainer, describe_game_prediction
from nflp.agents.modeling.models import predict_proba
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger
from nflp.registry import ModelRegistry

log = get_logger(__name__)


@dataclass
class PredictionBatch:
    """Conjunto de predicciones generadas en una corrida."""

    frame: pd.DataFrame
    target: str
    model_id: str
    n_predictions: int


class Predictor:
    """Genera y persiste predicciones con explicaciones."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.registry = ModelRegistry(self.settings, self.db)

    # ------------------------------------------------------------------ util

    def _select_model(self, target: str, *, use_shadow: bool) -> tuple[str, dict] | None:
        """Elige el modelo a usar: producción por defecto, shadow si se pide.

        Si no hay ninguno en producción se recurre al shadow más reciente: es
        preferible predecir con un modelo validado pero aún no promovido que no
        predecir nada, y la columna ``is_shadow`` deja constancia.
        """
        model_id = (
            self.registry.current_shadow(target)
            if use_shadow
            else self.registry.current_production(target) or self.registry.current_shadow(target)
        )
        if model_id is None:
            log.warning("Sin modelo disponible para '%s'.", target)
            return None
        return model_id, self.registry.load(model_id)

    # ------------------------------------------------------------- predecir

    def predict_games(
        self,
        target: str,
        *,
        season: int | None = None,
        week: int | None = None,
        use_shadow: bool = False,
        persist: bool = True,
    ) -> PredictionBatch | None:
        """Predice los partidos indicados (por defecto, los que aún no se han jugado)."""
        selection = self._select_model(target, use_shadow=use_shadow)
        if selection is None:
            return None
        model_id, artifact = selection

        dataset = load_game_dataset(
            target,
            db=self.db,
            feature_version=self.settings.features.version,
            require_result=False,           # incluye partidos futuros
            min_prior_games=0,
        )

        mask = pd.Series(True, index=dataset.meta.index)
        if season is not None:
            mask &= dataset.meta["season"] == season
        if week is not None:
            mask &= dataset.meta["week"] == week
        if season is None and week is None:
            # Por defecto: partidos futuros.
            now = pd.Timestamp.utcnow().tz_localize(None)
            mask &= dataset.meta["kickoff_utc"] >= now

        subset = dataset.slice(mask)
        if len(subset) == 0:
            log.warning("No hay partidos que predecir con ese filtro.")
            return None

        # Alineación de columnas: el modelo espera exactamente sus features.
        feature_names = artifact["feature_names"]
        X = subset.X.reindex(columns=feature_names)
        model = artifact["model"]
        kind = artifact["kind"]

        frame = subset.meta.copy().reset_index(drop=True)
        X = X.reset_index(drop=True)

        if kind == "classification":
            raw = predict_proba(model, X)
            calibrator = artifact.get("calibrator")
            probability = calibrator.transform(raw) if calibrator else raw
            frame["probability"] = probability
            frame["point_estimate"] = probability
            frame["pred_std"] = np.nan
            frame["lower_80"] = np.nan
            frame["upper_80"] = np.nan
        else:
            prediction = model.predict(X)
            frame["point_estimate"] = prediction
            residuals = artifact.get("residuals")
            if residuals is not None:
                lower, upper = residuals.interval(prediction)
                frame["pred_std"] = residuals.sigma
                frame["lower_80"], frame["upper_80"] = lower, upper
                # Para el spread, el margen se traduce a probabilidad de victoria.
                frame["probability"] = (
                    margin_to_win_probability(prediction, residuals.sigma)
                    if target == "spread"
                    else np.nan
                )
            else:
                frame["pred_std"] = np.nan
                frame["lower_80"] = np.nan
                frame["upper_80"] = np.nan
                frame["probability"] = np.nan

        frame["confidence_level"] = (
            confidence_level(frame["probability"])
            if frame["probability"].notna().any()
            else "media"
        )

        # --- explicaciones SHAP ---
        explainer = Explainer(model, feature_names)
        matrix, base_value = explainer.shap_matrix(X)

        explanations: list[str] = []
        shap_payloads: list[str] = []

        for position in range(len(frame)):
            contributions = dict(
                zip(feature_names, matrix[position].astype(float), strict=False)
            )
            top = dict(sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:12])
            shap_payloads.append(json.dumps({k: round(v, 5) for k, v in top.items()}))
            explanations.append(
                describe_game_prediction(
                    contributions,
                    X.iloc[position],
                    home_team=str(frame.iloc[position]["home_team"]),
                    away_team=str(frame.iloc[position]["away_team"]),
                )
            )

        frame["shap_values"] = shap_payloads
        frame["shap_base_value"] = base_value
        frame["explanation_es"] = explanations
        frame["model_id"] = model_id
        frame["target"] = target
        frame["is_shadow"] = bool(use_shadow)

        if persist:
            self._persist(frame, target, model_id, use_shadow)

        log.info(
            "Generadas %d predicciones de '%s' con el modelo %s%s.",
            len(frame), target, model_id[:8], " (shadow)" if use_shadow else "",
        )
        return PredictionBatch(frame, target, model_id, len(frame))

    # ----------------------------------------------------------- persistencia

    def _persist(
        self, frame: pd.DataFrame, target: str, model_id: str, is_shadow: bool
    ) -> None:
        now = datetime.now(UTC).replace(tzinfo=None)

        rows = pd.DataFrame(
            {
                "prediction_id": [str(uuid.uuid4()) for _ in range(len(frame))],
                "game_id": frame["game_id"],
                "model_id": model_id,
                "target": target,
                "gsis_id": None,
                "point_estimate": frame["point_estimate"],
                "probability": frame["probability"],
                "pred_std": frame["pred_std"],
                "lower_80": frame["lower_80"],
                "upper_80": frame["upper_80"],
                "confidence_level": frame["confidence_level"],
                "shap_values": frame["shap_values"],
                "shap_base_value": frame["shap_base_value"],
                "explanation_es": frame["explanation_es"],
                "is_shadow": is_shadow,
                "as_of_ts": frame["as_of_ts"],
                "predicted_at": now,
            }
        )

        # Se borran las predicciones previas del mismo modelo para los mismos
        # partidos: reejecutar el comando no debe duplicar filas.
        game_list = ", ".join(f"'{g}'" for g in frame["game_id"])
        self.db.execute(
            f"DELETE FROM predictions WHERE model_id = '{model_id}' "
            f"AND target = '{target}' AND game_id IN ({game_list})"
        )
        self.db.write_df(rows, "predictions", mode="append")

    # --------------------------------------------------------------- props

    def predict_props(
        self,
        target: str,
        *,
        season: int | None = None,
        week: int | None = None,
        persist: bool = True,
    ) -> PredictionBatch | None:
        """Predice un prop de jugador para los partidos indicados."""
        from nflp.agents.modeling.props import filter_prop_rows

        selection = self._select_model(target, use_shadow=False)
        if selection is None:
            return None
        model_id, artifact = selection

        clause = []
        if season is not None:
            clause.append(f"f.season = {season}")
        if week is not None:
            clause.append(f"f.week = {week}")
        where = (" AND " + " AND ".join(clause)) if clause else ""

        frame = self.db.query(
            f"""
            SELECT f.*, g.kickoff_utc, g.home_team, g.away_team,
                   p.display_name AS player_name
            FROM features_player f
            JOIN games g ON g.game_id = f.game_id
            LEFT JOIN players p ON p.gsis_id = f.gsis_id
            WHERE f.feature_set_version = '{self.settings.features.version}'{where}
            """
        )
        if frame.empty:
            log.warning("Sin filas de jugador para predecir '%s'.", target)
            return None

        if season is None and week is None:
            # Mismo criterio por defecto que `predict_games`: sin filtro
            # explícito, solo próximos partidos. Sin esto, cada corrida del
            # pipeline semanal volvía a predecir las 16 temporadas completas
            # de historia en vez de solo la semana que realmente importa.
            frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], errors="coerce")
            now = pd.Timestamp.utcnow().tz_localize(None)
            frame = frame[frame["kickoff_utc"] >= now]
            if frame.empty:
                log.warning("No hay partidos futuros que proyectar para '%s'.", target)
                return None

        frame = filter_prop_rows(frame, target, self.settings.props.min_projected_snaps)
        if frame.empty:
            log.warning("Ningún jugador supera el umbral de participación para '%s'.", target)
            return None

        feature_names = artifact["feature_names"]
        X = frame.reindex(columns=feature_names).astype(float).reset_index(drop=True)
        frame = frame.reset_index(drop=True)

        model = artifact["model"]
        distribution = artifact["distribution"]
        prediction = np.maximum(model.predict(X), 0.0)

        lower, upper = distribution.interval(prediction)
        result = frame[["game_id", "gsis_id", "season", "week", "team", "player_name", "as_of_ts"]].copy()
        result["point_estimate"] = prediction
        result["pred_std"] = distribution.std(prediction)
        result["lower_80"], result["upper_80"] = lower, upper
        result["probability"] = np.nan
        result["confidence_level"] = "media"

        explainer = Explainer(model, feature_names)
        matrix, base_value = explainer.shap_matrix(X)

        from nflp.agents.modeling.explain import describe_contributions

        result["shap_values"] = [
            json.dumps(
                {
                    k: round(v, 5)
                    for k, v in sorted(
                        zip(feature_names, matrix[i].astype(float), strict=False),
                        key=lambda kv: -abs(kv[1]),
                    )[:10]
                }
            )
            for i in range(len(frame))
        ]
        result["shap_base_value"] = base_value
        result["explanation_es"] = [
            describe_contributions(
                dict(zip(feature_names, matrix[i].astype(float), strict=False)), X.iloc[i]
            )
            for i in range(len(frame))
        ]

        if persist:
            now = datetime.now(UTC).replace(tzinfo=None)
            rows = result.assign(
                prediction_id=[str(uuid.uuid4()) for _ in range(len(result))],
                model_id=model_id,
                target=target,
                is_shadow=False,
                predicted_at=now,
            )
            keep = [
                "prediction_id", "game_id", "model_id", "target", "gsis_id",
                "point_estimate", "probability", "pred_std", "lower_80", "upper_80",
                "confidence_level", "shap_values", "shap_base_value", "explanation_es",
                "is_shadow", "as_of_ts", "predicted_at",
            ]
            game_list = ", ".join(f"'{g}'" for g in result["game_id"].unique())
            self.db.execute(
                f"DELETE FROM predictions WHERE model_id = '{model_id}' "
                f"AND target = '{target}' AND game_id IN ({game_list})"
            )
            self.db.write_df(rows[keep], "predictions", mode="append")

        log.info("Generadas %d proyecciones de '%s'.", len(result), target)
        return PredictionBatch(result, target, model_id, len(result))
