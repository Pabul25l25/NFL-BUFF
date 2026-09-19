"""Retraining/Monitoring Agent.

Tres responsabilidades, todas orientadas a que un modelo degradado no se quede en
producción sin que nadie se entere:

1. **Drift de features** (PSI). Si la distribución de las features de esta
   temporada se aleja de aquella con la que se entrenó el modelo, sus supuestos
   ya no se sostienen. En la NFL esto pasa de verdad: cambios de reglamento,
   evolución del juego aéreo, la ventaja de campo cayendo de ~3 a ~1,8 puntos.

2. **Rendimiento en producción**. Se recalcula semana a semana comparando cada
   predicción con el resultado real, para ver la degradación a simple vista.

3. **Gate de promoción y rollback**. Un modelo en shadow solo se promueve si,
   tras N semanas corriendo en paralelo, **supera** al de producción. Y si el de
   producción se degrada más de lo tolerado, se revierte a la versión anterior.

El PSI se interpreta con los umbrales habituales de la industria:
< 0,10 estable · 0,10-0,25 cambio moderado · > 0,25 cambio severo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nflp.agents.backtesting.metrics import classification_metrics, regression_metrics
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger
from nflp.registry import ModelRegistry

log = get_logger(__name__)


@dataclass
class DriftReport:
    """Resultado de una comprobación de drift."""

    checked_at: datetime
    drifted_features: list[str] = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    should_retrain: bool = False

    def describe(self) -> str:
        if not self.drifted_features:
            return "Sin drift relevante: todas las features dentro del umbral."
        return (
            f"{len(self.drifted_features)} features con drift: "
            + ", ".join(self.drifted_features[:8])
            + ("..." if len(self.drifted_features) > 8 else "")
        )


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = 10
) -> float:
    """PSI entre una distribución de referencia y otra actual.

    Se construyen los cortes sobre los cuantiles de la referencia y se compara
    la proporción de masa que cae en cada bin. Un épsilon evita la división por
    cero cuando un bin queda vacío en una de las dos distribuciones.
    """
    # `astype(float)` explícito: las columnas booleanas pasan el filtro de
    # "numérico" de pandas pero rompen `quantile()` y la resta de numpy.
    reference_clean = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
    current_clean = pd.to_numeric(current, errors="coerce").dropna().astype(float)

    if len(reference_clean) < 50 or len(current_clean) < 20:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference_clean.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_counts = np.histogram(reference_clean, bins=edges)[0].astype(float)
    current_counts = np.histogram(current_clean, bins=edges)[0].astype(float)

    epsilon = 1e-6
    reference_pct = np.maximum(reference_counts / reference_counts.sum(), epsilon)
    current_pct = np.maximum(current_counts / current_counts.sum(), epsilon)

    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


class MonitoringAgent:
    """Vigila drift, rendimiento y el ciclo de vida de los modelos."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.registry = ModelRegistry(self.settings, self.db)

    # ------------------------------------------------------------------ drift

    def check_drift(
        self,
        *,
        reference_seasons: list[int] | None = None,
        current_season: int | None = None,
        persist: bool = True,
    ) -> DriftReport:
        """Compara la distribución de features actual con la de referencia."""
        features = self.db.query(
            f"SELECT * FROM features_game "
            f"WHERE feature_set_version = '{self.settings.features.version}'"
        )
        if features.empty:
            log.warning("Feature store vacío; no se puede medir drift.")
            return DriftReport(checked_at=datetime.now(UTC).replace(tzinfo=None))

        seasons = sorted(features["season"].unique())
        current = current_season or seasons[-1]
        reference = reference_seasons or [s for s in seasons if s < current][-5:]

        if not reference:
            log.warning("Sin temporadas de referencia para comparar.")
            return DriftReport(checked_at=datetime.now(UTC).replace(tzinfo=None))

        reference_frame = features[features["season"].isin(reference)]
        current_frame = features[features["season"] == current]

        exclude = {
            "game_id", "feature_set_version", "season", "week", "season_type",
            "home_team", "away_team", "as_of_ts", "feature_hash",
        }
        numeric_columns = [
            c for c in features.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
        ]

        rows: list[dict] = []
        drifted: list[str] = []
        threshold = self.settings.monitoring.psi_threshold

        for column in numeric_columns:
            psi = population_stability_index(reference_frame[column], current_frame[column])
            if np.isnan(psi):
                continue
            is_drifted = psi > threshold
            if is_drifted:
                drifted.append(column)
            rows.append(
                {
                    "drift_id": str(uuid.uuid4()),
                    "checked_at": datetime.now(UTC).replace(tzinfo=None),
                    "season": int(current),
                    "week": int(current_frame["week"].max()) if not current_frame.empty else None,
                    "feature_name": column,
                    "psi": psi,
                    "kl_divergence": None,
                    "reference_mean": float(reference_frame[column].mean()),
                    "current_mean": float(current_frame[column].mean()),
                    "is_drifted": is_drifted,
                }
            )

        metrics = pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame()

        if persist and not metrics.empty:
            self.db.write_df(metrics, "drift_metrics", mode="append")

        # Se dispara reentrenamiento si una parte sustancial del espacio de
        # features se ha movido, no por una feature suelta.
        should_retrain = len(drifted) >= max(3, len(numeric_columns) // 10)

        report = DriftReport(
            checked_at=datetime.now(UTC).replace(tzinfo=None),
            drifted_features=drifted,
            metrics=metrics,
            should_retrain=should_retrain,
        )
        log.info(
            "Drift (%s vs %d): %s%s",
            reference, current, report.describe(),
            " → se recomienda reentrenar." if should_retrain else "",
        )
        return report

    # ---------------------------------------------------- rendimiento en prod

    def compute_production_performance(
        self, target: str, *, persist: bool = True
    ) -> pd.DataFrame:
        """Rendimiento semana a semana de las predicciones ya resueltas.

        Los props (``receiving_yards``, ``rushing_yards``, ...) tienen su verdad
        de referencia en ``player_stats_weekly``, no en el marcador del partido.
        Antes de esta versión, un prop pasado aquí caía en el ``else`` genérico y
        se comparaba silenciosamente contra `home_score + away_score` — el mismo
        cálculo que ``total``. Nunca llegó a ejecutarse en producción porque los
        props no estaban en la lista de objetivos vigilados, pero era un error
        latente real. Se separa en dos caminos explícitos.
        """
        if target in self.settings.props.targets:
            return self._compute_prop_performance(target, persist=persist)
        return self._compute_game_performance(target, persist=persist)

    def _compute_game_performance(
        self, target: str, *, persist: bool = True
    ) -> pd.DataFrame:
        frame = self.db.query(
            f"""
            SELECT p.model_id, p.game_id, p.target, p.point_estimate, p.probability,
                   g.season, g.week, g.home_score, g.away_score
            FROM predictions p
            JOIN games g USING (game_id)
            WHERE p.target = '{target}'
              AND g.home_score IS NOT NULL
              AND p.is_shadow = FALSE
            """
        )
        if frame.empty:
            log.info("Sin predicciones resueltas para '%s'.", target)
            return pd.DataFrame()

        if target == "winner":
            frame["y_true"] = (frame["home_score"] > frame["away_score"]).astype(float)
        elif target == "spread":
            frame["y_true"] = frame["home_score"] - frame["away_score"]
        else:
            frame["y_true"] = frame["home_score"] + frame["away_score"]

        rows: list[dict] = []
        now = datetime.now(UTC).replace(tzinfo=None)

        for (model_id, season, week), group in frame.groupby(["model_id", "season", "week"]):
            row = {
                "perf_id": str(uuid.uuid4()),
                "model_id": model_id,
                "target": target,
                "season": int(season),
                "week": int(week),
                "n_predictions": len(group),
                "computed_at": now,
                "log_loss": None, "brier_score": None, "accuracy": None, "mae": None,
            }
            if target == "winner":
                metrics = classification_metrics(group["y_true"], group["probability"])
                row.update(
                    {
                        "log_loss": metrics.log_loss,
                        "brier_score": metrics.brier_score,
                        "accuracy": metrics.accuracy,
                    }
                )
            else:
                metrics = regression_metrics(group["y_true"], group["point_estimate"])
                row["mae"] = metrics.mae
            rows.append(row)

        performance = pd.DataFrame(rows)
        if persist and not performance.empty:
            self._persist_performance(performance)

        return performance

    def _persist_performance(self, performance: pd.DataFrame) -> None:
        """Guarda el rendimiento con upsert por (modelo, objetivo, temporada, semana).

        **Hallazgo real**: la versión anterior usaba ``append`` sin más. Cada vez
        que se corría el monitoreo (por ejemplo, dentro de ``nflp pipeline
        weekly``, pensado para ejecutarse repetidamente), se sumaba una fila
        duplicada para la misma semana ya calculada. Con eso, ``check_degradation``
        —que exige al menos 4 filas antes de considerar un rollback— se
        "cumplía" con una sola semana real repetida 4 veces, y estuvo a punto de
        disparar un rollback a partir del ruido estadístico de una sola semana de
        16 partidos. El upsert por semana real corrige la causa de raíz.
        """
        self.db.write_df(
            performance, "production_performance", mode="upsert",
            key_columns=["model_id", "target", "season", "week"],
        )

    def _compute_prop_performance(
        self, target: str, *, persist: bool = True
    ) -> pd.DataFrame:
        """Rendimiento semana a semana de un prop, contra su estadística real."""
        frame = self.db.query(
            f"""
            SELECT p.model_id, p.game_id, p.gsis_id, p.point_estimate,
                   g.season, g.week, s.{target} AS y_true
            FROM predictions p
            JOIN games g ON g.game_id = p.game_id
            JOIN player_stats_weekly s
              ON s.player_id = p.gsis_id AND s.season = g.season
             AND s.week = g.week AND s.season_type = g.season_type
            WHERE p.target = '{target}'
              AND p.gsis_id IS NOT NULL
              AND p.is_shadow = FALSE
              AND s.{target} IS NOT NULL
            """
        )
        if frame.empty:
            log.info("Sin proyecciones resueltas para '%s'.", target)
            return pd.DataFrame()

        rows: list[dict] = []
        now = datetime.now(UTC).replace(tzinfo=None)

        for (model_id, season, week), group in frame.groupby(["model_id", "season", "week"]):
            metrics = regression_metrics(group["y_true"], group["point_estimate"])
            rows.append(
                {
                    "perf_id": str(uuid.uuid4()),
                    "model_id": model_id,
                    "target": target,
                    "season": int(season),
                    "week": int(week),
                    "n_predictions": len(group),
                    "computed_at": now,
                    "log_loss": None, "brier_score": None, "accuracy": None,
                    "mae": metrics.mae,
                }
            )

        performance = pd.DataFrame(rows)
        if persist and not performance.empty:
            self._persist_performance(performance)

        return performance

    # ---------------------------------------------- promoción desde shadow

    def evaluate_shadow(self, target: str) -> dict[str, object]:
        """Decide si el modelo en shadow debe sustituir al de producción.

        Criterio: tras correr en paralelo, el shadow debe tener **mejor** métrica
        sobre los mismos partidos. Empatar no basta — cambiar de modelo tiene
        coste operativo y riesgo, así que la carga de la prueba recae en el nuevo.
        """
        shadow_id = self.registry.current_shadow(target)
        production_id = self.registry.current_production(target)

        if shadow_id is None:
            return {"action": "none", "reason": "no hay modelo en shadow"}

        if production_id is None:
            self.registry.promote(shadow_id, reason="no había modelo en producción")
            return {"action": "promote", "model_id": shadow_id, "reason": "primera promoción"}

        comparison = self.db.query(
            f"""
            SELECT model_id, avg(log_loss) AS log_loss, avg(mae) AS mae,
                   sum(n_predictions) AS n
            FROM production_performance
            WHERE target = '{target}' AND model_id IN ('{shadow_id}', '{production_id}')
            GROUP BY model_id
            """
        )

        if len(comparison) < 2:
            # Sin histórico comparable, se usan las métricas de validación.
            registry_rows = self.db.query(
                f"SELECT model_id, val_metric, status FROM model_registry "
                f"WHERE model_id IN ('{shadow_id}', '{production_id}')"
            )
            scores = dict(zip(registry_rows["model_id"], registry_rows["val_metric"], strict=True))
            shadow_score = scores.get(shadow_id)
            production_score = scores.get(production_id)

            if shadow_score is None or production_score is None:
                return {"action": "none", "reason": "métricas incompletas"}

            if shadow_score < production_score:
                self.registry.promote(
                    shadow_id,
                    reason=f"mejor validación ({shadow_score:.4f} < {production_score:.4f})",
                )
                return {"action": "promote", "model_id": shadow_id, "reason": "mejor en validación"}

            return {
                "action": "keep",
                "reason": f"el shadow no mejora ({shadow_score:.4f} >= {production_score:.4f})",
            }

        metric = "log_loss" if comparison["log_loss"].notna().any() else "mae"
        indexed = comparison.set_index("model_id")[metric]
        shadow_score = float(indexed.get(shadow_id, np.nan))
        production_score = float(indexed.get(production_id, np.nan))

        if np.isnan(shadow_score) or np.isnan(production_score):
            return {"action": "none", "reason": "métricas de producción incompletas"}

        if shadow_score < production_score:
            self.registry.promote(
                shadow_id,
                reason=f"mejor en producción real ({shadow_score:.4f} < {production_score:.4f})",
            )
            return {"action": "promote", "model_id": shadow_id, "reason": "mejor en shadow"}

        return {
            "action": "keep",
            "reason": f"el shadow no mejora ({shadow_score:.4f} >= {production_score:.4f})",
        }

    # ------------------------------------------------------------- rollback

    def check_degradation(self, target: str) -> dict[str, object]:
        """Revierte a la versión anterior si el modelo en producción se degrada."""
        production_id = self.registry.current_production(target)
        if production_id is None:
            return {"action": "none", "reason": "sin modelo en producción"}

        recent = self.db.query(
            f"""
            SELECT log_loss, mae, n_predictions, season, week
            FROM production_performance
            WHERE model_id = '{production_id}' AND target = '{target}'
            ORDER BY season DESC, week DESC LIMIT 6
            """
        )
        if len(recent) < 4:
            return {"action": "none", "reason": "histórico insuficiente para evaluar degradación"}

        metric = "log_loss" if recent["log_loss"].notna().any() else "mae"
        baseline = self.db.scalar(
            f"SELECT val_metric FROM model_registry WHERE model_id = '{production_id}'"
        )
        if baseline is None:
            return {"action": "none", "reason": "sin métrica de referencia"}

        observed = float(recent[metric].mean())
        degradation = (observed - float(baseline)) / abs(float(baseline))
        threshold = self.settings.monitoring.rollback_degradation

        if degradation > threshold:
            previous = self.registry.rollback(
                target,
                reason=(
                    f"degradación del {degradation:.1%} sobre la referencia "
                    f"({observed:.4f} vs {baseline:.4f}), umbral {threshold:.1%}"
                ),
            )
            return {
                "action": "rollback", "model_id": previous,
                "reason": f"degradación {degradation:.1%}",
            }

        return {
            "action": "none",
            "reason": f"degradación {degradation:.1%} dentro del umbral {threshold:.1%}",
        }

    # ------------------------------------------------------------------- run

    def run(self, targets: list[str] | None = None) -> dict[str, object]:
        """Ciclo completo de monitoreo.

        Por defecto vigila tanto los objetivos de partido (``winner``, ``spread``,
        ``total``) como los props (``receiving_yards``, ``rushing_yards``). Antes
        de esta versión los props quedaban fuera del monitor por defecto y nunca
        se promovían automáticamente de shadow a producción.
        """
        targets = targets or [*self.settings.modeling.targets, *self.settings.props.targets]

        drift = self.check_drift()
        results: dict[str, object] = {"drift": drift.describe(), "targets": {}}

        for target in targets:
            self.compute_production_performance(target)
            shadow = self.evaluate_shadow(target)
            degradation = self.check_degradation(target)
            results["targets"][target] = {  # type: ignore[index]
                "shadow": shadow,
                "degradacion": degradation,
            }

        return results
