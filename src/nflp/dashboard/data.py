"""Capa de acceso a datos del dashboard.

El dashboard **lee siempre de la base de datos**; no hay ningún resultado
hardcodeado. Cada consulta se cachea unos minutos para que la interfaz responda
sin machacar DuckDB, pero el caché expira solo, de modo que tras un
reentrenamiento o una nueva semana los datos aparecen sin tocar nada.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from nflp.core.config import get_settings
from nflp.core.db import get_db, validate_identifier

CACHE_TTL = 300  # segundos


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Ejecuta una consulta cacheada con parámetros enlazados (``?``)."""
    return get_db().query(sql, list(params) if params else None)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_predictions(
    target: str,
    season: int | None = None,
    week: int | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """Predicciones con su resultado real cuando el partido ya se jugó."""
    clauses = ["p.target = ?", "p.is_shadow = FALSE"]
    params: list = [target]
    if season is not None:
        clauses.append("g.season = ?")
        params.append(int(season))
    if week is not None:
        clauses.append("g.week = ?")
        params.append(int(week))
    if team:
        clauses.append("(g.home_team = ? OR g.away_team = ?)")
        params.extend([team, team])

    return query(
        f"""
        SELECT p.prediction_id, p.game_id, p.model_id, p.target,
               p.point_estimate, p.probability, p.pred_std,
               p.lower_80, p.upper_80, p.confidence_level,
               p.shap_values, p.explanation_es, p.predicted_at, p.as_of_ts,
               g.season, g.week, g.season_type, g.gameday, g.kickoff_utc,
               g.home_team, g.away_team, g.home_score, g.away_score,
               g.result AS actual_margin, g.total AS actual_total,
               g.roof, g.surface, g.stadium,
               r.version AS model_version, r.family AS model_family,
               w.temperature_c, w.wind_kph, w.precipitation_mm
        FROM predictions p
        JOIN games g USING (game_id)
        LEFT JOIN model_registry r ON r.model_id = p.model_id
        LEFT JOIN weather w ON w.game_id = p.game_id
        WHERE {' AND '.join(clauses)}
        -- Un mismo partido puede tener predicciones de varias versiones del
        -- modelo (se conservan todas en la tabla por trazabilidad). Aquí solo
        -- nos interesa la más reciente por partido; el historial completo
        -- vive en la pestaña "Modelos y trazabilidad".
        QUALIFY row_number() OVER (
            PARTITION BY p.game_id ORDER BY p.predicted_at DESC
        ) = 1
        ORDER BY g.kickoff_utc
        """,
        tuple(params),
    )


# Columnas reales de player_stats_weekly, una por cada prop soportado. Se usa
# para saber contra qué comparar la proyección una vez jugado el partido.
_PROP_ACTUAL_COLUMN = {
    "receiving_yards": "receiving_yards",
    "rushing_yards": "rushing_yards",
    "receptions": "receptions",
    "receiving_tds": "receiving_tds",
    "rushing_tds": "rushing_tds",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "sacks_suffered": "sacks_suffered",
}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_prop_predictions(
    target: str, season: int | None = None, week: int | None = None
) -> pd.DataFrame:
    """Proyecciones de props con el resultado real si ya se jugó."""
    clauses = ["p.target = ?", "p.gsis_id IS NOT NULL"]
    params: list = [target]
    if season is not None:
        clauses.append("g.season = ?")
        params.append(int(season))
    if week is not None:
        clauses.append("g.week = ?")
        params.append(int(week))

    # Nombre de columna: no puede enlazarse, así que se valida como identificador.
    actual_column = validate_identifier(_PROP_ACTUAL_COLUMN.get(target, target))

    return query(
        f"""
        SELECT p.game_id, p.gsis_id, p.point_estimate, p.lower_80, p.upper_80,
               p.explanation_es, p.shap_values,
               g.season, g.week, g.home_team, g.away_team, g.kickoff_utc,
               pl.display_name AS player_name, pl.position,
               s.{actual_column} AS actual_value,
               COALESCE(s.team, f.team) AS team, f.opponent_team, f.is_home
        FROM predictions p
        JOIN games g USING (game_id)
        LEFT JOIN players pl ON pl.gsis_id = p.gsis_id
        LEFT JOIN player_stats_weekly s
          ON s.player_id = p.gsis_id AND s.season = g.season AND s.week = g.week
         AND s.season_type = g.season_type
        LEFT JOIN (
            SELECT game_id, gsis_id, team, opponent_team, is_home
            FROM features_player
            QUALIFY row_number() OVER (
                PARTITION BY game_id, gsis_id
                ORDER BY feature_set_version DESC, as_of_ts DESC
            ) = 1
        ) f ON f.game_id = p.game_id AND f.gsis_id = p.gsis_id
        WHERE {' AND '.join(clauses)}
        -- misma razón que en load_predictions: solo la proyección más
        -- reciente por jugador y partido, no una por cada versión de modelo.
        QUALIFY row_number() OVER (
            PARTITION BY p.game_id, p.gsis_id ORDER BY p.predicted_at DESC
        ) = 1
        ORDER BY p.point_estimate DESC
        """,
        tuple(params),
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_model_registry() -> pd.DataFrame:
    """Registro completo de versiones de modelo."""
    return query(
        """
        SELECT model_id, target, family, version, status, metric_name,
               train_metric, val_metric, train_val_gap, feature_set_version,
               dataset_hash, train_seasons, n_train_rows, n_val_rows,
               promotion_blocked_reason, created_at, promoted_at, archived_at
        FROM model_registry
        ORDER BY target, version DESC
        """
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_backtest_results(target: str | None = None) -> pd.DataFrame:
    """Resultados del backtest más reciente."""
    clause = "AND target = ?" if target else ""
    run_id = get_db().scalar(
        f"SELECT run_id FROM backtest_results WHERE 1=1 {clause} "
        "ORDER BY created_at DESC LIMIT 1",
        [target] if target else None,
    )
    if not run_id:
        return pd.DataFrame()
    return query("SELECT * FROM backtest_results WHERE run_id = ?", (run_id,))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_calibration(target: str) -> pd.DataFrame:
    """Puntos de la curva de calibración de la última corrida."""
    run_id = get_db().scalar(
        "SELECT run_id FROM calibration_bins WHERE target = ? LIMIT 1", [target]
    )
    if not run_id:
        return pd.DataFrame()
    return query(
        "SELECT * FROM calibration_bins WHERE run_id = ? AND target = ? "
        "ORDER BY model_label, bin_index",
        (run_id, target),
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_production_performance(target: str) -> pd.DataFrame:
    """Rendimiento semanal del modelo en producción."""
    return query(
        """
        SELECT pp.*, r.version, r.family
        FROM production_performance pp
        LEFT JOIN model_registry r ON r.model_id = pp.model_id
        WHERE pp.target = ?
        ORDER BY pp.season, pp.week
        """,
        (target,),
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_drift() -> pd.DataFrame:
    """Última comprobación de drift."""
    checked_at = get_db().scalar("SELECT max(checked_at) FROM drift_metrics")
    if checked_at is None:
        return pd.DataFrame()
    return query(
        "SELECT * FROM drift_metrics WHERE checked_at = ? ORDER BY psi DESC",
        (checked_at,),
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_retraining_log(limit: int = 50) -> pd.DataFrame:
    """Bitácora de decisiones del agente de monitoreo."""
    return query(
        """
        SELECT occurred_at, target, action, model_id, previous_model_id, reason
        FROM retraining_log ORDER BY occurred_at DESC LIMIT ?
        """,
        (int(limit),),
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def available_filters() -> dict[str, list]:
    """Valores disponibles para los filtros de la barra lateral."""
    seasons = query("SELECT DISTINCT season FROM games ORDER BY season DESC")
    teams = query("SELECT DISTINCT team_abbr FROM teams ORDER BY team_abbr")
    weeks = query("SELECT DISTINCT week FROM games ORDER BY week")
    return {
        "seasons": seasons["season"].tolist() if not seasons.empty else [],
        "teams": teams["team_abbr"].tolist() if not teams.empty else [],
        "weeks": weeks["week"].tolist() if not weeks.empty else [],
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def database_status() -> dict[str, int]:
    """Número de filas por tabla, para el panel de estado."""
    db = get_db()
    tables = [
        "games", "features_game", "features_player", "predictions",
        "model_registry", "backtest_results", "weather", "injuries",
    ]
    return {name: db.row_count(name) for name in tables}


def parse_shap(payload: str | None) -> dict[str, float]:
    """Decodifica el JSON de valores SHAP almacenado en ``predictions``."""
    if not payload:
        return {}
    try:
        return {str(k): float(v) for k, v in json.loads(payload).items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def has_data() -> bool:
    """Comprueba que hay datos suficientes para mostrar el dashboard."""
    try:
        return get_db().row_count("games") > 0
    except Exception:  # noqa: BLE001 - la base puede no existir todavía
        return False


def settings():
    return get_settings()
