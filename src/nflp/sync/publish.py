"""Publica un subconjunto de la base local a Neon/Postgres.

**Por qué existe este módulo y no otro mecanismo.** El frontend en Vercel es
serverless: no tiene disco persistente ni acceso a la red doméstica del
usuario, así que no puede abrir ``data/nflp.duckdb`` bajo ningún concepto. La
única forma de que muestre datos reales es que alguien se los deje en una base
alcanzable desde internet. Ese "alguien" es este módulo, corriendo en la PC del
usuario al final de cada `nflp pipeline weekly`.

**Qué NO hace este módulo**: no reescribe ninguna lógica de negocio. Las dos
tablas "aplanadas" (`pub_predictions_games`, `pub_predictions_props`) se
generan ejecutando en DuckDB prácticamente las mismas consultas que ya usa el
dashboard de Streamlit (`src/nflp/dashboard/data.py`), para que el criterio de
"qué se une con qué" siga viviendo en un solo lugar.

**Resiliencia deliberada**: si Neon no está configurado (`sync.enabled=False`
o sin `postgres_url`), `publish_all()` no hace nada y no falla. Si Neon está
caído o da un error de red, se registra el fallo pero **no se relanza** — las
predicciones locales y el Streamlit ya generados no dependen de esto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from nflp.core.config import DatabaseConfig, Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Claves de upsert para las tablas que se replican fila a fila (no con
# `mode="replace"`, que borraría y reinsertaría todo cada semana).
_UPSERT_KEYS: dict[str, list[str]] = {
    "games": ["game_id"],
    "predictions": ["prediction_id"],
    "players": ["gsis_id"],
    "player_stats_weekly": ["player_id", "season", "week", "season_type"],
    "weather": ["game_id"],
    "pub_predictions_games": ["game_id", "target"],
    "pub_predictions_props": ["game_id", "gsis_id", "target"],
    "pub_parlay_game_corr": ["game_id"],
    "pub_parlay_game_legs": ["game_id"],
    "pub_parlay_calibration": ["model_id"],
}

# Props de conteo: la web los modela con Poisson(lambda). Si el modelo no guardó
# lambda, la media (point_estimate) es el lambda.
_COUNT_TARGETS = ("receiving_tds", "rushing_tds", "passing_tds", "sacks_suffered")

# Mismo criterio que `dashboard/data._PROP_ACTUAL_COLUMN`: una columna real por
# cada prop soportado.
_PROP_ACTUAL_COLUMNS = [
    "receiving_yards", "rushing_yards", "receptions", "receiving_tds",
    "rushing_tds", "passing_yards", "passing_tds", "sacks_suffered",
]

# --------------------------------------------------------------- consultas
# Idénticas en espíritu a `load_predictions`/`load_prop_predictions` del
# dashboard, generalizadas a TODOS los targets a la vez (en vez de uno por
# llamada) porque aquí no hay un usuario eligiendo una pestaña: se publica todo
# de una sola pasada semanal.

_QUERY_PREDICTIONS_GAMES = """
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
    WHERE p.is_shadow = FALSE AND p.gsis_id IS NULL
    QUALIFY row_number() OVER (
        PARTITION BY p.game_id, p.target ORDER BY p.predicted_at DESC
    ) = 1
    ORDER BY g.kickoff_utc
"""

_QUERY_PREDICTIONS_PROPS = f"""
    SELECT p.game_id, p.gsis_id, p.model_id, p.target,
           p.point_estimate, p.lower_80, p.upper_80,
           p.explanation_es, p.shap_values, p.predicted_at, p.as_of_ts,
           p.quantiles,
           CASE WHEN p.target IN ({", ".join(repr(t) for t in _COUNT_TARGETS)})
                THEN COALESCE(p.poisson_lambda, p.point_estimate)
           END AS poisson_lambda,
           g.season, g.week, g.home_team, g.away_team, g.kickoff_utc,
           pl.display_name AS player_name, pl.position,
           -- Equipo/rival/localía: primero lo guardado con la predicción; si falta
           -- (filas antiguas), features_player (una fila por jugador-partido) y, por
           -- último, player_stats_weekly (que no tiene filas de partidos sin jugar).
           COALESCE(p.team, f.team, s.team) AS team,
           COALESCE(p.opponent_team, f.opponent_team) AS opponent_team,
           COALESCE(p.is_home, f.is_home) AS is_home,
           {", ".join(f"s.{c}" for c in _PROP_ACTUAL_COLUMNS)}
    FROM predictions p
    -- predictions guarda filas de modelos anteriores: solo se publica el de producción.
    JOIN model_registry mr ON mr.model_id = p.model_id AND mr.status = 'production'
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
    WHERE p.gsis_id IS NOT NULL
    QUALIFY row_number() OVER (
        PARTITION BY p.game_id, p.gsis_id, p.target ORDER BY p.predicted_at DESC
    ) = 1
    ORDER BY p.point_estimate DESC
"""

# Parlays: último modelo por partido (los dos primeros) y última calibración.
_QUERY_PARLAY_CORR = """
    SELECT game_id, model_id, legs, matrix, as_of_ts
    FROM parlay_game_corr
    QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY as_of_ts DESC) = 1
"""

_QUERY_PARLAY_LEGS = """
    SELECT game_id, model_id, p_home_win, margin_mu, margin_sigma,
           total_mu, total_sigma, as_of_ts
    FROM parlay_game_legs
    QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY as_of_ts DESC) = 1
"""

_QUERY_PARLAY_CALIBRATION = """
    SELECT model_id, ece, logloss, ece_independent, logloss_independent, n, created_at
    FROM parlay_calibration
    ORDER BY created_at DESC
    LIMIT 1
"""


@dataclass
class PublishReport:
    """Resumen de una corrida de publicación."""

    tables: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None

    def add(self, table: str, rows: int) -> None:
        self.tables[table] = rows

    def total(self) -> int:
        return sum(self.tables.values())


def _cloud_db(settings: Settings) -> Database:
    """Segunda conexión, apuntando a Postgres, sin tocar la configuración local.

    Reutiliza `Database` tal cual — el backend ya sabe hablar con Postgres
    (`src/nflp/core/db.py`), así que no hace falta código nuevo de bajo nivel,
    solo una copia de `Settings` con el backend cambiado.
    """
    cloud_settings = settings.model_copy(deep=True)
    cloud_settings.database = DatabaseConfig(
        backend="postgresql", postgres_url=settings.sync.postgres_url
    )
    return Database(cloud_settings)


def publish_all(
    settings: Settings | None = None,
    local_db: Database | None = None,
) -> PublishReport:
    """Publica el subconjunto configurado de tablas a Neon.

    No lanza excepciones por fallos de red o de configuración incompleta: los
    devuelve en `PublishReport.error` para que quien llame (CLI, el paso del
    Orchestrator) decida qué hacer. El pipeline semanal, por ejemplo, solo
    registra el error y sigue — la publicación es la última etapa y nunca debe
    tirar abajo lo que ya se calculó localmente.
    """
    settings = settings or get_settings()
    report = PublishReport()

    if not settings.sync.enabled or not settings.sync.postgres_url:
        log.info(
            "Publicación a la nube desactivada (NFLP_SYNC__ENABLED=false o sin "
            "postgres_url); no se hace nada."
        )
        report.skipped = True
        return report

    local = local_db or get_db(settings)

    try:
        cloud = _cloud_db(settings)
    except Exception as exc:  # noqa: BLE001 - error de configuración/red, no debe propagar
        log.error("No se pudo conectar a Neon: %s", exc)
        report.error = str(exc)
        return report

    try:
        _publish_plain_tables(local, cloud, settings.sync.tables_replace, report)
        _publish_games_and_predictions(local, cloud, report)
        _publish_recent_player_data(local, cloud, settings.sync.recent_seasons, report)
        _publish_flattened_views(local, cloud, report)
        _touch_sync_meta(cloud, report)
    except Exception as exc:  # noqa: BLE001 - un fallo de red a mitad de camino no debe tirar el pipeline
        log.error("Publicación a la nube interrumpida: %s", exc)
        report.error = str(exc)
        return report

    log.info(
        "Publicación a la nube completada: %d tablas, %d filas en total.",
        len(report.tables), report.total(),
    )
    return report


def _publish_plain_tables(
    local: Database, cloud: Database, tables: list[str], report: PublishReport
) -> None:
    """Tablas pequeñas que se reemplazan enteras en cada corrida."""
    for table in tables:
        if not local.table_exists(table):
            log.debug("Tabla local '%s' no existe todavía; se omite.", table)
            continue
        frame = local.query(f"SELECT * FROM {table}")
        rows = cloud.write_df(frame, table, mode="replace")
        report.add(table, rows)
        log.debug("Publicado '%s': %d filas.", table, rows)


def _publish_games_and_predictions(
    local: Database, cloud: Database, report: PublishReport
) -> None:
    """Calendario completo y todas las predicciones, vía upsert."""
    for table in ("games", "predictions"):
        frame = local.query(f"SELECT * FROM {table}")
        rows = cloud.write_df(frame, table, mode="upsert", key_columns=_UPSERT_KEYS[table])
        report.add(table, rows)
        log.debug("Publicado '%s': %d filas (upsert).", table, rows)


def _publish_recent_player_data(
    local: Database, cloud: Database, recent_seasons: int, report: PublishReport
) -> None:
    """Jugadores y sus estadísticas/clima de las últimas N temporadas.

    El frontend nunca necesita comparar props de hace 10 años; limitarlo evita
    que la base de Neon crezca sin límite en el free tier.
    """
    max_season = local.scalar("SELECT max(season) FROM games")
    if max_season is None:
        return
    min_season = int(max_season) - recent_seasons + 1

    stats = local.query(
        f"SELECT * FROM player_stats_weekly WHERE season >= {min_season}"
    )
    rows = cloud.write_df(
        stats, "player_stats_weekly", mode="upsert",
        key_columns=_UPSERT_KEYS["player_stats_weekly"],
    )
    report.add("player_stats_weekly", rows)

    weather = local.query(
        f"""
        SELECT w.* FROM weather w JOIN games g USING (game_id)
        WHERE g.season >= {min_season}
        """
    )
    rows = cloud.write_df(weather, "weather", mode="upsert", key_columns=_UPSERT_KEYS["weather"])
    report.add("weather", rows)

    players = local.query(
        """
        SELECT DISTINCT pl.*
        FROM players pl
        WHERE pl.gsis_id IN (
            SELECT DISTINCT gsis_id FROM predictions WHERE gsis_id IS NOT NULL
        )
        """
    )
    rows = cloud.write_df(players, "players", mode="upsert", key_columns=_UPSERT_KEYS["players"])
    report.add("players", rows)


def _publish_flattened_views(local: Database, cloud: Database, report: PublishReport) -> None:
    """Las dos tablas ya unidas y listas para servir sin joins en el frontend."""
    games_view = local.query(_QUERY_PREDICTIONS_GAMES)
    rows = cloud.write_df(
        games_view, "pub_predictions_games", mode="upsert",
        key_columns=_UPSERT_KEYS["pub_predictions_games"],
    )
    report.add("pub_predictions_games", rows)

    props_view = local.query(_QUERY_PREDICTIONS_PROPS)
    rows = cloud.write_df(
        props_view, "pub_predictions_props", mode="upsert",
        key_columns=_UPSERT_KEYS["pub_predictions_props"],
    )
    report.add("pub_predictions_props", rows)

    for table, query in (
        ("pub_parlay_game_corr", _QUERY_PARLAY_CORR),
        ("pub_parlay_game_legs", _QUERY_PARLAY_LEGS),
        ("pub_parlay_calibration", _QUERY_PARLAY_CALIBRATION),
    ):
        source = table.removeprefix("pub_")
        if not local.table_exists(source):
            log.debug("Tabla local '%s' no existe todavía; se omite.", source)
            continue
        frame = local.query(query)
        if frame.empty:
            continue
        rows = cloud.write_df(frame, table, mode="upsert", key_columns=_UPSERT_KEYS[table])
        report.add(table, rows)


def _touch_sync_meta(cloud: Database, report: PublishReport) -> None:
    """Registra cuándo se sincronizó cada tabla, para el badge de estado en la UI."""
    now = datetime.now(UTC).replace(tzinfo=None)
    meta = pd.DataFrame(
        [
            {"table_name": table, "synced_at": now, "row_count": rows}
            for table, rows in report.tables.items()
        ]
    )
    if not meta.empty:
        cloud.write_df(meta, "sync_meta", mode="upsert", key_columns=["table_name"])
