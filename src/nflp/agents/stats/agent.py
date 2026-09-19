"""Stats Agent — estadísticas históricas por jugador y equipo.

Responsabilidad única: mantener las series semanales multi-temporada que
alimentan al Feature Agent. Regular season y playoffs se conservan **marcados por
separado** (columna ``season_type``), como exige la sección 2 del prompt: los
playoffs tienen una distribución distinta y mezclarlos sesga las medias móviles.

La selección de columnas es deliberadamente conservadora. nflverse publica 150
columnas por jugador y 138 por equipo; aquí se retienen las que tienen valor
predictivo conocido (EPA por fase, CPOE, cuotas de uso) en vez de arrastrarlo
todo. Menos columnas basura significa menos ruido para el modelo.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nflp.agents.data import normalize as norm
from nflp.agents.data.agent import IngestionReport
from nflp.agents.data.nflverse import NflverseClient
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    """Serie numérica de una columna, o NaN si la columna no existe.

    nflverse añade y renombra columnas entre versiones; esto evita que la
    ingesta se rompa por un cambio aguas arriba.
    """
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _str(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[column].astype("string")


# Métricas de jugador retenidas (el resto se descarta a propósito).
PLAYER_METRICS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "passing_air_yards", "passing_first_downs", "passing_epa",
    "passing_cpoe", "pacr",
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs", "rushing_epa",
    "rushing_fumbles_lost",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "target_share", "air_yards_share", "wopr", "racr", "fantasy_points_ppr",
]

# Métricas de equipo retenidas.
TEAM_METRICS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "passing_epa", "passing_cpoe", "passing_first_downs", "passing_air_yards",
    "sacks_suffered",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa", "rushing_first_downs",
    "receiving_yards", "receiving_epa",
    "def_sacks", "def_qb_hits", "def_interceptions", "def_pass_defended",
    "def_tackles_for_loss", "def_fumbles_forced", "def_tds",
    "fumbles_lost", "special_teams_tds",
]


class StatsAgent:
    """Ingesta de estadísticas semanales de jugador y equipo."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.client = NflverseClient(self.settings)

    def _log_ingestion(self, asset: str, rows: int) -> None:
        entry = pd.DataFrame(
            [
                {
                    "ingestion_id": str(uuid.uuid4()),
                    "source": "nflverse",
                    "asset": asset,
                    "season": None,
                    "rows_ingested": rows,
                    "etag": None,
                    "url": None,
                    "status": "ok",
                    "message": None,
                    "fetched_at": datetime.now(UTC).replace(tzinfo=None),
                }
            ]
        )
        self.db.write_df(entry, "ingestion_log", mode="append")

    # ------------------------------------------------------- stats jugador

    def ingest_player_stats(
        self, seasons: list[int] | None = None, *, force: bool = False
    ) -> int:
        seasons = seasons or self.settings.ingestion.seasons
        raw = self.client.load_seasonal("player_stats", seasons, force=force)
        if raw.empty:
            return 0

        frame = pd.DataFrame(
            {
                "player_id": _str(raw, "player_id"),
                "season": pd.to_numeric(raw["season"], errors="coerce").astype("Int64"),
                "week": pd.to_numeric(raw["week"], errors="coerce").astype("Int64"),
                "season_type": norm.normalize_season_type(raw["season_type"]),
                "game_id": _str(raw, "game_id"),
                "team": norm.normalize_team(raw["team"]),
                "opponent_team": norm.normalize_team(raw["opponent_team"]),
                "player_name": _str(raw, "player_display_name"),
                "position": _str(raw, "position"),
                "position_group": _str(raw, "position_group"),
            }
        )
        for metric in PLAYER_METRICS:
            frame[metric] = _num(raw, metric)

        frame = frame[frame["player_id"].notna() & frame["week"].notna()]
        frame = norm.deduplicate(
            frame, ["player_id", "season", "week", "season_type"], label="player_stats_weekly"
        )

        rows = self.db.write_df(
            frame,
            "player_stats_weekly",
            mode="upsert",
            key_columns=["player_id", "season", "week", "season_type"],
        )
        self._log_ingestion("stats_player_week", rows)
        log.info("Estadísticas de jugador: %d filas.", rows)
        return rows

    # --------------------------------------------------------- stats equipo

    def ingest_team_stats(
        self, seasons: list[int] | None = None, *, force: bool = False
    ) -> int:
        seasons = seasons or self.settings.ingestion.seasons
        raw = self.client.load_seasonal("team_stats", seasons, force=force)
        if raw.empty:
            return 0

        frame = pd.DataFrame(
            {
                "team": norm.normalize_team(raw["team"]),
                "season": pd.to_numeric(raw["season"], errors="coerce").astype("Int64"),
                "week": pd.to_numeric(raw["week"], errors="coerce").astype("Int64"),
                "season_type": norm.normalize_season_type(raw["season_type"]),
                "game_id": _str(raw, "game_id"),
                "opponent_team": norm.normalize_team(raw["opponent_team"]),
            }
        )
        for metric in TEAM_METRICS:
            frame[metric] = _num(raw, metric)

        frame = frame[frame["team"].notna() & frame["week"].notna()]
        frame = norm.deduplicate(
            frame, ["team", "season", "week", "season_type"], label="team_stats_weekly"
        )

        rows = self.db.write_df(
            frame,
            "team_stats_weekly",
            mode="upsert",
            key_columns=["team", "season", "week", "season_type"],
        )
        self._log_ingestion("stats_team_week", rows)
        log.info("Estadísticas de equipo: %d filas.", rows)
        return rows

    # ------------------------------------------------------------------- run

    def run(self, seasons: list[int] | None = None, *, force: bool = False) -> IngestionReport:
        seasons = seasons or self.settings.ingestion.seasons
        log.info("Stats Agent: temporadas %d-%d ...", min(seasons), max(seasons))

        report = IngestionReport()
        report.add("player_stats_weekly", self.ingest_player_stats(seasons, force=force))
        report.add("team_stats_weekly", self.ingest_team_stats(seasons, force=force))

        log.info("Stats Agent completado: %d filas.", report.total())
        return report
