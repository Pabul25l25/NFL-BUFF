"""Construcción del feature store a nivel jugador-partido (``features_player``).

Mismo contrato as-of que el de partido: cada fila lleva el instante de corte, las
medias móviles excluyen el partido objetivo y el estado de lesión se resuelve con
el último parte publicado **antes** de ese corte.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from nflp.agents.data.normalize import is_closed_roof
from nflp.agents.data.weather import is_adverse_weather
from nflp.agents.features.contracts import audit_features
from nflp.agents.features.player_form import PLAYER_EWM_HALFLIFE, build_player_form
from nflp.agents.features.team_form import build_team_form
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)


class PlayerFeatureAgent:
    """Materializa ``features_player`` para los modelos de props."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.version = self.settings.features.version

    def build(self, seasons: list[int] | None = None) -> pd.DataFrame:
        settings = self.settings
        log.info("Construyendo features de jugador (versión %s) ...", self.version)

        games = self.db.query(
            """
            SELECT game_id, season, week, season_type, kickoff_utc, home_team, away_team,
                   stadium_id, roof, surface, home_rest, away_rest, neutral_site,
                   home_score, away_score
            FROM games ORDER BY kickoff_utc, game_id
            """
        )
        games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"], errors="coerce")

        player_stats = self.db.query("SELECT * FROM player_stats_weekly")
        snap_counts = self.db.query("SELECT * FROM snap_counts")
        team_stats = self.db.query("SELECT * FROM team_stats_weekly")
        # El esqueleto de features parte del róster, no de las estadísticas: así
        # se pueden proyectar props de partidos futuros (ver player_form.py).
        rosters = self.db.query("SELECT * FROM rosters_weekly")
        # Metadatos estáticos del catálogo. `pfr_id` solo si la ingesta lo trae:
        # con él el cruce de snaps es por id oficial; sin él, se infiere.
        player_columns = set(self.db.get_columns("players"))
        wanted = [
            c for c in ("gsis_id", "display_name", "draft_round", "rookie_season", "pfr_id")
            if c in player_columns
        ]
        players = self.db.query(f"SELECT {', '.join(wanted)} FROM players")
        injuries = self.db.query("SELECT * FROM injuries")

        if rosters.empty:
            log.warning("Sin datos de róster; se omite el feature store de props.")
            return pd.DataFrame()

        team_form = build_team_form(
            team_stats, games,
            windows=settings.features.rolling_windows,
            halflife=settings.features.ewm_halflife,
        )

        # El corte as-of (kickoff - offset) y el parte de lesiones as-of se
        # resuelven dentro de build_player_form: el QB titular esperado, el rango
        # de profundidad y la oportunidad vacante dependen de ellos.
        frame = build_player_form(
            player_stats, games, snap_counts, team_form, rosters, players,
            window=settings.props.usage_window,
            injuries=injuries,
            as_of_offset_hours=settings.features.as_of_offset_hours,
            ewm_halflife=PLAYER_EWM_HALFLIFE,
        )
        if frame.empty:
            return frame

        # --- contexto de partido: descanso, clima, techo ---
        frame = self._attach_game_context(frame, games)

        # --- metadatos ---
        frame["feature_set_version"] = self.version
        frame["feature_hash"] = hashlib.sha256(
            ",".join(sorted(frame.columns)).encode("utf-8")
        ).hexdigest()[:16]

        # Normaliza los nombres a los de la tabla destino.
        frame = frame.rename(
            columns={
                f"target_share_r{settings.props.usage_window}": "target_share_r5",
                f"air_yards_share_r{settings.props.usage_window}": "air_yards_share_r5",
                f"wopr_r{settings.props.usage_window}": "wopr_r5",
                f"snap_pct_r{settings.props.usage_window}": "snap_pct_r5",
                f"carry_share_r{settings.props.usage_window}": "carry_share_r5",
                f"receiving_yards_r{settings.props.usage_window}": "receiving_yards_r5",
                f"receptions_r{settings.props.usage_window}": "receptions_r5",
                f"targets_r{settings.props.usage_window}": "targets_r5",
                f"receiving_epa_r{settings.props.usage_window}": "receiving_epa_r5",
                f"racr_r{settings.props.usage_window}": "racr_r5",
                f"yac_r{settings.props.usage_window}": "yac_r5",
                f"rushing_yards_r{settings.props.usage_window}": "rushing_yards_r5",
                f"carries_r{settings.props.usage_window}": "carries_r5",
                f"rushing_epa_r{settings.props.usage_window}": "rushing_epa_r5",
                f"ypc_r{settings.props.usage_window}": "ypc_r5",
                # Ampliación: recepciones, touchdowns, yardas de pase, sacks.
                f"receiving_tds_r{settings.props.usage_window}": "receiving_tds_r5",
                f"rushing_tds_r{settings.props.usage_window}": "rushing_tds_r5",
                f"passing_yards_r{settings.props.usage_window}": "passing_yards_r5",
                f"passing_tds_r{settings.props.usage_window}": "passing_tds_r5",
                f"passing_epa_r{settings.props.usage_window}": "passing_epa_r5",
                f"attempts_r{settings.props.usage_window}": "passing_attempts_r5",
                f"completions_r{settings.props.usage_window}": "completions_r5",
                f"sacks_suffered_r{settings.props.usage_window}": "sacks_suffered_r5",
            }
        )
        frame["routes_proxy_r5"] = frame.get("targets_r5")

        if seasons:
            frame = frame[frame["season"].isin(seasons)]

        target_columns = self.db.get_columns("features_player")
        output = frame[[c for c in target_columns if c in frame.columns]].copy()
        output = output.drop_duplicates(
            subset=["game_id", "gsis_id", "feature_set_version"], keep="last"
        )

        # Auditoría anti-leakage antes de escribir nada (mismo contrato que el
        # feature store de partido).
        report = audit_features(output, kickoff=frame.loc[output.index, "kickoff_utc"])
        if not report.is_clean:
            raise RuntimeError(report.describe())
        log.info(report.describe())

        rows = self.db.write_df(
            output, "features_player", mode="upsert",
            key_columns=["game_id", "gsis_id", "feature_set_version"],
        )
        log.info("Feature store de jugador: %d filas.", rows)
        return output

    def _attach_game_context(self, frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        """Añade descanso, clima y techo a cada fila de jugador."""
        weather = self.db.query(
            "SELECT game_id, temperature_c, wind_kph, precipitation_mm FROM weather"
        )
        context = games[["game_id", "roof", "home_rest", "away_rest"]].merge(
            weather, on="game_id", how="left"
        )
        frame = frame.merge(context, on="game_id", how="left")

        frame["rest_days"] = np.where(
            frame["is_home"], frame["home_rest"], frame["away_rest"]
        )
        frame["roof_is_closed"] = is_closed_roof(frame["roof"])
        frame["is_adverse_weather"] = is_adverse_weather(
            frame["temperature_c"], frame["wind_kph"], frame["precipitation_mm"]
        )
        # Bajo techo, el clima no aplica.
        closed = frame["roof_is_closed"].fillna(False)
        frame.loc[closed, ["temperature_c", "wind_kph"]] = np.nan
        frame.loc[closed, "is_adverse_weather"] = False

        return frame

    def run(self, seasons: list[int] | None = None) -> pd.DataFrame:
        return self.build(seasons)
