"""Forma de equipo: medias móviles de EPA y métricas derivadas.

Construye, para cada equipo y partido, una foto de su rendimiento **antes** de
ese partido. Todas las agregaciones pasan por las primitivas ``shifted_*`` de
`contracts.py`, que excluyen la fila actual: es lo que impide que un equipo
"sepa" cómo le fue en el partido que estamos prediciendo.

Métricas defensivas: nflverse no publica "EPA permitido", así que se deriva
cruzando cada partido consigo mismo — el EPA defensivo de un equipo es el EPA
ofensivo que generó su rival en ese mismo partido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflp.agents.features.contracts import shifted_ewm, shifted_rolling
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Métricas ofensivas que se invierten para obtener la versión defensiva.
_OFFENSIVE_SOURCE = [
    "passing_epa", "rushing_epa", "passing_cpoe", "passing_yards", "rushing_yards",
    "passing_first_downs", "rushing_first_downs", "sacks_suffered",
    "passing_interceptions", "fumbles_lost", "attempts", "carries",
    "def_sacks", "def_interceptions", "def_fumbles_forced",
]


def build_team_game_frame(
    team_stats: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Une estadísticas de equipo con el calendario en formato largo.

    Devuelve una fila por equipo y partido, con sus métricas ofensivas, las
    permitidas (del rival) y los puntos anotados/recibidos.
    """
    stats = team_stats.copy()
    for column in _OFFENSIVE_SOURCE:
        if column not in stats.columns:
            stats[column] = np.nan

    # --- métricas permitidas: las ofensivas del rival en el mismo partido ---
    opponent = stats[["game_id", "team"] + _OFFENSIVE_SOURCE].copy()
    opponent = opponent.rename(
        columns={"team": "opponent_team", **{c: f"opp_{c}" for c in _OFFENSIVE_SOURCE}}
    )
    merged = stats.merge(opponent, on=["game_id", "opponent_team"], how="left")

    # --- puntos y contexto del calendario ---
    schedule = games[
        [
            "game_id", "season", "week", "season_type", "kickoff_utc",
            "home_team", "away_team", "home_score", "away_score", "neutral_site",
        ]
    ].copy()

    home = schedule.rename(
        columns={
            "home_team": "team", "away_team": "opponent_team",
            "home_score": "points_for", "away_score": "points_against",
        }
    )
    home["is_home"] = True
    away = schedule.rename(
        columns={
            "away_team": "team", "home_team": "opponent_team",
            "away_score": "points_for", "home_score": "points_against",
        }
    )
    away["is_home"] = False
    long_schedule = pd.concat([home, away], ignore_index=True)

    frame = long_schedule.merge(
        merged.drop(columns=["season", "week", "season_type", "opponent_team"], errors="ignore"),
        on=["game_id", "team"],
        how="left",
    )

    frame["points_for"] = pd.to_numeric(frame["points_for"], errors="coerce")
    frame["points_against"] = pd.to_numeric(frame["points_against"], errors="coerce")
    frame["margin"] = frame["points_for"] - frame["points_against"]
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], errors="coerce")

    return frame.sort_values(["team", "kickoff_utc", "game_id"]).reset_index(drop=True)


def _add_derived_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Tasas compuestas por partido, antes de promediarlas."""
    attempts = frame["attempts"].fillna(0)
    carries = frame["carries"].fillna(0)
    sacks = frame["sacks_suffered"].fillna(0)

    dropbacks = attempts + sacks
    plays = dropbacks + carries

    frame["plays"] = plays.replace(0, np.nan)
    frame["pass_rate"] = np.where(plays > 0, dropbacks / plays, np.nan)
    frame["sack_rate_off"] = np.where(dropbacks > 0, sacks / dropbacks, np.nan)

    opp_attempts = frame["opp_attempts"].fillna(0)
    opp_sacks = frame["opp_sacks_suffered"].fillna(0)
    opp_dropbacks = opp_attempts + opp_sacks
    frame["sack_rate_def"] = np.where(opp_dropbacks > 0, opp_sacks / opp_dropbacks, np.nan)

    # EPA neto por jugada: combina ataque y defensa en una sola cifra.
    frame["epa_off"] = frame[["passing_epa", "rushing_epa"]].sum(axis=1, min_count=1)
    frame["epa_def"] = frame[["opp_passing_epa", "opp_rushing_epa"]].sum(axis=1, min_count=1)

    # Balón: pérdidas propias frente a robos de la defensa.
    committed = frame["passing_interceptions"].fillna(0) + frame["fumbles_lost"].fillna(0)
    forced = frame["def_interceptions"].fillna(0) + frame["def_fumbles_forced"].fillna(0)
    frame["turnover_margin"] = forced - committed

    return frame


def build_team_form(
    team_stats: pd.DataFrame,
    games: pd.DataFrame,
    *,
    windows: list[int],
    halflife: float,
) -> pd.DataFrame:
    """Calcula todas las features de forma por equipo y partido.

    Todas las medias excluyen el partido actual (``shift(1)``).
    """
    frame = build_team_game_frame(team_stats, games)
    frame = _add_derived_rates(frame)

    keys = ["team"]

    # --- medias móviles por ventana ---
    rolling_specs: dict[str, str] = {
        "epa_off": "epa_off",
        "epa_def": "epa_def",
        "points_for": "points_for",
        "points_against": "points_against",
        "margin": "margin",
        "passing_epa": "pass_epa_off",
        "rushing_epa": "rush_epa_off",
        "opp_passing_epa": "pass_epa_def",
        "opp_rushing_epa": "rush_epa_def",
        "passing_cpoe": "cpoe",
        "sack_rate_off": "sack_rate_off",
        "sack_rate_def": "sack_rate_def",
        "turnover_margin": "turnover_margin",
        "pass_rate": "pass_rate",
        "plays": "plays",
    }

    for source, label in rolling_specs.items():
        for window in windows:
            frame[f"{label}_r{window}"] = shifted_rolling(frame, keys, source, window)

    # --- forma reciente ponderada exponencialmente ---
    frame["epa_off_ewm"] = shifted_ewm(frame, keys, "epa_off", halflife)
    frame["epa_def_ewm"] = shifted_ewm(frame, keys, "epa_def", halflife)

    # --- splits por condición: rendimiento solo como local / solo como visitante ---
    # Se calculan dentro de cada grupo (equipo, condición) para que la media de
    # "local" no se contamine con partidos de visitante.
    frame["epa_off_by_venue_r8"] = shifted_rolling(
        frame.sort_values(["team", "is_home", "kickoff_utc"]),
        ["team", "is_home"],
        "epa_off",
        8,
    ).reindex(frame.index)

    # --- partidos previos disponibles (para filtrar filas con poca historia) ---
    frame["prior_games"] = (
        frame.groupby(keys, observed=True, sort=False).cumcount()
    )

    log.debug("Forma de equipo: %d filas equipo-partido.", len(frame))
    return frame
