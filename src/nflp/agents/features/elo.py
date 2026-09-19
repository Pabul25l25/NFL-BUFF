"""Ratings Elo — baseline y feature a la vez.

Elo cumple dos funciones en este proyecto:

1. **Baseline obligatorio** del backtest (sección 4 del prompt). Es un listón
   sorprendentemente duro: un Elo bien calibrado acierta ~63 % de los partidos y
   supera a muchos modelos de ML mal validados.
2. **Feature legítima** para los modelos supervisados. No hay leakage: el rating
   *previo* de cada equipo se calcula únicamente con partidos ya jugados.

Implementación al estilo FiveThirtyEight:

* Ventaja de campo expresada en puntos Elo.
* Multiplicador por margen de victoria, amortiguado para evitar que las palizas
  distorsionen el rating.
* Regresión a la media entre temporadas (los equipos cambian de plantilla).

El cálculo es estrictamente cronológico y secuencial: cada partido actualiza los
ratings que usará el siguiente. Por eso se materializa en ``elo_ratings`` y no se
recalcula en cada iteración del backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

INITIAL_RATING = 1500.0


@dataclass(frozen=True)
class EloConfig:
    """Parámetros del sistema Elo."""

    k_factor: float = 20.0
    home_advantage: float = 55.0        # ~2,2 puntos de spread
    mean_reversion: float = 0.25        # fracción que revierte a la media entre temporadas
    reversion_target: float = 1505.0
    mov_multiplier: bool = True         # ajuste por margen de victoria
    playoff_multiplier: float = 1.2     # los playoffs pesan algo más


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilidad de que A gane a B según la diferencia de rating."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(margin: float, rating_diff: float) -> float:
    """Amortiguador por margen de victoria (fórmula de FiveThirtyEight).

    Sin él, una paliza puntual dispararía el rating; el denominador reduce el
    efecto cuando el favorito ya era muy superior.
    """
    return float(np.log(abs(margin) + 1.0) * (2.2 / (rating_diff * 0.001 + 2.2)))


def compute_elo(games: pd.DataFrame, config: EloConfig | None = None) -> pd.DataFrame:
    """Calcula los ratings Elo partido a partido, en orden cronológico.

    Args:
        games: debe contener ``game_id``, ``season``, ``week``, ``season_type``,
            ``kickoff_utc``, ``home_team``, ``away_team``, ``home_score``,
            ``away_score``. Los partidos sin resultado no actualizan ratings pero
            sí reciben su rating previo (necesario para predecir).

    Returns:
        Un DataFrame en formato largo: una fila por equipo y partido, con
        ``elo_pre`` y ``elo_post``.
    """
    config = config or EloConfig()

    required = {
        "game_id", "season", "week", "season_type",
        "home_team", "away_team", "home_score", "away_score",
    }
    missing = required - set(games.columns)
    if missing:
        raise KeyError(f"compute_elo: faltan columnas {sorted(missing)}")

    ordered = games.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    last_season: int | None = None
    records: list[dict[str, object]] = []

    for row in ordered.itertuples(index=False):
        season = int(row.season)

        # --- regresión a la media al cambiar de temporada ---
        if last_season is not None and season != last_season:
            for team in list(ratings):
                ratings[team] = (
                    ratings[team] * (1 - config.mean_reversion)
                    + config.reversion_target * config.mean_reversion
                )
        last_season = season

        home, away = row.home_team, row.away_team
        home_pre = ratings.get(home, INITIAL_RATING)
        away_pre = ratings.get(away, INITIAL_RATING)

        # Sede neutral: sin ventaja de campo.
        neutral = bool(getattr(row, "neutral_site", False))
        advantage = 0.0 if neutral else config.home_advantage

        home_score = row.home_score
        away_score = row.away_score
        has_result = pd.notna(home_score) and pd.notna(away_score)

        if has_result:
            margin = float(home_score) - float(away_score)
            if margin > 0:
                actual, rating_diff = 1.0, (home_pre + advantage) - away_pre
            elif margin < 0:
                actual, rating_diff = 0.0, away_pre - (home_pre + advantage)
            else:
                actual, rating_diff = 0.5, 0.0

            expected = expected_score(home_pre + advantage, away_pre)

            k = config.k_factor
            if str(row.season_type).upper() == "POST":
                k *= config.playoff_multiplier
            if config.mov_multiplier and margin != 0:
                k *= margin_multiplier(margin, rating_diff)

            change = k * (actual - expected)
            home_post = home_pre + change
            away_post = away_pre - change
            ratings[home] = home_post
            ratings[away] = away_post
        else:
            # Partido futuro: conserva el rating previo, no actualiza nada.
            home_post, away_post = home_pre, away_pre

        records.append(
            {
                "game_id": row.game_id, "season": season, "week": int(row.week),
                "team": home, "elo_pre": home_pre, "elo_post": home_post, "is_home": True,
            }
        )
        records.append(
            {
                "game_id": row.game_id, "season": season, "week": int(row.week),
                "team": away, "elo_pre": away_pre, "elo_post": away_post, "is_home": False,
            }
        )

    return pd.DataFrame.from_records(records)


def elo_win_probability(
    home_elo: pd.Series, away_elo: pd.Series, *, config: EloConfig | None = None,
    neutral_site: pd.Series | None = None,
) -> pd.Series:
    """Probabilidad de victoria local según Elo (baseline del backtest)."""
    config = config or EloConfig()
    advantage = config.home_advantage
    if neutral_site is not None:
        advantage = np.where(neutral_site.fillna(False), 0.0, advantage)
    diff = (home_elo + advantage) - away_elo
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def elo_to_spread(elo_diff: pd.Series) -> pd.Series:
    """Convierte diferencia de Elo en spread esperado (~25 puntos Elo = 1 punto)."""
    return elo_diff / 25.0


def build_and_store(db: Database | None = None, config: EloConfig | None = None) -> int:
    """Calcula el Elo de todo el histórico y lo materializa en la base de datos."""
    db = db or get_db()
    games = db.query(
        """
        SELECT game_id, season, week, season_type, kickoff_utc,
               home_team, away_team, home_score, away_score, neutral_site
        FROM games
        ORDER BY kickoff_utc, game_id
        """
    )
    if games.empty:
        log.warning("No hay partidos para calcular Elo.")
        return 0

    ratings = compute_elo(games, config)
    rows = db.write_df(ratings, "elo_ratings", mode="replace")
    log.info("Elo calculado para %d partidos (%d filas equipo-partido).", len(games), rows)
    return rows
