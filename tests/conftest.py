"""Fixtures compartidas.

Las pruebas corren **sin acceso a la red** y sobre una base de datos temporal, de
modo que la suite es determinista y rápida. Las pruebas que sí necesitan internet
llevan la marca ``@pytest.mark.network`` y se omiten por defecto.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflp.core.config import Settings, load_settings
from nflp.core.db import Database
from nflp.core.migrations import migrate

# Liga completa: 32 equipos → 16 partidos por semana, como la NFL real. Con
# menos equipos el dataset sintético queda por debajo del mínimo de filas que
# exige el walk-forward y las pruebas de integración no serían representativas.
TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuración apuntando a un directorio temporal."""
    config = load_settings()
    config.paths.data_dir = tmp_path
    config.paths.raw_dir = tmp_path / "raw"
    config.paths.artifacts_dir = tmp_path / "artifacts"
    config.paths.reports_dir = tmp_path / "reports"
    config.paths.duckdb_path = tmp_path / "test.duckdb"
    config.paths.ensure_dirs()
    config.database.backend = "duckdb"
    config.mlflow.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return config


@pytest.fixture
def db(settings: Settings) -> Database:
    """Base de datos temporal con el esquema aplicado."""
    database = Database(settings)
    migrate(database)
    return database


@pytest.fixture
def synthetic_games() -> pd.DataFrame:
    """Calendario sintético de 4 temporadas con resultados coherentes.

    Se construye con una señal real (cada equipo tiene una fuerza latente) más
    ruido, de modo que un modelo bien construido debe poder aprender algo y las
    pruebas de extremo a extremo sean significativas.
    """
    rng = np.random.default_rng(1729)
    strength = {team: rng.normal(0, 4) for team in TEAMS}

    records: list[dict] = []
    kickoff = datetime(2020, 9, 10, 17, 0)

    for season in (2020, 2021, 2022, 2023):
        for week in range(1, 15):
            shuffled = list(TEAMS)
            rng.shuffle(shuffled)
            for index in range(0, len(shuffled), 2):
                home, away = shuffled[index], shuffled[index + 1]
                margin = (
                    strength[home] - strength[away] + 2.0 + rng.normal(0, 10)
                )
                home_score = int(np.clip(round(22 + margin / 2), 0, 60))
                away_score = int(np.clip(round(22 - margin / 2), 0, 60))

                records.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away}_{home}",
                        "season": season,
                        "week": week,
                        "season_type": "REG",
                        "gameday": kickoff.date(),
                        "gametime": "13:00",
                        "kickoff_utc": kickoff,
                        "weekday": "Sunday",
                        "home_team": home,
                        "away_team": away,
                        "stadium_id": "KAN00",
                        "stadium": "Test Stadium",
                        "roof": "outdoors",
                        "surface": "grass",
                        "div_game": False,
                        "neutral_site": False,
                        "home_rest": 7,
                        "away_rest": 7,
                        "home_qb_id": None, "away_qb_id": None,
                        "home_qb_name": None, "away_qb_name": None,
                        "home_coach": None, "away_coach": None, "referee": None,
                        "home_score": home_score,
                        "away_score": away_score,
                        "result": home_score - away_score,
                        "total": home_score + away_score,
                        "overtime": False,
                    }
                )
                kickoff += timedelta(hours=3)
            kickoff += timedelta(days=7)
        kickoff += timedelta(days=200)

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_team_stats(synthetic_games: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas de equipo coherentes con el calendario sintético."""
    rng = np.random.default_rng(42)
    records: list[dict] = []

    for row in synthetic_games.itertuples(index=False):
        for team, opponent, points in (
            (row.home_team, row.away_team, row.home_score),
            (row.away_team, row.home_team, row.away_score),
        ):
            records.append(
                {
                    "team": team,
                    "season": row.season,
                    "week": row.week,
                    "season_type": "REG",
                    "game_id": row.game_id,
                    "opponent_team": opponent,
                    "attempts": float(rng.integers(25, 45)),
                    "carries": float(rng.integers(18, 35)),
                    "passing_yards": float(rng.normal(240, 60)),
                    "rushing_yards": float(rng.normal(110, 35)),
                    "passing_epa": float(rng.normal(0, 6)) + (points - 22) * 0.2,
                    "rushing_epa": float(rng.normal(0, 3)),
                    "passing_cpoe": float(rng.normal(0, 3)),
                    "sacks_suffered": float(rng.integers(0, 6)),
                    "passing_interceptions": float(rng.integers(0, 3)),
                    "fumbles_lost": float(rng.integers(0, 2)),
                    "def_sacks": float(rng.integers(0, 6)),
                    "def_interceptions": float(rng.integers(0, 3)),
                    "def_fumbles_forced": float(rng.integers(0, 2)),
                    "passing_first_downs": float(rng.integers(8, 20)),
                    "rushing_first_downs": float(rng.integers(3, 12)),
                    "passing_tds": float(rng.integers(0, 4)),
                    "rushing_tds": float(rng.integers(0, 3)),
                    "passing_air_yards": float(rng.normal(250, 60)),
                    "receiving_yards": float(rng.normal(240, 60)),
                    "receiving_epa": float(rng.normal(0, 5)),
                    "def_qb_hits": float(rng.integers(0, 10)),
                    "def_pass_defended": float(rng.integers(0, 10)),
                    "def_tackles_for_loss": float(rng.integers(0, 9)),
                    "def_tds": 0.0,
                    "special_teams_tds": 0.0,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def populated_db(
    db: Database, synthetic_games: pd.DataFrame, synthetic_team_stats: pd.DataFrame
) -> Database:
    """Base de datos temporal con datos sintéticos cargados."""
    from nflp.agents.data.stadiums import stadiums_frame

    db.write_df(stadiums_frame(), "stadiums", mode="replace")
    db.write_df(
        pd.DataFrame({"team_abbr": TEAMS, "team_name": TEAMS}), "teams", mode="replace"
    )
    db.write_df(synthetic_games, "games", mode="append")
    db.write_df(synthetic_team_stats, "team_stats_weekly", mode="append")

    # Líneas de mercado sintéticas, en su tabla aislada.
    rng = np.random.default_rng(7)
    market = pd.DataFrame(
        {
            "game_id": synthetic_games["game_id"],
            "spread_line": synthetic_games["result"] + rng.normal(0, 3, len(synthetic_games)),
            "total_line": synthetic_games["total"] + rng.normal(0, 3, len(synthetic_games)),
            "home_moneyline": -150.0,
            "away_moneyline": 130.0,
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
            "over_odds": -110.0,
            "under_odds": -110.0,
        }
    )
    db.write_df(market, "market_lines", mode="append")
    return db
