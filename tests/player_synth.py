"""Liga sintética mínima para probar el feature store de jugador.

No es un módulo de pruebas (no empieza por ``test_``): lo importan
``test_leakage.py`` y ``test_player_features.py``.

Diseño: 4 equipos, temporadas 2022 y 2023 de 8 semanas. La semana 8 de 2023 no
se ha jugado (sin estadísticas ni snaps), para comprobar que las filas futuras
existen. Escenarios fijos que las pruebas verifican:

* ``A_WR1`` pasa a reserva (RES) en 2023 semana 5 → oportunidad vacante en A.
* ``A_QB1`` consta Out (parte publicado antes del corte) en 2023 semana 6 →
  el titular esperado de A es ``A_QB2``.
* ``A_RB1`` consta Out en 2023 semana 4 pero con un parte publicado DESPUÉS del
  corte → debe ignorarse.
* ``C_WR1`` consta Out (antes del corte) en 2023 semana 4 pero aun así tiene
  estadísticas: dato incoherente a propósito, para que una media no desplazada
  en la oportunidad vacante sea detectable por la prueba de perturbación.
* ``A_WR2`` figura como "Michael Testman" en el catálogo y "Mike Testman" en
  snap_counts → el cruce de snaps debe resolverlo.
* ``B_WR3`` es novato de primera ronda en 2022 y ``A_ROOK`` en 2023 → prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TEAMS = ["A", "B", "C", "D"]
SLOTS = {
    "QB1": "QB", "QB2": "QB", "WR1": "WR", "WR2": "WR", "WR3": "WR",
    "TE1": "TE", "RB1": "RB", "RB2": "RB",
}
SEASONS = (2022, 2023)
WEEKS = range(1, 9)
FUTURE = (2023, 8)
PAIRINGS = [(("A", "B"), ("C", "D")), (("A", "C"), ("B", "D")), (("A", "D"), ("B", "C"))]
AS_OF_OFFSET_HOURS = 1.0


@dataclass
class League:
    games: pd.DataFrame
    rosters: pd.DataFrame
    players: pd.DataFrame
    player_stats: pd.DataFrame
    snap_counts: pd.DataFrame
    injuries: pd.DataFrame
    team_form: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame({"game_id": pd.Series(dtype=str), "team": pd.Series(dtype=str)})
    )


def _game_id(season: int, week: int, away: str, home: str) -> str:
    return f"{season}_{week:02d}_{away}_{home}"


def build_league(seed: int = 7) -> League:
    rng = np.random.default_rng(seed)

    games_rows: list[dict] = []
    for season in SEASONS:
        start = datetime(season, 9, 8, 17, 0)
        for week in WEEKS:
            kickoff = start + timedelta(days=7 * (week - 1))
            for home, away in PAIRINGS[(week - 1) % len(PAIRINGS)]:
                games_rows.append(
                    {
                        "game_id": _game_id(season, week, away, home), "season": season,
                        "week": week, "season_type": "REG", "kickoff_utc": kickoff,
                        "home_team": home, "away_team": away,
                    }
                )
    games = pd.DataFrame(games_rows)

    # --- catálogo ---
    player_rows: list[dict] = []
    for team in TEAMS:
        for slot, position in SLOTS.items():
            gsis = f"{team}_{slot}"
            player_rows.append(
                {
                    "gsis_id": gsis, "display_name": f"{team} {slot} Player",
                    "position": position, "draft_round": 2.0, "rookie_season": 2015,
                }
            )
    player_rows.append(
        {"gsis_id": "A_ROOK", "display_name": "Rookie Alpha", "position": "WR",
         "draft_round": 1.0, "rookie_season": 2023}
    )
    players = pd.DataFrame(player_rows)
    players.loc[players["gsis_id"] == "A_WR2", "display_name"] = "Michael Testman"
    players.loc[players["gsis_id"] == "B_WR3", ["draft_round", "rookie_season"]] = [1.0, 2022]

    # --- rósters semanales ---
    roster_rows: list[dict] = []
    for season in SEASONS:
        for week in WEEKS:
            for team in TEAMS:
                members = [(f"{team}_{slot}", position) for slot, position in SLOTS.items()]
                if team == "A" and season == 2023:
                    members.append(("A_ROOK", "WR"))
                for gsis, position in members:
                    status = "ACT"
                    if gsis == "A_WR1" and season == 2023 and week >= 5:
                        status = "RES"
                    roster_rows.append(
                        {"season": season, "week": week, "season_type": "REG", "team": team,
                         "gsis_id": gsis, "position": position, "depth_chart_pos": position,
                         "status": status}
                    )
    rosters = pd.DataFrame(roster_rows)

    # --- estadísticas (solo partidos jugados) ---
    active = rosters[rosters["status"] == "ACT"]
    long_games = pd.concat(
        [
            games.assign(team=games["home_team"], opponent_team=games["away_team"]),
            games.assign(team=games["away_team"], opponent_team=games["home_team"]),
        ],
        ignore_index=True,
    )
    played = long_games[
        ~((long_games["season"] == FUTURE[0]) & (long_games["week"] == FUTURE[1]))
    ]
    rows = played.merge(active, on=["season", "week", "season_type", "team"])
    # A_QB1 no juega la semana 6 de 2023 (Out).
    rows = rows[~((rows["gsis_id"] == "A_QB1") & (rows["season"] == 2023) & (rows["week"] == 6))]
    rows = rows.reset_index(drop=True)
    n = len(rows)
    is_pos = {p: rows["position"].eq(p).to_numpy() for p in ("QB", "WR", "TE", "RB")}
    starter_qb = rows["gsis_id"].str.endswith("QB1").to_numpy() | (
        (rows["gsis_id"] == "A_QB2") & (rows["season"] == 2023) & (rows["week"] == 6)
    ).to_numpy()

    targets = np.where(is_pos["WR"] | is_pos["TE"], rng.integers(1, 12, n), 0) + np.where(
        is_pos["RB"], rng.integers(0, 5, n), 0
    )
    receptions = np.minimum(targets, rng.integers(0, 10, n))
    carries = np.where(is_pos["RB"], rng.integers(3, 22, n), 0) + np.where(
        is_pos["QB"] & starter_qb, rng.integers(0, 5, n), 0
    )
    attempts = np.where(is_pos["QB"], np.where(starter_qb, rng.integers(25, 45, n), rng.integers(0, 3, n)), 0)
    stats = pd.DataFrame(
        {
            "player_id": rows["gsis_id"], "season": rows["season"], "week": rows["week"],
            "season_type": "REG", "game_id": rows["game_id"], "team": rows["team"],
            "opponent_team": rows["opponent_team"], "position": rows["position"],
            "targets": targets.astype(float), "receptions": receptions.astype(float),
            "receiving_yards": receptions * rng.uniform(5, 15, n),
            "receiving_air_yards": targets * rng.uniform(4, 12, n),
            "receiving_yards_after_catch": receptions * rng.uniform(1, 5, n),
            "receiving_epa": rng.normal(0, 2, n), "receiving_tds": rng.integers(0, 2, n).astype(float),
            "carries": carries.astype(float), "rushing_yards": carries * rng.uniform(2, 6, n),
            "rushing_epa": rng.normal(0, 2, n), "rushing_tds": rng.integers(0, 2, n).astype(float),
            "attempts": attempts.astype(float),
            "completions": (attempts * rng.uniform(0.5, 0.75, n)).round(),
            "passing_yards": attempts * rng.uniform(5, 9, n),
            "passing_tds": np.where(attempts > 0, rng.integers(0, 4, n), 0).astype(float),
            "passing_epa": np.where(attempts > 0, rng.normal(2, 8, n), 0.0),
            "sacks_suffered": np.where(attempts > 0, rng.integers(0, 5, n), 0).astype(float),
        }
    )
    team_targets = stats.groupby(["game_id", "team"])["targets"].transform("sum")
    stats["target_share"] = np.where(team_targets > 0, stats["targets"] / team_targets, np.nan)
    team_air = stats.groupby(["game_id", "team"])["receiving_air_yards"].transform("sum")
    stats["air_yards_share"] = np.where(team_air > 0, stats["receiving_air_yards"] / team_air, np.nan)
    stats["wopr"] = 1.5 * stats["target_share"] + 0.7 * stats["air_yards_share"]
    stats["racr"] = np.where(
        stats["receiving_air_yards"] > 0, stats["receiving_yards"] / stats["receiving_air_yards"], np.nan
    )

    # --- snaps (id PFR, nombre con apodo para A_WR2) ---
    names = players.set_index("gsis_id")["display_name"]
    snaps = stats[["game_id", "player_id", "team", "season", "week", "position"]].copy()
    snaps["pfr_player_id"] = "pfr_" + snaps["player_id"]
    snaps["player"] = snaps["player_id"].map(names)
    snaps.loc[snaps["player_id"] == "A_WR2", "player"] = "Mike Testman"
    snaps["season_type"] = "REG"
    snaps["offense_pct"] = rng.uniform(0.2, 1.0, len(snaps))
    snaps["offense_snaps"] = (snaps["offense_pct"] * 65).round()
    snaps = snaps.drop(columns=["player_id"])

    # --- partes de lesiones ---
    def as_of(season: int, week: int, team: str) -> pd.Timestamp:
        row = games[
            (games["season"] == season) & (games["week"] == week)
            & ((games["home_team"] == team) | (games["away_team"] == team))
        ].iloc[0]
        return pd.Timestamp(row["kickoff_utc"]) - pd.Timedelta(hours=AS_OF_OFFSET_HOURS)

    injuries = pd.DataFrame(
        [
            {"season": 2023, "week": 6, "season_type": "REG", "team": "A", "gsis_id": "A_QB1",
             "position": "QB", "report_status": "Out", "practice_status": "Did Not Participate In Practice",
             "date_modified": as_of(2023, 6, "A") - pd.Timedelta(days=2)},
            {"season": 2023, "week": 4, "season_type": "REG", "team": "C", "gsis_id": "C_WR1",
             "position": "WR", "report_status": "Out", "practice_status": None,
             "date_modified": as_of(2023, 4, "C") - pd.Timedelta(days=1)},
            {"season": 2023, "week": 4, "season_type": "REG", "team": "A", "gsis_id": "A_RB1",
             "position": "RB", "report_status": "Out", "practice_status": None,
             "date_modified": as_of(2023, 4, "A") + pd.Timedelta(minutes=5)},
        ]
    )

    return League(
        games=games, rosters=rosters, players=players, player_stats=stats,
        snap_counts=snaps, injuries=injuries,
    )


def build_features(league: League) -> pd.DataFrame:
    from nflp.agents.features.player_form import build_player_form

    def like_duckdb(frame: pd.DataFrame) -> pd.DataFrame:
        # DuckDB -> pandas entrega VARCHAR como dtype "string" e INTEGER nulables
        # como "Int64": la prueba debe usar los mismos tipos que producción.
        out = frame.copy()
        for column in out.columns:
            if out[column].dtype == object:
                out[column] = out[column].astype("string")
        for column in ("draft_round", "rookie_season"):
            if column in out.columns:
                out[column] = out[column].astype("Int64")
        return out

    return build_player_form(
        like_duckdb(league.player_stats), league.games, like_duckdb(league.snap_counts),
        league.team_form, like_duckdb(league.rosters), like_duckdb(league.players),
        window=5, injuries=like_duckdb(league.injuries),
        as_of_offset_hours=AS_OF_OFFSET_HOURS,
    )


def perturb_from(league: League, season: int, week: int, seed: int = 99) -> League:
    """Copia de la liga con TODO lo ocurrido desde (season, week) alterado.

    Cambia estadísticas y snaps de esos partidos y añade un parte de lesiones de
    esa semana publicado después del corte. Nada de eso puede mover una feature
    de una fila con fecha <= (season, week).
    """
    rng = np.random.default_rng(seed)
    stats = league.player_stats.copy()
    later = (stats["season"] > season) | ((stats["season"] == season) & (stats["week"] >= week))
    numeric = [
        c for c in stats.columns
        if c not in {"season", "week"} and pd.api.types.is_numeric_dtype(stats[c])
    ]
    stats.loc[later, numeric] = (
        stats.loc[later, numeric] * rng.uniform(0.1, 5.0, (int(later.sum()), len(numeric))) + 3.0
    )
    snaps = league.snap_counts.copy()
    snap_later = (snaps["season"] > season) | ((snaps["season"] == season) & (snaps["week"] >= week))
    snaps.loc[snap_later, "offense_pct"] = rng.uniform(0, 1, int(snap_later.sum()))

    target_games = league.games[(league.games["season"] == season) & (league.games["week"] == week)]
    kickoff = pd.Timestamp(target_games["kickoff_utc"].iloc[0])
    late_reports = pd.DataFrame(
        [
            {"season": season, "week": week, "season_type": "REG", "team": team,
             "gsis_id": f"{team}_{slot}", "position": SLOTS[slot], "report_status": "Out",
             "practice_status": "Did Not Participate In Practice",
             # Publicado justo en el corte: `<` estricto lo excluye.
             "date_modified": kickoff - pd.Timedelta(hours=AS_OF_OFFSET_HOURS)}
            for team in TEAMS for slot in ("QB1", "WR1", "RB1")
        ]
    )
    injuries = pd.concat([league.injuries, late_reports], ignore_index=True)
    return League(
        games=league.games, rosters=league.rosters, players=league.players,
        player_stats=stats, snap_counts=snaps, injuries=injuries, team_form=league.team_form,
    )
