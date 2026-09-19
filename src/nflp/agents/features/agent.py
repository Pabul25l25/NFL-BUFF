"""Feature Engineering Agent.

Ensambla el feature store a partir de los datos crudos. Todas las features
derivadas del rendimiento pasan por las primitivas ``shifted_*``, que excluyen el
partido objetivo; el contexto (descanso, viaje, altitud, superficie) se conoce al
publicarse el calendario; el clima y las lesiones se cortan en ``as_of_ts``.

Familias de features construidas:

* **Contexto** — divisional, sede neutral, playoffs, tramo de temporada, techo,
  superficie, altitud.
* **Descanso y viaje** — días de descanso, semana corta, salida de bye, distancia
  recorrida por el visitante (haversine) y desfase horario.
* **Clima** — temperatura, viento, precipitación y humedad, solo al aire libre.
* **Forma** — medias móviles de EPA por fase, puntos, CPOE, tasas de sack,
  margen de balón y ritmo, en varias ventanas más una exponencial.
* **Matchup** — diferenciales ataque-vs-defensa entre ambos equipos.
* **Elo** — rating previo de cada equipo.
* **Lesiones** — índice ponderado por gravedad e importancia posicional.
* **Estilo** — clúster K-means como feature auxiliar.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from nflp.agents.data.normalize import is_closed_roof, is_turf
from nflp.agents.data.stadiums import (
    HIGH_ALTITUDE_THRESHOLD_M,
    haversine_km,
    home_stadium_by_team,
)
from nflp.agents.data.weather import is_adverse_weather
from nflp.agents.features import elo as elo_module
from nflp.agents.features.clustering import build_style_clusters
from nflp.agents.features.contracts import audit_features
from nflp.agents.features.injuries import build_injury_index
from nflp.agents.features.team_form import build_team_form
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Columnas de forma que se replican con prefijo home_/away_.
_FORM_COLUMNS = {
    "epa_off_r3": "epa_off_r3",
    "epa_off_r5": "epa_off_r5",
    "epa_off_r8": "epa_off_r8",
    "epa_def_r3": "epa_def_r3",
    "epa_def_r5": "epa_def_r5",
    "epa_def_r8": "epa_def_r8",
    "pass_epa_off_r5": "pass_epa_off_r5",
    "rush_epa_off_r5": "rush_epa_off_r5",
    "pass_epa_def_r5": "pass_epa_def_r5",
    "rush_epa_def_r5": "rush_epa_def_r5",
    "epa_off_ewm": "epa_off_ewm",
    "epa_def_ewm": "epa_def_ewm",
    "points_for_r5": "points_for_r5",
    "points_against_r5": "points_against_r5",
    "margin_r5": "margin_r5",
    "cpoe_r5": "cpoe_r5",
    "sack_rate_off_r5": "sack_rate_off_r5",
    "sack_rate_def_r5": "sack_rate_def_r5",
    "turnover_margin_r5": "turnover_margin_r5",
    "pass_rate_r5": "pass_rate_r5",
}


class FeatureAgent:
    """Construye y materializa el feature store a nivel de partido."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.version = self.settings.features.version

    # ------------------------------------------------------------- carga

    def _load_games(self, seasons: list[int] | None) -> pd.DataFrame:
        clause = ""
        if seasons:
            clause = f"WHERE season IN ({', '.join(str(s) for s in seasons)})"
        games = self.db.query(
            f"""
            SELECT game_id, season, week, season_type, kickoff_utc, gameday,
                   home_team, away_team, stadium_id, roof, surface, div_game,
                   neutral_site, home_rest, away_rest, home_score, away_score
            FROM games {clause}
            ORDER BY kickoff_utc, game_id
            """
        )
        games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"], errors="coerce")
        return games

    # ------------------------------------------------------- contexto

    def _context_features(self, games: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
        frame = games.merge(
            stadiums[["stadium_id", "latitude", "longitude", "altitude_m", "timezone"]],
            on="stadium_id",
            how="left",
        )

        frame["is_divisional"] = frame["div_game"].fillna(False).astype(bool)
        frame["is_neutral_site"] = frame["neutral_site"].fillna(False).astype(bool)
        frame["is_postseason"] = frame["season_type"].eq("POST")
        frame["roof_is_closed"] = is_closed_roof(frame["roof"])
        frame["surface_is_turf"] = is_turf(frame["surface"])
        frame["is_high_altitude"] = (
            frame["altitude_m"].fillna(0) >= HIGH_ALTITUDE_THRESHOLD_M
        )

        # Tramo de temporada: el comportamiento de los equipos cambia entre el
        # arranque (plantillas sin rodar) y el tramo final (equipos eliminados
        # que descansan titulares).
        conditions = [
            frame["season_type"].eq("POST"),
            frame["week"] <= 4,
            frame["week"] <= 12,
        ]
        frame["week_phase"] = np.select(
            conditions, ["playoffs", "early", "mid"], default="late"
        )
        return frame

    # -------------------------------------------------- descanso y viaje

    def _rest_and_travel(self, frame: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
        frame["home_rest_days"] = pd.to_numeric(frame["home_rest"], errors="coerce")
        frame["away_rest_days"] = pd.to_numeric(frame["away_rest"], errors="coerce")
        frame["rest_advantage"] = frame["home_rest_days"] - frame["away_rest_days"]

        # Semana corta = jugar en jueves tras domingo (4 días o menos).
        frame["home_short_week"] = frame["home_rest_days"].le(4).fillna(False)
        frame["away_short_week"] = frame["away_rest_days"].le(4).fillna(False)
        # Salir de bye = 13 días o más de descanso.
        frame["home_off_bye"] = frame["home_rest_days"].ge(13).fillna(False)
        frame["away_off_bye"] = frame["away_rest_days"].ge(13).fillna(False)

        # --- distancia de viaje del visitante ---
        coords = stadiums.set_index("stadium_id")[["latitude", "longitude", "timezone"]]
        team_home = home_stadium_by_team()

        def travel_row(row: pd.Series) -> tuple[float, float]:
            origin_id = team_home.get(row["away_team"])
            if origin_id is None or origin_id not in coords.index:
                return float("nan"), float("nan")
            origin = coords.loc[origin_id]
            distance = haversine_km(
                float(origin["latitude"]), float(origin["longitude"]),
                float(row["latitude"]), float(row["longitude"]),
            )
            shift = _timezone_shift_hours(str(origin["timezone"]), str(row["timezone"]))
            return distance, shift

        travel = frame.apply(travel_row, axis=1, result_type="expand")
        frame["away_travel_km"] = travel[0]
        frame["away_tz_shift_hours"] = travel[1]
        return frame

    # ------------------------------------------------------------ clima

    def _weather_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        weather = self.db.query(
            """
            SELECT game_id, temperature_c, wind_kph, precipitation_mm,
                   humidity_pct, source AS weather_source
            FROM weather
            """
        )
        frame = frame.merge(weather, on="game_id", how="left")

        for column in ("temperature_c", "wind_kph", "precipitation_mm", "humidity_pct"):
            if column not in frame.columns:
                frame[column] = np.nan

        frame["is_adverse_weather"] = is_adverse_weather(
            frame["temperature_c"], frame["wind_kph"], frame["precipitation_mm"]
        )
        # Bajo techo el clima no aplica: se fuerza a nulo para que el modelo no
        # confunda "sin dato" con "buen tiempo".
        closed = frame["roof_is_closed"].fillna(False)
        frame.loc[closed, ["temperature_c", "wind_kph", "precipitation_mm", "humidity_pct"]] = np.nan
        frame.loc[closed, "is_adverse_weather"] = False
        return frame

    # ------------------------------------------------------------- run

    def build(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Construye el feature store a nivel de partido y lo materializa."""
        settings = self.settings
        log.info("Feature Agent: construyendo features (versión %s) ...", self.version)

        games = self._load_games(seasons)
        if games.empty:
            log.warning("No hay partidos que procesar.")
            return pd.DataFrame()

        stadiums = self.db.query("SELECT * FROM stadiums")

        # --- corte as-of: el invariante central ---
        offset = pd.Timedelta(hours=settings.features.as_of_offset_hours)
        games["as_of_ts"] = games["kickoff_utc"] - offset

        # --- forma de equipo (siempre sobre TODO el histórico) ---
        # Aunque solo se pidan unas temporadas, las medias móviles necesitan el
        # histórico completo para no truncar la ventana en la primera semana.
        team_stats = self.db.query("SELECT * FROM team_stats_weekly")
        all_games = self._load_games(None)
        team_form = build_team_form(
            team_stats,
            all_games,
            windows=settings.features.rolling_windows,
            halflife=settings.features.ewm_halflife,
        )

        # --- clústeres de estilo ---
        clusters = build_style_clusters(
            team_form,
            n_clusters=settings.features.style_clusters,
            random_state=settings.modeling.random_seed,
            min_prior_games=settings.features.min_prior_games,
        )
        if not clusters.empty:
            self.db.write_df(clusters, "team_style_clusters", mode="replace")

        # --- Elo ---
        elo_module.build_and_store(self.db)
        elo = self.db.query("SELECT game_id, team, elo_pre FROM elo_ratings")

        # --- contexto, descanso, viaje, clima ---
        frame = self._context_features(games, stadiums)
        frame = self._rest_and_travel(frame, stadiums)
        frame = self._weather_features(frame)

        # --- unión de la forma de cada equipo ---
        form_columns = ["game_id", "team", "prior_games", "epa_off_by_venue_r8"]
        form_columns += [c for c in _FORM_COLUMNS if c in team_form.columns]
        form = team_form[form_columns]

        frame = frame.merge(
            form.add_prefix("home_").rename(columns={"home_game_id": "game_id", "home_team": "home_team"}),
            on=["game_id", "home_team"],
            how="left",
        )
        frame = frame.merge(
            form.add_prefix("away_").rename(columns={"away_game_id": "game_id", "away_team": "away_team"}),
            on=["game_id", "away_team"],
            how="left",
        )

        frame = frame.rename(
            columns={
                "home_epa_off_by_venue_r8": "home_epa_off_at_home_r8",
                "away_epa_off_by_venue_r8": "away_epa_off_at_away_r8",
            }
        )

        # --- Elo por bando ---
        home_elo = elo.rename(columns={"team": "home_team", "elo_pre": "home_elo_pre"})
        away_elo = elo.rename(columns={"team": "away_team", "elo_pre": "away_elo_pre"})
        frame = frame.merge(home_elo, on=["game_id", "home_team"], how="left")
        frame = frame.merge(away_elo, on=["game_id", "away_team"], how="left")
        frame["elo_diff"] = frame["home_elo_pre"] - frame["away_elo_pre"]

        # --- lesiones con corte as-of ---
        injuries = self.db.query("SELECT * FROM injuries")
        injury_index = build_injury_index(
            injuries,
            frame[["game_id", "season", "week", "as_of_ts", "home_team", "away_team"]],
        )
        if not injury_index.empty:
            home_inj = injury_index.rename(
                columns={
                    "team": "home_team", "injury_index": "home_injury_index",
                    "starters_out": "home_starters_out", "qb_questionable": "home_qb_questionable",
                }
            )
            away_inj = injury_index.rename(
                columns={
                    "team": "away_team", "injury_index": "away_injury_index",
                    "starters_out": "away_starters_out", "qb_questionable": "away_qb_questionable",
                }
            )
            frame = frame.merge(home_inj, on=["game_id", "home_team"], how="left")
            frame = frame.merge(away_inj, on=["game_id", "away_team"], how="left")

        for column in (
            "home_injury_index", "away_injury_index",
            "home_starters_out", "away_starters_out",
        ):
            if column not in frame.columns:
                frame[column] = 0.0
            frame[column] = frame[column].fillna(0.0)
        for column in ("home_qb_questionable", "away_qb_questionable"):
            if column not in frame.columns:
                frame[column] = False
            # `astype(object)` antes del fillna evita el downcasting implícito
            # que pandas 2.2 marca como deprecado.
            frame[column] = frame[column].astype(object).where(frame[column].notna(), False).astype(bool)

        frame["injury_index_diff"] = frame["home_injury_index"] - frame["away_injury_index"]

        # --- clústeres por bando ---
        if not clusters.empty:
            home_cluster = clusters.rename(
                columns={"team": "home_team", "cluster_id": "home_style_cluster"}
            )[["season", "week", "home_team", "home_style_cluster"]]
            away_cluster = clusters.rename(
                columns={"team": "away_team", "cluster_id": "away_style_cluster"}
            )[["season", "week", "away_team", "away_style_cluster"]]
            frame = frame.merge(home_cluster, on=["season", "week", "home_team"], how="left")
            frame = frame.merge(away_cluster, on=["season", "week", "away_team"], how="left")
        for column in ("home_style_cluster", "away_style_cluster"):
            if column not in frame.columns:
                frame[column] = -1
            frame[column] = frame[column].fillna(-1).astype(int)
        frame["style_mismatch"] = frame["home_style_cluster"].ne(frame["away_style_cluster"])

        # --- diferenciales de matchup ---
        frame["epa_off_diff"] = frame["home_epa_off_r5"] - frame["away_epa_off_r5"]
        frame["epa_def_diff"] = frame["home_epa_def_r5"] - frame["away_epa_def_r5"]
        # EPA neto: ataque propio menos lo que concede la defensa. Más alto = mejor.
        frame["net_epa_diff"] = (
            (frame["home_epa_off_r5"] - frame["home_epa_def_r5"])
            - (frame["away_epa_off_r5"] - frame["away_epa_def_r5"])
        )
        frame["margin_diff_r5"] = frame["home_margin_r5"] - frame["away_margin_r5"]
        # Emparejamiento específico: pase local contra defensa aérea visitante.
        frame["pass_matchup_diff"] = frame["home_pass_epa_off_r5"] - frame["away_pass_epa_def_r5"]
        frame["rush_matchup_diff"] = frame["home_rush_epa_off_r5"] - frame["away_rush_epa_def_r5"]

        # --- metadatos y persistencia ---
        frame["feature_set_version"] = self.version
        frame["feature_hash"] = _hash_features(frame)

        target_columns = self.db.get_columns("features_game")
        output = frame[[c for c in target_columns if c in frame.columns]].copy()

        # Auditoría anti-leakage antes de escribir nada.
        report = audit_features(output, kickoff=frame["kickoff_utc"])
        if not report.is_clean:
            raise RuntimeError(report.describe())
        log.info(report.describe())

        rows = self.db.write_df(
            output,
            "features_game",
            mode="upsert",
            key_columns=["game_id", "feature_set_version"],
        )
        log.info("Feature store: %d filas escritas (versión %s).", rows, self.version)
        return output

    def run(self, seasons: list[int] | None = None) -> pd.DataFrame:
        return self.build(seasons)


def _timezone_shift_hours(origin_tz: str, destination_tz: str) -> float:
    """Diferencia horaria entre dos zonas, en horas.

    Negativo = el visitante viaja hacia el este y "pierde" horas, el caso más
    penalizado empíricamente (equipos de la costa oeste jugando a las 13:00 ET).
    """
    if not origin_tz or not destination_tz or origin_tz == destination_tz:
        return 0.0
    try:
        reference = pd.Timestamp("2024-10-15 13:00")
        origin_offset = reference.tz_localize(origin_tz).utcoffset()
        destination_offset = reference.tz_localize(destination_tz).utcoffset()
    except Exception:  # noqa: BLE001 - zona desconocida
        return float("nan")
    if origin_offset is None or destination_offset is None:
        return float("nan")
    return (destination_offset - origin_offset).total_seconds() / 3600.0


def _hash_features(frame: pd.DataFrame) -> pd.Series:
    """Hash estable del conjunto de columnas, para trazar versiones."""
    signature = ",".join(sorted(frame.columns))
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return pd.Series(digest, index=frame.index)
