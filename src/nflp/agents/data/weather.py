"""Clima por partido vía Open-Meteo (histórico y pronóstico).

Solo se consulta para partidos **al aire libre**: en estadios con techo cerrado el
clima es irrelevante y meter ruido ahí solo degrada el modelo. Los partidos bajo
techo se marcan con ``source='indoor'`` y valores nulos.

**Estrategia de peticiones**: en vez de una llamada por partido (miles), se hace
*una llamada por estadio* cubriendo todo el rango de fechas de sus partidos, y
después se cruza cada partido con la hora de su kickoff. Eso reduce ~7.000
peticiones a ~40, respetando de sobra los límites de la API gratuita.

**Nota de honestidad metodológica**: para partidos pasados se usan observaciones
de archivo, no el pronóstico que existía antes del kickoff. Es una concesión
menor —el pronóstico a 2-3 días es bastante preciso— pero introduce un leve
optimismo en el backtest. Queda marcada en la columna ``source``. Ver
`docs/anti-leakage.md`.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from nflp.agents.data.stadiums import get_stadium
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "relative_humidity_2m",
    "snowfall",
]

# El pronóstico de Open-Meteo alcanza ~16 días; más allá no hay dato.
FORECAST_HORIZON_DAYS = 16

# Umbrales para la bandera de clima adverso (viento fuerte, frío o precipitación).
ADVERSE_WIND_KPH = 25.0
ADVERSE_TEMP_C = 0.0
ADVERSE_PRECIP_MM = 1.0


class WeatherAgent:
    """Descarga y almacena el clima de cada partido al aire libre."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self._last_request = 0.0

    # ------------------------------------------------------------------ http

    def _throttle(self) -> None:
        delay = self.settings.weather.request_delay_seconds
        elapsed = time.monotonic() - self._last_request
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(6),
        # Espera larga: Open-Meteo pondera cada petición por volumen de datos y
        # responde 429 con ventanas de minuto. Reintentar rápido no sirve.
        wait=wait_exponential(multiplier=3, min=5, max=120),
        reraise=True,
    )
    def _request(self, url: str, params: dict[str, object]) -> dict:
        self._throttle()
        with httpx.Client(timeout=self.settings.ingestion.timeout_seconds) as client:
            response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _fetch_hourly(
        self, latitude: float, longitude: float, start: str, end: str, *, forecast: bool
    ) -> pd.DataFrame:
        """Serie horaria de un estadio en un rango de fechas, indexada en UTC."""
        params: dict[str, object] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        if forecast:
            url = self.settings.weather.forecast_url
            params["forecast_days"] = FORECAST_HORIZON_DAYS
        else:
            url = self.settings.weather.archive_url
            params["start_date"] = start
            params["end_date"] = end

        payload = self._request(url, params)
        hourly = payload.get("hourly") or {}
        if not hourly.get("time"):
            return pd.DataFrame()

        frame = pd.DataFrame(hourly)
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        return frame.dropna(subset=["time"]).set_index("time")

    # ------------------------------------------------------------------ run

    def run(self, seasons: list[int] | None = None, *, force: bool = False) -> int:
        """Puebla la tabla ``weather`` para las temporadas indicadas."""
        seasons = seasons or self.settings.ingestion.seasons
        season_list = ", ".join(str(s) for s in seasons)

        games = self.db.query(
            f"""
            SELECT g.game_id, g.kickoff_utc, g.gameday, g.stadium_id, g.roof
            FROM games g
            WHERE g.season IN ({season_list})
              AND g.kickoff_utc IS NOT NULL
            ORDER BY g.kickoff_utc
            """
        )
        if games.empty:
            log.warning("No hay partidos para consultar el clima.")
            return 0

        if not force:
            existing = self.db.query("SELECT game_id FROM weather")
            if not existing.empty:
                games = games[~games["game_id"].isin(existing["game_id"])]
        if games.empty:
            log.info("El clima ya estaba al día para esas temporadas.")
            return 0

        games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"])

        # --- partidos bajo techo: se marcan y no consumen petición alguna ---
        closed = games["roof"].astype("string").str.lower().isin(["dome", "closed"])
        indoor = games.loc[closed]
        outdoor = games.loc[~closed]

        records: list[dict[str, object]] = []
        now = datetime.now(UTC).replace(tzinfo=None)

        for game_id in indoor["game_id"]:
            records.append(
                {
                    "game_id": game_id,
                    "temperature_c": None, "wind_kph": None, "precipitation_mm": None,
                    "humidity_pct": None, "snowfall_cm": None,
                    "source": "indoor", "as_of_ts": now,
                }
            )

        # --- partidos al aire libre: una petición por estadio ---
        horizon = now + timedelta(days=FORECAST_HORIZON_DAYS)
        n_stadiums = outdoor["stadium_id"].nunique()
        log.info(
            "Clima: %d partidos al aire libre en %d estadios (%d bajo techo).",
            len(outdoor), n_stadiums, len(indoor),
        )

        for index, (stadium_id, group) in enumerate(outdoor.groupby("stadium_id"), start=1):
            stadium = get_stadium(str(stadium_id))
            if stadium is None:
                log.warning("Estadio desconocido '%s': %d partidos sin clima.", stadium_id, len(group))
                continue

            past = group[group["kickoff_utc"] < now]
            future = group[group["kickoff_utc"] >= now]

            series: list[pd.DataFrame] = []

            # Se trocea por temporada: Open-Meteo pondera cada petición por
            # volumen de datos, y un rango de 15 años de golpe devuelve 429.
            # Además, dentro de cada temporada solo pedimos la ventana real de
            # partidos, no el año entero.
            archive_cutoff = now - timedelta(days=6)  # el archivo va con retraso
            for _, season_games in past.groupby(past["kickoff_utc"].dt.year):
                start_ts = season_games["kickoff_utc"].min() - timedelta(days=1)
                end_ts = min(season_games["kickoff_utc"].max() + timedelta(days=1), archive_cutoff)
                if start_ts >= end_ts:
                    continue
                try:
                    series.append(
                        self._fetch_hourly(
                            float(stadium["latitude"]), float(stadium["longitude"]),
                            start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"),
                            forecast=False,
                        )
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    log.warning(
                        "Archivo de clima falló en %s (%s): %s",
                        stadium_id, start_ts.date(), exc,
                    )

            if not future.empty and (future["kickoff_utc"] <= horizon).any():
                try:
                    series.append(
                        self._fetch_hourly(
                            float(stadium["latitude"]), float(stadium["longitude"]),
                            "", "", forecast=True,
                        )
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    log.warning("Pronóstico de clima falló en %s: %s", stadium_id, exc)

            hourly = pd.concat(series) if series else pd.DataFrame()
            if not hourly.empty:
                hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()

            for _, game in group.iterrows():
                records.append(
                    self._match_game_hour(game, hourly, now)
                )

            log.debug("Clima %s (%d/%d): %d partidos.", stadium_id, index, n_stadiums, len(group))

        frame = pd.DataFrame(records)
        if frame.empty:
            return 0

        rows = self.db.write_df(frame, "weather", mode="upsert", key_columns=["game_id"])
        log.info("Clima: %d partidos actualizados.", rows)
        return rows

    @staticmethod
    def _match_game_hour(
        game: pd.Series, hourly: pd.DataFrame, now: datetime
    ) -> dict[str, object]:
        """Extrae la observación horaria más cercana al kickoff."""
        kickoff = game["kickoff_utc"]
        base: dict[str, object] = {
            "game_id": game["game_id"],
            "temperature_c": None, "wind_kph": None, "precipitation_mm": None,
            "humidity_pct": None, "snowfall_cm": None,
            "source": "unavailable",
            "as_of_ts": now,
        }
        if hourly.empty or pd.isna(kickoff):
            return base

        # Tolerancia de 2 horas: si no hay dato cercano, se deja nulo antes que
        # inventar un valor. LightGBM maneja los NaN de forma nativa.
        position = hourly.index.get_indexer([kickoff], method="nearest", tolerance=pd.Timedelta("2h"))
        if position[0] == -1:
            return base

        row = hourly.iloc[position[0]]
        base.update(
            {
                "temperature_c": _safe_float(row.get("temperature_2m")),
                "wind_kph": _safe_float(row.get("wind_speed_10m")),
                "precipitation_mm": _safe_float(row.get("precipitation")),
                "humidity_pct": _safe_float(row.get("relative_humidity_2m")),
                "snowfall_cm": _safe_float(row.get("snowfall")),
                "source": "archive" if kickoff < now else "forecast",
            }
        )
        return base


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):  # type: ignore[arg-type]
        return None
    return float(value)  # type: ignore[arg-type]


def is_adverse_weather(
    temperature_c: pd.Series, wind_kph: pd.Series, precipitation_mm: pd.Series
) -> pd.Series:
    """Bandera de clima adverso: viento fuerte, helada o precipitación apreciable."""
    return (
        (wind_kph >= ADVERSE_WIND_KPH)
        | (temperature_c <= ADVERSE_TEMP_C)
        | (precipitation_mm >= ADVERSE_PRECIP_MM)
    ).fillna(False)
