"""Tabla de referencia estática de estadios.

nflverse publica ``stadium_id`` y el nombre del estadio, pero **no** coordenadas,
altitud ni zona horaria. Esos tres datos son imprescindibles para:

* **Clima** — Open-Meteo requiere latitud y longitud.
* **Distancia de viaje** — fórmula de haversine entre estadios.
* **Desfase horario** — jugar a las 13:00 ET viniendo de la costa oeste.
* **Altitud** — Denver (1.610 m) y Ciudad de México (2.200 m) afectan al juego
  aéreo y a la fatiga de forma medible.

Los ``stadium_id`` corresponden a los valores reales de nflverse, incluidos los
estadios internacionales (Londres, México, Alemania, Brasil, España, Australia).

`roof_type` es el valor *por defecto* del recinto; para cada partido concreto se
usa el `roof` que publica nflverse, porque los techos retráctiles cambian de
estado según el día.
"""

from __future__ import annotations

import math

import pandas as pd

# stadium_id: (nombre, equipo principal, lat, lon, altitud_m, techo, superficie, tz)
_STADIUMS: dict[str, tuple[str, str | None, float, float, float, str, str, str]] = {
    # ---------------------------------------------------------------- activos
    "NYC01": ("MetLife Stadium", "NYG", 40.8135, -74.0745, 7, "outdoors", "fieldturf", "America/New_York"),
    "LAX01": ("SoFi Stadium", "LA", 33.9535, -118.3392, 30, "dome", "matrixturf", "America/Los_Angeles"),
    "KAN00": ("Arrowhead Stadium", "KC", 39.0489, -94.4839, 265, "outdoors", "grass", "America/Chicago"),
    "BOS00": ("Gillette Stadium", "NE", 42.0909, -71.2643, 92, "outdoors", "fieldturf", "America/New_York"),
    "PHI00": ("Lincoln Financial Field", "PHI", 39.9008, -75.1675, 3, "outdoors", "grass", "America/New_York"),
    "HOU00": ("NRG Stadium", "HOU", 29.6847, -95.4107, 15, "closed", "astroturf", "America/Chicago"),
    "BUF00": ("Highmark Stadium", "BUF", 42.7738, -78.7870, 183, "outdoors", "fieldturf", "America/New_York"),
    "PHO00": ("State Farm Stadium", "ARI", 33.5276, -112.2626, 340, "closed", "grass", "America/Phoenix"),
    "BAL00": ("M&T Bank Stadium", "BAL", 39.2780, -76.6227, 10, "outdoors", "grass", "America/New_York"),
    "GNB00": ("Lambeau Field", "GB", 44.5013, -88.0622, 195, "outdoors", "grass", "America/Chicago"),
    "NOR00": ("Caesars Superdome", "NO", 29.9511, -90.0812, 3, "dome", "fieldturf", "America/Chicago"),
    "TAM00": ("Raymond James Stadium", "TB", 27.9759, -82.5033, 8, "outdoors", "grass", "America/New_York"),
    "DEN00": ("Empower Field at Mile High", "DEN", 39.7439, -105.0201, 1610, "outdoors", "grass", "America/Denver"),
    "PIT00": ("Acrisure Stadium", "PIT", 40.4468, -80.0158, 220, "outdoors", "grass", "America/New_York"),
    "SEA00": ("Lumen Field", "SEA", 47.5952, -122.3316, 3, "outdoors", "fieldturf", "America/Los_Angeles"),
    "SFO01": ("Levi's Stadium", "SF", 37.4030, -121.9700, 8, "outdoors", "grass", "America/Los_Angeles"),
    "DAL00": ("AT&T Stadium", "DAL", 32.7473, -97.0945, 170, "closed", "matrixturf", "America/Chicago"),
    "DET00": ("Ford Field", "DET", 42.3400, -83.0456, 180, "dome", "fieldturf", "America/New_York"),
    "CAR00": ("Bank of America Stadium", "CAR", 35.2258, -80.8528, 220, "outdoors", "fieldturf", "America/New_York"),
    "CHI98": ("Soldier Field", "CHI", 41.8623, -87.6167, 180, "outdoors", "grass", "America/Chicago"),
    "NAS00": ("Nissan Stadium", "TEN", 36.1665, -86.7713, 130, "outdoors", "grass", "America/Chicago"),
    "WAS00": ("Northwest Stadium", "WAS", 38.9077, -76.8645, 60, "outdoors", "grass", "America/New_York"),
    "CIN00": ("Paycor Stadium", "CIN", 39.0955, -84.5161, 150, "outdoors", "fieldturf", "America/New_York"),
    "IND00": ("Lucas Oil Stadium", "IND", 39.7601, -86.1639, 218, "closed", "fieldturf", "America/Indiana/Indianapolis"),
    "MIA00": ("Hard Rock Stadium", "MIA", 25.9580, -80.2389, 2, "outdoors", "grass", "America/New_York"),
    "CLE00": ("Huntington Bank Field", "CLE", 41.5061, -81.6995, 175, "outdoors", "grass", "America/New_York"),
    "JAX00": ("EverBank Stadium", "JAX", 30.3239, -81.6373, 5, "outdoors", "grass", "America/New_York"),
    "MIN01": ("U.S. Bank Stadium", "MIN", 44.9736, -93.2575, 260, "dome", "sportturf", "America/Chicago"),
    "ATL97": ("Mercedes-Benz Stadium", "ATL", 33.7554, -84.4008, 320, "closed", "fieldturf", "America/New_York"),
    "VEG00": ("Allegiant Stadium", "LV", 36.0909, -115.1833, 650, "dome", "grass", "America/Los_Angeles"),
    # ------------------------------------------------------- históricos (<2020)
    "OAK00": ("Oakland Coliseum", "OAK", 37.7516, -122.2005, 13, "outdoors", "grass", "America/Los_Angeles"),
    "LAX99": ("Los Angeles Memorial Coliseum", "LA", 34.0141, -118.2879, 55, "outdoors", "grass", "America/Los_Angeles"),
    "LAX97": ("Dignity Health Sports Park", "LAC", 33.8644, -118.2611, 20, "outdoors", "grass", "America/Los_Angeles"),
    "ATL00": ("Georgia Dome", "ATL", 33.7576, -84.4009, 315, "dome", "fieldturf", "America/New_York"),
    "SDG00": ("Qualcomm Stadium", "SD", 32.7831, -117.1196, 30, "outdoors", "grass", "America/Los_Angeles"),
    "MIN98": ("TCF Bank Stadium", "MIN", 44.9765, -93.2245, 250, "outdoors", "fieldturf", "America/Chicago"),
    "STL00": ("Edward Jones Dome", "STL", 38.6328, -90.1885, 140, "dome", "astroturf", "America/Chicago"),
    # --------------------------------------------------------- internacionales
    "LON00": ("Wembley Stadium", None, 51.5560, -0.2795, 30, "outdoors", "grass", "Europe/London"),
    "LON01": ("Twickenham Stadium", None, 51.4560, -0.3417, 10, "outdoors", "grass", "Europe/London"),
    "LON02": ("Tottenham Hotspur Stadium", None, 51.6043, -0.0665, 30, "outdoors", "grass", "Europe/London"),
    "MEX00": ("Estadio Azteca", None, 19.3029, -99.1505, 2200, "outdoors", "grass", "America/Mexico_City"),
    "FRA00": ("Deutsche Bank Park", None, 50.0686, 8.6455, 110, "outdoors", "grass", "Europe/Berlin"),
    "GER00": ("Allianz Arena", None, 48.2188, 11.6247, 500, "outdoors", "grass", "Europe/Berlin"),
    "MUN01": ("Allianz Arena", None, 48.2188, 11.6247, 500, "outdoors", "grass", "Europe/Berlin"),
    "SAO00": ("Neo Química Arena", None, -23.5453, -46.4742, 780, "outdoors", "grass", "America/Sao_Paulo"),
    "RIO00": ("Estádio do Maracanã", None, -22.9121, -43.2302, 10, "outdoors", "grass", "America/Sao_Paulo"),
    "PAR00": ("Stade de France", None, 48.9245, 2.3601, 30, "outdoors", "grass", "Europe/Paris"),
    "MAD01": ("Santiago Bernabéu", None, 40.4531, -3.6883, 660, "closed", "grass", "Europe/Madrid"),
    "MEL00": ("Melbourne Cricket Ground", None, -37.8200, 144.9834, 30, "outdoors", "grass", "Australia/Melbourne"),
}

# Altitud (m) por encima de la cual se marca `is_high_altitude`.
HIGH_ALTITUDE_THRESHOLD_M = 1000.0


def stadiums_frame() -> pd.DataFrame:
    """Devuelve la tabla de estadios lista para insertar en la base de datos."""
    records = [
        {
            "stadium_id": stadium_id,
            "stadium_name": name,
            "team_abbr": team,
            "latitude": lat,
            "longitude": lon,
            "altitude_m": float(alt),
            "roof_type": roof,
            "surface": surface,
            "timezone": tz,
        }
        for stadium_id, (name, team, lat, lon, alt, roof, surface, tz) in _STADIUMS.items()
    ]
    return pd.DataFrame.from_records(records)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en kilómetros sobre la superficie terrestre entre dos puntos."""
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")

    earth_radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def get_stadium(stadium_id: str) -> dict[str, object] | None:
    """Datos de un estadio concreto, o ``None`` si no está en la tabla."""
    entry = _STADIUMS.get(stadium_id)
    if entry is None:
        return None
    name, team, lat, lon, alt, roof, surface, tz = entry
    return {
        "stadium_id": stadium_id,
        "stadium_name": name,
        "team_abbr": team,
        "latitude": lat,
        "longitude": lon,
        "altitude_m": float(alt),
        "roof_type": roof,
        "surface": surface,
        "timezone": tz,
    }


def home_stadium_by_team() -> dict[str, str]:
    """Mapa equipo -> stadium_id de su sede habitual.

    Se usa para estimar el origen del viaje del visitante cuando el partido se
    juega en sede neutral (Londres, México), donde ambos equipos viajan.
    """
    mapping: dict[str, str] = {}
    for stadium_id, (_, team, *_rest) in _STADIUMS.items():
        if team and team not in mapping:
            mapping[team] = stadium_id
    return mapping


def known_stadium_ids() -> set[str]:
    return set(_STADIUMS)
