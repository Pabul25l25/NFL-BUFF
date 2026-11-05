"""Normalización de los datos crudos de nflverse.

Problemas reales observados en la fuente que se corrigen aquí:

* ``surface`` trae ``'grass '`` con espacio final y cadenas vacías.
* ``roof`` trae cadenas vacías además de ``dome``/``closed``/``open``/``outdoors``.
* ``gameday`` + ``gametime`` vienen separados y en hora del Este; el pipeline
  necesita un ``kickoff_utc`` único para ordenar cronológicamente y para cortar
  el snapshot as-of.
* Las abreviaturas de equipo cambiaron con las mudanzas de franquicia
  (SD→LAC, OAK→LV, STL→LA), lo que rompe las series históricas por equipo.
"""

from __future__ import annotations

import pandas as pd

from nflp.core.logging import get_logger

log = get_logger(__name__)

# nflverse publica `gametime` en hora del Este, también para partidos en Europa.
NFL_SCHEDULE_TZ = "America/New_York"

# Superficies artificiales. Todo lo demás se considera césped natural.
TURF_SURFACES = {"fieldturf", "astroturf", "matrixturf", "sportturf", "a_turf", "astroplay"}

# Techos que dejan el campo cubierto durante el partido.
CLOSED_ROOFS = {"dome", "closed"}

# Reubicaciones de franquicia: se unifica al código actual para que las series
# históricas por equipo no se partan en dos.
TEAM_RELOCATIONS = {
    "SD": "LAC",
    "SDG": "LAC",
    "OAK": "LV",
    "STL": "LA",
    "LAR": "LA",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "SL": "LA",
}


def normalize_team(series: pd.Series) -> pd.Series:
    """Unifica abreviaturas de equipo al código actual."""
    return series.astype("string").str.strip().str.upper().replace(TEAM_RELOCATIONS)


def normalize_surface(series: pd.Series) -> pd.Series:
    """Limpia la superficie: minúsculas, sin espacios, vacío -> NA."""
    cleaned = series.astype("string").str.strip().str.lower()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})


def normalize_roof(series: pd.Series) -> pd.Series:
    """Limpia el tipo de techo: minúsculas, sin espacios, vacío -> NA."""
    cleaned = series.astype("string").str.strip().str.lower()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})


def is_turf(surface: pd.Series) -> pd.Series:
    """True si la superficie es artificial."""
    return normalize_surface(surface).isin(TURF_SURFACES).fillna(False)


def is_closed_roof(roof: pd.Series) -> pd.Series:
    """True si el partido se juega bajo techo (sin efecto del clima)."""
    return normalize_roof(roof).isin(CLOSED_ROOFS).fillna(False)


def normalize_season_type(series: pd.Series) -> pd.Series:
    """Unifica el tipo de temporada a REG / POST.

    El prompt exige que regular season y playoffs queden marcadas por separado:
    los playoffs tienen una distribución distinta (mejores equipos, más presión,
    sin partidos irrelevantes de final de temporada).
    """
    cleaned = series.astype("string").str.strip().str.upper()
    return cleaned.map(lambda v: "REG" if v == "REG" else ("POST" if pd.notna(v) else pd.NA)).astype(
        "string"
    )


def game_type_to_season_type(game_type: pd.Series) -> pd.Series:
    """Convierte el ``game_type`` de nflverse (REG, WC, DIV, CON, SB) a REG/POST."""
    cleaned = game_type.astype("string").str.strip().str.upper()
    return cleaned.map(lambda v: "REG" if v == "REG" else "POST").astype("string")


def build_kickoff_utc(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Combina fecha y hora del calendario en un timestamp UTC.

    `gametime` puede faltar en partidos futuros sin horario confirmado; en ese
    caso se asume las 13:00 ET, el horario más frecuente, y el margen as-of
    posterior absorbe la imprecisión.
    """
    date_part = pd.to_datetime(gameday, errors="coerce").dt.strftime("%Y-%m-%d")
    time_part = gametime.astype("string").str.strip()
    time_part = time_part.where(time_part.str.match(r"^\d{1,2}:\d{2}$", na=False), "13:00")

    combined = pd.to_datetime(
        date_part + " " + time_part,
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )

    # Localizamos en hora del Este y convertimos a UTC. `nonexistent`/`ambiguous`
    # cubren los saltos de horario de verano (relevantes en septiembre/noviembre).
    localized = combined.dt.tz_localize(
        NFL_SCHEDULE_TZ,
        nonexistent="shift_forward",
        ambiguous=True,
    )
    return localized.dt.tz_convert("UTC").dt.tz_localize(None)


def compute_as_of(kickoff_utc: pd.Series, offset_hours: float) -> pd.Series:
    """Instante de corte as-of: ``kickoff - offset``.

    Es el invariante central anti-leakage: ninguna feature puede usar datos
    posteriores a este timestamp.
    """
    return kickoff_utc - pd.Timedelta(hours=offset_hours)


def coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convierte a numérico las columnas indicadas que existan en el DataFrame."""
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def to_nullable_bool(series: pd.Series) -> pd.Series:
    """Convierte 0/1, 'TRUE'/'FALSE' y booleanos a un booleano con soporte de NA."""
    if series.dtype == bool:
        return series.astype("boolean")

    # En Python `1 == 1.0 == True`, así que una sola entrada cubre los tres
    # (y lo mismo para 0 / 0.0 / False). Añadirlas por separado sería redundante.
    lookup = {
        1: True, 0: False,
        "1": True, "0": False,
        "TRUE": True, "FALSE": False, "true": True, "false": False,
        "T": True, "F": False,
    }
    # `map` en vez de `replace`: evita el downcasting implícito que pandas 2.2
    # marca como deprecado, y deja en NA cualquier valor no reconocido.
    return series.map(lookup).astype("boolean")


def deduplicate(frame: pd.DataFrame, keys: list[str], *, label: str) -> pd.DataFrame:
    """Elimina duplicados por clave, avisando cuando los hay.

    nflverse ocasionalmente publica filas repetidas tras una corrección; si se
    insertan tal cual, violan la clave primaria.
    """
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise KeyError(f"{label}: faltan columnas clave {missing}")

    duplicated = frame.duplicated(subset=keys, keep="last")
    if duplicated.any():
        log.warning("%s: %d filas duplicadas descartadas.", label, int(duplicated.sum()))
    return frame.loc[~duplicated].copy()
