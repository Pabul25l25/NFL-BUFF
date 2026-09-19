"""Índice de disponibilidad por lesiones, con snapshot as-of estricto.

Las lesiones son la fuente de leakage más traicionera del dominio: el parte de
lesiones **se actualiza durante toda la semana**, y la versión final suele
publicarse a pocas horas del kickoff. Usar la última versión disponible hoy para
predecir un partido de hace tres años es leakage puro.

Aquí se resuelve usando ``date_modified``: para cada partido solo se consideran
los partes con ``date_modified < as_of_ts``.

El índice pondera dos dimensiones:

* **Gravedad del parte** — Out pesa mucho más que Questionable.
* **Importancia posicional** — perder al QB titular no es comparable a perder a
  un long snapper. Los pesos salen del impacto conocido por posición sobre el
  resultado, no de una escala inventada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflp.core.logging import get_logger

log = get_logger(__name__)

# Gravedad del parte de lesión.
STATUS_SEVERITY: dict[str, float] = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.35,
    "ir": 1.0,
    "pup": 1.0,
    "nfi": 1.0,
}

# Código numérico del estado (feature directa para los modelos de props).
STATUS_CODE: dict[str, int] = {
    "questionable": 1,
    "doubtful": 2,
    "out": 3,
    "ir": 3,
    "pup": 3,
    "nfi": 3,
}

PRACTICE_CODE: dict[str, int] = {
    "full participation in practice": 0,
    "limited participation in practice": 1,
    "did not participate in practice": 2,
}

# Peso por posición. El QB domina de forma desproporcionada, algo consistente
# con lo que muestran los datos de EPA por jugador.
POSITION_WEIGHT: dict[str, float] = {
    "QB": 10.0,
    "LT": 3.0, "RT": 2.5, "T": 2.5, "C": 2.0, "G": 1.8, "OL": 2.0,
    "WR": 2.5, "TE": 1.8, "RB": 1.8, "FB": 0.5,
    "EDGE": 2.5, "DE": 2.5, "DT": 2.0, "NT": 1.5, "DL": 2.0,
    "LB": 1.5, "ILB": 1.5, "OLB": 2.0, "MLB": 1.5,
    "CB": 2.5, "S": 1.8, "FS": 1.8, "SS": 1.8, "DB": 1.8,
    "K": 0.8, "P": 0.4, "LS": 0.2,
}
DEFAULT_POSITION_WEIGHT = 1.0


def _normalize_status(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def status_code(series: pd.Series) -> pd.Series:
    """Código numérico del estado del parte (0 = sano)."""
    return _normalize_status(series).map(STATUS_CODE).fillna(0).astype("int64")


def practice_code(series: pd.Series) -> pd.Series:
    """Código numérico de participación en entrenamientos (0 = completo)."""
    return _normalize_status(series).map(PRACTICE_CODE).fillna(0).astype("int64")


def build_injury_index(
    injuries: pd.DataFrame,
    game_as_of: pd.DataFrame,
) -> pd.DataFrame:
    """Índice de lesiones por equipo y partido, respetando el corte as-of.

    Args:
        injuries: tabla ``injuries`` con ``date_modified``.
        game_as_of: debe tener ``game_id``, ``season``, ``week``, ``as_of_ts``,
            ``home_team`` y ``away_team``.

    Returns:
        Una fila por ``(game_id, team)`` con ``injury_index``, ``starters_out`` y
        ``qb_questionable``.
    """
    if injuries.empty:
        log.warning("Sin datos de lesiones: el índice quedará a cero.")
        return pd.DataFrame(
            columns=["game_id", "team", "injury_index", "starters_out", "qb_questionable"]
        )

    reports = injuries.copy()
    reports["date_modified"] = pd.to_datetime(reports["date_modified"], errors="coerce")
    reports["severity"] = (
        _normalize_status(reports["report_status"]).map(STATUS_SEVERITY).fillna(0.0)
    )
    reports["position_weight"] = (
        reports["position"].astype("string").str.strip().str.upper()
        .map(POSITION_WEIGHT).fillna(DEFAULT_POSITION_WEIGHT)
    )
    reports["impact"] = reports["severity"] * reports["position_weight"]
    reports = reports[reports["impact"] > 0]

    # Emparejamos cada parte con los partidos de esa misma semana y equipo.
    long_games = pd.concat(
        [
            game_as_of[["game_id", "season", "week", "as_of_ts", "home_team"]]
            .rename(columns={"home_team": "team"}),
            game_as_of[["game_id", "season", "week", "as_of_ts", "away_team"]]
            .rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )

    merged = long_games.merge(reports, on=["season", "week", "team"], how="left")

    # --- CORTE AS-OF: solo partes publicados antes del instante de predicción ---
    # Los partes sin fecha se descartan: preferimos subestimar el impacto de las
    # lesiones antes que arriesgarnos a usar información del futuro.
    merged["as_of_ts"] = pd.to_datetime(merged["as_of_ts"], errors="coerce")
    visible = merged["date_modified"].notna() & (merged["date_modified"] < merged["as_of_ts"])
    merged = merged[visible | merged["gsis_id"].isna()]

    merged["impact"] = merged["impact"].fillna(0.0)
    merged["is_out"] = (
        _normalize_status(merged["report_status"]).isin(["out", "ir", "pup", "nfi"])
    )
    merged["is_qb_issue"] = (
        merged["position"].astype("string").str.upper().eq("QB")
        & merged["severity"].fillna(0).gt(0)
    )

    aggregated = (
        merged.groupby(["game_id", "team"], observed=True)
        .agg(
            injury_index=("impact", "sum"),
            starters_out=("is_out", "sum"),
            qb_questionable=("is_qb_issue", "any"),
        )
        .reset_index()
    )
    aggregated["starters_out"] = aggregated["starters_out"].astype(float)
    aggregated["qb_questionable"] = aggregated["qb_questionable"].fillna(False).astype(bool)

    log.debug("Índice de lesiones calculado para %d combinaciones equipo-partido.", len(aggregated))
    return aggregated


def player_injury_status(
    injuries: pd.DataFrame, player_as_of: pd.DataFrame
) -> pd.DataFrame:
    """Estado de lesión de cada jugador para su partido, con corte as-of.

    Args:
        player_as_of: ``game_id``, ``gsis_id``, ``season``, ``week``, ``as_of_ts``.
    """
    if injuries.empty:
        result = player_as_of[["game_id", "gsis_id"]].copy()
        result["injury_status_code"] = 0
        result["practice_status_code"] = 0
        return result

    reports = injuries.copy()
    reports["date_modified"] = pd.to_datetime(reports["date_modified"], errors="coerce")

    merged = player_as_of.merge(
        reports[["season", "week", "gsis_id", "report_status", "practice_status", "date_modified"]],
        on=["season", "week", "gsis_id"],
        how="left",
    )
    merged["as_of_ts"] = pd.to_datetime(merged["as_of_ts"], errors="coerce")

    # Fuera del corte as-of, el jugador se trata como sano (sin información).
    stale = merged["date_modified"].isna() | (merged["date_modified"] >= merged["as_of_ts"])
    merged.loc[stale, ["report_status", "practice_status"]] = np.nan

    merged["injury_status_code"] = status_code(merged["report_status"])
    merged["practice_status_code"] = practice_code(merged["practice_status"])

    return merged[["game_id", "gsis_id", "injury_status_code", "practice_status_code"]]
