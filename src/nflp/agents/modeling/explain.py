"""Explicabilidad: valores SHAP traducidos a lenguaje natural.

El requisito de la sección 7 no es mostrar un gráfico de SHAP, sino que el
usuario entienda **por qué** el modelo predijo lo que predijo. Un gráfico de
barras con `net_epa_diff = 0.23` no explica nada a quien no construyó el modelo.

Por eso este módulo hace dos cosas:

1. Calcula los valores SHAP (exactos sobre árboles, vía ``TreeExplainer``).
2. Los traduce a frases en español con el contexto real: no
   *"rest_advantage = 3"* sino *"el local llega con 3 días más de descanso"*.

Cuando una feature no tiene traducción registrada se muestra su nombre tal cual,
en vez de inventar una explicación.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nflp.core.logging import get_logger

log = get_logger(__name__)

# Traducción de cada feature a una frase con sentido. La función recibe el valor
# de la feature y el signo de su contribución SHAP, y devuelve la frase.
_DESCRIPTIONS: dict[str, str] = {
    "net_epa_diff": "diferencia de EPA neto entre ambos equipos",
    "epa_off_diff": "diferencia de EPA ofensivo",
    "epa_def_diff": "diferencia de EPA defensivo",
    "elo_diff": "diferencia de rating Elo",
    "home_elo_pre": "rating Elo del local",
    "away_elo_pre": "rating Elo del visitante",
    "margin_diff_r5": "diferencia de margen en los últimos 5 partidos",
    "pass_matchup_diff": "emparejamiento del juego aéreo local contra la defensa visitante",
    "rush_matchup_diff": "emparejamiento del juego terrestre local contra la defensa visitante",
    "rest_advantage": "ventaja de descanso del local",
    "home_rest_days": "días de descanso del local",
    "away_rest_days": "días de descanso del visitante",
    "away_travel_km": "distancia recorrida por el visitante",
    "away_tz_shift_hours": "cambio de zona horaria del visitante",
    "altitude_m": "altitud del estadio",
    "wind_kph": "viento",
    "temperature_c": "temperatura",
    "precipitation_mm": "precipitación",
    "is_adverse_weather": "clima adverso",
    "roof_is_closed": "estadio techado",
    "surface_is_turf": "superficie artificial",
    "injury_index_diff": "diferencia en el impacto de las lesiones",
    "home_injury_index": "impacto de las lesiones en el local",
    "away_injury_index": "impacto de las lesiones en el visitante",
    "home_qb_questionable": "QB local en duda",
    "away_qb_questionable": "QB visitante en duda",
    "home_starters_out": "titulares locales descartados",
    "away_starters_out": "titulares visitantes descartados",
    "is_divisional": "partido divisional",
    "is_neutral_site": "sede neutral",
    "home_epa_off_r5": "ataque local en los últimos 5 partidos",
    "away_epa_off_r5": "ataque visitante en los últimos 5 partidos",
    "home_epa_def_r5": "defensa local en los últimos 5 partidos",
    "away_epa_def_r5": "defensa visitante en los últimos 5 partidos",
    "home_epa_off_ewm": "forma ofensiva reciente del local",
    "away_epa_off_ewm": "forma ofensiva reciente del visitante",
    "home_pass_rate_r5": "tendencia al pase del local",
    "away_pass_rate_r5": "tendencia al pase del visitante",
    "home_turnover_margin_r5": "margen de balón del local",
    "away_turnover_margin_r5": "margen de balón del visitante",
    "home_margin_r5": "margen medio del local en sus últimos 5 partidos",
    "away_margin_r5": "margen medio del visitante en sus últimos 5 partidos",
    "home_points_for_r5": "puntos anotados por el local",
    "away_points_for_r5": "puntos anotados por el visitante",
    "home_points_against_r5": "puntos encajados por el local",
    "away_points_against_r5": "puntos encajados por el visitante",
    "home_epa_off_r3": "ataque local en los últimos 3 partidos",
    "away_epa_off_r3": "ataque visitante en los últimos 3 partidos",
    "home_epa_def_r3": "defensa local en los últimos 3 partidos",
    "away_epa_def_r3": "defensa visitante en los últimos 3 partidos",
    "home_epa_off_r8": "ataque local en los últimos 8 partidos",
    "away_epa_off_r8": "ataque visitante en los últimos 8 partidos",
    "home_epa_def_r8": "defensa local en los últimos 8 partidos",
    "away_epa_def_r8": "defensa visitante en los últimos 8 partidos",
    "home_epa_def_ewm": "forma defensiva reciente del local",
    "away_epa_def_ewm": "forma defensiva reciente del visitante",
    "home_pass_epa_off_r5": "juego aéreo del local",
    "away_pass_epa_off_r5": "juego aéreo del visitante",
    "home_rush_epa_off_r5": "juego terrestre del local",
    "away_rush_epa_off_r5": "juego terrestre del visitante",
    "home_pass_epa_def_r5": "defensa aérea del local",
    "away_pass_epa_def_r5": "defensa aérea del visitante",
    "home_rush_epa_def_r5": "defensa terrestre del local",
    "away_rush_epa_def_r5": "defensa terrestre del visitante",
    "home_cpoe_r5": "precisión de pase del local sobre lo esperado",
    "away_cpoe_r5": "precisión de pase del visitante sobre lo esperado",
    "home_sack_rate_off_r5": "capturas sufridas por el local",
    "away_sack_rate_off_r5": "capturas sufridas por el visitante",
    "home_sack_rate_def_r5": "presión defensiva del local",
    "away_sack_rate_def_r5": "presión defensiva del visitante",
    "home_epa_off_at_home_r8": "rendimiento del local jugando en casa",
    "away_epa_off_at_away_r8": "rendimiento del visitante jugando fuera",
    "home_style_cluster": "estilo de juego del local",
    "away_style_cluster": "estilo de juego del visitante",
    "humidity_pct": "humedad",
    "is_high_altitude": "estadio de altitud",
    "is_postseason": "partido de playoffs",
    "home_prior_games": "partidos previos del local",
    "away_prior_games": "partidos previos del visitante",
    "style_mismatch": "estilos de juego contrastados",
    "home_short_week": "semana corta para el local",
    "away_short_week": "semana corta para el visitante",
    "home_off_bye": "el local viene de semana de descanso",
    "away_off_bye": "el visitante viene de semana de descanso",
    # props
    "target_share_r5": "cuota de objetivos del jugador",
    "snap_pct_r5": "porcentaje de snaps del jugador",
    "carry_share_r5": "cuota de acarreos del jugador",
    "wopr_r5": "peso del jugador en el juego aéreo",
    "opp_pass_epa_def_r5": "defensa aérea del rival",
    "opp_rush_epa_def_r5": "defensa terrestre del rival",
    "team_plays_r5": "ritmo de juego del equipo",
    "injury_status_code": "estado físico del jugador",
    "receiving_tds_r5": "touchdowns de recepción recientes",
    "receiving_tds_r3": "touchdowns de recepción en los últimos 3 partidos",
    "rushing_tds_r5": "touchdowns de carrera recientes",
    "rushing_tds_r3": "touchdowns de carrera en los últimos 3 partidos",
    "passing_yards_r5": "yardas de pase recientes",
    "passing_yards_r3": "yardas de pase en los últimos 3 partidos",
    "passing_tds_r5": "touchdowns de pase recientes",
    "passing_tds_r3": "touchdowns de pase en los últimos 3 partidos",
    "passing_epa_r5": "eficacia de pase reciente (EPA)",
    "passing_attempts_r5": "intentos de pase recientes",
    "completions_r5": "pases completados recientes",
    "sacks_suffered_r5": "veces capturado recientemente",
    "sacks_suffered_r3": "veces capturado en los últimos 3 partidos",
    "opp_sack_rate_def_r5": "presión de pase del rival",
    "receptions_r3": "recepciones en los últimos 3 partidos",
    # props v2 (feature store de jugador v2)
    "target_share_r3": "cuota de objetivos en los últimos 3 partidos",
    "target_share_ewm": "tendencia reciente de la cuota de objetivos",
    "target_share_blend": "cuota de objetivos estimada (con prior de posición)",
    "air_yards_share_r5": "cuota de yardas aéreas del jugador",
    "air_yards_share_r3": "cuota de yardas aéreas en los últimos 3 partidos",
    "air_yards_share_ewm": "tendencia reciente de la cuota de yardas aéreas",
    "carry_share_r3": "cuota de acarreos en los últimos 3 partidos",
    "carry_share_ewm": "tendencia reciente de la cuota de acarreos",
    "carry_share_blend": "cuota de acarreos estimada (con prior de posición)",
    "snap_pct_r3": "porcentaje de snaps en los últimos 3 partidos",
    "snap_pct_ewm": "tendencia reciente del porcentaje de snaps",
    "snap_pct_blend": "porcentaje de snaps estimado (con prior de posición)",
    "targets_r5": "objetivos recientes",
    "targets_ewm": "tendencia reciente de objetivos",
    "targets_blend": "objetivos estimados (con prior de posición)",
    "receptions_r5": "recepciones recientes",
    "receptions_ewm": "tendencia reciente de recepciones",
    "receptions_blend": "recepciones estimadas (con prior de posición)",
    "receiving_yards_r5": "yardas de recepción recientes",
    "receiving_yards_r3": "yardas de recepción en los últimos 3 partidos",
    "receiving_yards_ewm": "tendencia reciente de yardas de recepción",
    "receiving_yards_blend": "yardas de recepción estimadas (con prior de posición)",
    "rushing_yards_r5": "yardas de carrera recientes",
    "rushing_yards_r3": "yardas de carrera en los últimos 3 partidos",
    "rushing_yards_ewm": "tendencia reciente de yardas de carrera",
    "rushing_yards_blend": "yardas de carrera estimadas (con prior de posición)",
    "carries_r5": "acarreos recientes",
    "carries_ewm": "tendencia reciente de acarreos",
    "carries_blend": "acarreos estimados (con prior de posición)",
    "passing_yards_ewm": "tendencia reciente de yardas de pase",
    "passing_yards_blend": "yardas de pase estimadas (con prior)",
    "passing_attempts_ewm": "tendencia reciente de intentos de pase",
    "opp_rec_yds_allowed_pos_r5": "yardas de recepción que concede el rival a su posición",
    "opp_receptions_allowed_pos_r5": "recepciones que concede el rival a su posición",
    "opp_targets_allowed_pos_r5": "objetivos que concede el rival a su posición",
    "opp_rush_yds_allowed_rb_r5": "yardas de carrera que concede el rival a los RB",
    "opp_points_allowed_r5": "puntos que concede el rival",
    "team_qb_epa_per_db_r5": "eficacia del QB titular del equipo",
    "team_qb_passing_yards_r5": "yardas de pase del QB titular del equipo",
    "team_qb_passing_yards_ewm": "tendencia de yardas de pase del QB titular",
    "team_qb_changed": "cambio de QB titular",
    "vacated_target_share": "objetivos liberados por bajas del equipo",
    "vacated_carry_share": "acarreos liberados por bajas del equipo",
    "depth_rank_proxy": "puesto en la rotación de su posición",
    "position_code": "posición del jugador",
    "practice_status_code": "participación en los entrenamientos",
    "games_played_prior": "partidos previos del jugador",
    "team_pass_rate_r5": "tendencia al pase del equipo",
    "team_epa_off_r5": "eficacia ofensiva del equipo",
    "receiving_epa_r5": "eficacia en recepción (EPA)",
    "rushing_epa_r5": "eficacia en carrera (EPA)",
    "racr_r5": "yardas por yarda aérea (RACR)",
    "yac_r5": "yardas tras la recepción",
    "ypc_r5": "yardas por acarreo",
}

# Unidades para mostrar el valor junto a la frase.
_UNITS: dict[str, str] = {
    "away_travel_km": " km",
    "wind_kph": " km/h",
    "temperature_c": " °C",
    "precipitation_mm": " mm",
    "altitude_m": " m",
    "home_rest_days": " días",
    "away_rest_days": " días",
    "rest_advantage": " días",
    "away_tz_shift_hours": " h",
}

_BOOLEAN_FEATURES = {
    "is_adverse_weather", "roof_is_closed", "surface_is_turf", "is_divisional",
    "is_neutral_site", "style_mismatch", "home_qb_questionable", "away_qb_questionable",
    "home_short_week", "away_short_week", "home_off_bye", "away_off_bye",
    "is_high_altitude", "is_postseason", "is_home", "team_qb_changed",
}


@dataclass
class Explanation:
    """Justificación de una predicción concreta."""

    shap_values: dict[str, float]
    base_value: float
    text: str

    def top_features(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.shap_values.items(), key=lambda kv: -abs(kv[1]))[:n]


class Explainer:
    """Calcula y traduce valores SHAP sobre modelos de árboles."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self._explainer: Any | None = None
        try:
            import shap

            self._explainer = shap.TreeExplainer(model)
        except Exception as exc:  # noqa: BLE001 - SHAP nunca debe tumbar la predicción
            log.warning("No se pudo inicializar SHAP: %s", exc)

    def shap_matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Matriz de valores SHAP y valor base del modelo."""
        if self._explainer is None:
            return np.zeros((len(X), len(self.feature_names))), 0.0

        values = self._explainer.shap_values(X)
        if isinstance(values, list):
            # Clasificación binaria: SHAP devuelve una matriz por clase.
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]

        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).ravel()[-1])

        return values, float(base)

    def explain(
        self,
        X: pd.DataFrame,
        *,
        context: pd.DataFrame | None = None,
        top_n: int = 4,
    ) -> list[Explanation]:
        """Genera una explicación por fila."""
        matrix, base = self.shap_matrix(X)
        explanations: list[Explanation] = []

        for position in range(len(X)):
            contributions = dict(
                zip(self.feature_names, matrix[position].astype(float), strict=False)
            )
            row = X.iloc[position]
            text = describe_contributions(contributions, row, top_n=top_n)
            explanations.append(
                Explanation(shap_values=contributions, base_value=base, text=text)
            )

        return explanations


def _format_value(feature: str, value: Any) -> str:
    """Formatea el valor de una feature para mostrarlo en la frase."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "sin dato"
    if feature in _BOOLEAN_FEATURES:
        return "sí" if float(value) >= 0.5 else "no"
    unit = _UNITS.get(feature, "")
    numeric = float(value)
    if abs(numeric) >= 100:
        return f"{numeric:,.0f}{unit}"
    if abs(numeric) >= 10:
        return f"{numeric:.1f}{unit}"
    return f"{numeric:.2f}{unit}"


def describe_contributions(
    contributions: dict[str, float],
    row: pd.Series,
    *,
    top_n: int = 4,
) -> str:
    """Convierte las contribuciones SHAP en una frase en español.

    El resultado tiene la forma:
    *"A favor del local: diferencia de EPA neto (0,23), ventaja de descanso
    (3 días). En contra: viento (32 km/h)."*
    """
    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    if not ranked:
        return "Sin factores destacados."

    positive: list[str] = []
    negative: list[str] = []

    for feature, contribution in ranked:
        if abs(contribution) < 1e-6:
            continue
        label = _DESCRIPTIONS.get(feature, feature.replace("_", " "))
        value = _format_value(feature, row.get(feature))
        phrase = f"{label} ({value})"
        (positive if contribution > 0 else negative).append(phrase)

    parts: list[str] = []
    if positive:
        parts.append("Empuja al alza: " + ", ".join(positive) + ".")
    if negative:
        parts.append("Empuja a la baja: " + ", ".join(negative) + ".")

    return " ".join(parts) if parts else "Sin factores destacados."


def describe_game_prediction(
    contributions: dict[str, float],
    row: pd.Series,
    *,
    home_team: str,
    away_team: str,
    top_n: int = 4,
) -> str:
    """Variante para partidos, que nombra a los equipos en vez de decir 'al alza'."""
    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    if not ranked:
        return "Sin factores destacados."

    favours_home: list[str] = []
    favours_away: list[str] = []

    for feature, contribution in ranked:
        if abs(contribution) < 1e-6:
            continue
        label = _DESCRIPTIONS.get(feature, feature.replace("_", " "))
        value = _format_value(feature, row.get(feature))
        phrase = f"{label} ({value})"
        (favours_home if contribution > 0 else favours_away).append(phrase)

    parts: list[str] = []
    if favours_home:
        parts.append(f"A favor de {home_team}: " + ", ".join(favours_home) + ".")
    if favours_away:
        parts.append(f"A favor de {away_team}: " + ", ".join(favours_away) + ".")

    return " ".join(parts) if parts else "Sin factores destacados."
