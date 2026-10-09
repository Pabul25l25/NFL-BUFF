"""Modelos de props de jugador (yardas, recepciones, touchdowns, sacks).

Diseño en dos capas, coherente con lo que dominan los datos:

1. **Uso proyectado** — cuota de objetivos, % de snaps, cuota de acarreos. Es el
   componente que más persiste entre semanas.
2. **Eficiencia** — yardas por objetivo, YAC, EPA. Mucho más ruidoso.

La salida es **distribucional**, no un número. Un prop se resuelve contra una
línea ("¿más de 62,5 yardas?"), así que lo útil no es la media sino la
distribución entera. Desde la v2 la distribución depende del jugador:

* **Continuos** (yardas, recepciones): rejilla de 19 cuantiles p05..p95 con
  LightGBM cuantil, monótona y calibrada de forma conformal sobre la temporada
  de calibración (ver ``distributions.py``).
* **Conteos dispersos** (touchdowns, sacks): Poisson con ``λ`` = predicción de
  un modelo con objetivo Poisson (una media, nunca la mediana).

La estimación puntual de los continuos usa el objetivo elegido por target en
un walk-forward (L1, L2, Huber o Tweedie; ver ``PROP_POINT_OBJECTIVE``).

**Componente semi-supervisado** (sección 3 del prompt). Para rookies y jugadores
con poco histórico se aplica pseudo-labeling: se entrena un modelo base con los
jugadores de muestra amplia, se generan pseudo-etiquetas para los de muestra
corta y se reentrena incluyéndolas con peso reducido. El control es un
**shrinkage empírico-Bayes** hacia la media de la posición. Desde la v2 del
feature store los priors de poca muestra ya vienen como features (``*_blend``),
así que el modelo nuevo no usa pseudo-labeling por defecto; el modelo heredado
(``legacy=True``) sí, como en producción.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nflp.agents.modeling.distributions import (
    FINE_LEVELS,
    QUANTILE_LEVELS,
    PoissonDistribution,
    QuantileGridDistribution,
    crps_fine_rows,
    crps_poisson_rows,
    crps_quantile_grid_rows,
    interval_coverage,
    lognormal_quantiles,
)
from nflp.agents.modeling.models import build_estimator, default_params
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Features de uso y eficiencia por objetivo.
#
# v2 del feature store: además de las medias móviles r3/r5 se añaden medias
# exponenciales (``*_ewm``), defensa rival por posición (``opp_*_pos_r5``),
# contexto del QB titular esperado (``team_qb_*``), oportunidad vacante por
# bajas (``vacated_*``), rol en la rotación (``depth_rank_proxy``) y priors de
# poca muestra (``*_blend``). Todas respetan el contrato as-of del feature
# store (ver tests/test_leakage.py). ``position_code`` se deriva aquí de la
# posición del róster (información previa al partido).
_RECEIVING_CORE = [
    "target_share_r5", "target_share_r3", "target_share_ewm", "target_share_blend",
    "air_yards_share_r5", "air_yards_share_r3", "air_yards_share_ewm", "wopr_r5",
    "snap_pct_r5", "snap_pct_r3", "snap_pct_ewm", "snap_pct_blend",
    "targets_r5", "targets_ewm", "targets_blend",
    "receptions_r5", "receptions_ewm", "receptions_blend",
    "receiving_yards_r5", "receiving_yards_r3", "receiving_yards_ewm", "receiving_yards_blend",
    "vacated_target_share", "depth_rank_proxy", "position_code",
    "team_qb_epa_per_db_r5", "team_qb_passing_yards_r5", "team_qb_passing_yards_ewm",
    "team_qb_changed",
    "opp_rec_yds_allowed_pos_r5", "opp_receptions_allowed_pos_r5", "opp_targets_allowed_pos_r5",
]
_RUSHING_CORE = [
    "carry_share_r5", "carry_share_r3", "carry_share_ewm", "carry_share_blend",
    "snap_pct_r5", "snap_pct_r3", "snap_pct_ewm", "snap_pct_blend",
    "carries_r5", "carries_ewm", "carries_blend",
    "rushing_yards_r5", "rushing_yards_r3", "rushing_yards_ewm", "rushing_yards_blend",
    "vacated_carry_share", "depth_rank_proxy", "position_code",
    "opp_rush_yds_allowed_rb_r5",
]
_PASSING_CORE = [
    "passing_yards_r5", "passing_yards_r3", "passing_yards_ewm", "passing_yards_blend",
    "passing_attempts_r5", "passing_attempts_ewm", "completions_r5", "passing_epa_r5",
    "team_qb_epa_per_db_r5", "team_qb_changed", "snap_pct_r5", "snap_pct_ewm",
]
_CONTEXT = [
    "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
    "injury_status_code", "practice_status_code", "is_home",
    "is_adverse_weather", "roof_is_closed", "games_played_prior",
]

PROP_FEATURE_SETS: dict[str, list[str]] = {
    "receiving_yards": [
        *_RECEIVING_CORE,
        "receiving_epa_r5", "racr_r5", "yac_r5",
        "opp_pass_epa_def_r5", "opp_points_allowed_r5",
        "wind_kph", "temperature_c", *_CONTEXT,
    ],
    "rushing_yards": [
        *_RUSHING_CORE,
        "rushing_epa_r5", "ypc_r5",
        "opp_rush_epa_def_r5", "opp_points_allowed_r5",
        "wind_kph", "temperature_c", *_CONTEXT,
    ],
    # ------------------------------------------------------------- ampliación
    "receptions": [
        *_RECEIVING_CORE, "receptions_r3",
        "receiving_epa_r5", "racr_r5",
        "opp_pass_epa_def_r5",
        "wind_kph", *_CONTEXT,
    ],
    "receiving_tds": [
        *_RECEIVING_CORE,
        "receiving_tds_r5", "receiving_tds_r3",
        "opp_pass_epa_def_r5", "opp_points_allowed_r5", *_CONTEXT,
    ],
    "rushing_tds": [
        *_RUSHING_CORE, "ypc_r5",
        "rushing_tds_r5", "rushing_tds_r3",
        "opp_rush_epa_def_r5", "opp_points_allowed_r5", *_CONTEXT,
    ],
    "passing_yards": [
        *_PASSING_CORE,
        "passing_tds_r5", "sacks_suffered_r5",
        "opp_pass_epa_def_r5", "opp_sack_rate_def_r5", "opp_points_allowed_r5",
        "wind_kph", "temperature_c", *_CONTEXT,
    ],
    "passing_tds": [
        *_PASSING_CORE,
        "passing_tds_r5", "passing_tds_r3",
        "opp_pass_epa_def_r5", "opp_points_allowed_r5", *_CONTEXT,
    ],
    # Veces que capturan al propio jugador (normalmente el QB). El contexto
    # relevante se invierte respecto a los props ofensivos: lo que importa es
    # la propia protección (proxy: sus propios sacks recibidos recientes) y
    # la presión que suele generar la defensa RIVAL (`opp_sack_rate_def_r5`).
    "sacks_suffered": [
        "sacks_suffered_r5", "sacks_suffered_r3",
        "passing_attempts_r5", "passing_attempts_ewm",
        "team_qb_epa_per_db_r5", "team_qb_changed", "snap_pct_r5",
        "opp_sack_rate_def_r5", "opp_pass_epa_def_r5",
        "wind_kph", *_CONTEXT,
    ],
}

# Listas v1, tal como estaban en producción hasta la v2. Se conservan para
# reproducir el modelo heredado en el backtest comparativo (``legacy=True``).
LEGACY_PROP_FEATURE_SETS: dict[str, list[str]] = {
    "receiving_yards": [
        "target_share_r5", "target_share_r3", "air_yards_share_r5", "wopr_r5",
        "snap_pct_r5", "snap_pct_r3",
        "receiving_yards_r5", "receiving_yards_r3", "receptions_r5", "targets_r5",
        "receiving_epa_r5", "racr_r5", "yac_r5",
        "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
        "opp_pass_epa_def_r5", "opp_points_allowed_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "wind_kph", "temperature_c", "is_adverse_weather", "roof_is_closed",
        "games_played_prior",
    ],
    "rushing_yards": [
        "carry_share_r5", "snap_pct_r5", "snap_pct_r3",
        "rushing_yards_r5", "rushing_yards_r3", "carries_r5", "rushing_epa_r5", "ypc_r5",
        "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
        "opp_rush_epa_def_r5", "opp_points_allowed_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "wind_kph", "temperature_c", "is_adverse_weather", "roof_is_closed",
        "games_played_prior",
    ],
    "receptions": [
        "target_share_r5", "target_share_r3", "air_yards_share_r5", "wopr_r5",
        "snap_pct_r5", "snap_pct_r3",
        "receptions_r5", "receptions_r3", "targets_r5",
        "receiving_yards_r5", "receiving_epa_r5", "racr_r5",
        "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
        "opp_pass_epa_def_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "wind_kph", "is_adverse_weather", "roof_is_closed",
        "games_played_prior",
    ],
    "receiving_tds": [
        "target_share_r5", "air_yards_share_r5", "wopr_r5", "snap_pct_r5",
        "receiving_yards_r5", "receptions_r5", "receiving_tds_r5", "receiving_tds_r3",
        "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
        "opp_pass_epa_def_r5", "opp_points_allowed_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "is_adverse_weather", "roof_is_closed", "games_played_prior",
    ],
    "rushing_tds": [
        "carry_share_r5", "snap_pct_r5", "rushing_yards_r5", "carries_r5", "ypc_r5",
        "rushing_tds_r5", "rushing_tds_r3",
        "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
        "opp_rush_epa_def_r5", "opp_points_allowed_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "is_adverse_weather", "roof_is_closed", "games_played_prior",
    ],
    "passing_yards": [
        "passing_yards_r5", "passing_yards_r3", "passing_attempts_r5", "completions_r5",
        "passing_tds_r5", "passing_epa_r5", "sacks_suffered_r5",
        "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
        "opp_pass_epa_def_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "wind_kph", "temperature_c", "is_adverse_weather", "roof_is_closed",
        "games_played_prior",
    ],
    "passing_tds": [
        "passing_yards_r5", "passing_attempts_r5", "completions_r5",
        "passing_tds_r5", "passing_tds_r3", "passing_epa_r5",
        "team_pass_rate_r5", "team_plays_r5", "team_epa_off_r5",
        "opp_pass_epa_def_r5", "opp_points_allowed_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "is_adverse_weather", "roof_is_closed", "games_played_prior",
    ],
    "sacks_suffered": [
        "sacks_suffered_r5", "sacks_suffered_r3", "passing_attempts_r5",
        "team_pass_rate_r5", "team_plays_r5",
        "opp_sack_rate_def_r5", "opp_pass_epa_def_r5",
        "injury_status_code", "practice_status_code", "is_home",
        "wind_kph", "is_adverse_weather", "roof_is_closed", "games_played_prior",
    ],
}

# Posiciones relevantes por objetivo.
PROP_POSITIONS: dict[str, tuple[str, ...]] = {
    "receiving_yards": ("WR", "TE", "RB"),
    "rushing_yards": ("RB", "QB", "WR", "FB"),
    "receptions": ("WR", "TE", "RB"),
    "receiving_tds": ("WR", "TE", "RB"),
    "rushing_tds": ("RB", "QB", "WR", "FB"),
    "passing_yards": ("QB",),
    "passing_tds": ("QB",),
    "sacks_suffered": ("QB",),
}

# Código numérico de posición (del róster, conocido antes del partido).
POSITION_CODES: dict[str, float] = {"WR": 0.0, "TE": 1.0, "RB": 2.0, "FB": 3.0, "QB": 4.0}

# Antes de 2016 el estado de róster de nflverse no es fiable (recomendación
# del features-agent): los props nuevos se entrenan con 2016 en adelante.
PROP_MIN_TRAIN_SEASON = 2016

# Props que son conteos de eventos poco frecuentes (touchdowns, sacks): se
# modelan como Poisson, con λ = predicción del modelo con objetivo Poisson.
SPARSE_COUNT_PROPS: frozenset[str] = frozenset(
    {"receiving_tds", "rushing_tds", "passing_tds", "sacks_suffered"}
)
CONTINUOUS_PROPS: frozenset[str] = frozenset(
    {"receiving_yards", "rushing_yards", "receptions", "passing_yards"}
)

# Suelo de la rejilla de cuantiles. Las yardas de carrera pueden ser negativas
# (rodillas del QB, pérdidas tras placaje), así que no se recortan.
QUANTILE_LOWER_BOUND: dict[str, float | None] = {"rushing_yards": None}

# Peso de las pseudo-etiquetas frente a las reales.
PSEUDO_LABEL_WEIGHT = 0.3

# El objetivo por defecto de `default_params` es MAE (mediana). Es degenerado
# en los conteos dispersos de SPARSE_COUNT_PROPS: la mediana de receiving_tds,
# rushing_tds, passing_tds o sacks_suffered es 0 para prácticamente cualquier
# jugador (incluidas estrellas), porque la mayoría de partidos individuales
# terminan en 0 eventos. Un regresor MAE aprende a predecir 0 siempre. Estos
# targets usan un objetivo de MEDIA: Poisson (log-enlace), que además da
# directamente el λ de la distribución servida. El modelo heredado usaba L2,
# que sigue disponible en `_LEGACY_SPARSE_COUNT_OBJECTIVE`.
_SPARSE_COUNT_OBJECTIVE: dict[str, dict[str, object]] = {
    "lightgbm": {"objective": "poisson"},
    "xgboost": {"objective": "count:poisson"},
    "catboost": {"loss_function": "Poisson"},
}
_LEGACY_SPARSE_COUNT_OBJECTIVE: dict[str, dict[str, object]] = {
    "lightgbm": {"objective": "regression"},
    "xgboost": {"objective": "reg:squarederror"},
    "catboost": {"loss_function": "RMSE"},
}

# Objetivos candidatos para la estimación puntual de los continuos.
POINT_OBJECTIVES: tuple[str, ...] = ("l1", "l2", "huber", "tweedie")

# Objetivo elegido por target en el walk-forward 2020-2023 (MAE, con CRPS y
# sesgo como información; ver data/reports/props_objectives.csv). Los conteos
# usan siempre Poisson.
PROP_POINT_OBJECTIVE: dict[str, str] = {
    "receiving_yards": "l1",
    "rushing_yards": "l1",
    "receptions": "l1",
    "passing_yards": "l1",
    "receiving_tds": "poisson",
    "rushing_tds": "poisson",
    "passing_tds": "poisson",
    "sacks_suffered": "poisson",
}

# Hiperparámetros afinados con Optuna (pliegues por temporada, ver
# `props_tuning.py`). Si el fichero no existe se usan los valores por defecto.
TUNED_PARAMS_PATH = Path(__file__).with_name("props_params.json")


def load_tuned_params() -> dict[str, dict[str, Any]]:
    if not TUNED_PARAMS_PATH.exists():
        return {}
    try:
        return json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("No se pudieron leer los hiperparámetros afinados: %s", exc)
        return {}


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas en el propio modelado (solo información previa)."""
    if "position" in frame.columns and "position_code" not in frame.columns:
        frame = frame.copy()
        frame["position_code"] = frame["position"].map(POSITION_CODES).astype(float)
    return frame


def point_objective_params(objective: str, y: np.ndarray | pd.Series) -> dict[str, Any]:
    """Parámetros LightGBM del objetivo puntual.

    * ``huber``: el ``alpha`` de LightGBM es el umbral δ en unidades del
      objetivo, así que se escala con la dispersión robusta de la etiqueta
      (1,345·σ, la constante clásica de Huber). Con el δ=0,9 por defecto se
      comportaría como L1 en yardas.
    * ``tweedie``: potencia 1,5 (entre Poisson y Gamma), adecuada para una
      variable no negativa con masa en 0 y cola derecha.
    """
    if objective == "l1":
        return {"objective": "regression_l1"}
    if objective == "l2":
        return {"objective": "regression"}
    if objective == "huber":
        values = np.asarray(y, dtype=float)
        values = values[~np.isnan(values)]
        mad = float(np.median(np.abs(values - np.median(values)))) if values.size else 1.0
        sigma = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-3)
        return {"objective": "huber", "alpha": 1.345 * sigma}
    if objective == "tweedie":
        return {"objective": "tweedie", "tweedie_variance_power": 1.5}
    if objective == "poisson":
        return {"objective": "poisson"}
    raise ValueError(f"Objetivo desconocido: {objective}")


@dataclass
class PropDistribution:
    """Distribución heredada (v1): log-normal desplazada global.

    Se conserva para cargar los artefactos de producción anteriores y como
    referencia en el backtest. Los modelos nuevos usan ``QuantileGridDistribution``
    (continuos) o ``PoissonDistribution`` (conteos).
    """

    sigma_log: float = 0.85
    shift: float = 1.0
    residual_scale: float = 1.0

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> PropDistribution:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
        valid = ~(np.isnan(truth) | np.isnan(prediction)) & (prediction > 0)
        if valid.sum() < 50:
            log.warning("Pocos residuos para ajustar la distribución del prop.")
            return self

        # Las yardas de carrera PUEDEN ser negativas (pérdida de yardas tras un
        # placaje). El desplazamiento se calcula a partir de los datos para que
        # todo quede positivo antes del logaritmo; fijarlo en 1 producía -inf.
        observed_min = float(np.min(truth[valid]))
        self.shift = max(1.0, -observed_min + 1.0)

        log_ratio = np.log(truth[valid] + self.shift) - np.log(prediction[valid] + self.shift)
        log_ratio = log_ratio[np.isfinite(log_ratio)]
        if log_ratio.size < 50:
            log.warning("Residuos logarítmicos insuficientes tras filtrar no finitos.")
            return self
        mad = float(np.median(np.abs(log_ratio - np.median(log_ratio))))
        self.sigma_log = max(mad * 1.4826, 0.25)
        self.residual_scale = float(np.exp(np.median(log_ratio)))
        return self

    def probability_over(self, predictions: np.ndarray, line: np.ndarray | float) -> np.ndarray:
        """P(yardas reales > línea)."""
        from scipy.stats import norm

        mu = np.maximum(np.asarray(predictions, dtype=float), 0.0) * self.residual_scale
        threshold = np.asarray(line, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(threshold + self.shift) - np.log(mu + self.shift)) / self.sigma_log
        probability = 1.0 - norm.cdf(z)
        return np.clip(np.nan_to_num(probability, nan=0.5), 1e-6, 1 - 1e-6)

    def interval(
        self, predictions: np.ndarray, confidence: float = 0.80
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.stats import norm

        mu = np.maximum(np.asarray(predictions, dtype=float), 0.0) * self.residual_scale
        z = norm.ppf(0.5 + confidence / 2.0)
        lower = np.exp(np.log(mu + self.shift) - z * self.sigma_log) - self.shift
        upper = np.exp(np.log(mu + self.shift) + z * self.sigma_log) - self.shift
        return np.maximum(lower, 0.0), upper

    def quantiles(self, predictions: np.ndarray) -> np.ndarray:
        """Rejilla p05..p95 de esta log-normal (para persistir y puntuar)."""
        return lognormal_quantiles(self, predictions)

    def std(self, predictions: np.ndarray) -> np.ndarray:
        """Desviación típica implícita de la log-normal desplazada."""
        mu = np.maximum(np.asarray(predictions, dtype=float), 0.0) * self.residual_scale
        sigma_squared = float(np.clip(self.sigma_log**2, 1e-6, 4.0))
        variance = (np.exp(sigma_squared) - 1.0) * (mu + self.shift) ** 2 * np.exp(sigma_squared)
        result = np.sqrt(np.maximum(np.nan_to_num(variance, nan=0.0), 0.0))
        return np.maximum(result, 1.0)


@dataclass
class PropModel:
    """Modelo de prop entrenado, con su distribución y metadatos."""

    target: str
    model: object
    distribution: Any
    feature_names: list[str]
    family: str
    used_pseudo_labels: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    objective: str = "l1"
    params: dict[str, Any] = field(default_factory=dict)
    legacy: bool = False

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        aligned = X.reindex(columns=self.feature_names).astype(float)
        return np.maximum(self.model.predict(aligned), 0.0)  # type: ignore[attr-defined]

    # ------------------------------------------------- distribución servida

    @property
    def distribution_kind(self) -> str:
        return getattr(self.distribution, "kind", "lognormal")

    def quantiles(self, X: pd.DataFrame, point: np.ndarray | None = None) -> np.ndarray:
        point = self.predict(X) if point is None else point
        return served_quantiles(self.distribution, point, X)

    def interval(self, X: pd.DataFrame, point: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        point = self.predict(X) if point is None else point
        return served_interval(self.distribution, point, X)

    def std(self, X: pd.DataFrame, point: np.ndarray | None = None) -> np.ndarray:
        point = self.predict(X) if point is None else point
        return served_std(self.distribution, point, X)


# ---------------------------------------------------------------------------
# Interfaz común a las tres distribuciones (heredada, rejilla, Poisson)
# ---------------------------------------------------------------------------


def served_quantiles(distribution: Any, point: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    kind = getattr(distribution, "kind", "lognormal")
    if kind == "lognormal":
        return distribution.quantiles(point)
    return distribution.quantiles(point, X)


def served_interval(
    distribution: Any, point: np.ndarray, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    kind = getattr(distribution, "kind", "lognormal")
    if kind == "lognormal":
        return distribution.interval(point)
    return distribution.interval(point, X)


def served_std(distribution: Any, point: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    kind = getattr(distribution, "kind", "lognormal")
    if kind == "lognormal":
        return distribution.std(point)
    return distribution.std(point, X)


def served_crps_rows(
    distribution: Any, y: np.ndarray, point: np.ndarray, X: pd.DataFrame, *, exact: bool = False
) -> np.ndarray:
    """CRPS por fila de la distribución que se sirve.

    * Poisson → CRPS exacto.
    * Rejilla nueva → CRPS de la rejilla p05..p95.
    * Log-normal heredada → rejilla p05..p95 (comparable con la nueva en los
      continuos) o, con ``exact=True``, rejilla fina de 200 niveles ≈ CRPS
      exacto (comparable con la Poisson en los conteos).
    """
    kind = getattr(distribution, "kind", "lognormal")
    if kind == "poisson":
        return crps_poisson_rows(y, distribution.lam(point))
    if kind == "lognormal" and exact:
        point = np.asarray(point, dtype=float)
        return crps_fine_rows(
            y, lambda rows: lognormal_quantiles(distribution, point[rows], FINE_LEVELS)
        )
    return crps_quantile_grid_rows(y, served_quantiles(distribution, point, X))


def served_crps(
    distribution: Any, y: np.ndarray, point: np.ndarray, X: pd.DataFrame, *, exact: bool = False
) -> float:
    """CRPS medio de la distribución servida (ver ``served_crps_rows``)."""
    rows = served_crps_rows(distribution, y, point, X, exact=exact)
    rows = rows[~np.isnan(rows)]
    return float(rows.mean()) if rows.size else float("nan")


def served_coverage(distribution: Any, y: np.ndarray, point: np.ndarray, X: pd.DataFrame) -> float:
    """Cobertura del 80 %: intervalo literal (continuos) o PIT aleatorizado (Poisson)."""
    kind = getattr(distribution, "kind", "lognormal")
    if kind == "poisson":
        return distribution.coverage(y, point)
    lower, upper = served_interval(distribution, point, X)
    return interval_coverage(y, lower, upper)


def empirical_bayes_shrinkage(
    values: pd.Series, groups: pd.Series, n_observations: pd.Series, *, prior_weight: float = 4.0
) -> pd.Series:
    """Shrinkage empírico-Bayes hacia la media del grupo.

    Es el **control** frente al pseudo-labeling. La idea es antigua y sólida: la
    media de un jugador con 2 partidos es poco fiable, así que se le acerca a la
    media de su posición. El peso del prior equivale a "partidos ficticios": con
    ``prior_weight=4``, un jugador con 4 partidos reales queda a medio camino
    entre su propia media y la de su posición.
    """
    group_means = values.groupby(groups).transform("mean")
    counts = n_observations.fillna(0).clip(lower=0)
    weight = counts / (counts + prior_weight)
    return weight * values.fillna(group_means) + (1 - weight) * group_means


def resolve_params(
    target: str,
    *,
    family: str = "lightgbm",
    random_state: int = 1729,
    objective: str | None = None,
    params: dict[str, Any] | None = None,
    y_train: np.ndarray | pd.Series | None = None,
) -> tuple[dict[str, Any], str]:
    """Hiperparámetros finales (defecto ← afinados ← explícitos ← objetivo)."""
    base = default_params(family, "regression", random_state)
    if family == "lightgbm":
        tuned = load_tuned_params().get(target, {})
        base.update({k: v for k, v in tuned.items() if k != "objective"})
    if params:
        base.update(params)

    if target in SPARSE_COUNT_PROPS:
        chosen = "poisson"
        base.update(_SPARSE_COUNT_OBJECTIVE.get(family, {}))
    else:
        chosen = objective or PROP_POINT_OBJECTIVE.get(target, "l1")
        if family == "lightgbm":
            base.pop("alpha", None)
            base.pop("tweedie_variance_power", None)
            base.update(point_objective_params(chosen, y_train if y_train is not None else [1.0]))
    return base, chosen


def train_prop_model(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    target: str,
    *,
    family: str = "lightgbm",
    random_state: int = 1729,
    use_pseudo_labels: bool = False,
    objective: str | None = None,
    params: dict[str, Any] | None = None,
    legacy: bool = False,
    with_distribution: bool = True,
) -> PropModel:
    """Entrena un modelo de prop con su distribución predictiva.

    Args:
        train: filas de entrenamiento, con la columna objetivo presente.
        calib: filas **posteriores** en el tiempo, para calibrar la distribución.
        objective: objetivo puntual de los continuos (l1, l2, huber, tweedie).
        legacy: reproduce el modelo v1 (features v1, L1/L2 por defecto,
            pseudo-labeling y log-normal global) para comparar.
        with_distribution: si es False solo se entrena el modelo puntual (útil
            para la búsqueda de hiperparámetros).
    """
    if legacy:
        return _train_legacy(train, calib, target, family=family, random_state=random_state)

    train = add_derived_features(train)
    calib = add_derived_features(calib)
    features = [f for f in PROP_FEATURE_SETS[target] if f in train.columns]
    if not features:
        raise ValueError(f"Sin features disponibles para el prop '{target}'.")

    X_train = train[features].astype(float)
    y_train = train[target].astype(float)

    final_params, chosen = resolve_params(
        target, family=family, random_state=random_state,
        objective=objective, params=params, y_train=y_train,
    )
    fit_y = y_train.clip(lower=0.0) if chosen in {"tweedie", "poisson"} else y_train
    model = build_estimator(family, "regression", final_params)

    used_pseudo = False
    if use_pseudo_labels and "is_low_sample" in train.columns:
        model, used_pseudo = _fit_with_pseudo_labels(
            model, train.assign(**{target: fit_y}), features, target, family, final_params
        )
    else:
        model.fit(X_train, fit_y)

    distribution: Any
    if target in SPARSE_COUNT_PROPS:
        distribution = PoissonDistribution()
    else:
        distribution = QuantileGridDistribution(
            feature_names=features,
            lower_bound=QUANTILE_LOWER_BOUND.get(target, 0.0),
        )
        if with_distribution:
            distribution.fit(X_train, y_train.to_numpy(), final_params)
            if not calib.empty:
                distribution.calibrate(calib[features].astype(float), calib[target].astype(float).to_numpy())

    return PropModel(
        target=target,
        model=model,
        distribution=distribution,
        feature_names=features,
        family=family,
        used_pseudo_labels=used_pseudo,
        objective=chosen,
        params=final_params,
    )


def _train_legacy(
    train: pd.DataFrame, calib: pd.DataFrame, target: str, *, family: str, random_state: int
) -> PropModel:
    """El modelo de producción v1, sin cambios, como referencia."""
    features = [f for f in LEGACY_PROP_FEATURE_SETS[target] if f in train.columns]
    params = default_params(family, "regression", random_state)
    if target in SPARSE_COUNT_PROPS:
        params = {**params, **_LEGACY_SPARSE_COUNT_OBJECTIVE.get(family, {})}
    model = build_estimator(family, "regression", params)
    model, used_pseudo = (
        _fit_with_pseudo_labels(model, train, features, target, family, params)
        if "is_low_sample" in train.columns
        else (model.fit(train[features].astype(float), train[target].astype(float)) or model, False)
    )
    distribution = PropDistribution()
    if not calib.empty:
        distribution.fit(
            calib[target].astype(float).to_numpy(),
            np.maximum(model.predict(calib[features].astype(float)), 0.0),
        )
    return PropModel(
        target=target, model=model, distribution=distribution, feature_names=features,
        family=family, used_pseudo_labels=used_pseudo,
        objective="poisson_l2_legacy" if target in SPARSE_COUNT_PROPS else "l1",
        params=params, legacy=True,
    )


def _fit_with_pseudo_labels(
    model: object,
    train: pd.DataFrame,
    features: list[str],
    target: str,
    family: str,
    params: dict,
) -> tuple[object, bool]:
    """Pseudo-labeling para jugadores con poco histórico.

    Procedimiento:
    1. Se entrena un modelo base **solo** con jugadores de muestra amplia, cuyas
       features de uso son fiables.
    2. Ese modelo genera pseudo-etiquetas para las filas de muestra corta.
    3. Se reentrena con todo, dando a las pseudo-etiquetas un peso reducido
       (``PSEUDO_LABEL_WEIGHT``) para que no dominen el ajuste.

    El peso bajo es deliberado: las pseudo-etiquetas son predicciones, no
    observaciones, y tratarlas como verdad amplifica los errores del modelo base.
    """
    low_sample = train["is_low_sample"].fillna(False).astype(bool)
    reliable = train.loc[~low_sample]

    if len(reliable) < 500 or low_sample.sum() < 50:
        log.debug("Pseudo-labeling omitido: muestra insuficiente.")
        model.fit(train[features].astype(float), train[target].astype(float))  # type: ignore[attr-defined]
        return model, False

    base = build_estimator(family, "regression", params)
    base.fit(reliable[features].astype(float), reliable[target].astype(float))

    pseudo_X = train.loc[low_sample, features].astype(float)
    pseudo_y = np.maximum(base.predict(pseudo_X), 0.0)

    X_all = train[features].astype(float)
    y_all = train[target].astype(float).copy()
    weights = np.where(low_sample.to_numpy(), PSEUDO_LABEL_WEIGHT, 1.0)

    blended = y_all.copy()
    blended.loc[low_sample] = (
        0.5 * y_all.loc[low_sample].to_numpy() + 0.5 * pseudo_y
    )

    model.fit(X_all, blended, sample_weight=weights)  # type: ignore[attr-defined]
    log.debug(
        "Pseudo-labeling aplicado a %d filas de poca muestra (peso %.1f).",
        int(low_sample.sum()), PSEUDO_LABEL_WEIGHT,
    )
    return model, True


def filter_prop_rows(frame: pd.DataFrame, target: str, min_snaps: int = 10) -> pd.DataFrame:
    """Filtra las filas relevantes para un prop.

    Se descartan jugadores fuera de posición y los que no proyectan suficiente
    participación: predecir las yardas de un tercer receptor que juega 5 snaps es
    ruido puro, y contaminaría las métricas.
    """
    positions = PROP_POSITIONS[target]
    subset = frame[frame["position"].isin(positions)].copy()

    if "snap_pct_r5" in subset.columns:
        # Umbral relativo: snap_pct viene en 0-100 en nflverse.
        threshold = min_snaps / 100.0 if subset["snap_pct_r5"].max() <= 1.5 else float(min_snaps)
        subset = subset[subset["snap_pct_r5"].fillna(0) >= threshold]

    if "games_played_prior" in subset.columns:
        subset = subset[subset["games_played_prior"] >= 1]

    return add_derived_features(subset)


__all__ = [
    "CONTINUOUS_PROPS",
    "LEGACY_PROP_FEATURE_SETS",
    "PROP_FEATURE_SETS",
    "PROP_MIN_TRAIN_SEASON",
    "QUANTILE_LEVELS",
    "SPARSE_COUNT_PROPS",
    "PropDistribution",
    "PropModel",
    "empirical_bayes_shrinkage",
    "filter_prop_rows",
    "served_coverage",
    "served_crps",
    "served_crps_rows",
    "served_interval",
    "served_quantiles",
    "served_std",
    "train_prop_model",
]
