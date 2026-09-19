"""Modelos de props de jugador (yardas de recepción y de carrera).

Diseño en dos capas, coherente con lo que dominan los datos:

1. **Uso proyectado** — cuota de objetivos, % de snaps, cuota de acarreos. Es el
   componente que más persiste entre semanas.
2. **Eficiencia** — yardas por objetivo, YAC, EPA. Mucho más ruidoso.

La salida es **distribucional**, no un número. Un prop se resuelve contra una
línea ("¿más de 62,5 yardas?"), así que lo útil no es la media sino P(superar la
línea). La dispersión se estima a partir de los residuos de validación, y la
distribución es log-normal desplazada: las yardas están acotadas por abajo en 0 y
tienen cola derecha larga (un receptor puede hacer 180 yardas, nunca −20).

**Componente semi-supervisado** (sección 3 del prompt). Para rookies y jugadores
con poco histórico se aplica pseudo-labeling: se entrena un modelo base con los
jugadores de muestra amplia, se generan pseudo-etiquetas para los de muestra
corta y se reentrena incluyéndolas con peso reducido.

Se implementa porque el encargo lo pide, pero se reporta con honestidad: el
control es un **shrinkage empírico-Bayes** hacia la media de la posición, que es
metodológicamente más defendible y suele igualar o superar al pseudo-labeling en
este dominio. El backtest compara ambos y el dashboard muestra cuál gana.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nflp.agents.modeling.models import build_estimator, default_params
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Features de uso y eficiencia por objetivo.
PROP_FEATURE_SETS: dict[str, list[str]] = {
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
    # ------------------------------------------------------------- ampliación
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
    # Veces que capturan al propio jugador (normalmente el QB). El contexto
    # relevante se invierte respecto a los props ofensivos: lo que importa es
    # la propia protección (proxy: sus propios sacks recibidos recientes) y
    # la presión que suele generar la defensa RIVAL (`opp_sack_rate_def_r5`).
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

# Props que son conteos de eventos poco frecuentes (touchdowns, sacks), no
# yardas continuas. La distribución log-normal desplazada de PropDistribution
# sigue siendo una aproximación razonable —da un rango y una P(over)— pero es
# menos fiel aquí que en yardas: un conteo disperso (mayoría de partidos en 0)
# no es realmente continuo. No se bloquea por eso, se documenta: un modelo de
# conteo dedicado (Poisson/binomial negativa) sería más correcto si esto se
# vuelve crítico. Ver docs/decisiones-modelado.md.
SPARSE_COUNT_PROPS: frozenset[str] = frozenset(
    {"receiving_tds", "rushing_tds", "passing_tds", "sacks_suffered"}
)

# Peso de las pseudo-etiquetas frente a las reales.
PSEUDO_LABEL_WEIGHT = 0.3


@dataclass
class PropDistribution:
    """Distribución predictiva de un prop de yardas.

    Log-normal desplazada: las yardas no bajan de 0 y tienen cola derecha larga.
    Modelarlas como normal produce probabilidades negativas en líneas bajas.
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

        # Residuos en escala logarítmica, que es donde son aproximadamente
        # simétricos para una variable con cola derecha.
        log_ratio = np.log(truth[valid] + self.shift) - np.log(prediction[valid] + self.shift)
        log_ratio = log_ratio[np.isfinite(log_ratio)]
        if log_ratio.size < 50:
            log.warning("Residuos logarítmicos insuficientes tras filtrar no finitos.")
            return self
        # Desviación robusta: los partidos de 180 yardas son reales pero raros.
        mad = float(np.median(np.abs(log_ratio - np.median(log_ratio))))
        self.sigma_log = max(mad * 1.4826, 0.25)
        self.residual_scale = float(np.exp(np.median(log_ratio)))

        log.debug(
            "Distribución del prop: sigma_log=%.3f, escala=%.3f", self.sigma_log, self.residual_scale
        )
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

    def std(self, predictions: np.ndarray) -> np.ndarray:
        """Desviación típica implícita de la log-normal desplazada."""
        mu = np.maximum(np.asarray(predictions, dtype=float), 0.0) * self.residual_scale
        # Var[X] de una log-normal con mediana (mu + shift) en escala original.
        sigma_squared = float(np.clip(self.sigma_log**2, 1e-6, 4.0))
        variance = (np.exp(sigma_squared) - 1.0) * (mu + self.shift) ** 2 * np.exp(sigma_squared)
        result = np.sqrt(np.maximum(np.nan_to_num(variance, nan=0.0), 0.0))
        # Suelo de dispersión: una desviación nula haría explotar el CRPS.
        return np.maximum(result, 1.0)


@dataclass
class PropModel:
    """Modelo de prop entrenado, con su distribución y metadatos."""

    target: str
    model: object
    distribution: PropDistribution
    feature_names: list[str]
    family: str
    used_pseudo_labels: bool = False
    metrics: dict[str, float] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        aligned = X.reindex(columns=self.feature_names)
        return np.maximum(self.model.predict(aligned), 0.0)  # type: ignore[attr-defined]


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


def train_prop_model(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    target: str,
    *,
    family: str = "lightgbm",
    random_state: int = 1729,
    use_pseudo_labels: bool = True,
) -> PropModel:
    """Entrena un modelo de prop con su distribución predictiva.

    Args:
        train: filas de entrenamiento, con la columna objetivo presente.
        calib: filas posteriores en el tiempo, para ajustar la distribución.
    """
    features = [f for f in PROP_FEATURE_SETS[target] if f in train.columns]
    if not features:
        raise ValueError(f"Sin features disponibles para el prop '{target}'.")

    X_train = train[features].astype(float)
    y_train = train[target].astype(float)

    params = default_params(family, "regression", random_state)
    model = build_estimator(family, "regression", params)

    used_pseudo = False
    if use_pseudo_labels and "is_low_sample" in train.columns:
        model, used_pseudo = _fit_with_pseudo_labels(
            model, train, features, target, family, params
        )
    else:
        model.fit(X_train, y_train)

    distribution = PropDistribution()
    if not calib.empty:
        calib_X = calib[features].astype(float)
        distribution.fit(
            calib[target].astype(float).to_numpy(),
            np.maximum(model.predict(calib_X), 0.0),
        )

    return PropModel(
        target=target,
        model=model,
        distribution=distribution,
        feature_names=features,
        family=family,
        used_pseudo_labels=used_pseudo,
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
    # Las filas de poca muestra conservan su etiqueta real si la tienen; la
    # pseudo-etiqueta solo actúa como regularizador vía el peso.
    weights = np.where(low_sample.to_numpy(), PSEUDO_LABEL_WEIGHT, 1.0)

    # Mezcla: la etiqueta de las filas dudosas se acerca a la predicción del
    # modelo base, lo que reduce la varianza de esas observaciones ruidosas.
    blended = y_all.copy()
    blended.loc[low_sample] = (
        0.5 * y_all.loc[low_sample].to_numpy() + 0.5 * pseudo_y
    )

    model.fit(X_all, blended, sample_weight=weights)  # type: ignore[attr-defined]
    log.info(
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

    return subset
