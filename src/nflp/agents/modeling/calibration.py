"""Calibración de probabilidades y distribuciones predictivas.

Un modelo puede acertar mucho y mentir sistemáticamente sobre su confianza. Para
decidir sobre una predicción —apostar, o simplemente confiar— lo que importa es
que, cuando el modelo dice 70 %, ocurra el 70 % de las veces.

Dos mecanismos:

* **Clasificación** — calibración isotónica (o de Platt) ajustada sobre un
  conjunto de validación *posterior* en el tiempo al de entrenamiento. Ajustarla
  sobre los mismos datos de entrenamiento no calibra nada: los memoriza.
* **Regresión** — el GBM devuelve un punto; la incertidumbre se estima a partir
  de la dispersión de los residuos de validación. Eso convierte "el margen será
  +3,2" en una distribución sobre la que sí se puede calcular P(cubrir).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from nflp.core.logging import get_logger

log = get_logger(__name__)

# Muestra mínima para que la calibración isotónica sea estable. Por debajo se
# usa Platt automáticamente. Una temporada NFL completa son ~285 partidos, así
# que en la práctica casi siempre se acaba usando Platt — y está bien.
ISOTONIC_MIN_SAMPLES = 1500


@dataclass
class ProbabilityCalibrator:
    """Calibrador de probabilidades ajustado sobre validación temporal."""

    method: str = "isotonic"
    _model: object | None = None
    _fitted: bool = False

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> ProbabilityCalibrator:
        probability = np.asarray(probabilities, dtype=float).ravel()
        outcome = np.asarray(outcomes, dtype=float).ravel()
        valid = ~(np.isnan(probability) | np.isnan(outcome))
        probability, outcome = probability[valid], outcome[valid]

        # La isotónica es no paramétrica y necesita MUCHA muestra: con ~300
        # partidos ajusta una escalera que no generaliza y empeora el log-loss
        # fuera de muestra (verificado empíricamente en este dataset: 0,62 ->
        # 0,73). Platt, al tener dos parámetros, es estable con cientos de casos.
        # El umbral es alto a propósito: una temporada NFL son ~285 partidos.
        if probability.size < ISOTONIC_MIN_SAMPLES and self.method == "isotonic":
            log.info(
                "Muestra de calibración pequeña (%d < %d): se usa Platt en vez de isotónica.",
                probability.size, ISOTONIC_MIN_SAMPLES,
            )
            self.method = "sigmoid"

        if self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(probability, outcome)
        else:
            model = LogisticRegression(C=1e6, solver="lbfgs")
            model.fit(probability.reshape(-1, 1), outcome)

        self._model = model
        self._fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            return np.asarray(probabilities, dtype=float)

        probability = np.asarray(probabilities, dtype=float).ravel()
        if self.method == "isotonic":
            calibrated = self._model.predict(probability)  # type: ignore[union-attr]
        else:
            calibrated = self._model.predict_proba(probability.reshape(-1, 1))[:, 1]  # type: ignore[union-attr]

        # Nunca devolvemos 0 ni 1 exactos: el log-loss diverge.
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


@dataclass
class ResidualDistribution:
    """Distribución predictiva estimada a partir de los residuos de validación.

    Convierte una predicción puntual en una distribución, que es lo que permite
    responder "¿cuál es la probabilidad de que el total supere 47,5?".
    """

    sigma: float = 13.5
    residual_mean: float = 0.0

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> ResidualDistribution:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(y_pred, dtype=float)
        valid = ~(np.isnan(truth) | np.isnan(prediction))
        residuals = truth[valid] - prediction[valid]

        if residuals.size < 30:
            log.warning("Pocos residuos (%d): se conserva sigma por defecto.", residuals.size)
            return self

        self.residual_mean = float(np.mean(residuals))
        # Desviación robusta: la NFL produce palizas ocasionales que inflan la
        # desviación clásica. El MAD escalado resiste mejor esos extremos.
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        robust_sigma = mad * 1.4826
        classic_sigma = float(np.std(residuals, ddof=1))
        self.sigma = max(robust_sigma, classic_sigma * 0.85, 1e-3)

        log.debug(
            "Distribución de residuos: sigma=%.2f (robusta=%.2f, clásica=%.2f), sesgo=%.2f",
            self.sigma, robust_sigma, classic_sigma, self.residual_mean,
        )
        return self

    def probability_over(self, predictions: np.ndarray, line: np.ndarray | float) -> np.ndarray:
        """P(valor real > línea) dada la predicción puntual."""
        from scipy.stats import norm

        mu = np.asarray(predictions, dtype=float) + self.residual_mean
        threshold = np.asarray(line, dtype=float)
        return np.clip(1.0 - norm.cdf(threshold, loc=mu, scale=self.sigma), 1e-6, 1 - 1e-6)

    def interval(
        self, predictions: np.ndarray, confidence: float = 0.80
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intervalo central de la distribución predictiva."""
        from scipy.stats import norm

        mu = np.asarray(predictions, dtype=float) + self.residual_mean
        z = norm.ppf(0.5 + confidence / 2.0)
        return mu - z * self.sigma, mu + z * self.sigma


def confidence_level(probability: pd.Series | np.ndarray) -> pd.Series:
    """Traduce una probabilidad a un nivel de confianza legible.

    Se basa en la distancia a 0,5: una probabilidad de 0,52 es prácticamente un
    lanzamiento de moneda y el dashboard debe decirlo, no disfrazarlo.
    """
    values = pd.Series(np.asarray(probability, dtype=float))
    distance = (values - 0.5).abs()
    return pd.Series(
        np.select(
            [distance >= 0.20, distance >= 0.10],
            ["alta", "media"],
            default="baja",
        ),
        index=values.index,
    )


def margin_to_win_probability(margin: np.ndarray, sigma: float = 13.5) -> np.ndarray:
    """Convierte un margen esperado en probabilidad de victoria local."""
    from scipy.stats import norm

    return np.clip(norm.cdf(np.asarray(margin, dtype=float) / sigma), 1e-6, 1 - 1e-6)
