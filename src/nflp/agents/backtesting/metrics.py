"""Métricas de evaluación.

El prompt es explícito: **en predicción probabilística la calibración importa más
que el accuracy crudo**. Un modelo que acierta el 65 % pero dice "80 % de
probabilidad" cuando en realidad es 65 % está mintiendo, y sobre esa mentira no se
puede decidir nada.

Por eso el conjunto de métricas prioriza:

* **Log-loss** — penaliza con dureza la confianza mal puesta.
* **Brier score** — error cuadrático sobre probabilidades; descomponible.
* **ECE** (Expected Calibration Error) — desviación media entre probabilidad
  declarada y frecuencia observada.
* **Curvas de calibración** — para verlo, no solo medirlo.

Y solo después el accuracy, que es la métrica más fácil de inflar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from nflp.core.logging import get_logger

log = get_logger(__name__)

# Recorte de probabilidades: log-loss diverge en 0 y 1.
EPSILON = 1e-6


@dataclass
class ClassificationMetrics:
    log_loss: float
    brier_score: float
    ece: float
    accuracy: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "accuracy": self.accuracy,
            "n": float(self.n),
        }


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    n: int
    crps: float | None = None

    def as_dict(self) -> dict[str, float]:
        values = {"mae": self.mae, "rmse": self.rmse, "n": float(self.n)}
        if self.crps is not None:
            values["crps"] = self.crps
        return values


@dataclass
class CalibrationCurve:
    """Puntos de la curva de fiabilidad."""

    bin_lower: list[float] = field(default_factory=list)
    bin_upper: list[float] = field(default_factory=list)
    mean_predicted: list[float] = field(default_factory=list)
    observed_frequency: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bin_index": range(len(self.bin_lower)),
                "bin_lower": self.bin_lower,
                "bin_upper": self.bin_upper,
                "mean_predicted": self.mean_predicted,
                "observed_frequency": self.observed_frequency,
                "n_samples": self.counts,
            }
        )


def _clean_pair(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alinea y elimina los pares con valores nulos."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(truth) | np.isnan(prediction))
    return truth[valid], prediction[valid]


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Error de calibración esperado: |probabilidad declarada − frecuencia real|.

    Se pondera por el número de casos en cada bin, de modo que un bin casi vacío
    no distorsiona el resultado.
    """
    truth, probability = _clean_pair(y_true, y_prob)
    if truth.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probability, edges[1:-1], right=False)

    total_error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(probability[mask].mean() - truth[mask].mean())
        total_error += (count / truth.size) * gap

    return float(total_error)


def calibration_curve(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Curva de fiabilidad: probabilidad media declarada vs frecuencia observada."""
    truth, probability = _clean_pair(y_true, y_prob)
    curve = CalibrationCurve()
    if truth.size == 0:
        return curve

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probability, edges[1:-1], right=False)

    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        curve.bin_lower.append(float(edges[bin_index]))
        curve.bin_upper.append(float(edges[bin_index + 1]))
        curve.mean_predicted.append(float(probability[mask].mean()))
        curve.observed_frequency.append(float(truth[mask].mean()))
        curve.counts.append(count)

    return curve


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> ClassificationMetrics:
    """Conjunto completo de métricas probabilísticas."""
    truth, probability = _clean_pair(y_true, y_prob)
    if truth.size == 0:
        return ClassificationMetrics(np.nan, np.nan, np.nan, np.nan, 0)

    clipped = np.clip(probability, EPSILON, 1 - EPSILON)

    # `labels=[0,1]` evita que sklearn falle si un segmento solo tiene una clase.
    try:
        loss = float(log_loss(truth, clipped, labels=[0, 1]))
    except ValueError:
        loss = float("nan")

    return ClassificationMetrics(
        log_loss=loss,
        brier_score=float(brier_score_loss(truth, clipped)) if truth.size else float("nan"),
        ece=expected_calibration_error(truth, clipped, n_bins),
        accuracy=float(((clipped >= 0.5).astype(float) == truth).mean()),
        n=int(truth.size),
    )


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_std: pd.Series | np.ndarray | None = None,
) -> RegressionMetrics:
    """MAE, RMSE y, si se aporta desviación, CRPS."""
    truth, prediction = _clean_pair(y_true, y_pred)
    if truth.size == 0:
        return RegressionMetrics(np.nan, np.nan, 0)

    mae = float(mean_absolute_error(truth, prediction))
    rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))

    crps = None
    if y_std is not None:
        sigma = np.asarray(y_std, dtype=float)
        if sigma.size == truth.size:
            crps = crps_gaussian(truth, prediction, sigma)

    return RegressionMetrics(mae=mae, rmse=rmse, n=int(truth.size), crps=crps)


def crps_gaussian(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> float:
    """CRPS para una predicción gaussiana (forma cerrada).

    El CRPS evalúa la **distribución** completa, no solo el punto central. Es la
    métrica correcta para props: un modelo que acierta la media pero se equivoca
    en la dispersión produce probabilidades over/under malas.
    """
    from scipy.stats import norm

    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    standardized = (np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float)) / sigma
    score = sigma * (
        standardized * (2 * norm.cdf(standardized) - 1)
        + 2 * norm.pdf(standardized)
        - 1 / np.sqrt(np.pi)
    )
    return float(np.nanmean(score))


def ats_win_pct(
    actual_margin: pd.Series, predicted_margin: pd.Series, market_spread: pd.Series
) -> float:
    """Porcentaje de aciertos contra el spread del mercado.

    Se apuesta al lado que el modelo considera infravalorado y se mira si cubrió.
    Los empates contra la línea (push) se excluyen, como en una casa real.

    El umbral de rentabilidad con vig estándar (-110) es **52,4 %**: por debajo de
    eso, acertar más de la mitad sigue siendo perder dinero.
    """
    frame = pd.DataFrame(
        {"actual": actual_margin, "predicted": predicted_margin, "spread": market_spread}
    ).dropna()
    if frame.empty:
        return float("nan")

    # El modelo apuesta al local cuando espera que gane por más de la línea.
    bets_home = frame["predicted"] > frame["spread"]
    covered_home = frame["actual"] > frame["spread"]

    push = frame["actual"] == frame["spread"]
    decided = frame[~push]
    if decided.empty:
        return float("nan")

    wins = (bets_home[~push] == covered_home[~push]).sum()
    return float(wins / len(decided))


BREAKEVEN_ATS = 0.524  # umbral de rentabilidad con cuota -110
