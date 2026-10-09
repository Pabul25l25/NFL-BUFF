"""Distribuciones predictivas por jugador para los props y su evaluación.

Dos familias, según la naturaleza del objetivo:

* **Continuos** (yardas, recepciones): rejilla de 19 cuantiles p05..p95 por
  jugador, producida por LightGBM con pérdida cuantil (un modelo por nivel).
  La rejilla se ordena (monotonía garantizada aunque los modelos se crucen) y
  se **calibra de forma conformal** sobre la temporada de calibración, que es
  posterior a la de entrenamiento:

  1. Por nivel ``tau``: se desplaza el cuantil por el cuantil ``tau`` de los
     residuos de calibración ``y - q_tau(x)``. Así, en calibración,
     ``P(y <= q_tau) ≈ tau`` para cada nivel (CQR generalizado a la rejilla).
  2. Un único factor de escala alrededor de la mediana, elegido para que el
     intervalo p10–p90 cubra el 80 % en calibración. Escalar alrededor de la
     mediana conserva la monotonía de la rejilla.

  La dispersión pasa a depender del jugador: un receptor estrella y un
  tercer tight end ya no comparten la misma anchura relativa.

* **Conteos dispersos** (touchdowns, sacks): Poisson con ``λ`` = predicción
  puntual (modelo con objetivo Poisson, que estima la media). El intervalo y
  los cuantiles salen de la Poisson; el CRPS es exacto.

El CRPS de cada familia puntúa la distribución que realmente se sirve:

* continuos → ``2 · media_tau pinball_tau(y, q_tau)`` sobre la rejilla;
* Poisson → ``Σ_k (F(k) − 1{y ≤ k})²``, exacto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nflp.core.logging import get_logger

log = get_logger(__name__)

# p05, p10, ..., p95 — 19 niveles equiespaciados.
QUANTILE_LEVELS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
_LEVELS = np.asarray(QUANTILE_LEVELS)
# Índices del intervalo central del 80 % y de la mediana dentro de la rejilla.
P10_INDEX = QUANTILE_LEVELS.index(0.10)
P50_INDEX = QUANTILE_LEVELS.index(0.50)
P90_INDEX = QUANTILE_LEVELS.index(0.90)


# =============================================================================
# Utilidades de rejilla
# =============================================================================


def make_monotone(grid: np.ndarray) -> np.ndarray:
    """Ordena cada fila: los cuantiles de un mismo jugador nunca se cruzan."""
    return np.sort(np.asarray(grid, dtype=float), axis=1)


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Cuantil empírico con la corrección de muestra finita del split conformal."""
    n = scores.size
    if n == 0:
        return 0.0
    # Nivel corregido: ceil((n+1)·τ)/n por arriba de la mediana, floor por
    # debajo — el cuantil se aleja de la mediana lo justo para garantizar la
    # cobertura unilateral en muestra finita.
    if level >= 0.5:
        adjusted = min(np.ceil((n + 1) * level) / n, 1.0)
    else:
        adjusted = max(np.floor((n + 1) * level) / n, 0.0)
    return float(np.quantile(scores, adjusted))


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fracción de observaciones dentro de [lower, upper] (inclusivo)."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = ~(np.isnan(y) | np.isnan(lower) | np.isnan(upper))
    if not valid.any():
        return float("nan")
    return float(np.mean((y[valid] >= lower[valid]) & (y[valid] <= upper[valid])))


# =============================================================================
# CRPS
# =============================================================================


def pinball_loss(y: np.ndarray, quantiles: np.ndarray, levels: np.ndarray | None = None) -> np.ndarray:
    """Pérdida pinball por fila y nivel: matriz (n, n_levels)."""
    taus = np.asarray(QUANTILE_LEVELS if levels is None else levels, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    diff = y - np.asarray(quantiles, dtype=float)
    return np.maximum(taus * diff, (taus - 1.0) * diff)


def crps_quantile_grid_rows(
    y: np.ndarray, quantiles: np.ndarray, levels: np.ndarray | None = None
) -> np.ndarray:
    """CRPS por fila a partir de una rejilla de cuantiles (NaN si falta dato).

    ``CRPS(F, y) = 2 ∫_0^1 ρ_τ(y − F⁻¹(τ)) dτ``; la integral se aproxima con la
    media sobre los niveles de la rejilla. Con 19 niveles equiespaciados la
    aproximación ignora las colas extremas (τ < 0,05 y τ > 0,95), igual para
    todos los modelos comparados, así que la comparación es justa.
    """
    loss = pinball_loss(y, quantiles, levels)
    return 2.0 * loss.mean(axis=1)


def crps_quantile_grid(
    y: np.ndarray, quantiles: np.ndarray, levels: np.ndarray | None = None
) -> float:
    """CRPS medio de una rejilla de cuantiles (ver ``crps_quantile_grid_rows``)."""
    rows = crps_quantile_grid_rows(y, quantiles, levels)
    rows = rows[~np.isnan(rows)]
    return float(rows.mean()) if rows.size else float("nan")


def crps_poisson_rows(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """CRPS exacto por fila de una Poisson(λ) frente a conteos enteros.

    ``CRPS = Σ_{k≥0} (F(k) − 1{y ≤ k})²``. La suma se trunca donde la cola es
    despreciable (F(k) > 1 − 1e-10) y siempre por encima de ``y``.
    """
    from scipy.stats import poisson

    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)
    out = np.full(y.shape, np.nan)
    valid = ~(np.isnan(y) | np.isnan(lam))
    if not valid.any():
        return out
    yv, lv = y[valid], lam[valid]
    upper = int(max(np.max(yv), poisson.ppf(1 - 1e-10, np.max(lv)))) + 2
    ks = np.arange(upper + 1, dtype=float)
    cdf = poisson.cdf(ks[None, :], lv[:, None])
    indicator = (yv[:, None] <= ks[None, :]).astype(float)
    out[valid] = ((cdf - indicator) ** 2).sum(axis=1)
    return out


def crps_poisson(y: np.ndarray, lam: np.ndarray) -> float:
    """CRPS exacto medio de una Poisson(λ)."""
    rows = crps_poisson_rows(y, lam)
    rows = rows[~np.isnan(rows)]
    return float(rows.mean()) if rows.size else float("nan")


def randomized_pit_covered(
    y: np.ndarray, lam: np.ndarray, *, confidence: float = 0.80, random_state: int = 1729
) -> np.ndarray:
    """Indicador por fila de PIT aleatorizado dentro del 80 % central (NaN si falta dato).

    Un intervalo entero ``[ppf(0,1), ppf(0,9)]`` de una Poisson nunca cubre
    exactamente el 80 % (con λ=0,3 cubre el 96 %), así que su cobertura
    literal no dice si la distribución está bien calibrada. El PIT aleatorizado
    ``u = F(y−1) + V·(F(y) − F(y−1))``, con ``V ~ U(0,1)``, es uniforme si el
    modelo es correcto; la fracción de ``u`` en [0,1; 0,9] debe ser ≈ 0,80.
    """
    from scipy.stats import poisson

    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)
    out = np.full(y.shape, np.nan)
    valid = ~(np.isnan(y) | np.isnan(lam))
    if not valid.any():
        return out
    rng = np.random.default_rng(random_state)
    low = poisson.cdf(y[valid] - 1, lam[valid])
    high = poisson.cdf(y[valid], lam[valid])
    u = low + rng.uniform(size=low.size) * (high - low)
    tail = (1.0 - confidence) / 2.0
    out[valid] = ((u >= tail) & (u <= 1.0 - tail)).astype(float)
    return out


def randomized_pit_coverage(
    y: np.ndarray, lam: np.ndarray, *, confidence: float = 0.80, random_state: int = 1729
) -> float:
    """Cobertura central del 80 % por PIT aleatorizado (ver ``randomized_pit_covered``)."""
    rows = randomized_pit_covered(y, lam, confidence=confidence, random_state=random_state)
    rows = rows[~np.isnan(rows)]
    return float(rows.mean()) if rows.size else float("nan")


# =============================================================================
# Poisson
# =============================================================================


@dataclass
class PoissonDistribution:
    """Distribución de conteo: Poisson con λ igual a la predicción puntual."""

    kind: str = "poisson"
    min_lambda: float = 1e-6

    def lam(self, point: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(point, dtype=float), self.min_lambda)

    def quantiles(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        from scipy.stats import poisson

        return poisson.ppf(_LEVELS[None, :], self.lam(point)[:, None]).astype(float)

    def interval(
        self, point: np.ndarray, X: pd.DataFrame | None = None, confidence: float = 0.80
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.stats import poisson

        tail = (1.0 - confidence) / 2.0
        lam = self.lam(point)
        return poisson.ppf(tail, lam).astype(float), poisson.ppf(1.0 - tail, lam).astype(float)

    def std(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        return np.sqrt(self.lam(point))

    def probability_over(self, point: np.ndarray, line: np.ndarray | float) -> np.ndarray:
        """P(Y > línea). Con línea 0,5 es P(al menos un evento)."""
        from scipy.stats import poisson

        return poisson.sf(np.floor(np.asarray(line, dtype=float)), self.lam(point))

    def crps(self, y: np.ndarray, point: np.ndarray, X: pd.DataFrame | None = None) -> float:
        return crps_poisson(y, self.lam(point))

    def coverage(self, y: np.ndarray, point: np.ndarray, X: pd.DataFrame | None = None) -> float:
        """Cobertura del 80 % por PIT aleatorizado (la literal se reporta aparte)."""
        return randomized_pit_coverage(y, self.lam(point))


# =============================================================================
# Rejilla de cuantiles (LightGBM cuantil + calibración conformal)
# =============================================================================


@dataclass
class QuantileGridDistribution:
    """Rejilla p05..p95 por jugador, monótona y calibrada de forma conformal."""

    feature_names: list[str]
    models: list[Any] = field(default_factory=list)
    level_shift: np.ndarray = field(default_factory=lambda: np.zeros(len(QUANTILE_LEVELS)))
    scale: float = 1.0
    lower_bound: float | None = 0.0
    calibration_coverage: float = float("nan")
    kind: str = "quantile_grid"

    # ------------------------------------------------------------- ajuste

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> QuantileGridDistribution:
        """Un LightGBM con pérdida cuantil por nivel de la rejilla."""
        from lightgbm import LGBMRegressor

        base = {
            k: v for k, v in params.items()
            if k not in {"objective", "alpha", "tweedie_variance_power"}
        }
        features = X.reindex(columns=self.feature_names).astype(float)
        target = np.asarray(y, dtype=float)
        self.models = []
        for tau in QUANTILE_LEVELS:
            model = LGBMRegressor(**base, objective="quantile", alpha=tau)
            model.fit(features, target)
            self.models.append(model)
        return self

    def raw_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        features = X.reindex(columns=self.feature_names).astype(float)
        grid = np.column_stack([m.predict(features) for m in self.models])
        return make_monotone(grid)

    def calibrate(self, X: pd.DataFrame, y: np.ndarray, *, target_coverage: float = 0.80) -> QuantileGridDistribution:
        """Calibración conformal sobre datos posteriores al entrenamiento."""
        truth = np.asarray(y, dtype=float)
        valid = ~np.isnan(truth)
        if valid.sum() < 50:
            log.warning("Pocos datos para calibrar la rejilla de cuantiles; se deja sin calibrar.")
            return self

        raw = self.raw_quantiles(X.loc[valid] if isinstance(X, pd.DataFrame) else X[valid])
        truth = truth[valid]

        # Paso 1: desplazamiento por nivel (CQR sobre cada cuantil).
        residuals = truth[:, None] - raw
        self.level_shift = np.array(
            [_conformal_quantile(residuals[:, j], tau) for j, tau in enumerate(QUANTILE_LEVELS)]
        )
        self.scale = 1.0
        shifted = self._finalize(raw + self.level_shift)

        # Paso 2: un solo factor de escala alrededor de la mediana para que el
        # intervalo p10–p90 cubra exactamente el objetivo en calibración. Se
        # busca por bisección: la cobertura es monótona en la escala.
        median = shifted[:, P50_INDEX : P50_INDEX + 1]
        spread = shifted - median

        def coverage_at(k: float) -> float:
            grid = self._clip(median + k * spread)
            return interval_coverage(truth, grid[:, P10_INDEX], grid[:, P90_INDEX])

        low, high = 0.3, 3.0
        if coverage_at(low) > target_coverage:
            self.scale = low
        elif coverage_at(high) < target_coverage:
            self.scale = high
        else:
            for _ in range(40):
                mid = 0.5 * (low + high)
                if coverage_at(mid) < target_coverage:
                    low = mid
                else:
                    high = mid
            self.scale = high
        self.calibration_coverage = coverage_at(self.scale)
        log.debug(
            "Rejilla calibrada: escala=%.3f, cobertura 80%% en calibración=%.3f",
            self.scale, self.calibration_coverage,
        )
        return self

    # ----------------------------------------------------------- predicción

    def _clip(self, grid: np.ndarray) -> np.ndarray:
        if self.lower_bound is not None:
            grid = np.maximum(grid, self.lower_bound)
        return make_monotone(grid)

    def _finalize(self, grid: np.ndarray) -> np.ndarray:
        return self._clip(make_monotone(grid))

    def quantiles(self, point: np.ndarray | None, X: pd.DataFrame) -> np.ndarray:
        shifted = make_monotone(self.raw_quantiles(X) + self.level_shift)
        median = shifted[:, P50_INDEX : P50_INDEX + 1]
        return self._clip(median + self.scale * (shifted - median))

    def interval(
        self, point: np.ndarray | None, X: pd.DataFrame, confidence: float = 0.80
    ) -> tuple[np.ndarray, np.ndarray]:
        tail = round((1.0 - confidence) / 2.0, 2)
        grid = self.quantiles(point, X)
        return grid[:, QUANTILE_LEVELS.index(tail)], grid[:, QUANTILE_LEVELS.index(round(1 - tail, 2))]

    def std(self, point: np.ndarray | None, X: pd.DataFrame) -> np.ndarray:
        """Desviación implícita: (p90 − p10) / (2·1,2816), como en una normal."""
        grid = self.quantiles(point, X)
        return np.maximum((grid[:, P90_INDEX] - grid[:, P10_INDEX]) / 2.5631, 1e-6)

    def probability_over(self, point: np.ndarray | None, X: pd.DataFrame, line: float) -> np.ndarray:
        """P(Y > línea) interpolando linealmente la CDF de la rejilla."""
        grid = self.quantiles(point, X)
        out = np.empty(grid.shape[0])
        for i, row in enumerate(grid):
            cdf = np.interp(line, row, _LEVELS, left=0.0, right=1.0)
            out[i] = 1.0 - cdf
        return np.clip(out, 1e-6, 1 - 1e-6)

    def crps(self, y: np.ndarray, point: np.ndarray | None, X: pd.DataFrame) -> float:
        return crps_quantile_grid(y, self.quantiles(point, X))

    def coverage(self, y: np.ndarray, point: np.ndarray | None, X: pd.DataFrame) -> float:
        lower, upper = self.interval(point, X)
        return interval_coverage(y, lower, upper)


def lognormal_quantiles(
    distribution: Any, predictions: np.ndarray, levels: np.ndarray | None = None
) -> np.ndarray:
    """Rejilla de cuantiles de la log-normal desplazada heredada (``PropDistribution``).

    Por defecto p05..p95. Sirve para puntuar el modelo de producción actual con
    el mismo CRPS que el nuevo, y para persistir cuantiles aunque el target
    siga en el modelo antiguo.
    """
    from scipy.stats import norm

    taus = _LEVELS if levels is None else np.asarray(levels, dtype=float)
    mu = np.maximum(np.asarray(predictions, dtype=float), 0.0) * distribution.residual_scale
    z = norm.ppf(taus)[None, :]
    grid = np.exp(np.log(mu + distribution.shift)[:, None] + z * distribution.sigma_log) - distribution.shift
    return make_monotone(np.maximum(grid, 0.0))


# Rejilla fina (regla del punto medio) para un CRPS prácticamente exacto de
# cualquier distribución dada su función cuantil.
FINE_LEVELS = (np.arange(200) + 0.5) / 200.0


def crps_fine_rows(y: np.ndarray, quantile_fn: Any, *, chunk: int = 20000) -> np.ndarray:
    """CRPS por fila con 200 niveles: converge al CRPS exacto (también en discretas)."""
    y = np.asarray(y, dtype=float)
    out = np.empty(y.shape)
    for start in range(0, y.size, chunk):
        stop = start + chunk
        grid = quantile_fn(slice(start, stop))
        out[start:stop] = crps_quantile_grid_rows(y[start:stop], grid, FINE_LEVELS)
    return out


def quantiles_to_json(grid: np.ndarray, decimals: int = 2) -> list[str]:
    """Serializa cada fila de la rejilla como lista JSON (p05..p95)."""
    import json

    return [json.dumps([round(float(v), decimals) for v in row]) for row in np.asarray(grid)]
