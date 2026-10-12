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

* **Conteos sobredispersos** (sacks): binomial negativa con media ``μ`` = la
  misma predicción del modelo Poisson y tamaño ``r`` (``Var = μ + μ²/r``)
  ajustado por máxima verosimilitud **solo en la temporada de calibración**,
  constante o log-lineal en la media (se elige por AIC en calibración). La
  Poisson es el límite ``r → ∞``.

El CRPS de cada familia puntúa la distribución que realmente se sirve:

* continuos → ``2 · media_tau pinball_tau(y, q_tau)`` sobre la rejilla;
* Poisson y binomial negativa → ``Σ_k (F(k) − 1{y ≤ k})²``, exacto.

La **media** de la distribución servida (``mean_estimate``) es λ en Poisson,
μ en binomial negativa y, en la rejilla, la integral de la función cuantil con
las mismas colas que usa la CDF de la rejilla (ver ``grid_mean``).
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


def grid_tail_slopes(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pendientes de las colas de la rejilla, en unidades del objetivo por unidad z.

    Cola inferior: la del tramo p05–p10 en escala normal; superior: p90–p95.
    Si un tramo es plano (p. ej. p05 = p10 = 0 por el suelo) se usa la
    pendiente global p05–p95. Es la misma extrapolación que usa la CDF de la
    rejilla en el modelo de parlays (``parlay.grid_cdf``).
    """
    from scipy.stats import norm

    grid = np.asarray(grid, dtype=float)
    z = norm.ppf(_LEVELS)
    overall = np.maximum((grid[:, -1] - grid[:, 0]) / (z[-1] - z[0]), 1e-6)
    low = (grid[:, 1] - grid[:, 0]) / (z[1] - z[0])
    high = (grid[:, -1] - grid[:, -2]) / (z[-1] - z[-2])
    return np.where(low > 1e-9, low, overall), np.where(high > 1e-9, high, overall)


def grid_mean(grid: np.ndarray, lower_bound: float | None = 0.0) -> np.ndarray:
    """Media implícita de una rejilla p05..p95: ``E[Y] = ∫_0^1 Q(u) du``.

    * Entre p05 y p95, ``Q`` es lineal a trozos en ``u`` (la CDF de la rejilla
      interpola linealmente en probabilidad): regla del trapecio, exacta.
    * Colas: ``Q(u) = q_extremo + s·(Φ⁻¹(u) − z_extremo)``, lineal en escala
      normal con la pendiente del tramo extremo (``grid_tail_slopes``), y la
      cola inferior acotada por el suelo de la rejilla. Integral cerrada con
      ``∫_0^a Φ⁻¹(u) du = −φ(Φ⁻¹(a))``.

    Con una rejilla normal exacta devuelve la media de la normal.
    """
    from scipy.stats import norm

    grid = np.sort(np.asarray(grid, dtype=float), axis=1)
    taus = _LEVELS
    a, b = float(taus[0]), float(taus[-1])
    za, zb = float(norm.ppf(a)), float(norm.ppf(b))
    s_low, s_high = grid_tail_slopes(grid)

    interior = (np.diff(taus)[None, :] * 0.5 * (grid[:, :-1] + grid[:, 1:])).sum(axis=1)

    q0 = grid[:, 0]
    if lower_bound is None:
        lower = a * (q0 - s_low * za) - s_low * norm.pdf(za)
    else:
        # Por debajo de u* = Φ(z*) la cola toca el suelo y queda en lower_bound.
        z_star = za - np.maximum(q0 - lower_bound, 0.0) / s_low
        u_star = norm.cdf(z_star)
        lower = (
            lower_bound * u_star
            + (a - u_star) * (q0 - s_low * za)
            + s_low * (norm.pdf(z_star) - norm.pdf(za))
        )
    q1 = grid[:, -1]
    upper = (1.0 - b) * (q1 - s_high * zb) + s_high * norm.pdf(zb)
    return interior + lower + upper


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


def count_cdf(k: np.ndarray, mu: np.ndarray, size: np.ndarray | None = None) -> np.ndarray:
    """CDF de un conteo: Poisson(μ) o, con ``size`` finito, binomial negativa(μ, r).

    Parametrización NB2: ``Var = μ + μ²/r``; en scipy, ``nbinom(n=r, p=r/(r+μ))``.
    Las filas con ``size`` NaN o infinito son Poisson (el límite ``r → ∞``).
    """
    from scipy.stats import nbinom, poisson

    if size is None:
        return poisson.cdf(np.asarray(k, dtype=float), np.maximum(np.asarray(mu, dtype=float), 1e-9))
    k, mu, size = np.broadcast_arrays(
        np.asarray(k, dtype=float),
        np.maximum(np.asarray(mu, dtype=float), 1e-9),
        np.asarray(size, dtype=float),
    )
    out = np.empty(k.shape)
    is_nb = np.isfinite(size)
    if is_nb.any():
        r = size[is_nb]
        out[is_nb] = nbinom.cdf(k[is_nb], r, r / (r + mu[is_nb]))
    if (~is_nb).any():
        out[~is_nb] = poisson.cdf(k[~is_nb], mu[~is_nb])
    return out


def count_ppf(q: np.ndarray | float, mu: np.ndarray, size: np.ndarray | None = None) -> np.ndarray:
    """Cuantil de un conteo (Poisson o binomial negativa, ver ``count_cdf``)."""
    from scipy.stats import nbinom, poisson

    if size is None:
        return poisson.ppf(q, np.maximum(np.asarray(mu, dtype=float), 1e-9)).astype(float)
    q, mu, size = np.broadcast_arrays(
        np.asarray(q, dtype=float),
        np.maximum(np.asarray(mu, dtype=float), 1e-9),
        np.asarray(size, dtype=float),
    )
    out = np.empty(q.shape)
    is_nb = np.isfinite(size)
    if is_nb.any():
        r = size[is_nb]
        out[is_nb] = nbinom.ppf(q[is_nb], r, r / (r + mu[is_nb]))
    if (~is_nb).any():
        out[~is_nb] = poisson.ppf(q[~is_nb], mu[~is_nb])
    return out


def crps_count_rows(
    y: np.ndarray, mu: np.ndarray, size: np.ndarray | None = None, *, chunk: int = 5000
) -> np.ndarray:
    """CRPS exacto por fila de un conteo: Poisson(μ) o binomial negativa(μ, r).

    ``CRPS = Σ_{k≥0} (F(k) − 1{y ≤ k})²``. La suma se trunca donde la cola es
    despreciable (``F(k) > 1 − 1e-10`` en la fila de cola más larga del bloque)
    y siempre por encima de ``y``.
    """
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    sizes = None if size is None else np.broadcast_to(np.asarray(size, dtype=float), y.shape)
    out = np.full(y.shape, np.nan)
    valid = ~(np.isnan(y) | np.isnan(mu))
    idx = np.flatnonzero(valid)
    for start in range(0, idx.size, chunk):
        rows = idx[start : start + chunk]
        yv, mv = y[rows], mu[rows]
        sv = None if sizes is None else sizes[rows]
        tail = count_ppf(1 - 1e-10, mv, sv)
        upper = int(max(np.max(yv), np.max(tail))) + 2
        ks = np.arange(upper + 1, dtype=float)
        cdf = count_cdf(ks[None, :], mv[:, None], None if sv is None else sv[:, None])
        indicator = (yv[:, None] <= ks[None, :]).astype(float)
        out[rows] = ((cdf - indicator) ** 2).sum(axis=1)
    return out


def crps_count(y: np.ndarray, mu: np.ndarray, size: np.ndarray | None = None) -> float:
    """CRPS exacto medio de un conteo (ver ``crps_count_rows``)."""
    rows = crps_count_rows(y, mu, size)
    rows = rows[~np.isnan(rows)]
    return float(rows.mean()) if rows.size else float("nan")


def crps_poisson_rows(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """CRPS exacto por fila de una Poisson(λ) frente a conteos enteros."""
    return crps_count_rows(y, lam)


def crps_poisson(y: np.ndarray, lam: np.ndarray) -> float:
    """CRPS exacto medio de una Poisson(λ)."""
    return crps_count(y, lam)


def randomized_pit_covered(
    y: np.ndarray,
    lam: np.ndarray,
    *,
    size: np.ndarray | None = None,
    confidence: float = 0.80,
    random_state: int = 1729,
) -> np.ndarray:
    """Indicador por fila de PIT aleatorizado dentro del 80 % central (NaN si falta dato).

    Un intervalo entero ``[ppf(0,1), ppf(0,9)]`` de una Poisson nunca cubre
    exactamente el 80 % (con λ=0,3 cubre el 96 %), así que su cobertura
    literal no dice si la distribución está bien calibrada. El PIT aleatorizado
    ``u = F(y−1) + V·(F(y) − F(y−1))``, con ``V ~ U(0,1)``, es uniforme si el
    modelo es correcto; la fracción de ``u`` en [0,1; 0,9] debe ser ≈ 0,80.
    Con ``size`` la CDF es la de la binomial negativa(λ, r).
    """
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)
    out = np.full(y.shape, np.nan)
    valid = ~(np.isnan(y) | np.isnan(lam))
    if not valid.any():
        return out
    sv = None if size is None else np.broadcast_to(np.asarray(size, dtype=float), y.shape)[valid]
    rng = np.random.default_rng(random_state)
    low = count_cdf(y[valid] - 1, lam[valid], sv)
    high = count_cdf(y[valid], lam[valid], sv)
    u = low + rng.uniform(size=low.size) * (high - low)
    tail = (1.0 - confidence) / 2.0
    out[valid] = ((u >= tail) & (u <= 1.0 - tail)).astype(float)
    return out


def randomized_pit_coverage(
    y: np.ndarray,
    lam: np.ndarray,
    *,
    size: np.ndarray | None = None,
    confidence: float = 0.80,
    random_state: int = 1729,
) -> float:
    """Cobertura central del 80 % por PIT aleatorizado (ver ``randomized_pit_covered``)."""
    rows = randomized_pit_covered(
        y, lam, size=size, confidence=confidence, random_state=random_state
    )
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

    def mean(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        """Media de la Poisson: λ."""
        return self.lam(point)

    def size(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        """Sin dispersión extra: tamaño infinito (Poisson)."""
        return np.full(np.asarray(point).shape, np.inf)


# =============================================================================
# Binomial negativa (conteos sobredispersos)
# =============================================================================


def negbin_loglik_rows(y: np.ndarray, mu: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Log-verosimilitud por fila de la NB2(μ, r): estable con ``gammaln``."""
    from scipy.special import gammaln

    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    r = np.asarray(size, dtype=float)
    return (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
        + r * (np.log(r) - np.log(r + mu)) + y * (np.log(mu) - np.log(r + mu))
    )


def poisson_loglik_rows(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    return y * np.log(mu) - mu - gammaln(y + 1.0)


@dataclass
class NegativeBinomialDistribution:
    """Binomial negativa NB2: media ``μ`` = predicción puntual, tamaño ``r`` ajustado.

    ``Var(Y) = μ + μ²/r``. El tamaño puede depender de la media:
    ``log r = a + b·log μ`` (``b = 0`` → ``r`` constante). Ambas variantes se
    ajustan por máxima verosimilitud sobre la temporada de **calibración**
    (posterior al entrenamiento del modelo de media, anterior a validación) y
    se elige la de menor AIC. Nunca se ajusta con validación ni con prueba.
    """

    kind: str = "negbin"
    log_size_intercept: float = float(np.log(10.0))
    log_size_slope: float = 0.0
    min_lambda: float = 1e-6
    min_size: float = 0.05
    max_size: float = 1e4
    calibration_coverage: float = float("nan")
    calibration: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------- ajuste

    def _size_from(self, mu: np.ndarray, a: float, b: float) -> np.ndarray:
        log_r = a + b * np.log(np.maximum(mu, self.min_lambda))
        return np.clip(np.exp(np.clip(log_r, -20.0, 20.0)), self.min_size, self.max_size)

    def fit(self, y: np.ndarray, point: np.ndarray) -> NegativeBinomialDistribution:
        """MLE del tamaño ``r`` con la media fija (predicción del modelo de media).

        ``y`` y ``point`` deben ser de la temporada de calibración.
        """
        from scipy.optimize import minimize, minimize_scalar

        y = np.asarray(y, dtype=float)
        mu = self.lam(point)
        valid = ~(np.isnan(y) | np.isnan(mu))
        y, mu = y[valid], mu[valid]
        if y.size < 50:
            log.warning("Pocos datos para ajustar la dispersión NB; se deja r por defecto.")
            return self

        lo, hi = np.log(self.min_size), np.log(self.max_size)

        def nll(a: float, b: float) -> float:
            return -float(negbin_loglik_rows(y, mu, self._size_from(mu, a, b)).sum())

        const = minimize_scalar(lambda a: nll(a, 0.0), bounds=(lo, hi), method="bounded")
        a_const = float(const.x)
        nll_const = float(const.fun)
        slope = minimize(
            lambda t: nll(float(t[0]), float(t[1])), x0=np.array([a_const, 0.0]),
            method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
        )
        a_slope, b_slope = float(slope.x[0]), float(slope.x[1])
        nll_slope = float(slope.fun)

        nll_pois = -float(poisson_loglik_rows(y, mu).sum())
        aic_const = 2.0 * nll_const + 2.0
        aic_slope = 2.0 * nll_slope + 4.0
        if aic_slope < aic_const:
            self.log_size_intercept, self.log_size_slope = a_slope, b_slope
        else:
            self.log_size_intercept, self.log_size_slope = a_const, 0.0
        self.calibration_coverage = randomized_pit_coverage(y, mu, size=self.size(mu))
        self.calibration = {
            "n": float(y.size),
            "size_const": float(self._size_from(np.ones(1), a_const, 0.0)[0]),
            "aic_poisson": 2.0 * nll_pois,
            "aic_nb_const": aic_const,
            "aic_nb_mean": aic_slope,
            "log_size_intercept": self.log_size_intercept,
            "log_size_slope": self.log_size_slope,
            "mean_y": float(y.mean()),
            "mean_mu": float(mu.mean()),
            "var_over_mean": float(y.var() / max(y.mean(), 1e-9)),
            "coverage_80": self.calibration_coverage,
        }
        log.info(
            "NB calibrada: log r = %.3f %+.3f·log μ (r medio %.2f) · AIC Poisson %.1f, "
            "NB const %.1f, NB(μ) %.1f · cobertura 80%% en calibración %.3f",
            self.log_size_intercept, self.log_size_slope, float(np.mean(self.size(mu))),
            2.0 * nll_pois, aic_const, aic_slope, self.calibration_coverage,
        )
        return self

    # ----------------------------------------------------------- predicción

    def lam(self, point: np.ndarray) -> np.ndarray:
        """Media ``μ`` (mismo nombre que en la Poisson por compatibilidad)."""
        return np.maximum(np.asarray(point, dtype=float), self.min_lambda)

    def size(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        return self._size_from(self.lam(point), self.log_size_intercept, self.log_size_slope)

    def mean(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        return self.lam(point)

    def quantiles(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        mu = self.lam(point)
        return count_ppf(_LEVELS[None, :], mu[:, None], self.size(mu)[:, None])

    def interval(
        self, point: np.ndarray, X: pd.DataFrame | None = None, confidence: float = 0.80
    ) -> tuple[np.ndarray, np.ndarray]:
        tail = (1.0 - confidence) / 2.0
        mu = self.lam(point)
        size = self.size(mu)
        return count_ppf(tail, mu, size), count_ppf(1.0 - tail, mu, size)

    def std(self, point: np.ndarray, X: pd.DataFrame | None = None) -> np.ndarray:
        mu = self.lam(point)
        return np.sqrt(mu + mu**2 / self.size(mu))

    def probability_over(self, point: np.ndarray, line: np.ndarray | float) -> np.ndarray:
        """P(Y > línea)."""
        mu = self.lam(point)
        return 1.0 - count_cdf(np.floor(np.asarray(line, dtype=float)), mu, self.size(mu))

    def crps_rows(self, y: np.ndarray, point: np.ndarray) -> np.ndarray:
        mu = self.lam(point)
        return crps_count_rows(y, mu, self.size(mu))

    def crps(self, y: np.ndarray, point: np.ndarray, X: pd.DataFrame | None = None) -> float:
        rows = self.crps_rows(y, point)
        rows = rows[~np.isnan(rows)]
        return float(rows.mean()) if rows.size else float("nan")

    def coverage(self, y: np.ndarray, point: np.ndarray, X: pd.DataFrame | None = None) -> float:
        """Cobertura del 80 % por PIT aleatorizado con la CDF de la NB."""
        mu = self.lam(point)
        return randomized_pit_coverage(y, mu, size=self.size(mu))


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
    # Desplazamiento conformal de la estimación puntual (mediana L1) para que
    # P(y < punto) ≈ 0,5 en calibración (ver ``calibrate_point``).
    point_shift: float = 0.0
    point_calibration: dict[str, float] = field(default_factory=dict)

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

    def calibrate_point(self, y: np.ndarray, point: np.ndarray) -> QuantileGridDistribution:
        """Centra la mediana puntual (modelo L1) en la temporada de calibración.

        El modelo puntual se entrena con L1 en temporadas anteriores y su
        mediana puede quedar descentrada en temporadas posteriores. Se le suma
        el cuantil conformal 0,5 de los residuos de calibración ``y − punto``,
        de modo que en calibración ``P(y < punto) ≈ 0,5``. El objetivo L1 no
        cambia y la rejilla tampoco (su p50 ya se centra en el paso 1 de
        ``calibrate``), así que sigue monótona.
        """
        truth = np.asarray(y, dtype=float)
        point = np.asarray(point, dtype=float)
        valid = ~(np.isnan(truth) | np.isnan(point))
        if valid.sum() < 50:
            return self
        residuals = truth[valid] - point[valid]
        self.point_shift = _conformal_quantile(residuals, 0.5)
        shifted = self.shift_point(point[valid])
        self.point_calibration = {
            "shift": float(self.point_shift),
            "frac_below_before": float(np.mean(truth[valid] < point[valid])),
            "frac_below_after": float(np.mean(truth[valid] < shifted)),
        }
        return self

    def shift_point(self, point: np.ndarray) -> np.ndarray:
        """Estimación puntual servida: punto L1 + desplazamiento, con suelo 0."""
        # getattr: los artefactos anteriores a este campo no lo tienen.
        shift = float(getattr(self, "point_shift", 0.0))
        return np.maximum(np.asarray(point, dtype=float) + shift, 0.0)

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

    def mean(self, point: np.ndarray | None, X: pd.DataFrame) -> np.ndarray:
        """Media de la distribución servida (ver ``grid_mean``)."""
        return grid_mean(self.quantiles(point, X), self.lower_bound)

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
