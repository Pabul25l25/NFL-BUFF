"""Correlación de parlays del mismo partido: cópula gaussiana por rol.

Un parlay del mismo partido no es un producto de probabilidades: si el QB
lanza 320 yardas, su WR1 probablemente supera su línea de recepción, y si el
local gana por 14, su RB1 acumula acarreos en el último cuarto. Este módulo
estima esa dependencia y la sirve como una matriz de correlación por partido.

**Modelo.** Cada pierna (prop de jugador o pierna de partido) conserva su
distribución marginal servida — rejilla de cuantiles p05..p95 (continuos),
Poisson(λ) (conteos) o normal (margen y total del partido) — y la
dependencia se modela con una **cópula gaussiana**: la puntuación normal
``z = Φ⁻¹(PIT)`` de las piernas de un mismo partido es normal multivariante
con matriz ``R``.

**Puntuaciones normales fuera de muestra.** Las ``z`` se calculan con
predicciones walk-forward (entrenamiento ≤ S−2, calibración S−1, prueba S;
la misma maquinaria que ``PropsAgent.backtest``) y la distribución que se
sirve. Los resultados son enteros, así que el PIT es **aleatorizado**:
``u ~ U(F(y⁻), F(y))``, con ``F(y⁻) = F(y − 0,5)`` en rejillas y normales (las
líneas son x,5) y ``F(y − 1)`` en Poisson. Así el evento "más de la línea"
es exactamente ``u > F(línea)`` y la cópula de las ``u`` es una cópula válida
del vector discreto (transformación distribucional).

**Estimación por rol.** Los jugadores se etiquetan por rol dentro de su equipo
según el uso previo al partido (QB, RB1, RB2, WR1, WR2, WR3, TE1 y OTH para
el resto) y cada par de piernas por ``(rol_a, target_a, rol_b, target_b,
relación)`` con relación ∈ {same_player, teammate, opponent, game}. Las
piernas de partido usan el rol ``G``; frente a ``g:margin`` se usa el margen
con el signo del equipo del jugador. La correlación de Pearson de cada clave
se contrae hacia 0: ``ρ·n/(n+k)``, con ``k`` elegido por validación temporal
(verosimilitud compuesta gaussiana por pares en temporadas posteriores).

**Matriz por partido.** Se ensambla con las estimaciones por clave, se
proyecta a la correlación semidefinida positiva más cercana (recorte de
autovalores y renormalización de la diagonal) y, tras redondear a 3
decimales, se verifica que siga siendo semidefinida.

**Regla de publicación.** El backtest de calibración (parlays sintéticos de
2–4 piernas en temporadas de prueba, correlaciones estimadas solo con
temporadas anteriores) compara cópula e independencia. Solo se publica la
versión correlada si su ECE **y** su log loss son ≤ los de la independencia;
si no, se guarda la identidad y así consta.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from nflp.agents.modeling.distributions import QUANTILE_LEVELS
from nflp.agents.modeling.props import (
    PROP_MIN_TRAIN_SEASON,
    QUANTILE_LOWER_BOUND,
    SPARSE_COUNT_PROPS,
    train_prop_model,
)
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

# =============================================================================
# Constantes
# =============================================================================

KIND_CORRELATED = "gaussian_copula"
KIND_IDENTITY = "identity"

ROLES: tuple[str, ...] = ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "OTH")
GAME_ROLE = "G"
GAME_TARGETS: tuple[str, ...] = ("margin", "total")
RELATIONS: tuple[str, ...] = ("same_player", "teammate", "opponent", "game")
KEY_COLUMNS = ["relation", "role_a", "target_a", "role_b", "target_b"]

# Rejilla de k para el shrinkage ρ·n/(n+k).
SHRINKAGE_GRID: tuple[float, ...] = (0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200)

# Primera temporada con puntuaciones fuera de muestra (entrenamiento desde
# PROP_MIN_TRAIN_SEASON = 2016, calibración en S−1).
OOS_FIRST_SEASON = 2019
BACKTEST_TEST_SEASONS: tuple[int, ...] = (2023, 2024, 2025)

POSITION_GROUP: dict[str, str] = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "QB": "QB"}
_ROLE_SLOTS: dict[str, tuple[str, ...]] = {
    "QB": ("QB",), "RB": ("RB1", "RB2"), "WR": ("WR1", "WR2", "WR3"), "TE": ("TE1",),
}
OUT_STATUS_CODE = 3

U_EPS = 1e-6
_LEVELS = np.asarray(QUANTILE_LEVELS, dtype=float)
_ZLEVELS = norm.ppf(_LEVELS)
_QCOLS = [f"q{i:02d}" for i in range(len(QUANTILE_LEVELS))]


def player_leg(gsis_id: str, target: str) -> str:
    """Clave de pierna de jugador: ``p:<gsis_id>:<target>``."""
    return f"p:{gsis_id}:{target}"


# =============================================================================
# Roles dentro del equipo (uso previo al partido)
# =============================================================================

_USAGE_COLUMNS = [
    "passing_attempts_ewm", "passing_attempts_r5", "carries_ewm", "carries_r5",
    "targets_ewm", "targets_r5", "target_share_ewm", "target_share_r5",
    "snap_pct_ewm", "snap_pct_r5", "injury_status_code",
]


def _first_available(frame: pd.DataFrame, *names: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name in frame.columns:
            out = out.fillna(pd.to_numeric(frame[name], errors="coerce").astype(float))
    return out


def snap_threshold(snap_pct: pd.Series, min_snaps: float) -> float:
    """Mismo umbral que ``filter_prop_rows`` (snap_pct en 0-1 o en 0-100)."""
    values = pd.to_numeric(snap_pct, errors="coerce")
    return min_snaps / 100.0 if values.max() <= 1.5 else float(min_snaps)


def assign_roles(players: pd.DataFrame, *, min_snaps: float = 10.0) -> pd.DataFrame:
    """Rol de cada jugador dentro de su equipo y partido, solo con datos previos.

    El orden se basa en el uso proyectado (medias exponenciales desplazadas del
    feature store): intentos de pase para QB, acarreos + objetivos para RB y
    cuota de objetivos para WR/TE; desempate por % de snaps. Solo compiten los
    jugadores elegibles para props (umbral de snaps y sin parte de baja); el
    resto queda como ``OTH``.

    Returns:
        DataFrame ``game_id, gsis_id, role``.
    """
    frame = players.copy()
    group = frame["position"].map(POSITION_GROUP)
    qb_usage = _first_available(frame, "passing_attempts_ewm", "passing_attempts_r5")
    carries = _first_available(frame, "carries_ewm", "carries_r5")
    targets = _first_available(frame, "targets_ewm", "targets_r5")
    rb_usage = carries.fillna(0.0) + targets.fillna(0.0)
    rb_usage = rb_usage.where(carries.notna() | targets.notna())
    rec_usage = _first_available(frame, "target_share_ewm", "target_share_r5")
    usage = pd.Series(
        np.select(
            [group.eq("QB"), group.eq("RB"), group.isin(["WR", "TE"])],
            [qb_usage, rb_usage, rec_usage],
            default=np.nan,
        ),
        index=frame.index,
    )
    snaps = _first_available(frame, "snap_pct_ewm", "snap_pct_r5")

    eligible = group.notna()
    if "injury_status_code" in frame.columns:
        injury = pd.to_numeric(frame["injury_status_code"], errors="coerce")
        eligible &= ~(injury >= OUT_STATUS_CODE)
    if "snap_pct_r5" in frame.columns and frame["snap_pct_r5"].notna().any():
        threshold = snap_threshold(frame["snap_pct_r5"], min_snaps)
        eligible &= pd.to_numeric(frame["snap_pct_r5"], errors="coerce").fillna(0) >= threshold

    frame["_group"] = group
    frame["_usage"] = usage.fillna(-1.0)
    frame["_snaps"] = snaps.fillna(-1.0)
    ranked = frame.loc[eligible].sort_values(
        ["game_id", "team", "_group", "_usage", "_snaps", "gsis_id"],
        ascending=[True, True, True, False, False, True],
    )
    ranked["_rank"] = ranked.groupby(["game_id", "team", "_group"]).cumcount()

    def role_of(row_group: str, rank: int) -> str:
        slots = _ROLE_SLOTS.get(row_group, ())
        return slots[rank] if rank < len(slots) else "OTH"

    ranked["role"] = [role_of(g, int(r)) for g, r in zip(ranked["_group"], ranked["_rank"], strict=True)]
    out = frame[["game_id", "gsis_id"]].copy()
    out["role"] = "OTH"
    out.loc[ranked.index, "role"] = ranked["role"]
    return out.drop_duplicates(["game_id", "gsis_id"]).reset_index(drop=True)


# =============================================================================
# Marginales servidas: CDF y PIT aleatorizado
# =============================================================================


def grid_cdf(x: np.ndarray, grid: np.ndarray, lower_bound: float | None = 0.0) -> np.ndarray:
    """CDF de una rejilla de cuantiles p05..p95, por fila.

    Interpolación lineal en probabilidad entre cuantiles (como
    ``QuantileGridDistribution.probability_over``) y, fuera de [p05, p95],
    extrapolación lineal en escala normal con la pendiente del tramo extremo,
    para que el PIT no se sature en 0 o 1. Por debajo de ``lower_bound`` (suelo
    de la variable) la CDF es 0.
    """
    x = np.asarray(x, dtype=float).ravel()
    grid = np.sort(np.asarray(grid, dtype=float), axis=1)
    n, m = grid.shape
    out = np.empty(n)
    j = (grid <= x[:, None]).sum(axis=1) - 1

    interior = (j >= 0) & (j < m - 1)
    if interior.any():
        rows = np.flatnonzero(interior)
        jj = j[rows]
        q0, q1 = grid[rows, jj], grid[rows, jj + 1]
        out[rows] = _LEVELS[jj] + (x[rows] - q0) / (q1 - q0) * (_LEVELS[jj + 1] - _LEVELS[jj])

    # Escala (unidades del objetivo por unidad z) de cada cola.
    overall = np.maximum((grid[:, -1] - grid[:, 0]) / (_ZLEVELS[-1] - _ZLEVELS[0]), 1e-6)
    upper = j == m - 1
    if upper.any():
        slope = (grid[:, -1] - grid[:, -2]) / (_ZLEVELS[-1] - _ZLEVELS[-2])
        slope = np.where(slope > 1e-9, slope, overall)
        out[upper] = norm.cdf(_ZLEVELS[-1] + (x[upper] - grid[upper, -1]) / slope[upper])
    lower = j < 0
    if lower.any():
        slope = (grid[:, 1] - grid[:, 0]) / (_ZLEVELS[1] - _ZLEVELS[0])
        slope = np.where(slope > 1e-9, slope, overall)
        out[lower] = norm.cdf(_ZLEVELS[0] - (grid[lower, 0] - x[lower]) / slope[lower])
    if lower_bound is not None:
        out[x < lower_bound] = 0.0
    return np.clip(out, 0.0, 1.0)


def marginal_cdf(
    kind: np.ndarray,
    x: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    lower_bound: np.ndarray | None = None,
    lam: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """``F(x)`` de cada pierna según su familia (grid, poisson o normal).

    En Poisson se evalúa en ``floor(x)`` (la masa es entera). ``lower_bound``
    por fila; NaN = sin suelo.
    """
    kind = np.asarray(kind)
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    is_grid = kind == "grid"
    if is_grid.any():
        bounds = np.full(x.shape, np.nan) if lower_bound is None else np.asarray(lower_bound, float)
        rows = np.flatnonzero(is_grid)
        for bound in np.unique(np.where(np.isnan(bounds[rows]), np.inf, bounds[rows])):
            sel = rows[np.where(np.isnan(bounds[rows]), np.inf, bounds[rows]) == bound]
            out[sel] = grid_cdf(x[sel], grid[sel], None if np.isinf(bound) else float(bound))
    is_pois = kind == "poisson"
    if is_pois.any():
        out[is_pois] = poisson.cdf(np.floor(x[is_pois]), np.maximum(lam[is_pois], 1e-9))
    is_norm = kind == "normal"
    if is_norm.any():
        out[is_norm] = norm.cdf(x[is_norm], loc=mu[is_norm], scale=np.maximum(sigma[is_norm], 1e-9))
    return out


def randomized_pit(
    kind: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    **params: Any,
) -> np.ndarray:
    """PIT aleatorizado de resultados enteros: ``u ~ U(F(y⁻), F(y))``.

    * grid y normal: ``F(y⁻) = F(y − 0,5)``, ``F(y) = F(y + 0,5)`` (continuidad);
    * poisson: ``F(y − 1)`` y ``F(y)``.
    Con el modelo correcto ``u`` es U(0,1) exacta.
    """
    kind = np.asarray(kind)
    y = np.asarray(y, dtype=float)
    is_pois = kind == "poisson"
    lo_x = np.where(is_pois, y - 1.0, y - 0.5)
    hi_x = np.where(is_pois, y, y + 0.5)
    lo = marginal_cdf(kind, lo_x, **params)
    hi = marginal_cdf(kind, hi_x, **params)
    lo = np.minimum(lo, hi)
    return lo + rng.uniform(size=y.size) * (hi - lo)


def normal_scores(u: np.ndarray) -> np.ndarray:
    """``z = Φ⁻¹(u)`` con ``u`` recortado a [1e-6, 1 − 1e-6]."""
    return norm.ppf(np.clip(np.asarray(u, dtype=float), U_EPS, 1.0 - U_EPS))


def half_line(value: np.ndarray) -> np.ndarray:
    """Línea x,5 más cercana: ``floor(v) + 0,5`` (0,2 → 0,5; 53,9 → 53,5)."""
    return np.floor(np.asarray(value, dtype=float)) + 0.5


# =============================================================================
# Puntuaciones fuera de muestra (walk-forward)
# =============================================================================


def served_variant(registry: Any, target: str) -> str:
    """``legacy`` si producción sirve el modelo v1 (log-normal), si no ``v2``."""
    try:
        production = registry.load_production(target)
    except Exception as exc:  # noqa: BLE001 - sin producción → v2
        log.debug("Sin producción cargable para %s: %s", target, exc)
        return "v2"
    if production is None:
        return "v2"
    kind = getattr(production[1].get("distribution"), "kind", "lognormal")
    return "legacy" if kind == "lognormal" else "v2"


def prop_marginals(target: str, model: Any, frame: pd.DataFrame) -> dict[str, Any]:
    """Marginal servida de un prop: λ (conteos) o rejilla (continuos).

    Los conteos se sirven como Poisson(λ = estimación puntual) aunque el
    artefacto sea el log-normal heredado (es el λ de reserva que usa la web).
    """
    point = model.predict(frame)
    if target in SPARSE_COUNT_PROPS:
        return {"kind": "poisson", "lam": np.maximum(point, 1e-6), "point": point}
    return {"kind": "grid", "grid": model.quantiles(frame, point), "point": point}


def collect_prop_oos(
    agent: Any,
    target: str,
    seasons: Iterable[int],
    *,
    variant: str = "v2",
    random_state: int = 1729,
) -> pd.DataFrame:
    """Predicciones walk-forward de un prop con su marginal servida.

    Para cada temporada ``S``: entrenamiento ≤ S−2 (desde 2016 en v2),
    calibración de la distribución en S−1, predicción de S. Ninguna fila de S
    ni posterior entra en el modelo que la predice.
    """
    frame = agent.load_dataset(target)
    blocks: list[pd.DataFrame] = []
    lower_bound = QUANTILE_LOWER_BOUND.get(target, 0.0)
    for season in sorted(set(int(s) for s in seasons)):
        test = frame[frame["season"] == season]
        calib = frame[frame["season"] == season - 1]
        if test.empty or calib.empty:
            continue
        if variant == "legacy":
            fit = frame[frame["season"] < season - 1]
        else:
            fit = frame[(frame["season"] >= PROP_MIN_TRAIN_SEASON) & (frame["season"] < season - 1)]
        if fit.empty:
            continue
        model = train_prop_model(
            fit, calib, target, legacy=variant == "legacy", random_state=random_state
        )
        marg = prop_marginals(target, model, test)
        out = test[["game_id", "gsis_id", "season", "week", "team", "opponent_team",
                    "is_home", "position"]].reset_index(drop=True).copy()
        out["target"] = target
        out["y"] = test[target].astype(float).to_numpy()
        out["line_base"] = pd.to_numeric(test[f"{target}_r5"], errors="coerce").to_numpy()
        out["kind"] = marg["kind"]
        out["point"] = marg["point"]
        out["lam"] = marg.get("lam", np.full(len(out), np.nan))
        grid = marg.get("grid")
        grid = np.full((len(out), len(_QCOLS)), np.nan) if grid is None else np.asarray(grid)
        out[_QCOLS] = grid
        out["lower_bound"] = np.nan if lower_bound is None else float(lower_bound)
        out["variant"] = variant
        blocks.append(out)
        log.info("  parlay OOS '%s' %d: %d filas (%s).", target, season, len(out), variant)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def collect_game_oos(
    db: Database,
    settings: Settings,
    seasons: Iterable[int],
    registry: Any,
) -> pd.DataFrame:
    """Margen y total fuera de muestra: N(μ, σ) por partido.

    Familia e hiperparámetros del modelo de producción de ``spread`` y
    ``total``; entrenamiento ≤ S−2 (con el mismo filtro de historia mínima que
    en producción), σ y sesgo de los residuos en S−1 (``ResidualDistribution``),
    predicción de todos los partidos jugados de S.
    """
    from nflp.agents.modeling.calibration import ResidualDistribution
    from nflp.agents.modeling.datasets import load_game_dataset
    from nflp.agents.modeling.models import build_estimator, default_params

    frames: dict[str, pd.DataFrame] = {}
    for target, label in (("spread", "margin"), ("total", "total")):
        family, params = "lightgbm", default_params("lightgbm", "regression", settings.modeling.random_seed)
        model_id = registry.current_production(target)
        if model_id:
            record = registry.get(model_id)
            if not record.empty:
                family = str(record.iloc[0]["family"])
                params = json.loads(record.iloc[0]["hyperparameters"] or "{}")
        train_ds = load_game_dataset(
            target, db=db, feature_version=settings.features.version,
            min_prior_games=settings.features.min_prior_games,
        )
        all_ds = load_game_dataset(
            target, db=db, feature_version=settings.features.version, min_prior_games=0,
        )
        features = train_ds.feature_names
        rows: list[pd.DataFrame] = []
        for season in sorted(set(int(s) for s in seasons)):
            fit_mask = train_ds.meta["season"] <= season - 2
            cal_mask = train_ds.meta["season"] == season - 1
            test_mask = all_ds.meta["season"] == season
            if fit_mask.sum() < 300 or cal_mask.sum() < 30 or test_mask.sum() == 0:
                continue
            model = build_estimator(family, "regression", params)
            model.fit(train_ds.X.loc[fit_mask, features], train_ds.y.loc[fit_mask])
            residuals = ResidualDistribution().fit(
                train_ds.y.loc[cal_mask].to_numpy(),
                model.predict(train_ds.X.loc[cal_mask, features]),
            )
            X_test = all_ds.X.loc[test_mask].reindex(columns=features)
            meta = all_ds.meta.loc[test_mask, ["game_id", "season", "week", "home_team", "away_team"]]
            block = meta.reset_index(drop=True).copy()
            block[f"{label}_mu"] = model.predict(X_test) + residuals.residual_mean
            block[f"{label}_sigma"] = residuals.sigma
            block[f"{label}_y"] = all_ds.y.loc[test_mask].to_numpy()
            rows.append(block)
        frames[label] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        log.info("  parlay OOS de partido '%s' (%s): %d partidos.", label, family, len(frames[label]))

    if frames["margin"].empty or frames["total"].empty:
        return pd.DataFrame()
    return frames["margin"].merge(
        frames["total"][["game_id", "total_mu", "total_sigma", "total_y"]], on="game_id", how="inner"
    )


def player_legs_from_oos(oos: pd.DataFrame, roles: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Piernas de jugador con rol y puntuación normal ``z`` (PIT aleatorizado)."""
    legs = oos.merge(roles, on=["game_id", "gsis_id"], how="left")
    legs["role"] = legs["role"].fillna("OTH")
    legs["leg"] = [player_leg(g, t) for g, t in zip(legs["gsis_id"], legs["target"], strict=True)]
    legs["is_home"] = legs["is_home"].astype(bool)
    params = _marginal_params(legs)
    legs["z"] = normal_scores(randomized_pit(legs["kind"].to_numpy(), legs["y"].to_numpy(), rng, **params))
    legs["line"] = half_line(legs["line_base"].to_numpy())
    return legs


def game_legs_from_oos(games: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Piernas de partido (``g:margin`` local − visitante y ``g:total``) con ``z``."""
    blocks = []
    for label in GAME_TARGETS:
        block = games[["game_id", "season", "week", "home_team"]].copy()
        block["gsis_id"] = None
        block["team"] = None
        block["is_home"] = True
        block["role"] = GAME_ROLE
        block["target"] = label
        block["leg"] = f"g:{label}"
        block["kind"] = "normal"
        block["mu"] = games[f"{label}_mu"].to_numpy()
        block["sigma"] = games[f"{label}_sigma"].to_numpy()
        block["y"] = games[f"{label}_y"].to_numpy()
        block["line"] = half_line(block["mu"].to_numpy())
        blocks.append(block)
    legs = pd.concat(blocks, ignore_index=True)
    params = _marginal_params(legs)
    legs["z"] = normal_scores(randomized_pit(legs["kind"].to_numpy(), legs["y"].to_numpy(), rng, **params))
    return legs


def _marginal_params(legs: pd.DataFrame) -> dict[str, Any]:
    n = len(legs)

    def col(name: str) -> np.ndarray:
        return legs[name].to_numpy(dtype=float) if name in legs.columns else np.full(n, np.nan)

    grid = legs[_QCOLS].to_numpy(dtype=float) if set(_QCOLS) <= set(legs.columns) else np.full((n, len(_QCOLS)), np.nan)
    return {
        "grid": grid, "lower_bound": col("lower_bound"),
        "lam": col("lam"), "mu": col("mu"), "sigma": col("sigma"),
    }


# =============================================================================
# Estadísticos por par y correlaciones por clave
# =============================================================================


def _canonical_order(role_a: pd.Series, target_a: pd.Series, role_b: pd.Series, target_b: pd.Series) -> np.ndarray:
    """True si hay que intercambiar a y b para que la clave sea canónica."""
    return ((role_a + "|" + target_a) > (role_b + "|" + target_b)).to_numpy()


def canonical_key(relation: str, role_a: str, target_a: str, role_b: str, target_b: str) -> tuple[str, ...]:
    """Clave canónica de un par. Jugador×partido: el jugador va primero."""
    if role_a == GAME_ROLE and role_b != GAME_ROLE:
        return (relation, role_b, target_b, role_a, target_a)
    if role_b == GAME_ROLE and role_a != GAME_ROLE:
        return (relation, role_a, target_a, role_b, target_b)
    if f"{role_a}|{target_a}" > f"{role_b}|{target_b}":
        return (relation, role_b, target_b, role_a, target_a)
    return (relation, role_a, target_a, role_b, target_b)


def _aggregate(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    work = frame[KEY_COLUMNS].copy()
    xs, ys = frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)
    work["n"] = 1
    work["sx"], work["sy"] = xs, ys
    work["sxx"], work["syy"], work["sxy"] = xs * xs, ys * ys, xs * ys
    return work.groupby(KEY_COLUMNS, as_index=False).sum()


def pair_stats(player_legs: pd.DataFrame, game_legs: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos suficientes por (temporada, clave): n, Σx, Σy, Σx², Σy², Σxy.

    Con ellos se obtiene Pearson de cualquier rango de temporadas sumando, sin
    volver a recorrer los pares.
    """
    out: list[pd.DataFrame] = []
    margin = game_legs[game_legs["target"] == "margin"][["game_id", "z"]].rename(columns={"z": "z_margin"})
    total = game_legs[game_legs["target"] == "total"][["game_id", "z"]].rename(columns={"z": "z_total"})
    seasons = sorted(set(player_legs["season"]).union(set(game_legs["season"])))
    for season in seasons:
        p = player_legs.loc[
            player_legs["season"] == season,
            ["game_id", "leg", "gsis_id", "team", "is_home", "role", "target", "z"],
        ]
        # --- jugador × jugador ---
        if not p.empty:
            m = p.merge(p, on="game_id", suffixes=("_a", "_b"))
            m = m[m["leg_a"] < m["leg_b"]]
            m["relation"] = np.where(
                m["gsis_id_a"] == m["gsis_id_b"], "same_player",
                np.where(m["team_a"] == m["team_b"], "teammate", "opponent"),
            )
            swap = _canonical_order(m["role_a"], m["target_a"], m["role_b"], m["target_b"])
            for left, right in (("role_a", "role_b"), ("target_a", "target_b"), ("z_a", "z_b")):
                a_vals, b_vals = m[left].to_numpy().copy(), m[right].to_numpy().copy()
                m[left] = np.where(swap, b_vals, a_vals)
                m[right] = np.where(swap, a_vals, b_vals)
            out.append(_aggregate(m, "z_a", "z_b").assign(season=season))

            # --- jugador × piernas de partido ---
            pg = p.merge(margin, on="game_id").merge(total, on="game_id")
            if not pg.empty:
                sign = np.where(pg["is_home"].astype(bool), 1.0, -1.0)
                for label, column, signed in (("margin", "z_margin", True), ("total", "z_total", False)):
                    frame = pd.DataFrame({
                        "relation": "game", "role_a": pg["role"].to_numpy(),
                        "target_a": pg["target"].to_numpy(), "role_b": GAME_ROLE, "target_b": label,
                        "x": pg["z"].to_numpy(),
                        "y": pg[column].to_numpy() * (sign if signed else 1.0),
                    })
                    out.append(_aggregate(frame, "x", "y").assign(season=season))

        # --- margen × total ---
        gs = game_legs[game_legs["season"] == season]
        mt = gs[gs["target"] == "margin"][["game_id", "z"]].merge(
            gs[gs["target"] == "total"][["game_id", "z"]], on="game_id", suffixes=("_m", "_t")
        )
        if not mt.empty:
            frame = pd.DataFrame({
                "relation": "game", "role_a": GAME_ROLE, "target_a": "margin",
                "role_b": GAME_ROLE, "target_b": "total",
                "x": mt["z_m"].to_numpy(), "y": mt["z_t"].to_numpy(),
            })
            out.append(_aggregate(frame, "x", "y").assign(season=season))

    if not out:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "season", "n", "sx", "sy", "sxx", "syy", "sxy"])
    stats = pd.concat(out, ignore_index=True)
    return stats.groupby([*KEY_COLUMNS, "season"], as_index=False).sum()


def pearson_from_sums(n: np.ndarray, sx: np.ndarray, sy: np.ndarray, sxx: np.ndarray,
                      syy: np.ndarray, sxy: np.ndarray, symmetric: np.ndarray | None = None) -> np.ndarray:
    """Pearson a partir de sumas. En claves simétricas (mismo rol y target a
    ambos lados) se usan media y varianza combinadas de los dos lados, porque
    el orden a/b es arbitrario."""
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = sx / n, sy / n
        vx, vy = sxx / n - mx**2, syy / n - my**2
        cov = sxy / n - mx * my
        if symmetric is not None and np.any(symmetric):
            pooled_m = (sx + sy) / (2 * n)
            pooled_v = (sxx + syy) / (2 * n) - pooled_m**2
            cov = np.where(symmetric, sxy / n - pooled_m**2, cov)
            vx = np.where(symmetric, pooled_v, vx)
            vy = np.where(symmetric, pooled_v, vy)
        rho = cov / np.sqrt(vx * vy)
    rho = np.where((n >= 3) & np.isfinite(rho), rho, 0.0)
    return np.clip(rho, -0.99, 0.99)


def shrink(rho: np.ndarray, n: np.ndarray, k: float) -> np.ndarray:
    """Contracción hacia 0: ``ρ·n/(n+k)``."""
    n = np.asarray(n, dtype=float)
    return np.asarray(rho, dtype=float) * n / (n + float(k)) if k > 0 else np.asarray(rho, dtype=float)


def summarize_stats(stats: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """Suma los estadísticos de las temporadas dadas y calcula ρ por clave."""
    seasons = list(seasons)
    subset = stats[stats["season"].isin(seasons)]
    if subset.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "n", "rho_raw"])
    agg = subset.groupby(KEY_COLUMNS, as_index=False)[["n", "sx", "sy", "sxx", "syy", "sxy"]].sum()
    symmetric = ((agg["role_a"] == agg["role_b"]) & (agg["target_a"] == agg["target_b"])).to_numpy()
    agg["rho_raw"] = pearson_from_sums(
        agg["n"].to_numpy(), agg["sx"].to_numpy(), agg["sy"].to_numpy(),
        agg["sxx"].to_numpy(), agg["syy"].to_numpy(), agg["sxy"].to_numpy(), symmetric,
    )
    return agg


def composite_loglik(rho: np.ndarray, n: np.ndarray, sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray) -> float:
    """Log-verosimilitud gaussiana bivariante (z estándar) sumada por clave."""
    rho = np.clip(np.asarray(rho, dtype=float), -0.99, 0.99)
    one = 1.0 - rho**2
    ll = -0.5 * n * np.log(one) - (sxx - 2.0 * rho * sxy + syy) / (2.0 * one)
    return float(np.sum(ll))


def choose_shrinkage(
    stats: pd.DataFrame,
    seasons: Iterable[int],
    *,
    n_folds: int = 3,
    grid: Iterable[float] = SHRINKAGE_GRID,
) -> tuple[float, pd.DataFrame]:
    """Elige ``k`` con validación temporal entre las temporadas dadas.

    Cada pliegue estima ρ con las temporadas **anteriores** a la de validación
    V y puntúa la verosimilitud compuesta por pares en V. Se devuelve el ``k``
    con mayor suma sobre pliegues.
    """
    seasons = sorted(set(int(s) for s in seasons))
    folds = [v for v in seasons[1:]][-n_folds:]
    grid = list(grid)
    scores = {k: 0.0 for k in grid}
    for valid in folds:
        train = summarize_stats(stats, [s for s in seasons if s < valid])
        target = summarize_stats(stats, [valid])
        if train.empty or target.empty:
            continue
        merged = target.merge(train[[*KEY_COLUMNS, "n", "rho_raw"]], on=KEY_COLUMNS,
                              how="left", suffixes=("", "_train"))
        n_train = merged["n_train"].fillna(0.0).to_numpy()
        rho_train = merged["rho_raw_train"].fillna(0.0).to_numpy()
        for k in grid:
            scores[k] += composite_loglik(
                shrink(rho_train, n_train, k), merged["n"].to_numpy(dtype=float),
                merged["sxx"].to_numpy(), merged["syy"].to_numpy(), merged["sxy"].to_numpy(),
            )
    table = pd.DataFrame({"k": grid, "loglik": [scores[k] for k in grid]})
    if not folds or table["loglik"].eq(0).all():
        return float(grid[len(grid) // 2]), table
    best = float(table.loc[table["loglik"].idxmax(), "k"])
    return best, table


@dataclass
class CorrelationModel:
    """Correlaciones contraídas por clave ``(relación, rol_a, target_a, rol_b, target_b)``."""

    table: pd.DataFrame
    k: float
    seasons: list[int]
    kind: str = KIND_CORRELATED
    _lookup: dict[tuple[str, ...], float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._lookup and not self.table.empty:
            keys = zip(*(self.table[c] for c in KEY_COLUMNS), strict=True)
            self._lookup = dict(zip(keys, self.table["rho"].astype(float), strict=True))

    def rho(self, relation: str, role_a: str, target_a: str, role_b: str, target_b: str) -> float:
        if self.kind == KIND_IDENTITY:
            return 0.0
        return self._lookup.get(canonical_key(relation, role_a, target_a, role_b, target_b), 0.0)

    def to_dict(self) -> dict[str, Any]:
        cols = [*KEY_COLUMNS, "n", "rho_raw", "rho"]
        return {
            "kind": self.kind, "k": self.k, "seasons": self.seasons,
            "table": self.table[cols].assign(
                n=self.table["n"].astype(int),
                rho_raw=self.table["rho_raw"].round(5), rho=self.table["rho"].round(5),
            ).to_dict(orient="records"),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CorrelationModel:
        table = pd.DataFrame(payload.get("table", []))
        if table.empty:
            table = pd.DataFrame(columns=[*KEY_COLUMNS, "n", "rho_raw", "rho"])
        return cls(table=table, k=float(payload["k"]), seasons=list(payload["seasons"]),
                   kind=payload.get("kind", KIND_CORRELATED))


def fit_correlations(stats: pd.DataFrame, seasons: Iterable[int], k: float) -> CorrelationModel:
    """Correlaciones contraídas con las temporadas dadas (y solo esas)."""
    seasons = sorted(set(int(s) for s in seasons))
    table = summarize_stats(stats, seasons)
    if not table.empty:
        table["rho"] = shrink(table["rho_raw"].to_numpy(), table["n"].to_numpy(), k)
    else:
        table["rho"] = []
    return CorrelationModel(table=table, k=float(k), seasons=seasons)


def fit_before(stats: pd.DataFrame, test_season: int, *, n_folds: int = 3) -> CorrelationModel:
    """Modelo para evaluar ``test_season``: k y ρ solo con temporadas anteriores."""
    available = sorted(int(s) for s in stats["season"].unique() if int(s) < test_season)
    if not available:
        return CorrelationModel(table=pd.DataFrame(columns=[*KEY_COLUMNS, "n", "rho_raw", "rho"]),
                                k=0.0, seasons=[])
    k, _ = choose_shrinkage(stats, available, n_folds=n_folds)
    return fit_correlations(stats, available, k)


# =============================================================================
# Matriz por partido
# =============================================================================


def nearest_psd_correlation(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Correlación semidefinida positiva más cercana por recorte de autovalores.

    Se simetriza, se recortan los autovalores a ``eps`` y se renormaliza la
    diagonal a 1 (``D^{-1/2} A D^{-1/2}``, congruencia que conserva la
    semidefinición).
    """
    sym = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    values, vectors = np.linalg.eigh(sym)
    clipped = (vectors * np.maximum(values, eps)) @ vectors.T
    d = np.sqrt(np.clip(np.diag(clipped), 1e-12, None))
    out = clipped / np.outer(d, d)
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 1.0)
    return out


def round_psd(matrix: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Redondea y garantiza que la matriz redondeada sigue siendo semidefinida.

    Si el redondeo crea un autovalor negativo, se mezcla con la identidad
    ``(1−δ)R + δI`` con el menor δ de una escala que lo evite.
    """
    for delta in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        candidate = np.round((1.0 - delta) * matrix + delta * np.eye(len(matrix)), decimals)
        np.fill_diagonal(candidate, 1.0)
        if len(candidate) == 0 or np.linalg.eigvalsh(candidate).min() >= 0.0:
            return candidate
    return np.eye(len(matrix))


def build_correlation_matrix(legs: pd.DataFrame, model: CorrelationModel, *, psd: bool = True) -> np.ndarray:
    """Matriz de correlación latente de las piernas de un partido.

    ``legs`` necesita ``gsis_id``, ``team``, ``is_home``, ``role`` y ``target``
    (las piernas de partido llevan ``role='G'``). Frente a ``g:margin`` (local −
    visitante) la correlación estimada con el margen del equipo del jugador
    cambia de signo si el jugador es visitante.
    """
    n = len(legs)
    matrix = np.eye(n)
    if model.kind == KIND_IDENTITY or n == 0:
        return matrix
    roles = legs["role"].tolist()
    targets = legs["target"].tolist()
    players = legs["gsis_id"].tolist()
    teams = legs["team"].tolist()
    home = legs["is_home"].astype(bool).tolist()
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = roles[i] == GAME_ROLE, roles[j] == GAME_ROLE
            if gi or gj:
                relation = "game"
            elif players[i] == players[j]:
                relation = "same_player"
            elif teams[i] == teams[j]:
                relation = "teammate"
            else:
                relation = "opponent"
            value = model.rho(relation, roles[i], targets[i], roles[j], targets[j])
            if gi != gj:
                game_target = targets[i] if gi else targets[j]
                player_home = home[j] if gi else home[i]
                if game_target == "margin" and not player_home:
                    value = -value
            matrix[i, j] = matrix[j, i] = value
    return nearest_psd_correlation(matrix) if psd else matrix


# =============================================================================
# Backtest de calibración con parlays sintéticos
# =============================================================================


def leg_threshold(legs: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad marginal de "más" y umbral latente ``t = Φ⁻¹(F(línea))``."""
    legs = legs.copy()
    params = _marginal_params(legs)
    cdf = marginal_cdf(legs["kind"].to_numpy(), legs["line"].to_numpy(dtype=float), **params)
    legs["p_over"] = 1.0 - cdf
    legs["t"] = norm.ppf(np.clip(cdf, U_EPS, 1.0 - U_EPS))
    legs["over_hit"] = legs["y"].to_numpy(dtype=float) > legs["line"].to_numpy(dtype=float)
    return legs


def _draw_latent(matrix: np.ndarray, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(matrix)
        factor = vectors * np.sqrt(np.maximum(values, 0.0))
    return rng.standard_normal((n_sims, matrix.shape[0])) @ factor.T


def simulate_parlays(
    legs: pd.DataFrame,
    model: CorrelationModel,
    *,
    n_per_game: int = 40,
    n_sims: int = 20000,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Parlays sintéticos de 2–4 piernas por partido: cópula vs independencia.

    Mitad "aleatorios" (piernas de todo el partido) y mitad "mismo equipo"
    (piernas de un equipo + piernas de partido), que es donde la correlación
    importa. Lado over/under al azar. La probabilidad independiente es el
    producto exacto de marginales; la de cópula, Monte Carlo con ``n_sims``.
    """
    rows: list[dict[str, Any]] = []
    for game_id, game in legs.groupby("game_id", sort=False):
        game = game[np.isfinite(game["line"]) & np.isfinite(game["t"])].reset_index(drop=True)
        if len(game) < 4:
            continue
        matrix = build_correlation_matrix(game, model)
        latent = _draw_latent(matrix, n_sims, rng)
        above = latent > game["t"].to_numpy()[None, :]
        p_over = game["p_over"].to_numpy()
        over_hit = game["over_hit"].to_numpy()
        teams = [t for t in game["team"].dropna().unique()]
        is_game_leg = (game["role"] == GAME_ROLE).to_numpy()
        for _ in range(n_per_game):
            size = int(rng.integers(2, 5))
            stratum = "random" if rng.random() < 0.5 or not teams else "same_team"
            if stratum == "same_team":
                team = teams[int(rng.integers(len(teams)))]
                pool = np.flatnonzero((game["team"] == team).to_numpy() | is_game_leg)
            else:
                pool = np.arange(len(game))
            if len(pool) < size:
                continue
            idx = rng.choice(pool, size=size, replace=False)
            over = rng.random(size) < 0.5
            p_indep = float(np.prod(np.where(over, p_over[idx], 1.0 - p_over[idx])))
            p_copula = float(np.mean(np.all(above[:, idx] == over[None, :], axis=1)))
            hit = bool(np.all(over_hit[idx] == over))
            rows.append({
                "game_id": game_id, "season": int(game["season"].iloc[0]), "n_legs": size,
                "stratum": stratum, "p_indep": p_indep, "p_copula": p_copula, "hit": hit,
                "legs": ",".join(game["leg"].iloc[idx].tolist()),
                "sides": "".join("O" if o else "U" for o in over),
                "direction": "all_over" if over.all() else ("all_under" if not over.any() else "mixed"),
            })
    return pd.DataFrame(rows)


def ece_score(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE con 10 bins de igual anchura en [0, 1], ponderado por tamaño del bin."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += sel.sum() / p.size * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def log_loss(p: np.ndarray, y: np.ndarray, eps: float) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = bins == b
        rows.append({
            "bin": f"[{b / n_bins:.1f}, {(b + 1) / n_bins:.1f})", "n": int(sel.sum()),
            "p_mean": float(p[sel].mean()) if sel.any() else float("nan"),
            "freq": float(y[sel].mean()) if sel.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def calibration_metrics(parlays: pd.DataFrame, n_sims: int) -> dict[str, float]:
    eps = 1.0 / (2.0 * n_sims)
    y = parlays["hit"].to_numpy(dtype=float)
    return {
        "ece": ece_score(parlays["p_copula"].to_numpy(), y),
        "logloss": log_loss(parlays["p_copula"].to_numpy(), y, eps),
        "brier": float(np.mean((parlays["p_copula"].to_numpy() - y) ** 2)),
        "ece_independent": ece_score(parlays["p_indep"].to_numpy(), y),
        "logloss_independent": log_loss(parlays["p_indep"].to_numpy(), y, eps),
        "brier_independent": float(np.mean((parlays["p_indep"].to_numpy() - y) ** 2)),
        "hit_rate": float(y.mean()) if y.size else float("nan"),
        "mean_p_copula": float(parlays["p_copula"].mean()) if y.size else float("nan"),
        "mean_p_indep": float(parlays["p_indep"].mean()) if y.size else float("nan"),
        "n": int(y.size),
    }


# =============================================================================
# Orquestación: ajuste, backtest, persistencia y predicción
# =============================================================================


@dataclass
class ParlayFitResult:
    model_id: str
    model: CorrelationModel
    metrics: dict[str, float]
    by_season: pd.DataFrame
    by_legs: pd.DataFrame
    by_stratum: pd.DataFrame
    by_direction: pd.DataFrame
    reliability: dict[str, pd.DataFrame]
    k_by_test_season: dict[int, float]
    k_scores: pd.DataFrame
    publish_correlated: bool
    marginal: dict[str, float]
    parlays: pd.DataFrame
    report_path: Path | None = None
    artifact_path: Path | None = None


class ParlayAgent:
    """Ajusta, valida y sirve el modelo de correlación de parlays."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        from nflp.agents.modeling.props_agent import PropsAgent

        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.props = PropsAgent(self.settings, self.db)
        self.registry = self.props.registry
        self.cache_dir = self.settings.paths.artifacts_dir / "parlay"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ OOS + roles

    def _completed_seasons(self) -> list[int]:
        frame = self.db.query(
            "SELECT DISTINCT season FROM games WHERE home_score IS NOT NULL ORDER BY season"
        )
        return [int(s) for s in frame["season"] if int(s) >= OOS_FIRST_SEASON]

    def oos_scores(self, *, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Piernas fuera de muestra (jugador y partido) con su ``z``; con caché en disco."""
        seasons = self._completed_seasons()
        counts = self.db.query(
            "SELECT season, count(*) AS n FROM games WHERE home_score IS NOT NULL GROUP BY season"
        )
        completed = {int(r.season): int(r.n) for r in counts.itertuples(index=False)}
        seed = self.settings.modeling.random_seed
        prop_blocks = []
        for target in self.settings.props.targets:
            variant = served_variant(self.registry, target)
            path = self.cache_dir / f"oos_{target}.parquet"
            cached = pd.read_parquet(path) if path.exists() and not refresh else pd.DataFrame()
            have = set(cached["season"].unique()) if not cached.empty else set()
            # La temporada en curso se recalcula si hay partidos jugados nuevos.
            current = max(seasons)
            stale = (not cached.empty) and (cached["variant"].iloc[0] != variant)
            grown = cached.empty or (
                cached.loc[cached["season"] == current, "game_id"].nunique() < completed[current]
            )
            missing = [s for s in seasons if s not in have or (s == current and grown) or stale]
            if stale:
                cached = pd.DataFrame()
            if missing:
                fresh = collect_prop_oos(self.props, target, missing, variant=variant, random_state=seed)
                keep = cached[~cached["season"].isin(missing)] if not cached.empty else cached
                cached = pd.concat([keep, fresh], ignore_index=True)
                cached.to_parquet(path, index=False)
            prop_blocks.append(cached)
        props = pd.concat(prop_blocks, ignore_index=True)

        game_path = self.cache_dir / "oos_games.parquet"
        games = pd.read_parquet(game_path) if game_path.exists() and not refresh else pd.DataFrame()
        have = set(games["season"].unique()) if not games.empty else set()
        current = max(seasons)
        grown = games.empty or games.loc[games["season"] == current, "game_id"].nunique() < completed[current]
        missing = [s for s in seasons if s not in have or (s == current and grown)]
        if missing:
            fresh = collect_game_oos(self.db, self.settings, missing, self.registry)
            keep = games[~games["season"].isin(missing)] if not games.empty else games
            games = pd.concat([keep, fresh], ignore_index=True)
            games.to_parquet(game_path, index=False)
        return props, games

    def load_roles(self, game_ids: Iterable[str] | None = None, seasons: Iterable[int] | None = None) -> pd.DataFrame:
        cols = ", ".join(["game_id", "gsis_id", "team", "position", *_USAGE_COLUMNS])
        clauses = [f"feature_set_version = '{self.settings.features.version}'"]
        if seasons is not None:
            clauses.append(f"season IN ({', '.join(str(int(s)) for s in seasons)})")
        if game_ids is not None:
            ids = ", ".join(f"'{g}'" for g in game_ids)
            clauses.append(f"game_id IN ({ids})")
        frame = self.db.query(f"SELECT {cols} FROM features_player WHERE {' AND '.join(clauses)}")
        return assign_roles(frame, min_snaps=self.settings.props.min_projected_snaps)

    def build_legs(self, *, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        props, games = self.oos_scores(refresh=refresh)
        rng = np.random.default_rng(self.settings.modeling.random_seed)
        roles = self.load_roles(seasons=sorted(props["season"].unique()))
        player = player_legs_from_oos(props, roles, rng)
        # Solo partidos con piernas de partido y de jugador a la vez.
        game = game_legs_from_oos(games, rng)
        common = set(player["game_id"]) & set(game["game_id"])
        player = player[player["game_id"].isin(common)].reset_index(drop=True)
        game = game[game["game_id"].isin(common)].reset_index(drop=True)
        return player, game

    # --------------------------------------------------------------- ajuste

    def fit(
        self,
        *,
        test_seasons: Iterable[int] = BACKTEST_TEST_SEASONS,
        n_per_game: int = 40,
        n_sims: int = 20000,
        refresh: bool = False,
        persist: bool = True,
    ) -> ParlayFitResult:
        """Backtest walk-forward de calibración y modelo final con todo el histórico."""
        player, game = self.build_legs(refresh=refresh)
        stats = pair_stats(player, game)
        rng = np.random.default_rng(self.settings.modeling.random_seed + 7)

        all_legs = leg_threshold(pd.concat([player, game], ignore_index=True))
        parlays_list = []
        k_by_season: dict[int, float] = {}
        for season in sorted(int(s) for s in test_seasons):
            model = fit_before(stats, season)
            if model.table.empty:
                log.warning("Sin temporadas previas para evaluar %d.", season)
                continue
            assert max(model.seasons) < season  # sin fuga: solo temporadas anteriores
            k_by_season[season] = model.k
            season_legs = all_legs[all_legs["season"] == season]
            parlays_list.append(simulate_parlays(season_legs, model, n_per_game=n_per_game,
                                                 n_sims=n_sims, rng=rng))
            log.info("Parlays %d: k=%.0f, %d parlays.", season, model.k, len(parlays_list[-1]))
        parlays = pd.concat(parlays_list, ignore_index=True)

        metrics = calibration_metrics(parlays, n_sims)
        publish = (metrics["ece"] <= metrics["ece_independent"]) and (
            metrics["logloss"] <= metrics["logloss_independent"]
        )

        def by(column: str) -> pd.DataFrame:
            return pd.DataFrame(
                [{column: key, **calibration_metrics(group, n_sims)} for key, group in parlays.groupby(column)]
            )

        test_legs = all_legs[
            all_legs["season"].isin(list(k_by_season)) & np.isfinite(all_legs["line"])
            & np.isfinite(all_legs["p_over"])
        ]
        marginal = {
            "ece": ece_score(test_legs["p_over"].to_numpy(), test_legs["over_hit"].to_numpy(float)),
            "logloss": log_loss(test_legs["p_over"].to_numpy(), test_legs["over_hit"].to_numpy(float), 1e-6),
            "n": int(len(test_legs)),
        }

        # Modelo final: todas las temporadas con puntuaciones fuera de muestra.
        seasons = sorted(int(s) for s in stats["season"].unique())
        k_final, k_scores = choose_shrinkage(stats, seasons)
        final = fit_correlations(stats, seasons, k_final)
        if not publish:
            final.kind = KIND_IDENTITY

        result = ParlayFitResult(
            model_id=str(uuid.uuid4()), model=final, metrics=metrics,
            by_season=by("season"), by_legs=by("n_legs"), by_stratum=by("stratum"),
            by_direction=by("direction"),
            reliability={
                "copula": reliability_table(parlays["p_copula"], parlays["hit"]),
                "independent": reliability_table(parlays["p_indep"], parlays["hit"]),
            },
            k_by_test_season=k_by_season, k_scores=k_scores, publish_correlated=publish,
            marginal=marginal, parlays=parlays,
        )
        log.info(
            "Parlays: ECE %.4f vs %.4f (indep) · log loss %.4f vs %.4f · n=%d → %s",
            metrics["ece"], metrics["ece_independent"], metrics["logloss"],
            metrics["logloss_independent"], metrics["n"],
            "cópula" if publish else "identidad",
        )
        if persist:
            self._persist_model(result)
        return result

    def _persist_model(self, result: ParlayFitResult) -> None:
        now = datetime.now(UTC).replace(tzinfo=None)
        artifact = {
            "model_id": result.model_id,
            "created_at": now.isoformat(),
            "kind": result.model.kind,
            "publish_correlated": result.publish_correlated,
            "metrics": result.metrics,
            "k_by_test_season": {str(k): v for k, v in result.k_by_test_season.items()},
            "model": result.model.to_dict(),
        }
        path = self.cache_dir / f"parlay_{result.model_id}.json"
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
        result.artifact_path = path
        row = pd.DataFrame([{
            "model_id": result.model_id,
            "ece": result.metrics["ece"] if result.publish_correlated else result.metrics["ece_independent"],
            "logloss": result.metrics["logloss"] if result.publish_correlated else result.metrics["logloss_independent"],
            "ece_independent": result.metrics["ece_independent"],
            "logloss_independent": result.metrics["logloss_independent"],
            "n": int(result.metrics["n"]),
            "created_at": now,
        }])
        self.db.write_df(row, "parlay_calibration", mode="upsert", key_columns=["model_id"])

    # ------------------------------------------------------------- servicio

    def latest_model(self) -> tuple[str, CorrelationModel] | None:
        frame = self.db.query(
            "SELECT model_id FROM parlay_calibration ORDER BY created_at DESC LIMIT 1"
        ) if self.db.table_exists("parlay_calibration") else pd.DataFrame()
        if frame.empty:
            return None
        model_id = str(frame.iloc[0]["model_id"])
        path = self.cache_dir / f"parlay_{model_id}.json"
        if not path.exists():
            log.warning("Falta el artefacto del modelo de parlay %s.", model_id)
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_id, CorrelationModel.from_dict(payload["model"])

    def predict(
        self,
        *,
        season: int | None = None,
        week: int | None = None,
        persist: bool = True,
        model: tuple[str, CorrelationModel] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filas de ``parlay_game_legs`` y ``parlay_game_corr`` para los partidos indicados.

        Por defecto, los partidos aún no jugados. Las piernas de jugador salen
        de las predicciones de props de los modelos en producción (última
        corrida por partido); las de partido, de las de ``spread`` y ``total``.
        """
        selection = model or self.latest_model()
        if selection is None:
            log.warning("Sin modelo de parlay; ejecuta `nflp model parlay-corr`.")
            return pd.DataFrame(), pd.DataFrame()
        model_id, corr = selection

        clauses = []
        if season is not None:
            clauses.append(f"g.season = {int(season)}")
        if week is not None:
            clauses.append(f"g.week = {int(week)}")
        if not clauses:
            now = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
            clauses.append(f"g.home_score IS NULL AND g.kickoff_utc >= TIMESTAMP '{now}'")
        games = self.db.query(
            f"SELECT game_id, season, week, home_team, away_team FROM games g WHERE {' AND '.join(clauses)}"
        )
        if games.empty:
            log.warning("No hay partidos para el modelo de parlay con ese filtro.")
            return pd.DataFrame(), pd.DataFrame()
        ids = ", ".join(f"'{g}'" for g in games["game_id"])

        targets = list(self.settings.props.targets)
        production = {t: self.registry.current_production(t) for t in [*targets, "spread", "total"]}
        model_ids = ", ".join(f"'{m}'" for m in production.values() if m)
        preds = self.db.query(
            f"""
            SELECT game_id, gsis_id, target, model_id, point_estimate, pred_std,
                   lower_80, upper_80, team, opponent_team, is_home, as_of_ts
            FROM predictions
            WHERE game_id IN ({ids}) AND NOT is_shadow AND model_id IN ({model_ids or "''"})
            QUALIFY row_number() OVER (
                PARTITION BY game_id, coalesce(gsis_id, ''), target ORDER BY predicted_at DESC
            ) = 1
            """
        )
        preds = preds[[production.get(t) == m for t, m in zip(preds["target"], preds["model_id"], strict=True)]]

        roles = self.load_roles(game_ids=games["game_id"].tolist())
        legs_rows, corr_rows = [], []
        role_order = {r: i for i, r in enumerate(ROLES)}
        target_order = {t: i for i, t in enumerate(targets)}
        for game_row in games.itertuples(index=False):
            gp = preds[preds["game_id"] == game_row.game_id]
            player = gp[gp["target"].isin(targets) & gp["gsis_id"].notna() & gp["team"].notna()].copy()
            player = player.merge(roles, on=["game_id", "gsis_id"], how="left")
            player["role"] = player["role"].fillna("OTH")
            player["is_home"] = player["team"] == game_row.home_team
            player["_home_first"] = (~player["is_home"]).astype(int)
            player["_r"] = player["role"].map(role_order)
            player["_t"] = player["target"].map(target_order)
            player = player.sort_values(["_home_first", "_r", "gsis_id", "_t"]).reset_index(drop=True)
            player["leg"] = [player_leg(g, t) for g, t in zip(player["gsis_id"], player["target"], strict=True)]

            spread = gp[gp["target"] == "spread"]
            total = gp[gp["target"] == "total"]
            game_legs = pd.DataFrame()
            if not spread.empty and not total.empty:
                s, t = spread.iloc[0], total.iloc[0]
                margin_mu = float((s["lower_80"] + s["upper_80"]) / 2.0)
                total_mu = float((t["lower_80"] + t["upper_80"]) / 2.0)
                margin_sigma, total_sigma = float(s["pred_std"]), float(t["pred_std"])
                as_of = max(pd.Timestamp(s["as_of_ts"]), pd.Timestamp(t["as_of_ts"]))
                legs_rows.append({
                    "game_id": game_row.game_id, "model_id": model_id,
                    "p_home_win": float(norm.cdf(margin_mu / margin_sigma)),
                    "margin_mu": margin_mu, "margin_sigma": margin_sigma,
                    "total_mu": total_mu, "total_sigma": total_sigma, "as_of_ts": as_of,
                })
                game_legs = pd.DataFrame({
                    "leg": ["g:margin", "g:total"], "gsis_id": [None, None], "team": [None, None],
                    "is_home": [True, True], "role": [GAME_ROLE, GAME_ROLE], "target": list(GAME_TARGETS),
                })
            all_legs = pd.concat(
                [player[["leg", "gsis_id", "team", "is_home", "role", "target"]], game_legs], ignore_index=True
            )
            if all_legs.empty:
                continue
            matrix = round_psd(build_correlation_matrix(all_legs, corr))
            as_of_values = pd.to_datetime(gp["as_of_ts"], errors="coerce").dropna()
            corr_rows.append({
                "game_id": game_row.game_id, "model_id": model_id,
                "legs": json.dumps(all_legs["leg"].tolist()),
                "matrix": json.dumps(matrix.tolist()),
                "as_of_ts": as_of_values.max() if not as_of_values.empty else None,
            })

        legs_frame, corr_frame = pd.DataFrame(legs_rows), pd.DataFrame(corr_rows)
        if persist and not corr_frame.empty:
            # Una sola fila vigente por partido: se retiran las de modelos de
            # parlay anteriores para esos partidos (la publicación toma la de
            # mayor as_of_ts y no sabría desempatar entre modelos).
            game_list = ", ".join(f"'{g}'" for g in corr_frame["game_id"])
            self.db.execute(f"DELETE FROM parlay_game_corr WHERE game_id IN ({game_list})")
            self.db.execute(f"DELETE FROM parlay_game_legs WHERE game_id IN ({game_list})")
            self.db.write_df(corr_frame, "parlay_game_corr", mode="append")
            if not legs_frame.empty:
                self.db.write_df(legs_frame, "parlay_game_legs", mode="append")
        log.info("Parlay: %d partidos con matriz (%s), %d con piernas de partido.",
                 len(corr_frame), corr.kind, len(legs_frame))
        return legs_frame, corr_frame


# =============================================================================
# Informe
# =============================================================================


def top_correlations(model: CorrelationModel, n: int = 25, min_pairs: int = 200) -> pd.DataFrame:
    table = model.table.copy()
    table = table[table["n"] >= min_pairs]
    table["abs"] = table["rho"].abs()
    return table.sort_values("abs", ascending=False).head(n)[[*KEY_COLUMNS, "n", "rho_raw", "rho"]]


def sanity_pairs(model: CorrelationModel) -> pd.DataFrame:
    checks = [
        ("teammate", "QB", "passing_yards", "WR1", "receiving_yards"),
        ("teammate", "WR1", "receiving_yards", "WR2", "receiving_yards"),
        ("game", "RB1", "rushing_yards", "G", "margin"),
        ("game", "QB", "passing_yards", "G", "total"),
        ("game", "QB", "passing_yards", "G", "margin"),
        ("same_player", "WR1", "receptions", "WR1", "receiving_yards"),
        ("same_player", "RB1", "receiving_tds", "RB1", "rushing_tds"),
        ("same_player", "QB", "passing_tds", "QB", "passing_yards"),
        ("game", "WR1", "receiving_tds", "G", "total"),
        ("game", "RB1", "rushing_tds", "G", "total"),
        ("game", "QB", "sacks_suffered", "G", "margin"),
        ("opponent", "QB", "passing_yards", "QB", "passing_yards"),
        ("game", "G", "margin", "G", "total"),
    ]
    lookup = model.table.set_index(KEY_COLUMNS) if not model.table.empty else None
    rows = []
    for key in checks:
        canon = canonical_key(*key)
        if lookup is not None and canon in lookup.index:
            rec = lookup.loc[canon]
            rows.append({"pair": " × ".join([f"{key[1]} {key[2]}", f"{key[3]} {key[4]}"]),
                         "relation": key[0], "n": int(rec["n"]),
                         "rho_raw": float(rec["rho_raw"]), "rho": float(rec["rho"])})
    return pd.DataFrame(rows)


def _md_table(frame: pd.DataFrame, floats: int = 4) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda v: "" if pd.isna(v) else f"{v:.{floats}f}")
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    sep = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_parlay_report(result: ParlayFitResult, path: Path, *, n_sims: int, n_per_game: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = result.metrics
    decision = (
        "**Se publica la cópula correlada**: ECE y log loss ≤ independencia."
        if result.publish_correlated
        else "**Se publica la identidad (independencia)**: la cópula no mejora ECE y log loss a la vez."
    )
    overall = pd.DataFrame([
        {"modelo": "cópula", "ece": m["ece"], "logloss": m["logloss"], "brier": m["brier"],
         "p_media": m["mean_p_copula"]},
        {"modelo": "independencia", "ece": m["ece_independent"], "logloss": m["logloss_independent"],
         "brier": m["brier_independent"], "p_media": m["mean_p_indep"]},
    ])
    cols = ["n", "hit_rate", "mean_p_copula", "mean_p_indep", "ece", "ece_independent",
            "logloss", "logloss_independent"]
    reliab = result.reliability["copula"].merge(
        result.reliability["independent"], on="bin", suffixes=("_copula", "_indep")
    )
    lines = [
        "# Calibración del modelo de correlación de parlays (mismo partido)",
        "",
        f"Modelo `{result.model_id}` · tipo publicado: `{result.model.kind}` · "
        f"k final = {result.model.k:.0f} · temporadas de estimación {result.model.seasons}",
        "",
        decision,
        "",
        "## Método",
        "",
        "* Puntuaciones normales `z = Φ⁻¹(PIT aleatorizado)` fuera de muestra: props con",
        "  entrenamiento ≤ S−2, calibración de la distribución en S−1 y prueba en S, con la",
        "  marginal servida (rejilla p05..p95 en continuos, Poisson(λ) en conteos);",
        "  margen y total con el modelo de producción reentrenado ≤ S−2 y σ de residuos en S−1.",
        "* Correlación de Pearson por (rol_a, target_a, rol_b, target_b, relación), contraída",
        "  `ρ·n/(n+k)`; k por validación temporal (verosimilitud compuesta por pares).",
        "* Para cada temporada de prueba las correlaciones y k se estiman **solo con",
        "  temporadas anteriores**: " + ", ".join(
            f"{s}: k={k:.0f}" for s, k in result.k_by_test_season.items()),
        f"* Parlays sintéticos de 2–4 piernas ({n_per_game} por partido; mitad al azar, mitad",
        "  de un mismo equipo + piernas de partido); línea = media de 5 partidos previos",
        "  redondeada a x,5 (jugador) o μ redondeada a x,5 (partido); over/under al azar.",
        f"* Probabilidad conjunta: Monte Carlo con la cópula ({n_sims:,} simulaciones por partido)",
        "  frente al producto de marginales (independencia). ECE con 10 bins de igual anchura.",
        "",
        "## Resultado global",
        "",
        f"{m['n']:,} parlays, tasa de acierto {m['hit_rate']:.4f}.",
        "",
        _md_table(overall),
        "",
        f"Referencia (marginales, una pierna): ECE {result.marginal['ece']:.4f} · "
        f"log loss {result.marginal['logloss']:.4f} · n={result.marginal['n']:,}.",
        "",
        "## Por temporada de prueba",
        "",
        _md_table(result.by_season[["season", *cols]]),
        "",
        "## Por número de piernas",
        "",
        _md_table(result.by_legs[["n_legs", *cols]]),
        "",
        "## Por estrato",
        "",
        _md_table(result.by_stratum[["stratum", *cols]]),
        "",
        "## Por dirección de las piernas",
        "",
        "Con lados al azar, las correlaciones positivas y negativas se compensan en",
        "promedio; donde la dependencia pesa es en parlays de una sola dirección.",
        "",
        _md_table(result.by_direction[["direction", *cols]]),
        "",
        "## Tabla de fiabilidad",
        "",
        _md_table(reliab[["bin", "n_copula", "p_mean_copula", "freq_copula", "p_mean_indep",
                          "n_indep", "freq_indep"]]),
        "",
        "## Correlaciones de control (modelo final)",
        "",
        _md_table(sanity_pairs(result.model), floats=3),
        "",
        "## Mayores correlaciones (|ρ| contraída, n ≥ 200)",
        "",
        _md_table(top_correlations(result.model), floats=3),
        "",
        "## Elección de k (modelo final, validación temporal)",
        "",
        _md_table(result.k_scores, floats=1),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = [
    "BACKTEST_TEST_SEASONS",
    "CorrelationModel",
    "KIND_CORRELATED",
    "KIND_IDENTITY",
    "ParlayAgent",
    "ParlayFitResult",
    "assign_roles",
    "build_correlation_matrix",
    "canonical_key",
    "choose_shrinkage",
    "fit_before",
    "fit_correlations",
    "grid_cdf",
    "marginal_cdf",
    "nearest_psd_correlation",
    "pair_stats",
    "randomized_pit",
    "round_psd",
    "shrink",
    "simulate_parlays",
    "write_parlay_report",
]
