"""Selección del objetivo puntual y búsqueda de hiperparámetros de los props.

Todo es **temporal**: cada pliegue entrena con temporadas anteriores y evalúa
una temporada posterior completa. Nunca se baraja. Las temporadas de
calibración y validación del entrenamiento final (las más recientes) quedan
fuera de estos pliegues, para que sus métricas no estén contaminadas por la
selección.

* ``compare_objectives`` — L1 / L2 / Huber / Tweedie para los continuos (y
  Poisson frente a L2 para los conteos). Para que el CRPS sea comparable entre
  objetivos, cada punto se acompaña de la rejilla de residuos de la temporada
  de calibración (``punto + cuantiles de y − punto``); también se puntúa la
  rejilla LightGBM cuantil, que es la que se sirve.
* ``tune_prop`` — Optuna acotado (≈40 pruebas) minimizando el MAE medio (o la
  devianza Poisson en los conteos) sobre pliegues por temporada.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from nflp.agents.modeling.distributions import (
    P10_INDEX,
    P90_INDEX,
    QUANTILE_LEVELS,
    PoissonDistribution,
    QuantileGridDistribution,
    crps_poisson,
    crps_quantile_grid,
    interval_coverage,
    make_monotone,
    randomized_pit_coverage,
)
from nflp.agents.modeling.props import (
    POINT_OBJECTIVES,
    PROP_FEATURE_SETS,
    QUANTILE_LOWER_BOUND,
    SPARSE_COUNT_PROPS,
    TUNED_PARAMS_PATH,
    add_derived_features,
    load_tuned_params,
    train_prop_model,
)
from nflp.core.logging import get_logger

log = get_logger(__name__)

_LEVELS = np.asarray(QUANTILE_LEVELS)


def residual_grid(point_calib: np.ndarray, y_calib: np.ndarray, point: np.ndarray, lower: float | None) -> np.ndarray:
    """Rejilla ``punto + cuantiles empíricos de (y − punto)`` en calibración."""
    residuals = np.asarray(y_calib, dtype=float) - np.asarray(point_calib, dtype=float)
    offsets = np.quantile(residuals[~np.isnan(residuals)], _LEVELS)
    grid = np.asarray(point, dtype=float)[:, None] + offsets[None, :]
    if lower is not None:
        grid = np.maximum(grid, lower)
    return make_monotone(grid)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def compare_objectives(
    frame: pd.DataFrame,
    target: str,
    test_seasons: list[int],
    *,
    random_state: int = 1729,
) -> pd.DataFrame:
    """Walk-forward por temporada: entrena < S−1, calibra en S−1, evalúa en S."""
    frame = add_derived_features(frame)
    rows: list[dict[str, Any]] = []
    objectives = ["poisson", "l2"] if target in SPARSE_COUNT_PROPS else list(POINT_OBJECTIVES)
    lower = QUANTILE_LOWER_BOUND.get(target, 0.0)

    for season in test_seasons:
        fit = frame[frame["season"] < season - 1]
        calib = frame[frame["season"] == season - 1]
        test = frame[frame["season"] == season]
        if fit.empty or calib.empty or test.empty:
            continue
        y_test = test[target].astype(float).to_numpy()
        y_calib = calib[target].astype(float).to_numpy()

        for objective in objectives:
            if target in SPARSE_COUNT_PROPS and objective == "l2":
                # resolve_params fuerza Poisson en conteos; L2 (el objetivo
                # heredado) se construye a mano para compararlo.
                from nflp.agents.modeling.models import build_estimator
                from nflp.agents.modeling.props import resolve_params

                features = [f for f in PROP_FEATURE_SETS[target] if f in fit.columns]
                params, _ = resolve_params(target, random_state=random_state)
                l2 = build_estimator("lightgbm", "regression", {**params, "objective": "regression"})
                l2.fit(fit[features].astype(float), fit[target].astype(float))
                point = np.maximum(l2.predict(test[features].astype(float)), 1e-6)
            else:
                model = train_prop_model(fit, calib, target, objective=objective,
                                         random_state=random_state, with_distribution=False)
                point = model.predict(test)

            record: dict[str, Any] = {
                "target": target, "season": season, "variant": f"point_{objective}",
                "n": len(test),
                "mae": float(np.mean(np.abs(y_test - point))),
                "rmse": float(np.sqrt(np.mean((y_test - point) ** 2))),
                "bias": float(np.mean(point - y_test)),
            }
            if target in SPARSE_COUNT_PROPS:
                record["crps"] = crps_poisson(y_test, point)
                record["coverage_80"] = randomized_pit_coverage(y_test, point)
                record["poisson_deviance"] = poisson_deviance(y_test, point)
            else:
                point_calib = model.predict(calib)
                grid = residual_grid(point_calib, y_calib, point, lower)
                record["crps"] = crps_quantile_grid(y_test, grid)
                record["coverage_80"] = interval_coverage(y_test, grid[:, P10_INDEX], grid[:, P90_INDEX])
            rows.append(record)

        if target not in SPARSE_COUNT_PROPS:
            # La rejilla servida: LightGBM cuantil + calibración conformal.
            model = train_prop_model(fit, calib, target, random_state=random_state)
            grid = model.quantiles(test)
            rows.append({
                "target": target, "season": season, "variant": "quantile_grid_conformal",
                "n": len(test),
                "mae": float(np.mean(np.abs(y_test - grid[:, QUANTILE_LEVELS.index(0.5)]))),
                "rmse": float(np.sqrt(np.mean((y_test - grid[:, QUANTILE_LEVELS.index(0.5)]) ** 2))),
                "bias": float(np.mean(grid[:, QUANTILE_LEVELS.index(0.5)] - y_test)),
                "crps": crps_quantile_grid(y_test, grid),
                "coverage_80": interval_coverage(y_test, grid[:, P10_INDEX], grid[:, P90_INDEX]),
            })
        log.info("Objetivos de '%s' evaluados en %d.", target, season)

    return pd.DataFrame(rows)


def summarize_objectives(per_season: pd.DataFrame) -> pd.DataFrame:
    """Media ponderada por filas sobre las temporadas."""
    rows = []
    for (target, variant), group in per_season.groupby(["target", "variant"]):
        weights = group["n"].to_numpy(dtype=float)
        out: dict[str, Any] = {"target": target, "variant": variant, "n": int(weights.sum())}
        for column in ("mae", "rmse", "bias", "crps", "coverage_80", "poisson_deviance"):
            if column in group and group[column].notna().any():
                out[column] = float(np.average(group[column].fillna(0), weights=weights))
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["target", "mae"]).reset_index(drop=True)


# ----------------------------------------------------------------- Optuna


def tune_prop(
    frame: pd.DataFrame,
    target: str,
    *,
    objective: str | None,
    val_seasons: list[int],
    n_trials: int = 40,
    random_state: int = 1729,
) -> tuple[dict[str, Any], float, float]:
    """Optuna con pliegues por temporada. Devuelve (params, score_defecto, score_mejor)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frame = add_derived_features(frame)
    folds = [
        (frame[frame["season"] < s], frame[frame["season"] == s]) for s in val_seasons
    ]
    folds = [(tr, te) for tr, te in folds if not tr.empty and not te.empty]
    is_count = target in SPARSE_COUNT_PROPS

    def score(params: dict[str, Any] | None) -> float:
        values = []
        for train, test in folds:
            model = train_prop_model(
                train, train.head(0), target, objective=objective, params=params,
                random_state=random_state, with_distribution=False,
            )
            prediction = model.predict(test)
            truth = test[target].astype(float).to_numpy()
            values.append(
                poisson_deviance(truth, prediction) if is_count
                else float(np.mean(np.abs(truth - prediction)))
            )
        return float(np.mean(values))

    def objective_fn(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        }
        return score(params)

    # Se parte de los valores por defecto (sin afinar) como primera prueba.
    defaults = {
        "n_estimators": 250, "learning_rate": 0.02, "num_leaves": 7, "max_depth": 3,
        "min_child_samples": 80, "subsample": 0.7, "colsample_bytree": 0.5,
        "reg_alpha": 2.0, "reg_lambda": 10.0,
    }
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.enqueue_trial(defaults)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    baseline = study.trials[0].value
    log.info(
        "Optuna '%s': defecto=%.4f mejor=%.4f (%d pruebas, pliegues %s).",
        target, baseline, study.best_value, n_trials, val_seasons,
    )
    return {**study.best_params, "subsample_freq": 1}, float(baseline), float(study.best_value)


def save_tuned_params(target: str, params: dict[str, Any]) -> None:
    current = load_tuned_params()
    current[target] = params
    TUNED_PARAMS_PATH.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "compare_objectives",
    "poisson_deviance",
    "residual_grid",
    "save_tuned_params",
    "summarize_objectives",
    "tune_prop",
    "PoissonDistribution",
    "QuantileGridDistribution",
]
