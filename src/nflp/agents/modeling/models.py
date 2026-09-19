"""Familias de modelos y sus espacios de hiperparámetros.

Se comparan tres implementaciones de gradient boosting —LightGBM, XGBoost y
CatBoost— y gana la que mejor valida temporalmente, no la elegida de antemano.

**Por qué gradient boosting y no deep learning** (justificación pedida en la
sección 3 del prompt):

* El volumen es pequeño: ~285 partidos por temporada, ~4.400 en 16 temporadas.
  Una red con capacidad suficiente para capturar secuencias temporales tiene más
  parámetros que muestras.
* La señal es débil y el ruido alto. El resultado de un partido de NFL depende de
  un balón suelto que bota hacia un lado u otro. Los GBM con regularización
  fuerte degradan con elegancia ante ese ruido; las redes lo memorizan.
* La interpretabilidad no es opcional aquí: el dashboard debe justificar cada
  predicción con SHAP. Los valores SHAP sobre árboles son exactos y rápidos;
  sobre redes son aproximados y caros.
* El estado del arte público en predicción NFL (FiveThirtyEight, nflfastR) usa
  modelos de esta familia o incluso más simples. No hay evidencia de que las
  redes ganen en este dominio.

**Los valores por defecto son deliberadamente conservadores**: profundidad baja,
regularización L1/L2 activa y submuestreo. En un dominio ruidoso, el sobreajuste
es el modo de fallo por defecto, no una posibilidad remota.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from nflp.core.logging import get_logger

log = get_logger(__name__)

SUPPORTED_FAMILIES = ("lightgbm", "xgboost", "catboost")


class Estimator(Protocol):
    """Interfaz mínima que debe cumplir un modelo entrenable."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
    def predict(self, X: Any) -> np.ndarray: ...


@dataclass
class ModelSpec:
    """Configuración de una familia de modelos para un tipo de tarea."""

    family: str
    kind: str                     # classification | regression
    params: dict[str, Any]


# --------------------------------------------------------------------- defaults

def default_params(family: str, kind: str, random_state: int = 1729) -> dict[str, Any]:
    """Hiperparámetros por defecto, regularizados para un dominio ruidoso."""
    if family == "lightgbm":
        params = {
            "n_estimators": 250,
            "learning_rate": 0.02,
            "num_leaves": 7,           # muy bajo: árboles casi de decisión simple
            "max_depth": 3,            # profundidad mínima útil
            "min_child_samples": 80,   # exige soporte amplio antes de dividir
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.5,   # decorrelaciona árboles con 90 features
            "reg_alpha": 2.0,          # L1
            "reg_lambda": 10.0,        # L2 fuerte
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        if kind == "classification":
            params["objective"] = "binary"
        else:
            # MAE es más robusto a resultados extremos (palizas) que MSE.
            params["objective"] = "regression_l1"
        return params

    if family == "xgboost":
        params = {
            "n_estimators": 250,
            "learning_rate": 0.02,
            "max_depth": 3,
            "min_child_weight": 25,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        params["objective"] = (
            "binary:logistic" if kind == "classification" else "reg:absoluteerror"
        )
        return params

    if family == "catboost":
        return {
            "iterations": 250,
            "learning_rate": 0.02,
            "depth": 3,
            "l2_leaf_reg": 15.0,
            "rsm": 0.5,
            "random_seed": random_state,
            "verbose": 0,
            "allow_writing_files": False,
            "loss_function": "Logloss" if kind == "classification" else "MAE",
        }

    raise ValueError(f"Familia desconocida: {family}. Válidas: {SUPPORTED_FAMILIES}")


def build_estimator(family: str, kind: str, params: dict[str, Any] | None = None) -> Estimator:
    """Instancia un modelo de la familia indicada."""
    params = params or default_params(family, kind)

    if family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        return LGBMClassifier(**params) if kind == "classification" else LGBMRegressor(**params)

    if family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        return XGBClassifier(**params) if kind == "classification" else XGBRegressor(**params)

    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        return (
            CatBoostClassifier(**params) if kind == "classification" else CatBoostRegressor(**params)
        )

    raise ValueError(f"Familia desconocida: {family}")


# ------------------------------------------------------------- espacio de búsqueda

def suggest_params(trial: Any, family: str, kind: str, random_state: int = 1729) -> dict[str, Any]:
    """Espacio de búsqueda de Optuna.

    Los rangos están deliberadamente acotados hacia la regularización: se permite
    explorar, pero no llegar a configuraciones que memorizan el histórico.
    """
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 16),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
            "objective": "binary" if kind == "classification" else "regression_l1",
        }

    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "binary:logistic" if kind == "classification" else "reg:absoluteerror",
        }

    if family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "depth": trial.suggest_int("depth", 2, 5),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 60.0, log=True),
            "rsm": trial.suggest_float("rsm", 0.3, 0.8),
            "random_seed": random_state,
            "verbose": 0,
            "allow_writing_files": False,
            "loss_function": "Logloss" if kind == "classification" else "MAE",
        }

    raise ValueError(f"Familia desconocida: {family}")


def predict_proba(model: Any, X: Any) -> np.ndarray:
    """Probabilidad de la clase positiva, homogénea entre las tres librerías."""
    proba = model.predict_proba(X)
    array = np.asarray(proba)
    return array[:, 1] if array.ndim == 2 else array


def feature_importances(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Importancias del modelo, normalizadas a suma 1."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {}
    values = np.asarray(importances, dtype=float)
    total = values.sum()
    if total <= 0:
        return {}
    return dict(sorted(zip(feature_names, values / total, strict=False), key=lambda kv: -kv[1]))
