"""Motor de validación walk-forward.

Es la pieza que decide si este proyecto vale algo. La regla es absoluta: **para
predecir la semana N solo se puede usar información anterior a la semana N.**

Esquema (ventana expansiva):

```
temporada 2015  |████ entrena ████|→ predice sem. 1-4
                |████ entrena ██████|→ predice sem. 5-8
                |████ entrena ████████|→ predice sem. 9-12
temporada 2016  |████ entrena ██████████|→ ...
```

El modelo se reentrena cada ``retrain_every_weeks`` semanas. Reentrenar cada
semana sería más fiel a producción pero multiplica el coste por cuatro sin
cambiar materialmente las conclusiones; el parámetro es configurable.

Lo que **no** se hace y es el error habitual: un `train_test_split` aleatorio.
Sobre series temporales, eso entrena con partidos de 2024 para predecir 2016,
produce métricas excelentes y un modelo que pierde dinero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nflp.agents.modeling.calibration import ProbabilityCalibrator, ResidualDistribution
from nflp.agents.modeling.datasets import TARGET_SPECS, Dataset
from nflp.agents.modeling.models import build_estimator, default_params, predict_proba
from nflp.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    """Parámetros de la corrida walk-forward."""

    start_season: int = 2015
    end_season: int = 2025
    retrain_every_weeks: int = 4
    min_train_seasons: int = 3
    family: str = "catboost"
    params: dict | None = None
    calibrate: bool = True
    # Temporadas finales del bloque de entrenamiento reservadas a calibrar.
    calibration_seasons: int = 2


@dataclass
class WalkForwardResult:
    """Predicciones fuera de muestra de una corrida completa."""

    predictions: pd.DataFrame
    target: str
    kind: str
    n_retrains: int
    feature_names: list[str] = field(default_factory=list)


def run_walk_forward(
    dataset: Dataset,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Ejecuta la validación walk-forward y devuelve predicciones fuera de muestra.

    Cada fila del resultado fue predicha por un modelo que **no vio ese partido
    ni ninguno posterior** durante su entrenamiento.
    """
    config = config or WalkForwardConfig()
    kind = dataset.kind
    params = config.params or default_params(config.family, kind)

    meta = dataset.meta.copy()
    meta["_row"] = np.arange(len(meta))

    # Orden cronológico canónico: (temporada, tipo, semana). Los playoffs van
    # después de la temporada regular aunque su numeración de semana se solape.
    meta["_phase"] = meta["season_type"].map({"REG": 0, "POST": 1}).fillna(0).astype(int)
    meta = meta.sort_values(["season", "_phase", "week", "kickoff_utc"])

    seasons = sorted(meta["season"].unique())
    evaluable = [
        s for s in seasons
        if config.start_season <= s <= config.end_season
        and s - min(seasons) >= config.min_train_seasons
    ]
    if not evaluable:
        raise ValueError(
            f"No hay temporadas evaluables entre {config.start_season} y {config.end_season} "
            f"con {config.min_train_seasons} temporadas mínimas de historia."
        )

    log.info(
        "Walk-forward '%s' (%s): temporadas %d-%d, reentrenamiento cada %d semanas.",
        dataset.target, config.family, evaluable[0], evaluable[-1], config.retrain_every_weeks,
    )

    blocks = _build_blocks(meta, evaluable, config.retrain_every_weeks)
    records: list[pd.DataFrame] = []
    n_retrains = 0

    for block_index, block in enumerate(blocks, start=1):
        test_rows = block["rows"]
        # Entrenamiento: TODO lo estrictamente anterior al primer partido del bloque.
        cutoff = block["start_ts"]
        train_mask = meta["kickoff_utc"] < cutoff
        train_rows = meta.loc[train_mask, "_row"].to_numpy()

        if len(train_rows) < 300:
            log.debug("Bloque %d omitido: solo %d filas de entrenamiento.", block_index, len(train_rows))
            continue

        train_set = _subset(dataset, train_rows)
        test_set = _subset(dataset, test_rows)

        # Sub-corte para calibrar: las últimas temporadas del bloque de
        # entrenamiento. El modelo se entrena sin ellas, calibra con ellas.
        calib_set = None
        fit_set = train_set
        if config.calibrate:
            train_seasons_sorted = sorted(train_set.meta["season"].unique())
            if len(train_seasons_sorted) > config.calibration_seasons + 1:
                calib_cutoff = train_seasons_sorted[-config.calibration_seasons]
                is_calib = train_set.meta["season"] >= calib_cutoff
                if is_calib.sum() >= 150 and (~is_calib).sum() >= 300:
                    fit_set = train_set.slice(~is_calib)
                    calib_set = train_set.slice(is_calib)

        model = build_estimator(config.family, kind, params)
        model.fit(fit_set.X, fit_set.y)
        n_retrains += 1

        frame = test_set.meta.copy()
        frame["y_true"] = test_set.y.to_numpy()

        if kind == "classification":
            raw = predict_proba(model, test_set.X)
            if calib_set is not None:
                calibrator = ProbabilityCalibrator("isotonic").fit(
                    predict_proba(model, calib_set.X), calib_set.y.to_numpy()
                )
                frame["y_prob"] = calibrator.transform(raw)
            else:
                frame["y_prob"] = np.clip(raw, 1e-6, 1 - 1e-6)
            frame["y_pred"] = (frame["y_prob"] >= 0.5).astype(float)
        else:
            frame["y_pred"] = model.predict(test_set.X)
            if calib_set is not None:
                residuals = ResidualDistribution().fit(
                    calib_set.y.to_numpy(), model.predict(calib_set.X)
                )
            else:
                residuals = ResidualDistribution()
            frame["y_std"] = residuals.sigma
            lower, upper = residuals.interval(frame["y_pred"].to_numpy())
            frame["lower_80"], frame["upper_80"] = lower, upper

        records.append(frame)

        if block_index % 10 == 0 or block_index == len(blocks):
            log.info(
                "  bloque %d/%d — %s sem. %s — %d partidos (entrenado con %d).",
                block_index, len(blocks), block["season"], block["weeks"],
                len(test_rows), len(fit_set),
            )

    if not records:
        raise RuntimeError("El walk-forward no produjo predicciones.")

    predictions = pd.concat(records, ignore_index=True)
    log.info(
        "Walk-forward completado: %d predicciones fuera de muestra, %d reentrenamientos.",
        len(predictions), n_retrains,
    )

    return WalkForwardResult(
        predictions=predictions,
        target=dataset.target,
        kind=kind,
        n_retrains=n_retrains,
        feature_names=dataset.feature_names,
    )


def _build_blocks(
    meta: pd.DataFrame, evaluable_seasons: list[int], every: int
) -> list[dict]:
    """Agrupa las semanas evaluables en bloques de reentrenamiento."""
    blocks: list[dict] = []

    for season in evaluable_seasons:
        season_rows = meta[meta["season"] == season]
        if season_rows.empty:
            continue

        # Ordenación por fase y semana, para que los playoffs queden al final.
        ordered_weeks = (
            season_rows[["_phase", "week"]]
            .drop_duplicates()
            .sort_values(["_phase", "week"])
            .to_records(index=False)
        )

        for start in range(0, len(ordered_weeks), every):
            chunk = ordered_weeks[start : start + every]
            selector = pd.Series(False, index=season_rows.index)
            for phase, week in chunk:
                selector |= (season_rows["_phase"] == phase) & (season_rows["week"] == week)

            block_rows = season_rows.loc[selector]
            if block_rows.empty:
                continue

            blocks.append(
                {
                    "season": season,
                    "weeks": f"{chunk[0][1]}-{chunk[-1][1]}",
                    "rows": block_rows["_row"].to_numpy(),
                    "start_ts": block_rows["kickoff_utc"].min(),
                }
            )

    return blocks


def _subset(dataset: Dataset, rows: np.ndarray) -> Dataset:
    """Subconjunto por posiciones de fila."""
    return Dataset(
        X=dataset.X.iloc[rows],
        y=dataset.y.iloc[rows],
        meta=dataset.meta.iloc[rows],
        target=dataset.target,
        kind=dataset.kind,
        feature_names=dataset.feature_names,
        dataset_hash=dataset.dataset_hash,
    )


def walk_forward_all_families(
    dataset: Dataset, families: list[str], config: WalkForwardConfig
) -> dict[str, WalkForwardResult]:
    """Ejecuta el walk-forward para varias familias y devuelve cada resultado."""
    results: dict[str, WalkForwardResult] = {}
    for family in families:
        family_config = WalkForwardConfig(
            **{**config.__dict__, "family": family, "params": None}
        )
        results[family] = run_walk_forward(dataset, family_config)
    return results


def default_metric_name(target: str) -> str:
    return TARGET_SPECS.get(target, {}).get("metric", "mae")
