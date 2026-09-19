"""Agente de entrenamiento y validación de props de jugador.

Mantiene la misma disciplina que los modelos de partido: corte temporal,
walk-forward, calibración fuera de muestra y gate de promoción.

La evaluación usa **CRPS** además de MAE. El CRPS mide la calidad de toda la
distribución predictiva, que es lo que realmente se usa: un prop se resuelve
contra una línea, así que un modelo que acierta la media pero se equivoca en la
dispersión produce probabilidades over/under malas aunque su MAE sea bueno.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd

from nflp.agents.backtesting.metrics import crps_gaussian, regression_metrics
from nflp.agents.modeling.props import (
    PROP_FEATURE_SETS,
    PropModel,
    empirical_bayes_shrinkage,
    filter_prop_rows,
    train_prop_model,
)
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger
from nflp.registry import ModelRecord, ModelRegistry

log = get_logger(__name__)


@dataclass
class PropBacktestResult:
    """Resultado de validar un prop."""

    target: str
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    n_rows: int


class PropsAgent:
    """Entrena, valida y registra los modelos de props."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.registry = ModelRegistry(self.settings, self.db)

    # ------------------------------------------------------------------ datos

    def load_dataset(self, target: str, seasons: list[int] | None = None) -> pd.DataFrame:
        """Une el feature store de jugador con el resultado real del prop."""
        clause = ""
        if seasons:
            clause = f"AND f.season IN ({', '.join(str(s) for s in seasons)})"

        frame = self.db.query(
            f"""
            SELECT f.*, s.{target} AS {target}, g.kickoff_utc
            FROM features_player f
            JOIN player_stats_weekly s
              ON s.player_id = f.gsis_id AND s.season = f.season
             AND s.week = f.week AND s.season_type = f.season_type
            JOIN games g ON g.game_id = f.game_id
            WHERE f.feature_set_version = '{self.settings.features.version}'
              AND s.{target} IS NOT NULL
              {clause}
            ORDER BY g.kickoff_utc
            """
        )
        if frame.empty:
            raise ValueError(
                f"Sin datos para el prop '{target}'. "
                "¿Se ejecutó `nflp features build-players`?"
            )

        frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], errors="coerce")
        filtered = filter_prop_rows(frame, target, self.settings.props.min_projected_snaps)

        log.info(
            "Prop '%s': %d filas tras filtrar (de %d), %d jugadores.",
            target, len(filtered), len(frame), filtered["gsis_id"].nunique(),
        )
        return filtered

    # ------------------------------------------------------------ walk-forward

    def backtest(
        self,
        target: str,
        *,
        start_season: int | None = None,
        end_season: int | None = None,
        compare_semisupervised: bool = True,
    ) -> PropBacktestResult:
        """Valida el prop con walk-forward y compara las variantes.

        Compara tres enfoques:
        1. **Base** — solo jugadores de muestra amplia.
        2. **Pseudo-labeling** — el semi-supervisado que pide el prompt.
        3. **Shrinkage empírico-Bayes** — el control metodológico.
        """
        config = self.settings.backtest
        start = start_season or config.start_season
        end = end_season or config.end_season

        frame = self.load_dataset(target)
        seasons = sorted(frame["season"].unique())
        evaluable = [
            s for s in seasons
            if start <= s <= end and s - min(seasons) >= config.min_train_seasons
        ]
        if not evaluable:
            raise ValueError(f"Sin temporadas evaluables para el prop '{target}'.")

        variants = {"base": False}
        if compare_semisupervised:
            variants["pseudo_labeling"] = True

        collected: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}

        for season in evaluable:
            train = frame[frame["season"] < season]
            test = frame[frame["season"] == season]
            if len(train) < 1000 or test.empty:
                continue

            # Última temporada del bloque de entrenamiento: calibración.
            calib_season = sorted(train["season"].unique())[-1]
            fit = train[train["season"] < calib_season]
            calib = train[train["season"] == calib_season]
            if fit.empty or calib.empty:
                fit, calib = train, train.tail(0)

            for name, use_pseudo in variants.items():
                model = train_prop_model(
                    fit, calib, target,
                    random_state=self.settings.modeling.random_seed,
                    use_pseudo_labels=use_pseudo,
                )
                prediction = model.predict(test)
                block = test[["game_id", "gsis_id", "season", "week", "position", "team"]].copy()
                block["y_true"] = test[target].to_numpy()
                block["y_pred"] = prediction
                block["y_std"] = model.distribution.std(prediction)
                block["variant"] = name
                collected[name].append(block)

            log.info("  prop '%s' temporada %d validada (%d filas).", target, season, len(test))

        rows: list[dict] = []
        all_predictions: list[pd.DataFrame] = []

        for name, blocks in collected.items():
            if not blocks:
                continue
            predictions = pd.concat(blocks, ignore_index=True)
            all_predictions.append(predictions)

            metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
            crps = crps_gaussian(
                predictions["y_true"].to_numpy(),
                predictions["y_pred"].to_numpy(),
                predictions["y_std"].to_numpy(),
            )
            rows.append(
                {
                    "variante": name, "n": metrics.n,
                    "mae": round(metrics.mae, 3), "rmse": round(metrics.rmse, 3),
                    "crps": round(crps, 3),
                }
            )

        # --- control: shrinkage empírico-Bayes sobre la media reciente ---
        if all_predictions:
            baseline = all_predictions[0].copy()
            recent = f"{target}_r5"
            source = self.load_dataset(target)[["game_id", "gsis_id", recent, "games_played_prior", "position"]]
            merged = baseline.merge(source, on=["game_id", "gsis_id"], how="left", suffixes=("", "_s"))

            shrunk = empirical_bayes_shrinkage(
                merged[recent], merged["position_s"].fillna(merged["position"]),
                merged["games_played_prior"],
            )
            metrics = regression_metrics(merged["y_true"], shrunk)
            rows.append(
                {
                    "variante": "control_shrinkage_eb", "n": metrics.n,
                    "mae": round(metrics.mae, 3), "rmse": round(metrics.rmse, 3),
                    "crps": None,
                }
            )

            naive = regression_metrics(merged["y_true"], merged[recent])
            rows.append(
                {
                    "variante": "baseline_media_5_partidos", "n": naive.n,
                    "mae": round(naive.mae, 3), "rmse": round(naive.rmse, 3), "crps": None,
                }
            )

        comparison = pd.DataFrame(rows).sort_values("mae")
        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

        return PropBacktestResult(
            target=target, comparison=comparison,
            predictions=predictions, n_rows=len(predictions),
        )

    # --------------------------------------------------------------- entrenar

    def train(
        self, target: str, *, val_seasons: int = 2, register: bool = True
    ) -> PropModel:
        """Entrena el modelo final de un prop y lo registra."""
        frame = self.load_dataset(target)
        seasons = sorted(frame["season"].unique())
        if len(seasons) < 4:
            raise ValueError(f"Se necesitan al menos 4 temporadas para el prop '{target}'.")

        val_cutoff = seasons[-val_seasons]
        calib_cutoff = seasons[-(val_seasons + 1)]

        fit = frame[frame["season"] < calib_cutoff]
        calib = frame[(frame["season"] >= calib_cutoff) & (frame["season"] < val_cutoff)]
        validation = frame[frame["season"] >= val_cutoff]

        model = train_prop_model(
            fit, calib, target,
            random_state=self.settings.modeling.random_seed,
            use_pseudo_labels=True,
        )

        train_pred = model.predict(fit)
        val_pred = model.predict(validation)
        train_metrics = regression_metrics(fit[target], train_pred)
        val_metrics = regression_metrics(validation[target], val_pred)

        crps = crps_gaussian(
            validation[target].to_numpy(), val_pred, model.distribution.std(val_pred)
        )
        model.metrics = {
            "train_mae": train_metrics.mae, "val_mae": val_metrics.mae,
            "val_rmse": val_metrics.rmse, "val_crps": crps,
        }

        log.info(
            "Prop '%s': MAE train=%.2f val=%.2f · RMSE=%.2f · CRPS=%.2f",
            target, train_metrics.mae, val_metrics.mae, val_metrics.rmse, crps,
        )

        if register:
            decision = self.registry.evaluate_promotion(
                train_metrics.mae, val_metrics.mae, lower_is_better=True
            )
            log.info(decision.describe())

            record = ModelRecord(
                model_id=str(uuid.uuid4()),
                target=target,
                family=model.family,
                version=self.registry.next_version(target),
                feature_set_version=self.settings.features.version,
                feature_names=model.feature_names,
                hyperparameters={"pseudo_labels": model.used_pseudo_labels},
                dataset_hash="",
                train_seasons=f"{fit['season'].min()}-{fit['season'].max()}",
                n_train_rows=len(fit),
                n_val_rows=len(validation),
                metric_name="mae",
                train_metric=train_metrics.mae,
                val_metric=val_metrics.mae,
                metrics=model.metrics,
                status="shadow",
            )
            self.registry.register(
                record,
                {
                    "model": model.model,
                    "distribution": model.distribution,
                    "feature_names": model.feature_names,
                    "target": target,
                    "kind": "prop",
                    "family": model.family,
                },
                decision=decision,
            )

        return model

    def train_all(self) -> list[PropModel]:
        return [self.train(target) for target in self.settings.props.targets]


def prop_feature_names(target: str) -> list[str]:
    return PROP_FEATURE_SETS[target]
