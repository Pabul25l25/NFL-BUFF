"""Agente de entrenamiento y validación de props de jugador.

Mantiene la misma disciplina que los modelos de partido: corte temporal,
walk-forward, calibración fuera de muestra y gate de promoción.

La evaluación usa **CRPS** además de MAE, y el CRPS puntúa la distribución que
realmente se sirve: rejilla de cuantiles p05..p95 para los continuos, Poisson
o binomial negativa exactas para los conteos (ver ``distributions.py``). Un modelo que acierta la
media pero se equivoca en la dispersión produce probabilidades over/under malas
aunque su MAE sea bueno. Además se mide la **cobertura del 80 %** (literal en
continuos, PIT aleatorizado en conteos) y el **sesgo** medio.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nflp.agents.backtesting.metrics import regression_metrics
from nflp.agents.modeling.distributions import (
    P10_INDEX,
    P50_INDEX,
    P90_INDEX,
    QUANTILE_LEVELS,
    PoissonDistribution,
    crps_poisson_rows,
    crps_quantile_grid_rows,
    interval_coverage,
    randomized_pit_covered,
)
from nflp.agents.modeling.props import (
    PROP_FEATURE_SETS,
    PROP_MIN_TRAIN_SEASON,
    QUANTILE_LOWER_BOUND,
    SPARSE_COUNT_PROPS,
    PropModel,
    filter_prop_rows,
    served_coverage,
    served_crps,
    served_crps_rows,
    served_dispersion,
    served_interval,
    served_mean,
    served_point,
    served_quantiles,
    train_prop_model,
)
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger
from nflp.registry import ModelRecord, ModelRegistry

log = get_logger(__name__)

# Banda de cobertura del 80 % exigida para promover un prop.
COVERAGE_BAND: tuple[float, float] = (0.75, 0.85)


@dataclass
class PropBacktestResult:
    """Resultado de validar un prop."""

    target: str
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    n_rows: int
    by_season: pd.DataFrame | None = None


def score_rows(
    y: np.ndarray,
    point: np.ndarray,
    crps: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    covered: np.ndarray | None = None,
) -> pd.DataFrame:
    """Métricas por fila: error, CRPS y cobertura del 80 %.

    ``covered`` permite pasar la cobertura por PIT aleatorizado (Poisson); si
    no se pasa, se usa la cobertura literal del intervalo.
    """
    y = np.asarray(y, dtype=float)
    point = np.asarray(point, dtype=float)
    literal = ((y >= lower) & (y <= upper)).astype(float)
    return pd.DataFrame(
        {
            "y_true": y, "y_pred": point, "lower_80": lower, "upper_80": upper,
            "abs_err": np.abs(y - point), "sq_err": (y - point) ** 2,
            "err": point - y, "crps": crps,
            "covered_80": literal if covered is None else covered,
            "covered_literal": literal,
        }
    )


def aggregate_scores(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = rows.groupby(by, dropna=False)
    out = grouped.agg(
        n=("y_true", "size"),
        mae=("abs_err", "mean"),
        rmse=("sq_err", lambda s: float(np.sqrt(s.mean()))),
        bias=("err", "mean"),
        crps=("crps", "mean"),
        coverage_80=("covered_80", "mean"),
        coverage_literal=("covered_literal", "mean"),
        mean_pred=("y_pred", "mean"),
        mean_true=("y_true", "mean"),
    )
    return out.reset_index()


def distribution_diagnostics(
    distribution: Any, y: np.ndarray, point: np.ndarray, X: pd.DataFrame, target: str
) -> dict[str, float]:
    """Sesgo de la media servida y calibración de la mediana (informativos).

    * ``val_mean_bias`` = media(``mean_estimate`` − real): debe ser ≈ 0.
    * ``val_frac_below_p50`` / ``val_frac_below_point``: fracción de reales por
      debajo de la mediana de la rejilla y de la estimación puntual (≈ 0,5).
    * ``val_nb_size_mean``: tamaño r medio si la distribución es binomial negativa.
    """
    y = np.asarray(y, dtype=float)
    point = np.asarray(point, dtype=float)
    out = {
        "val_mean_bias": float(np.mean(served_mean(distribution, point, X, target=target) - y)),
        "val_frac_below_point": float(np.mean(y < point)),
    }
    if target not in SPARSE_COUNT_PROPS:
        out["val_frac_below_p50"] = float(
            np.mean(y < served_quantiles(distribution, point, X)[:, P50_INDEX])
        )
    size = served_dispersion(distribution, point)
    if np.isfinite(size).any():
        out["val_nb_size_mean"] = float(np.nanmean(size))
    return out


def evaluate_candidate(
    candidate: dict[str, float], production: dict[str, float] | None
) -> tuple[bool, str]:
    """Regla de promoción de props: MAE **y** CRPS mejoran y cobertura en banda."""
    coverage = candidate.get("val_coverage_80", float("nan"))
    low, high = COVERAGE_BAND
    if not (low <= coverage <= high):
        return False, f"cobertura 80% en validación {coverage:.3f} fuera de [{low}, {high}]"
    if not production:
        return True, "sin modelo de producción con el que comparar"
    mae_ok = candidate["val_mae"] < production["val_mae"]
    prod_crps = production.get("val_crps")
    if prod_crps is None or pd.isna(prod_crps):
        # Producción anterior a props v2 sin CRPS: solo se puede comparar el MAE.
        crps_ok = True
        crps_text = f"CRPS {candidate['val_crps']:.4f} vs n/d"
    else:
        crps_ok = candidate["val_crps"] < prod_crps
        crps_text = f"CRPS {candidate['val_crps']:.4f} vs {prod_crps:.4f}"
    detail = (
        f"MAE {candidate['val_mae']:.4f} vs {production['val_mae']:.4f}, "
        f"{crps_text}, cobertura {coverage:.3f}"
    )
    if mae_ok and crps_ok:
        return True, "mejora MAE y CRPS: " + detail
    return False, "no mejora MAE y CRPS a la vez: " + detail


class PropsAgent:
    """Entrena, valida y registra los modelos de props."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.registry = ModelRegistry(self.settings, self.db)

    # ------------------------------------------------------------------ datos

    def load_dataset(
        self, target: str, seasons: list[int] | None = None, *, min_season: int | None = None
    ) -> pd.DataFrame:
        """Une el feature store de jugador con el resultado real del prop."""
        clauses = []
        if seasons:
            clauses.append(f"AND f.season IN ({', '.join(str(int(s)) for s in seasons)})")
        if min_season is not None:
            clauses.append(f"AND f.season >= {int(min_season)}")

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
              {' '.join(clauses)}
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
        """Walk-forward por temporada: nuevo vs actual vs media de 5 partidos.

        Para cada temporada evaluada ``S``:

        * **nuevo** — features v2, entrenado en ``[2016, S−2]``, distribución
          calibrada en ``S−1``.
        * **actual** — el modelo de producción v1 (features v1, L1/L2,
          pseudo-labeling, log-normal global) entrenado en ``< S−1`` y con la
          log-normal ajustada en ``S−1``.
        * **media_5_partidos** — ``{target}_r5`` como punto; Poisson(r5) en
          conteos y ``r5 + cuantiles de residuos de S−1`` en continuos.

        Todas las variantes se puntúan con las mismas métricas (CRPS de la
        distribución servida, cobertura del 80 %, sesgo).
        """
        config = self.settings.backtest
        frame = self.load_dataset(target)
        seasons = sorted(int(s) for s in frame["season"].unique())
        first = PROP_MIN_TRAIN_SEASON + config.min_train_seasons + 1
        start = max(start_season or first, first)
        end = end_season or max(seasons)
        evaluable = [s for s in seasons if start <= s <= end]
        if not evaluable:
            raise ValueError(f"Sin temporadas evaluables para el prop '{target}'.")

        is_count = target in SPARSE_COUNT_PROPS
        recent = f"{target}_r5"
        lower_bound = QUANTILE_LOWER_BOUND.get(target, 0.0)
        seed = self.settings.modeling.random_seed
        blocks: list[pd.DataFrame] = []

        for season in evaluable:
            test = frame[frame["season"] == season]
            calib = frame[frame["season"] == season - 1]
            if test.empty or calib.empty:
                continue
            meta = test[["game_id", "gsis_id", "season", "week", "position", "team"]].reset_index(drop=True)
            y = test[target].astype(float).to_numpy()

            # --- nuevo ---
            fit_new = frame[(frame["season"] >= PROP_MIN_TRAIN_SEASON) & (frame["season"] < season - 1)]
            new = train_prop_model(fit_new, calib, target, random_state=seed)
            point = new.predict(test)
            lower, upper = new.interval(test, point)
            crps = served_crps_rows(new.distribution, y, point, test)
            covered = (
                randomized_pit_covered(y, point, size=served_dispersion(new.distribution, point))
                if is_count else None
            )
            blocks.append(meta.join(score_rows(y, point, crps, lower, upper, covered)).assign(variant="nuevo"))

            # --- actual (v1) ---
            fit_old = frame[frame["season"] < season - 1]
            old = train_prop_model(fit_old, calib, target, legacy=True, random_state=seed)
            point = old.predict(test)
            lower, upper = old.interval(test, point)
            # El modelo actual sirve una log-normal también en conteos; en ellos se
            # puntúa con el CRPS casi exacto (rejilla fina) para compararlo con la
            # Poisson exacta del nuevo.
            crps = served_crps_rows(old.distribution, y, point, test, exact=is_count)
            blocks.append(meta.join(score_rows(y, point, crps, lower, upper)).assign(variant="actual"))

            # --- media de 5 partidos ---
            point = np.maximum(test[recent].astype(float).fillna(0.0).to_numpy(), 0.0)
            if is_count:
                lower, upper = PoissonDistribution().interval(point)
                crps = crps_poisson_rows(y, point)
                covered = randomized_pit_covered(y, point)
            else:
                from nflp.agents.modeling.props_tuning import residual_grid

                calib_point = np.maximum(calib[recent].astype(float).fillna(0.0).to_numpy(), 0.0)
                grid = residual_grid(calib_point, calib[target].astype(float).to_numpy(), point, lower_bound)
                lower, upper = grid[:, P10_INDEX], grid[:, P90_INDEX]
                crps = crps_quantile_grid_rows(y, grid)
                covered = None
            blocks.append(meta.join(score_rows(y, point, crps, lower, upper, covered))
                          .assign(variant="media_5_partidos"))

            log.info("  prop '%s' temporada %d validada (%d filas).", target, season, len(test))

        predictions = pd.concat(blocks, ignore_index=True)
        predictions["target"] = target
        comparison = aggregate_scores(predictions, ["variant"]).sort_values("crps")
        by_season = aggregate_scores(predictions, ["variant", "season"])
        comparison.insert(0, "target", target)
        by_season.insert(0, "target", target)

        return PropBacktestResult(
            target=target, comparison=comparison, predictions=predictions,
            n_rows=int((predictions["variant"] == "nuevo").sum()), by_season=by_season,
        )

    # --------------------------------------------------------------- entrenar

    def _split(self, frame: pd.DataFrame, val_seasons: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        seasons = sorted(frame["season"].unique())
        if len(seasons) < 4:
            raise ValueError("Se necesitan al menos 4 temporadas para un prop.")
        val_cutoff = seasons[-val_seasons]
        calib_cutoff = seasons[-(val_seasons + 1)]
        fit = frame[frame["season"] < calib_cutoff]
        calib = frame[(frame["season"] >= calib_cutoff) & (frame["season"] < val_cutoff)]
        validation = frame[frame["season"] >= val_cutoff]
        return fit, calib, validation

    def evaluate_artifact(self, artifact: dict[str, Any], validation: pd.DataFrame) -> dict[str, float]:
        """Métricas de un artefacto registrado sobre filas de validación dadas."""
        target = artifact["target"]
        distribution = artifact["distribution"]
        point = served_point(artifact["model"], distribution, validation, artifact["feature_names"])
        y = validation[target].astype(float).to_numpy()
        metrics = regression_metrics(y, point)
        lower, upper = served_interval(distribution, point, validation)
        return {
            "val_mae": metrics.mae,
            "val_rmse": metrics.rmse,
            "val_bias": float(np.mean(point - y)),
            "val_crps": served_crps(
                distribution, y, point, validation, exact=target in SPARSE_COUNT_PROPS
            ),
            "val_coverage_80": served_coverage(distribution, y, point, validation),
            "val_coverage_literal": interval_coverage(y, lower, upper),
            **distribution_diagnostics(distribution, y, point, validation, target),
            "n_val": float(len(validation)),
        }

    def train(
        self,
        target: str,
        *,
        val_seasons: int = 2,
        register: bool = True,
        min_season: int | None = PROP_MIN_TRAIN_SEASON,
    ) -> PropModel:
        """Entrena el modelo final de un prop y lo registra en shadow.

        Corte temporal: entrenamiento ``[2016, T−3]``, calibración de la
        distribución en ``T−2`` y validación en las dos últimas temporadas.
        """
        frame = self.load_dataset(target, min_season=min_season)
        fit, calib, validation = self._split(frame, val_seasons)

        model = train_prop_model(fit, calib, target, random_state=self.settings.modeling.random_seed)

        train_pred = model.predict(fit)
        val_pred = model.predict(validation)
        train_metrics = regression_metrics(fit[target], train_pred)
        val_metrics = regression_metrics(validation[target], val_pred)
        y_val = validation[target].astype(float).to_numpy()
        lower, upper = model.interval(validation, val_pred)

        model.metrics = {
            "train_mae": train_metrics.mae,
            "val_mae": val_metrics.mae,
            "val_rmse": val_metrics.rmse,
            "val_bias": float(np.mean(val_pred - y_val)),
            "val_crps": served_crps(
                model.distribution, y_val, val_pred, validation, exact=target in SPARSE_COUNT_PROPS
            ),
            "val_coverage_80": served_coverage(model.distribution, y_val, val_pred, validation),
            "val_coverage_literal": interval_coverage(y_val, lower, upper),
            "calib_coverage_80": float(getattr(model.distribution, "calibration_coverage", np.nan)),
            **distribution_diagnostics(model.distribution, y_val, val_pred, validation, target),
        }
        for key, value in getattr(model.distribution, "calibration", {}).items():
            model.metrics[f"calib_{key}"] = float(value)
        for key, value in getattr(model.distribution, "point_calibration", {}).items():
            model.metrics[f"calib_point_{key}"] = float(value)

        # La producción vigente, puntuada sobre exactamente las mismas filas y
        # con el mismo CRPS: la comparación de promoción es manzanas con manzanas.
        production = self.registry.load_production(target)
        if production is not None:
            prod_metrics = self.evaluate_artifact(production[1], validation)
            model.metrics.update({f"prod_{k}": v for k, v in prod_metrics.items()})

        log.info(
            "Prop '%s' (%s, %s): MAE train=%.3f val=%.3f · CRPS=%.3f · cobertura80=%.3f · "
            "sesgo=%+.3f · sesgo media=%+.3f",
            target, model.objective, model.distribution_kind, train_metrics.mae, val_metrics.mae,
            model.metrics["val_crps"], model.metrics["val_coverage_80"], model.metrics["val_bias"],
            model.metrics["val_mean_bias"],
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
                hyperparameters={
                    "pseudo_labels": model.used_pseudo_labels,
                    "objective": model.objective,
                    "distribution": model.distribution_kind,
                    **(
                        {
                            "nb_log_size_intercept": model.distribution.log_size_intercept,
                            "nb_log_size_slope": model.distribution.log_size_slope,
                        }
                        if model.distribution_kind == "negbin" else {}
                    ),
                    "min_train_season": min_season,
                    **{k: v for k, v in model.params.items() if k not in {"verbose", "n_jobs"}},
                },
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
                    "distribution_kind": model.distribution_kind,
                    "quantile_levels": list(QUANTILE_LEVELS),
                    "objective": model.objective,
                    "feature_names": model.feature_names,
                    "target": target,
                    "kind": "prop",
                    "family": model.family,
                },
                decision=decision,
            )
            model.metrics["train_val_gap"] = record.train_val_gap
            self._last_record = record

        return model

    def promote_if_better(self, target: str, model: PropModel) -> tuple[bool, str]:
        """Promueve el último shadow registrado si mejora a producción (ver ``evaluate_candidate``)."""
        record = getattr(self, "_last_record", None)
        if record is None or record.target != target:
            return False, "no hay modelo recién registrado para este target"
        if record.status != "shadow":
            return False, f"el modelo quedó en estado '{record.status}' (gate de brecha)"
        production = None
        if "prod_val_mae" in model.metrics:
            production = {
                k.removeprefix("prod_"): v for k, v in model.metrics.items() if k.startswith("prod_")
            }
        allowed, reason = evaluate_candidate(model.metrics, production)
        if allowed:
            self.registry.promote(record.model_id, reason=f"props v2: {reason}")
        log.info("Promoción de '%s': %s — %s", target, "SÍ" if allowed else "NO", reason)
        return allowed, reason

    def train_all(self) -> list[PropModel]:
        return [self.train(target) for target in self.settings.props.targets]


def write_backtest_report(results: list[PropBacktestResult], directory: Path) -> tuple[Path, Path]:
    """Escribe ``props_backtest.md`` y ``props_backtest.csv`` (global y por temporada)."""
    directory.mkdir(parents=True, exist_ok=True)
    overall = pd.concat([r.comparison.assign(season="all") for r in results], ignore_index=True)
    seasons = pd.concat([r.by_season for r in results if r.by_season is not None], ignore_index=True)
    table = pd.concat([overall, seasons], ignore_index=True)
    columns = ["target", "variant", "season", "n", "mae", "rmse", "bias", "crps",
               "coverage_80", "coverage_literal", "mean_pred", "mean_true"]
    table = table[columns]
    csv_path = directory / "props_backtest.csv"
    table.to_csv(csv_path, index=False, float_format="%.4f")

    def fmt(frame: pd.DataFrame) -> str:
        shown = frame.copy()
        for column in ("mae", "rmse", "bias", "crps", "mean_pred", "mean_true"):
            shown[column] = shown[column].map(lambda v: f"{v:.3f}")
        for column in ("coverage_80", "coverage_literal"):
            shown[column] = shown[column].map(lambda v: f"{v:.3f}")
        header = "| " + " | ".join(shown.columns) + " |"
        sep = "|" + "|".join("---" for _ in shown.columns) + "|"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
        return "\n".join([header, sep, *body])

    lines = [
        "# Backtest walk-forward de props",
        "",
        "Generado por `PropsAgent.backtest` (`uv run nflp backtest props`). Para cada",
        "temporada S se entrena con temporadas anteriores, se calibra la distribución",
        "en S−1 y se evalúa S completa. Nunca hay split aleatorio.",
        "",
        "* **nuevo** — features v2, entrenamiento desde 2016, objetivo elegido por target,",
        "  hiperparámetros afinados con Optuna, rejilla de cuantiles conformal (continuos)",
        "  o Poisson (conteos).",
        "* **actual** — el modelo v1 de producción (features v1, L1 en continuos y L2 en",
        "  conteos, pseudo-labeling, log-normal global).",
        "* **media_5_partidos** — media de los 5 partidos previos; Poisson(r5) en conteos y",
        "  r5 + cuantiles de residuos de S−1 en continuos.",
        "",
        "Métricas: CRPS de la distribución servida (rejilla p05..p95 en continuos y en el",
        "modelo actual; Poisson exacto en conteos para nuevo y media_5_partidos),",
        "`coverage_80` = cobertura del intervalo p10–p90 (continuos) o PIT aleatorizado",
        "dentro del 80 % central (conteos Poisson); `coverage_literal` = fracción de",
        "observaciones dentro de [lower_80, upper_80]; `bias` = media(predicción − real).",
        "",
        "**Aviso de selección**: el objetivo puntual se eligió con el walk-forward 2020–2023",
        "y los hiperparámetros con pliegues 2021–2023, así que en esas temporadas el",
        "modelo nuevo está en muestra respecto a la selección. Las temporadas limpias son",
        "2024 en adelante (2026 incompleta).",
        "",
        "## Global",
        "",
        fmt(overall[columns]),
        "",
        "## Por temporada",
        "",
    ]
    for target in seasons["target"].unique():
        lines += [f"### {target}", "", fmt(seasons[seasons["target"] == target][columns]
                                            .sort_values(["season", "variant"])), ""]
    md_path = directory / "props_backtest.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path


def prop_feature_names(target: str) -> list[str]:
    return PROP_FEATURE_SETS[target]


__all__ = [
    "COVERAGE_BAND",
    "PropBacktestResult",
    "PropsAgent",
    "distribution_diagnostics",
    "evaluate_candidate",
    "served_quantiles",
    "write_backtest_report",
]
