"""Backtesting Agent — orquesta el walk-forward, los baselines y la segmentación.

Produce la tabla que responde la única pregunta que importa: **¿el modelo aporta
algo que no aporten ya un Elo simple y la línea del mercado?**

Los resultados se reportan sin maquillar. Si el modelo pierde contra el mercado,
la tabla lo dice.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nflp.agents.backtesting.metrics import (
    ats_win_pct,
    calibration_curve,
    classification_metrics,
    regression_metrics,
)
from nflp.agents.backtesting.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)
from nflp.agents.features.elo import EloConfig, elo_win_probability
from nflp.agents.modeling.datasets import load_game_dataset
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestReport:
    """Resultado completo de una corrida de backtest."""

    run_id: str
    target: str
    overall: pd.DataFrame          # una fila por modelo/baseline
    segments: pd.DataFrame         # métricas por segmento
    calibration: pd.DataFrame      # puntos de la curva de calibración
    predictions: pd.DataFrame      # predicciones fuera de muestra
    n_retrains: int


class BacktestingAgent:
    """Validación walk-forward con baselines y segmentación."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)

    # ------------------------------------------------------------------ run

    def run(
        self,
        target: str,
        *,
        start_season: int | None = None,
        end_season: int | None = None,
        family: str | None = None,
        persist: bool = True,
    ) -> BacktestReport:
        """Ejecuta el backtest completo de un objetivo."""
        backtest_config = self.settings.backtest
        start = start_season or backtest_config.start_season
        end = end_season or backtest_config.end_season
        run_id = str(uuid.uuid4())

        dataset = load_game_dataset(
            target,
            db=self.db,
            feature_version=self.settings.features.version,
            min_prior_games=self.settings.features.min_prior_games,
        )

        config = WalkForwardConfig(
            start_season=start,
            end_season=end,
            retrain_every_weeks=backtest_config.retrain_every_weeks,
            min_train_seasons=backtest_config.min_train_seasons,
            family=family or "catboost",
        )
        result = run_walk_forward(dataset, config)

        # --- contexto necesario para los baselines ---
        context = self._load_context(result.predictions["game_id"].tolist())
        frame = result.predictions.merge(context, on="game_id", how="left")

        # --- evaluación del modelo y de cada baseline ---
        rows = self._evaluate_all(frame, target, result)
        overall = pd.DataFrame(rows).sort_values(
            "log_loss" if result.kind == "classification" else "mae"
        )

        segments = self._evaluate_segments(frame, target, result.kind)
        calibration = self._calibration_points(frame, target, result.kind, run_id)

        report = BacktestReport(
            run_id=run_id,
            target=target,
            overall=overall,
            segments=segments,
            calibration=calibration,
            predictions=frame,
            n_retrains=result.n_retrains,
        )

        if persist:
            self._persist(report)

        return report

    # -------------------------------------------------------------- contexto

    def _load_context(self, game_ids: list[str]) -> pd.DataFrame:
        """Carga Elo, líneas de mercado y contexto para evaluar baselines.

        Este método es el **único** punto del pipeline que lee ``market_lines``,
        y solo para comparar. Ver `docs/anti-leakage.md`.
        """
        placeholders = ", ".join(f"'{g}'" for g in game_ids)
        context = self.db.query(
            f"""
            SELECT g.game_id, g.home_score, g.away_score,
                   g.home_score - g.away_score AS actual_margin,
                   g.home_score + g.away_score AS actual_total,
                   f.home_elo_pre, f.away_elo_pre, f.is_neutral_site,
                   f.is_adverse_weather, f.week_phase, f.temperature_c, f.wind_kph,
                   m.spread_line, m.total_line, m.home_moneyline, m.away_moneyline
            FROM games g
            JOIN features_game f ON f.game_id = g.game_id
            LEFT JOIN market_lines m ON m.game_id = g.game_id
            WHERE g.game_id IN ({placeholders})
            """
        )
        return context

    # ------------------------------------------------------------ evaluación

    def _evaluate_all(
        self, frame: pd.DataFrame, target: str, result: WalkForwardResult
    ) -> list[dict]:
        """Métricas del modelo y de todos los baselines obligatorios."""
        rows: list[dict] = []

        if result.kind == "classification":
            # --- modelo ---
            rows.append(self._class_row("modelo", frame["y_true"], frame["y_prob"], frame))

            # --- baseline naive: el local siempre gana ---
            naive = pd.Series(0.57, index=frame.index)
            rows.append(self._class_row("baseline_naive_local", frame["y_true"], naive, frame))

            # --- baseline Elo ---
            elo_probability = elo_win_probability(
                frame["home_elo_pre"], frame["away_elo_pre"],
                config=EloConfig(), neutral_site=frame["is_neutral_site"],
            )
            rows.append(self._class_row("baseline_elo", frame["y_true"], elo_probability, frame))

            # --- baseline mercado: la línea de cierre ---
            market = self._market_win_probability(frame)
            rows.append(self._class_row("baseline_mercado", frame["y_true"], market, frame))

        else:
            actual = frame["y_true"]

            rows.append(self._reg_row("modelo", actual, frame["y_pred"], frame, target))

            if target == "spread":
                elo_margin = (frame["home_elo_pre"] - frame["away_elo_pre"]) / 25.0
                elo_margin = elo_margin + np.where(
                    frame["is_neutral_site"].fillna(False), 0.0, EloConfig().home_advantage / 25.0
                )
                rows.append(self._reg_row("baseline_elo", actual, elo_margin, frame, target))
                rows.append(
                    self._reg_row(
                        "baseline_naive_local", actual,
                        pd.Series(2.0, index=frame.index), frame, target,
                    )
                )
                rows.append(
                    self._reg_row("baseline_mercado", actual, frame["spread_line"], frame, target)
                )
            else:
                # Totales: media histórica como naive, línea del mercado como listón.
                rows.append(
                    self._reg_row(
                        "baseline_media_historica", actual,
                        pd.Series(float(actual.mean()), index=frame.index), frame, target,
                    )
                )
                rows.append(
                    self._reg_row("baseline_mercado", actual, frame["total_line"], frame, target)
                )

        return rows

    def _class_row(
        self, label: str, y_true: pd.Series, y_prob: pd.Series, frame: pd.DataFrame
    ) -> dict:
        metrics = classification_metrics(y_true, y_prob)
        return {
            "modelo": label,
            "n": metrics.n,
            "log_loss": round(metrics.log_loss, 4),
            "brier": round(metrics.brier_score, 4),
            "ece": round(metrics.ece, 4),
            "accuracy": round(metrics.accuracy, 4),
        }

    def _reg_row(
        self, label: str, y_true: pd.Series, y_pred: pd.Series,
        frame: pd.DataFrame, target: str,
    ) -> dict:
        metrics = regression_metrics(y_true, y_pred)
        row = {
            "modelo": label,
            "n": metrics.n,
            "mae": round(metrics.mae, 4),
            "rmse": round(metrics.rmse, 4),
        }
        if target == "spread" and "spread_line" in frame.columns:
            row["ats_%"] = round(
                100 * ats_win_pct(y_true, y_pred, frame["spread_line"]), 2
            )
        return row

    def _market_win_probability(self, frame: pd.DataFrame) -> pd.Series:
        """Probabilidad del mercado, sin vig, a partir de las moneylines.

        Sin quitar el margen de la casa, las dos probabilidades suman ~1,05 y el
        log-loss del mercado sale artificialmente malo — una comparación injusta
        a nuestro favor. Cuando falta la moneyline se recurre al spread.
        """
        from scipy.stats import norm

        from nflp.agents.modeling.baselines import moneyline_to_probability

        home = pd.Series(moneyline_to_probability(frame["home_moneyline"]), index=frame.index)
        away = pd.Series(moneyline_to_probability(frame["away_moneyline"]), index=frame.index)
        total = home + away
        devigged = (home / total).where(total > 0)

        from_spread = pd.Series(
            norm.cdf(frame["spread_line"].to_numpy(dtype=float) / 13.5), index=frame.index
        ).where(frame["spread_line"].notna())

        return devigged.fillna(from_spread)

    # ------------------------------------------------------------- segmentos

    def _evaluate_segments(
        self, frame: pd.DataFrame, target: str, kind: str
    ) -> pd.DataFrame:
        """Métricas desagregadas por temporada, equipo, clima y tramo.

        Un modelo puede tener buena métrica global y fallar sistemáticamente en
        un segmento concreto (partidos con viento, playoffs, un equipo raro).
        Reportar solo el agregado esconde exactamente ese tipo de problema.
        """
        rows: list[dict] = []

        definitions: list[tuple[str, pd.Series]] = [
            ("season", frame["season"].astype(str)),
            ("week_phase", frame["week_phase"].astype(str)),
            (
                "weather",
                np.where(
                    frame["is_adverse_weather"].fillna(False), "adverso",
                    np.where(frame["temperature_c"].isna(), "techo_cerrado", "normal"),
                ),
            ),
            ("home_team", frame["home_team"].astype(str)),
        ]

        for segment_type, values in definitions:
            series = pd.Series(values, index=frame.index)
            for value, group in frame.groupby(series, observed=True):
                if len(group) < 10:
                    continue
                row = {"segment_type": segment_type, "segment_value": str(value), "n_games": len(group)}
                if kind == "classification":
                    metrics = classification_metrics(group["y_true"], group["y_prob"])
                    row.update(
                        {
                            "log_loss": metrics.log_loss, "brier_score": metrics.brier_score,
                            "ece": metrics.ece, "accuracy": metrics.accuracy,
                        }
                    )
                else:
                    metrics = regression_metrics(group["y_true"], group["y_pred"])
                    row.update({"mae": metrics.mae, "rmse": metrics.rmse})
                    if target == "spread" and "spread_line" in group.columns:
                        row["ats_win_pct"] = ats_win_pct(
                            group["y_true"], group["y_pred"], group["spread_line"]
                        )
                rows.append(row)

        return pd.DataFrame(rows)

    def _calibration_points(
        self, frame: pd.DataFrame, target: str, kind: str, run_id: str
    ) -> pd.DataFrame:
        """Curva de calibración del modelo y de los baselines probabilísticos."""
        if kind != "classification":
            return pd.DataFrame()

        pieces: list[pd.DataFrame] = []
        sources = {
            "modelo": frame["y_prob"],
            "baseline_elo": elo_win_probability(
                frame["home_elo_pre"], frame["away_elo_pre"],
                config=EloConfig(), neutral_site=frame["is_neutral_site"],
            ),
            "baseline_mercado": self._market_win_probability(frame),
        }

        for label, probability in sources.items():
            curve = calibration_curve(frame["y_true"], probability).to_frame()
            if curve.empty:
                continue
            curve["run_id"] = run_id
            curve["model_label"] = label
            curve["target"] = target
            pieces.append(curve)

        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    # ----------------------------------------------------------- persistencia

    def _persist(self, report: BacktestReport) -> None:
        """Guarda resultados globales, por segmento y de calibración."""
        now = datetime.now(UTC).replace(tzinfo=None)

        global_rows = [
            {
                "result_id": str(uuid.uuid4()),
                "run_id": report.run_id,
                "model_label": row["modelo"],
                "target": report.target,
                "segment_type": "overall",
                "segment_value": None,
                "n_games": int(row.get("n", 0)),
                "log_loss": row.get("log_loss"),
                "brier_score": row.get("brier"),
                "ece": row.get("ece"),
                "accuracy": row.get("accuracy"),
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
                "crps": None,
                "ats_win_pct": (row.get("ats_%") / 100.0) if row.get("ats_%") is not None else None,
                "beats_market": None,
                "created_at": now,
            }
            for _, row in report.overall.iterrows()
        ]

        segment_rows = [
            {
                "result_id": str(uuid.uuid4()),
                "run_id": report.run_id,
                "model_label": "modelo",
                "target": report.target,
                "segment_type": row["segment_type"],
                "segment_value": row["segment_value"],
                "n_games": int(row["n_games"]),
                "log_loss": row.get("log_loss"),
                "brier_score": row.get("brier_score"),
                "ece": row.get("ece"),
                "accuracy": row.get("accuracy"),
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
                "crps": None,
                "ats_win_pct": row.get("ats_win_pct"),
                "beats_market": None,
                "created_at": now,
            }
            for _, row in report.segments.iterrows()
        ]

        combined = pd.DataFrame(global_rows + segment_rows)
        if not combined.empty:
            self.db.write_df(combined, "backtest_results", mode="append")

        if not report.calibration.empty:
            self.db.write_df(report.calibration, "calibration_bins", mode="append")

        log.info(
            "Backtest %s persistido: %d filas de resultados.",
            report.run_id[:8], len(combined),
        )
