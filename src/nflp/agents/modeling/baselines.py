"""Baselines obligatorios del backtest (sección 4 del prompt).

Sin baselines, un log-loss de 0,65 no significa nada. Con ellos, significa todo:

* **Naive** — el local siempre gana. Acierta ~57 % histórico. Cualquier modelo que
  no lo supere es ruido.
* **Elo** — sorprendentemente competitivo (~63 % de acierto). Es el listón que
  separa "he aprendido algo" de "he reordenado la tabla de posiciones".
* **Mercado** — la línea de cierre. Es el listón *real*: agrega la información de
  miles de apostadores profesionales. Superarlo de forma consistente es difícil, y
  este proyecto reporta la distancia sin maquillarla.

Los baselines de mercado son el **único** punto del sistema que lee
``market_lines``. Ver `docs/anti-leakage.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nflp.agents.features.elo import EloConfig, elo_win_probability
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Desviación típica histórica del margen de victoria en la NFL (~13,5 puntos).
# Se usa para convertir un spread en probabilidad de victoria.
NFL_MARGIN_STD = 13.5

# Ventaja de campo histórica en puntos (ha bajado desde ~3 hasta ~1,8 en los
# últimos años; se usa un valor intermedio conservador).
HOME_FIELD_POINTS = 2.0


@dataclass
class BaselinePrediction:
    """Predicciones de un baseline para un conjunto de partidos."""

    label: str
    game_id: pd.Series
    home_win_prob: pd.Series | None = None
    predicted_margin: pd.Series | None = None
    predicted_total: pd.Series | None = None


def naive_home_baseline(games: pd.DataFrame) -> BaselinePrediction:
    """El equipo local siempre gana, con la probabilidad base histórica."""
    n = len(games)
    return BaselinePrediction(
        label="naive_local",
        game_id=games["game_id"],
        home_win_prob=pd.Series(np.full(n, 0.57), index=games.index),
        predicted_margin=pd.Series(np.full(n, HOME_FIELD_POINTS), index=games.index),
    )


def elo_baseline(
    features: pd.DataFrame, *, config: EloConfig | None = None
) -> BaselinePrediction:
    """Baseline Elo: probabilidad y margen derivados del rating previo."""
    config = config or EloConfig()
    neutral = features.get("is_neutral_site")

    probability = elo_win_probability(
        features["home_elo_pre"],
        features["away_elo_pre"],
        config=config,
        neutral_site=neutral if neutral is not None else None,
    )
    advantage = np.where(
        neutral.fillna(False) if neutral is not None else False, 0.0, config.home_advantage
    )
    margin = (features["home_elo_pre"] + advantage - features["away_elo_pre"]) / 25.0

    return BaselinePrediction(
        label="elo",
        game_id=features["game_id"],
        home_win_prob=probability,
        predicted_margin=margin,
    )


def market_baseline(
    games: pd.DataFrame, db: Database | None = None
) -> BaselinePrediction:
    """Baseline de mercado: la línea de cierre.

    **Este es el único lugar del sistema que lee ``market_lines``.** Se usa solo
    para comparar, nunca para entrenar.

    `spread_line` en nflverse es positivo cuando el local es favorito, y expresa
    los puntos que se espera que gane el local.
    """
    db = db or get_db()
    lines = db.query(
        "SELECT game_id, spread_line, total_line, home_moneyline, away_moneyline FROM market_lines"
    )
    merged = games[["game_id"]].merge(lines, on="game_id", how="left")

    # El spread se convierte en probabilidad asumiendo que el margen se
    # distribuye normal alrededor de la línea.
    from scipy.stats import norm

    probability = pd.Series(
        norm.cdf(merged["spread_line"].to_numpy(dtype=float) / NFL_MARGIN_STD),
        index=merged.index,
    )
    probability = probability.where(merged["spread_line"].notna())

    return BaselinePrediction(
        label="mercado_cierre",
        game_id=merged["game_id"],
        home_win_prob=probability,
        predicted_margin=merged["spread_line"],
        predicted_total=merged["total_line"],
    )


def moneyline_to_probability(moneyline: pd.Series) -> pd.Series:
    """Convierte una cuota americana en probabilidad implícita (con vig)."""
    values = pd.to_numeric(moneyline, errors="coerce")
    return np.where(
        values < 0,
        -values / (-values + 100.0),
        100.0 / (values + 100.0),
    )


def devig_two_way(prob_a: pd.Series, prob_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Elimina el margen de la casa normalizando dos probabilidades implícitas.

    Sin esto, las probabilidades del mercado suman ~1,05 y el log-loss del
    baseline sale artificialmente malo — comparación injusta a nuestro favor.
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def season_average_total(
    features: pd.DataFrame, games: pd.DataFrame
) -> BaselinePrediction:
    """Baseline de totales: la media de puntos de las temporadas anteriores.

    Es el equivalente del 'naive' para over/under.
    """
    scored = games.dropna(subset=["home_score", "away_score"]).copy()
    scored["game_total"] = scored["home_score"] + scored["away_score"]

    # Media de la temporada anterior — nunca de la actual (evita leakage).
    by_season = scored.groupby("season")["game_total"].mean()
    prior = by_season.shift(1)
    # Antes de la primera temporada con historia, la media histórica de la NFL.
    prior = prior.fillna(44.0)

    predicted = features["season"].map(prior).fillna(44.0)
    return BaselinePrediction(
        label="media_temporada_previa",
        game_id=features["game_id"],
        predicted_total=predicted,
    )
