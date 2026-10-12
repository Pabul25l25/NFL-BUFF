"""Publish de props: contexto del partido, distribuciones y solo el modelo de producción."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from nflp.core.db import Database
from nflp.sync.publish import (
    _QUERY_PARLAY_CALIBRATION,
    _QUERY_PARLAY_CORR,
    _QUERY_PARLAY_LEGS,
    _QUERY_PREDICTIONS_PROPS,
)

GAME = "2099_01_AAA_BBB"


def _registry(db: Database, model_id: str, target: str, status: str, version: int) -> None:
    db.write_df(
        pd.DataFrame([{
            "model_id": model_id, "target": target, "family": "lightgbm",
            "version": version, "status": status, "created_at": datetime.now(),
        }]),
        "model_registry",
    )


def _pred(pid: str, model: str, target: str, **extra: object) -> dict:
    return {
        "prediction_id": pid, "game_id": GAME, "model_id": model, "target": target,
        "gsis_id": "00-0000001", "point_estimate": 55.0, "is_shadow": False,
        "predicted_at": datetime.now(), **extra,
    }


def _seed(db: Database) -> None:
    # Columnas de la migración 011; IF NOT EXISTS para no depender de su orden.
    db.execute_script(
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS mean_estimate DOUBLE;"
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS nb_dispersion DOUBLE;"
    )
    kickoff = datetime.now() + timedelta(days=3)
    db.write_df(
        pd.DataFrame([{
            "game_id": GAME, "season": 2099, "week": 1,
            "season_type": "REG", "kickoff_utc": kickoff,
            "home_team": "BBB", "away_team": "AAA",
        }]),
        "games",
    )
    _registry(db, "m1", "receiving_yards", "production", 2)
    _registry(db, "m0", "receiving_yards", "archived", 1)
    _registry(db, "m2", "receiving_tds", "production", 1)
    q = [float(i) for i in range(19)]
    db.write_df(
        pd.DataFrame([
            _pred("p1", "m1", "receiving_yards", quantiles=str(q)),
            _pred("p_old", "m0", "receiving_yards"),  # modelo viejo: no debe salir
            _pred("p_td", "m2", "receiving_tds", point_estimate=0.4),  # sin lambda -> media
        ]),
        "predictions",
    )
    rows = []
    for version in ("v1", "v2"):  # duplicado por versión: debe salir una sola fila
        rows.append({
            "game_id": GAME, "gsis_id": "00-0000001",
            "feature_set_version": version, "season": 2099, "week": 1,
            "season_type": "REG", "team": "AAA", "opponent_team": "BBB",
            "is_home": False, "as_of_ts": datetime.now(),
        })
    db.write_df(pd.DataFrame(rows), "features_player")


def test_props_upcoming_game_has_team_opponent_and_home_flag(db: Database) -> None:
    _seed(db)
    frame = db.query(_QUERY_PREDICTIONS_PROPS)

    row = frame[frame["target"] == "receiving_yards"].iloc[0]
    assert row["team"] == "AAA"
    assert row["opponent_team"] == "BBB"
    assert row["is_home"] == False  # noqa: E712
    assert pd.notna(row["is_home"])


def test_props_only_production_model_and_distribution_columns(db: Database) -> None:
    _seed(db)
    frame = db.query(_QUERY_PREDICTIONS_PROPS)

    assert set(frame["model_id"]) == {"m1", "m2"}
    assert len(frame) == 2
    cont = frame[frame["target"] == "receiving_yards"].iloc[0]
    assert cont["quantiles"].startswith("[0.0")
    assert pd.isna(cont["poisson_lambda"])  # continuo: sin lambda
    td = frame[frame["target"] == "receiving_tds"].iloc[0]
    assert td["poisson_lambda"] == 0.4  # COALESCE(poisson_lambda, point_estimate)


def test_props_mean_estimate_and_dispersion(db: Database) -> None:
    _seed(db)
    db.execute("UPDATE predictions SET mean_estimate = 61.5 WHERE prediction_id = 'p1'")
    db.execute("UPDATE predictions SET nb_dispersion = 3.2 WHERE prediction_id = 'p_td'")
    frame = db.query(_QUERY_PREDICTIONS_PROPS)
    cont = frame[frame["target"] == "receiving_yards"].iloc[0]
    assert cont["mean_estimate"] == 61.5
    assert pd.isna(cont["nb_dispersion"])
    td = frame[frame["target"] == "receiving_tds"].iloc[0]
    assert td["mean_estimate"] == 0.4  # fallback a point_estimate
    assert td["nb_dispersion"] == 3.2


def test_props_predictions_team_wins_over_features_fallback(db: Database) -> None:
    _seed(db)
    db.execute("UPDATE predictions SET team = 'ZZZ', opponent_team = 'YYY', is_home = TRUE "
               "WHERE prediction_id = 'p1'")
    row = db.query(_QUERY_PREDICTIONS_PROPS).query("target == 'receiving_yards'").iloc[0]
    assert (row["team"], row["opponent_team"], bool(row["is_home"])) == ("ZZZ", "YYY", True)


def test_parlay_queries_pick_latest_per_game(db: Database) -> None:
    t0, t1 = datetime(2099, 1, 1), datetime(2099, 1, 2)
    # Mismo esquema que la migración 009; IF NOT EXISTS para no depender de su orden.
    db.execute_script("""
        CREATE TABLE IF NOT EXISTS parlay_game_corr (
            game_id VARCHAR, model_id VARCHAR, legs VARCHAR, matrix VARCHAR, as_of_ts TIMESTAMP);
        CREATE TABLE IF NOT EXISTS parlay_game_legs (
            game_id VARCHAR, model_id VARCHAR, p_home_win DOUBLE, margin_mu DOUBLE,
            margin_sigma DOUBLE, total_mu DOUBLE, total_sigma DOUBLE, as_of_ts TIMESTAMP);
        CREATE TABLE IF NOT EXISTS parlay_calibration (
            model_id VARCHAR, ece DOUBLE, logloss DOUBLE, ece_independent DOUBLE,
            logloss_independent DOUBLE, n INTEGER, created_at TIMESTAMP);
    """)
    db.write_df(pd.DataFrame([
        {"game_id": GAME, "model_id": "old", "legs": "[]", "matrix": "[]", "as_of_ts": t0},
        {"game_id": GAME, "model_id": "new", "legs": '["a"]', "matrix": "[[1.0]]", "as_of_ts": t1},
    ]), "parlay_game_corr")
    db.write_df(pd.DataFrame([
        {"game_id": GAME, "model_id": m, "p_home_win": p, "margin_mu": 1.0, "margin_sigma": 13.0,
         "total_mu": 44.0, "total_sigma": 10.0, "as_of_ts": t}
        for m, p, t in (("old", 0.4, t0), ("new", 0.6, t1))
    ]), "parlay_game_legs")
    db.write_df(pd.DataFrame([
        {"model_id": m, "ece": e, "logloss": 0.6, "ece_independent": 0.1,
         "logloss_independent": 0.7, "n": 100, "created_at": t}
        for m, e, t in (("old", 0.05, t0), ("new", 0.02, t1))
    ]), "parlay_calibration")

    assert db.query(_QUERY_PARLAY_CORR)["model_id"].tolist() == ["new"]
    assert db.query(_QUERY_PARLAY_LEGS)["p_home_win"].tolist() == [0.6]
    assert db.query(_QUERY_PARLAY_CALIBRATION)["model_id"].tolist() == ["new"]
