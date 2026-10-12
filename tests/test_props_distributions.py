"""Distribuciones por jugador de los props v2.

* La rejilla de cuantiles es monótona en cada fila, también tras calibrar.
* La calibración conformal lleva la cobertura p10–p90 al ~80 % fuera de muestra.
* El CRPS de la Poisson es exacto (coincide con E|X−y| − ½E|X−X'|).
* Las predicciones de props persisten la rejilla, λ y equipo/rival/localía.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nflp.agents.modeling.distributions import (
    FINE_LEVELS,
    P10_INDEX,
    P90_INDEX,
    QUANTILE_LEVELS,
    PoissonDistribution,
    QuantileGridDistribution,
    crps_fine_rows,
    crps_poisson,
    crps_poisson_rows,
    crps_quantile_grid,
    interval_coverage,
    make_monotone,
    randomized_pit_coverage,
)
from nflp.agents.modeling.props import (
    _SPARSE_COUNT_OBJECTIVE,
    PROP_FEATURE_SETS,
    SPARSE_COUNT_PROPS,
    point_objective_params,
    train_prop_model,
)
from nflp.agents.modeling.props_agent import evaluate_candidate

_FAST = {
    "n_estimators": 80, "learning_rate": 0.1, "num_leaves": 7, "max_depth": 3,
    "min_child_samples": 30, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
    "random_state": 0, "verbose": -1, "n_jobs": 1,
}


def _heteroscedastic(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """y ≥ 0 con media y dispersión que dependen de x (como unas yardas)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.normal(size=n)
    mean = 10 + 60 * x1
    y = np.maximum(rng.gamma(shape=2.0, scale=mean / 2.0) + 2 * x2, 0.0)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


# --------------------------------------------------------------- monotonía


def test_make_monotone_sorts_each_row() -> None:
    grid = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    out = make_monotone(grid)
    assert (np.diff(out, axis=1) >= 0).all()
    assert sorted(grid[0]) == list(out[0])


def test_quantile_grid_is_monotone_and_has_19_levels() -> None:
    X, y = _heteroscedastic(3000, seed=1)
    Xc, yc = _heteroscedastic(1500, seed=2)
    dist = QuantileGridDistribution(feature_names=["x1", "x2"]).fit(X, y, _FAST)
    dist.calibrate(Xc, yc)
    grid = dist.quantiles(None, Xc)

    assert len(QUANTILE_LEVELS) == 19
    assert QUANTILE_LEVELS[0] == 0.05 and QUANTILE_LEVELS[-1] == 0.95
    assert grid.shape == (len(Xc), 19)
    assert (np.diff(grid, axis=1) >= -1e-9).all()
    assert (grid >= 0).all()  # suelo en 0 por defecto
    lower, upper = dist.interval(None, Xc)
    np.testing.assert_allclose(lower, grid[:, P10_INDEX])
    np.testing.assert_allclose(upper, grid[:, P90_INDEX])


def test_quantile_grid_width_depends_on_player() -> None:
    """La dispersión ya no es global: más media → intervalo más ancho."""
    X, y = _heteroscedastic(4000, seed=3)
    Xc, yc = _heteroscedastic(1500, seed=4)
    dist = QuantileGridDistribution(feature_names=["x1", "x2"]).fit(X, y, _FAST).calibrate(Xc, yc)
    probe = pd.DataFrame({"x1": [0.05, 0.95], "x2": [0.0, 0.0]})
    lower, upper = dist.interval(None, probe)
    widths = upper - lower
    assert widths[1] > 2 * widths[0]


# ------------------------------------------------------- calibración conformal


def test_conformal_calibration_reaches_80_percent_out_of_sample() -> None:
    X, y = _heteroscedastic(4000, seed=5)
    Xc, yc = _heteroscedastic(3000, seed=6)
    Xt, yt = _heteroscedastic(6000, seed=7)

    # Rejilla mal calibrada a propósito: muy pocos árboles → cuantiles
    # encogidos hacia el valor inicial.
    poor = {**_FAST, "n_estimators": 5, "learning_rate": 0.05}
    dist = QuantileGridDistribution(feature_names=["x1", "x2"]).fit(X, y, poor)
    raw = dist.raw_quantiles(Xt)
    raw_coverage = interval_coverage(yt, raw[:, P10_INDEX], raw[:, P90_INDEX])

    dist.calibrate(Xc, yc)
    assert dist.calibration_coverage == pytest.approx(0.80, abs=0.01)
    lower, upper = dist.interval(None, Xt)
    coverage = interval_coverage(yt, lower, upper)
    assert 0.77 <= coverage <= 0.83
    assert abs(coverage - 0.80) < abs(raw_coverage - 0.80)


def test_quantile_crps_beats_constant_grid() -> None:
    X, y = _heteroscedastic(4000, seed=8)
    Xc, yc = _heteroscedastic(2000, seed=9)
    Xt, yt = _heteroscedastic(3000, seed=10)
    dist = QuantileGridDistribution(feature_names=["x1", "x2"]).fit(X, y, _FAST).calibrate(Xc, yc)
    model_crps = crps_quantile_grid(yt, dist.quantiles(None, Xt))
    constant = np.tile(np.quantile(y, QUANTILE_LEVELS), (len(yt), 1))
    assert model_crps < crps_quantile_grid(yt, constant)


# ---------------------------------------------------------------- Poisson


def _crps_energy_form(y: float, lam: float, k_max: int = 80) -> float:
    """CRPS = E|X − y| − ½ E|X − X'|, por suma directa sobre el soporte."""
    from scipy.stats import poisson

    ks = np.arange(k_max)
    p = poisson.pmf(ks, lam)
    first = float(np.sum(p * np.abs(ks - y)))
    second = float(np.sum(p[:, None] * p[None, :] * np.abs(ks[:, None] - ks[None, :])))
    return first - 0.5 * second


@pytest.mark.parametrize(("y", "lam"), [(0, 0.2), (1, 0.2), (0, 2.3), (3, 2.3), (7, 1.0)])
def test_poisson_crps_is_exact(y: int, lam: float) -> None:
    exact = crps_poisson_rows(np.array([y], dtype=float), np.array([lam]))[0]
    assert exact == pytest.approx(_crps_energy_form(y, lam), abs=1e-8)


def test_poisson_crps_matches_fine_quantile_grid() -> None:
    from scipy.stats import poisson

    rng = np.random.default_rng(0)
    lam = rng.uniform(0.1, 3.0, 500)
    y = rng.poisson(lam).astype(float)
    exact = crps_poisson(y, lam)
    fine = crps_fine_rows(y, lambda rows: poisson.ppf(FINE_LEVELS[None, :], lam[rows][:, None])).mean()
    assert fine == pytest.approx(exact, rel=0.02)


def test_poisson_crps_rewards_the_true_rate() -> None:
    rng = np.random.default_rng(1)
    lam = rng.uniform(0.1, 3.0, 5000)
    y = rng.poisson(lam).astype(float)
    assert crps_poisson(y, lam) < crps_poisson(y, lam * 1.5)
    assert crps_poisson(y, lam) < crps_poisson(y, np.full_like(lam, lam.mean()))


def test_randomized_pit_coverage_is_80_percent_for_true_poisson() -> None:
    rng = np.random.default_rng(2)
    lam = rng.uniform(0.05, 3.0, 20000)
    y = rng.poisson(lam).astype(float)
    assert randomized_pit_coverage(y, lam) == pytest.approx(0.80, abs=0.015)
    # La cobertura literal del intervalo entero queda muy por encima.
    lower, upper = PoissonDistribution().interval(lam)
    assert interval_coverage(y, lower, upper) > 0.85


def test_poisson_distribution_interval_and_quantiles() -> None:
    dist = PoissonDistribution()
    lam = np.array([0.3, 2.3])
    grid = dist.quantiles(lam)
    assert grid.shape == (2, 19)
    assert (np.diff(grid, axis=1) >= 0).all()
    lower, upper = dist.interval(lam)
    assert list(lower) == [0.0, 0.0]
    assert list(upper) == [1.0, 4.0]


# ------------------------------------------------------ objetivos y promoción


def test_sparse_counts_use_poisson_mean_objective() -> None:
    """Los conteos dispersos no pueden usar L1 (colapsaría a 0)."""
    assert _SPARSE_COUNT_OBJECTIVE["lightgbm"]["objective"] == "poisson"
    rng = np.random.default_rng(3)
    n = 4000
    frame = pd.DataFrame({f: rng.normal(size=n) for f in PROP_FEATURE_SETS["receiving_tds"]})
    frame["position"] = "WR"
    frame["receiving_tds"] = rng.poisson(np.exp(-1.6 + 0.6 * frame["target_share_r5"])).astype(float)
    model = train_prop_model(frame.iloc[:3000], frame.iloc[3000:], "receiving_tds", params=_FAST)
    prediction = model.predict(frame.iloc[3000:])
    assert model.objective == "poisson"
    assert model.distribution_kind == "poisson"
    assert prediction.mean() == pytest.approx(frame["receiving_tds"].mean(), rel=0.25)
    assert prediction.std() > 0.02
    assert "receiving_tds" in SPARSE_COUNT_PROPS


def test_huber_delta_scales_with_target() -> None:
    small = point_objective_params("huber", np.array([0, 1, 2, 3, 4], dtype=float))
    large = point_objective_params("huber", np.array([0, 10, 20, 30, 40], dtype=float))
    assert large["alpha"] == pytest.approx(10 * small["alpha"])


def test_promotion_requires_mae_crps_and_coverage() -> None:
    production = {"val_mae": 10.0, "val_crps": 8.0}
    good = {"val_mae": 9.5, "val_crps": 7.5, "val_coverage_80": 0.80}
    assert evaluate_candidate(good, production)[0]
    assert not evaluate_candidate({**good, "val_crps": 8.1}, production)[0]
    assert not evaluate_candidate({**good, "val_mae": 10.1}, production)[0]
    assert not evaluate_candidate({**good, "val_coverage_80": 0.86}, production)[0]
    assert not evaluate_candidate({**good, "val_coverage_80": 0.74}, production)[0]


# ------------------------------------------------------------- persistencia


def _seed_upcoming(db, target: str, n_players: int = 6) -> None:  # type: ignore[no-untyped-def]
    kickoff = datetime.now() + timedelta(days=3)
    db.write_df(
        pd.DataFrame([{
            "game_id": "2099_01_AAA_BBB", "season": 2099, "week": 1,
            "season_type": "REG", "kickoff_utc": kickoff,
            "home_team": "BBB", "away_team": "AAA",
        }]),
        "games",
    )
    rng = np.random.default_rng(4)
    rows = []
    for i in range(n_players):
        team, opponent, home = ("AAA", "BBB", False) if i % 2 == 0 else ("BBB", "AAA", True)
        row = {
            "game_id": "2099_01_AAA_BBB", "gsis_id": f"00-000000{i}",
            "feature_set_version": db.settings.features.version if hasattr(db, "settings") else "v2",
            "season": 2099, "week": 1, "season_type": "REG",
            "team": team, "opponent_team": opponent, "is_home": home,
            "position": "WR", "as_of_ts": datetime.now(),
            "snap_pct_r5": 0.8, "games_played_prior": 10,
        }
        for feature in PROP_FEATURE_SETS[target]:
            row.setdefault(feature, float(rng.normal()))
        rows.append(row)
    db.write_df(pd.DataFrame(rows), "features_player")


def _register_production(settings, db, model, target: str) -> str:  # type: ignore[no-untyped-def]
    import uuid

    from nflp.registry import ModelRecord, ModelRegistry

    registry = ModelRegistry(settings, db)
    record = ModelRecord(
        model_id=str(uuid.uuid4()), target=target, family="lightgbm", version=1,
        feature_set_version=settings.features.version, feature_names=model.feature_names,
        hyperparameters={}, dataset_hash="", train_seasons="2000-2001",
        n_train_rows=1, n_val_rows=1, metric_name="mae", train_metric=1.0, val_metric=1.0,
    )
    registry._log_to_mlflow = lambda *a, **k: None  # type: ignore[method-assign]
    registry.register(record, {
        "model": model.model, "distribution": model.distribution,
        "feature_names": model.feature_names, "target": target, "kind": "prop",
        "family": "lightgbm",
    })
    assert registry.promote(record.model_id)
    return record.model_id


@pytest.mark.parametrize("target", ["receiving_tds", "receiving_yards"])
def test_predictions_persist_quantiles_lambda_and_team(settings, db, target: str) -> None:  # type: ignore[no-untyped-def]
    from nflp.agents.modeling.predict import Predictor

    rng = np.random.default_rng(5)
    n = 2500
    features = PROP_FEATURE_SETS[target]
    frame = pd.DataFrame({f: rng.normal(size=n) for f in features})
    frame["position"] = "WR"
    if target in SPARSE_COUNT_PROPS:
        frame[target] = rng.poisson(0.4, n).astype(float)
    else:
        frame[target] = np.maximum(rng.gamma(2.0, 15.0, n) + 5 * frame[features[0]], 0.0)
    model = train_prop_model(frame.iloc[:1800], frame.iloc[1800:], target, params=_FAST)
    model_id = _register_production(settings, db, model, target)

    _seed_upcoming(db, target)
    db.execute(
        f"UPDATE features_player SET feature_set_version = '{settings.features.version}'"
    )
    batch = Predictor(settings, db).predict_props(target, season=2099, week=1)
    assert batch is not None and batch.n_predictions == 6

    stored = db.query(
        f"SELECT * FROM predictions WHERE target = '{target}' ORDER BY gsis_id"
    )
    assert len(stored) == 6
    assert (stored["model_id"] == model_id).all()
    assert stored["as_of_ts"].notna().all() and stored["predicted_at"].notna().all()
    assert stored["probability"].isna().all()  # nulo por diseño en props
    assert set(stored["team"]) == {"AAA", "BBB"}
    assert set(stored["opponent_team"]) == {"AAA", "BBB"}
    assert (stored["team"] != stored["opponent_team"]).all()
    assert stored["is_home"].map(bool).tolist() == [False, True, False, True, False, True]

    for _, row in stored.iterrows():
        grid = json.loads(row["quantiles"])
        assert len(grid) == 19
        assert all(b >= a for a, b in zip(grid, grid[1:], strict=False))
        assert row["lower_80"] == pytest.approx(grid[1], abs=0.01)
        assert row["upper_80"] == pytest.approx(grid[-2], abs=0.01)

    assert stored["nb_dispersion"].isna().all()  # Poisson y rejilla: sin r
    assert stored["mean_estimate"].notna().all()
    if target in SPARSE_COUNT_PROPS:
        assert stored["poisson_lambda"].notna().all()
        np.testing.assert_allclose(stored["poisson_lambda"], stored["point_estimate"])
        np.testing.assert_allclose(stored["mean_estimate"], stored["point_estimate"])
    else:
        assert stored["poisson_lambda"].isna().all()
        from nflp.agents.modeling.distributions import grid_mean

        grids = np.array([json.loads(q) for q in stored["quantiles"]])
        # La media se calcula con la rejilla sin redondear; tolerancia del redondeo a 2 decimales.
        np.testing.assert_allclose(stored["mean_estimate"], grid_mean(grids, 0.0), atol=0.02)
