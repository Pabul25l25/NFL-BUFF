"""Binomial negativa para conteos sobredispersos y media de la distribución servida.

* El CRPS de la NB es exacto (coincide con la integral numérica de (F − 1{x ≥ y})²).
* Los cuantiles de la NB son monótonos y el intervalo coincide con p10/p90.
* El tamaño r se ajusta solo con la temporada de calibración.
* ``mean_estimate``: media exacta de rejillas conocidas, λ en Poisson, μ en NB.
* Las predicciones persisten ``mean_estimate`` y ``nb_dispersion``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm, nbinom, norm

from nflp.agents.modeling.distributions import (
    P10_INDEX,
    P90_INDEX,
    QUANTILE_LEVELS,
    NegativeBinomialDistribution,
    PoissonDistribution,
    crps_count_rows,
    crps_fine_rows,
    grid_mean,
    randomized_pit_coverage,
)
from nflp.agents.modeling.props import (
    COUNT_DISTRIBUTION,
    PROP_FEATURE_SETS,
    served_crps,
    served_dispersion,
    served_mean,
    train_prop_model,
)

_LEVELS = np.asarray(QUANTILE_LEVELS)
_FAST = {
    "n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "max_depth": 3,
    "min_child_samples": 30, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
    "random_state": 0, "verbose": -1, "n_jobs": 1,
}


# ------------------------------------------------------------------ CRPS NB


@pytest.mark.parametrize(
    ("y", "mu", "r"), [(0, 0.3, 2.0), (2, 2.1, 5.5), (7, 2.5, 1.2), (1, 4.0, 50.0), (12, 6.0, 0.8)]
)
def test_negbin_crps_matches_numerical_integration(y: int, mu: float, r: float) -> None:
    """CRPS = ∫ (F(x) − 1{x ≥ y})² dx, con F escalonada: integral numérica fina."""
    exact = crps_count_rows(np.array([float(y)]), np.array([mu]), np.array([r]))[0]
    upper = nbinom.ppf(1 - 1e-12, r, r / (r + mu)) + y + 5
    xs = np.linspace(0.0, upper, int(upper * 20000) + 1)
    cdf = nbinom.cdf(np.floor(xs), r, r / (r + mu))
    numeric = np.trapezoid((cdf - (xs >= y)) ** 2, xs)
    assert exact == pytest.approx(numeric, abs=2e-4)


def test_negbin_crps_matches_energy_form_and_fine_grid() -> None:
    rng = np.random.default_rng(1)
    mu, r = 2.2, 4.0
    y = rng.negative_binomial(r, r / (r + mu), size=300).astype(float)
    exact = crps_count_rows(y, np.full(y.size, mu), np.full(y.size, r))
    # E|X − y| − ½ E|X − X'| por Monte Carlo grande.
    draws = rng.negative_binomial(r, r / (r + mu), size=400_000).astype(float)
    half_spread = 0.5 * np.mean(np.abs(draws[:200_000] - draws[200_000:]))
    energy = np.array([np.mean(np.abs(draws - v)) for v in y[:20]]) - half_spread
    np.testing.assert_allclose(exact[:20], energy, atol=0.01)
    # Rejilla fina de 200 niveles (la que se usa para la log-normal heredada).
    fine = crps_fine_rows(y, lambda rows: nbinom.ppf(
        ((np.arange(200) + 0.5) / 200)[None, :], r, r / (r + mu)
    ).repeat(y[rows].size, axis=0))
    assert fine.mean() == pytest.approx(exact.mean(), rel=0.02)


def test_negbin_with_huge_size_is_poisson() -> None:
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.1, 3.0, 500)
    y = rng.poisson(mu).astype(float)
    nb = crps_count_rows(y, mu, np.full(mu.size, 1e7))
    pois = crps_count_rows(y, mu)
    np.testing.assert_allclose(nb, pois, atol=1e-5)
    # Tamaño NaN/inf → Poisson exacta.
    np.testing.assert_allclose(crps_count_rows(y, mu, np.full(mu.size, np.inf)), pois)


# ------------------------------------------------------- cuantiles NB


def test_negbin_quantiles_monotone_and_interval_matches_grid() -> None:
    dist = NegativeBinomialDistribution(log_size_intercept=np.log(3.0), log_size_slope=0.4)
    point = np.array([0.05, 0.4, 1.0, 2.3, 3.7, 8.0])
    grid = dist.quantiles(point)
    assert grid.shape == (6, 19)
    assert np.all(np.diff(grid, axis=1) >= 0)
    # Más media → cuantiles que no bajan.
    assert np.all(np.diff(grid, axis=0) >= 0)
    lower, upper = dist.interval(point)
    np.testing.assert_array_equal(lower, grid[:, P10_INDEX])
    np.testing.assert_array_equal(upper, grid[:, P90_INDEX])
    # El tamaño depende de la media: log r = a + b·log μ.
    np.testing.assert_allclose(dist.size(point), np.clip(3.0 * point**0.4, 0.05, 1e4))
    # Sobredispersión: más ancha que la Poisson con la misma media.
    assert np.all(dist.std(point) > PoissonDistribution().std(point))


def test_negbin_fit_recovers_size_and_calibrates_coverage() -> None:
    rng = np.random.default_rng(3)
    mu = rng.uniform(1.0, 4.0, 6000)
    y = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(float)
    dist = NegativeBinomialDistribution().fit(y, mu)
    assert dist.size(np.array([2.0]))[0] == pytest.approx(5.0, rel=0.25)
    assert dist.calibration["aic_nb_const"] < dist.calibration["aic_poisson"]
    # Fuera de muestra (otra extracción): PIT aleatorizado ≈ 80 %.
    y_new = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(float)
    assert dist.coverage(y_new, mu) == pytest.approx(0.80, abs=0.02)
    assert randomized_pit_coverage(y_new, mu) < 0.76  # la Poisson se queda corta


def test_negbin_fit_on_poisson_data_degenerates_to_poisson() -> None:
    rng = np.random.default_rng(4)
    mu = rng.uniform(0.1, 2.0, 5000)
    y = rng.poisson(mu).astype(float)
    dist = NegativeBinomialDistribution().fit(y, mu)
    # Sin sobredispersión el AIC no mejora apreciablemente y la NB ajustada es
    # prácticamente la Poisson (desviación típica a < 5 % de √μ).
    assert dist.calibration["aic_nb_const"] > dist.calibration["aic_poisson"] - 4.0
    point = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(dist.std(point), np.sqrt(point), rtol=0.05)


def test_dispersion_is_fitted_only_on_calibration_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """El r de sacks sale de la temporada de calibración, nunca del entrenamiento."""
    assert COUNT_DISTRIBUTION["sacks_suffered"] == "negbin"
    rng = np.random.default_rng(5)
    features = PROP_FEATURE_SETS["sacks_suffered"]

    def season_frame(n: int, season: int, size: float | None) -> pd.DataFrame:
        frame = pd.DataFrame({f: rng.normal(size=n) for f in features})
        frame["season"] = season
        frame["position"] = "QB"
        mu = np.exp(0.7 + 0.3 * frame[features[0]].to_numpy())
        if size is None:
            frame["sacks_suffered"] = rng.poisson(mu).astype(float)
        else:
            frame["sacks_suffered"] = rng.negative_binomial(size, size / (size + mu)).astype(float)
        return frame

    # Entrenamiento muy sobredisperso (r = 0,7); calibración casi Poisson (r = 40).
    train = season_frame(3000, 2022, 0.7)
    calib = season_frame(3000, 2023, 40.0)

    seen: list[np.ndarray] = []
    original = NegativeBinomialDistribution.fit

    def spy(self: NegativeBinomialDistribution, y: np.ndarray, point: np.ndarray) -> NegativeBinomialDistribution:
        seen.append(np.asarray(y, dtype=float).copy())
        return original(self, y, point)

    monkeypatch.setattr(NegativeBinomialDistribution, "fit", spy)
    model = train_prop_model(train, calib, "sacks_suffered", params=_FAST)

    assert model.distribution_kind == "negbin"
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], calib["sacks_suffered"].to_numpy())
    # r refleja la calibración (casi Poisson), no el entrenamiento (r = 0,7).
    assert model.distribution.size(np.array([2.0]))[0] > 10.0


# ------------------------------------------------------- mean_estimate


def test_grid_mean_of_normal_grid_is_exact() -> None:
    grid = (40.0 + 12.0 * norm.ppf(_LEVELS))[None, :]
    assert grid_mean(grid, None)[0] == pytest.approx(40.0, abs=1e-9)
    # Con suelo por debajo de p05 la cola se trunca en el suelo: E[max(X, 25)].
    floor = 15.0
    z = (floor - 40.0) / 12.0
    expected = floor * norm.cdf(z) + 40.0 * (1 - norm.cdf(z)) + 12.0 * norm.pdf(z)
    clipped = np.maximum(grid, floor)
    assert grid_mean(clipped, floor)[0] == pytest.approx(expected, abs=1e-6)


def test_grid_mean_of_skewed_grid_is_close() -> None:
    """Log-normal (cola derecha, como unas yardas): error pequeño por la cola lineal en z."""
    dist = lognorm(s=0.6, scale=45.0)
    grid = dist.ppf(_LEVELS)[None, :]
    assert grid_mean(grid, 0.0)[0] == pytest.approx(dist.mean(), rel=0.02)


def test_grid_mean_all_at_floor_is_floor() -> None:
    assert grid_mean(np.zeros((2, 19)), 0.0) == pytest.approx([0.0, 0.0], abs=1e-6)


def test_served_mean_and_dispersion_for_counts() -> None:
    point = np.array([0.2, 1.5, 3.0])
    X = pd.DataFrame(index=range(3))
    np.testing.assert_allclose(served_mean(PoissonDistribution(), point, X), point)
    assert np.isnan(served_dispersion(PoissonDistribution(), point)).all()
    nb = NegativeBinomialDistribution(log_size_intercept=np.log(6.0))
    np.testing.assert_allclose(served_mean(nb, point, X), point)
    np.testing.assert_allclose(served_dispersion(nb, point), 6.0)
    y = np.array([0.0, 2.0, 5.0])
    assert served_crps(nb, y, point, X) == pytest.approx(
        crps_count_rows(y, point, np.full(3, 6.0)).mean()
    )


# ------------------------------------------------------- mediana puntual


def test_point_median_shift_is_fitted_on_calibration_and_keeps_grid() -> None:
    """Un punto sesgado se centra en calibración: P(y < punto) ≈ 0,5; la rejilla no cambia."""
    from nflp.agents.modeling.distributions import P50_INDEX, QuantileGridDistribution

    rng = np.random.default_rng(6)
    n = 4000
    X = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.normal(size=n)})
    y = np.maximum(rng.gamma(2.0, (10 + 50 * X["x1"].to_numpy()) / 2.0), 0.0)
    train, calib, test = slice(0, 2000), slice(2000, 3000), slice(3000, n)
    dist = QuantileGridDistribution(feature_names=["x1", "x2"]).fit(X.iloc[train], y[train], _FAST)
    dist.calibrate(X.iloc[calib], y[calib])
    grid_before = dist.quantiles(None, X.iloc[test])

    # Punto deliberadamente bajo (como una mediana L1 que se quedó corta).
    biased = np.maximum(np.median(y[train]) * (0.4 + X["x1"].to_numpy()) - 6.0, 0.0)
    assert np.mean(y[calib] < biased[calib]) < 0.4
    dist.calibrate_point(y[calib], biased[calib])

    assert dist.point_shift > 0
    shifted_calib = dist.shift_point(biased[calib])
    assert np.mean(y[calib] < shifted_calib) == pytest.approx(0.5, abs=0.02)
    assert dist.point_calibration["frac_below_after"] == pytest.approx(0.5, abs=0.02)
    # Fuera de muestra también se centra (misma distribución).
    assert np.mean(y[test] < dist.shift_point(biased[test])) == pytest.approx(0.5, abs=0.07)
    # La rejilla (y su p50) no cambia y sigue monótona.
    grid_after = dist.quantiles(None, X.iloc[test])
    np.testing.assert_array_equal(grid_before, grid_after)
    assert np.all(np.diff(grid_after, axis=1) >= 0)
    assert np.mean(y[calib] < dist.quantiles(None, X.iloc[calib])[:, P50_INDEX]) == pytest.approx(0.5, abs=0.02)


def test_old_grid_artifacts_without_shift_still_serve() -> None:
    """Artefactos anteriores (sin ``point_shift``) sirven el punto sin desplazar."""
    from nflp.agents.modeling.distributions import QuantileGridDistribution

    dist = QuantileGridDistribution(feature_names=["x"])
    del dist.point_shift  # como un pickle anterior al campo
    np.testing.assert_allclose(dist.shift_point(np.array([-1.0, 3.0])), [0.0, 3.0])


# ------------------------------------------------------- persistencia


def test_predictions_persist_mean_estimate_and_nb_dispersion(settings, db) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime, timedelta

    from nflp.agents.modeling.predict import Predictor
    from nflp.registry import ModelRecord, ModelRegistry

    target = "sacks_suffered"
    rng = np.random.default_rng(7)
    features = PROP_FEATURE_SETS[target]
    frame = pd.DataFrame({f: rng.normal(size=2400) for f in features})
    frame["position"] = "QB"
    mu = np.exp(0.7 + 0.3 * frame[features[0]].to_numpy())
    frame[target] = rng.negative_binomial(4.0, 4.0 / (4.0 + mu)).astype(float)
    model = train_prop_model(frame.iloc[:1600], frame.iloc[1600:], target, params=_FAST)
    assert model.distribution_kind == "negbin"

    registry = ModelRegistry(settings, db)
    registry._log_to_mlflow = lambda *a, **k: None  # type: ignore[method-assign]
    record = ModelRecord(
        model_id="nb-model", target=target, family="lightgbm", version=1,
        feature_set_version=settings.features.version, feature_names=model.feature_names,
        hyperparameters={}, dataset_hash="", train_seasons="2000-2001",
        n_train_rows=1, n_val_rows=1, metric_name="mae", train_metric=1.0, val_metric=1.0,
    )
    registry.register(record, {
        "model": model.model, "distribution": model.distribution,
        "feature_names": model.feature_names, "target": target, "kind": "prop", "family": "lightgbm",
    })
    assert registry.promote(record.model_id)

    db.write_df(pd.DataFrame([{
        "game_id": "2099_01_AAA_BBB", "season": 2099, "week": 1, "season_type": "REG",
        "kickoff_utc": datetime.now() + timedelta(days=3), "home_team": "BBB", "away_team": "AAA",
    }]), "games")
    rows = []
    for i in range(4):
        row = {
            "game_id": "2099_01_AAA_BBB", "gsis_id": f"00-000001{i}",
            "feature_set_version": settings.features.version, "season": 2099, "week": 1,
            "season_type": "REG", "team": "AAA" if i % 2 == 0 else "BBB",
            "opponent_team": "BBB" if i % 2 == 0 else "AAA", "is_home": i % 2 == 1,
            "position": "QB", "as_of_ts": datetime.now(), "snap_pct_r5": 0.9, "games_played_prior": 10,
        }
        for feature in features:
            row.setdefault(feature, float(rng.normal()))
        rows.append(row)
    db.write_df(pd.DataFrame(rows), "features_player")

    batch = Predictor(settings, db).predict_props(target, season=2099, week=1)
    assert batch is not None and batch.n_predictions == 4
    stored = db.query(f"SELECT * FROM predictions WHERE target = '{target}' ORDER BY gsis_id")
    assert (stored["model_id"] == "nb-model").all()
    assert stored["as_of_ts"].notna().all() and stored["predicted_at"].notna().all()
    assert stored["probability"].isna().all()
    np.testing.assert_allclose(stored["mean_estimate"], stored["point_estimate"])
    expected_r = model.distribution.size(stored["point_estimate"].to_numpy())
    np.testing.assert_allclose(stored["nb_dispersion"], expected_r)
    assert stored["poisson_lambda"].isna().all()  # no es Poisson
    for _, row in stored.iterrows():
        import json

        grid = json.loads(row["quantiles"])
        assert len(grid) == 19 and all(b >= a for a, b in zip(grid, grid[1:], strict=False))
