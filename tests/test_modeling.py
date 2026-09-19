"""Pruebas de modelado: cortes temporales, calibración, métricas y gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflp.agents.backtesting.metrics import (
    BREAKEVEN_ATS,
    ats_win_pct,
    calibration_curve,
    classification_metrics,
    crps_gaussian,
    expected_calibration_error,
    regression_metrics,
)
from nflp.agents.features.elo import (
    EloConfig,
    compute_elo,
    elo_win_probability,
    expected_score,
)
from nflp.agents.modeling.calibration import (
    ISOTONIC_MIN_SAMPLES,
    ProbabilityCalibrator,
    ResidualDistribution,
    confidence_level,
)
from nflp.agents.modeling.datasets import Dataset, season_folds, temporal_split_three_way
from nflp.agents.modeling.props import PropDistribution, empirical_bayes_shrinkage
from nflp.registry import ModelRegistry


def _make_dataset(seasons: list[int], rows_per_season: int = 100) -> Dataset:
    rng = np.random.default_rng(0)
    records = []
    for season in seasons:
        for index in range(rows_per_season):
            records.append(
                {
                    "season": season,
                    "week": (index % 17) + 1,
                    "season_type": "REG",
                    "game_id": f"{season}_{index}",
                    "home_team": "KC",
                    "away_team": "BUF",
                    "kickoff_utc": pd.Timestamp(f"{season}-09-10") + pd.Timedelta(days=index),
                }
            )
    meta = pd.DataFrame(records)
    X = pd.DataFrame(rng.normal(size=(len(meta), 4)), columns=list("abcd"))
    y = pd.Series(rng.normal(size=len(meta)))
    return Dataset(X=X, y=y, meta=meta, target="spread", kind="regression")


class TestTemporalSplits:
    """Los cortes deben ser temporales, nunca aleatorios."""

    def test_three_way_split_is_chronological(self) -> None:
        dataset = _make_dataset([2018, 2019, 2020, 2021, 2022])
        train, calib, val = temporal_split_three_way(
            dataset, calib_seasons=1, val_seasons=1
        )

        assert train.meta["season"].max() < calib.meta["season"].min()
        assert calib.meta["season"].max() < val.meta["season"].min()
        assert len(train) + len(calib) + len(val) == len(dataset)

    def test_three_way_split_requires_enough_seasons(self) -> None:
        dataset = _make_dataset([2020, 2021])
        with pytest.raises(ValueError, match="tres bandas"):
            temporal_split_three_way(dataset, calib_seasons=1, val_seasons=1)

    def test_season_folds_never_train_on_future(self) -> None:
        dataset = _make_dataset([2016, 2017, 2018, 2019, 2020, 2021])
        folds = season_folds(dataset, n_splits=3)

        assert folds, "Debería producir al menos un fold"
        for train_mask, val_mask in folds:
            train_seasons = dataset.meta.loc[train_mask, "season"]
            val_seasons = dataset.meta.loc[val_mask, "season"]
            assert train_seasons.max() < val_seasons.min(), (
                "Un fold entrena con datos posteriores a su validación"
            )

    def test_folds_are_disjoint(self) -> None:
        dataset = _make_dataset([2018, 2019, 2020, 2021])
        for train_mask, val_mask in season_folds(dataset, n_splits=2):
            assert not (train_mask & val_mask).any()

    def test_dataset_hash_is_deterministic(self) -> None:
        first = _make_dataset([2019, 2020])
        second = _make_dataset([2019, 2020])
        assert first.dataset_hash == second.dataset_hash


class TestMetrics:
    def test_log_loss_penalises_overconfidence(self) -> None:
        truth = np.array([1.0, 1.0, 0.0, 0.0])
        cautious = np.array([0.6, 0.6, 0.4, 0.4])
        overconfident = np.array([0.99, 0.01, 0.99, 0.01])

        assert (
            classification_metrics(truth, cautious).log_loss
            < classification_metrics(truth, overconfident).log_loss
        )

    def test_perfect_calibration_has_zero_ece(self) -> None:
        rng = np.random.default_rng(3)
        probability = rng.uniform(0.05, 0.95, 20000)
        truth = (rng.uniform(size=20000) < probability).astype(float)
        assert expected_calibration_error(truth, probability) < 0.02

    def test_miscalibration_is_detected(self) -> None:
        rng = np.random.default_rng(3)
        probability = np.full(5000, 0.9)
        truth = (rng.uniform(size=5000) < 0.5).astype(float)  # dice 90%, acierta 50%
        assert expected_calibration_error(truth, probability) > 0.3

    def test_calibration_curve_bins(self) -> None:
        rng = np.random.default_rng(1)
        probability = rng.uniform(0, 1, 2000)
        truth = (rng.uniform(size=2000) < probability).astype(float)
        curve = calibration_curve(truth, probability, n_bins=10)
        assert len(curve.mean_predicted) > 5
        assert sum(curve.counts) == 2000

    def test_regression_metrics(self) -> None:
        truth = np.array([10.0, 20.0, 30.0])
        prediction = np.array([12.0, 18.0, 33.0])
        metrics = regression_metrics(truth, prediction)
        assert metrics.mae == pytest.approx(7 / 3)
        assert metrics.rmse > metrics.mae

    def test_crps_rewards_correct_spread(self) -> None:
        """Un CRPS menor indica una distribución mejor dimensionada."""
        rng = np.random.default_rng(5)
        truth = rng.normal(0, 10, 5000)
        mu = np.zeros(5000)

        correct = crps_gaussian(truth, mu, np.full(5000, 10.0))
        too_narrow = crps_gaussian(truth, mu, np.full(5000, 2.0))
        too_wide = crps_gaussian(truth, mu, np.full(5000, 40.0))

        assert correct < too_narrow
        assert correct < too_wide

    def test_ats_excludes_pushes(self) -> None:
        actual = pd.Series([7.0, -3.0, 0.0])
        predicted = pd.Series([5.0, -5.0, 2.0])
        spread = pd.Series([3.0, -1.0, 0.0])   # la tercera es push
        result = ats_win_pct(actual, predicted, spread)
        assert 0.0 <= result <= 1.0

    def test_breakeven_threshold(self) -> None:
        """Con cuota -110 hay que acertar el 52,4% para no perder."""
        assert BREAKEVEN_ATS == pytest.approx(0.524, abs=0.001)

    def test_empty_input_returns_nan(self) -> None:
        metrics = classification_metrics(np.array([]), np.array([]))
        assert metrics.n == 0
        assert np.isnan(metrics.log_loss)


class TestElo:
    def test_equal_ratings_give_even_odds(self) -> None:
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_stronger_team_favoured(self) -> None:
        assert expected_score(1700, 1400) > 0.8

    def test_home_advantage_applied(self) -> None:
        probability = elo_win_probability(
            pd.Series([1500.0]), pd.Series([1500.0]), config=EloConfig()
        )
        assert probability.iloc[0] > 0.5

    def test_neutral_site_removes_advantage(self) -> None:
        probability = elo_win_probability(
            pd.Series([1500.0]), pd.Series([1500.0]),
            config=EloConfig(), neutral_site=pd.Series([True]),
        )
        assert probability.iloc[0] == pytest.approx(0.5)

    def test_ratings_are_zero_sum(self, synthetic_games: pd.DataFrame) -> None:
        """Lo que gana un equipo lo pierde el otro."""
        ratings = compute_elo(synthetic_games)
        change = ratings["elo_post"] - ratings["elo_pre"]
        by_game = change.groupby(ratings["game_id"]).sum()
        assert by_game.abs().max() < 1e-6

    def test_unplayed_games_do_not_update(self) -> None:
        games = pd.DataFrame(
            {
                "game_id": ["g1"], "season": [2024], "week": [1], "season_type": ["REG"],
                "kickoff_utc": [pd.Timestamp("2024-09-08")],
                "home_team": ["KC"], "away_team": ["BUF"],
                "home_score": [np.nan], "away_score": [np.nan],
                "neutral_site": [False],
            }
        )
        ratings = compute_elo(games)
        assert (ratings["elo_pre"] == ratings["elo_post"]).all()

    def test_winning_increases_rating(self, synthetic_games: pd.DataFrame) -> None:
        ratings = compute_elo(synthetic_games.head(1))
        game = synthetic_games.iloc[0]
        winner = game["home_team"] if game["result"] > 0 else game["away_team"]
        row = ratings[ratings["team"] == winner].iloc[0]
        assert row["elo_post"] > row["elo_pre"]


class TestCalibration:
    def test_platt_used_for_small_samples(self) -> None:
        """Con poca muestra la isotónica sobreajusta: debe caer a Platt."""
        rng = np.random.default_rng(0)
        n = 300
        probability = rng.uniform(0.1, 0.9, n)
        truth = (rng.uniform(size=n) < probability).astype(float)

        calibrator = ProbabilityCalibrator("isotonic").fit(probability, truth)
        assert calibrator.method == "sigmoid"
        assert n < ISOTONIC_MIN_SAMPLES

    def test_calibration_improves_biased_probabilities(self) -> None:
        rng = np.random.default_rng(1)
        n = 4000
        true_probability = rng.uniform(0.1, 0.9, n)
        truth = (rng.uniform(size=n) < true_probability).astype(float)
        # Probabilidades sesgadas al alza.
        biased = np.clip(true_probability + 0.15, 0.01, 0.99)

        calibrator = ProbabilityCalibrator("sigmoid").fit(biased, truth)
        calibrated = calibrator.transform(biased)

        assert (
            expected_calibration_error(truth, calibrated)
            < expected_calibration_error(truth, biased)
        )

    def test_output_never_hits_zero_or_one(self) -> None:
        rng = np.random.default_rng(2)
        probability = rng.uniform(0, 1, 500)
        truth = (rng.uniform(size=500) < probability).astype(float)
        calibrated = ProbabilityCalibrator("sigmoid").fit(probability, truth).transform(
            np.array([0.0, 1.0])
        )
        assert (calibrated > 0).all()
        assert (calibrated < 1).all()

    def test_unfitted_calibrator_is_identity(self) -> None:
        values = np.array([0.3, 0.7])
        np.testing.assert_allclose(ProbabilityCalibrator().transform(values), values)

    def test_residual_distribution_estimates_sigma(self) -> None:
        rng = np.random.default_rng(4)
        prediction = rng.normal(0, 5, 2000)
        truth = prediction + rng.normal(0, 13.5, 2000)

        distribution = ResidualDistribution().fit(truth, prediction)
        assert 10.0 < distribution.sigma < 17.0

    def test_probability_over_is_monotonic(self) -> None:
        distribution = ResidualDistribution(sigma=13.5)
        prediction = np.array([3.0, 3.0, 3.0])
        probabilities = distribution.probability_over(prediction, np.array([-7.0, 0.0, 7.0]))
        assert probabilities[0] > probabilities[1] > probabilities[2]

    def test_interval_contains_prediction(self) -> None:
        distribution = ResidualDistribution(sigma=13.5)
        lower, upper = distribution.interval(np.array([3.0]))
        assert lower[0] < 3.0 < upper[0]

    def test_confidence_levels(self) -> None:
        levels = confidence_level(np.array([0.52, 0.65, 0.85]))
        assert levels.iloc[0] == "baja"
        assert levels.iloc[1] == "media"
        assert levels.iloc[2] == "alta"


class TestPropDistribution:
    def test_handles_negative_yards(self) -> None:
        """Las yardas de carrera pueden ser negativas: no debe romperse."""
        rng = np.random.default_rng(6)
        prediction = np.abs(rng.normal(40, 15, 500)) + 5
        truth = prediction + rng.normal(0, 20, 500)
        truth[:20] = -6.0  # pérdidas de yardas reales

        distribution = PropDistribution().fit(truth, prediction)
        assert np.isfinite(distribution.sigma_log)
        assert distribution.shift > 6.0

    def test_probability_over_is_monotonic(self) -> None:
        distribution = PropDistribution(sigma_log=0.7, shift=1.0)
        prediction = np.array([60.0, 60.0])
        probabilities = distribution.probability_over(prediction, np.array([40.0, 90.0]))
        assert probabilities[0] > probabilities[1]

    def test_std_is_always_positive(self) -> None:
        distribution = PropDistribution()
        assert (distribution.std(np.array([0.0, 50.0, 200.0])) > 0).all()

    def test_interval_lower_bound_non_negative(self) -> None:
        distribution = PropDistribution(sigma_log=1.2)
        lower, _ = distribution.interval(np.array([5.0]))
        assert lower[0] >= 0.0


class TestShrinkage:
    def test_low_sample_pulled_toward_group_mean(self) -> None:
        values = pd.Series([100.0, 10.0, 55.0, 55.0])
        groups = pd.Series(["WR"] * 4)
        counts = pd.Series([1.0, 1.0, 20.0, 20.0])

        shrunk = empirical_bayes_shrinkage(values, groups, counts, prior_weight=4.0)
        group_mean = values.mean()

        # Los de poca muestra se acercan a la media; los de mucha, apenas.
        assert abs(shrunk.iloc[0] - group_mean) < abs(values.iloc[0] - group_mean)
        assert abs(shrunk.iloc[2] - values.iloc[2]) < 1.0


class TestPromotionGate:
    """El gate debe bloquear el sobreajuste y también el leakage."""

    def test_small_gap_allowed(self, settings, db) -> None:
        registry = ModelRegistry(settings, db)
        decision = registry.evaluate_promotion(0.60, 0.63)
        assert decision.allowed

    def test_large_gap_blocked(self, settings, db) -> None:
        registry = ModelRegistry(settings, db)
        decision = registry.evaluate_promotion(0.40, 0.70)
        assert not decision.allowed
        assert "sobreajuste" in decision.reason

    def test_validation_much_better_is_suspicious(self, settings, db) -> None:
        """Validar mucho mejor que entrenar suele delatar leakage."""
        registry = ModelRegistry(settings, db)
        decision = registry.evaluate_promotion(0.70, 0.40)
        assert not decision.allowed
        assert "leakage" in decision.reason

    def test_missing_metrics_blocked(self, settings, db) -> None:
        registry = ModelRegistry(settings, db)
        assert not registry.evaluate_promotion(np.nan, 0.6).allowed
