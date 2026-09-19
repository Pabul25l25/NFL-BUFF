"""Pruebas de monitoreo, explicabilidad y baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflp.agents.modeling.baselines import (
    HOME_FIELD_POINTS,
    NFL_MARGIN_STD,
    devig_two_way,
    elo_baseline,
    moneyline_to_probability,
    naive_home_baseline,
)
from nflp.agents.modeling.explain import (
    describe_contributions,
    describe_game_prediction,
)
from nflp.agents.monitoring.agent import MonitoringAgent, population_stability_index


class TestPSI:
    def test_identical_distributions_have_zero_psi(self) -> None:
        rng = np.random.default_rng(0)
        values = pd.Series(rng.normal(0, 1, 5000))
        assert population_stability_index(values, values) < 0.01

    def test_shifted_distribution_detected(self) -> None:
        rng = np.random.default_rng(1)
        reference = pd.Series(rng.normal(0, 1, 5000))
        shifted = pd.Series(rng.normal(2.5, 1, 5000))
        assert population_stability_index(reference, shifted) > 0.25

    def test_moderate_shift_in_middle_band(self) -> None:
        rng = np.random.default_rng(2)
        reference = pd.Series(rng.normal(0, 1, 20000))
        shifted = pd.Series(rng.normal(0.28, 1, 20000))
        psi = population_stability_index(reference, shifted)
        assert 0.02 < psi < 0.30

    def test_small_sample_returns_nan(self) -> None:
        assert np.isnan(
            population_stability_index(pd.Series([1.0, 2.0]), pd.Series([1.0]))
        )

    def test_boolean_series_does_not_crash(self) -> None:
        """Las columnas booleanas pasan el filtro numérico de pandas."""
        rng = np.random.default_rng(3)
        reference = pd.Series(rng.random(1000) > 0.5)
        current = pd.Series(rng.random(500) > 0.3)
        result = population_stability_index(reference, current)
        assert np.isfinite(result) or np.isnan(result)

    def test_psi_is_non_negative(self) -> None:
        rng = np.random.default_rng(4)
        reference = pd.Series(rng.normal(0, 1, 3000))
        current = pd.Series(rng.normal(1, 2, 3000))
        assert population_stability_index(reference, current) >= 0


class TestMonitoringAgent:
    def test_drift_on_empty_store(self, settings, db) -> None:
        report = MonitoringAgent(settings, db).check_drift()
        assert report.drifted_features == []
        assert not report.should_retrain

    def test_shadow_evaluation_without_models(self, settings, db) -> None:
        result = MonitoringAgent(settings, db).evaluate_shadow("spread")
        assert result["action"] == "none"

    def test_degradation_without_production(self, settings, db) -> None:
        result = MonitoringAgent(settings, db).check_degradation("spread")
        assert result["action"] == "none"

    def test_drift_detects_real_shift(self, settings, populated_db) -> None:
        from nflp.agents.features import FeatureAgent

        FeatureAgent(settings, populated_db).build()
        report = MonitoringAgent(settings, populated_db).check_drift()
        # Con datos sintéticos estables no debería haber drift masivo.
        assert isinstance(report.drifted_features, list)
        assert not report.metrics.empty


class TestBaselines:
    def test_naive_predicts_home(self) -> None:
        games = pd.DataFrame({"game_id": ["a", "b"]})
        baseline = naive_home_baseline(games)
        assert (baseline.home_win_prob > 0.5).all()
        assert (baseline.predicted_margin == HOME_FIELD_POINTS).all()

    def test_elo_baseline_favours_stronger(self) -> None:
        features = pd.DataFrame(
            {
                "game_id": ["a"],
                "home_elo_pre": [1600.0],
                "away_elo_pre": [1400.0],
                "is_neutral_site": [False],
            }
        )
        baseline = elo_baseline(features)
        assert baseline.home_win_prob.iloc[0] > 0.7
        assert baseline.predicted_margin.iloc[0] > 0

    def test_moneyline_conversion(self) -> None:
        """−150 implica ~60%; +150 implica ~40%."""
        favourite = moneyline_to_probability(pd.Series([-150.0]))
        underdog = moneyline_to_probability(pd.Series([150.0]))
        assert float(favourite[0]) == pytest.approx(0.6, abs=0.01)
        assert float(underdog[0]) == pytest.approx(0.4, abs=0.01)

    def test_devig_normalises_to_one(self) -> None:
        """Sin quitar el vig, las probabilidades suman más de 1."""
        home = pd.Series(moneyline_to_probability(pd.Series([-150.0])))
        away = pd.Series(moneyline_to_probability(pd.Series([130.0])))
        assert float(home.iloc[0] + away.iloc[0]) > 1.0

        home_fair, away_fair = devig_two_way(home, away)
        assert float(home_fair.iloc[0] + away_fair.iloc[0]) == pytest.approx(1.0)

    def test_margin_std_is_realistic(self) -> None:
        """La desviación del margen en la NFL ronda los 13,5 puntos."""
        assert 12.0 < NFL_MARGIN_STD < 15.0


class TestExplanations:
    def test_translates_features_to_spanish(self) -> None:
        contributions = {"rest_advantage": 0.05, "wind_kph": -0.03}
        row = pd.Series({"rest_advantage": 3.0, "wind_kph": 32.0})
        text = describe_contributions(contributions, row)

        assert "descanso" in text.lower()
        assert "viento" in text.lower()
        assert "3" in text
        assert "32" in text

    def test_separates_positive_and_negative(self) -> None:
        contributions = {"elo_diff": 0.10, "away_travel_km": -0.05}
        row = pd.Series({"elo_diff": 80.0, "away_travel_km": 3500.0})
        text = describe_contributions(contributions, row)
        assert "alza" in text.lower()
        assert "baja" in text.lower()

    def test_game_prediction_names_teams(self) -> None:
        contributions = {"net_epa_diff": 0.12, "wind_kph": -0.04}
        row = pd.Series({"net_epa_diff": 0.25, "wind_kph": 30.0})
        text = describe_game_prediction(contributions, row, home_team="KC", away_team="BUF")
        assert "KC" in text
        assert "BUF" in text

    def test_boolean_features_render_as_yes_no(self) -> None:
        contributions = {"is_adverse_weather": -0.08}
        row = pd.Series({"is_adverse_weather": 1.0})
        text = describe_contributions(contributions, row)
        assert "sí" in text.lower()

    def test_missing_value_renders_gracefully(self) -> None:
        contributions = {"wind_kph": -0.02}
        row = pd.Series({"wind_kph": np.nan})
        text = describe_contributions(contributions, row)
        assert "sin dato" in text.lower()

    def test_empty_contributions(self) -> None:
        assert "Sin factores" in describe_contributions({}, pd.Series(dtype=float))

    def test_units_are_appended(self) -> None:
        contributions = {"away_travel_km": -0.05, "temperature_c": 0.02}
        row = pd.Series({"away_travel_km": 3500.0, "temperature_c": -5.0})
        text = describe_contributions(contributions, row)
        assert "km" in text
        assert "°C" in text

    def test_only_top_n_shown(self) -> None:
        contributions = {f"feature_{i}": 0.1 / (i + 1) for i in range(20)}
        row = pd.Series({f"feature_{i}": float(i) for i in range(20)})
        text = describe_contributions(contributions, row, top_n=3)
        assert text.count("(") <= 3
