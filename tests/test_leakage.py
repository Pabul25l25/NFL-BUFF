"""Pruebas del contrato anti-leakage.

Estas son las pruebas más importantes del proyecto. Un fallo aquí invalida
cualquier métrica que produzca el sistema: un backtest con leakage da resultados
espectaculares y pierde dinero en producción.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nflp.agents.features.contracts import (
    DENYLIST,
    MARKET_COLUMNS,
    OUTCOME_COLUMNS,
    LeakageError,
    assert_no_leakage,
    audit_features,
    feature_columns,
    shifted_ewm,
    shifted_expanding,
    shifted_rolling,
)


class TestDenylist:
    """La denylist debe bloquear resultados y datos de mercado."""

    def test_outcomes_are_denied(self) -> None:
        for column in ("result", "home_score", "away_score", "total"):
            assert column in DENYLIST, f"'{column}' debería estar prohibida"

    def test_market_columns_are_denied(self) -> None:
        for column in ("spread_line", "total_line", "home_moneyline", "away_moneyline"):
            assert column in DENYLIST, f"'{column}' debería estar prohibida"

    def test_outcome_and_market_sets_are_disjoint(self) -> None:
        assert not (OUTCOME_COLUMNS & MARKET_COLUMNS)

    def test_feature_columns_excludes_denylist(self) -> None:
        frame = pd.DataFrame(
            {
                "epa_off_diff": [0.1], "elo_diff": [50.0],
                "result": [7], "spread_line": [-3.0], "game_id": ["x"],
            }
        )
        names = feature_columns(frame)
        assert "epa_off_diff" in names
        assert "elo_diff" in names
        assert "result" not in names
        assert "spread_line" not in names
        assert "game_id" not in names


class TestAudit:
    """La auditoría debe detectar cada tipo de leakage."""

    def _clean_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epa_off_diff": [0.1, 0.2],
                "elo_diff": [50.0, -30.0],
                "as_of_ts": pd.to_datetime(["2024-09-08 16:00", "2024-09-15 16:00"]),
            }
        )

    def test_clean_frame_passes(self) -> None:
        report = audit_features(self._clean_frame())
        assert report.is_clean, report.describe()

    def test_detects_outcome_column(self) -> None:
        frame = self._clean_frame()
        frame["result"] = [7, -3]
        report = audit_features(frame)
        assert not report.is_clean
        assert "result" in report.denied_columns

    def test_detects_market_column(self) -> None:
        frame = self._clean_frame()
        frame["spread_line"] = [-3.0, 6.5]
        report = audit_features(frame)
        assert not report.is_clean
        assert "spread_line" in report.denied_columns

    def test_detects_suspicious_name(self) -> None:
        frame = self._clean_frame()
        frame["actual_points_home"] = [24, 17]
        report = audit_features(frame)
        assert not report.is_clean
        assert "actual_points_home" in report.suspicious_columns

    def test_detects_missing_as_of(self) -> None:
        frame = self._clean_frame().drop(columns=["as_of_ts"])
        report = audit_features(frame)
        assert not report.is_clean
        assert report.missing_as_of == len(frame)

    def test_detects_as_of_after_kickoff(self) -> None:
        """El corte as-of nunca puede ser posterior al kickoff."""
        frame = self._clean_frame()
        kickoff = pd.to_datetime(["2024-09-08 17:00", "2024-09-15 13:00"])
        # La segunda fila tiene as_of 16:00 > kickoff 13:00 → leakage.
        report = audit_features(frame, kickoff=pd.Series(kickoff))
        assert not report.is_clean
        assert report.rows_after_as_of == 1

    def test_assert_raises(self) -> None:
        frame = self._clean_frame()
        frame["home_score"] = [24, 17]
        with pytest.raises(LeakageError, match="home_score"):
            assert_no_leakage(frame)


class TestShiftedAggregations:
    """Las primitivas de agregación deben excluir siempre la fila actual."""

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "team": ["KC"] * 5 + ["BUF"] * 5,
                "value": [10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    def test_rolling_excludes_current_row(self) -> None:
        frame = self._frame()
        result = shifted_rolling(frame, ["team"], "value", window=3)

        # La primera fila de cada grupo no tiene historia previa.
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[5])
        # La segunda fila ve solo la primera.
        assert result.iloc[1] == pytest.approx(10.0)
        # La cuarta ve las tres anteriores: (10+20+30)/3 = 20.
        assert result.iloc[3] == pytest.approx(20.0)
        # Crucialmente, NUNCA incluye su propio valor.
        assert result.iloc[4] == pytest.approx(30.0)  # (20+30+40)/3

    def test_rolling_respects_group_boundaries(self) -> None:
        """La media de un equipo no puede contaminarse con la de otro."""
        frame = self._frame()
        result = shifted_rolling(frame, ["team"], "value", window=3)
        # La primera fila de BUF debe ser NaN, no la media de KC.
        assert pd.isna(result.iloc[5])
        assert result.iloc[6] == pytest.approx(1.0)

    def test_ewm_excludes_current_row(self) -> None:
        frame = self._frame()
        result = shifted_ewm(frame, ["team"], "value", halflife=2.0)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)

    def test_expanding_excludes_current_row(self) -> None:
        frame = self._frame()
        result = shifted_expanding(frame, ["team"], "value")
        assert pd.isna(result.iloc[0])
        assert result.iloc[2] == pytest.approx(15.0)  # (10+20)/2

    def test_no_future_information_leaks_backwards(self) -> None:
        """Cambiar un valor futuro no debe alterar ninguna media pasada."""
        frame = self._frame()
        baseline = shifted_rolling(frame, ["team"], "value", window=3)

        modified = frame.copy()
        modified.loc[4, "value"] = 9999.0  # último partido de KC
        after = shifted_rolling(modified, ["team"], "value", window=3)

        # Las filas 0-4 no pueden haber cambiado: ninguna mira hacia adelante.
        pd.testing.assert_series_equal(
            baseline.iloc[:5], after.iloc[:5], check_names=False
        )


class TestMarketIsolation:
    """Las líneas de mercado deben estar estructuralmente aisladas."""

    def test_market_lines_in_separate_table(self, populated_db) -> None:
        """`market_lines` existe como tabla propia, fuera de `games`."""
        assert populated_db.table_exists("market_lines")
        game_columns = set(populated_db.get_columns("games"))
        for column in MARKET_COLUMNS:
            assert column not in game_columns, (
                f"'{column}' no debe vivir en `games`: rompería el aislamiento"
            )

    def test_feature_store_has_no_market_columns(self, populated_db) -> None:
        feature_columns_set = set(populated_db.get_columns("features_game"))
        assert not (feature_columns_set & MARKET_COLUMNS)

    def test_feature_store_has_no_outcome_columns(self, populated_db) -> None:
        feature_columns_set = set(populated_db.get_columns("features_game"))
        leaked = feature_columns_set & OUTCOME_COLUMNS
        assert not leaked, f"El feature store expone resultados: {leaked}"
