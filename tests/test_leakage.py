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


# ============================================================================
# Feature store de jugador (v2): cada columna nueva, verificada por perturbación.
# ============================================================================

# Columnas añadidas en feature_set_version v2 (migración 006).
PLAYER_V2_COLUMNS = [
    "target_share_r3", "air_yards_share_r3", "carry_share_r3",
    "target_share_ewm", "air_yards_share_ewm", "carry_share_ewm", "snap_pct_ewm",
    "receiving_yards_ewm", "rushing_yards_ewm", "receptions_ewm", "targets_ewm",
    "passing_yards_ewm", "carries_ewm", "passing_attempts_ewm",
    "opp_rec_yds_allowed_pos_r5", "opp_receptions_allowed_pos_r5",
    "opp_targets_allowed_pos_r5", "opp_rush_yds_allowed_rb_r5",
    "team_qb_epa_per_db_r5", "team_qb_passing_yards_r5", "team_qb_passing_yards_ewm",
    "team_qb_changed",
    "vacated_target_share", "vacated_carry_share", "depth_rank_proxy",
    "target_share_blend", "carry_share_blend", "snap_pct_blend", "targets_blend",
    "receptions_blend", "receiving_yards_blend", "rushing_yards_blend", "carries_blend",
    "passing_yards_blend",
]

# Semana objetivo de la prueba de perturbación (ver tests/player_synth.py).
_TARGET = (2023, 4)


@pytest.fixture(scope="module")
def player_leakage_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features con datos reales y con todo lo ocurrido desde la semana objetivo alterado."""
    from player_synth import build_features, build_league, perturb_from

    league = build_league()
    baseline = build_features(league)
    perturbed = build_features(perturb_from(league, *_TARGET))
    return baseline, perturbed


def _rows_up_to_target(frame: pd.DataFrame) -> pd.DataFrame:
    season, week = _TARGET
    mask = (frame["season"] < season) | ((frame["season"] == season) & (frame["week"] <= week))
    return frame[mask].sort_values(["game_id", "gsis_id"]).reset_index(drop=True)


class TestPlayerFeatureLeakage:
    """El valor de la semana w no puede depender de nada ocurrido en la semana w o después.

    Se alteran las estadísticas y los snaps de la semana objetivo y de todas las
    posteriores, y se añaden partes de lesiones publicados en el mismo instante
    del corte (``date_modified == as_of_ts``, que el ``<`` estricto excluye).
    Ninguna feature de una fila con fecha <= semana objetivo puede moverse.
    """

    @pytest.mark.parametrize("column", PLAYER_V2_COLUMNS)
    def test_new_column_ignores_target_week_and_future(
        self, player_leakage_frames: tuple[pd.DataFrame, pd.DataFrame], column: str
    ) -> None:
        baseline, perturbed = player_leakage_frames
        before = _rows_up_to_target(baseline)
        after = _rows_up_to_target(perturbed)
        assert column in before.columns, f"'{column}' no se construye"
        # La prueba solo tiene sentido si la columna tiene valores en la ventana.
        assert before[column].notna().any(), f"'{column}' es todo NaN en la ventana probada"
        pd.testing.assert_series_equal(
            before[column].astype(float), after[column].astype(float),
            check_names=False, rtol=1e-12, atol=1e-12,
        )

    def test_every_stored_column_ignores_target_week_and_future(
        self, player_leakage_frames: tuple[pd.DataFrame, pd.DataFrame], db
    ) -> None:
        """Lo mismo para TODAS las columnas que se escriben en features_player."""
        baseline, perturbed = player_leakage_frames
        stored = [
            c for c in db.get_columns("features_player")
            if c in baseline.columns and c not in {"feature_set_version", "feature_hash"}
        ]
        before = _rows_up_to_target(baseline)[stored]
        after = _rows_up_to_target(perturbed)[stored]
        pd.testing.assert_frame_equal(before, after, check_dtype=False, rtol=1e-12, atol=1e-12)

    def test_target_week_rows_exist_and_changed_data_is_really_different(
        self, player_leakage_frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Control: la perturbación sí cambia las filas posteriores (la prueba no es vacía)."""
        baseline, perturbed = player_leakage_frames
        season, week = _TARGET
        later_b = baseline[(baseline["season"] == season) & (baseline["week"] == week + 1)]
        later_p = perturbed[(perturbed["season"] == season) & (perturbed["week"] == week + 1)]
        assert not later_b.empty
        diff = (
            later_b.sort_values(["game_id", "gsis_id"])["targets_ewm"].to_numpy()
            != later_p.sort_values(["game_id", "gsis_id"])["targets_ewm"].to_numpy()
        )
        assert diff.any()

    def test_player_store_has_no_market_or_outcome_columns(self, db) -> None:
        columns = set(db.get_columns("features_player"))
        assert not (columns & DENYLIST)

    def test_player_frame_passes_audit(
        self, player_leakage_frames: tuple[pd.DataFrame, pd.DataFrame], db
    ) -> None:
        baseline, _ = player_leakage_frames
        stored = [c for c in db.get_columns("features_player") if c in baseline.columns]
        report = audit_features(baseline[stored], kickoff=baseline["kickoff_utc"])
        assert report.is_clean, report.describe()
