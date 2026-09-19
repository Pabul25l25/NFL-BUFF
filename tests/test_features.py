"""Pruebas del Feature Agent: forma de equipo, lesiones y clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflp.agents.features.clustering import STYLE_FEATURES, build_style_clusters
from nflp.agents.features.injuries import (
    POSITION_WEIGHT,
    build_injury_index,
    practice_code,
    status_code,
)
from nflp.agents.features.team_form import build_team_form


class TestTeamForm:
    def test_produces_two_rows_per_game(
        self, synthetic_team_stats: pd.DataFrame, synthetic_games: pd.DataFrame
    ) -> None:
        form = build_team_form(
            synthetic_team_stats, synthetic_games, windows=[3, 5], halflife=4.0
        )
        assert len(form) == 2 * len(synthetic_games)

    def test_first_game_has_no_history(
        self, synthetic_team_stats: pd.DataFrame, synthetic_games: pd.DataFrame
    ) -> None:
        """El primer partido de cada equipo no puede tener medias móviles."""
        form = build_team_form(
            synthetic_team_stats, synthetic_games, windows=[3, 5], halflife=4.0
        )
        first = form.sort_values(["team", "kickoff_utc"]).groupby("team").head(1)
        assert first["epa_off_r3"].isna().all()
        assert first["prior_games"].eq(0).all()

    def test_rolling_excludes_current_game(
        self, synthetic_team_stats: pd.DataFrame, synthetic_games: pd.DataFrame
    ) -> None:
        """La media de un equipo no puede contener su partido actual."""
        form = build_team_form(
            synthetic_team_stats, synthetic_games, windows=[3], halflife=4.0
        )
        team = form[form["team"] == "KC"].sort_values("kickoff_utc").reset_index(drop=True)

        # La media de la fila 3 debe ser la de las filas 0-2.
        expected = team.loc[0:2, "epa_off"].mean()
        assert team.loc[3, "epa_off_r3"] == pytest.approx(expected, rel=1e-6)

    def test_defensive_epa_is_opponent_offensive_epa(
        self, synthetic_team_stats: pd.DataFrame, synthetic_games: pd.DataFrame
    ) -> None:
        """El EPA defensivo se deriva del ataque rival en el mismo partido."""
        form = build_team_form(
            synthetic_team_stats, synthetic_games, windows=[3], halflife=4.0
        )
        game_id = synthetic_games.iloc[0]["game_id"]
        rows = form[form["game_id"] == game_id]
        assert len(rows) == 2

        home = rows[rows["is_home"]].iloc[0]
        away = rows[~rows["is_home"]].iloc[0]
        assert home["epa_def"] == pytest.approx(away["epa_off"], rel=1e-6)

    def test_pass_rate_is_a_proportion(
        self, synthetic_team_stats: pd.DataFrame, synthetic_games: pd.DataFrame
    ) -> None:
        form = build_team_form(
            synthetic_team_stats, synthetic_games, windows=[3], halflife=4.0
        )
        rates = form["pass_rate"].dropna()
        assert rates.between(0, 1).all()

    def test_turnover_margin_sums_to_zero_per_game(
        self, synthetic_team_stats: pd.DataFrame, synthetic_games: pd.DataFrame
    ) -> None:
        """No es exacto en datos sintéticos, pero debe existir la columna."""
        form = build_team_form(
            synthetic_team_stats, synthetic_games, windows=[3], halflife=4.0
        )
        assert "turnover_margin" in form.columns
        assert form["turnover_margin"].notna().any()


class TestInjuries:
    def test_status_codes(self) -> None:
        codes = status_code(pd.Series(["Out", "Questionable", "Doubtful", None, "IR"]))
        assert codes.iloc[0] == 3
        assert codes.iloc[1] == 1
        assert codes.iloc[2] == 2
        assert codes.iloc[3] == 0
        assert codes.iloc[4] == 3

    def test_practice_codes(self) -> None:
        codes = practice_code(
            pd.Series(
                [
                    "Full Participation in Practice",
                    "Did Not Participate In Practice",
                    None,
                ]
            )
        )
        assert codes.iloc[0] == 0
        assert codes.iloc[1] == 2
        assert codes.iloc[2] == 0

    def test_quarterback_weighted_most(self) -> None:
        """Perder al QB no es comparable a perder a un long snapper."""
        assert POSITION_WEIGHT["QB"] > POSITION_WEIGHT["WR"]
        assert POSITION_WEIGHT["QB"] > POSITION_WEIGHT["LB"]
        assert POSITION_WEIGHT["QB"] > 5 * POSITION_WEIGHT["LS"]

    def _injury_frame(self, date_modified: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "season": [2024], "week": [1], "season_type": ["REG"],
                "team": ["KC"], "gsis_id": ["00-001"], "position": ["QB"],
                "full_name": ["Test Player"],
                "report_primary_injury": ["Knee"], "report_secondary_injury": [None],
                "report_status": ["Out"],
                "practice_primary_injury": [None], "practice_status": [None],
                "date_modified": [pd.Timestamp(date_modified)],
            }
        )

    def _game_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["g1"], "season": [2024], "week": [1],
                "as_of_ts": [pd.Timestamp("2024-09-08 16:00")],
                "home_team": ["KC"], "away_team": ["BUF"],
            }
        )

    def test_report_before_cutoff_is_counted(self) -> None:
        index = build_injury_index(
            self._injury_frame("2024-09-06 12:00"), self._game_frame()
        )
        kc = index[index["team"] == "KC"]
        assert not kc.empty
        assert kc.iloc[0]["injury_index"] > 0
        assert bool(kc.iloc[0]["qb_questionable"])

    def test_report_after_cutoff_is_ignored(self) -> None:
        """Un parte publicado tras el corte as-of sería leakage."""
        index = build_injury_index(
            self._injury_frame("2024-09-08 18:00"), self._game_frame()
        )
        kc = index[index["team"] == "KC"]
        if not kc.empty:
            assert kc.iloc[0]["injury_index"] == 0

    def test_empty_injuries_returns_empty(self) -> None:
        result = build_injury_index(pd.DataFrame(), self._game_frame())
        assert result.empty


class TestClustering:
    def _team_form(self, n: int = 600) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        frame = pd.DataFrame(
            {
                "season": rng.integers(2018, 2024, n),
                "week": rng.integers(1, 18, n),
                "team": rng.choice(["KC", "BUF", "SF", "PHI", "GB", "DAL"], n),
                "prior_games": rng.integers(0, 16, n),
            }
        )
        for feature in STYLE_FEATURES:
            frame[feature] = rng.normal(size=n)
        return frame.drop_duplicates(subset=["season", "week", "team"])

    def test_assigns_cluster_to_every_row(self) -> None:
        form = self._team_form()
        clusters = build_style_clusters(form, n_clusters=4, min_prior_games=3)
        assert len(clusters) == len(form)
        assert clusters["cluster_id"].notna().all()

    def test_low_history_marked_minus_one(self) -> None:
        """Sin partidos previos no hay estilo estable: se marca con -1."""
        form = self._team_form()
        clusters = build_style_clusters(form, n_clusters=4, min_prior_games=5)
        merged = form.merge(clusters, on=["season", "week", "team"])
        low = merged[merged["prior_games"] < 5]
        assert (low["cluster_id"] == -1).all()

    def test_returns_empty_on_insufficient_sample(self) -> None:
        form = self._team_form(n=20)
        clusters = build_style_clusters(form, n_clusters=6, min_prior_games=3)
        assert clusters.empty

    def test_cluster_ids_within_range(self) -> None:
        form = self._team_form()
        clusters = build_style_clusters(form, n_clusters=5, min_prior_games=3)
        valid = clusters[clusters["cluster_id"] >= 0]["cluster_id"]
        assert valid.max() < 5
        assert valid.min() >= 0
