"""Pruebas de significado del feature store de jugador (v2).

Las pruebas de leakage por perturbación viven en ``test_leakage.py``; aquí se
comprueba que cada familia nueva calcula lo que dice calcular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from player_synth import FUTURE, build_features, build_league

from nflp.agents.features.player_form import build_snap_crosswalk


@pytest.fixture(scope="module")
def league():
    return build_league()


@pytest.fixture(scope="module")
def features(league) -> pd.DataFrame:
    return build_features(league)


def _row(frame: pd.DataFrame, gsis: str, season: int, week: int) -> pd.Series:
    match = frame[(frame["gsis_id"] == gsis) & (frame["season"] == season) & (frame["week"] == week)]
    assert len(match) == 1, f"{gsis} {season}-{week}: {len(match)} filas"
    return match.iloc[0]


class TestSkeleton:
    def test_future_week_has_rows_with_history_features(self, features: pd.DataFrame) -> None:
        future = features[(features["season"] == FUTURE[0]) & (features["week"] == FUTURE[1])]
        assert not future.empty
        for column in ("targets_ewm", "target_share_r3", "opp_rec_yds_allowed_pos_r5",
                       "team_qb_passing_yards_r5", "snap_pct_ewm"):
            assert future[column].notna().any(), column


class TestEwm:
    def test_ewm_matches_manual_computation(self, league, features: pd.DataFrame) -> None:
        stats = league.player_stats
        history = stats[stats["player_id"] == "C_WR1"].sort_values(["season", "week"])
        prior = history[(history["season"] == 2022) & (history["week"] < 6)]["targets"]
        expected = prior.ewm(halflife=3.0).mean().iloc[-1]
        assert _row(features, "C_WR1", 2022, 6)["targets_ewm"] == pytest.approx(expected)


class TestOpponentByPosition:
    def test_value_is_shifted_mean_for_own_position(self, league, features: pd.DataFrame) -> None:
        row = _row(features, "A_TE1", 2023, 3)
        opponent = row["opponent_team"]
        games = league.games
        stats = league.player_stats
        opp_games = games[(games["home_team"] == opponent) | (games["away_team"] == opponent)]
        opp_games = opp_games[opp_games["kickoff_utc"] < row["kickoff_utc"]].sort_values("kickoff_utc")
        last5 = opp_games.tail(5)
        allowed = []
        for _, g in last5.iterrows():
            offense = g["away_team"] if g["home_team"] == opponent else g["home_team"]
            te = stats[(stats["game_id"] == g["game_id"]) & (stats["team"] == offense)
                       & (stats["position"] == "TE")]
            allowed.append(te["receiving_yards"].sum())
        assert row["opp_rec_yds_allowed_pos_r5"] == pytest.approx(np.mean(allowed))

    def test_quarterbacks_get_no_receiving_allowed(self, features: pd.DataFrame) -> None:
        assert features.loc[features["position"] == "QB", "opp_rec_yds_allowed_pos_r5"].isna().all()


class TestTeamQb:
    def test_backup_starts_when_starter_out_before_cutoff(self, features: pd.DataFrame) -> None:
        wr = _row(features, "A_WR2", 2023, 6)
        backup = _row(features, "A_QB2", 2023, 6)
        assert wr["team_qb_passing_yards_r5"] == pytest.approx(backup["passing_yards_r5"])
        assert wr["team_qb_changed"] == 1.0

    def test_regular_starter_otherwise(self, features: pd.DataFrame) -> None:
        wr = _row(features, "A_WR2", 2023, 5)
        starter = _row(features, "A_QB1", 2023, 5)
        assert wr["team_qb_passing_yards_r5"] == pytest.approx(starter["passing_yards_r5"])
        assert wr["team_qb_changed"] == 0.0


class TestVacatedOpportunity:
    def test_teammate_moved_to_reserve_vacates_share(self, league, features: pd.DataFrame) -> None:
        wr2 = _row(features, "A_WR2", 2023, 5)
        stats = league.player_stats
        a_games = league.games[
            ((league.games["home_team"] == "A") | (league.games["away_team"] == "A"))
            & (league.games["kickoff_utc"] < wr2["kickoff_utc"])
        ].sort_values("kickoff_utc").tail(5)
        wr1 = stats[(stats["player_id"] == "A_WR1") & stats["game_id"].isin(a_games["game_id"])]
        assert wr2["vacated_target_share"] == pytest.approx(wr1["target_share"].mean())

    def test_report_published_after_cutoff_is_ignored(self, features: pd.DataFrame) -> None:
        # A_RB1 tiene un Out publicado 5 minutos DESPUÉS del corte en 2023-4.
        rb2 = _row(features, "A_RB2", 2023, 4)
        assert rb2["vacated_carry_share"] == pytest.approx(0.0)
        assert _row(features, "A_RB1", 2023, 4)["injury_status_code"] == 0

    def test_roster_status_ignored_before_reliable_season(self, league, monkeypatch) -> None:
        """En temporadas cuyo róster semanal no es una foto previa, RES no cuenta."""
        import nflp.agents.features.player_form as player_form

        monkeypatch.setattr(player_form, "ROSTER_STATUS_RELIABLE_FROM_SEASON", 2024)
        frame = build_features(league)
        assert _row(frame, "A_WR2", 2023, 5)["vacated_target_share"] == pytest.approx(0.0)

    def test_roster_dependent_columns_null_before_reliable_season(self, league, monkeypatch) -> None:
        import nflp.agents.features.player_form as player_form

        monkeypatch.setattr(player_form, "ROSTER_STATUS_RELIABLE_FROM_SEASON", 2023)
        frame = build_features(league)
        early = frame[frame["season"] == 2022]
        late = frame[frame["season"] == 2023]
        for column in player_form.ROSTER_DEPENDENT_COLUMNS:
            assert early[column].isna().all(), column
            assert late[column].notna().any(), column

    def test_no_absences_means_zero(self, features: pd.DataFrame) -> None:
        assert _row(features, "B_WR1", 2023, 5)["vacated_target_share"] == pytest.approx(0.0)


class TestDepthRankProxy:
    def test_ranks_are_unique_within_position_group(self, features: pd.DataFrame) -> None:
        ranked = features.dropna(subset=["depth_rank_proxy"]).copy()
        ranked["_grp"] = ranked["position"].replace({"FB": "RB"})
        dupes = ranked.duplicated(subset=["game_id", "team", "_grp", "depth_rank_proxy"])
        assert not dupes.any()

    def test_out_player_has_no_rank(self, features: pd.DataFrame) -> None:
        assert np.isnan(_row(features, "A_QB1", 2023, 6)["depth_rank_proxy"])
        assert _row(features, "A_QB2", 2023, 6)["depth_rank_proxy"] == 1.0


class TestLowSampleBlend:
    def test_rookie_without_history_gets_prior_from_previous_seasons(
        self, features: pd.DataFrame
    ) -> None:
        rookie = _row(features, "A_ROOK", 2023, 1)
        assert rookie["games_played_prior"] == 0
        assert bool(rookie["is_low_sample"])
        assert np.isnan(rookie["target_share_r5"])
        # Prior = cuota media de B_WR3 (WR de 1.ª ronda) en sus 4 primeros partidos de 2022.
        b = features[(features["gsis_id"] == "B_WR3") & (features["season"] == 2022)]
        b = b.sort_values("kickoff_utc")
        first_four = b[b["games_played_prior"] < 4]["target_share"]
        assert rookie["target_share_blend"] == pytest.approx(first_four.mean())

    def test_veteran_blend_equals_recent_average(self, features: pd.DataFrame) -> None:
        row = _row(features, "C_WR1", 2023, 3)
        assert row["games_played_prior"] >= 4
        assert row["target_share_blend"] == pytest.approx(row["target_share_r5"])


class TestSnapCrosswalk:
    def test_nickname_is_matched(self, features: pd.DataFrame) -> None:
        played = features[(features["gsis_id"] == "A_WR2") & (features["season"] == 2022)]
        assert played["offense_pct"].notna().all()
        assert played["snap_pct_ewm"].iloc[1:].notna().all()

    def test_ambiguous_last_name_resolved_by_initial_and_position(self) -> None:
        keys = pd.DataFrame(
            {"game_id": ["g1"], "team": ["BUF"], "gsis_id": ["G1"],
             "player_name": ["Gabe Davis"], "position": ["WR"]}
        )
        snaps = pd.DataFrame(
            {"game_id": ["g1", "g1", "g1"], "team": ["BUF"] * 3,
             "pfr_player_id": ["DaviGa00", "DaviRa00", "DaviMi00"],
             "player": ["Gabriel Davis", "Ray Davis", "Mike Davis"],
             "position": ["WR", "RB", "LB"]}
        )
        crosswalk = build_snap_crosswalk(keys, snaps)
        assert crosswalk.set_index("gsis_id")["pfr_player_id"].to_dict() == {"G1": "DaviGa00"}

    def test_official_pfr_id_takes_precedence(self) -> None:
        keys = pd.DataFrame(
            {"game_id": ["g1"], "team": ["BUF"], "gsis_id": ["G1"],
             "player_name": ["Gabe Davis"], "position": ["WR"]}
        )
        snaps = pd.DataFrame(
            {"game_id": ["g1"], "team": ["BUF"], "pfr_player_id": ["DaviGa00"],
             "player": ["Gabriel Davis"], "position": ["WR"]}
        )
        players = pd.DataFrame({"gsis_id": ["G1"], "pfr_id": ["DaviGa00"]})
        crosswalk = build_snap_crosswalk(keys, snaps, players)
        assert len(crosswalk) == 1
        assert crosswalk.iloc[0]["pfr_player_id"] == "DaviGa00"
