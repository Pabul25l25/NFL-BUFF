"""Pruebas del Data Agent: normalización y separación del mercado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflp.agents.data.normalize import (
    build_kickoff_utc,
    compute_as_of,
    deduplicate,
    game_type_to_season_type,
    is_closed_roof,
    is_turf,
    normalize_roof,
    normalize_surface,
    normalize_team,
    to_nullable_bool,
)
from nflp.agents.data.stadiums import (
    HIGH_ALTITUDE_THRESHOLD_M,
    get_stadium,
    haversine_km,
    home_stadium_by_team,
    stadiums_frame,
)


class TestNormalization:
    def test_surface_strips_whitespace(self) -> None:
        """nflverse publica 'grass ' con espacio final: debe limpiarse."""
        result = normalize_surface(pd.Series(["grass ", "GRASS", "fieldturf", ""]))
        assert result.iloc[0] == "grass"
        assert result.iloc[1] == "grass"
        assert result.iloc[2] == "fieldturf"
        assert pd.isna(result.iloc[3])

    def test_turf_detection(self) -> None:
        result = is_turf(pd.Series(["grass ", "fieldturf", "matrixturf", "grass"]))
        assert not result.iloc[0]
        assert result.iloc[1]
        assert result.iloc[2]
        assert not result.iloc[3]

    def test_roof_normalization(self) -> None:
        result = normalize_roof(pd.Series(["Dome", "outdoors", "", "CLOSED"]))
        assert result.iloc[0] == "dome"
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == "closed"

    def test_closed_roof_detection(self) -> None:
        """'open' es un techo retráctil abierto: el clima SÍ afecta."""
        result = is_closed_roof(pd.Series(["dome", "closed", "outdoors", "open"]))
        assert result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]
        assert not result.iloc[3]

    def test_team_relocations_unified(self) -> None:
        """Las mudanzas de franquicia no deben partir las series históricas."""
        result = normalize_team(pd.Series(["SD", "OAK", "STL", "KC"]))
        assert result.iloc[0] == "LAC"
        assert result.iloc[1] == "LV"
        assert result.iloc[2] == "LA"
        assert result.iloc[3] == "KC"

    def test_season_type_separates_playoffs(self) -> None:
        result = game_type_to_season_type(pd.Series(["REG", "WC", "DIV", "CON", "SB"]))
        assert result.iloc[0] == "REG"
        assert (result.iloc[1:] == "POST").all()

    def test_nullable_bool(self) -> None:
        result = to_nullable_bool(pd.Series([1, 0, "TRUE", "false", None]))
        assert result.iloc[0] is np.True_ or bool(result.iloc[0]) is True
        assert bool(result.iloc[1]) is False
        assert bool(result.iloc[2]) is True
        assert bool(result.iloc[3]) is False
        assert pd.isna(result.iloc[4])

    def test_deduplicate_keeps_last(self) -> None:
        frame = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
        result = deduplicate(frame, ["k"], label="test")
        assert len(result) == 2
        assert result[result["k"] == "a"]["v"].iloc[0] == 2

    def test_deduplicate_raises_on_missing_key(self) -> None:
        with pytest.raises(KeyError):
            deduplicate(pd.DataFrame({"a": [1]}), ["b"], label="test")


class TestKickoff:
    def test_converts_eastern_to_utc(self) -> None:
        """13:00 ET en septiembre (EDT, UTC-4) son las 17:00 UTC."""
        result = build_kickoff_utc(pd.Series(["2024-09-08"]), pd.Series(["13:00"]))
        assert result.iloc[0] == pd.Timestamp("2024-09-08 17:00")

    def test_handles_winter_offset(self) -> None:
        """En enero (EST, UTC-5) las 13:00 ET son las 18:00 UTC."""
        result = build_kickoff_utc(pd.Series(["2024-01-14"]), pd.Series(["13:00"]))
        assert result.iloc[0] == pd.Timestamp("2024-01-14 18:00")

    def test_missing_time_defaults(self) -> None:
        result = build_kickoff_utc(pd.Series(["2024-09-08"]), pd.Series([None]))
        assert pd.notna(result.iloc[0])

    def test_as_of_precedes_kickoff(self) -> None:
        kickoff = pd.Series([pd.Timestamp("2024-09-08 17:00")])
        as_of = compute_as_of(kickoff, offset_hours=1.0)
        assert as_of.iloc[0] == pd.Timestamp("2024-09-08 16:00")
        assert (as_of < kickoff).all()


class TestStadiums:
    def test_all_stadiums_have_coordinates(self) -> None:
        frame = stadiums_frame()
        assert frame["latitude"].notna().all()
        assert frame["longitude"].notna().all()
        assert frame["timezone"].notna().all()

    def test_coordinates_are_plausible(self) -> None:
        frame = stadiums_frame()
        assert frame["latitude"].between(-90, 90).all()
        assert frame["longitude"].between(-180, 180).all()

    def test_denver_is_high_altitude(self) -> None:
        denver = get_stadium("DEN00")
        assert denver is not None
        assert denver["altitude_m"] >= HIGH_ALTITUDE_THRESHOLD_M

    def test_mexico_city_is_high_altitude(self) -> None:
        mexico = get_stadium("MEX00")
        assert mexico is not None
        assert mexico["altitude_m"] > 2000

    def test_sea_level_stadiums_not_high_altitude(self) -> None:
        for stadium_id in ("MIA00", "SEA00", "NOR00"):
            stadium = get_stadium(stadium_id)
            assert stadium is not None
            assert stadium["altitude_m"] < HIGH_ALTITUDE_THRESHOLD_M

    def test_unknown_stadium_returns_none(self) -> None:
        assert get_stadium("XXX99") is None

    def test_every_team_has_home_stadium(self) -> None:
        mapping = home_stadium_by_team()
        for team in ("KC", "BUF", "SF", "GB", "DEN", "MIA"):
            assert team in mapping


class TestHaversine:
    def test_zero_distance(self) -> None:
        assert haversine_km(39.0, -94.0, 39.0, -94.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_kc_to_buffalo(self) -> None:
        """Arrowhead a Highmark: ~1.320 km en línea recta."""
        distance = haversine_km(39.0489, -94.4839, 42.7738, -78.7870)
        assert 1250 < distance < 1400

    def test_coast_to_coast(self) -> None:
        """Seattle a Miami: ~4.400 km."""
        distance = haversine_km(47.5952, -122.3316, 25.9580, -80.2389)
        assert 4200 < distance < 4600

    def test_symmetry(self) -> None:
        forward = haversine_km(39.0, -94.0, 42.0, -78.0)
        backward = haversine_km(42.0, -78.0, 39.0, -94.0)
        assert forward == pytest.approx(backward)

    def test_nan_input_returns_nan(self) -> None:
        assert np.isnan(haversine_km(np.nan, -94.0, 42.0, -78.0))


class TestMarketSeparation:
    """La separación estructural entre features y mercado."""

    def test_games_and_market_are_separate_tables(self, populated_db) -> None:
        games_columns = set(populated_db.get_columns("games"))
        market_columns = set(populated_db.get_columns("market_lines"))

        assert "spread_line" in market_columns
        assert "spread_line" not in games_columns
        assert "total_line" in market_columns
        assert "total_line" not in games_columns
        # El único punto de unión debe ser la clave.
        assert games_columns & market_columns == {"game_id"}


class TestInjuryAvailabilityFallback:
    def _games(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "season": [2025, 2025], "week": [1, 1],
                "home_team": ["KC", "BUF"], "away_team": ["BAL", "MIA"],
                "gameday": ["2025-09-04", "2025-09-07"],
                "kickoff_utc": pd.to_datetime(["2025-09-05 00:20", "2025-09-07 17:00"]),
            }
        )

    def _inj(self, teams: list[str], stamp=pd.NaT) -> pd.DataFrame:
        return pd.DataFrame(
            {"season": [2025] * len(teams), "week": [1] * len(teams), "team": teams,
             "date_modified": pd.Series([stamp] * len(teams), dtype="datetime64[ns]")}
        )

    def test_missing_timestamp_visible_before_kickoff(self) -> None:
        from nflp.agents.data.agent import derive_injury_availability

        out = derive_injury_availability(self._inj(["KC", "BAL", "BUF"]), self._games())
        assert out["date_modified"].notna().all()
        # Jueves: miercoles 16:00 ET = 20:00 UTC; domingo: sabado 16:00 ET.
        assert out.loc[0, "date_modified"] == pd.Timestamp("2025-09-03 20:00")
        assert out.loc[2, "date_modified"] == pd.Timestamp("2025-09-06 20:00")

    def test_never_visible_after_kickoff(self) -> None:
        from nflp.agents.data.agent import derive_injury_availability

        games = self._games()
        games.loc[0, "kickoff_utc"] = pd.Timestamp("2025-09-03 12:00")  # kickoff anomalo
        out = derive_injury_availability(self._inj(["KC"]), games)
        assert out.loc[0, "date_modified"] < games.loc[0, "kickoff_utc"]
        teams = ["KC", "BAL", "BUF", "MIA"]
        full = derive_injury_availability(self._inj(teams), self._games())
        g = self._games()
        kick = {"KC": 0, "BAL": 0, "BUF": 1, "MIA": 1}
        for i, team in enumerate(teams):
            assert full.loc[i, "date_modified"] < g.loc[kick[team], "kickoff_utc"]

    def test_existing_timestamp_kept_and_bye_stays_nat(self) -> None:
        from nflp.agents.data.agent import derive_injury_availability

        real = pd.Timestamp("2025-09-05 18:00")
        kept = derive_injury_availability(self._inj(["KC"], real), self._games())
        assert kept.loc[0, "date_modified"] == real
        bye = derive_injury_availability(self._inj(["DEN"]), self._games())
        assert pd.isna(bye.loc[0, "date_modified"])
