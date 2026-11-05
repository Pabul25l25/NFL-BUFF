"""Data Agent — ingesta y normalización.

Responsabilidad única: dejar en la base de datos una copia limpia, normalizada y
auditable de calendario, equipos, estadios, rosters y lesiones.

**Decisión de diseño clave**: al ingerir el calendario, las líneas de apuestas se
separan de `games` y se escriben en `market_lines`. Esa separación es la que hace
estructuralmente imposible que el Feature Agent las use por accidente. Ver
`docs/anti-leakage.md`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from nflp.agents.data import normalize as norm
from nflp.agents.data.nflverse import NflverseClient
from nflp.agents.data.stadiums import stadiums_frame
from nflp.core.config import Settings, get_settings
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class IngestionReport:
    """Resumen de una corrida de ingesta."""

    tables: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, table: str, rows: int) -> None:
        self.tables[table] = self.tables.get(table, 0) + rows

    def total(self) -> int:
        return sum(self.tables.values())


class DataAgent:
    """Ingesta de calendario, equipos, estadios, rosters y lesiones."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.client = NflverseClient(self.settings)

    # ------------------------------------------------------------------ util

    def _log_ingestion(
        self,
        source: str,
        asset: str,
        rows: int,
        *,
        season: int | None = None,
        status: str = "ok",
        message: str | None = None,
        url: str | None = None,
    ) -> None:
        """Registra la ingesta para auditoría (tabla ``ingestion_log``)."""
        entry = pd.DataFrame(
            [
                {
                    "ingestion_id": str(uuid.uuid4()),
                    "source": source,
                    "asset": asset,
                    "season": season,
                    "rows_ingested": rows,
                    "etag": None,
                    "url": url,
                    "status": status,
                    "message": message,
                    "fetched_at": datetime.now(UTC).replace(tzinfo=None),
                }
            ]
        )
        self.db.write_df(entry, "ingestion_log", mode="append")

    # ------------------------------------------------------------- estadios

    def ingest_stadiums(self) -> int:
        """Siembra la tabla de referencia de estadios (datos estáticos)."""
        frame = stadiums_frame()
        rows = self.db.write_df(frame, "stadiums", mode="replace")
        self._log_ingestion("static", "stadiums", rows)
        log.info("Estadios: %d filas.", rows)
        return rows

    # --------------------------------------------------------------- equipos

    def ingest_teams(self, *, force: bool = False) -> int:
        """Ingesta el catálogo de equipos."""
        raw = self.client.load_static("teams", force=force)

        frame = pd.DataFrame(
            {
                "team_abbr": norm.normalize_team(raw["team_abbr"]),
                "team_name": raw.get("team_name"),
                "team_conf": raw.get("team_conf"),
                "team_division": raw.get("team_division"),
                "team_color": raw.get("team_color"),
                "team_color2": raw.get("team_color2"),
                "team_logo_espn": raw.get("team_logo_espn"),
            }
        )
        frame = norm.deduplicate(frame, ["team_abbr"], label="teams")

        rows = self.db.write_df(frame, "teams", mode="replace")
        self._log_ingestion("nflverse", "teams", rows)
        log.info("Equipos: %d filas.", rows)
        return rows

    # -------------------------------------------------------------- partidos

    def ingest_games(self, seasons: list[int] | None = None, *, force: bool = False) -> tuple[int, int]:
        """Ingesta el calendario y **separa** las líneas de mercado.

        Returns:
            (filas en ``games``, filas en ``market_lines``)
        """
        seasons = seasons or self.settings.ingestion.seasons
        raw = self.client.load_static("games", force=force)
        raw = raw[raw["season"].isin(seasons)].copy()

        if raw.empty:
            log.warning("No hay partidos para las temporadas solicitadas.")
            return 0, 0

        kickoff = norm.build_kickoff_utc(raw["gameday"], raw["gametime"])
        season_type = norm.game_type_to_season_type(raw["game_type"])

        # --- games: contexto del partido y resultados, SIN datos de mercado ---
        games = pd.DataFrame(
            {
                "game_id": raw["game_id"].astype("string"),
                "season": pd.to_numeric(raw["season"], errors="coerce").astype("Int64"),
                "week": pd.to_numeric(raw["week"], errors="coerce").astype("Int64"),
                "season_type": season_type,
                "gameday": pd.to_datetime(raw["gameday"], errors="coerce"),
                "gametime": raw["gametime"].astype("string"),
                "kickoff_utc": kickoff,
                "weekday": raw["weekday"].astype("string"),
                "home_team": norm.normalize_team(raw["home_team"]),
                "away_team": norm.normalize_team(raw["away_team"]),
                "stadium_id": raw["stadium_id"].astype("string"),
                "stadium": raw["stadium"].astype("string"),
                "roof": norm.normalize_roof(raw["roof"]),
                "surface": norm.normalize_surface(raw["surface"]),
                "div_game": norm.to_nullable_bool(raw["div_game"]),
                "neutral_site": raw["location"].astype("string").str.strip().str.lower().eq("neutral"),
                "home_rest": pd.to_numeric(raw["home_rest"], errors="coerce").astype("Int64"),
                "away_rest": pd.to_numeric(raw["away_rest"], errors="coerce").astype("Int64"),
                "home_qb_id": raw["home_qb_id"].astype("string"),
                "away_qb_id": raw["away_qb_id"].astype("string"),
                "home_qb_name": raw["home_qb_name"].astype("string"),
                "away_qb_name": raw["away_qb_name"].astype("string"),
                "home_coach": raw["home_coach"].astype("string"),
                "away_coach": raw["away_coach"].astype("string"),
                "referee": raw["referee"].astype("string"),
                # resultados — nunca se usan como features
                "home_score": pd.to_numeric(raw["home_score"], errors="coerce").astype("Int64"),
                "away_score": pd.to_numeric(raw["away_score"], errors="coerce").astype("Int64"),
                "result": pd.to_numeric(raw["result"], errors="coerce").astype("Int64"),
                "total": pd.to_numeric(raw["total"], errors="coerce").astype("Int64"),
                "overtime": norm.to_nullable_bool(raw["overtime"]),
            }
        )
        games = norm.deduplicate(games, ["game_id"], label="games")
        games_rows = self.db.write_df(games, "games", mode="upsert", key_columns=["game_id"])

        # --- market_lines: AISLADO. Solo el Backtesting Agent lo consume. ---
        market = pd.DataFrame(
            {
                "game_id": raw["game_id"].astype("string"),
                "spread_line": pd.to_numeric(raw["spread_line"], errors="coerce"),
                "total_line": pd.to_numeric(raw["total_line"], errors="coerce"),
                "home_moneyline": pd.to_numeric(raw["home_moneyline"], errors="coerce"),
                "away_moneyline": pd.to_numeric(raw["away_moneyline"], errors="coerce"),
                "home_spread_odds": pd.to_numeric(raw["home_spread_odds"], errors="coerce"),
                "away_spread_odds": pd.to_numeric(raw["away_spread_odds"], errors="coerce"),
                "over_odds": pd.to_numeric(raw["over_odds"], errors="coerce"),
                "under_odds": pd.to_numeric(raw["under_odds"], errors="coerce"),
            }
        )
        market = norm.deduplicate(market, ["game_id"], label="market_lines")
        # Solo guardamos filas con al menos una línea publicada.
        market = market[market[["spread_line", "total_line"]].notna().any(axis=1)]
        market_rows = self.db.write_df(market, "market_lines", mode="upsert", key_columns=["game_id"])

        self._log_ingestion("nflverse", "games", games_rows)
        self._log_ingestion("nflverse", "market_lines", market_rows)
        log.info("Partidos: %d filas · líneas de mercado: %d filas.", games_rows, market_rows)
        return games_rows, market_rows

    # ------------------------------------------------------------- jugadores

    def ingest_players(self, *, force: bool = False) -> int:
        """Ingesta el catálogo de jugadores."""
        raw = self.client.load_static("players", force=force)

        # nflverse cambió el nombre de la columna de id entre versiones.
        id_column = next(
            (c for c in ("gsis_id", "player_id", "gsis_it_id") if c in raw.columns), None
        )
        if id_column is None:
            raise KeyError("No se encontró columna de identificador de jugador.")

        frame = pd.DataFrame(
            {
                "gsis_id": raw[id_column].astype("string"),
                "display_name": raw.get("display_name", raw.get("football_name")),
                "position": raw.get("position"),
                "position_group": raw.get("position_group"),
                "birth_date": pd.to_datetime(raw.get("birth_date"), errors="coerce"),
                "height": pd.to_numeric(raw.get("height"), errors="coerce"),
                "weight": pd.to_numeric(raw.get("weight"), errors="coerce"),
                "college": raw.get("college_name", raw.get("college")),
                "rookie_season": pd.to_numeric(raw.get("rookie_season"), errors="coerce").astype("Int64"),
                "draft_round": pd.to_numeric(raw.get("draft_round"), errors="coerce").astype("Int64"),
                "draft_pick": pd.to_numeric(raw.get("draft_pick"), errors="coerce").astype("Int64"),
                "headshot_url": raw.get("headshot"),
            }
        )
        frame = frame[frame["gsis_id"].notna()]
        frame = norm.deduplicate(frame, ["gsis_id"], label="players")

        rows = self.db.write_df(frame, "players", mode="replace")
        self._log_ingestion("nflverse", "players", rows)
        log.info("Jugadores: %d filas.", rows)
        return rows

    # --------------------------------------------------------------- rosters

    def ingest_rosters(self, seasons: list[int] | None = None, *, force: bool = False) -> int:
        """Ingesta los rosters semanales."""
        seasons = seasons or self.settings.ingestion.seasons
        raw = self.client.load_seasonal("weekly_rosters", seasons, force=force)
        if raw.empty:
            return 0

        frame = pd.DataFrame(
            {
                "season": pd.to_numeric(raw["season"], errors="coerce").astype("Int64"),
                "week": pd.to_numeric(raw["week"], errors="coerce").astype("Int64"),
                "season_type": norm.normalize_season_type(raw.get("game_type", pd.Series("REG", index=raw.index))),
                "team": norm.normalize_team(raw["team"]),
                "gsis_id": raw["gsis_id"].astype("string"),
                "position": raw.get("position"),
                "depth_chart_pos": raw.get("depth_chart_position"),
                "jersey_number": pd.to_numeric(raw.get("jersey_number"), errors="coerce").astype("Int64"),
                "status": raw.get("status"),
                "years_exp": pd.to_numeric(raw.get("years_exp"), errors="coerce").astype("Int64"),
            }
        )
        frame = frame[frame["gsis_id"].notna() & frame["week"].notna()]
        frame = norm.deduplicate(frame, ["season", "week", "team", "gsis_id"], label="rosters_weekly")

        rows = self.db.write_df(
            frame, "rosters_weekly", mode="upsert", key_columns=["season", "week", "team", "gsis_id"]
        )
        self._log_ingestion("nflverse", "weekly_rosters", rows)
        log.info("Rosters semanales: %d filas.", rows)
        return rows

    # -------------------------------------------------------------- lesiones

    def ingest_injuries(self, seasons: list[int] | None = None, *, force: bool = False) -> int:
        """Ingesta los partes de lesiones.

        ``date_modified`` es crítico: permite reconstruir qué se sabía antes del
        kickoff sin mirar información futura.
        """
        seasons = seasons or self.settings.ingestion.seasons
        raw = self.client.load_seasonal("injuries", seasons, force=force)
        if raw.empty:
            return 0

        frame = pd.DataFrame(
            {
                "season": pd.to_numeric(raw["season"], errors="coerce").astype("Int64"),
                "week": pd.to_numeric(raw["week"], errors="coerce").astype("Int64"),
                "season_type": norm.game_type_to_season_type(raw["game_type"]),
                "team": norm.normalize_team(raw["team"]),
                "gsis_id": raw["gsis_id"].astype("string"),
                "position": raw.get("position"),
                "full_name": raw.get("full_name"),
                "report_primary_injury": raw.get("report_primary_injury"),
                "report_secondary_injury": raw.get("report_secondary_injury"),
                "report_status": raw.get("report_status"),
                "practice_primary_injury": raw.get("practice_primary_injury"),
                "practice_status": raw.get("practice_status"),
                "date_modified": _parse_injury_timestamp(raw),
            }
        )
        frame = frame[frame["gsis_id"].notna() & frame["week"].notna()]
        frame = norm.deduplicate(frame, ["season", "week", "team", "gsis_id"], label="injuries")
        frame = self._fill_injury_availability(frame)

        rows = self.db.write_df(
            frame, "injuries", mode="upsert", key_columns=["season", "week", "team", "gsis_id"]
        )
        self._log_ingestion("nflverse", "injuries", rows)
        log.info("Lesiones: %d filas.", rows)
        return rows

    def _fill_injury_availability(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rellena ``date_modified`` ausente con la disponibilidad conservadora."""
        if not frame["date_modified"].isna().any():
            return frame
        games = self.db.query(
            "SELECT season, week, home_team, away_team, gameday, kickoff_utc FROM games"
        )
        return derive_injury_availability(frame, games)

    # ----------------------------------------------------------------- snaps

    def ingest_snap_counts(self, seasons: list[int] | None = None, *, force: bool = False) -> int:
        """Ingesta los conteos de snaps (base de la proyección de uso para props)."""
        seasons = seasons or self.settings.ingestion.seasons
        raw = self.client.load_seasonal("snap_counts", seasons, force=force)
        if raw.empty:
            return 0

        frame = pd.DataFrame(
            {
                "game_id": raw["game_id"].astype("string"),
                "pfr_player_id": raw["pfr_player_id"].astype("string"),
                "player": raw.get("player"),
                "position": raw.get("position"),
                "team": norm.normalize_team(raw["team"]),
                "season": pd.to_numeric(raw["season"], errors="coerce").astype("Int64"),
                "week": pd.to_numeric(raw["week"], errors="coerce").astype("Int64"),
                "season_type": norm.game_type_to_season_type(raw["game_type"]),
                "offense_snaps": pd.to_numeric(raw.get("offense_snaps"), errors="coerce"),
                "offense_pct": pd.to_numeric(raw.get("offense_pct"), errors="coerce"),
                "defense_snaps": pd.to_numeric(raw.get("defense_snaps"), errors="coerce"),
                "defense_pct": pd.to_numeric(raw.get("defense_pct"), errors="coerce"),
                "st_snaps": pd.to_numeric(raw.get("st_snaps"), errors="coerce"),
                "st_pct": pd.to_numeric(raw.get("st_pct"), errors="coerce"),
            }
        )
        frame = frame[frame["game_id"].notna() & frame["pfr_player_id"].notna()]
        frame = norm.deduplicate(frame, ["game_id", "pfr_player_id"], label="snap_counts")

        rows = self.db.write_df(
            frame, "snap_counts", mode="upsert", key_columns=["game_id", "pfr_player_id"]
        )
        self._log_ingestion("nflverse", "snap_counts", rows)
        log.info("Snap counts: %d filas.", rows)
        return rows

    # ------------------------------------------------------------------- run

    def run(self, seasons: list[int] | None = None, *, force: bool = False) -> IngestionReport:
        """Ejecuta la ingesta completa del agente."""
        seasons = seasons or self.settings.ingestion.seasons
        log.info(
            "Data Agent: ingiriendo temporadas %d-%d ...", min(seasons), max(seasons)
        )
        report = IngestionReport()

        report.add("stadiums", self.ingest_stadiums())
        report.add("teams", self.ingest_teams(force=force))

        games_rows, market_rows = self.ingest_games(seasons, force=force)
        report.add("games", games_rows)
        report.add("market_lines", market_rows)

        report.add("players", self.ingest_players(force=force))
        report.add("rosters_weekly", self.ingest_rosters(seasons, force=force))
        report.add("injuries", self.ingest_injuries(seasons, force=force))
        report.add("snap_counts", self.ingest_snap_counts(seasons, force=force))

        log.info("Data Agent completado: %d filas en total.", report.total())
        return report


def _parse_injury_timestamp(raw: pd.DataFrame) -> pd.Series:
    """Extrae ``date_modified`` de forma tolerante a cambios de esquema.

    **Hallazgo de esta ingesta**: nflverse eliminó la columna ``date_modified`` del
    archivo de lesiones a partir de la temporada 2025 (confirmado: 2024 y anteriores
    la traen, 2025 y 2026 no). Es la columna de la que depende todo el corte as-of de
    lesiones — ver `docs/anti-leakage.md`.

    Sin ella, no hay forma honesta de saber cuándo se publicó un parte dentro de la
    semana. La postura del proyecto es la misma que ante cualquier dato incierto:
    **no inventar un timestamp**. Se devuelve ``NaT`` para esas filas, lo que hace
    que `build_injury_index` las trate como "no visibles todavía" — el índice de
    lesiones sale en 0 para esas temporadas en vez de arriesgar un leak. Es una
    pérdida de señal real, documentada, preferible a fabricar datos.

    Si en el futuro nflverse repone la columna con otro nombre, basta con añadirlo
    a `candidates`.
    """
    candidates = ("date_modified", "last_updated", "modified_at")
    for column in candidates:
        if column in raw.columns:
            return pd.to_datetime(raw[column], errors="coerce", utc=True).dt.tz_localize(None)

    log.warning(
        "La fuente de lesiones no trae columna de fecha de modificación (%s); "
        "el índice de lesiones quedará en 0 para estas filas (no se fabrica un "
        "timestamp). Ver docs/anti-leakage.md.",
        ", ".join(candidates),
    )
    return pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")


# Hora (ET) del dia previo al partido a la que se considera disponible el parte final.
INJURY_FALLBACK_HOUR_ET = 16


def derive_injury_availability(injuries: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Rellena ``date_modified`` ausente con una marca conservadora y sin leakage.

    Cuando nflverse no trae ``date_modified`` (2025+), el parte final de la semana se
    considera disponible a las 16:00 ET del dia anterior al partido del equipo
    (``gameday - 1 dia``), y nunca mas tarde que 1 minuto antes del kickoff. Las
    filas con ``date_modified`` real se conservan intactas. Equipos sin partido esa
    semana (bye) o sin fecha de partido quedan en NaT: no se fabrica nada.
    """
    result = injuries.copy()
    missing = result["date_modified"].isna()
    if not missing.any() or games.empty:
        return result

    cols = ["season", "week", "team", "gameday", "kickoff_utc"]
    long_games = pd.concat(
        [
            games.rename(columns={"home_team": "team"})[cols],
            games.rename(columns={"away_team": "team"})[cols],
        ],
        ignore_index=True,
    )
    long_games["gameday"] = pd.to_datetime(long_games["gameday"], errors="coerce")
    long_games["kickoff_utc"] = pd.to_datetime(long_games["kickoff_utc"], errors="coerce")
    # Un equipo juega una vez por semana; ante duplicados, el kickoff mas temprano.
    long_games = (
        long_games.dropna(subset=["gameday"])
        .sort_values("kickoff_utc")
        .drop_duplicates(["season", "week", "team"], keep="first")
        .reset_index(drop=True)
    )

    local = (
        long_games["gameday"].dt.normalize()
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=INJURY_FALLBACK_HOUR_ET)
    ).dt.tz_localize(norm.NFL_SCHEDULE_TZ, nonexistent="shift_forward", ambiguous=True)
    avail = local.dt.tz_convert("UTC").dt.tz_localize(None)
    cap = long_games["kickoff_utc"] - pd.Timedelta(minutes=1)
    long_games["fallback_ts"] = avail.where(cap.isna() | (avail <= cap), cap)

    keys = ["season", "week", "team"]
    left = result[keys].copy()
    right = long_games[keys + ["fallback_ts"]].copy()
    for frame in (left, right):
        frame["season"] = frame["season"].astype("int64")
        frame["week"] = frame["week"].astype("int64")
    merged = left.merge(right, on=keys, how="left")
    fallback = pd.Series(merged["fallback_ts"].to_numpy(), index=result.index)
    result.loc[missing, "date_modified"] = fallback[missing]
    log.warning(
        "Lesiones sin date_modified: %d filas con disponibilidad derivada "
        "(gameday-1 %d:00 ET, tope kickoff-1min); %d quedan en NaT (sin partido).",
        int(missing.sum()), INJURY_FALLBACK_HOUR_ET, int(result["date_modified"].isna().sum()),
    )
    return result


# Hora (ET) del día previo al partido a la que se considera disponible el parte final.
INJURY_FALLBACK_HOUR_ET = 16
