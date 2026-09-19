"""Features a nivel jugador-partido para los modelos de props.

**Tesis central del módulo**: en props de yardas, el *uso* domina sobre la
eficiencia. Un receptor mediocre con 10 objetivos por partido produce más yardas
que uno excelente con 3. Por eso la proyección se construye en dos capas:

1. **Uso proyectado** — cuota de objetivos, porcentaje de snaps, cuota de
   acarreos. Es lo que mejor persiste de una semana a otra.
2. **Eficiencia** — yardas por objetivo, YAC, EPA. Mucho más ruidosa; se pondera
   menos y se suaviza con ventanas más largas.

Todas las agregaciones usan las primitivas ``shifted_*``: el partido que se
predice nunca entra en su propia media.

**Decisión de diseño — el esqueleto es el róster, no las estadísticas.**
La primera versión de este módulo construía una fila por jugador *únicamente*
para los partidos que ya tenían estadísticas registradas. Eso hacía imposible
predecir props de un partido futuro: no hay estadísticas de un partido que aún
no se jugó, así que esos jugadores simplemente no aparecían en el feature store.

La solución es partir del **róster activo** de cada equipo semana a semana
(tabla ``rosters_weekly``, disponible desde que se publica el calendario) cruzado
con el calendario, y añadir las estadísticas reales con un ``LEFT JOIN`` cuando
existen. Para un partido futuro, la fila existe (viene del róster) pero sus
estadísticas "de ese partido" son nulas — lo cual es correcto y no leakage: la
media móvil (`shifted_rolling`) nunca las usa de todas formas, solo usa los
partidos *anteriores* ya jugados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflp.agents.features.contracts import shifted_rolling
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Posiciones con props de yardas relevantes.
SKILL_POSITIONS = ("WR", "TE", "RB", "FB", "QB")

# Por debajo de este número de partidos previos, el jugador se considera de
# "poca muestra" y entra en la rama semi-supervisada.
LOW_SAMPLE_THRESHOLD = 4

# Estados de róster que cuentan como "disponible esta semana". Practice squad
# (DEV), reserva/lesionado (RES), cortado (CUT), retirado (RET) y exento (EXE)
# quedan fuera: proyectar sus props no tiene sentido.
ACTIVE_ROSTER_STATUSES = ("ACT",)

# Métricas de jugador que se usan para saber si una fila "tiene estadísticas
# reales" (partido ya jugado) frente a una fila que solo viene del róster
# (partido futuro o el jugador no jugó esa semana).
_STAT_PRESENCE_COLUMNS = (
    "targets", "carries", "receptions", "attempts",
)


def build_player_form(
    player_stats: pd.DataFrame,
    games: pd.DataFrame,
    snap_counts: pd.DataFrame,
    team_form: pd.DataFrame,
    rosters: pd.DataFrame,
    players: pd.DataFrame,
    *,
    window: int = 5,
) -> pd.DataFrame:
    """Construye las features de jugador para cada partido, pasado o futuro.

    Args:
        team_form: salida de ``build_team_form``, para el contexto de equipo y
            las métricas defensivas del rival.
        rosters: tabla ``rosters_weekly`` — define QUÉ jugadores existen cada
            semana, incluidas las semanas todavía no jugadas.
        players: catálogo ``players`` — aporta el nombre para el cruce de snaps.
    """
    schedule = games[
        ["game_id", "season", "week", "season_type", "kickoff_utc", "home_team", "away_team"]
    ].copy()
    schedule["kickoff_utc"] = pd.to_datetime(schedule["kickoff_utc"], errors="coerce")

    # --- esqueleto: róster activo x calendario del equipo, pasado y futuro ---
    roster = rosters[rosters["position"].isin(SKILL_POSITIONS)].copy()
    if "status" in roster.columns:
        roster = roster[roster["status"].isin(ACTIVE_ROSTER_STATUSES)]
    if roster.empty:
        log.warning("Sin jugadores de posición ofensiva en el róster activo.")
        return pd.DataFrame()

    home_side = schedule.assign(
        team=schedule["home_team"], opponent_team=schedule["away_team"], is_home=True
    )
    away_side = schedule.assign(
        team=schedule["away_team"], opponent_team=schedule["home_team"], is_home=False
    )
    long_schedule = pd.concat([home_side, away_side], ignore_index=True)

    frame = roster.merge(
        long_schedule[
            ["game_id", "season", "week", "season_type", "kickoff_utc", "team", "opponent_team", "is_home"]
        ],
        on=["season", "week", "season_type", "team"],
        how="inner",
    )
    frame = frame.drop_duplicates(subset=["game_id", "gsis_id"])

    # Nombre del jugador (para el cruce de snaps por nombre) y posición
    # normalizada desde el catálogo, con la del róster como respaldo.
    names = players[["gsis_id", "display_name"]].rename(columns={"display_name": "player_name"})
    frame = frame.merge(names, on="gsis_id", how="left")
    frame["player_name"] = frame["player_name"].fillna(frame["gsis_id"])

    # --- estadísticas reales, solo donde existen (partidos ya jugados) ---
    from nflp.agents.stats.agent import PLAYER_METRICS

    stat_columns = [c for c in PLAYER_METRICS if c in player_stats.columns]
    stats = player_stats.rename(columns={"player_id": "gsis_id"})[
        ["game_id", "gsis_id", *stat_columns]
    ]
    frame = frame.merge(stats, on=["game_id", "gsis_id"], how="left")

    frame = frame.sort_values(["gsis_id", "kickoff_utc"]).reset_index(drop=True)

    # --- snaps: se cruzan por nombre+equipo+partido (nflverse usa id de PFR) ---
    frame = _attach_snaps(frame, snap_counts)

    # --- métricas derivadas por partido, antes de promediar ---
    frame["ypc"] = np.where(
        frame.get("carries", pd.Series(dtype=float)).fillna(0) > 0,
        frame.get("rushing_yards", np.nan) / frame.get("carries", np.nan),
        np.nan,
    )
    frame["yac"] = frame.get("receiving_yards_after_catch", np.nan)

    keys = ["gsis_id"]

    # --- uso (el driver dominante) ---
    usage_metrics = {
        "target_share": "target_share",
        "air_yards_share": "air_yards_share",
        "wopr": "wopr",
        "offense_pct": "snap_pct",
    }
    for source, label in usage_metrics.items():
        if source in frame.columns:
            frame[f"{label}_r{window}"] = shifted_rolling(frame, keys, source, window)
            frame[f"{label}_r3"] = shifted_rolling(frame, keys, source, 3)

    # --- producción reciente ---
    production_metrics = [
        "receiving_yards", "receptions", "targets", "receiving_epa", "racr",
        "rushing_yards", "carries", "rushing_epa", "ypc", "yac",
        # Ampliación: recepciones, touchdowns, yardas de pase de QB y sacks
        # recibidos. Los datos ya venían de nflverse; solo faltaba calcular su
        # media móvil, con el mismo patrón que las yardas de recepción/carrera.
        "receiving_tds", "rushing_tds", "passing_yards", "passing_tds",
        "passing_epa", "attempts", "completions", "sacks_suffered",
    ]
    for metric in production_metrics:
        if metric in frame.columns:
            frame[f"{metric}_r{window}"] = shifted_rolling(frame, keys, metric, window)
            frame[f"{metric}_r3"] = shifted_rolling(frame, keys, metric, 3)

    # --- cuota de acarreos dentro del equipo ---
    if "carries" in frame.columns:
        team_carries = (
            frame.groupby(["game_id", "team"], observed=True)["carries"].transform("sum")
        )
        frame["carry_share"] = np.where(
            team_carries > 0, frame["carries"] / team_carries, np.nan
        )
        frame[f"carry_share_r{window}"] = shifted_rolling(frame, keys, "carry_share", window)

    # --- historia disponible: cuenta PARTIDOS JUGADOS, no semanas en el róster ---
    # Un jugador puede figurar en el róster varias semanas sin ver el campo
    # (suplente, healthy scratch). "games_played_prior" debe reflejar partidos
    # reales, porque alimenta el umbral de la rama semi-supervisada.
    present = [c for c in _STAT_PRESENCE_COLUMNS if c in frame.columns]
    frame["_played_this_row"] = (
        frame[present].notna().any(axis=1) if present else pd.Series(False, index=frame.index)
    )
    frame["games_played_prior"] = (
        frame.groupby(keys, observed=True, sort=False)["_played_this_row"]
        .transform(lambda s: s.shift(1, fill_value=False).cumsum())
        .astype(int)
    )
    frame["is_low_sample"] = frame["games_played_prior"] < LOW_SAMPLE_THRESHOLD
    frame = frame.drop(columns=["_played_this_row"])

    # --- contexto de equipo propio y del rival ---
    frame = _attach_team_context(frame, team_form, window)

    has_real_stats = frame[present].notna().any(axis=1) if present else pd.Series(False, index=frame.index)
    log.info(
        "Features de jugador: %d filas (%d jugadores, %d con poca muestra, %d de partidos aún no jugados).",
        len(frame), frame["gsis_id"].nunique(), int(frame["is_low_sample"].sum()),
        int((~has_real_stats).sum()),
    )
    return frame


def _attach_snaps(frame: pd.DataFrame, snap_counts: pd.DataFrame) -> pd.DataFrame:
    """Une el porcentaje de snaps ofensivos.

    nflverse identifica a los jugadores en ``snap_counts`` con el id de Pro
    Football Reference, distinto del gsis_id de las estadísticas. Sin una tabla
    de correspondencia fiable se cruza por (partido, equipo, nombre normalizado),
    que cubre la gran mayoría de casos. Solo existe para partidos ya jugados;
    para partidos futuros queda en NaN, como corresponde.
    """
    if snap_counts.empty:
        frame["offense_pct"] = np.nan
        return frame

    snaps = snap_counts[["game_id", "team", "player", "offense_pct", "offense_snaps"]].copy()
    snaps["_name"] = _normalize_name(snaps["player"])
    snaps = snaps.drop_duplicates(subset=["game_id", "team", "_name"], keep="last")

    frame["_name"] = _normalize_name(frame["player_name"])
    merged = frame.merge(
        snaps[["game_id", "team", "_name", "offense_pct", "offense_snaps"]],
        on=["game_id", "team", "_name"],
        how="left",
    )
    coverage = merged["offense_pct"].notna().mean()
    log.debug("Cobertura de snaps tras el cruce por nombre: %.1f%%", 100 * coverage)

    return merged.drop(columns=["_name"])


def _normalize_name(series: pd.Series) -> pd.Series:
    """Normaliza nombres para el cruce: sin puntos, sufijos ni mayúsculas."""
    cleaned = (
        series.astype("string").str.normalize("NFKD")
        .str.encode("ascii", errors="ignore").str.decode("ascii")
        .str.lower()
        .str.replace(r"[.'`]", "", regex=True)
        .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)
        .str.strip()
    )
    return cleaned


def _attach_team_context(
    frame: pd.DataFrame, team_form: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Añade el contexto del equipo propio y la defensa del rival.

    ``team_form`` ya cubre partidos futuros (sus medias móviles solo dependen de
    partidos anteriores), así que esta unión funciona igual para pasado y futuro.
    """
    columns = ["game_id", "team", f"pass_rate_r{window}", f"plays_r{window}", f"epa_off_r{window}"]
    available = [c for c in columns if c in team_form.columns]
    own = team_form[available].rename(
        columns={
            f"pass_rate_r{window}": "team_pass_rate_r5",
            f"plays_r{window}": "team_plays_r5",
            f"epa_off_r{window}": "team_epa_off_r5",
        }
    )
    frame = frame.merge(own, on=["game_id", "team"], how="left")

    # Defensa del rival: se toma su fila en el mismo partido.
    # `sack_rate_def` incluido a propósito para el prop de sacks del QB: es la
    # tasa a la que los RIVALES del equipo son capturados, es decir, qué tan
    # buena es su presión de pase.
    opponent_columns = [
        "game_id", "team",
        f"pass_epa_def_r{window}", f"rush_epa_def_r{window}", f"points_against_r{window}",
        f"sack_rate_def_r{window}",
    ]
    opponent_available = [c for c in opponent_columns if c in team_form.columns]
    opponent = team_form[opponent_available].rename(
        columns={
            "team": "opponent_team",
            f"pass_epa_def_r{window}": "opp_pass_epa_def_r5",
            f"rush_epa_def_r{window}": "opp_rush_epa_def_r5",
            f"points_against_r{window}": "opp_points_allowed_r5",
            f"sack_rate_def_r{window}": "opp_sack_rate_def_r5",
        }
    )
    return frame.merge(opponent, on=["game_id", "opponent_team"], how="left")
