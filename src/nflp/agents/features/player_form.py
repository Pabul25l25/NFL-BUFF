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

**Familias añadidas en v2** (todas con su sello temporal razonado):

* EWM (``*_ewm``, ``shifted_ewm``) de uso y producción.
* Defensa rival por posición (``opp_*_allowed_*_r5``): media desplazada de lo
  que la defensa rival concedió a WR/TE/RB en sus partidos anteriores.
* QB titular esperado del equipo (``team_qb_*``): se elige con datos previos
  (carga de dropbacks desplazada) y el parte de lesiones con corte as-of.
* Oportunidad vacante (``vacated_*``): cuota previa de compañeros que constan
  como baja antes del corte (parte de lesiones as-of o fuera del róster activo).
* Rango de profundidad *proxy* (``depth_rank_proxy``): orden de uso previo
  dentro de la posición; ``depth_chart_pos`` no trae rango, solo la posición.
* Mezcla con prior para poca muestra (``*_blend``): contracción hacia el uso de
  jugadores comparables en temporadas *anteriores*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflp.agents.features.contracts import shifted_ewm, shifted_expanding, shifted_rolling
from nflp.agents.features.injuries import player_injury_status
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Posiciones con props de yardas relevantes.
SKILL_POSITIONS = ("WR", "TE", "RB", "FB", "QB")

# Grupo posicional usado para la defensa rival, el rango de profundidad y el
# prior de poca muestra. El FB se trata como RB.
POSITION_GROUP: dict[str, str] = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "QB": "QB"}

# Por debajo de este número de partidos previos, el jugador se considera de
# "poca muestra" y entra en la rama semi-supervisada.
LOW_SAMPLE_THRESHOLD = 4

# Estados de róster que cuentan como "disponible esta semana". Practice squad
# (DEV), reserva/lesionado (RES), cortado (CUT), retirado (RET) y exento (EXE)
# quedan fuera: proyectar sus props no tiene sentido.
ACTIVE_ROSTER_STATUSES = ("ACT",)

# Estados que NO confirman una baja antes del corte as-of. INA (inactivo del día
# de partido) se publica ~90 minutos antes del kickoff y no existe todavía cuando
# se predice entre semana, así que no puede usarse para marcar una ausencia: haría
# que el entrenamiento viera información que en producción no está disponible.
PRESENT_OR_UNKNOWN_STATUSES = ("ACT", "INA")

# Primera temporada en la que el estado de ``rosters_weekly`` es una foto PREVIA
# al partido. Comprobado en los datos: desde 2016, el 99.8-100% de los jugadores
# con uso en la semana w figuran ACT en el róster de la semana w. En 2011-2015
# solo el 86-90%: el estado de esas temporadas arrastra información posterior
# (p. ej. un IR futuro), así que no puede usarse para inferir bajas.
ROSTER_STATUS_RELIABLE_FROM_SEASON = 2016

# Columnas que dependen de qué compañeros figuran ACT esa semana.
ROSTER_DEPENDENT_COLUMNS = (
    "team_qb_epa_per_db_r5", "team_qb_passing_yards_r5", "team_qb_passing_yards_ewm",
    "team_qb_changed", "depth_rank_proxy",
)

# Código de parte de lesión a partir del cual el jugador se considera baja.
OUT_STATUS_CODE = 3
DOUBTFUL_STATUS_CODE = 2

# Vida media (en partidos) de las medias exponenciales de jugador.
PLAYER_EWM_HALFLIFE = 3.0

# Un compañero cuenta para la oportunidad vacante si tuvo uso en alguno de los
# últimos N partidos del equipo.
VACATED_RECENT_GAMES = 3

# Métricas de jugador que se usan para saber si una fila "tiene estadísticas
# reales" (partido ya jugado) frente a una fila que solo viene del róster
# (partido futuro o el jugador no jugó esa semana).
_STAT_PRESENCE_COLUMNS = (
    "targets", "carries", "receptions", "attempts",
)

# Medias exponenciales: columna de origen -> etiqueta de salida.
EWM_METRICS: dict[str, str] = {
    "target_share": "target_share",
    "air_yards_share": "air_yards_share",
    "carry_share": "carry_share",
    "offense_pct": "snap_pct",
    "receiving_yards": "receiving_yards",
    "rushing_yards": "rushing_yards",
    "receptions": "receptions",
    "targets": "targets",
    "passing_yards": "passing_yards",
    "carries": "carries",
    "attempts": "passing_attempts",
}

# Métricas con versión "blend" (prior para poca muestra): origen -> etiqueta.
BLEND_METRICS: dict[str, str] = {
    "target_share": "target_share",
    "carry_share": "carry_share",
    "offense_pct": "snap_pct",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "rushing_yards": "rushing_yards",
    "carries": "carries",
    "passing_yards": "passing_yards",
}


def build_player_form(
    player_stats: pd.DataFrame,
    games: pd.DataFrame,
    snap_counts: pd.DataFrame,
    team_form: pd.DataFrame,
    rosters: pd.DataFrame,
    players: pd.DataFrame,
    *,
    window: int = 5,
    injuries: pd.DataFrame | None = None,
    as_of_offset_hours: float = 1.0,
    ewm_halflife: float = PLAYER_EWM_HALFLIFE,
) -> pd.DataFrame:
    """Construye las features de jugador para cada partido, pasado o futuro.

    Args:
        team_form: salida de ``build_team_form``, para el contexto de equipo y
            las métricas defensivas del rival.
        rosters: tabla ``rosters_weekly`` completa (todos los estados) — define
            QUÉ jugadores existen cada semana, incluidas las semanas todavía no
            jugadas, y quién está fuera del róster activo.
        players: catálogo ``players`` — nombre (cruce de snaps), ``draft_round`` y
            ``rookie_season`` (prior de poca muestra) y ``pfr_id`` si existe.
        injuries: tabla ``injuries``; se resuelve con corte as-of estricto.
        as_of_offset_hours: margen antes del kickoff para ``as_of_ts``.
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
    # DuckDB devuelve VARCHAR como dtype "string" de pandas: sus comparaciones
    # dan booleanos con NA, que np.select / .loc no aceptan. Se pasa a object.
    frame["position"] = frame["position"].astype(object)

    # Nombre del jugador (para el cruce de snaps) y metadatos estáticos del
    # catálogo (draft, temporada de novato): se conocen antes de cualquier partido.
    meta_columns = [
        c for c in ("gsis_id", "display_name", "draft_round", "rookie_season")
        if c in players.columns
    ]
    meta = (
        players[meta_columns].drop_duplicates(subset=["gsis_id"])
        .rename(columns={"display_name": "player_name"})
    )
    frame = frame.merge(meta, on="gsis_id", how="left")
    if "player_name" not in frame.columns:
        frame["player_name"] = np.nan
    frame["player_name"] = frame["player_name"].fillna(frame["gsis_id"])

    # --- estadísticas reales, solo donde existen (partidos ya jugados) ---
    from nflp.agents.stats.agent import PLAYER_METRICS

    stats_full = _with_carry_share(player_stats)
    stat_columns = [c for c in PLAYER_METRICS if c in stats_full.columns]
    if "carry_share" in stats_full.columns:
        stat_columns.append("carry_share")
    stats = stats_full.rename(columns={"player_id": "gsis_id"})[
        ["game_id", "gsis_id", *stat_columns]
    ].drop_duplicates(subset=["game_id", "gsis_id"], keep="last")
    frame = frame.merge(stats, on=["game_id", "gsis_id"], how="left")

    # --- corte as-of y parte de lesiones (solo partes publicados antes) ---
    frame["as_of_ts"] = frame["kickoff_utc"] - pd.Timedelta(hours=as_of_offset_hours)
    frame = _attach_injury_status(frame, injuries)

    # --- snaps: id de PFR -> gsis_id (catálogo si existe; si no, inferido) ---
    frame = _attach_snaps(frame, snap_counts, players)

    frame = frame.sort_values(["gsis_id", "kickoff_utc", "game_id"]).reset_index(drop=True)

    # --- métricas derivadas por partido, antes de promediar ---
    frame["ypc"] = np.where(
        frame.get("carries", pd.Series(dtype=float)).fillna(0) > 0,
        frame.get("rushing_yards", np.nan) / frame.get("carries", np.nan),
        np.nan,
    )
    frame["yac"] = frame.get("receiving_yards_after_catch", np.nan)
    if "attempts" in frame.columns:
        sacks = frame["sacks_suffered"] if "sacks_suffered" in frame.columns else 0.0
        frame["dropbacks"] = frame["attempts"] + pd.Series(sacks, index=frame.index).fillna(0)

    keys = ["gsis_id"]

    # --- uso (el driver dominante) ---
    usage_metrics = {
        "target_share": "target_share",
        "air_yards_share": "air_yards_share",
        "wopr": "wopr",
        "offense_pct": "snap_pct",
        "carry_share": "carry_share",
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
        "passing_epa", "attempts", "completions", "sacks_suffered", "dropbacks",
    ]
    for metric in production_metrics:
        if metric in frame.columns:
            frame[f"{metric}_r{window}"] = shifted_rolling(frame, keys, metric, window)
            frame[f"{metric}_r3"] = shifted_rolling(frame, keys, metric, 3)

    # --- forma reciente ponderada (EWM, excluye el partido objetivo) ---
    for source, label in EWM_METRICS.items():
        if source in frame.columns:
            frame[f"{label}_ewm"] = shifted_ewm(frame, keys, source, ewm_halflife)

    # --- historia disponible: cuenta PARTIDOS JUGADOS, no semanas en el róster ---
    # Un jugador puede figurar en el róster varias semanas sin ver el campo
    # (suplente, healthy scratch). "games_played_prior" debe reflejar partidos
    # reales, porque alimenta el umbral de la rama semi-supervisada.
    present = [c for c in _STAT_PRESENCE_COLUMNS if c in frame.columns]
    frame["_played"] = (
        frame[present].notna().any(axis=1) if present else pd.Series(False, index=frame.index)
    )
    frame["games_played_prior"] = (
        frame.groupby(keys, observed=True, sort=False)["_played"]
        .transform(lambda s: s.shift(1, fill_value=False).cumsum())
        .astype(int)
    )
    frame["is_low_sample"] = frame["games_played_prior"] < LOW_SAMPLE_THRESHOLD

    # --- contexto de equipo propio y del rival ---
    frame = _attach_team_context(frame, team_form, window)
    frame = _attach_opponent_position_defense(frame, stats_full, long_schedule, window)

    # --- disponibilidad: QB titular, rango de profundidad, oportunidad vacante ---
    frame = _attach_team_qb(frame, stats_full, long_schedule, window)
    frame = _attach_depth_rank_proxy(frame)
    frame = _attach_vacated_opportunity(
        frame, stats_full, long_schedule, rosters, injuries,
        as_of_offset_hours=as_of_offset_hours, window=window,
    )

    # El QB titular esperado y el rango de profundidad eligen entre los
    # compañeros del róster ACT. Antes de ROSTER_STATUS_RELIABLE_FROM_SEASON ese
    # estado no es una foto previa al partido, así que esas columnas se anulan.
    unreliable = frame["season"] < ROSTER_STATUS_RELIABLE_FROM_SEASON
    frame.loc[unreliable, [*ROSTER_DEPENDENT_COLUMNS]] = np.nan

    # --- prior para poca muestra ---
    frame = frame.sort_values(["gsis_id", "kickoff_utc", "game_id"]).reset_index(drop=True)
    frame = _attach_low_sample_blend(frame, window)

    log.info(
        "Features de jugador: %d filas (%d jugadores, %d con poca muestra, %d de partidos aún no jugados).",
        len(frame), frame["gsis_id"].nunique(), int(frame["is_low_sample"].sum()),
        int((~frame["_played"]).sum()),
    )
    return frame.drop(columns=["_played"])


# =============================================================== utilidades


def _with_carry_share(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Añade la cuota de acarreos del partido sobre el total del equipo.

    El denominador sale de *todas* las filas de estadísticas del equipo en ese
    partido (no solo del róster activo de posiciones ofensivas), para que un
    jugador subido del practice squad no infle la cuota de los demás.
    """
    stats = player_stats.copy()
    if "carries" not in stats.columns or stats.empty:
        return stats
    team_carries = stats.groupby(["game_id", "team"], observed=True)["carries"].transform("sum")
    stats["carry_share"] = np.where(team_carries > 0, stats["carries"] / team_carries, np.nan)
    return stats


def _attach_injury_status(frame: pd.DataFrame, injuries: pd.DataFrame | None) -> pd.DataFrame:
    """Código de parte de lesión de cada fila, con corte as-of estricto."""
    if injuries is None or injuries.empty:
        frame["injury_status_code"] = 0
        frame["practice_status_code"] = 0
        return frame
    status = player_injury_status(
        injuries, frame[["game_id", "season", "week", "as_of_ts", "gsis_id"]]
    )
    # Un jugador traspasado puede tener dos partes la misma semana: el peor manda.
    status = status.groupby(["game_id", "gsis_id"], as_index=False)[
        ["injury_status_code", "practice_status_code"]
    ].max()
    frame = frame.drop(columns=["injury_status_code", "practice_status_code"], errors="ignore")
    frame = frame.merge(status, on=["game_id", "gsis_id"], how="left")
    frame["injury_status_code"] = frame["injury_status_code"].fillna(0).astype(int)
    frame["practice_status_code"] = frame["practice_status_code"].fillna(0).astype(int)
    return frame


# ==================================================================== snaps


def _attach_snaps(
    frame: pd.DataFrame, snap_counts: pd.DataFrame, players: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Une el porcentaje de snaps ofensivos por identidad de jugador.

    nflverse identifica a los jugadores en ``snap_counts`` con el id de Pro
    Football Reference, distinto del gsis_id. La v1 cruzaba por nombre exacto en
    cada partido, lo que perdía a quien figura con apodo en una fuente y nombre
    completo en la otra ("Gabe"/"Gabriel"). Ahora se construye una tabla
    ``pfr_player_id -> gsis_id`` (ver ``build_snap_crosswalk``) y se cruza por
    ``(game_id, gsis_id)``. Solo existe para partidos ya jugados; para partidos
    futuros queda en NaN, como corresponde (solo se usa vía ``shifted_*``).
    """
    if snap_counts.empty:
        frame["offense_pct"] = np.nan
        frame["offense_snaps"] = np.nan
        return frame

    crosswalk = build_snap_crosswalk(
        frame[["game_id", "team", "gsis_id", "player_name", "position"]], snap_counts, players
    )
    snaps = snap_counts[["game_id", "pfr_player_id", "offense_pct", "offense_snaps"]].merge(
        crosswalk, on="pfr_player_id", how="inner"
    )
    snaps = snaps.drop_duplicates(subset=["game_id", "gsis_id"], keep="last")
    merged = frame.merge(
        snaps[["game_id", "gsis_id", "offense_pct", "offense_snaps"]],
        on=["game_id", "gsis_id"],
        how="left",
    )
    log.debug(
        "Cobertura de snaps: %.1f%% de filas (%d ids PFR mapeados).",
        100 * merged["offense_pct"].notna().mean(), len(crosswalk),
    )
    return merged


def build_snap_crosswalk(
    keys: pd.DataFrame, snap_counts: pd.DataFrame, players: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Tabla ``pfr_player_id -> gsis_id`` para cruzar ``snap_counts``.

    Orden de preferencia:

    1. ``players.pfr_id`` si el catálogo lo trae (mapeo oficial).
    2. Inferido de los datos: pares (gsis, pfr) que coinciden en el mismo partido
       y equipo por nombre normalizado exacto; después, entre los que quedan sin
       pareja en ese partido, por apellido + inicial y por apellido solo, exigiendo
       mismo grupo posicional y unicidad a ambos lados. Cada par suma un voto y se
       conserva el emparejamiento mutuamente mejor, de modo que un cruce espurio
       aislado no puede imponerse a decenas de coincidencias exactas.

    La identidad de un jugador no es información de rendimiento: usar todos los
    partidos para fijarla no filtra nada del futuro.
    """
    direct = pd.DataFrame(columns=["gsis_id", "pfr_player_id"])
    if players is not None and "pfr_id" in players.columns:
        direct = (
            players[["gsis_id", "pfr_id"]].dropna()
            .rename(columns={"pfr_id": "pfr_player_id"})
            .drop_duplicates(subset=["pfr_player_id"])
            .drop_duplicates(subset=["gsis_id"])
        )

    left = keys.drop_duplicates(subset=["game_id", "gsis_id"]).copy()
    left["_name"] = _normalize_name(left["player_name"])
    left["_pos"] = left["position"].astype(object).map(POSITION_GROUP)
    right = snap_counts[["game_id", "team", "pfr_player_id", "player", "position"]].drop_duplicates(
        subset=["game_id", "pfr_player_id"]
    ).copy()
    right["_name"] = _normalize_name(right["player"])
    right["_pos"] = right["position"].astype(object).map(POSITION_GROUP)

    # 1) nombre exacto dentro de (partido, equipo); ante ambigüedad, misma posición.
    exact = left.merge(right, on=["game_id", "team", "_name"], suffixes=("", "_snap"))
    exact["_same_pos"] = exact["_pos"].eq(exact["_pos_snap"]).fillna(False).astype(bool)
    exact = exact.sort_values("_same_pos", ascending=False, kind="mergesort")
    exact = exact.drop_duplicates(subset=["game_id", "gsis_id"]).drop_duplicates(
        subset=["game_id", "pfr_player_id"]
    )
    votes = [exact[["game_id", "gsis_id", "pfr_player_id"]]]

    # 2) apodos: apellido + inicial, luego apellido solo, entre los sin pareja.
    matched_left = exact[["game_id", "gsis_id"]]
    matched_right = exact[["game_id", "pfr_player_id"]]
    for key_builder in (_last_and_initial, _last_name):
        lu = _anti_join(left, matched_left, ["game_id", "gsis_id"])
        ru = _anti_join(right[right["_pos"].notna()], matched_right, ["game_id", "pfr_player_id"])
        if lu.empty or ru.empty:
            break
        lu = lu.assign(_k=key_builder(lu["_name"]))
        ru = ru.assign(_k=key_builder(ru["_name"]))
        cand = lu.merge(ru, on=["game_id", "team", "_pos", "_k"], suffixes=("", "_snap"))
        cand = cand[
            ~cand.duplicated(subset=["game_id", "gsis_id"], keep=False)
            & ~cand.duplicated(subset=["game_id", "pfr_player_id"], keep=False)
        ]
        votes.append(cand[["game_id", "gsis_id", "pfr_player_id"]])
        matched_left = pd.concat([matched_left, cand[["game_id", "gsis_id"]]], ignore_index=True)
        matched_right = pd.concat(
            [matched_right, cand[["game_id", "pfr_player_id"]]], ignore_index=True
        )

    tally = (
        pd.concat(votes, ignore_index=True)
        .groupby(["gsis_id", "pfr_player_id"], as_index=False).size()
        .sort_values("size", ascending=False, kind="mergesort")
    )
    best_for_pfr = tally.drop_duplicates(subset=["pfr_player_id"])
    best_for_gsis = tally.drop_duplicates(subset=["gsis_id"])
    derived = best_for_pfr.merge(
        best_for_gsis[["gsis_id", "pfr_player_id"]], on=["gsis_id", "pfr_player_id"]
    )[["gsis_id", "pfr_player_id"]]

    # El mapeo oficial manda; lo inferido completa los huecos.
    derived = derived[
        ~derived["pfr_player_id"].isin(direct["pfr_player_id"])
        & ~derived["gsis_id"].isin(direct["gsis_id"])
    ]
    if direct.empty:
        return derived.reset_index(drop=True)
    return pd.concat([direct, derived], ignore_index=True)


def _anti_join(frame: pd.DataFrame, exclude: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    marker = exclude[on].drop_duplicates().assign(_hit=True)
    merged = frame.merge(marker, on=on, how="left")
    return merged[merged["_hit"].isna()].drop(columns=["_hit"])


def _last_name(names: pd.Series) -> pd.Series:
    return names.str.split().str[-1]


def _last_and_initial(names: pd.Series) -> pd.Series:
    return names.str.split().str[-1] + "|" + names.str[:1]


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


# ========================================================== equipo y rival


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


_ALLOWED_METRICS = {
    "receiving_yards": "opp_rec_yds_allowed",
    "receptions": "opp_receptions_allowed",
    "targets": "opp_targets_allowed",
    "rushing_yards": "opp_rush_yds_allowed",
}


def _attach_opponent_position_defense(
    frame: pd.DataFrame, player_stats: pd.DataFrame, long_schedule: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Producción concedida por la defensa rival a cada grupo posicional.

    Para cada defensa y partido se suma lo que produjeron los WR, TE y RB del
    ataque contrario; después se promedia con ``shifted_rolling`` sobre los
    partidos de esa defensa ordenados por kickoff, de modo que el partido
    objetivo nunca entra. Cada jugador recibe el valor de *su* posición:

    * ``opp_rec_yds_allowed_pos_r5`` / ``opp_receptions_allowed_pos_r5`` /
      ``opp_targets_allowed_pos_r5`` — a WR, TE o RB según el jugador (NaN en QB).
    * ``opp_rush_yds_allowed_rb_r5`` — yardas de carrera concedidas a RB (todas
      las filas: es la defensa de carrera del rival frente a corredores).
    """
    out_columns = [
        "opp_rec_yds_allowed_pos_r5", "opp_receptions_allowed_pos_r5",
        "opp_targets_allowed_pos_r5", "opp_rush_yds_allowed_rb_r5",
    ]
    metrics = [m for m in _ALLOWED_METRICS if m in player_stats.columns]
    if player_stats.empty or not metrics or "position" not in player_stats.columns:
        for column in out_columns:
            frame[column] = np.nan
        return frame

    stats = player_stats[["game_id", "team", "position", *metrics]].copy()
    stats["_pos"] = stats["position"].astype(object).map(POSITION_GROUP)

    # Un partido "jugado" para un ataque = tiene alguna fila de estadísticas.
    played = (
        stats.groupby(["game_id", "team"], observed=True).size().rename("_rows").reset_index()
    )
    per_pos = (
        stats[stats["_pos"].isin(["WR", "TE", "RB"])]
        .groupby(["game_id", "team", "_pos"], observed=True)[metrics].sum()
        .unstack("_pos")
    )
    per_pos.columns = [f"{metric}__{pos}" for metric, pos in per_pos.columns]
    per_pos = per_pos.reset_index()

    defense = long_schedule[["game_id", "kickoff_utc", "team", "opponent_team"]].rename(
        columns={"team": "defense_team", "opponent_team": "offense_team"}
    )
    defense = defense.merge(
        played.rename(columns={"team": "offense_team"}), on=["game_id", "offense_team"], how="left"
    ).merge(
        per_pos.rename(columns={"team": "offense_team"}), on=["game_id", "offense_team"], how="left"
    )
    value_columns = [c for c in defense.columns if "__" in c]
    was_played = defense["_rows"].notna()
    # Partido jugado sin producción de una posición = 0 concedido, no "sin dato".
    defense.loc[was_played, value_columns] = defense.loc[was_played, value_columns].fillna(0.0)
    defense = defense.sort_values(["defense_team", "kickoff_utc", "game_id"]).reset_index(drop=True)

    rolled = defense[["game_id", "defense_team"]].copy()
    for column in value_columns:
        rolled[f"{column}_r"] = shifted_rolling(defense, ["defense_team"], column, window)
    rolled = rolled.rename(columns={"defense_team": "opponent_team"})
    frame = frame.merge(rolled, on=["game_id", "opponent_team"], how="left")

    group = frame["position"].astype(object).map(POSITION_GROUP)
    conditions = [group.eq(pos).fillna(False).to_numpy(dtype=bool) for pos in ("WR", "TE", "RB")]
    for metric in ("receiving_yards", "receptions", "targets"):
        label = _ALLOWED_METRICS[metric]
        choices = [
            frame.get(f"{metric}__{pos}_r", pd.Series(np.nan, index=frame.index))
            for pos in ("WR", "TE", "RB")
        ]
        frame[f"{label}_pos_r5"] = np.select(
            conditions, [c.to_numpy(dtype=float) for c in choices], default=np.nan
        )
    frame["opp_rush_yds_allowed_rb_r5"] = frame.get(
        "rushing_yards__RB_r", pd.Series(np.nan, index=frame.index)
    )
    for column in out_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.drop(columns=[f"{c}_r" for c in value_columns], errors="ignore")


# ================================================== disponibilidad y roles


def _attach_team_qb(
    frame: pd.DataFrame, player_stats: pd.DataFrame, long_schedule: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Calidad del QB que se espera que sea titular del equipo en el partido.

    **Elección del titular, solo con información previa al corte:**
    candidatos = QB del róster activo de esa semana cuyo parte as-of no es Out.
    Se ordena por (no-Doubtful primero, carga reciente desc., partidos previos
    desc.). La carga reciente es ``shifted_rolling`` de los dropbacks de cada fila
    del QB, donde una fila sin estadísticas cuenta 0 (estaba disponible y no jugó)
    salvo que ese día constara como Out (entonces no penaliza: NaN).

    Ojo: ``games.home_qb_id``/``away_qb_id`` NO se usan: en el histórico registran
    al QB que efectivamente jugó, un dato posterior al kickoff.

    Columnas: ``team_qb_epa_per_db_r5`` (EPA de pase / dropbacks en sus últimos
    partidos, ambos desplazados), ``team_qb_passing_yards_r5``,
    ``team_qb_passing_yards_ewm`` y ``team_qb_changed`` (1 si el titular esperado
    no fue quien más pases lanzó en el partido anterior del equipo).
    """
    out_columns = [
        "team_qb_epa_per_db_r5", "team_qb_passing_yards_r5",
        "team_qb_passing_yards_ewm", "team_qb_changed",
    ]
    is_qb = frame["position"].eq("QB").fillna(False).astype(bool)
    if not is_qb.any() or "attempts" not in frame.columns:
        for column in out_columns:
            frame[column] = np.nan
        return frame

    qb = frame.loc[is_qb].sort_values(["gsis_id", "kickoff_utc", "game_id"]).copy()
    load = qb["dropbacks"] if "dropbacks" in qb.columns else qb["attempts"]
    qb["_qb_load"] = np.where(
        qb["_played"], load.fillna(0.0),
        np.where(qb["injury_status_code"] >= OUT_STATUS_CODE, np.nan, 0.0),
    )
    qb["_qb_load_r3"] = shifted_rolling(qb, ["gsis_id"], "_qb_load", 3)
    qb["_doubtful"] = qb["injury_status_code"].ge(DOUBTFUL_STATUS_CODE)

    candidates = qb[qb["injury_status_code"] < OUT_STATUS_CODE].sort_values(
        ["game_id", "team", "_doubtful", "_qb_load_r3", "games_played_prior", "gsis_id"],
        ascending=[True, True, True, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    starter = candidates.drop_duplicates(subset=["game_id", "team"]).copy()

    epa = starter.get(f"passing_epa_r{window}", pd.Series(np.nan, index=starter.index))
    dropbacks = starter.get(f"dropbacks_r{window}", pd.Series(np.nan, index=starter.index))
    starter["team_qb_epa_per_db_r5"] = np.where(dropbacks > 0, epa / dropbacks, np.nan)
    starter["team_qb_passing_yards_r5"] = starter.get(f"passing_yards_r{window}", np.nan)
    starter["team_qb_passing_yards_ewm"] = starter.get("passing_yards_ewm", np.nan)

    # Quién más pases lanzó en el partido ANTERIOR del equipo (rezago de 1).
    leaders = player_stats[["game_id", "team", "player_id", "attempts"]].dropna(subset=["attempts"])
    leaders = leaders[leaders["attempts"] > 0].sort_values(
        ["game_id", "team", "attempts", "player_id"], ascending=[True, True, False, True]
    ).drop_duplicates(subset=["game_id", "team"])
    sequence = long_schedule[["game_id", "team", "kickoff_utc"]].sort_values(
        ["team", "kickoff_utc", "game_id"]
    )
    sequence = sequence.merge(
        leaders[["game_id", "team", "player_id"]], on=["game_id", "team"], how="left"
    )
    sequence["_prev_leader"] = sequence.groupby("team", sort=False)["player_id"].shift(1)
    starter = starter.merge(
        sequence[["game_id", "team", "_prev_leader"]], on=["game_id", "team"], how="left"
    )
    starter["team_qb_changed"] = np.where(
        starter["_prev_leader"].isna(), np.nan,
        (starter["gsis_id"] != starter["_prev_leader"]).astype(float),
    )

    frame = frame.drop(columns=out_columns, errors="ignore")
    return frame.merge(starter[["game_id", "team", *out_columns]], on=["game_id", "team"], how="left")


def _attach_depth_rank_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    """Rango de profundidad aproximado dentro del grupo posicional del equipo.

    ``rosters_weekly.depth_chart_pos`` solo trae la *posición* del depth chart,
    no el orden, y no hay depth charts fechados en la base. Como sustituto se
    ordena a los jugadores disponibles (parte as-of distinto de Out) por su uso
    previo: ``snap_pct_ewm`` y, para desempatar, objetivos + acarreos EWM. Ambos
    son desplazados, así que el rango no depende del partido objetivo. 1 = el de
    más uso. Los jugadores que constan como Out quedan en NaN.
    """
    group = frame["position"].map(POSITION_GROUP)
    available = frame["injury_status_code"] < OUT_STATUS_CODE
    snap = frame.get("snap_pct_ewm", pd.Series(np.nan, index=frame.index)).fillna(-1.0)
    touches = (
        frame.get("targets_ewm", pd.Series(0.0, index=frame.index)).fillna(0.0)
        + frame.get("carries_ewm", pd.Series(0.0, index=frame.index)).fillna(0.0)
    )
    ranking = pd.DataFrame(
        {
            "game_id": frame["game_id"], "team": frame["team"], "_grp": group,
            "_snap": snap, "_touches": touches, "gsis_id": frame["gsis_id"],
        }
    )[available & group.notna()]
    ranking = ranking.sort_values(
        ["game_id", "team", "_grp", "_snap", "_touches", "gsis_id"],
        ascending=[True, True, True, False, False, True],
        kind="mergesort",
    )
    rank = ranking.groupby(["game_id", "team", "_grp"], sort=False).cumcount() + 1
    frame["depth_rank_proxy"] = rank.reindex(frame.index).astype(float)
    return frame


def _attach_vacated_opportunity(
    frame: pd.DataFrame,
    player_stats: pd.DataFrame,
    long_schedule: pd.DataFrame,
    rosters: pd.DataFrame,
    injuries: pd.DataFrame | None,
    *,
    as_of_offset_hours: float,
    window: int,
) -> pd.DataFrame:
    """Cuota de objetivos/acarreos que dejan libre los compañeros de baja.

    Para cada (equipo, partido) se arma un panel con los jugadores que tuvieron
    uso en ese equipo. Para cada uno, con primitivas desplazadas sobre la
    secuencia de partidos *del equipo*:

    * ``share_r`` = media de su cuota en los partidos que jugó de los últimos
      ``window`` del equipo (``shifted_rolling``),
    * ``recent`` = tuvo uso en alguno de los últimos ``VACATED_RECENT_GAMES``.

    Se le considera baja si, **antes del corte as-of**, su parte de lesiones dice
    Out/IR/PUP/NFI, o si el róster de esa semana del equipo lo tiene fuera del
    activo (RES, CUT, DEV... o ya no figura). INA no cuenta (ver
    ``PRESENT_OR_UNKNOWN_STATUSES``). Si el róster de esa semana todavía no
    existe, la ausencia en el róster no se infiere. Antes de
    ``ROSTER_STATUS_RELIABLE_FROM_SEASON`` solo cuenta el parte de lesiones.

    ``vacated_target_share`` / ``vacated_carry_share`` = suma de ``share_r`` de los
    compañeros recientes que son baja, excluido el propio jugador.
    """
    out_columns = ["vacated_target_share", "vacated_carry_share"]
    share_columns = [c for c in ("target_share", "carry_share") if c in player_stats.columns]
    if player_stats.empty or not share_columns:
        for column in out_columns:
            frame[column] = np.nan
        return frame

    sequence = long_schedule[
        ["game_id", "season", "week", "season_type", "kickoff_utc", "team"]
    ].sort_values(["team", "kickoff_utc", "game_id"]).reset_index(drop=True)
    sequence["_tg"] = sequence.groupby("team", sort=False).cumcount()

    usage = player_stats[["game_id", "team", "player_id", *share_columns]].rename(
        columns={"player_id": "gsis_id"}
    )
    usage = usage[(usage[share_columns].fillna(0) > 0).any(axis=1)]
    usage = usage.drop_duplicates(subset=["game_id", "gsis_id"]).merge(
        sequence[["game_id", "team", "_tg"]], on=["game_id", "team"], how="inner"
    )
    if usage.empty:
        for column in out_columns:
            frame[column] = np.nan
        return frame

    # Panel denso (equipo, jugador) x partidos del equipo entre su primera
    # aparición y VACATED_RECENT_GAMES partidos después de la última. Fuera de ese
    # rango "recent" sería falso de todos modos, así que el recorte no altera
    # ningún valor (y por tanto no introduce dependencia del futuro).
    spans = usage.groupby(["team", "gsis_id"], as_index=False)["_tg"].agg(first="min", last="max")
    team_max = sequence.groupby("team")["_tg"].max()
    spans["end"] = np.minimum(
        spans["last"] + VACATED_RECENT_GAMES, spans["team"].map(team_max)
    )
    lengths = (spans["end"] - spans["first"] + 1).to_numpy()
    panel = spans.loc[spans.index.repeat(lengths), ["team", "gsis_id", "first"]].reset_index(drop=True)
    panel["_tg"] = panel["first"] + panel.groupby(["team", "gsis_id"], sort=False).cumcount()
    panel = panel.drop(columns=["first"]).merge(sequence, on=["team", "_tg"], how="left")
    panel = panel.merge(
        usage[["team", "gsis_id", "_tg", *share_columns]], on=["team", "gsis_id", "_tg"], how="left"
    )
    panel["_used"] = panel[share_columns].notna().any(axis=1).astype(float)
    panel = panel.sort_values(["team", "gsis_id", "_tg"]).reset_index(drop=True)

    group_keys = ["team", "gsis_id"]
    for column in share_columns:
        panel[f"_{column}_r"] = shifted_rolling(panel, group_keys, column, window)
    panel["_recent"] = (
        shifted_rolling(panel, group_keys, "_used", VACATED_RECENT_GAMES).fillna(0) > 0
    )

    # --- ¿es baja confirmada antes del corte? ---
    panel["as_of_ts"] = panel["kickoff_utc"] - pd.Timedelta(hours=as_of_offset_hours)
    panel["_out"] = False
    if injuries is not None and not injuries.empty:
        status = player_injury_status(
            injuries, panel[["game_id", "season", "week", "as_of_ts", "gsis_id"]]
        )
        worst = status.groupby(["game_id", "gsis_id"], as_index=False)["injury_status_code"].max()
        panel = panel.merge(worst, on=["game_id", "gsis_id"], how="left")
        panel["_out"] = panel["injury_status_code"].fillna(0).ge(OUT_STATUS_CODE)
        panel = panel.drop(columns=["injury_status_code"])

    if not rosters.empty and "status" in rosters.columns:
        roster_keys = ["season", "week", "season_type", "team"]
        roster_keys = [k for k in roster_keys if k in rosters.columns]
        weekly = rosters[[*roster_keys, "gsis_id", "status"]].drop_duplicates(
            subset=[*roster_keys, "gsis_id"], keep="last"
        )
        published = weekly[roster_keys].drop_duplicates().assign(_roster_published=True)
        panel = panel.merge(weekly, on=[*roster_keys, "gsis_id"], how="left")
        panel = panel.merge(published, on=roster_keys, how="left")
        off_roster = (
            panel["_roster_published"].notna()
            & ~panel["status"].isin(PRESENT_OR_UNKNOWN_STATUSES).fillna(False).astype(bool)
            & (panel["season"] >= ROSTER_STATUS_RELIABLE_FROM_SEASON)
        )
        panel["_out"] = panel["_out"] | off_roster

    contribution = panel[["game_id", "team", "gsis_id"]].copy()
    labels = {"target_share": "vacated_target_share", "carry_share": "vacated_carry_share"}
    for column in share_columns:
        contribution[labels[column]] = np.where(
            panel["_recent"] & panel["_out"], panel[f"_{column}_r"].fillna(0.0), 0.0
        )
    value_columns = [labels[c] for c in share_columns]
    team_total = contribution.groupby(["game_id", "team"], as_index=False)[value_columns].sum()
    own = contribution.rename(columns={c: f"_own_{c}" for c in value_columns})

    frame = frame.drop(columns=out_columns, errors="ignore")
    frame = frame.merge(team_total, on=["game_id", "team"], how="left")
    frame = frame.merge(own, on=["game_id", "team", "gsis_id"], how="left")
    for column in value_columns:
        frame[column] = frame[column] - frame[f"_own_{column}"].fillna(0.0)
        frame = frame.drop(columns=[f"_own_{column}"])
    for column in out_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


# ================================================= prior para poca muestra


def _attach_low_sample_blend(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """``*_blend``: uso reciente con contracción hacia un prior si hay poca muestra.

    * Con ``games_played_prior >= LOW_SAMPLE_THRESHOLD`` y media reciente
      disponible, ``blend`` = la media desplazada de ``window`` partidos.
    * Si no, ``blend = (n * propia + k * prior) / (n + k)``, con ``propia`` = media
      de carrera desplazada (``shifted_expanding``), ``n`` = observaciones previas
      y ``k = LOW_SAMPLE_THRESHOLD``.

    El ``prior`` es el valor medio en sus primeros partidos de jugadores del mismo
    grupo posicional y tramo de draft (R1, R2, R3, R4+, UDFA), calculado solo con
    **temporadas anteriores** (``shifted_expanding`` sobre la serie por temporada).
    Así un novato no arranca con NaN→0, ni con información de su propia temporada.

    ``frame`` debe venir ordenado por ``gsis_id`` y kickoff.
    """
    k = LOW_SAMPLE_THRESHOLD
    sources = {s: label for s, label in BLEND_METRICS.items() if s in frame.columns}
    if not sources:
        return frame

    group = frame["position"].astype(object).map(POSITION_GROUP).fillna("OTHER")
    if "draft_round" in frame.columns:
        draft_round = pd.to_numeric(frame["draft_round"], errors="coerce").astype("float64").to_numpy()
    else:
        draft_round = np.full(len(frame), np.nan)
    bucket = pd.Series(
        np.select(
            [draft_round == 1, draft_round == 2, draft_round == 3, draft_round >= 4],
            ["R1", "R2", "R3", "R4+"],
            default="UDFA",
        ),
        index=frame.index,
    )
    work = pd.DataFrame({"season": frame["season"], "_grp": group, "_bucket": bucket})

    # Pool del prior: primeros partidos de carrera (los veteranos que aparecen
    # en la primera temporada de datos con pocas filas no son novatos).
    early = frame["_played"] & (frame["games_played_prior"] < k)
    if "rookie_season" in frame.columns:
        rookie_season = pd.to_numeric(frame["rookie_season"], errors="coerce").astype("float64")
        early &= rookie_season.isna() | ((frame["season"] - rookie_season) <= 1)
    pool = pd.concat([work, frame[list(sources)]], axis=1)[early]

    seasons = sorted(frame["season"].dropna().unique())
    cells = work[["_grp", "_bucket"]].drop_duplicates()
    grid = cells.merge(pd.DataFrame({"season": seasons}), how="cross")
    season_level = grid.merge(
        pool.groupby(["_grp", "_bucket", "season"], as_index=False)[list(sources)].mean(),
        on=["_grp", "_bucket", "season"],
        how="left",
    ).sort_values(["_grp", "_bucket", "season"]).reset_index(drop=True)
    priors = season_level[["_grp", "_bucket", "season"]].copy()
    for source in sources:
        priors[f"_prior_{source}"] = shifted_expanding(season_level, ["_grp", "_bucket"], source)

    looked_up = work.merge(priors, on=["_grp", "_bucket", "season"], how="left")
    looked_up.index = frame.index

    games_prior = frame["games_played_prior"].astype(float)
    for source, label in sources.items():
        own = shifted_expanding(frame, ["gsis_id"], source)
        n_obs = (
            frame.groupby("gsis_id", sort=False)[source]
            .transform(lambda s: s.notna().shift(1, fill_value=False).cumsum())
            .astype(float)
        )
        prior = looked_up[f"_prior_{source}"]
        shrunk = np.where(
            own.notna(),
            (n_obs * own + k * prior.fillna(own)) / (n_obs + k),
            prior,
        )
        recent = frame.get(f"{label}_r{window}", pd.Series(np.nan, index=frame.index))
        frame[f"{label}_blend"] = np.where(
            (games_prior >= k) & recent.notna(), recent, shrunk
        )
    return frame
