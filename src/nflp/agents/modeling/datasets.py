"""Construcción de datasets de entrenamiento.

Une el feature store con los resultados reales y produce ``(X, y)`` listos para
entrenar, aplicando siempre el contrato anti-leakage antes de devolver nada.

Punto crítico: los **cortes son temporales**, nunca aleatorios. Un `train_test_split`
aleatorio sobre datos de series temporales mezcla partidos de 2024 en el
entrenamiento y de 2015 en la validación, produce métricas excelentes y un modelo
inservible. Aquí todos los cortes se hacen por (temporada, semana).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nflp.agents.features.contracts import assert_no_leakage, feature_columns
from nflp.core.db import Database, get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

# Definición de cada objetivo: cómo se calcula y de qué tipo es.
TARGET_SPECS: dict[str, dict[str, str]] = {
    "winner": {"kind": "classification", "metric": "log_loss"},
    "spread": {"kind": "regression", "metric": "mae"},
    "total": {"kind": "regression", "metric": "mae"},
}


@dataclass
class Dataset:
    """Matriz de features y objetivo, con su metadata de trazabilidad."""

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame                     # game_id, season, week, equipos, kickoff
    target: str
    kind: str
    feature_names: list[str] = field(default_factory=list)
    dataset_hash: str = ""

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(self.X.columns)
        if not self.dataset_hash:
            self.dataset_hash = compute_dataset_hash(self.X, self.y)

    def __len__(self) -> int:
        return len(self.X)

    def slice(self, mask: pd.Series) -> Dataset:
        """Subconjunto por máscara booleana, preservando la alineación."""
        return Dataset(
            X=self.X.loc[mask],
            y=self.y.loc[mask],
            meta=self.meta.loc[mask],
            target=self.target,
            kind=self.kind,
            feature_names=self.feature_names,
        )


def compute_dataset_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """Hash reproducible del dataset.

    Permite verificar que un backtest es reproducible: mismo hash implica mismos
    datos de entrada, así que las métricas deben coincidir exactamente.
    """
    hasher = hashlib.sha256()
    hasher.update(",".join(sorted(X.columns)).encode("utf-8"))
    hasher.update(str(X.shape).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(X, index=False).values.tobytes())
    hasher.update(pd.util.hash_pandas_object(y, index=False).values.tobytes())
    return hasher.hexdigest()[:16]


def build_target(games: pd.DataFrame, target: str) -> pd.Series:
    """Calcula la variable objetivo a partir de los resultados."""
    home = pd.to_numeric(games["home_score"], errors="coerce")
    away = pd.to_numeric(games["away_score"], errors="coerce")

    if target == "winner":
        # Los empates (muy raros en la NFL) se descartan aguas arriba.
        return (home > away).astype("float64").where(home.notna() & away.notna())
    if target == "spread":
        return home - away
    if target == "total":
        return home + away
    raise ValueError(f"Objetivo desconocido: {target}")


def load_game_dataset(
    target: str,
    *,
    db: Database | None = None,
    feature_version: str | None = None,
    seasons: list[int] | None = None,
    require_result: bool = True,
    min_prior_games: int = 3,
) -> Dataset:
    """Carga el dataset de un objetivo a nivel de partido.

    Args:
        require_result: si es ``True`` solo devuelve partidos ya jugados
            (entrenamiento); si es ``False``, incluye los futuros (predicción).
        min_prior_games: descarta partidos en los que algún equipo tiene menos
            historia que este mínimo — sus medias móviles son puro ruido.
    """
    if target not in TARGET_SPECS:
        raise ValueError(f"Objetivo desconocido: {target}. Válidos: {list(TARGET_SPECS)}")

    db = db or get_db()
    version = feature_version or "v1"

    clause = [f"f.feature_set_version = '{version}'"]
    if seasons:
        clause.append(f"g.season IN ({', '.join(str(s) for s in seasons)})")
    if require_result:
        clause.append("g.home_score IS NOT NULL AND g.away_score IS NOT NULL")

    frame = db.query(
        f"""
        SELECT f.*, g.kickoff_utc, g.home_score, g.away_score
        FROM features_game f
        JOIN games g USING (game_id)
        WHERE {' AND '.join(clause)}
        ORDER BY g.kickoff_utc, f.game_id
        """
    )
    if frame.empty:
        raise ValueError(
            f"Sin datos para el objetivo '{target}'. "
            "¿Se ejecutó `nflp features build`?"
        )

    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], errors="coerce")
    y = build_target(frame, target)

    # Empates: sin ganador que predecir. Se exige que ambos marcadores existan:
    # en un partido aún no jugado los dos son NULL, y `NULL == NULL` da NA, lo
    # que descartaría justamente los partidos que queremos predecir.
    if target == "winner":
        both_played = frame["home_score"].notna() & frame["away_score"].notna()
        ties = (frame["home_score"].eq(frame["away_score"]) & both_played).fillna(False)
        if ties.any():
            log.debug("Descartados %d empates.", int(ties.sum()))
            frame, y = frame.loc[~ties], y.loc[~ties]

    if require_result:
        valid = y.notna()
        frame, y = frame.loc[valid], y.loc[valid]

    # Partidos sin historia suficiente: las medias móviles no son informativas.
    if min_prior_games > 0 and "home_epa_off_r3" in frame.columns:
        has_history = frame["home_epa_off_r3"].notna() & frame["away_epa_off_r3"].notna()
        dropped = int((~has_history).sum())
        if dropped:
            log.debug("Descartados %d partidos sin historia suficiente.", dropped)
        frame, y = frame.loc[has_history], y.loc[has_history]

    meta_columns = [
        "game_id", "season", "week", "season_type",
        "home_team", "away_team", "kickoff_utc", "as_of_ts",
    ]
    meta = frame[[c for c in meta_columns if c in frame.columns]].copy()

    # `frame` lleva a propósito home_score/away_score: son la materia prima del
    # objetivo. La matriz de features se construye excluyéndolas, y la auditoría
    # se ejecuta sobre X —lo que realmente ve el modelo—, no sobre `frame`.
    names = feature_columns(frame)
    names = [
        n for n in names
        if n not in {"home_score", "away_score", "season", "week", "home_team", "away_team",
                     "season_type", "weather_source", "week_phase"}
    ]
    X = frame[names].copy()
    X = _encode_features(X)

    # --- CONTRATO ANTI-LEAKAGE: se verifica lo que entra al modelo ---
    # `as_of_ts` se añade solo para la auditoría y se retira acto seguido: es
    # metadato de control, nunca una feature.
    audit_frame = X.copy()
    audit_frame["as_of_ts"] = frame["as_of_ts"].to_numpy()
    assert_no_leakage(audit_frame, kickoff=frame["kickoff_utc"])

    log.info(
        "Dataset '%s': %d filas, %d features (temporadas %s-%s).",
        target, len(X), len(X.columns),
        meta["season"].min() if not meta.empty else "?",
        meta["season"].max() if not meta.empty else "?",
    )

    return Dataset(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True).astype(float),
        meta=meta.reset_index(drop=True),
        target=target,
        kind=TARGET_SPECS[target]["kind"],
    )


def _encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convierte booleanos a enteros y deja el resto en float.

    Los GBM manejan los NaN de forma nativa, así que **no se imputa**: un valor
    ausente (no hay dato de clima, no hay historia) es información en sí mismo, y
    rellenarlo con la media lo destruiría.
    """
    encoded = X.copy()
    for column in encoded.columns:
        series = encoded[column]
        if series.dtype == bool or str(series.dtype) == "boolean":
            encoded[column] = series.astype("float64")
        elif series.dtype == object or str(series.dtype) == "string":
            # Categóricas residuales: se codifican por frecuencia de aparición.
            encoded[column] = pd.factorize(series)[0].astype("float64")
            encoded.loc[series.isna(), column] = np.nan
        else:
            encoded[column] = pd.to_numeric(series, errors="coerce").astype("float64")
    return encoded


def temporal_split(
    dataset: Dataset, *, val_seasons: int = 2
) -> tuple[Dataset, Dataset]:
    """Corte temporal train/validación por temporadas completas.

    Las últimas ``val_seasons`` temporadas quedan para validación. Nunca aleatorio.
    """
    seasons = sorted(dataset.meta["season"].unique())
    if len(seasons) <= val_seasons:
        raise ValueError(
            f"Se necesitan más de {val_seasons} temporadas para el corte temporal "
            f"(hay {len(seasons)})."
        )

    cutoff = seasons[-val_seasons]
    is_val = dataset.meta["season"] >= cutoff

    log.debug(
        "Corte temporal: entrenamiento < %d (%d filas), validación >= %d (%d filas).",
        cutoff, int((~is_val).sum()), cutoff, int(is_val.sum()),
    )
    return dataset.slice(~is_val), dataset.slice(is_val)


def temporal_split_three_way(
    dataset: Dataset, *, calib_seasons: int = 1, val_seasons: int = 1
) -> tuple[Dataset, Dataset, Dataset]:
    """Corte temporal a tres bandas: entrenamiento / calibración / validación.

    **Por qué hacen falta tres y no dos.** El calibrador debe ajustarse sobre
    predicciones que el modelo no vio al entrenar. Pero si además se *evalúa*
    sobre ese mismo conjunto, la métrica sale optimista: el calibrador ya se
    adaptó a esos datos concretos. Se necesita un tercer bloque, posterior a
    ambos, para medir de forma honesta.

    Los tres bloques son consecutivos en el tiempo:
    ``[... entrenamiento ...] [calibración] [validación]``
    """
    seasons = sorted(dataset.meta["season"].unique())
    needed = calib_seasons + val_seasons + 1
    if len(seasons) < needed:
        raise ValueError(
            f"Se necesitan al menos {needed} temporadas para el corte a tres bandas "
            f"(hay {len(seasons)})."
        )

    val_cutoff = seasons[-val_seasons]
    calib_cutoff = seasons[-(val_seasons + calib_seasons)]

    season_series = dataset.meta["season"]
    is_val = season_series >= val_cutoff
    is_calib = (season_series >= calib_cutoff) & (~is_val)
    is_train = ~(is_val | is_calib)

    log.debug(
        "Corte a tres bandas — entrenamiento: <%d (%d filas) · calibración: %d-%d (%d) · "
        "validación: >=%d (%d).",
        calib_cutoff, int(is_train.sum()), calib_cutoff, val_cutoff - 1, int(is_calib.sum()),
        val_cutoff, int(is_val.sum()),
    )
    return dataset.slice(is_train), dataset.slice(is_calib), dataset.slice(is_val)


def season_folds(dataset: Dataset, n_splits: int = 5) -> list[tuple[pd.Series, pd.Series]]:
    """Validación cruzada temporal: k-fold **por temporada**, con ventana expansiva.

    Cada fold entrena con todas las temporadas anteriores y valida en la
    siguiente. Es el equivalente honesto de un k-fold para series temporales:
    nunca se entrena con el futuro.
    """
    seasons = sorted(dataset.meta["season"].unique())
    if len(seasons) < n_splits + 1:
        n_splits = max(1, len(seasons) - 1)

    folds: list[tuple[pd.Series, pd.Series]] = []
    validation_seasons = seasons[-n_splits:]

    for season in validation_seasons:
        train_mask = dataset.meta["season"] < season
        val_mask = dataset.meta["season"] == season
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append((train_mask, val_mask))

    log.debug("Validación cruzada temporal: %d folds por temporada.", len(folds))
    return folds
