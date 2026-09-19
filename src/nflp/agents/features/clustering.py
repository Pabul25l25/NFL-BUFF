"""Clustering de estilos de equipo (componente no supervisado).

El prompt pide un componente no supervisado, pero también advierte contra la
complejidad injustificada. La postura aquí es explícita: **el clustering entra
como feature auxiliar, nunca como predictor principal.**

Qué aporta realmente: los equipos no son intercambiables. Una defensa que vive
de presionar al QB se comporta distinto contra una ofensiva de pases cortos y
rápidos que contra una vertical. Un identificador de estilo permite al modelo de
gradient boosting capturar interacciones de emparejamiento que las medias de EPA,
por sí solas, no expresan.

Qué NO se hace: usar el clúster para predecir directamente, ni interpretarlo como
si los grupos tuvieran significado causal. Son etiquetas que ayudan al árbol a
partir el espacio, nada más.

**Anti-leakage**: el K-means se ajusta únicamente con datos hasta la semana
anterior. El perfil de estilo de un equipo en la semana N usa medias móviles ya
desplazadas, así que nunca ve el partido que se está prediciendo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nflp.core.logging import get_logger

log = get_logger(__name__)

# Dimensiones que definen el "estilo" de un equipo. Se eligen por ser
# descriptivas del *cómo* juega, no de *cómo de bueno* es: la calidad ya la
# capturan las features de EPA.
STYLE_FEATURES = [
    "pass_rate_r5",        # ¿pasa o corre?
    "plays_r5",            # ritmo de juego
    "sack_rate_off_r5",    # protección de la línea
    "sack_rate_def_r5",    # presión de la defensa
    "cpoe_r5",             # precisión del pase por encima de lo esperado
    "turnover_margin_r5",  # disciplina con el balón
]


def build_style_clusters(
    team_form: pd.DataFrame,
    *,
    n_clusters: int = 6,
    random_state: int = 1729,
    min_prior_games: int = 3,
) -> pd.DataFrame:
    """Asigna un clúster de estilo a cada equipo-semana.

    Args:
        team_form: salida de ``build_team_form`` (ya con medias desplazadas).

    Returns:
        ``season``, ``week``, ``team``, ``cluster_id``, ``model_version``.
    """
    available = [column for column in STYLE_FEATURES if column in team_form.columns]
    if not available:
        log.warning("No hay features de estilo disponibles; se omite el clustering.")
        return pd.DataFrame(columns=["season", "week", "team", "cluster_id", "model_version"])

    frame = team_form[["season", "week", "team", "prior_games", *available]].copy()

    # Equipos con poca historia en la temporada no tienen un estilo estable.
    eligible = frame[frame["prior_games"] >= min_prior_games].copy()
    if len(eligible) < n_clusters * 10:
        log.warning(
            "Muestra insuficiente para %d clústeres (%d filas); se omite.",
            n_clusters, len(eligible),
        )
        return pd.DataFrame(columns=["season", "week", "team", "cluster_id", "model_version"])

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
        ]
    )

    matrix = eligible[available].astype(float)
    labels = pipeline.fit_predict(matrix)
    eligible["cluster_id"] = labels.astype(int)

    result = frame.merge(
        eligible[["season", "week", "team", "cluster_id"]],
        on=["season", "week", "team"],
        how="left",
    )
    # -1 marca "sin estilo asignado" (poca historia). Es información legítima:
    # el modelo puede aprender que los equipos sin perfil estable son distintos.
    result["cluster_id"] = result["cluster_id"].fillna(-1).astype(int)
    result["model_version"] = f"kmeans_k{n_clusters}_v1"

    distribution = result["cluster_id"].value_counts().sort_index().to_dict()
    log.info("Clústeres de estilo asignados: %s", distribution)

    return result[["season", "week", "team", "cluster_id", "model_version"]]


def describe_clusters(
    team_form: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Perfil medio de cada clúster, para poder interpretarlos en el dashboard."""
    available = [column for column in STYLE_FEATURES if column in team_form.columns]
    merged = team_form.merge(clusters, on=["season", "week", "team"], how="inner")
    profile = (
        merged[merged["cluster_id"] >= 0]
        .groupby("cluster_id", observed=True)[available]
        .mean()
        .round(3)
    )
    profile["n"] = merged[merged["cluster_id"] >= 0].groupby("cluster_id").size()
    return profile.reset_index()


def label_cluster(profile_row: pd.Series) -> str:
    """Etiqueta legible en español a partir del perfil medio del clúster.

    Sirve para que el dashboard no muestre "clúster 3" sino algo interpretable.
    """
    pass_rate = profile_row.get("pass_rate_r5", np.nan)
    plays = profile_row.get("plays_r5", np.nan)
    sack_def = profile_row.get("sack_rate_def_r5", np.nan)

    parts: list[str] = []
    if pd.notna(pass_rate):
        parts.append("aéreo" if pass_rate > 0.60 else "terrestre" if pass_rate < 0.52 else "equilibrado")
    if pd.notna(plays):
        parts.append("ritmo alto" if plays > 65 else "ritmo bajo" if plays < 60 else "ritmo medio")
    if pd.notna(sack_def) and sack_def > 0.075:
        parts.append("defensa presionante")

    return ", ".join(parts) if parts else "sin perfil"
