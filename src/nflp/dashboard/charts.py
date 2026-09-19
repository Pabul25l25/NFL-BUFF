"""Gráficos del dashboard.

Paleta sobria y consistente, legible en tema claro y oscuro. Los colores
codifican significado, no decoran: verde para lo que favorece al local o mejora,
rojo para lo contrario, gris para las referencias.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Paleta con contraste suficiente en ambos temas.
POSITIVE = "#2E8B57"
NEGATIVE = "#C1444A"
NEUTRAL = "#8A8A93"
ACCENT = "#3B6EA5"
MARKET = "#B5892E"
GRID = "rgba(128,128,128,0.18)"

_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"size": 13},
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
    "hoverlabel": {"font_size": 12},
}

# Nombres legibles de las features en los gráficos SHAP.
_LABELS = {
    "net_epa_diff": "EPA neto (dif.)",
    "epa_off_diff": "EPA ofensivo (dif.)",
    "epa_def_diff": "EPA defensivo (dif.)",
    "elo_diff": "Elo (dif.)",
    "home_elo_pre": "Elo local",
    "away_elo_pre": "Elo visitante",
    "margin_diff_r5": "Margen 5 part. (dif.)",
    "rest_advantage": "Ventaja de descanso",
    "away_travel_km": "Viaje del visitante",
    "wind_kph": "Viento",
    "temperature_c": "Temperatura",
    "injury_index_diff": "Lesiones (dif.)",
    "pass_matchup_diff": "Matchup aéreo",
    "rush_matchup_diff": "Matchup terrestre",
    "home_margin_r5": "Margen local 5 part.",
    "away_margin_r5": "Margen visitante 5 part.",
    "home_points_against_r5": "Puntos encajados local",
    "away_points_against_r5": "Puntos encajados visitante",
    "target_share_r5": "Cuota de objetivos",
    "snap_pct_r5": "% de snaps",
    "carry_share_r5": "Cuota de acarreos",
    "opp_pass_epa_def_r5": "Defensa aérea rival",
    "opp_rush_epa_def_r5": "Defensa terrestre rival",
}


def _label(feature: str) -> str:
    return _LABELS.get(feature, feature.replace("_", " "))


def shap_chart(
    shap_values: dict[str, float], positive_label: str, negative_label: str, top_n: int = 8
) -> go.Figure:
    """Barras horizontales con las contribuciones SHAP más relevantes."""
    ranked = sorted(shap_values.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    ranked.reverse()  # Plotly dibuja de abajo arriba.

    features = [_label(f) for f, _ in ranked]
    values = [v for _, v in ranked]
    colours = [POSITIVE if v > 0 else NEGATIVE for v in values]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Contribuciones — verde favorece a {positive_label}, rojo a {negative_label}",
        xaxis_title="Valor SHAP",
        height=max(260, 34 * len(ranked) + 90),
        showlegend=False,
        **_LAYOUT,
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=NEUTRAL)
    figure.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return figure


def calibration_chart(frame: pd.DataFrame) -> go.Figure:
    """Curva de fiabilidad: probabilidad declarada vs frecuencia observada."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
            line={"dash": "dash", "color": NEUTRAL, "width": 1.5},
            hoverinfo="skip",
        )
    )

    colours = {"modelo": ACCENT, "baseline_elo": POSITIVE, "baseline_mercado": MARKET}
    names = {
        "modelo": "Nuestro modelo",
        "baseline_elo": "Elo",
        "baseline_mercado": "Mercado (cierre)",
    }

    for label, group in frame.groupby("model_label"):
        group = group.sort_values("mean_predicted")
        figure.add_trace(
            go.Scatter(
                x=group["mean_predicted"],
                y=group["observed_frequency"],
                mode="lines+markers",
                name=names.get(str(label), str(label)),
                line={"color": colours.get(str(label), NEUTRAL), "width": 2.5},
                marker={"size": 8},
                customdata=group["n_samples"],
                hovertemplate=(
                    "Predicho: %{x:.1%}<br>Observado: %{y:.1%}"
                    "<br>n = %{customdata}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    figure.update_layout(
        title="Curva de calibración — victoria local",
        xaxis_title="Probabilidad predicha",
        yaxis_title="Frecuencia observada",
        height=480,
        legend={"orientation": "h", "y": -0.18},
        **_LAYOUT,
    )
    figure.update_xaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID)
    figure.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID)
    return figure


def performance_chart(frame: pd.DataFrame, target: str) -> go.Figure:
    """Evolución semanal de la métrica del modelo en producción."""
    metric = "log_loss" if target == "winner" else "mae"
    if metric not in frame.columns or frame[metric].isna().all():
        metric = "mae" if "mae" in frame.columns else "log_loss"

    frame = frame.dropna(subset=[metric]).copy()
    frame["etiqueta"] = frame["season"].astype(str) + "-S" + frame["week"].astype(str)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["etiqueta"], y=frame[metric],
            mode="lines+markers", name=metric,
            line={"color": ACCENT, "width": 2.5}, marker={"size": 7},
            hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
        )
    )

    # Media móvil de 4 semanas: la señal semanal es demasiado ruidosa para
    # detectar degradación a simple vista.
    if len(frame) >= 4:
        rolling = frame[metric].rolling(4, min_periods=2).mean()
        figure.add_trace(
            go.Scatter(
                x=frame["etiqueta"], y=rolling,
                mode="lines", name="Media móvil (4 sem.)",
                line={"color": NEGATIVE, "width": 2, "dash": "dot"},
                hovertemplate="%{x}<br>media: %{y:.4f}<extra></extra>",
            )
        )

    figure.update_layout(
        title=f"Degradación en producción — {metric}",
        xaxis_title="Temporada-semana",
        yaxis_title=metric,
        height=420,
        legend={"orientation": "h", "y": -0.22},
        **_LAYOUT,
    )
    figure.update_xaxes(gridcolor=GRID)
    figure.update_yaxes(gridcolor=GRID)
    return figure


def comparison_chart(overall: pd.DataFrame, metric: str) -> go.Figure:
    """Barras comparando el modelo contra cada baseline."""
    frame = overall.dropna(subset=[metric]).sort_values(metric)
    colours = [
        ACCENT if label == "modelo"
        else MARKET if label == "baseline_mercado"
        else NEUTRAL
        for label in frame["model_label"]
    ]

    figure = go.Figure(
        go.Bar(
            x=frame["model_label"], y=frame[metric],
            marker_color=colours,
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{metric} — menor es mejor",
        yaxis_title=metric,
        height=380,
        showlegend=False,
        **_LAYOUT,
    )
    figure.update_yaxes(gridcolor=GRID)
    return figure
