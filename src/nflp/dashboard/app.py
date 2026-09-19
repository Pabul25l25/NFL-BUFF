"""Dashboard de NFL-Predictor (Streamlit).

Cubre los seis requisitos de la sección 7 del prompt:

1. Predicción principal con probabilidad y nivel de confianza.
2. **Justificación explicable por predicción** (SHAP traducido a español).
3. Comparación predicción vs resultado real, con métricas acumuladas.
4. Vista de calibración y degradación del modelo en producción.
5. Filtros por equipo, semana, temporada y tipo de predicción.
6. Historial de versiones de modelo (trazabilidad completa).

Todo se lee de la capa de datos; no hay ningún número hardcodeado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nflp.dashboard import data as dd
from nflp.dashboard.charts import (
    calibration_chart,
    performance_chart,
    shap_chart,
)

st.set_page_config(
    page_title="NFL-Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET_LABELS = {
    "winner": "Ganador",
    "spread": "Spread (margen)",
    "total": "Total (over/under)",
}


def main() -> None:
    st.title("🏈 NFL-Predictor")
    st.caption(
        "Predicciones con backtesting walk-forward y justificación explicable. "
        "Las líneas del mercado se usan solo como benchmark, nunca como features."
    )

    if not dd.has_data():
        st.error(
            "No hay datos en la base. Ejecuta primero:\n\n"
            "```\nuv run nflp db init\n"
            "uv run nflp data ingest\n"
            "uv run nflp features build\n"
            "uv run nflp model train --target all\n```"
        )
        return

    filters = dd.available_filters()

    with st.sidebar:
        st.header("Filtros")
        season = st.selectbox(
            "Temporada", options=filters["seasons"], index=0 if filters["seasons"] else None
        )
        week_options = ["Todas"] + [str(w) for w in filters["weeks"]]
        week_choice = st.selectbox("Semana", options=week_options, index=0)
        week = None if week_choice == "Todas" else int(week_choice)

        team_options = ["Todos"] + filters["teams"]
        team_choice = st.selectbox("Equipo", options=team_options, index=0)
        team = None if team_choice == "Todos" else team_choice

        st.divider()
        st.caption("Estado de la base de datos")
        for name, rows in dd.database_status().items():
            st.caption(f"{name}: **{rows:,}**")

    tabs = st.tabs(
        [
            "Predicciones",
            "Resultados y aciertos",
            "Calibración",
            "Rendimiento en producción",
            "Props de jugador",
            "Modelos y trazabilidad",
        ]
    )

    with tabs[0]:
        render_predictions(season, week, team)
    with tabs[1]:
        render_results(season, week, team)
    with tabs[2]:
        render_calibration()
    with tabs[3]:
        render_production()
    with tabs[4]:
        render_props(season, week)
    with tabs[5]:
        render_registry()


# ============================================================== predicciones


def render_predictions(season: int | None, week: int | None, team: str | None) -> None:
    st.subheader("Predicciones por partido")

    target = st.radio(
        "Tipo de predicción",
        options=list(TARGET_LABELS),
        format_func=lambda t: TARGET_LABELS[t],
        horizontal=True,
        key="pred_target",
    )

    frame = dd.load_predictions(target, season, week, team)
    if frame.empty:
        st.info(
            "No hay predicciones para este filtro. "
            "Genera algunas con `uv run nflp predict --season "
            f"{season or 2026}`."
        )
        return

    upcoming = frame[frame["home_score"].isna()]
    played = frame[frame["home_score"].notna()]

    if not upcoming.empty:
        st.markdown(f"**Próximos partidos** ({len(upcoming)})")
        _render_prediction_cards(upcoming, target)

    if not played.empty:
        with st.expander(f"Partidos ya jugados ({len(played)})", expanded=upcoming.empty):
            _render_prediction_cards(played.tail(20), target, show_actual=True)


def _render_prediction_cards(
    frame: pd.DataFrame, target: str, *, show_actual: bool = False
) -> None:
    """Tarjeta por partido con predicción, confianza y justificación SHAP."""
    for _, row in frame.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        header = f"**{matchup}** · semana {row['week']}"

        with st.container(border=True):
            columns = st.columns([2.2, 1.1, 1.1, 1.3])

            with columns[0]:
                st.markdown(header)
                stadium = row.get("stadium") or "—"
                weather_bits = []
                if pd.notna(row.get("temperature_c")):
                    weather_bits.append(f"{row['temperature_c']:.0f} °C")
                if pd.notna(row.get("wind_kph")):
                    weather_bits.append(f"viento {row['wind_kph']:.0f} km/h")
                suffix = f" · {', '.join(weather_bits)}" if weather_bits else ""
                st.caption(f"{stadium}{suffix}")

            with columns[1]:
                st.metric("Predicción", _format_prediction(row, target))

            with columns[2]:
                probability = row.get("probability")
                if pd.notna(probability):
                    shown = max(probability, 1 - probability) if target == "winner" else probability
                    st.metric("Probabilidad", f"{shown:.1%}")
                elif pd.notna(row.get("lower_80")):
                    st.metric("Rango 80%", f"{row['lower_80']:.0f}–{row['upper_80']:.0f}")

            with columns[3]:
                confidence = str(row.get("confidence_level") or "—")
                colour = {"alta": "🟢", "media": "🟡", "baja": "🔴"}.get(confidence, "⚪")
                st.metric("Confianza", f"{colour} {confidence}")
                if show_actual and pd.notna(row.get("home_score")):
                    st.caption(
                        f"Real: {row['away_team']} {row['away_score']:.0f} – "
                        f"{row['home_score']:.0f} {row['home_team']}"
                    )

            explanation = row.get("explanation_es")
            if explanation:
                st.markdown(f"💡 {explanation}")

            shap_values = dd.parse_shap(row.get("shap_values"))
            if shap_values:
                with st.expander("Ver contribuciones SHAP"):
                    st.plotly_chart(
                        shap_chart(shap_values, row["home_team"], row["away_team"]),
                        width="stretch",
                        key=f"shap_{row['prediction_id']}",
                    )

            model_version = row.get("model_version")
            if pd.notna(model_version):
                st.caption(
                    f"Modelo {row.get('model_family')} v{int(model_version)} · "
                    f"predicho el {pd.to_datetime(row['predicted_at']):%d/%m/%Y %H:%M} · "
                    f"corte as-of {pd.to_datetime(row['as_of_ts']):%d/%m %H:%M}"
                )


def _format_prediction(row: pd.Series, target: str) -> str:
    if target == "winner":
        probability = row.get("probability")
        if pd.isna(probability):
            return "—"
        return f"{row['home_team']}" if probability >= 0.5 else f"{row['away_team']}"
    if target == "spread":
        return f"{row['point_estimate']:+.1f}"
    return f"{row['point_estimate']:.1f}"


# ========================================================== resultados reales


def render_results(season: int | None, week: int | None, team: str | None) -> None:
    st.subheader("Predicción vs. resultado real")

    target = st.radio(
        "Tipo de predicción",
        options=list(TARGET_LABELS),
        format_func=lambda t: TARGET_LABELS[t],
        horizontal=True,
        key="results_target",
    )

    frame = dd.load_predictions(target, season, week, team)
    frame = frame[frame["home_score"].notna()]

    if frame.empty:
        st.info("Todavía no hay partidos jugados con predicción registrada.")
        return

    if target == "winner":
        frame["acierto"] = (
            (frame["probability"] >= 0.5) == (frame["home_score"] > frame["away_score"])
        )
        accuracy = frame["acierto"].mean()
        brier = float(
            np.mean(
                (frame["probability"] - (frame["home_score"] > frame["away_score"]).astype(float))
                ** 2
            )
        )
        columns = st.columns(3)
        columns[0].metric("Aciertos", f"{accuracy:.1%}")
        columns[1].metric("Brier score", f"{brier:.4f}")
        columns[2].metric("Partidos", f"{len(frame):,}")
        display = frame[
            ["week", "away_team", "home_team", "probability", "away_score", "home_score", "acierto"]
        ].rename(
            columns={
                "week": "Sem", "away_team": "Visitante", "home_team": "Local",
                "probability": "P(local)", "away_score": "Pts vis.",
                "home_score": "Pts loc.", "acierto": "Acierto",
            }
        )
    else:
        actual = frame["actual_margin"] if target == "spread" else frame["actual_total"]
        frame["error"] = frame["point_estimate"] - actual
        mae = frame["error"].abs().mean()
        columns = st.columns(3)
        columns[0].metric("MAE", f"{mae:.2f}")
        columns[1].metric("Sesgo medio", f"{frame['error'].mean():+.2f}")
        columns[2].metric("Partidos", f"{len(frame):,}")
        frame["real"] = actual
        display = frame[
            ["week", "away_team", "home_team", "point_estimate", "real", "error"]
        ].rename(
            columns={
                "week": "Sem", "away_team": "Visitante", "home_team": "Local",
                "point_estimate": "Predicción", "real": "Real", "error": "Error",
            }
        )

    st.dataframe(display, width="stretch", hide_index=True)

    # --- comparación contra baselines del último backtest ---
    backtest = dd.load_backtest_results(target)
    if not backtest.empty:
        overall = backtest[backtest["segment_type"] == "overall"]
        if not overall.empty:
            st.markdown("**Backtest walk-forward — comparación contra baselines**")
            columns_to_show = ["model_label", "n_games", "log_loss", "brier_score", "ece",
                               "accuracy", "mae", "rmse", "ats_win_pct"]
            available = [c for c in columns_to_show if c in overall.columns]
            st.dataframe(
                overall[available].dropna(axis=1, how="all").round(4),
                width="stretch", hide_index=True,
            )
            st.caption(
                "El benchmark real es `baseline_mercado` (línea de cierre). "
                "Superar a `baseline_elo` y `baseline_naive_local` es el mínimo exigible."
            )


# ============================================================== calibración


def render_calibration() -> None:
    st.subheader("Calibración del modelo")
    st.caption(
        "Si el modelo dice «70 %», debería acertar el 70 % de las veces. "
        "La diagonal es la calibración perfecta."
    )

    frame = dd.load_calibration("winner")
    if frame.empty:
        st.info(
            "Sin datos de calibración. Ejecuta "
            "`uv run nflp backtest run --target winner`."
        )
        return

    st.plotly_chart(calibration_chart(frame), width="stretch")

    backtest = dd.load_backtest_results("winner")
    if not backtest.empty:
        overall = backtest[backtest["segment_type"] == "overall"]
        if not overall.empty and "ece" in overall.columns:
            st.markdown("**Error de calibración esperado (ECE) — menor es mejor**")
            st.dataframe(
                overall[["model_label", "ece", "log_loss", "brier_score"]].round(4),
                width="stretch", hide_index=True,
            )

        segments = backtest[backtest["segment_type"] == "season"]
        if not segments.empty:
            st.markdown("**Rendimiento por temporada**")
            st.dataframe(
                segments[["segment_value", "n_games", "log_loss", "brier_score", "accuracy"]]
                .rename(columns={"segment_value": "Temporada", "n_games": "Partidos"})
                .round(4),
                width="stretch", hide_index=True,
            )


# ======================================================= producción y drift


def render_production() -> None:
    st.subheader("Rendimiento del modelo en producción")

    target = st.radio(
        "Objetivo",
        options=list(TARGET_LABELS),
        format_func=lambda t: TARGET_LABELS[t],
        horizontal=True,
        key="prod_target",
    )

    performance = dd.load_production_performance(target)
    if performance.empty:
        st.info(
            "Sin histórico de producción. Se genera al ejecutar "
            "`uv run nflp monitor check` tras acumular predicciones resueltas."
        )
    else:
        st.plotly_chart(performance_chart(performance, target), width="stretch")

    st.divider()
    st.markdown("**Drift de features**")
    st.caption(
        "PSI: <0,10 estable · 0,10–0,25 cambio moderado · >0,25 cambio severo. "
        "Un drift generalizado indica que el entorno cambió y conviene reentrenar."
    )

    drift = dd.load_drift()
    if drift.empty:
        st.info("Sin comprobaciones de drift. Ejecuta `uv run nflp monitor drift`.")
        return

    drift = drift.copy()
    drift["estado"] = np.select(
        [drift["psi"] > 0.25, drift["psi"] > 0.10],
        ["severo", "moderado"],
        default="estable",
    )
    st.dataframe(
        drift[["feature_name", "psi", "reference_mean", "current_mean", "estado"]]
        .rename(
            columns={
                "feature_name": "Feature", "psi": "PSI",
                "reference_mean": "Media referencia", "current_mean": "Media actual",
                "estado": "Estado",
            }
        )
        .head(25)
        .round(4),
        width="stretch", hide_index=True,
    )


# ==================================================================== props


PROP_LABELS = {
    "receiving_yards": "Yardas de recepción",
    "rushing_yards": "Yardas de carrera",
    "receptions": "Recepciones",
    "receiving_tds": "Touchdowns de recepción",
    "rushing_tds": "Touchdowns de carrera",
    "passing_yards": "Yardas de pase",
    "passing_tds": "Touchdowns de pase",
    "sacks_suffered": "Veces capturado (QB)",
}

# Props de conteo disperso (mayoría de partidos en 0): la proyección es una
# aproximación más basta que en las yardas, que son continuas.
_SPARSE_PROPS = {"receiving_tds", "rushing_tds", "passing_tds", "sacks_suffered"}


def render_props(season: int | None, week: int | None) -> None:
    st.subheader("Proyecciones de jugador")

    target = st.radio(
        "Prop",
        options=list(PROP_LABELS),
        format_func=lambda t: PROP_LABELS[t],
        horizontal=True,
        key="prop_target",
    )
    if target in _SPARSE_PROPS:
        st.caption(
            "⚠️ Conteo poco frecuente (la mayoría de partidos son 0): la proyección "
            "es una aproximación más basta que en yardas continuas."
        )

    frame = dd.load_prop_predictions(target, season, week)
    if frame.empty:
        st.info(
            "Sin proyecciones. Genera algunas con "
            f"`uv run nflp predict --target {target} --season {season or 2026}`."
        )
        return

    frame["real"] = frame["actual_value"]

    display = frame[
        ["week", "player_name", "position", "team", "point_estimate",
         "lower_80", "upper_80", "real"]
    ].rename(
        columns={
            "week": "Sem", "player_name": "Jugador", "position": "Pos",
            "team": "Equipo", "point_estimate": "Proyección",
            "lower_80": "Mín 80%", "upper_80": "Máx 80%", "real": "Real",
        }
    )
    st.dataframe(display.round(1), width="stretch", hide_index=True)

    resolved = frame[frame["real"].notna()]
    if not resolved.empty:
        mae = (resolved["point_estimate"] - resolved["real"]).abs().mean()
        within = (
            (resolved["real"] >= resolved["lower_80"])
            & (resolved["real"] <= resolved["upper_80"])
        ).mean()
        columns = st.columns(3)
        columns[0].metric("MAE", f"{mae:.1f} yardas")
        columns[1].metric("Cobertura del intervalo 80%", f"{within:.1%}")
        columns[2].metric("Proyecciones resueltas", f"{len(resolved):,}")
        if abs(within - 0.80) > 0.08:
            st.caption(
                "⚠️ La cobertura se aleja del 80% nominal: la distribución predictiva "
                "está mal dimensionada (demasiado estrecha o demasiado ancha)."
            )

    selected = st.selectbox(
        "Ver justificación de un jugador",
        options=["—"] + frame["player_name"].dropna().unique().tolist(),
    )
    if selected != "—":
        row = frame[frame["player_name"] == selected].iloc[0]
        st.markdown(f"💡 {row['explanation_es']}")
        shap_values = dd.parse_shap(row.get("shap_values"))
        if shap_values:
            st.plotly_chart(
                shap_chart(shap_values, "más yardas", "menos yardas"),
                width="stretch",
            )


# ============================================================== trazabilidad


def render_registry() -> None:
    st.subheader("Registro de modelos y trazabilidad")

    registry = dd.load_model_registry()
    if registry.empty:
        st.info("Sin modelos registrados. Ejecuta `uv run nflp model train --target all`.")
        return

    status_emoji = {
        "production": "🟢 producción",
        "shadow": "🟡 shadow",
        "rejected": "🔴 rechazado",
        "archived": "⚪ archivado",
    }
    registry = registry.copy()
    registry["estado"] = registry["status"].map(status_emoji).fillna(registry["status"])

    st.dataframe(
        registry[
            ["target", "family", "version", "estado", "metric_name",
             "train_metric", "val_metric", "train_val_gap", "n_train_rows",
             "train_seasons", "created_at"]
        ].rename(
            columns={
                "target": "Objetivo", "family": "Familia", "version": "Ver.",
                "estado": "Estado", "metric_name": "Métrica",
                "train_metric": "Train", "val_metric": "Validación",
                "train_val_gap": "Brecha", "n_train_rows": "Filas train",
                "train_seasons": "Temporadas", "created_at": "Creado",
            }
        ).round(4),
        width="stretch", hide_index=True,
    )

    blocked = registry[registry["status"] == "rejected"]
    if not blocked.empty:
        st.markdown("**Modelos bloqueados por el gate de promoción**")
        st.caption(
            "El gate bloquea automáticamente cualquier modelo cuya brecha "
            "train/validación supere el umbral configurado: una brecha grande "
            "significa que memorizó el histórico."
        )
        for _, row in blocked.iterrows():
            st.warning(
                f"**{row['target']} v{int(row['version'])}** ({row['family']}) — "
                f"{row['promotion_blocked_reason']}"
            )

    st.divider()
    st.markdown("**Bitácora de reentrenamiento**")
    log_frame = dd.load_retraining_log()
    if log_frame.empty:
        st.info("Sin eventos registrados.")
    else:
        st.dataframe(
            log_frame.rename(
                columns={
                    "occurred_at": "Fecha", "target": "Objetivo", "action": "Acción",
                    "model_id": "Modelo", "previous_model_id": "Modelo anterior",
                    "reason": "Motivo",
                }
            ),
            width="stretch", hide_index=True,
        )


if __name__ == "__main__":
    main()
