"""Generación del reporte HTML de backtest.

Se regenera en cada corrida (requisito de la sección 4). El reporte es
autocontenido —sin dependencias externas ni CDN— para que pueda abrirse sin
conexión y archivarse junto a los resultados.

El tono del reporte es deliberadamente neutro: si el modelo pierde contra el
mercado, el reporte lo dice con la misma claridad con la que diría lo contrario.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nflp.core.config import get_settings
from nflp.core.db import get_db
from nflp.core.logging import get_logger

log = get_logger(__name__)

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.5rem;
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #fbfbfa; color: #1a1a19;
}
@media (prefers-color-scheme: dark) {
  body { background: #17181c; color: #e8e8e6; }
  .card { background: #1f2026 !important; border-color: #33343c !important; }
  th { background: #26272e !important; }
  tr:nth-child(even) td { background: #1c1d23 !important; }
  .muted { color: #9a9aa2 !important; }
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; letter-spacing: -.01em; }
h3 { font-size: .95rem; margin: 1.5rem 0 .5rem; font-weight: 600; }
.muted { color: #6b6b73; font-size: .875rem; }
.card {
  background: #fff; border: 1px solid #e4e4e1; border-radius: 10px;
  padding: 1.25rem; margin: 1rem 0;
}
table { width: 100%; border-collapse: collapse; font-size: .875rem; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
th, td { padding: .5rem .6rem; text-align: right; border-bottom: 1px solid #ececea; }
th:first-child, td:first-child { text-align: left; }
th { background: #f5f5f3; font-weight: 600; font-size: .8rem; white-space: nowrap; }
tr:nth-child(even) td { background: #fafaf9; }
.win { color: #15803d; font-weight: 600; }
.loss { color: #b91c1c; font-weight: 600; }
.model-row td { font-weight: 650; }
.badge {
  display: inline-block; padding: .15rem .5rem; border-radius: 5px;
  font-size: .75rem; font-weight: 600;
}
.badge-ok { background: #dcfce7; color: #14532d; }
.badge-no { background: #fee2e2; color: #7f1d1d; }
.note {
  border-left: 3px solid #d4d4d0; padding: .5rem 0 .5rem .9rem;
  margin: 1rem 0; font-size: .875rem; color: #55555c;
}
"""


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:,.1f}"
    if isinstance(value, (int,)):
        return f"{value:,}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, *, highlight_model: bool = False) -> str:
    if frame.empty:
        return '<p class="muted">Sin datos.</p>'

    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body: list[str] = []

    for _, row in frame.iterrows():
        is_model = highlight_model and str(row.iloc[0]) == "modelo"
        cells = "".join(f"<td>{_format_value(v)}</td>" for v in row)
        body.append(f'<tr class="{"model-row" if is_model else ""}">{cells}</tr>')

    return (
        f'<div class="scroll"><table><thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _verdict_block(overall: pd.DataFrame) -> str:
    """Bloque de veredicto: comparación honesta contra cada baseline."""
    metric = "log_loss" if "log_loss" in overall.columns else "mae"
    try:
        model_score = float(overall.loc[overall["modelo"] == "modelo", metric].iloc[0])
    except (IndexError, KeyError):
        return ""

    labels = {
        "baseline_elo": "Elo",
        "baseline_naive_local": "modelo naive",
        "baseline_media_historica": "media histórica",
        "baseline_mercado": "línea del mercado",
    }

    items: list[str] = []
    for rival, label in labels.items():
        subset = overall.loc[overall["modelo"] == rival, metric]
        if subset.empty or pd.isna(subset.iloc[0]):
            continue
        rival_score = float(subset.iloc[0])
        delta = rival_score - model_score
        pct = 100 * delta / abs(rival_score) if rival_score else 0.0
        badge = "badge-ok" if delta > 0 else "badge-no"
        verdict = "supera" if delta > 0 else "NO supera"
        items.append(
            f'<li><span class="badge {badge}">{verdict}</span> '
            f"a {html.escape(label)} — {metric} {delta:+.4f} ({pct:+.1f}%)</li>"
        )

    if not items:
        return ""
    return f'<ul style="list-style:none;padding:0;line-height:2.1">{"".join(items)}</ul>'


def render_report(reports: list[Any], output_path: Path | None = None) -> Path:
    """Genera el reporte HTML a partir de los objetos ``BacktestReport``."""
    settings = get_settings()
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    path = output_path or settings.paths.reports_dir / (
        f"backtest_{timestamp:%Y%m%d_%H%M%S}.html"
    )

    sections: list[str] = []

    for report in reports:
        overall = report.overall
        segments = report.segments

        section = [f"<h2>Objetivo: {html.escape(report.target)}</h2>"]
        section.append(
            f'<p class="muted">{len(report.predictions):,} predicciones fuera de muestra · '
            f"{report.n_retrains} reentrenamientos · corrida {html.escape(report.run_id[:8])}</p>"
        )

        section.append('<div class="card">')
        section.append("<h3>Comparación global contra baselines</h3>")
        section.append(_table(overall, highlight_model=True))
        section.append(_verdict_block(overall))
        section.append("</div>")

        if not segments.empty:
            for segment_type, label in (
                ("season", "Por temporada"),
                ("week_phase", "Por tramo de temporada"),
                ("weather", "Por condición climática"),
            ):
                subset = segments[segments["segment_type"] == segment_type].drop(
                    columns=["segment_type"]
                )
                if subset.empty:
                    continue
                subset = subset.rename(columns={"segment_value": "segmento", "n_games": "n"})
                section.append('<div class="card">')
                section.append(f"<h3>{label}</h3>")
                section.append(_table(subset.round(4)))
                section.append("</div>")

            worst = segments[segments["segment_type"] == "home_team"]
            if not worst.empty:
                metric = "log_loss" if "log_loss" in worst.columns else "mae"
                worst = worst.nlargest(8, metric)[
                    ["segment_value", "n_games", metric]
                ].rename(columns={"segment_value": "equipo local", "n_games": "n"})
                section.append('<div class="card">')
                section.append("<h3>Equipos peor predichos (mayor error)</h3>")
                section.append(_table(worst.round(4)))
                section.append(
                    '<p class="muted">Un error concentrado en pocos equipos suele indicar '
                    "un cambio estructural no capturado (entrenador nuevo, QB lesionado toda "
                    "la temporada) más que un fallo general del modelo.</p>"
                )
                section.append("</div>")

        sections.append("".join(section))

    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Backtest NFL-Predictor</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Reporte de backtest — NFL-Predictor</h1>
<p class="muted">Generado el {timestamp:%d/%m/%Y a las %H:%M}</p>
<div class="note">
  <strong>Metodología.</strong> Validación walk-forward estricta: cada predicción la
  emitió un modelo entrenado únicamente con partidos anteriores. Las líneas del mercado
  se usan como <em>benchmark</em>, nunca como features. La métrica principal es el
  log-loss (clasificación) o el MAE (regresión), no el accuracy: en predicción
  probabilística, la calibración importa más que acertar el lado.
</div>
{"".join(sections)}
<p class="muted" style="margin-top:3rem">
  Umbral de rentabilidad contra el spread con cuota −110: <strong>52,4 %</strong>.
  Un ATS por debajo de esa cifra pierde dinero aunque supere el 50 %.
</p>
</div></body></html>"""

    path.write_text(document, encoding="utf-8")
    log.info("Reporte HTML generado: %s", path)
    return path


def render_from_db(run_id: str | None = None) -> Path:
    """Regenera un reporte a partir de los resultados almacenados."""
    db = get_db()

    if run_id is None:
        run_id = db.scalar(
            "SELECT run_id FROM backtest_results ORDER BY created_at DESC LIMIT 1"
        )
    if not run_id:
        raise ValueError("No hay corridas de backtest almacenadas.")

    results = db.query(f"SELECT * FROM backtest_results WHERE run_id = '{run_id}'")
    if results.empty:
        raise ValueError(f"Corrida no encontrada: {run_id}")

    reports = []
    for target, group in results.groupby("target"):
        overall = (
            group[group["segment_type"] == "overall"]
            .rename(columns={"model_label": "modelo", "n_games": "n"})
            .drop(columns=["result_id", "run_id", "segment_type", "segment_value",
                           "target", "created_at", "beats_market"], errors="ignore")
            .dropna(axis=1, how="all")
        )
        segments = (
            group[group["segment_type"] != "overall"]
            .drop(columns=["result_id", "run_id", "model_label", "target",
                           "created_at", "beats_market"], errors="ignore")
            .dropna(axis=1, how="all")
        )

        reports.append(
            type(
                "StoredReport",
                (),
                {
                    "target": target, "overall": overall, "segments": segments,
                    "predictions": pd.DataFrame(index=range(int(group["n_games"].max() or 0))),
                    "n_retrains": 0, "run_id": run_id, "calibration": pd.DataFrame(),
                },
            )()
        )

    return render_report(reports)
