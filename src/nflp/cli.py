"""Interfaz de línea de comandos de NFL-Predictor.

Cada agente expone su propio grupo de comandos, de modo que puede ejecutarse de
forma totalmente independiente (requisito de la sección 1 del prompt)::

    nflp db init
    nflp data ingest --seasons 2011:2026
    nflp features build --season 2026
    nflp model train --target spread --tune
    nflp backtest run --from 2015 --to 2025
    nflp predict --season 2026 --week 2
    nflp monitor drift
    nflp dashboard

Las importaciones pesadas (LightGBM, SHAP, MLflow, Prefect) se hacen dentro de
cada comando para que arrancar la CLI sea instantáneo.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from nflp import __version__
from nflp.core.config import get_settings
from nflp.core.logging import setup_logging

console = Console()

app = typer.Typer(
    name="nflp",
    help="NFL-Predictor — predicción de resultados NFL con backtesting riguroso.",
    no_args_is_help=True,
    add_completion=False,
)

db_app = typer.Typer(name="db", help="Gestión del esquema de base de datos.", no_args_is_help=True)
data_app = typer.Typer(name="data", help="Data Agent — ingesta y normalización.", no_args_is_help=True)
stats_app = typer.Typer(name="stats", help="Stats Agent — estadísticas históricas.", no_args_is_help=True)
features_app = typer.Typer(name="features", help="Feature Engineering Agent.", no_args_is_help=True)
model_app = typer.Typer(name="model", help="Modeling Agent — entrenamiento y versionado.", no_args_is_help=True)
backtest_app = typer.Typer(name="backtest", help="Backtesting Agent — walk-forward.", no_args_is_help=True)
monitor_app = typer.Typer(name="monitor", help="Monitoring Agent — drift y promoción.", no_args_is_help=True)
pipeline_app = typer.Typer(name="pipeline", help="Orquestador — flows de Prefect.", no_args_is_help=True)

app.add_typer(db_app)
app.add_typer(data_app)
app.add_typer(stats_app)
app.add_typer(features_app)
app.add_typer(model_app)
app.add_typer(backtest_app)
app.add_typer(monitor_app)
app.add_typer(pipeline_app)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Activa el log de nivel DEBUG."),
) -> None:
    """Configura el logging antes de ejecutar cualquier subcomando."""
    settings = get_settings()
    setup_logging(
        "DEBUG" if verbose else settings.logging.level,
        rich_tracebacks=settings.logging.rich_tracebacks,
    )


@app.command()
def version() -> None:
    """Muestra la versión instalada."""
    console.print(f"NFL-Predictor [bold cyan]{__version__}[/bold cyan]")


def _parse_seasons(spec: str | None) -> list[int]:
    """Convierte '2011:2026' o '2020,2021' en una lista de temporadas."""
    settings = get_settings()
    if not spec:
        return settings.ingestion.seasons
    spec = spec.strip()
    if ":" in spec:
        start, end = spec.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in spec.split(",") if part.strip()]


# =============================================================================
# db
# =============================================================================


@db_app.command("init")
def db_init() -> None:
    """Crea el esquema aplicando todas las migraciones pendientes."""
    from nflp.core.db import get_db
    from nflp.core.migrations import current_version, migrate

    settings = get_settings()
    settings.paths.ensure_dirs()
    db = get_db(settings)

    applied = migrate(db)
    if applied:
        console.print(f"[green]Aplicadas {len(applied)} migraciones.[/green]")
    else:
        console.print("[yellow]No había migraciones pendientes.[/yellow]")
    console.print(f"Versión de esquema: [bold]{current_version(db)}[/bold]")
    console.print(f"Backend: [bold]{db.backend}[/bold]")
    if db.backend == "duckdb":
        console.print(f"Fichero: {settings.paths.duckdb_path}")


@db_app.command("status")
def db_status() -> None:
    """Muestra las tablas del esquema y su número de filas."""
    from nflp.core.db import get_db
    from nflp.core.migrations import current_version

    db = get_db()
    console.print(f"Versión de esquema: [bold]{current_version(db)}[/bold]  ·  backend: {db.backend}")

    tables = [
        "teams", "stadiums", "games", "players", "rosters_weekly", "injuries",
        "player_stats_weekly", "team_stats_weekly", "snap_counts", "weather",
        "market_lines", "ingestion_log", "features_game", "features_player",
        "elo_ratings", "team_style_clusters", "model_registry", "predictions",
        "backtest_results", "calibration_bins", "drift_metrics", "retraining_log",
        "production_performance",
    ]

    table = Table(title="Estado de la base de datos", header_style="bold cyan")
    table.add_column("Tabla")
    table.add_column("Filas", justify="right")
    table.add_column("Estado")

    for name in tables:
        if db.table_exists(name):
            count = db.row_count(name)
            estado = "[green]ok[/green]" if count else "[yellow]vacía[/yellow]"
            table.add_row(name, f"{count:,}", estado)
        else:
            table.add_row(name, "-", "[red]no existe[/red]")

    console.print(table)


@db_app.command("reset")
def db_reset(
    yes: bool = typer.Option(False, "--yes", help="Confirma el borrado sin preguntar."),
) -> None:
    """Borra la base DuckDB y vuelve a crear el esquema desde cero."""
    settings = get_settings()
    if settings.database.backend != "duckdb":
        console.print("[red]`reset` solo está soportado con el backend DuckDB.[/red]")
        raise typer.Exit(code=1)

    path = settings.paths.duckdb_path
    if not yes:
        confirmed = typer.confirm(f"Esto borrará {path}. ¿Continuar?")
        if not confirmed:
            console.print("Cancelado.")
            raise typer.Exit()

    if path.exists():
        path.unlink()
        console.print(f"[yellow]Borrado {path}[/yellow]")
    db_init()


# =============================================================================
# data
# =============================================================================


@data_app.command("ingest")
def data_ingest(
    seasons: str = typer.Option(None, "--seasons", help="Rango '2011:2026' o lista '2024,2025'."),
    force: bool = typer.Option(False, "--force", help="Ignora la caché y vuelve a descargar."),
) -> None:
    """Ingesta calendario, equipos, estadios, rosters, lesiones y snaps."""
    from nflp.agents.data import DataAgent

    agent = DataAgent()
    report = agent.run(_parse_seasons(seasons), force=force)

    table = Table(title="Ingesta completada", header_style="bold cyan")
    table.add_column("Tabla")
    table.add_column("Filas", justify="right")
    for name, rows in report.tables.items():
        table.add_row(name, f"{rows:,}")
    console.print(table)


@data_app.command("stadiums")
def data_stadiums() -> None:
    """Siembra únicamente la tabla de referencia de estadios."""
    from nflp.agents.data import DataAgent

    rows = DataAgent().ingest_stadiums()
    console.print(f"[green]Estadios sembrados: {rows}[/green]")


@data_app.command("weather")
def data_weather(
    seasons: str = typer.Option(None, "--seasons", help="Rango '2011:2026' o lista."),
    force: bool = typer.Option(False, "--force", help="Recalcula el clima ya almacenado."),
) -> None:
    """Descarga clima histórico y pronóstico para los partidos al aire libre."""
    from nflp.agents.data.weather import WeatherAgent

    rows = WeatherAgent().run(_parse_seasons(seasons), force=force)
    console.print(f"[green]Clima: {rows:,} partidos actualizados.[/green]")


# =============================================================================
# stats
# =============================================================================


@stats_app.command("build")
def stats_build(
    seasons: str = typer.Option(None, "--seasons", help="Rango '2011:2026' o lista."),
    force: bool = typer.Option(False, "--force", help="Ignora la caché y vuelve a descargar."),
) -> None:
    """Ingesta estadísticas semanales de jugador y equipo."""
    from nflp.agents.stats import StatsAgent

    report = StatsAgent().run(_parse_seasons(seasons), force=force)

    table = Table(title="Estadísticas ingeridas", header_style="bold cyan")
    table.add_column("Tabla")
    table.add_column("Filas", justify="right")
    for name, rows in report.tables.items():
        table.add_row(name, f"{rows:,}")
    console.print(table)


# =============================================================================
# features
# =============================================================================


@features_app.command("build")
def features_build(
    seasons: str = typer.Option(None, "--seasons", help="Rango '2011:2026' o lista."),
) -> None:
    """Construye el feature store aplicando el contrato anti-leakage."""
    from nflp.agents.features import FeatureAgent

    frame = FeatureAgent().run(_parse_seasons(seasons) if seasons else None)
    console.print(f"[green]Feature store: {len(frame):,} filas.[/green]")


@features_app.command("build-players")
def features_build_players(
    seasons: str = typer.Option(None, "--seasons", help="Rango '2011:2026' o lista."),
) -> None:
    """Construye el feature store a nivel jugador (para los modelos de props)."""
    from nflp.agents.features.player_agent import PlayerFeatureAgent

    frame = PlayerFeatureAgent().run(_parse_seasons(seasons) if seasons else None)
    console.print(f"[green]Feature store de jugador: {len(frame):,} filas.[/green]")


@features_app.command("audit")
def features_audit() -> None:
    """Audita el feature store en busca de leakage."""
    from nflp.agents.features import audit_features
    from nflp.core.db import get_db

    db = get_db()
    frame = db.query(
        """
        SELECT f.*, g.kickoff_utc
        FROM features_game f JOIN games g USING (game_id)
        """
    )
    if frame.empty:
        console.print("[yellow]El feature store está vacío.[/yellow]")
        raise typer.Exit(code=1)

    kickoff = frame.pop("kickoff_utc")
    report = audit_features(frame, kickoff=kickoff)

    if report.is_clean:
        console.print(f"[green]{report.describe()}[/green]")
    else:
        console.print(f"[red]{report.describe()}[/red]")
        raise typer.Exit(code=1)


# =============================================================================
# model
# =============================================================================


@model_app.command("train")
def model_train(
    target: str = typer.Option("spread", "--target", help="winner | spread | total | all"),
    tune: bool = typer.Option(False, "--tune", help="Búsqueda de hiperparámetros con Optuna."),
    families: str = typer.Option(None, "--families", help="Lista, p. ej. 'lightgbm,xgboost'."),
    val_seasons: int = typer.Option(2, "--val-seasons", help="Temporadas reservadas a validación."),
) -> None:
    """Entrena un modelo comparando familias y aplicando el gate de promoción."""
    from nflp.agents.modeling import ModelingAgent

    agent = ModelingAgent()
    family_list = [f.strip() for f in families.split(",")] if families else None
    targets = ["winner", "spread", "total"] if target == "all" else [target]

    for name in targets:
        result = agent.train(name, families=family_list, tune=tune, val_seasons=val_seasons)

        console.print(f"\n[bold cyan]Objetivo: {name}[/bold cyan]")
        console.print(result.comparison.to_string(index=False))

        colour = {"shadow": "green", "rejected": "red"}.get(result.status, "yellow")
        console.print(
            f"Ganador: [bold]{result.family}[/bold] v{result.version} · "
            f"{result.metric_name} val = [bold]{result.val_metric:.4f}[/bold] · "
            f"brecha = {result.train_val_gap:+.1%} · "
            f"estado = [{colour}]{result.status}[/{colour}]"
        )
        if result.blocked_reason:
            console.print(f"[red]Promoción bloqueada: {result.blocked_reason}[/red]")


@model_app.command("train-props")
def model_train_props(
    target: str = typer.Option("all", "--target", help="receiving_yards | rushing_yards | all"),
) -> None:
    """Entrena los modelos de props de jugador."""
    from nflp.agents.modeling.props_agent import PropsAgent

    agent = PropsAgent()
    targets = agent.settings.props.targets if target == "all" else [target]

    for name in targets:
        model = agent.train(name)
        console.print(
            f"[bold cyan]{name}[/bold cyan] · familia {model.family} · "
            f"MAE val = [bold]{model.metrics['val_mae']:.2f}[/bold] · "
            f"CRPS = {model.metrics['val_crps']:.2f} · "
            f"pseudo-labeling: {'sí' if model.used_pseudo_labels else 'no'}"
        )


@backtest_app.command("props")
def backtest_props(
    target: str = typer.Option("all", "--target", help="receiving_yards | rushing_yards | all"),
    from_season: int = typer.Option(None, "--from"),
    to_season: int = typer.Option(None, "--to"),
) -> None:
    """Valida los props con walk-forward y compara las variantes."""
    from nflp.agents.modeling.props_agent import PropsAgent

    agent = PropsAgent()
    targets = agent.settings.props.targets if target == "all" else [target]

    for name in targets:
        result = agent.backtest(name, start_season=from_season, end_season=to_season)
        console.print(f"\n[bold cyan]Prop: {name}[/bold cyan] ({result.n_rows:,} predicciones)")
        console.print(result.comparison.to_string(index=False))


@model_app.command("list")
def model_list(
    target: str = typer.Option(None, "--target", help="Filtra por objetivo."),
) -> None:
    """Lista las versiones registradas de modelos."""
    from nflp.registry import ModelRegistry

    frame = ModelRegistry().list_models(target)
    if frame.empty:
        console.print("[yellow]No hay modelos registrados.[/yellow]")
        return

    table = Table(title="Registro de modelos", header_style="bold cyan")
    for column in ("objetivo", "familia", "ver", "estado", "métrica", "train", "val", "brecha"):
        table.add_column(column)

    for _, row in frame.iterrows():
        colour = {"production": "green", "shadow": "yellow", "rejected": "red"}.get(
            row["status"], "white"
        )
        table.add_row(
            row["target"], row["family"], str(row["version"]),
            f"[{colour}]{row['status']}[/{colour}]",
            str(row["metric_name"]),
            f"{row['train_metric']:.4f}" if pd_notna(row["train_metric"]) else "-",
            f"{row['val_metric']:.4f}" if pd_notna(row["val_metric"]) else "-",
            f"{row['train_val_gap']:+.1%}" if pd_notna(row["train_val_gap"]) else "-",
        )
    console.print(table)


@model_app.command("promote")
def model_promote(
    model_id: str = typer.Argument(..., help="Identificador del modelo a promover."),
) -> None:
    """Promueve un modelo a producción (archiva el anterior)."""
    from nflp.registry import ModelRegistry

    if ModelRegistry().promote(model_id):
        console.print(f"[green]Modelo {model_id} promovido a producción.[/green]")
    else:
        console.print("[red]No se pudo promover el modelo.[/red]")
        raise typer.Exit(code=1)


@model_app.command("rollback")
def model_rollback(
    target: str = typer.Argument(..., help="Objetivo sobre el que hacer rollback."),
) -> None:
    """Vuelve a la versión anterior de producción."""
    from nflp.registry import ModelRegistry

    previous = ModelRegistry().rollback(target)
    if previous:
        console.print(f"[yellow]Rollback completado: modelo {previous} en producción.[/yellow]")
    else:
        console.print("[red]No hay versión anterior disponible.[/red]")
        raise typer.Exit(code=1)


# =============================================================================
# predict
# =============================================================================


@app.command("predict")
def predict(
    season: int = typer.Option(None, "--season", help="Temporada a predecir."),
    week: int = typer.Option(None, "--week", help="Semana a predecir."),
    target: str = typer.Option("all", "--target", help="winner | spread | total | props | all"),
    shadow: bool = typer.Option(False, "--shadow", help="Usa el modelo en shadow."),
) -> None:
    """Genera predicciones con justificación SHAP y las persiste con trazabilidad."""
    from nflp.agents.modeling.predict import Predictor

    predictor = Predictor()
    game_targets = ["winner", "spread", "total"]
    prop_targets = predictor.settings.props.targets

    if target == "all":
        selected_games, selected_props = game_targets, prop_targets
    elif target == "props":
        selected_games, selected_props = [], prop_targets
    elif target in prop_targets:
        selected_games, selected_props = [], [target]
    else:
        selected_games, selected_props = [target], []

    for name in selected_games:
        batch = predictor.predict_games(name, season=season, week=week, use_shadow=shadow)
        if batch is None:
            console.print(f"[yellow]Sin predicciones para '{name}'.[/yellow]")
            continue
        _show_game_predictions(batch)

    for name in selected_props:
        batch = predictor.predict_props(name, season=season, week=week)
        if batch is None:
            console.print(f"[yellow]Sin proyecciones para '{name}'.[/yellow]")
            continue
        _show_prop_predictions(batch)


def _show_game_predictions(batch: object) -> None:
    frame = batch.frame  # type: ignore[attr-defined]
    target = batch.target  # type: ignore[attr-defined]

    table = Table(
        title=f"Predicciones — {target} (modelo {batch.model_id[:8]})",  # type: ignore[attr-defined]
        header_style="bold cyan",
    )
    table.add_column("Partido")
    table.add_column("Sem", justify="right")
    table.add_column("Predicción", justify="right")
    table.add_column("Prob.", justify="right")
    table.add_column("Confianza")

    for _, row in frame.head(20).iterrows():
        if target == "winner":
            prediction = f"{row['home_team']} gana" if row["probability"] >= 0.5 else f"{row['away_team']} gana"
            probability = f"{max(row['probability'], 1 - row['probability']):.1%}"
        elif target == "spread":
            prediction = f"{row['point_estimate']:+.1f}"
            probability = f"{row['probability']:.1%}" if pd_notna(row["probability"]) else "-"
        else:
            prediction = f"{row['point_estimate']:.1f}"
            probability = "-"

        table.add_row(
            f"{row['away_team']} @ {row['home_team']}",
            str(row["week"]),
            prediction,
            probability,
            str(row["confidence_level"]),
        )

    console.print(table)

    # Muestra la justificación del primer partido como ejemplo.
    if not frame.empty:
        first = frame.iloc[0]
        console.print(
            f"[dim]Ejemplo de justificación ({first['away_team']} @ {first['home_team']}):[/dim]\n"
            f"  {first['explanation_es']}\n"
        )


def _show_prop_predictions(batch: object) -> None:
    frame = batch.frame  # type: ignore[attr-defined]

    table = Table(title=f"Proyecciones — {batch.target}", header_style="bold cyan")  # type: ignore[attr-defined]
    table.add_column("Jugador")
    table.add_column("Equipo")
    table.add_column("Proyección", justify="right")
    table.add_column("Rango 80%", justify="right")

    top = frame.nlargest(15, "point_estimate")
    for _, row in top.iterrows():
        table.add_row(
            str(row["player_name"] or row["gsis_id"]),
            str(row["team"]),
            f"{row['point_estimate']:.1f}",
            f"{row['lower_80']:.0f} – {row['upper_80']:.0f}",
        )
    console.print(table)


# =============================================================================
# backtest
# =============================================================================


@backtest_app.command("run")
def backtest_run(
    target: str = typer.Option("all", "--target", help="winner | spread | total | all"),
    from_season: int = typer.Option(None, "--from", help="Primera temporada evaluada."),
    to_season: int = typer.Option(None, "--to", help="Última temporada evaluada."),
    family: str = typer.Option("catboost", "--family", help="Familia de modelo a validar."),
    report: bool = typer.Option(True, "--report/--no-report", help="Genera el reporte HTML."),
) -> None:
    """Ejecuta la validación walk-forward contra todos los baselines."""
    from nflp.agents.backtesting.agent import BacktestingAgent
    from nflp.agents.backtesting.report import render_report

    agent = BacktestingAgent()
    targets = ["winner", "spread", "total"] if target == "all" else [target]
    reports = []

    for name in targets:
        result = agent.run(name, start_season=from_season, end_season=to_season, family=family)
        reports.append(result)

        console.print(f"\n[bold cyan]Backtest walk-forward — objetivo: {name}[/bold cyan]")
        console.print(
            f"{len(result.predictions):,} predicciones fuera de muestra · "
            f"{result.n_retrains} reentrenamientos"
        )
        console.print(result.overall.to_string(index=False))

        _print_verdict(result)

    if report and reports:
        path = render_report(reports)
        console.print(f"\n[green]Reporte HTML generado: {path}[/green]")


def _print_verdict(result: object) -> None:
    """Imprime el veredicto honesto: ¿le ganamos al mercado y a Elo?"""
    overall = result.overall  # type: ignore[attr-defined]
    metric = "log_loss" if "log_loss" in overall.columns else "mae"

    try:
        model_score = float(overall.loc[overall["modelo"] == "modelo", metric].iloc[0])
    except (IndexError, KeyError):
        return

    for rival, etiqueta in (
        ("baseline_elo", "Elo"),
        ("baseline_naive_local", "naive"),
        ("baseline_media_historica", "media histórica"),
        ("baseline_mercado", "mercado"),
    ):
        subset = overall.loc[overall["modelo"] == rival, metric]
        if subset.empty or pd_isna(subset.iloc[0]):
            continue
        rival_score = float(subset.iloc[0])
        delta = rival_score - model_score           # positivo = ganamos nosotros
        pct = 100 * delta / abs(rival_score) if rival_score else 0.0
        if delta > 0:
            console.print(f"  [green]✓ supera a {etiqueta}[/green] ({metric} {delta:+.4f}, {pct:+.1f}%)")
        else:
            console.print(f"  [red]✗ NO supera a {etiqueta}[/red] ({metric} {delta:+.4f}, {pct:+.1f}%)")


@backtest_app.command("report")
def backtest_report(
    run_id: str = typer.Option(None, "--run-id", help="Corrida concreta; por defecto la última."),
) -> None:
    """Regenera el reporte HTML de una corrida ya ejecutada."""
    from nflp.agents.backtesting.report import render_from_db

    path = render_from_db(run_id)
    console.print(f"[green]Reporte generado: {path}[/green]")


# =============================================================================
# monitor
# =============================================================================


@monitor_app.command("drift")
def monitor_drift(
    season: int = typer.Option(None, "--season", help="Temporada a comparar."),
    top: int = typer.Option(15, "--top", help="Cuántas features mostrar."),
) -> None:
    """Mide el drift de las features respecto a las temporadas de referencia."""
    from nflp.agents.monitoring import MonitoringAgent

    report = MonitoringAgent().check_drift(current_season=season)
    console.print(report.describe())

    if report.metrics.empty:
        return

    table = Table(title="Índice de estabilidad poblacional (PSI)", header_style="bold cyan")
    table.add_column("Feature")
    table.add_column("PSI", justify="right")
    table.add_column("Media ref.", justify="right")
    table.add_column("Media actual", justify="right")
    table.add_column("Estado")

    for _, row in report.metrics.head(top).iterrows():
        psi = row["psi"]
        if psi > 0.25:
            estado = "[red]severo[/red]"
        elif psi > 0.10:
            estado = "[yellow]moderado[/yellow]"
        else:
            estado = "[green]estable[/green]"
        table.add_row(
            row["feature_name"], f"{psi:.4f}",
            f"{row['reference_mean']:.3f}", f"{row['current_mean']:.3f}", estado,
        )
    console.print(table)

    if report.should_retrain:
        console.print("[yellow]Se recomienda reentrenar: drift generalizado.[/yellow]")


@monitor_app.command("check")
def monitor_check() -> None:
    """Ciclo completo: drift, rendimiento, promoción desde shadow y rollback."""
    from nflp.agents.monitoring import MonitoringAgent

    results = MonitoringAgent().run()
    console.print(f"[bold]Drift:[/bold] {results['drift']}")

    for target, info in results["targets"].items():  # type: ignore[index]
        console.print(f"\n[bold cyan]{target}[/bold cyan]")
        console.print(f"  shadow: {info['shadow'].get('action')} — {info['shadow'].get('reason')}")
        console.print(
            f"  degradación: {info['degradacion'].get('action')} — "
            f"{info['degradacion'].get('reason')}"
        )


# =============================================================================
# pipeline (Prefect)
# =============================================================================


@pipeline_app.command("weekly")
def pipeline_weekly(
    season: int = typer.Option(None, "--season"),
    week: int = typer.Option(None, "--week"),
) -> None:
    """Ejecuta el flow semanal de Prefect."""
    from nflp.orchestrator import weekly_flow

    result = weekly_flow(season=season, week=week)
    console.print("[green]Flow semanal completado.[/green]")
    console.print(result)


@pipeline_app.command("retrain")
def pipeline_retrain(
    tune: bool = typer.Option(False, "--tune", help="Activa el tuning con Optuna."),
) -> None:
    """Ejecuta el flow de reentrenamiento con gate de promoción."""
    from nflp.orchestrator import retrain_flow

    retrain_flow(tune=tune)
    console.print("[green]Flow de reentrenamiento completado.[/green]")


@pipeline_app.command("backtest")
def pipeline_backtest() -> None:
    """Ejecuta el flow de backtesting."""
    from nflp.orchestrator import backtest_flow

    backtest_flow()
    console.print("[green]Flow de backtest completado.[/green]")


@pipeline_app.command("full")
def pipeline_full(
    seasons: str = typer.Option(None, "--seasons"),
    tune: bool = typer.Option(False, "--tune"),
) -> None:
    """Pipeline completo desde cero: ingesta, features, entrenamiento y backtest."""
    from nflp.orchestrator import full_pipeline

    full_pipeline(seasons=_parse_seasons(seasons) if seasons else None, tune=tune)
    console.print("[green]Pipeline completo terminado.[/green]")


# =============================================================================
# dashboard
# =============================================================================


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8501, "--port", help="Puerto del servidor."),
) -> None:
    """Lanza el dashboard de Streamlit."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print(f"[cyan]Iniciando dashboard en http://localhost:{port} ...[/cyan]")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(port),
            "--server.headless", "true",
        ],
        check=False,
    )


def pd_isna(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


def pd_notna(value: object) -> bool:
    import pandas as pd

    return bool(pd.notna(value))


if __name__ == "__main__":
    app()
