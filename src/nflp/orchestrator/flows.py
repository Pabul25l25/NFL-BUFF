"""Orquestación con Prefect 3.

Define los flows que encadenan a los agentes. Cada tarea envuelve a un agente que
también puede ejecutarse por separado desde la CLI: Prefect aporta reintentos,
observabilidad y programación, pero no es un requisito para que nada funcione.

Flows disponibles:

* ``weekly_flow`` — **el comando semanal único**: ingesta, features, reentrena,
  promueve si corresponde y predice, para partidos y para props. Pensado para
  correr los martes, cuando ya se han cerrado los partidos del fin de semana y
  se publican los primeros partes de lesiones de la semana siguiente.
* ``retrain_flow`` — solo reentrenamiento con gate de promoción (sin ingesta ni
  predicción), para cuando ya tienes los datos al día y solo quieres reentrenar.
* ``backtest_flow`` — validación walk-forward completa.
* ``full_pipeline`` — todo lo anterior, para una puesta en marcha desde cero.

**Orden dentro de ``weekly_flow``, y por qué importa**: primero se reentrena y se
deja que el monitor decida si el modelo nuevo reemplaza al de producción, y
**solo después** se generan las predicciones. Invertir ese orden —predecir antes
de promover— es un error real que se detectó en la práctica: las predicciones
quedaban firmadas por el modelo saliente aunque uno mejor ya estuviera listo.

Los reintentos están puestos donde tienen sentido: en las tareas que dependen de
la red (ingesta, clima). Reintentar un entrenamiento que falló por un error de
datos no arregla nada.
"""

from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

from nflp.core.config import get_settings
from nflp.core.logging import setup_logging

# ============================================================ tareas de datos


@task(name="ingesta-nflverse", retries=3, retry_delay_seconds=30, log_prints=True)
def ingest_data_task(seasons: list[int] | None = None, force: bool = False) -> dict[str, int]:
    """Data Agent: calendario, equipos, estadios, rosters, lesiones, snaps."""
    from nflp.agents.data import DataAgent

    logger = get_run_logger()
    report = DataAgent().run(seasons, force=force)
    logger.info("Ingesta completada: %s filas.", report.total())
    return report.tables


@task(name="ingesta-estadisticas", retries=3, retry_delay_seconds=30)
def ingest_stats_task(seasons: list[int] | None = None, force: bool = False) -> dict[str, int]:
    """Stats Agent: estadísticas semanales de jugador y equipo."""
    from nflp.agents.stats import StatsAgent

    return StatsAgent().run(seasons, force=force).tables


@task(name="ingesta-clima", retries=4, retry_delay_seconds=60)
def ingest_weather_task(seasons: list[int] | None = None, force: bool = False) -> int:
    """Clima histórico y pronóstico (Open-Meteo).

    ``force=True`` en el ciclo semanal a propósito: el pronóstico de los
    próximos partidos mejora a medida que se acerca la fecha, así que conviene
    refrescarlo aunque ya hubiera un valor guardado.
    """
    from nflp.agents.data.weather import WeatherAgent

    return WeatherAgent().run(seasons, force=force)


# ========================================================== tareas de features


@task(name="features-partido")
def build_game_features_task(seasons: list[int] | None = None) -> int:
    """Feature Agent: features a nivel partido, con auditoría anti-leakage."""
    from nflp.agents.features import FeatureAgent

    return len(FeatureAgent().run(seasons))


@task(name="features-jugador")
def build_player_features_task(seasons: list[int] | None = None) -> int:
    """Features a nivel jugador para los modelos de props.

    Cubre partidos ya jugados y también futuros: el esqueleto sale del róster
    activo de cada equipo, no de estadísticas que todavía no existen.
    """
    from nflp.agents.features.player_agent import PlayerFeatureAgent

    return len(PlayerFeatureAgent().run(seasons))


@task(name="auditoria-anti-leakage")
def audit_features_task() -> bool:
    """Verifica el contrato anti-leakage. Falla el flow si detecta un problema.

    Es una tarea propia y no un detalle interno: si el feature store tiene
    leakage, todo lo que venga después carece de valor, así que el pipeline debe
    detenerse aquí y no seguir entrenando.
    """
    from nflp.agents.features import audit_features
    from nflp.core.db import get_db

    logger = get_run_logger()
    db = get_db()
    frame = db.query(
        "SELECT f.*, g.kickoff_utc FROM features_game f JOIN games g USING (game_id)"
    )
    if frame.empty:
        raise RuntimeError("El feature store está vacío.")

    kickoff = frame.pop("kickoff_utc")
    report = audit_features(frame, kickoff=kickoff)
    if not report.is_clean:
        raise RuntimeError(report.describe())

    logger.info(report.describe())
    return True


# ========================================================== tareas de modelado


@task(name="entrenar-modelo")
def train_model_task(target: str, tune: bool = False) -> dict[str, Any]:
    """Modeling Agent: entrena, calibra y aplica el gate de promoción."""
    from nflp.agents.modeling import ModelingAgent

    logger = get_run_logger()
    result = ModelingAgent().train(target, tune=tune)
    logger.info(
        "Modelo '%s': %s v%d · %s=%.4f · brecha %+.1f%% · %s",
        target, result.family, result.version, result.metric_name,
        result.val_metric, 100 * result.train_val_gap, result.status,
    )
    return {
        "model_id": result.model_id, "target": target, "status": result.status,
        "val_metric": result.val_metric, "gap": result.train_val_gap,
    }


@task(name="entrenar-props")
def train_props_task(target: str) -> dict[str, Any]:
    """Entrena un modelo de prop de jugador."""
    from nflp.agents.modeling.props_agent import PropsAgent

    model = PropsAgent().train(target)
    return {"target": target, "metrics": model.metrics}


@task(name="predecir")
def predict_task(target: str, season: int | None = None, week: int | None = None) -> int:
    """Genera predicciones de partido con justificación SHAP."""
    from nflp.agents.modeling.predict import Predictor

    batch = Predictor().predict_games(target, season=season, week=week)
    return batch.n_predictions if batch else 0


@task(name="predecir-props")
def predict_props_task(target: str, season: int | None = None, week: int | None = None) -> int:
    """Genera proyecciones de props de jugador con justificación SHAP."""
    from nflp.agents.modeling.predict import Predictor

    batch = Predictor().predict_props(target, season=season, week=week)
    return batch.n_predictions if batch else 0


@task(name="backtest")
def backtest_task(target: str) -> dict[str, Any]:
    """Backtesting Agent: walk-forward contra todos los baselines."""
    from nflp.agents.backtesting.agent import BacktestingAgent

    report = BacktestingAgent().run(target)
    return {
        "target": target,
        "run_id": report.run_id,
        "overall": report.overall.to_dict(orient="records"),
    }


# ======================================================== tareas de monitoreo


@task(name="monitoreo")
def monitor_task(targets: list[str] | None = None) -> dict[str, Any]:
    """Monitoring Agent: drift, rendimiento, promoción y rollback.

    Sin argumentos, vigila tanto los objetivos de partido como los props — ver
    ``MonitoringAgent.run()``.
    """
    from nflp.agents.monitoring import MonitoringAgent

    logger = get_run_logger()
    results = MonitoringAgent().run(targets)
    logger.info("Monitoreo: %s", results.get("drift"))
    return results


# ================================================================== flows


@flow(
    name="nflp-semanal",
    description=(
        "El comando semanal único: ingesta, features, reentrena, promueve y "
        "predice — partidos y props."
    ),
    task_runner=ThreadPoolTaskRunner(max_workers=2),
)
def weekly_flow(season: int | None = None, week: int | None = None) -> dict[str, Any]:
    """Ciclo semanal completo, alineado al calendario NFL (sugerido: martes).

    Orden, y por qué es ese orden:

    1. **Ingesta** — calendario, lesiones, rosters, estadísticas, clima.
    2. **Features** — se reconstruyen con los datos frescos (partido y jugador).
    3. **Auditoría anti-leakage** — si falla, el flow se detiene aquí.
    4. **Reentrenamiento** — partidos y props, con el gate de over/underfitting.
    5. **Monitoreo** — decide si cada modelo nuevo reemplaza al de producción.
    6. **Predicción** — la última, para que use siempre el modelo ya promovido.
    """
    setup_logging(get_settings().logging.level)
    logger = get_run_logger()
    logger.info("Iniciando ciclo semanal NFL-Predictor.")

    settings = get_settings()
    current = [settings.ingestion.end_season]

    # --- 1. ingesta: datos y estadísticas en paralelo, clima después ---
    data = ingest_data_task.submit(current)
    stats = ingest_stats_task.submit(current)
    data.result()
    stats.result()
    ingest_weather_task(current, force=True)

    # --- 2-3. features y auditoría ---
    build_game_features_task(None)
    build_player_features_task(None)
    audit_features_task()

    # --- 4. reentrenamiento ---
    training = {target: train_model_task(target) for target in settings.modeling.targets}
    for prop in settings.props.targets:
        training[prop] = train_props_task(prop)

    # --- 5. monitoreo y promoción (incluye partidos y props) ---
    monitoring = monitor_task()

    # --- 6. predicción, ya con los modelos promovidos ---
    predictions = {
        target: predict_task(target, season, week) for target in settings.modeling.targets
    }
    for prop in settings.props.targets:
        predictions[prop] = predict_props_task(prop, season, week)

    logger.info("Ciclo semanal completado.")
    return {"entrenamiento": training, "monitoreo": monitoring, "predicciones": predictions}


@flow(name="nflp-reentrenamiento", description="Reentrenamiento con gate de promoción.")
def retrain_flow(tune: bool = False, targets: list[str] | None = None) -> dict[str, Any]:
    """Reentrena todos los objetivos y evalúa la promoción desde shadow.

    A diferencia de ``weekly_flow``, no ingiere datos nuevos ni predice — es
    para cuando los datos ya están al día y solo hace falta reentrenar.
    """
    setup_logging(get_settings().logging.level)
    settings = get_settings()
    targets = targets or settings.modeling.targets

    build_game_features_task(None)
    build_player_features_task(None)
    audit_features_task()

    results = {target: train_model_task(target, tune) for target in targets}
    for prop in settings.props.targets:
        results[prop] = train_props_task(prop)

    results["monitoreo"] = monitor_task()
    return results


@flow(name="nflp-backtest", description="Validación walk-forward completa.")
def backtest_flow(targets: list[str] | None = None) -> dict[str, Any]:
    """Ejecuta el backtest de todos los objetivos y regenera el reporte."""
    setup_logging(get_settings().logging.level)
    targets = targets or get_settings().modeling.targets
    return {target: backtest_task(target) for target in targets}


@flow(name="nflp-completo", description="Puesta en marcha desde cero.")
def full_pipeline(seasons: list[int] | None = None, tune: bool = False) -> dict[str, Any]:
    """Pipeline completo: ingesta histórica, features, entrenamiento y backtest."""
    setup_logging(get_settings().logging.level)
    settings = get_settings()
    seasons = seasons or settings.ingestion.seasons

    ingest_data_task(seasons)
    ingest_stats_task(seasons)
    ingest_weather_task(seasons)

    build_game_features_task(None)
    build_player_features_task(None)
    audit_features_task()

    results: dict[str, Any] = {}
    for target in settings.modeling.targets:
        results[target] = train_model_task(target, tune)
    for prop in settings.props.targets:
        results[prop] = train_props_task(prop)

    results["monitoreo"] = monitor_task()
    results["predicciones"] = {
        target: predict_task(target) for target in settings.modeling.targets
    }
    for prop in settings.props.targets:
        results["predicciones"][prop] = predict_props_task(prop)

    results["backtest"] = {t: backtest_task(t) for t in settings.modeling.targets}
    return results
