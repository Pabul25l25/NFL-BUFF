# Prompt para Claude Code — NFL Predictor Platform

```
Quiero que construyas un proyecto completo llamado "NFL-Predictor": una plataforma de
predicción de resultados NFL (spread, moneyline, totales y props de jugadores) con
foco en confiabilidad estadística, backtesting riguroso y reentrenamiento continuo
sin overfitting. Actúa como arquitecto de datos + ingeniero ML senior. No optimices
por velocidad de entrega, optimiza por corrección metodológica.

## 1. ARQUITECTURA GENERAL
Diseña el proyecto como un sistema multi-agente orquestado, donde cada agente es un
módulo independiente con responsabilidad única, testeable por separado:

- **Data Agent**: ingesta y normalización de datos (calendario, rosters, ubicaciones/
  estadios, lesiones, clima).
- **Stats Agent**: estadísticas históricas por jugador y equipo (multi-temporada),
  incluye ETL de fuentes públicas (nflverse/nflfastR, Pro Football Reference, ESPN API,
  Open-Meteo/NOAA para clima histórico y forecast).
- **Feature Engineering Agent**: construye features derivados (EPA, DVOA-like metrics,
  rolling averages, efectos de clima/altitud/superficie, descanso entre partidos,
  travel distance, home/away splits, matchup-specific stats).
- **Modeling Agent**: entrena y versiona modelos (ver sección 3).
- **Backtesting Agent**: valida modelos con walk-forward testing temporal (nunca
  random split — los datos son series temporales).
- **Retraining/Monitoring Agent**: detecta drift, dispara reentrenamiento, controla
  over/underfitting.
- **Orchestrator**: coordina los agentes, define pipeline (Airflow/Prefect o dagster),
  logging centralizado.

Cada agente debe poder correr de forma independiente vía CLI y como parte del pipeline
completo.

## 2. FUENTES Y ESTRUCTURA DE DATOS
- Calendario y ubicaciones de partidos (estadio, superficie, techo abierto/cerrado).
- Rosters actualizados por equipo con status de lesión (questionable/out/IR).
- Estadísticas por jugador y equipo, mínimo 5-8 temporadas históricas (regular season
  + playoffs, marcadas por separado).
- Clima histórico y forecast por partido (temperatura, viento, precipitación,
  humedad) — solo relevante para estadios sin techo cerrado.
- Define un esquema de base de datos (PostgreSQL o DuckDB para análisis local) con
  tablas normalizadas: games, teams, players, player_stats_weekly, team_stats_weekly,
  weather, injuries, rosters. Incluye migraciones versionadas.
- Todo dato ingerido debe tener timestamp de "as-of" (snapshot) para evitar data
  leakage — ninguna feature puede usar información no disponible antes del kickoff.

## 3. MODELADO
Implementa y compara múltiples enfoques, no te quedes con uno solo:
- **Supervisado** (core): gradient boosting (XGBoost/LightGBM/CatBoost) para spread y
  totales; modelos separados por tipo de predicción (resultado, spread, over/under,
  props de jugador).
- **Semi-supervisado**: para proyecciones de jugadores con datos incompletos (rookies,
  lesionados con poco histórico) usando técnicas de label propagation o pseudo-labeling.
- **No supervisado**: clustering de estilos de equipo/jugador (para detectar matchups
  atípicos) como feature adicional, no como predictor principal.
- Si usas modelos de deep learning (LSTM/Transformer para secuencias temporales de
  performance), define épocas con early stopping basado en validación temporal, no en
  número fijo de épocas.
- Documenta explícitamente por qué cada modelo es apropiado para su tarea — evita
  complejidad injustificada. Este es un dominio de datos ruidosos con alta varianza
  (NFL), prioriza modelos robustos e interpretables sobre modelos "impresionantes".

## 4. BACKTESTING Y VALIDACIÓN (crítico)
- Walk-forward validation estricta: entrena solo con datos anteriores a la fecha de
  predicción, nunca con información futura.
- Métricas: log-loss, Brier score, calibration curves (no solo accuracy — en apuestas/
  predicción probabilística la calibración importa más que el accuracy crudo).
- Compara contra baselines: línea de casas de apuestas (si está disponible como
  benchmark), modelo naive (home team favorito), modelo Elo simple.
- Reporta performance por segmento: por temporada, por equipo, por condición
  climática, por semana de temporada (early/mid/late/playoffs).
- Implementa un dashboard o reporte automatizado (HTML/notebook) que se regenere en
  cada backtest.

## 5. CONTROL DE OVER/UNDERFITTING Y REENTRENAMIENTO CONTINUO
- Define un pipeline de reentrenamiento programado (semanal, alineado al calendario
  NFL) con validación automática antes de promover un modelo nuevo a producción
  (shadow deployment: el modelo nuevo corre en paralelo antes de reemplazar al actual).
- Usa regularización (L1/L2, max_depth, min_child_weight en GBM) y cross-validation
  temporal (k-fold por temporada, no aleatorio) para tuning de hiperparámetros.
- Monitorea la brecha train/validation/test en cada reentrenamiento; define un umbral
  automático que bloquee la promoción de un modelo si la brecha excede X% (parametrizable).
- Guarda versión de cada modelo con metadata (fecha, features usados, métricas,
  hash de dataset) para poder hacer rollback.

## 6. STACK TÉCNICO SUGERIDO (ajusta si tienes mejor criterio)
- Python 3.11+, Poetry o uv para dependencias.
- DuckDB o PostgreSQL para almacenamiento.
- XGBoost/LightGBM, scikit-learn, optuna para tuning.
- Prefect o Dagster para orquestación.
- MLflow para tracking de experimentos y versionado de modelos.
- pytest con cobertura para cada agente.
- Docker Compose para reproducibilidad local.

## 7. DASHBOARD DE RESULTADOS Y JUSTIFICACIONES
Construye un dashboard web (Streamlit o FastAPI+React, tú decides según velocidad de
desarrollo vs pulido) que muestre, para cada partido próximo y pasado:
- Predicción principal (ganador, spread, total) con probabilidad y nivel de confianza.
- **Justificación explicable por predicción**: top features que más influyeron
  (SHAP values o equivalente), no solo el número final — el usuario debe poder ver
  "por qué" el modelo predijo lo que predijo (ej. "ventaja por descanso +3 días",
  "clima adverso reduce total esperado", "ausencia de WR1 afecta pase").
- Comparación predicción vs resultado real una vez jugado el partido (para partidos
  históricos), con métricas de acierto acumuladas.
- Vista de calibración y performance del modelo en producción (accuracy rolling,
  log-loss por semana) para detectar degradación a simple vista.
- Filtros por equipo, semana, temporada y tipo de predicción (spread/total/props).
- Historial de versiones de modelo usado por predicción (trazabilidad: qué modelo
  generó qué número y cuándo).

El dashboard debe leer directamente de la capa de datos/modelos (no hardcodear
resultados) y actualizarse automáticamente tras cada ciclo de reentrenamiento o
nueva semana de partidos.

## 8. ENTREGABLES ESPERADOS
1. Estructura de repo completa con separación clara por agente/módulo.
2. Scripts de ingesta de datos funcionales (con manejo de rate limits y caché).
3. Pipeline de features documentado.
4. Al menos un modelo entrenado end-to-end con resultados de backtest reproducibles.
5. Dashboard funcional con predicciones, justificaciones (SHAP) y métricas de
   performance del modelo.
6. README técnico explicando arquitectura, decisiones de modelado y cómo correr todo.
7. Plan de reentrenamiento documentado (cron/schedule sugerido).

Antes de escribir código, propón la estructura de carpetas y el esquema de base de
datos, y espera mi confirmación antes de implementar.
```
