# Arquitectura

## Principio de diseño

Sistema multi-agente donde **cada agente tiene una responsabilidad única** y puede
ejecutarse de forma aislada. Prefect aporta orquestación, reintentos y observabilidad,
pero no es un requisito: cualquier agente corre por CLI sin que haya nada más levantado.

Esa independencia no es estética. Significa que se puede reingerir una temporada sin
reentrenar, reconstruir features sin volver a descargar, o depurar el cálculo de Elo sin
levantar un servidor.

```
                        ┌──────────────────┐
                        │   Orchestrator   │  Prefect 3
                        │  flows + logging │
                        └────────┬─────────┘
                                 │
     ┌───────────┬───────────┬───┴───────┬────────────┬──────────────┐
     ▼           ▼           ▼           ▼            ▼              ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│  Data   │ │  Stats  │ │Features │ │ Modeling │ │Backtest  │ │ Monitoring │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘
     │           │           │           │            │             │
     └───────────┴───────────┴─────┬─────┴────────────┴─────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  DuckDB / PostgreSQL         │
                    │  + MLflow (artefactos)       │
                    └──────────────┬───────────────┘
                                   ▼
                          ┌─────────────────┐
                          │   Dashboard     │  Streamlit
                          └─────────────────┘
```

---

## Los agentes

### Data Agent — `nflp.agents.data`

Ingesta y normalización. Es la frontera entre el mundo exterior, que es sucio, y el resto
del sistema, que asume datos limpios.

| Responsabilidad | Detalle |
|---|---|
| Descarga | Parquet de nflverse por HTTPS, con caché validada por ETag |
| Reintentos | `tenacity` con backoff exponencial; 404 por temporada no publicada se omite con aviso |
| Normalización | `'grass '` → `grass`, `SD` → `LAC`, kickoff ET → UTC |
| Estadios | Tabla de referencia estática con lat/lon/altitud/zona horaria |
| Clima | Open-Meteo, **una petición por estadio-temporada** |
| **Separación del mercado** | Las líneas van a `market_lines`, no a `games` |

**Decisión de diseño — el clima por estadio-temporada**: la implementación ingenua hace
una petición por partido (~7.000) y agota la cuota. La primera versión agrupaba por
estadio con el rango completo de 15 años, y Open-Meteo devolvía `429` porque pondera cada
petición por volumen de datos. La versión final trocea por estadio y temporada, pidiendo
solo la ventana real de partidos: ~650 peticiones ligeras, sin un solo 429.

### Stats Agent — `nflp.agents.stats`

Estadísticas semanales de jugador y equipo, multi-temporada, con regular season y
playoffs **marcados por separado** (`season_type`). Mezclarlos sesga las medias móviles:
los playoffs tienen otra distribución —mejores equipos, sin partidos irrelevantes de
final de temporada—.

De las 150 columnas que publica nflverse por jugador se retienen ~30, elegidas por tener
valor predictivo conocido (EPA por fase, CPOE, cuotas de uso). Arrastrarlo todo añade
ruido, no información.

### Feature Engineering Agent — `nflp.agents.features`

El agente con más superficie. Construye ocho familias de features:

| Familia | Ejemplos | Fuente de verdad |
|---|---|---|
| Contexto | divisional, sede neutral, techo, superficie, altitud | calendario (conocido en mayo) |
| Descanso y viaje | días de descanso, semana corta, salida de bye, km, husos | calendario + estadios |
| Clima | temperatura, viento, precipitación, humedad | Open-Meteo (solo al aire libre) |
| Forma | EPA por fase en ventanas 3/5/8 + exponencial | `shifted_rolling` |
| Matchup | diferenciales ataque-vs-defensa | derivadas |
| Elo | rating previo de cada equipo | cronológico estricto |
| Lesiones | índice ponderado por gravedad y posición | snapshot as-of |
| Estilo | clúster K-means | auxiliar |

**Métricas defensivas**: nflverse no publica "EPA permitido". Se deriva cruzando cada
partido consigo mismo — el EPA defensivo de un equipo es el EPA ofensivo que generó su
rival en ese mismo partido.

Todo pasa por el [contrato anti-leakage](anti-leakage.md), que se verifica antes de
escribir nada en el feature store.

### Modeling Agent — `nflp.agents.modeling`

Entrena, compara familias, calibra, explica y versiona. Ver
[decisiones-modelado.md](decisiones-modelado.md).

Estructura interna:

```
modeling/
├── datasets.py     construcción de (X, y) + cortes temporales
├── models.py       familias GBM y espacios de búsqueda
├── calibration.py  Platt/isotónica + distribuciones de residuos
├── explain.py      SHAP → frases en español
├── predict.py      predicción con trazabilidad
├── props.py        modelos de props + distribución log-normal
├── props_agent.py  entrenamiento y validación de props
├── baselines.py    naive, Elo, mercado
└── agent.py        orquestación del entrenamiento
```

### Backtesting Agent — `nflp.agents.backtesting`

Walk-forward con ventana expansiva. Para cada bloque de semanas, entrena con **todo lo
estrictamente anterior** y predice ese bloque. 66 reentrenamientos cubren 11 temporadas.

Evalúa contra los tres baselines obligatorios y segmenta por temporada, tramo, clima y
equipo. Genera un reporte HTML autocontenido en cada corrida.

### Retraining/Monitoring Agent — `nflp.agents.monitoring`

- **Drift**: PSI por feature contra las últimas cinco temporadas de referencia.
- **Rendimiento en producción**: métricas semana a semana de predicciones ya resueltas.
- **Promoción**: un shadow solo sustituye a producción si **mejora**; empatar no basta.
- **Rollback**: si producción se degrada más del umbral, vuelve a la versión anterior.

---

## Capa de datos

### Por qué DuckDB por defecto

- Sin servidor: el proyecto arranca con `uv sync` y nada más.
- Lee parquet remoto directamente, lo que simplifica la ingesta.
- Analítica columnar, que es exactamente el patrón de acceso del feature store.
- Un fichero de ~50 MB con 16 temporadas completas.

PostgreSQL está soportado mediante `NFLP_DATABASE__BACKEND=postgresql` y las mismas
migraciones, para el despliegue con Docker Compose.

### Por qué migraciones SQL y no Alembic

Decisión consciente, documentada por si extraña. `duckdb-engine` tiene huecos conocidos
con el autogenerate y el mapeo de tipos de Alembic. El SQL plano que escribimos es válido
en ambos motores, y el runner de [`migrations.py`](../src/nflp/core/migrations.py) aporta
lo que realmente se necesita: versionado, orden, idempotencia y **verificación de
checksum** —editar una migración ya aplicada lanza un error explícito en vez de dejar dos
bases divergentes—.

### Esquema

Ver [esquema-datos.md](esquema-datos.md) para el detalle completo. La estructura general:

```
referencia:  teams · stadiums · players
hechos:      games · rosters_weekly · injuries · player_stats_weekly ·
             team_stats_weekly · snap_counts · weather
aislada:     market_lines                    ← solo la lee el Backtesting Agent
features:    features_game · features_player · elo_ratings · team_style_clusters
registro:    model_registry · predictions · backtest_results · calibration_bins
monitoreo:   drift_metrics · retraining_log · production_performance
auditoría:   ingestion_log · schema_migrations
```

---

## Registro de modelos: dos capas

**MLflow** es la fuente de verdad de artefactos y métricas de experimento.
**La tabla `model_registry`** espeja el estado del ciclo de vida.

La duplicación es deliberada: el dashboard y el gate de promoción consultan estado
constantemente, y no deberían depender de que el servidor de MLflow esté levantado. Un
rollback es una sentencia SQL.

Ciclo de vida: `rejected` → `shadow` → `production` → `archived`. Ningún modelo pasa
directo a producción.

> **Nota**: el backend de ficheros de MLflow (`./mlruns`) quedó en modo mantenimiento en
> MLflow 3 y rechaza escrituras. Se usa SQLite (`sqlite:///data/mlflow.db`).

---

## Flujo de datos completo

```
nflverse (parquet)  ──┐
Open-Meteo (JSON)   ──┼──→ Data + Stats Agents ──→ tablas de hechos
                      │
                      └──→ market_lines (aislada)
                                  │
tablas de hechos ──→ Feature Agent ──→ features_game
                          │                features_player
                          │                elo_ratings
                          ▼
                   auditoría anti-leakage  ← aborta si falla
                          │
                          ▼
                   Modeling Agent ──→ model_registry + MLflow
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          Backtesting          Predictor
                │                   │
                │                   ▼
                │            predictions (+ SHAP)
                ▼                   │
        backtest_results            │
        calibration_bins            │
                └─────────┬─────────┘
                          ▼
                      Dashboard
```

---

## Decisiones transversales

**Configuración tipada con sobrescritura por entorno.** Todo vive en `settings.yaml` y
cualquier valor se sobrescribe con `NFLP_SECCION__CLAVE`. Pydantic valida al arrancar,
así que un error de configuración falla de inmediato y no a mitad de un entrenamiento.

**Importaciones perezosas en la CLI.** LightGBM, SHAP, MLflow y Prefect se importan
dentro de cada comando. `nflp --help` responde al instante.

**Los NaN no se imputan.** Los GBM los manejan de forma nativa, y un valor ausente es
información: "no hay dato de clima" significa estadio techado, "no hay historia" significa
primera semana. Rellenar con la media destruye esa señal.

**Escrituras idempotentes.** Todas las cargas usan `upsert` con clave explícita.
Reejecutar cualquier comando es seguro.

**Logging en español, identificadores en inglés.** Los mensajes los lee una persona; los
identificadores los lee el intérprete y deben seguir la convención del ecosistema.
