# NFL-Predictor

> **¿No programas?** Este README es técnico. Empieza en
> [`QUE_ES_ESTO.md`](QUE_ES_ESTO.md) (qué es, cómo funciona y por qué, sin jerga) y
> [`COMO_USARLO.md`](COMO_USARLO.md) (paso a paso para instalarlo y usarlo).

Plataforma de predicción de resultados NFL —**spread, moneyline, totales y props de
jugadores**— construida con foco en **corrección metodológica** antes que en velocidad
de entrega: backtesting walk-forward estricto, calibración probabilística, control
explícito de overfitting y reentrenamiento continuo con gate de promoción.

> **Principio rector**: en un dominio tan ruidoso como la NFL (~285 partidos por
> temporada), un modelo que parece impresionante casi siempre está sobreajustado. Este
> proyecto prioriza modelos robustos e interpretables, y mide su valor contra el único
> listón que importa: **la línea de cierre del mercado**.

---

## Índice

- [Arranque rápido](#arranque-rápido)
- [Arquitectura](#arquitectura)
- [Fuentes de datos](#fuentes-de-datos)
- [El contrato anti-leakage](#el-contrato-anti-leakage)
- [Modelado](#modelado)
- [Backtesting](#backtesting)
- [Reentrenamiento y promoción](#reentrenamiento-y-promoción)
- [Dashboard](#dashboard)
- [Comandos disponibles](#comandos-disponibles)
- [Desarrollo y pruebas](#desarrollo-y-pruebas)
- [Docker](#docker)
- [Limitaciones conocidas](#limitaciones-conocidas)

---

## Arranque rápido

Requiere [`uv`](https://docs.astral.sh/uv/) (gestiona también el intérprete de Python).

```powershell
uv sync                                        # instala dependencias y Python 3.12
uv run nflp db init                            # crea el esquema (migraciones versionadas)

uv run nflp data ingest --seasons 2011:2026    # calendario, rosters, lesiones, snaps (~7 min)
uv run nflp stats build --seasons 2011:2026    # estadísticas de jugador y equipo
uv run nflp data weather --seasons 2011:2026   # clima histórico y pronóstico (~15 min)

uv run nflp features build                     # feature store de partido
uv run nflp features build-players             # feature store de jugador (props)
uv run nflp features audit                     # verificación anti-leakage

uv run nflp model train --target all           # entrena ganador, spread y total
uv run nflp model train-props                  # entrena los props
uv run nflp monitor check                      # promueve de shadow a producción

uv run nflp backtest run --target all --from 2015 --to 2025
uv run nflp predict --season 2026 --week 3     # predicciones con justificación SHAP
uv run nflp dashboard                          # Streamlit en http://localhost:8501
```

O todo de una vez, orquestado por Prefect:

```powershell
uv run nflp pipeline full --seasons 2011:2026
```

Para ver el estado en cualquier momento:

```powershell
uv run nflp db status
uv run nflp model list
```

---

## Arquitectura

Sistema multi-agente. Cada agente es un módulo con **responsabilidad única**, testeable
por separado y ejecutable de forma independiente por CLI o como tarea dentro de los
flows de Prefect.

```
                        ┌──────────────────┐
                        │   Orchestrator   │  Prefect 3
                        │  (flows + logs)  │
                        └────────┬─────────┘
                                 │
     ┌───────────┬───────────┬───┴───────┬────────────┬──────────────┐
     ▼           ▼           ▼           ▼            ▼              ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│  Data   │ │  Stats  │ │Features │ │ Modeling │ │Backtest  │ │ Monitoring │
│ Agent   │ │ Agent   │ │ Agent   │ │  Agent   │ │  Agent   │ │   Agent    │
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

| Agente | Responsabilidad | Módulo |
|---|---|---|
| **Data** | Calendario, estadios, rosters, lesiones, clima | `nflp.agents.data` |
| **Stats** | Estadísticas históricas por jugador y equipo, multi-temporada | `nflp.agents.stats` |
| **Features** | EPA rolling, descanso, viaje, altitud, clima, lesiones, clústeres | `nflp.agents.features` |
| **Modeling** | Entrenamiento, calibración, versionado, SHAP | `nflp.agents.modeling` |
| **Backtesting** | Walk-forward, baselines, calibración, segmentación | `nflp.agents.backtesting` |
| **Monitoring** | Drift PSI, gate de promoción, shadow, rollback | `nflp.agents.monitoring` |

### Estructura del repositorio

```
src/nflp/
├── core/          config tipada, logging, acceso a BD, migraciones
├── agents/        los seis agentes, uno por subpaquete
├── orchestrator/  flows de Prefect
├── registry/      wrapper de MLflow + gate de promoción
├── dashboard/     aplicación Streamlit
└── cli.py         punto de entrada Typer
migrations/        SQL versionado (portable DuckDB ↔ PostgreSQL)
config/            settings.yaml, features.yaml, models.yaml
docs/              arquitectura, esquema, anti-leakage, decisiones de modelado
tests/             pytest, un subpaquete por agente
```

---

## Fuentes de datos

Todo el histórico proviene de **[nflverse](https://github.com/nflverse/nflverse-data)**,
descargado como parquet por HTTPS directo —sin API key, con caché validada por ETag.

| Dato | Fuente | Cobertura verificada |
|---|---|---|
| Calendario, estadios, líneas | `schedules/games.parquet` | 1999–2026 · 7.548 partidos |
| Estadísticas de jugador | `stats_player/stats_player_week_{season}.parquet` | 150 columnas |
| Estadísticas de equipo | `stats_team/stats_team_week_{season}.parquet` | 138 columnas |
| Lesiones | `injuries/injuries_{season}.parquet` | con `date_modified` |
| Snaps | `snap_counts/snap_counts_{season}.parquet` | desde 2012 |
| Rosters semanales | `weekly_rosters/roster_weekly_{season}.parquet` | desde 2002 |
| Clima | [Open-Meteo](https://open-meteo.com/) archive + forecast | sin API key |

Las **coordenadas y altitud de los estadios** no existen en nflverse; se mantienen como
tabla de referencia estática en `src/nflp/agents/data/stadiums.py`, imprescindible para
el clima, la distancia de viaje y el efecto altitud (Denver, +1.600 m).

---

## El contrato anti-leakage

Es la parte más importante del proyecto. Un backtest con leakage produce métricas
excelentes y pérdidas reales.

**Invariantes, verificados por pruebas automáticas:**

1. **Toda** fila del feature store lleva `as_of_ts` = `kickoff − 1 hora`. Ninguna feature
   puede construirse con información posterior.
2. Las medias móviles usan `shift(1)`: agregan **solo semanas estrictamente anteriores**.
   Nunca incluyen el partido objetivo.
3. Las lesiones se resuelven al último parte con `date_modified < as_of_ts`.
4. `result`, `home_score`, `away_score` y `total` están en una **denylist**: si alguna
   aparece en la matriz de features, la prueba falla y el entrenamiento se aborta.
5. **Las líneas de mercado viven en una tabla aparte (`market_lines`) a la que el
   Feature Agent no accede.** El aislamiento es estructural, no una convención: solo el
   Backtesting Agent las lee, y únicamente como benchmark.

Detalle completo en [`docs/anti-leakage.md`](docs/anti-leakage.md).

### Por qué el mercado es benchmark y no feature

Incluir `spread_line` como feature dispara la precisión aparente —pero el modelo se
limita a reaprender la línea, y **deja de ser medible si aporta valor real**. Al
excluirla, cada backtest responde la única pregunta que importa: *¿le ganamos al
mercado?* La respuesta se reporta sin maquillar, incluso cuando es que no.

---

## Modelado

| Tarea | Modelo | Por qué |
|---|---|---|
| Ganador | Gradient boosting binario + calibración isotónica | La calibración importa más que el accuracy crudo |
| Spread | Regresor sobre el margen local → P(cubrir) | Predice margen; la cobertura se deriva |
| Total | Regresor + residuos distribucionales | Permite P(over/under) para cualquier línea |
| Props | Proyección de uso → regresor + distribución | El uso (snaps, target share) domina la varianza |
| Estilos | K-means sobre perfiles de equipo | Detecta matchups atípicos — **feature auxiliar, no predictor** |
| Poca muestra | Pseudo-labeling + shrinkage empírico-Bayes | Rookies y lesionados con histórico corto |

Se entrenan **LightGBM, XGBoost y CatBoost** y gana el mejor en validación temporal, no
el elegido de antemano.

### Por qué no hay deep learning

Es una decisión deliberada y documentada. Con ~285 partidos por temporada, LSTM y
Transformers sobreajustan sin aportar señal frente a un GBM regularizado, y destruyen la
interpretabilidad que el dashboard necesita para justificar cada predicción. El propio
encargo pide priorizar robustez e interpretabilidad sobre modelos impresionantes.
Razonamiento completo en [`docs/decisiones-modelado.md`](docs/decisiones-modelado.md).

---

## Backtesting

**Walk-forward estricto**: para cada (temporada, semana), se entrena únicamente con
partidos anteriores y se predice esa semana. Ventana expansiva. Nunca `random_split` —
son series temporales.

**Métricas**: log-loss, Brier score, ECE y curvas de calibración, MAE/RMSE, ATS%, CRPS
para props.

**Baselines obligatorios**:
1. Naive — el local siempre gana
2. Elo simple con regresión a la media entre temporadas
3. **Línea de cierre del mercado** — el listón real

**Segmentación** (se reporta por separado): temporada, equipo, condición climática y
tramo de temporada (early / mid / late / playoffs).

Cada corrida regenera un **reporte HTML** en `data/reports/`.

### Resultados medidos

Walk-forward sobre **2015-2025**, 66 reentrenamientos, ~3.000 predicciones fuera de
muestra por objetivo. Reproducible con `uv run nflp backtest run --target all`.

**Ganador (moneyline)**

| Modelo | Log-loss | Brier | ECE | Accuracy |
|---|---|---|---|---|
| Mercado (cierre) | **0,6121** | 0,2122 | 0,0181 | 66,1 % |
| **Nuestro modelo** | 0,6329 | 0,2211 | **0,0158** | 65,0 % |
| Elo | 0,6373 | 0,2230 | 0,0321 | 64,2 % |
| Naive (local) | 0,6890 | 0,2479 | 0,0203 | 55,0 % |

**Spread**

| Modelo | MAE | RMSE | ATS % |
|---|---|---|---|
| Mercado (cierre) | **9,807** | 12,721 | 50,97 |
| **Nuestro modelo** | 10,133 | 13,128 | 51,07 |
| Elo | 10,216 | 13,148 | 49,75 |
| Naive (local) | 10,974 | 14,161 | 51,24 |

**Total (over/under)**

| Modelo | MAE | RMSE |
|---|---|---|
| Mercado (cierre) | **10,442** | 13,225 |
| **Nuestro modelo** | 10,701 | 13,624 |
| Media histórica | 10,983 | 13,883 |

**Props** (validación temporal): yardas de recepción MAE 16,30 · CRPS 12,09 ·
yardas de carrera MAE 8,48 · CRPS 7,12.

### Lectura honesta de estos números

El modelo **supera a Elo y al naive en los tres objetivos**, y **pierde contra el mercado**
por un 2,5-3,4 %. Eso es exactamente lo que cabe esperar de un modelo que no ve la línea:
batir de forma consistente a la línea de cierre es muy difícil, y cualquier sistema que
afirme hacerlo con holgura merece una auditoría de leakage antes que un aplauso.

Dos matices que sí son favorables:

- **Nuestra calibración es mejor que la del mercado** (ECE 0,0158 vs 0,0181). Las
  probabilidades son menos certeras pero más fieles a sí mismas.
- **El ATS es 51,07 %**, por debajo del umbral de rentabilidad (52,4 % con cuota −110).
  Dicho claramente: **este sistema no es rentable apostando**, y el reporte lo dice.

---

## Reentrenamiento y promoción

Pipeline semanal alineado al calendario NFL (martes, tras cerrar la semana).

```
entrenar → validar → ¿brecha train/val < umbral? ──no──→ BLOQUEADO (queda registrado)
                             │sí
                             ▼
                      shadow (N semanas)
                             │
                    ¿supera a producción?
                    │sí               │no
                    ▼                 ▼
               promover           descartar
```

- La **brecha train/validación** por encima del umbral (por defecto 15 %) **bloquea** la
  promoción automáticamente.
- Todo modelo nuevo corre en **shadow** en paralelo antes de sustituir al de producción.
- Cada versión guarda metadatos completos (fecha, features, hiperparámetros, métricas,
  `dataset_hash`), lo que permite **rollback** inmediato.
- El drift se mide con PSI por feature y dispara reentrenamiento anticipado.

Detalle en [`docs/plan-reentrenamiento.md`](docs/plan-reentrenamiento.md).

---

## Dashboard

Streamlit, leyendo siempre de la capa de datos —nada hardcodeado.

- Predicción por partido: ganador, spread y total, con probabilidad y nivel de confianza.
- **Justificación explicable**: valores SHAP traducidos a lenguaje natural
  («descanso +3 días», «viento de 32 km/h reduce el total esperado», «ausencia de WR1»).
- Predicción vs. resultado real en partidos ya jugados, con métricas acumuladas.
- Vista de calibración y degradación semanal para detectar deterioro a simple vista.
- Filtros por equipo, semana, temporada y tipo de predicción.
- Historial de versiones: qué modelo generó qué número y cuándo.

---

## Comandos disponibles

```
nflp db init | status | reset                     Esquema y migraciones
nflp data ingest | weather | stadiums             Data Agent
nflp stats build                                  Stats Agent
nflp features build | build-players | audit       Feature Agent
nflp model train | train-props | list             Modeling Agent
nflp model promote | rollback                     Ciclo de vida de modelos
nflp backtest run | props | report                Backtesting Agent
nflp predict                                      Predicciones con SHAP
nflp monitor drift | check                        Monitoring Agent
nflp pipeline weekly | retrain | backtest | full  Orquestador Prefect
nflp dashboard                                    Streamlit
```

Cualquier comando acepta `--help`. Cada agente corre de forma **independiente**, así que
se puede reingerir una temporada sin reentrenar, o reconstruir features sin volver a
descargar nada.

---

## Desarrollo y pruebas

```powershell
uv run pytest                      # pruebas rápidas (sin red)
uv run pytest --cov=nflp           # con cobertura
uv run pytest -m network           # pruebas que sí acceden a internet
uv run ruff check src tests        # linting
```

Las pruebas usan fixtures de parquet recortados en `tests/fixtures/`, así que la suite
principal corre sin conexión y es determinista.

---

## Docker

```bash
docker compose up -d      # postgres + mlflow + prefect + streamlit
```

> **Nota honesta**: `docker-compose.yml` y `Dockerfile` se entregan según lo pedido,
> pero **no han podido verificarse** porque la máquina de desarrollo no tiene Docker
> instalado. Todo el resto del stack —DuckDB, MLflow, Prefect y Streamlit— sí corre y se
> prueba localmente sin contenedores.

---

## Limitaciones conocidas

Se documentan explícitamente en lugar de esconderse:

- **Clima histórico**: se usan observaciones de archivo, no el pronóstico que existía
  antes del kickoff. Es una concesión menor (el pronóstico a 3 días es bastante preciso)
  pero introduce un optimismo leve en el backtest. Marcado en `weather.source`.
- **Props sin línea de mercado**: nflverse no publica líneas de props, así que esos
  modelos se evalúan contra su propia distribución (CRPS) y no contra el mercado.
- **Docker sin verificar** (ver arriba).
- **`pandas` fijado a `>=2.2,<3`**: SHAP y MLflow aún no son fiables con pandas 3.
- **Cobertura de snaps desde 2012**: las temporadas anteriores tienen features de uso
  incompletas, y por eso la ingesta arranca en 2011 y el backtest en 2015.
