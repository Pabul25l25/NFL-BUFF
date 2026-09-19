# Esquema de base de datos

SQL deliberadamente portable entre **DuckDB** (por defecto) y **PostgreSQL**: solo se usan
`VARCHAR`, `INTEGER`, `DOUBLE PRECISION`, `BOOLEAN`, `DATE` y `TIMESTAMP`, válidos en
ambos motores. Las migraciones viven en [`migrations/`](../migrations/) y las aplica el
runner de [`core/migrations.py`](../src/nflp/core/migrations.py).

```powershell
uv run nflp db init      # aplica las migraciones pendientes
uv run nflp db status    # filas por tabla
```

---

## Mapa general

```
REFERENCIA          teams · stadiums · players
HECHOS              games · rosters_weekly · injuries · player_stats_weekly
                    team_stats_weekly · snap_counts · weather
AISLADA             market_lines                ← solo el Backtesting Agent
FEATURE STORE       features_game · features_player · elo_ratings
                    team_style_clusters
REGISTRO            model_registry · predictions
EVALUACIÓN          backtest_results · calibration_bins
MONITOREO           drift_metrics · retraining_log · production_performance
AUDITORÍA           ingestion_log · schema_migrations
```

---

## 001 — Tablas núcleo

### `teams`
Catálogo de equipos. PK `team_abbr`. Las abreviaturas se unifican al código actual
(`SD`→`LAC`, `OAK`→`LV`, `STL`→`LA`) para que las series históricas no se partan.

### `stadiums`
**Tabla de referencia estática**, sembrada desde
[`agents/data/stadiums.py`](../src/nflp/agents/data/stadiums.py). nflverse publica
`stadium_id` y nombre, pero **no** coordenadas ni altitud.

| Columna | Para qué |
|---|---|
| `latitude`, `longitude` | Open-Meteo y distancia de viaje (haversine) |
| `altitude_m` | Denver (1.610 m) y Ciudad de México (2.200 m) afectan al juego |
| `timezone` | Desfase horario del visitante |
| `roof_type`, `surface` | Valor por defecto del recinto |

Incluye los 30 estadios activos, 7 históricos y 12 internacionales (Londres, México,
Alemania, Brasil, España, Australia).

> `roof_type` aquí es el valor por defecto. Para cada partido se usa el `roof` que publica
> nflverse, porque los techos retráctiles cambian de estado según el día.

### `games`
PK `game_id`. Calendario, contexto y **resultados**.

```sql
game_id, season, week, season_type,          -- REG | POST, separados
gameday, gametime, kickoff_utc, weekday,
home_team, away_team, stadium_id, stadium,
roof, surface, div_game, neutral_site,
home_rest, away_rest,
home_qb_id, away_qb_id, home_coach, away_coach, referee,
home_score, away_score, result, total, overtime   -- ← resultados
```

Las últimas cinco columnas están en la **denylist**: son la materia prima del objetivo,
nunca features. `kickoff_utc` se calcula combinando `gameday` + `gametime` (hora del Este)
y convirtiendo a UTC, con manejo de los saltos de horario de verano.

### `injuries`
PK `(season, week, team, gsis_id)`.

**`date_modified` es la columna crítica del proyecto**: permite reconstruir qué se sabía
del parte de lesiones *antes* del kickoff. Sin ella no habría forma honesta de usar
lesiones como feature.

### `player_stats_weekly` / `team_stats_weekly`
De las 150 columnas de jugador y 138 de equipo que publica nflverse se retienen ~30 y ~26
respectivamente, elegidas por valor predictivo conocido: EPA por fase, CPOE, cuotas de
uso (`target_share`, `air_yards_share`, `wopr`, `racr`).

`season_type` separa regular season de playoffs, como exige el encargo.

### `weather`
PK `game_id`. Solo se puebla para partidos al aire libre.

| `source` | Significado |
|---|---|
| `archive` | Observación histórica de Open-Meteo |
| `forecast` | Pronóstico (partidos dentro de los próximos 16 días) |
| `indoor` | Estadio techado: el clima no aplica, valores nulos |
| `unavailable` | Sin dato (partido futuro lejano o estadio sin coordenadas) |

Distinguir `indoor` de `unavailable` importa: el primero significa "no aplica", el
segundo "no lo sabemos". El modelo puede aprender cosas distintas de cada uno.

### `market_lines` — **la tabla aislada**
PK `game_id`. Spread, total y moneylines de cierre, con sus cuotas.

```sql
game_id, spread_line, total_line,
home_moneyline, away_moneyline,
home_spread_odds, away_spread_odds, over_odds, under_odds
```

**Existe en una tabla separada a propósito.** El Feature Engineering Agent no la consulta
nunca; el único punto del código que la lee es `BacktestingAgent._load_context()`, y solo
como benchmark. El aislamiento es estructural, no una convención. Ver
[anti-leakage.md](anti-leakage.md).

Una prueba verifica que `games` y `market_lines` no comparten más columna que `game_id`.

### `ingestion_log`
Auditoría: fuente, asset, filas, ETag, estado y timestamp de cada ingesta.

---

## 002 — Feature store

### `features_game`
PK `(game_id, feature_set_version)`. Formato ancho y explícito —no EAV— porque el número
de features es estable y así el entrenamiento lee una sola tabla sin pivotar.

**Columnas de control**, presentes en cada fila:

| Columna | Invariante |
|---|---|
| `as_of_ts` | `NOT NULL`. Instante de corte = `kickoff − 1h` |
| `feature_set_version` | Cambiarla invalida el store y obliga a reentrenar |
| `feature_hash` | Hash del conjunto de columnas, para trazar versiones |

Las ~82 features se agrupan en: contexto, descanso y viaje, clima, forma local, forma
visitante, splits por condición, diferenciales de matchup, Elo, lesiones y estilo. El
detalle está comentado en [`002_feature_store.sql`](../migrations/002_feature_store.sql).

### `features_player`
PK `(game_id, gsis_id, feature_set_version)`. Alimenta los modelos de props.

Dos capas: **uso proyectado** (cuota de objetivos, % de snaps, cuota de acarreos) y
**producción reciente** (yardas, EPA, YAC). La primera es la que domina la predicción.

`is_low_sample` marca a los jugadores con menos de 4 partidos previos y dispara la rama
semi-supervisada.

### `elo_ratings`
PK `(game_id, team)`. Formato largo: dos filas por partido, con `elo_pre` y `elo_post`.

Se materializa porque el cálculo es secuencial y estrictamente cronológico —cada partido
actualiza los ratings que usará el siguiente—, y recalcularlo en cada iteración del
walk-forward sería muy costoso.

### `team_style_clusters`
PK `(season, week, team)`. Clúster K-means de estilo. `cluster_id = -1` significa "sin
perfil estable" (poca historia), que es información legítima.

---

## 003 — Registro, evaluación y monitoreo

### `model_registry`
Espejo consultable del registro de MLflow. Existe para que el dashboard y el gate de
promoción no dependan de que el servidor de MLflow esté levantado, y para que un rollback
sea una sentencia SQL.

| Grupo | Columnas |
|---|---|
| Identidad | `model_id`, `target`, `family`, `version` |
| Reproducibilidad | `feature_names`, `hyperparameters`, `dataset_hash`, `feature_set_version` |
| Entrenamiento | `train_seasons`, `n_train_rows`, `n_val_rows` |
| Métricas | `metric_name`, `train_metric`, `val_metric`, `train_val_gap`, `metrics_json` |
| Ciclo de vida | `status`, `promotion_blocked_reason`, `created_at`, `promoted_at`, `archived_at` |

`status` ∈ `{rejected, shadow, production, archived}`.

### `predictions`
**Trazabilidad completa** (requisito de la sección 7): cada número mostrado en el
dashboard apunta al modelo exacto que lo generó y al instante en que se generó.

```sql
prediction_id, game_id, model_id, target, gsis_id,
point_estimate, probability, pred_std, lower_80, upper_80, confidence_level,
shap_values,          -- JSON: {feature: contribución}
shap_base_value,
explanation_es,       -- justificación en lenguaje natural
is_shadow,            -- TRUE = modelo en shadow, no visible al usuario
as_of_ts, predicted_at
```

### `backtest_results`
Una fila por modelo/baseline y segmento. `segment_type` ∈ `{overall, season, team,
weather, week_phase}`.

Incluye métricas probabilísticas (`log_loss`, `brier_score`, `ece`, `accuracy`), de
regresión (`mae`, `rmse`, `crps`) y de apuestas (`ats_win_pct`).

### `calibration_bins`
Puntos de la curva de fiabilidad, para graficar sin recalcular.

### `drift_metrics`
PSI y divergencia KL por feature y semana. `is_drifted` marca los que superan el umbral.

### `retraining_log`
Bitácora auditable de cada decisión del Monitoring Agent: `retrain`, `promote`, `block`,
`rollback`, `shadow_start`, con su motivo.

### `production_performance`
Métricas semana a semana del modelo en producción. Alimenta la vista de degradación.

---

## Volúmenes reales

Tras ingerir 2011-2026 completo:

| Tabla | Filas |
|---|---|
| `games` | 4.368 |
| `market_lines` | 4.144 |
| `players` | 24.824 |
| `rosters_weekly` | 622.573 |
| `snap_counts` | 326.103 |
| `player_stats_weekly` | ~270.000 |
| `injuries` | 81.631 |
| `weather` | 4.341 |
| `features_game` | 4.368 |
| `features_player` | 89.860 |

Fichero DuckDB resultante: ~50 MB.

---

## Cambiar a PostgreSQL

```powershell
$env:NFLP_DATABASE__BACKEND = "postgresql"
$env:NFLP_DATABASE__POSTGRES_URL = "postgresql+psycopg://nflp:nflp@localhost:5432/nflp"
uv run nflp db init
```

Las mismas migraciones se aplican sin cambios. Con Docker Compose el servicio `postgres`
ya queda configurado.
