# Plan de reentrenamiento

## Calendario

El ciclo está alineado al calendario NFL. La ventana buena es el **martes**: ya se han
cerrado todos los partidos del fin de semana (incluido el Monday Night), nflverse ha
publicado las estadísticas actualizadas, y los primeros partes de lesiones de la semana
siguiente empiezan a salir el miércoles.

| Cuándo | Qué | Comando |
|---|---|---|
| **Martes 06:00** | Ciclo semanal completo | `nflp pipeline weekly` |
| **Miércoles 06:00** | Refresco de lesiones y clima | `nflp data ingest --seasons 2026` + `nflp data weather` |
| **Jueves / Domingo 09:00** | Repredicción con partes actualizados | `nflp predict --season 2026 --week N` |
| **Martes (quincenal)** | Reentrenamiento con evaluación de shadow | `nflp pipeline retrain` |
| **Fin de temporada** | Reentrenamiento completo con tuning | `nflp pipeline retrain --tune` |
| **Pretemporada** | Backtest completo y revisión de features | `nflp backtest run --target all` |

**Por qué el reentrenamiento es quincenal y no semanal**: una semana añade ~16 partidos a
un conjunto de ~4.400. El modelo no cambia de forma apreciable, y reentrenar cada semana
multiplica el riesgo de introducir una regresión sin beneficio medible. La repredicción
sí es más frecuente, porque las features (lesiones, clima) cambian aunque el modelo no.

## Programación

### Windows — Programador de tareas

```powershell
schtasks /create /tn "NFLP semanal" /tr "cmd /c cd /d D:\NFL && uv run nflp pipeline weekly" /sc weekly /d TUE /st 06:00
schtasks /create /tn "NFLP predicciones" /tr "cmd /c cd /d D:\NFL && uv run nflp predict --season 2026" /sc weekly /d THU,SUN /st 09:00
```

### Linux / macOS — cron

```cron
0 6 * * 2   cd /opt/nflp && uv run nflp pipeline weekly           >> logs/weekly.log 2>&1
0 6 * * 3   cd /opt/nflp && uv run nflp data ingest --seasons 2026 >> logs/injuries.log 2>&1
0 9 * * 4,0 cd /opt/nflp && uv run nflp predict --season 2026     >> logs/predict.log 2>&1
0 7 * * 2   cd /opt/nflp && uv run nflp pipeline retrain          >> logs/retrain.log 2>&1
```

### Prefect (recomendado en producción)

```powershell
uv run prefect server start                     # terminal aparte
uv run prefect deploy --all
uv run prefect deployment run "nflp-semanal/weekly"
```

Prefect aporta reintentos con backoff, historial de ejecuciones, alertas ante fallo y
visualización del DAG. Los flows están definidos en
[`orchestrator/flows.py`](../src/nflp/orchestrator/flows.py).

---

## El ciclo de vida de un modelo

```
        entrenar
           │
           ▼
    ¿brecha train/val < 15%?
     │ no              │ sí
     ▼                 ▼
 RECHAZADO         SHADOW  ──── predice en paralelo, marcado is_shadow=TRUE
 (registrado           │
  con motivo)          │  tras ≥2 semanas
                       ▼
              ¿supera a producción?
               │ no            │ sí
               ▼               ▼
           descartar       PRODUCCIÓN ──→ el anterior pasa a ARCHIVADO
                                 │
                                 │  si se degrada >10%
                                 ▼
                             ROLLBACK ──→ vuelve el archivado
```

### El gate de promoción

```
brecha = (métrica_validación − métrica_entrenamiento) / |métrica_entrenamiento|
```

Si `brecha > max_train_val_gap` (por defecto 15 %), el modelo se registra con estado
`rejected` y `promote()` lo **rechaza explícitamente**. No se puede saltar a mano; hay una
prueba que lo verifica.

También se bloquea el caso simétrico: validación más de un 15 % **mejor** que
entrenamiento indica leakage o un corte temporal mal hecho.

### Shadow deployment

Un modelo en shadow predice sobre los mismos partidos que el de producción, con
`is_shadow = TRUE`. El usuario no ve sus números; el sistema sí los compara.

La promoción exige **mejorar**, no empatar: cambiar de modelo tiene coste operativo y
riesgo, así que la carga de la prueba recae en el candidato.

### Rollback

`check_degradation()` compara el rendimiento reciente (últimas 6 semanas) con la métrica
de validación registrada. Si la degradación supera el 10 %, revierte automáticamente y
deja constancia en `retraining_log`.

Manual:

```powershell
uv run nflp model rollback spread
```

---

## Disparadores de reentrenamiento anticipado

Además del calendario, hay cuatro señales que justifican reentrenar antes de tiempo:

1. **Drift generalizado** — `should_retrain=True` cuando al menos el 10 % de las features
   (mínimo 3) superan PSI 0,20.
2. **Degradación sostenida** — la media móvil de 4 semanas empeora más del 10 % respecto
   a la referencia.
3. **Cambio estructural conocido** — cambio de reglamento, expansión del calendario,
   nuevo formato de playoffs. Esto no lo detecta ningún PSI: lo decide una persona.
4. **Cambio en el feature store** — features nuevas o corregidas invalidan los modelos
   existentes. Al subir `features.version`, hay que reentrenar todo.

### Sobre el drift al principio de temporada

Es esperable y **no** indica un problema. En las primeras semanas las medias móviles se
calculan con muy pocos partidos, así que sus distribuciones difieren naturalmente de la
referencia histórica.

En la comprobación actual (2026, semana 2) aparecen 41 features con drift, encabezadas
por `temperature_c` y las medias de EPA de ventana corta. Es exactamente el patrón
esperable con dos jornadas disputadas. Conviene tratar el drift como señal a partir de la
semana 6.

---

## Versionado y reproducibilidad

Cada versión registra:

| Campo | Para qué sirve |
|---|---|
| `feature_names` | Saber con qué entrenó |
| `hyperparameters` | Reproducir el ajuste |
| `dataset_hash` | Verificar que los datos son los mismos |
| `train_seasons`, `n_train_rows` | Contexto del entrenamiento |
| `train_metric`, `val_metric`, `train_val_gap` | Evaluar el gate |
| `promotion_blocked_reason` | Entender por qué se rechazó |
| `mlflow_run_id` | Enlazar con el artefacto |

**Reproducibilidad**: mismo `dataset_hash` + mismos hiperparámetros + misma semilla
(`modeling.random_seed = 1729`) implica métricas idénticas. Si no coinciden, algo cambió
aguas arriba y hay que averiguar qué.

---

## Runbook de incidencias

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| Todos los modelos salen `rejected` | Hiperparámetros demasiado permisivos | Regularizar más; **no** subir el umbral del gate |
| `LeakageError` al entrenar | Feature nueva con nombre o contenido prohibido | Revisar `contracts.py`; el error indica la columna |
| Log-loss mejora de golpe | Casi siempre, leakage | Auditar la feature nueva antes de celebrar |
| 429 de Open-Meteo | Rango de fechas demasiado amplio | Ya troceado por temporada; subir `weather.request_delay_seconds` |
| 404 de nflverse | Temporada aún no publicada | Normal; se omite con aviso |
| Drift masivo en semana 1-4 | Medias móviles con poca muestra | Ignorar hasta la semana 6 |
| Predicciones vacías | No hay modelo en producción | `nflp model list` y promover, o `nflp monitor check` |

---

## Coste de cada operación

Medido en la máquina de desarrollo (Windows, sin GPU):

| Operación | Tiempo |
|---|---|
| Ingesta completa 2011-2026 | ~7 min (primera vez); ~30 s con caché |
| Clima completo | ~15 min (troceado por estadio-temporada) |
| Feature store de partido | ~10 s |
| Feature store de jugador | ~20 s |
| Entrenar los 3 objetivos | ~30 s |
| Entrenar con `--tune` | ~15 min |
| Backtest walk-forward (1 objetivo) | ~60 s |
| Ciclo semanal completo | ~3 min |
