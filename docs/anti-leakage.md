# El contrato anti-leakage

Este es el documento más importante del proyecto.

Un backtest con *data leakage* produce métricas espectaculares y pérdidas reales. Es el
modo de fallo por defecto en predicción deportiva, y casi nunca se anuncia: el modelo no
avisa de que hizo trampa, simplemente da un log-loss de 0,45 donde el mercado da 0,61,
y quien lo construyó celebra en vez de sospechar.

**Regla general**: si un modelo supera holgadamente a la línea de cierre, lo más probable
no es que sea bueno, sino que esté mirando el futuro.

---

## Los cinco invariantes

### 1. Toda fila de feature lleva `as_of_ts`

```
as_of_ts = kickoff_utc − as_of_offset_hours   (por defecto, 1 hora)
```

Es el instante hasta el que se permitió mirar datos. Ninguna feature puede construirse
con información posterior. Está en el esquema como columna `NOT NULL` de `features_game`
y `features_player`, no como convención opcional.

El margen de una hora absorbe la imprecisión del horario de kickoff cuando nflverse aún
no lo ha confirmado (partidos futuros sin hora fijada).

### 2. Las agregaciones móviles excluyen el partido objetivo

Toda media móvil pasa por las primitivas de [`contracts.py`](../src/nflp/agents/features/contracts.py):

```python
def shifted_rolling(frame, group_keys, value_column, window, min_periods=1):
    grouped = frame.groupby(group_keys, observed=True, sort=False)[value_column]
    return grouped.transform(
        lambda serie: serie.shift(1).rolling(window=window, min_periods=min_periods).mean()
    )
```

El `.shift(1)` **dentro de cada grupo** es lo que impide que el partido que se está
prediciendo entre en su propia media. Sin él, el EPA del equipo local en la semana 8
incluiría su rendimiento en la semana 8 — el resultado que queremos predecir, filtrado
en la entrada.

Hay una prueba específica que lo verifica de la forma más directa posible: se altera un
valor futuro y se comprueba que **ninguna media anterior cambia**
([`test_leakage.py::test_no_future_information_leaks_backwards`](../tests/test_leakage.py)).

### 3. Las lesiones se resuelven al último parte anterior al corte

El parte de lesiones es la fuente de leakage más traicionera del dominio: **se actualiza
durante toda la semana**, y la versión definitiva suele publicarse a pocas horas del
kickoff. Usar el estado final conocido hoy para predecir un partido de 2018 es leakage
puro, y además del tipo que más infla las métricas, porque el estado final contiene
información que solo existió cuando ya era tarde para apostar.

La solución usa `date_modified`, que nflverse sí publica:

```python
visible = merged["date_modified"].notna() & (merged["date_modified"] < merged["as_of_ts"])
```

Los partes sin fecha se **descartan**. Preferimos subestimar el impacto de las lesiones
antes que arriesgarnos a usar información del futuro.

### 4. La denylist bloquea resultados y mercado

Definida en [`contracts.py`](../src/nflp/agents/features/contracts.py):

| Grupo | Columnas |
|---|---|
| `OUTCOME_COLUMNS` | `result`, `total`, `home_score`, `away_score`, `overtime`, … |
| `MARKET_COLUMNS` | `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`, odds, … |

Además hay patrones sospechosos por expresión regular (`^actual_`, `_actual$`, `^final_`,
`vegas`, `moneyline`, …) que detectan features nuevas con nombres delatores.

`assert_no_leakage()` se ejecuta **antes de cada entrenamiento**, dentro de
`load_game_dataset()`. Si detecta algo, lanza `LeakageError` y el entrenamiento se aborta.
Fallar ahí es infinitamente preferible a entrenar con métricas infladas.

### 5. El mercado vive en otra tabla

Esta es la decisión de diseño de la que más partido se saca.

Las líneas de apuestas **no están en `games`**. Viven en `market_lines`, una tabla
separada que el Feature Engineering Agent **no consulta nunca**. El aislamiento es
estructural, no una convención que alguien pueda olvidar:

```
games          ──┐
team_stats     ──┼──→ Feature Agent ──→ features_game ──→ modelos
injuries       ──┤
weather        ──┘

market_lines   ─────→ Backtesting Agent (solo comparación)
```

El único punto del código que lee `market_lines` es
[`BacktestingAgent._load_context()`](../src/nflp/agents/backtesting/agent.py), y solo
para evaluar el baseline. Hay una prueba que verifica que las dos tablas no comparten
más columna que `game_id`.

---

## Por qué el mercado es benchmark y no feature

Incluir `spread_line` como feature dispara la precisión aparente de inmediato. La línea
de cierre agrega la información de miles de apostadores profesionales, y un modelo que la
recibe como entrada aprende sobre todo a reproducirla.

El problema es que entonces **deja de ser medible si el modelo aporta algo**. Un modelo
market-aware con log-loss 0,60 puede ser excelente o puede ser un pasamanos caro de la
línea; no hay forma de distinguirlo desde dentro.

Al excluirla, cada backtest responde la única pregunta que importa: *¿le ganamos al
mercado?* Y la respuesta se reporta sin maquillar. En la corrida actual sobre 3.018
partidos fuera de muestra:

| Modelo | Log-loss | Brier | ECE | Accuracy |
|---|---|---|---|---|
| Mercado (cierre) | **0,6121** | 0,2122 | 0,0181 | 66,1 % |
| Nuestro modelo | 0,6329 | 0,2211 | **0,0158** | 65,0 % |
| Elo | 0,6373 | 0,2230 | 0,0321 | 64,2 % |
| Naive (local) | 0,6890 | 0,2479 | 0,0203 | 55,0 % |

El modelo supera a Elo y al naive, y **pierde contra el mercado por un 3,4 %**. Eso es
exactamente lo que cabe esperar de un modelo honesto que no ve la línea. El dato
interesante es la columna ECE: nuestras probabilidades están **mejor calibradas** que las
del mercado, aunque sean menos certeras.

---

## Concesiones conocidas

Se documentan en vez de esconderse.

### Clima histórico observado, no pronosticado

Para partidos pasados usamos las observaciones del archivo de Open-Meteo, no el
pronóstico que existía 24 horas antes del kickoff. Estrictamente, el modelo "sabe" que
llovió cuando antes del partido solo se podía saber que probablemente llovería.

- **Impacto**: leve. El pronóstico meteorológico a 1-3 días es bastante preciso en las
  variables que usamos (temperatura, viento), y el clima contribuye poco a la predicción
  comparado con EPA o Elo.
- **Trazabilidad**: queda marcado en `weather.source` (`archive` vs `forecast`).
- **Cómo eliminarlo**: sustituir el archivo por la API de pronósticos históricos de
  Open-Meteo, que devuelve el pronóstico tal como se emitió. Es un cambio acotado a
  `WeatherAgent._fetch_hourly()`.

### El calendario completo se conoce de antemano

`games.parquet` contiene toda la temporada desde antes de que empiece, así que features
de contexto como `away_travel_km` o `is_divisional` se calculan sobre partidos futuros.
Esto **no es leakage**: la NFL publica el calendario en mayo. Es información legítimamente
disponible antes del kickoff.

### Los clústeres de estilo se ajustan sobre todo el histórico

El K-means de estilos se entrena con el conjunto completo de perfiles de equipo-semana.
Los *perfiles* que consume ya están desplazados (usan `shifted_rolling`), así que ninguna
fila ve su propio partido, pero las **fronteras de los clústeres** sí se estiman con datos
de todo el periodo.

- **Impacto**: muy bajo. El clúster entra como feature auxiliar de baja importancia, y lo
  que aporta es una partición del espacio de estilos que es bastante estable en el tiempo.
- **Cómo eliminarlo**: reajustar el K-means dentro de cada bloque del walk-forward. Se
  descartó porque multiplica el coste de la validación por un beneficio marginal.

---

## Cómo verificarlo

```powershell
# Auditoría del feature store completo
uv run nflp features audit

# Batería de pruebas del contrato
uv run pytest tests/test_leakage.py -v
```

La auditoría comprueba, sobre las 4.368 filas del feature store:

1. Que ninguna columna de la denylist aparece en la matriz de features.
2. Que ninguna columna tiene un nombre sospechoso.
3. Que todas las filas tienen `as_of_ts`.
4. Que ninguna tiene `as_of_ts` posterior a su kickoff.

---

## Señales de alarma

Si al trabajar sobre este proyecto observas alguna de estas, sospecha de leakage antes
que de genialidad:

- El modelo supera a la línea de cierre por un margen amplio y estable.
- La métrica de validación es **mejor** que la de entrenamiento de forma consistente
  (el gate de promoción bloquea este caso automáticamente, precisamente por eso).
- Una feature nueva dispara el rendimiento de golpe.
- El ATS supera el 55 % de forma sostenida. El techo realista de los mejores modelos
  públicos ronda el 53-54 %, y el umbral de rentabilidad está en 52,4 %.
