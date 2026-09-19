# Decisiones de modelado

Este documento explica **por qué** cada modelo es el que es. El prompt pedía justificar
explícitamente cada elección y evitar complejidad injustificada; lo que sigue es esa
justificación, incluidas las decisiones de *no* hacer algo.

---

## El contexto: un dominio de señal débil

Antes de cualquier elección concreta, conviene fijar las magnitudes del problema, porque
casi todo se deriva de ellas:

| Hecho | Consecuencia |
|---|---|
| ~285 partidos por temporada | Régimen de datos escasos: 16 temporadas son ~4.400 filas |
| El mercado acierta el 66 % | El techo práctico está cerca; el margen de mejora es estrecho |
| Desviación del margen ≈ 13,5 puntos | Una parte grande del resultado es irreducible |
| Un balón suelto cambia el partido | Alta varianza estructural, no ruido de medición |

En un dominio así, **el sobreajuste es el modo de fallo por defecto**, no una posibilidad
remota. Todas las decisiones siguientes están sesgadas hacia la robustez.

---

## Modelos principales

### Ganador (moneyline) — gradient boosting binario + calibración

**Qué**: clasificador binario sobre "gana el local", con calibración de Platt ajustada
sobre un bloque temporal posterior al de entrenamiento.

**Por qué gradient boosting**: las relaciones son no lineales y con interacciones
(la ventaja de descanso importa más en semana corta; el viento importa solo al aire
libre). Un GBM las captura sin que haya que especificarlas, y con regularización fuerte
degrada con elegancia ante el ruido.

**Por qué la calibración importa más que el accuracy**: un modelo que acierta el 65 %
pero dice "80 %" está mintiendo sobre su propia confianza, y sobre esa mentira no se
puede decidir nada. El prompt lo pedía explícitamente y los resultados lo confirman:
nuestro ECE (0,0158) es **mejor que el del mercado** (0,0181) aunque nuestro log-loss sea
peor.

### Spread — regresor sobre el margen

**Qué**: se predice el margen del local (`home_score − away_score`) y de ahí se deriva
P(cubrir) usando la distribución de residuos.

**Por qué predecir el margen y no la cobertura directamente**: el margen es una variable
continua con mucha más información que un binario. Entrenar sobre "cubrió / no cubrió"
tira toda la señal de *por cuánto*, y además ataría el modelo a la línea del mercado,
que hemos decidido no usar.

**Por qué objetivo MAE y no MSE**: la NFL produce palizas ocasionales (márgenes de 40+).
Con MSE, esos partidos dominan el gradiente y el modelo se deforma para acomodarlos. El
MAE es robusto a esa cola.

### Total (over/under) — regresor + residuos distribucionales

**Qué**: se predice la suma de puntos y se acompaña de una distribución de residuos que
permite responder P(over) para **cualquier** línea, no solo la del mercado.

**Por qué distribucional**: un total se resuelve contra una línea. Un modelo que acierta
la media pero se equivoca en la dispersión produce probabilidades over/under malas aunque
su MAE sea excelente.

---

## Comparación de familias

Se entrenan **LightGBM, XGBoost y CatBoost** y gana la que mejor valida temporalmente, no
la elegida de antemano. Los resultados sobre el objetivo `winner`:

| Familia | Log-loss train | Log-loss val | Brecha |
|---|---|---|---|
| LightGBM | 0,5810 | 0,6225 | +7,1 % |
| XGBoost | 0,5810 | 0,6250 | +7,6 % |
| **CatBoost** | 0,6039 | **0,6203** | **+2,7 %** |

CatBoost gana de forma consistente, y no por casualidad: su *ordered boosting* está
diseñado precisamente para reducir el sobreajuste en datasets pequeños, que es
exactamente nuestro caso. También es el que menor brecha train/validación produce.

---

## Por qué NO hay deep learning

Es una decisión deliberada, y el prompt la contemplaba explícitamente
(«evita complejidad injustificada», «prioriza modelos robustos e interpretables sobre
modelos impresionantes»).

**1. El volumen no da.** Un LSTM o un Transformer con capacidad suficiente para modelar
secuencias temporales de rendimiento tiene más parámetros que muestras tenemos. Con 4.400
partidos, la red memoriza.

**2. La señal es demasiado débil.** Las arquitecturas secuenciales brillan cuando hay
estructura temporal rica que explotar. Aquí la autocorrelación del rendimiento de un
equipo entre semanas es real pero modesta, y una media móvil exponencial ya la captura
casi por completo. La capacidad adicional se dedica a memorizar ruido.

**3. Destruye la interpretabilidad que el dashboard necesita.** El requisito de la
sección 7 es que el usuario vea *por qué*. Los valores SHAP sobre árboles son **exactos**
(algoritmo TreeSHAP) y se calculan en milisegundos; sobre redes son aproximados, caros y
sensibles a la elección de la distribución de referencia.

**4. No hay evidencia de que gane en este dominio.** El estado del arte público en
predicción NFL —FiveThirtyEight, nflfastR, los modelos de mercado— usa regresión
logística, Elo y gradient boosting. Si las redes tuvieran ventaja aquí, ya se sabría.

**Cuándo reconsiderarlo**: si se incorporaran datos de tracking jugada a jugada (Next Gen
Stats a nivel de trayectoria), el volumen crecería en varios órdenes de magnitud y la
estructura secuencial sería real. Ahí sí tendría sentido.

---

## Componente no supervisado: clústeres de estilo

**Qué**: K-means sobre seis dimensiones que describen *cómo* juega un equipo (tendencia al
pase, ritmo, protección, presión defensiva, precisión, disciplina con el balón), no *cómo
de bueno* es —la calidad ya la capturan las features de EPA.

**Por qué**: los equipos no son intercambiables. Una defensa que vive de presionar al QB
se comporta distinto contra una ofensiva de pases cortos y rápidos que contra una
vertical. Un identificador de estilo permite al árbol capturar interacciones de
emparejamiento que las medias de EPA, por sí solas, no expresan.

**Qué NO se hace**: usarlo como predictor principal, ni interpretar los grupos como si
tuvieran significado causal. Son etiquetas que ayudan a partir el espacio de features,
nada más. Su importancia SHAP es baja y eso está bien.

Los equipos con menos de 3 partidos previos reciben clúster `-1`, que es información
legítima: el modelo puede aprender que un equipo sin perfil estable se comporta distinto.

---

## Componente semi-supervisado: pseudo-labeling en props

**Qué**: para jugadores con menos de 4 partidos de histórico (rookies, recién llegados de
una lesión), se entrena un modelo base solo con jugadores de muestra amplia, se generan
pseudo-etiquetas y se reentrena incluyéndolas con peso reducido (0,3).

**Por qué el peso es bajo**: las pseudo-etiquetas son predicciones, no observaciones.
Tratarlas como verdad amplifica los errores del modelo base y crea un bucle de
retroalimentación. El peso reducido las convierte en un regularizador, que es lo único
que pueden ser honestamente.

**La honestidad del caso**: se implementa porque el encargo lo pide, pero el control que
lo acompaña es un **shrinkage empírico-Bayes** hacia la media de la posición —la idea
clásica de que la media de un jugador con 2 partidos no es fiable y conviene acercarla a
la de su grupo. Es metodológicamente más defendible, más barato y más interpretable.

El backtest de props compara ambos enfoques más un baseline de media móvil, y el
dashboard muestra cuál gana. Si el shrinkage iguala o supera al pseudo-labeling —que es
lo habitual en este dominio— el resultado se reporta tal cual.

---

## Control de over/underfitting

### Regularización por defecto, agresiva

Los hiperparámetros de partida son deliberadamente conservadores:

| Parámetro | Valor | Motivo |
|---|---|---|
| `max_depth` / `depth` | 3 | Árboles casi de decisión simple |
| `num_leaves` | 7 | Limita la complejidad efectiva |
| `min_child_samples` | 80 | Exige soporte amplio antes de dividir |
| `colsample_bytree` | 0,5 | Decorrelaciona árboles con ~82 features |
| `reg_alpha` (L1) | 2,0 | Poda features irrelevantes |
| `reg_lambda` (L2) | 10,0 | Penaliza pesos grandes |
| `learning_rate` | 0,02 | Aprendizaje lento, más estable |

Estos valores no son arbitrarios: los valores iniciales, más permisivos, producían
brechas train/validación del 40-48 % y el gate de promoción los **bloqueaba**. La
respuesta correcta fue regularizar más, no relajar el umbral.

### Validación cruzada temporal, nunca aleatoria

El tuning con Optuna usa k-fold **por temporada** con ventana expansiva. Cada fold entrena
con todas las temporadas anteriores y valida en la siguiente. Un `KFold` aleatorio sobre
series temporales mezcla 2024 en entrenamiento y 2016 en validación, produce métricas
excelentes y elige hiperparámetros que sobreajustan.

### El gate de promoción

```
brecha = (métrica_validación − métrica_entrenamiento) / |métrica_entrenamiento|
```

Si `brecha > 15 %`, el modelo queda marcado como `rejected` y **no puede promoverse**,
por buenas que sean sus métricas absolutas.

También se bloquea el caso simétrico: si la validación es **más del 15 % mejor** que el
entrenamiento, es señal de leakage o de un corte temporal mal hecho. Un modelo no debería
rendir mejor en datos que no ha visto.

### Corte a tres bandas

```
[······ entrenamiento ······][ calibración ][ validación ]
```

Hacen falta tres bloques, no dos. El calibrador debe ajustarse sobre predicciones que el
modelo no vio al entrenar; pero si además se *evalúa* sobre ese mismo conjunto, la métrica
sale optimista porque el calibrador ya se adaptó a esos datos.

**Esto se detectó en la práctica.** Con dos bloques, la calibración isotónica parecía
mejorar el log-loss. Al medirla de forma honesta sobre un tercer bloque, lo **empeoraba**
de 0,621 a 0,728: con ~285 partidos, la isotónica ajusta una escalera que no generaliza.
De ahí el umbral de 1.500 muestras por debajo del cual se usa Platt automáticamente.

---

## Elo: baseline y feature a la vez

Elo cumple dos funciones y merece mención propia porque suele infravalorarse.

Como **baseline** es un listón duro: 64,2 % de acierto y log-loss 0,6373 en nuestro
backtest, a solo un 0,7 % de nuestro modelo completo con 82 features. Cualquiera que
presente un modelo de ML para NFL sin compararlo contra Elo está ocultando información.

Como **feature** es legítima y no introduce leakage: el rating *previo* de cada equipo se
calcula únicamente con partidos ya jugados, en orden cronológico estricto.

La implementación sigue a FiveThirtyEight: multiplicador por margen de victoria
amortiguado (para que las palizas no disparen el rating), ventaja de campo de 55 puntos
Elo (~2,2 puntos de spread) y regresión a la media del 25 % entre temporadas.

---

## Props de conteo disperso (touchdowns, sacks)

Los props se ampliaron de dos (yardas de recepción y de carrera) a ocho, sumando
recepciones, touchdowns de recepción y carrera, yardas de pase, touchdowns de pase y
sacks recibidos por el QB. Los datos crudos ya estaban ingeridos desde el principio
(`PLAYER_METRICS` en `stats/agent.py` ya los incluía); solo faltaba conectar las
medias móviles y el objetivo en `props.py`.

**Limitación conocida y no resuelta**: touchdowns y sacks son conteos de eventos poco
frecuentes —la mayoría de partidos terminan en 0—, no variables continuas como las
yardas. `PropDistribution` (log-normal desplazada) sigue dando un rango y una
P(over/under) razonables, pero es una aproximación menos fiel aquí que en yardas: un
conteo disperso se modela mejor con Poisson o binomial negativa. No se implementó un
modelo de conteo dedicado por alcance del cambio; queda marcado en
`props.SPARSE_COUNT_PROPS` y avisado en el dashboard. Sería el siguiente paso natural
si estos props se vuelven críticos.

## Qué se probó y no funcionó

Documentar los callejones sin salida ahorra tiempo a quien venga después.

- **Calibración isotónica con una temporada de datos**: empeora el log-loss fuera de
  muestra (0,621 → 0,728). Sustituida por Platt con umbral automático.
- **Hiperparámetros permisivos** (`max_depth=4`, `num_leaves=15`, `reg_lambda=2`):
  brechas del 40-48 %, bloqueadas por el gate. La solución fue regularizar, no ceder.
- **Distribución normal para props**: las yardas de carrera pueden ser negativas y tienen
  cola derecha larga. Una normal produce probabilidades absurdas en líneas bajas. Se usa
  log-normal desplazada, con desplazamiento calculado a partir de los datos.
