# NFL-Predictor — explicado sin tecnicismos

Este documento explica el proyecto para alguien que no programa: **qué es**, **qué
partes tiene**, **qué herramientas usa y por qué se eligió cada una**, y **para qué
sirve realmente** (y para qué NO sirve).

Si lo que buscas es solo usarlo día a día, ve a [`COMO_USARLO.md`](COMO_USARLO.md).

---

## 1. ¿Qué es esto, en una frase?

Un programa que **descarga automáticamente el historial de la NFL** (16 años de
partidos), **aprende patrones** de esos datos, y **predice** quién va a ganar cada
partido, por cuánto margen, cuántos puntos totales se van a anotar, y cuántas yardas
va a hacer un jugador concreto — explicando siempre **por qué** llegó a esa conclusión.

Piénsalo como un analista deportivo muy metódico que:
- Nunca se cansa de revisar estadísticas.
- Nunca hace trampa mirando el resultado antes de predecir.
- Siempre te dice qué factores tuvo en cuenta ("el equipo local descansó 3 días más",
  "hace mucho viento y eso baja el marcador esperado").
- Es honesto cuando se equivoca o cuando pierde contra las casas de apuestas.

---

## 2. ¿Para qué sirve? (y para qué no)

### Sirve para

- **Ver predicciones razonadas** de los próximos partidos, con el nivel de confianza
  del programa (alta / media / baja).
- **Entender el "por qué"** detrás de cada predicción, en frases normales, no en
  números crípticos.
- **Comparar** qué tan bueno es el programa contra métodos simples (como "siempre gana
  el local") y contra las líneas de las casas de apuestas.
- **Ver el historial**: qué predijo en partidos pasados y si acertó.
- **Proyectar estadísticas de jugadores** de **próximos** partidos —yardas de
  recepción y de carrera, recepciones, touchdowns, yardas de pase y sacks recibidos
  por el mariscal de campo— con un rango probable, no solo un número seco.

### NO sirve para

- **Ganar dinero apostando de forma garantizada.** Esto se explica con toda claridad
  más abajo: el programa, medido honestamente, **pierde** contra las casas de apuestas
  por un margen pequeño. Se construyó para ser honesto sobre esto, no para venderte una
  ilusión.
- **Predecir con certeza absoluta.** El fútbol americano tiene mucho azar (un balón que
  rebota hacia un lado o el otro cambia partidos). Ningún sistema, ni el de las casas de
  apuestas profesionales, elimina esa incertidumbre.

---

## 3. ¿De dónde saca la información?

Todo dato que usa el programa viene de fuentes públicas y gratuitas, sin necesidad de
pagar ninguna suscripción ni clave de acceso:

| Fuente | Qué aporta |
|---|---|
| **nflverse** | Un proyecto de código abierto (gratuito, mantenido por aficionados y analistas) que recopila calendarios, resultados, estadísticas de cada jugador y equipo, lesiones y quién juega, desde 1999 hasta hoy. |
| **Open-Meteo** | Un servicio gratuito de clima que da la temperatura, el viento y la lluvia de cada estadio al aire libre, tanto de partidos pasados como del pronóstico de los próximos. |

El programa descarga esta información automáticamente — no hay que buscarla a mano en
ningún lado, ni copiar datos de una página web.

**Un detalle importante para la honestidad del sistema**: la línea de apuestas de las
casas (el "spread", el "over/under") también viene de estos datos, pero **se guarda
aparte y nunca se usa para enseñarle al programa a predecir**. Se usa solo al final,
como comparación — el examen contra el que se mide el programa, no un chuleta que se le
da de antemano. Esto se explica en la sección 6.

---

## 4. ¿Cómo "aprende" a predecir?

En vez de programar reglas fijas ("si el equipo A tiene mejor récord, gana"), se le
muestran al programa **miles de partidos pasados con su resultado real**, y el programa
mismo encuentra los patrones que mejor explican quién ganó y por cuánto.

Es parecido a cómo un aficionado experimentado, tras ver miles de partidos, "intuye"
quién va a ganar sin poder explicar la regla exacta — solo que aquí el programa sí puede
decirte qué factores pesó más.

Los factores que el programa aprende a valorar incluyen, entre otros ochenta:

- Qué tan bien ha jugado cada equipo en sus últimos partidos (ataque y defensa).
- Cuántos días de descanso tiene cada equipo antes de este partido.
- Cuánto tuvo que viajar el equipo visitante.
- El clima esperado (viento fuerte y frío bajan el marcador esperado).
- Si hay jugadores lesionados o en duda, y qué tan importantes son esos jugadores.
- La altitud del estadio (jugar en Denver o Ciudad de México es distinto a nivel del
  mar).
- Un "ranking de fuerza" de cada equipo que se actualiza partido a partido (parecido al
  ranking Elo del ajedrez).

**Una regla que se cuidó con mucho detalle**: el programa **nunca** puede ver el
resultado de un partido, ni ningún dato posterior a la hora de inicio, al construir la
predicción de ese mismo partido. Suena obvio, pero es el error más común (y más fácil de
cometer sin darse cuenta) al construir este tipo de sistemas — y produce predicciones
que parecen geniales en las pruebas pero fallan en la vida real. Hay pruebas
automáticas que verifican esto constantemente.

---

## 5. Las piezas del programa (los "agentes")

El proyecto está dividido en seis módulos independientes, cada uno con un trabajo
específico — como un equipo donde cada persona tiene su rol:

| Módulo | Su trabajo, en una frase |
|---|---|
| **Recolector de datos** | Descarga y limpia calendarios, lesiones, clima, plantillas. |
| **Estadístico** | Organiza las estadísticas históricas de jugadores y equipos. |
| **Constructor de pistas** | Convierte los datos crudos en las ~80 "pistas" que el modelo usa para predecir (descanso, forma reciente, clima, etc.). |
| **Entrenador de modelos** | Le "enseña" al programa a predecir con esas pistas, y revisa que no esté memorizando en vez de aprendiendo. |
| **Evaluador (backtesting)** | Pone a prueba al modelo con partidos históricos que nunca vio, como un examen sorpresa, y compara los resultados contra las casas de apuestas. |
| **Vigilante (monitoreo)** | Revisa periódicamente si el modelo se está "oxidando" con el tiempo y decide si conviene actualizarlo. |
| **Orquestador** | Ejecuta los otros seis, en el orden correcto, con un solo comando. |

Cada uno se puede usar por separado — eso permite, por ejemplo, traer solo el clima sin
reentrenar nada. Pero para el uso normal de cada semana no hace falta llamarlos uno por
uno: el **Orquestador** los encadena automáticamente con `uv run nflp pipeline weekly`
(ver [`COMO_USARLO.md`](COMO_USARLO.md)). El orden importa, y se cuidó a propósito:
primero se trae la información nueva, después se reentrena, después el Vigilante decide
si el modelo nuevo reemplaza al anterior, y **solo al final** se generan las
predicciones — así siempre quedan firmadas por la versión correcta del modelo, nunca
por la que está a punto de quedar obsoleta.

---

## 6. Decisiones clave — y por qué se tomaron así

### 6.1 La línea de apuestas se guarda aparte, nunca se usa para enseñar

Si le mostráramos al programa la línea de la casa de apuestas mientras aprende, el
programa haría trampa: aprendería a copiar esa línea en vez de razonar por sí mismo. Sus
números se verían espectaculares en las pruebas, pero no significarían nada — sería
como enseñarle a un estudiante las respuestas del examen antes de dárselo.

Por eso, la línea de apuestas se guarda en un lugar separado del resto de la información,
al que el "constructor de pistas" no tiene acceso. Solo se usa al final, como la nota de
un examen que el programa no vio de antemano.

### 6.2 Se compara siempre contra métodos simples

Antes de creerle a un modelo complicado, hay que asegurarse de que le gana a algo
simple. Por eso el programa siempre se compara contra tres referencias:

1. **"Siempre gana el local"** — la predicción más tonta posible.
2. **Un ranking de fuerza simple** (parecido al de ajedrez) — sin inteligencia
   artificial, solo aritmética.
3. **La línea de la casa de apuestas** — el verdadero examen difícil.

### 6.3 Se usan tres "cerebros" distintos y gana el mejor, no el primero

En vez de comprometerse con un solo tipo de modelo, se entrenan tres variantes distintas
(llamadas LightGBM, XGBoost y CatBoost — nombres de librerías de programación, no
importan los nombres) y se queda el que mejor predice partidos que nunca vio. No se
elige a dedo de antemano.

### 6.4 No se usó "inteligencia artificial de última generación" (redes neuronales)

Esto es deliberado, y quizás sorprenda: se decidió explícitamente **no** usar las
técnicas de IA más avanzadas (del tipo que usa ChatGPT), aunque suenan más
impresionantes. La razón es sencilla: esas técnicas necesitan **enormes** cantidades de
datos para funcionar bien, y la NFL solo juega ~285 partidos por temporada. Con tan
pocos datos, una técnica muy potente termina "memorizando" los partidos pasados en vez
de aprender patrones generales — como un estudiante que memoriza las respuestas de un
examen anterior en vez de entender la materia, y luego reprueba el examen nuevo.

Se prefirió una técnica más simple pero más confiable en este tipo de problema, y que
además permite explicar sus decisiones con claridad (ver 6.6).

### 6.5 Un "portero" que bloquea modelos que memorizan en vez de aprender

Cada vez que se entrena una versión nueva, un mecanismo automático compara qué tan bien
predijo los partidos con los que "estudió" contra qué tan bien predice partidos nuevos
que nunca vio. Si hay una diferencia muy grande, es señal de que memorizó en vez de
aprender, y esa versión **se bloquea automáticamente** — no se pone en uso, aunque en el
papel se vea muy bien.

Esto no es teórico: durante la construcción, varias versiones fueron bloqueadas por
esta razón, y la solución fue simplificar el modelo, no bajar la exigencia del filtro.

### 6.6 Cada predicción viene con su explicación

No basta con que el programa diga "gana el equipo A, 65% de probabilidad". Para cada
predicción, calcula automáticamente **qué factores pesaron más** y los traduce a una
frase en español, del estilo: *"A favor del local: viene de 3 días más de descanso,
mejor rendimiento reciente. En contra: el rival tiene mejor defensa contra el pase."*

### 6.7 Antes de usarse, un modelo nuevo "hace su prueba" en silencio

Cuando se entrena una versión mejorada, no reemplaza de inmediato a la que está en uso.
Primero corre "en la sombra" — genera sus propias predicciones en paralelo, sin que se
muestren — durante un tiempo. Solo si demuestra ser mejor que la versión actual, la
reemplaza. Si la versión actual empieza a fallar más de la cuenta, el sistema puede
regresar automáticamente a la versión anterior que funcionaba bien.

### 6.8 Ocho estadísticas de jugador, no solo dos

Al principio el programa solo proyectaba yardas de recepción y de carrera. Se amplió a
ocho: se sumaron recepciones, touchdowns de recepción y de carrera, yardas de pase,
touchdowns de pase y sacks recibidos por el mariscal de campo. Los datos para todas ya
venían incluidos en la fuente pública que usa el programa; lo que hacía falta era
conectarlos, siguiendo exactamente el mismo patrón ya probado con las dos primeras.

Una salvedad honesta: touchdowns y sacks son sucesos poco frecuentes (la mayoría de
partidos terminan en 0), así que la proyección ahí es más aproximada que en yardas,
que varían de forma continua. El dashboard lo avisa cuando corresponde.

### 6.9 Las estadísticas de jugador ahora se proyectan también para partidos futuros

Al principio, el programa solo podía calcular las yardas proyectadas de un jugador para
partidos que **ya se habían jugado** — útil para revisar aciertos, inútil para lo que
de verdad importa: saber qué esperar del partido del domingo que viene.

La causa era sutil: la proyección se armaba a partir de las estadísticas del propio
partido, que obviamente no existen todavía para un partido futuro. Se corrigió
cambiando la base de cálculo a **la lista de jugadores que va a jugar esa semana**
(el róster), combinada con su historial de partidos anteriores — que es, en el fondo,
la forma correcta y más honesta de proyectar algo que todavía no pasó.

### 6.10 El sistema se revisa a sí mismo antes de actuar sobre poca información

Durante una revisión de rutina se encontró que, con solo una semana de partidos
jugados en la temporada, el sistema estuvo a punto de **revertir un modelo** pensando
que había empeorado — cuando en realidad solo estaba viendo el ruido normal de 16
partidos, no una tendencia real. Se corrigió para que ese tipo de decisiones exija
varias semanas de evidencia real antes de actuar, no solo la misma semana contada
varias veces. Se documenta aquí porque es exactamente el tipo de error que este
proyecto se propuso evitar desde el principio, y encontrarlo y corregirlo confirma que
las revisiones funcionan.

### 6.11 El dashboard mostraba el mismo partido varias veces

Cada vez que se reentrena un modelo, el programa **conserva las predicciones de
versiones anteriores** en vez de borrarlas —es la trazabilidad de la que habla la
sección 5—, pero el panel visual no estaba eligiendo cuál de esas versiones mostrar, así
que un mismo partido podía aparecer repetido con porcentajes distintos. Se corrigió para
que la vista principal muestre siempre la predicción más reciente; el historial completo
de versiones sigue disponible, ahora en su propia pestaña.

---

## 7. Las herramientas usadas y por qué esas

No hace falta entender esto para usar el programa, pero por transparencia:

| Herramienta | Para qué se usó | Por qué se eligió |
|---|---|---|
| **Python** | El lenguaje en que está escrito todo el programa | Es el estándar de la industria para este tipo de análisis de datos |
| **uv** | Instala y organiza todas las piezas del programa | Simplifica la instalación a un solo comando, sin pasos manuales |
| **DuckDB** | La "base de datos" donde se guarda toda la información | No necesita instalar un servidor aparte; funciona con un solo archivo |
| **LightGBM / XGBoost / CatBoost** | Los tres "cerebros" que aprenden a predecir | Son robustos con pocos datos y permiten explicar sus decisiones (ver 6.6) |
| **SHAP** | Calcula qué factores pesaron en cada predicción | Es el método más confiable para explicar decisiones de este tipo de modelo |
| **MLflow** | Guarda un historial ordenado de cada versión del modelo entrenada | Permite comparar versiones y volver atrás si una falla |
| **Prefect** | Encadena automáticamente los pasos (traer datos → calcular → predecir) | Evita tener que ejecutar cada paso a mano y reintenta si algo falla |
| **Streamlit** | Construye la página web (el "dashboard") que ves en el navegador | Permite mostrar gráficos y tablas interactivas sin construir una app desde cero |

---

## 8. ¿Qué tan bueno es, honestamente?

Se probó el programa "prediciendo" 11 temporadas completas de partidos ya jugados
(2015-2025), como si en cada semana solo tuviera la información disponible hasta ese
momento — nunca haciendo trampa mirando resultados futuros.

| Comparado contra... | ¿Le ganamos? |
|---|---|
| "Siempre gana el local" | ✅ Sí, claramente |
| Un ranking de fuerza simple (tipo ajedrez) | ✅ Sí, por un margen pequeño |
| La línea de las casas de apuestas | ❌ No — perdemos por un margen pequeño (2,5-3,4%) |

**Esto es exactamente lo esperable y es una señal de que el sistema es honesto.** Las
casas de apuestas profesionales invierten enormes recursos en fijar sus líneas, y
superarlas de forma consistente es extremadamente difícil — tan difícil que si un
sistema casero afirmara lograrlo con facilidad, lo más probable es que tuviera un error
(por ejemplo, estar mirando información que no debería).

Un dato interesante a favor del programa: aunque acierta un poco menos que la casa de
apuestas, **cuando dice "70% de probabilidad", acierta más cerca de ese 70%** que la
propia casa de apuestas. Es decir, es un poco menos certero pero un poco más honesto
sobre su propia confianza.

**Conclusión práctica**: es una herramienta útil para entender partidos y ver
predicciones razonadas, no una máquina de ganar apuestas.

---

## 9. Preguntas frecuentes

**¿Necesito internet para usarlo?**
Solo para traer datos nuevos cada semana (Parte 3 de `COMO_USARLO.md`). Para ver el
dashboard con lo ya descargado, no.

**¿Cuesta dinero alguna de las herramientas que usa?**
No. Todo lo que usa el programa —los datos, las herramientas de programación— es
gratuito y de código abierto.

**¿Mis datos o mis apuestas quedan guardados en algún lado de internet?**
No. Todo vive únicamente en tu computadora, en la carpeta `D:\NFL`. El programa no
manda información a ningún servidor externo (excepto para *descargar* los datos
públicos de partidos y clima).

**¿Puedo pedir que agregue otro tipo de predicción (por ejemplo, otro tipo de
jugada)?**
Sí, el sistema está construido en piezas separadas justamente para poder ampliarlo sin
rehacer todo. Es cuestión de pedirlo.

**¿Por qué a veces dice "confianza baja"?**
Es literal: cuando la diferencia entre los equipos es pequeña, el partido está cerca de
ser un "lanzamiento de moneda", y el programa lo dice en vez de fingir seguridad que no
tiene.
