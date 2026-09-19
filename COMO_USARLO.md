# Cómo usar NFL-Predictor (guía sin tecnicismos)

Esta guía asume que no sabes programar. Solo vas a copiar y pegar líneas de texto en
una ventana. No necesitas entender lo que hacen para que funcionen.

> 💡 Si algo no funciona igual que aquí se describe, ve a la sección
> **[¿Algo no funcionó?](#algo-no-funcionó)** al final.

---

## Parte 1 — Ver el resultado ahora mismo

Si el proyecto ya está corriendo en esta computadora (pregúntale a quien te la entregó),
solo tienes que:

1. Abrir tu navegador (Chrome, Edge, el que uses).
2. Escribir en la barra de direcciones: `http://localhost:8501`
3. Presionar Enter.

Va a aparecer una página llamada **NFL-Predictor** con varias pestañas. Puedes explorar
sin miedo a romper nada: no hay ningún botón que borre datos.

Si esa dirección no carga nada, sigue con la Parte 2.

---

## Parte 2 — Abrir el programa (cada vez que reinicies la computadora)

El programa vive dentro de esta computadora, no en internet. Cada vez que la apagas, se
"apaga" con ella. Para volver a encenderlo:

### Paso 1: Abrir la Terminal

La Terminal es una ventana donde se escriben instrucciones en vez de hacer clic en
botones. En Windows se llama **PowerShell**.

- Presiona la tecla de Windows (⊞) en el teclado.
- Escribe `PowerShell`.
- Haz clic en el resultado "Windows PowerShell".

Se abre una ventana con fondo azul o negro y texto. Es normal, no es un error.

### Paso 2: Ir a la carpeta del proyecto

Copia esta línea (selecciónala con el mouse y presiona `Ctrl + C`), haz clic dentro de
la ventana de PowerShell, pega con clic derecho, y presiona `Enter`:

```
cd D:\NFL
```

### Paso 3: Encender el programa

Copia, pega y presiona Enter:

```
uv run nflp dashboard
```

Vas a ver texto apareciendo en la ventana durante unos 10-20 segundos. Cuando termine,
el navegador se abre solo. Si no se abre, ve tú mismo a `http://localhost:8501`.

### Paso 4: Dejarlo abierto

**No cierres la ventana negra de PowerShell** mientras uses el programa — es lo que lo
mantiene encendido. Puedes minimizarla.

### Para apagarlo

Haz clic dentro de la ventana negra y presiona `Ctrl + C`.

---

## Parte 3 — Actualizar todo con un solo comando

Los datos no se actualizan solos. Cada semana, después de que se jueguen los partidos,
hay que "avisarle" al programa para que traiga la información nueva, vuelva a estudiar
con los resultados frescos y recalcule las predicciones. Es como actualizar una app del
teléfono, pero manual.

**Un solo comando hace todo eso, en el orden correcto.** Abre PowerShell (Parte 2,
Pasos 1-2) y ejecuta:

```
cd D:\NFL
```
```
uv run nflp pipeline weekly
```

⏳ Tarda entre 2 y 4 minutos. Vas a ver bastante texto pasar — es normal, es el programa
reportando cada paso. Mientras corre, por dentro está haciendo, en este orden:

1. Trae el calendario, las lesiones, los rosters y las estadísticas nuevas.
2. Trae el clima de los próximos partidos.
3. Recalcula los datos internos que usa el modelo (partidos y jugadores).
4. Vuelve a "estudiar" con los resultados más recientes.
5. Decide si la versión nueva del modelo es mejor que la que ya estaba en uso — y
   **solo la reemplaza si de verdad mejora**. Si no mejora, sigue usando la anterior.
6. Genera las predicciones nuevas de partidos y de jugadores, ya con el modelo correcto.

Cuando termine, abre (o refresca con `F5`) el dashboard en el navegador. Ya vas a ver
todo actualizado.

> 📝 **No hace falta que memorices ningún otro comando para el uso semanal.** Este
> reemplaza a los 5-6 comandos que antes había que pegar uno por uno.

<details>
<summary>📋 Ver el detalle paso a paso (por si quieres correr solo una parte)</summary>

Si alguna vez necesitas correr solo un paso (por ejemplo, solo traer el clima sin
reentrenar nada), estos son los comandos por separado:

```
uv run nflp data ingest --seasons 2026        # calendario, lesiones, rosters
uv run nflp stats build --seasons 2026        # estadísticas de jugadores y equipos
uv run nflp data weather --seasons 2026 --force   # clima de los próximos partidos
uv run nflp features build                    # recalcula datos internos (partidos)
uv run nflp features build-players            # recalcula datos internos (jugadores)
uv run nflp model train --target all          # reentrena partidos
uv run nflp model train-props                 # reentrena jugadores
uv run nflp monitor check                     # decide si usar el modelo nuevo
uv run nflp predict --season 2026             # genera todas las predicciones
```

`uv run nflp pipeline weekly` es exactamente estos 9 pasos, en este orden, automáticos.

</details>

---

## Parte 4 — Ver y obtener estadísticas de jugadores (recepciones, touchdowns, yardas de pase...)

Además de predecir quién gana cada partido, el programa proyecta cuánto va a producir
un jugador concreto. Hoy cubre ocho tipos de estadística:

|     Estadística         |             A quién aplica                 |
|-------------------------|--------------------------------------------|
| Yardas de recepción     | Receptores, alas cerradas, corredores      |
| Yardas de carrera       | Corredores, mariscales de campo que corren |
| Recepciones             | Receptores, alas cerradas, corredores      |
| Touchdowns de recepción | Receptores, alas cerradas, corredores      |
| Touchdowns de carrera   | Corredores, mariscales de campo            |
| Yardas de pase          | Mariscales de campo (QB)                   |
| Touchdowns de pase      | Mariscales de campo (QB)                   |
| Veces capturado (sacks) | Mariscales de campo (QB)                   |

> Los touchdowns y los sacks son sucesos poco frecuentes (la mayoría de partidos son
> 0). El dashboard lo avisa: la proyección ahí es más aproximada que en yardas, que sí
> varían de forma continua partido a partido.

### Verlas en el dashboard

1. Abre el dashboard (Parte 2) y ve a la pestaña **"Props de jugador"**.
2. Elige el tipo de estadística con los botones de arriba.
3. En los filtros de la izquierda, elige la Temporada y la Semana que quieras ver.
4. Vas a ver una tabla con cada jugador, su proyección, y un rango de "seguramente
   entre X e Y". Haciendo clic en un jugador de la lista de abajo puedes ver
   también su justificación (por qué se espera ese número).

> Estas proyecciones se generan automáticamente cada vez que corres
> `uv run nflp pipeline weekly` (Parte 3) — no hace falta pedirlas aparte.

### Obtenerlas a mano para una semana o jugador concreto

Si quieres regenerarlas sin correr todo el pipeline semanal (por ejemplo, para probar
algo puntual), en PowerShell:

```
cd D:\NFL
```
```
uv run nflp predict --target passing_yards --season 2026 --week 3
```

Cambia `passing_yards` por cualquiera de: `receiving_yards`, `rushing_yards`,
`receptions`, `receiving_tds`, `rushing_tds`, `passing_tds`, `sacks_suffered`.
Cambia el número después de `--week` por la semana que quieras. Si quitas
`--week 3` por completo, te da las de toda la temporada. Para regenerar las ocho
de una sola vez: `uv run nflp predict --target props --season 2026`.

### ¿Se pueden agregar más estadísticas?

Sí — la forma en que está armado el programa permite sumar otros tipos (yardas después
de la atajada, primeros downs, intercepciones lanzadas, etc.) sin rehacer nada. Es
cuestión de pedirlo.

---

## Parte 5 — Instalarlo en una computadora nueva, desde cero

Solo necesario si mueves el proyecto a una PC que nunca lo ha tenido.

**1. Copia la carpeta** `D:\NFL` completa a la nueva computadora (USB, nube, como
prefieras).

**2. Instala la herramienta base** (`uv`). Abre PowerShell y pega:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Esto instala automáticamente todo lo necesario, incluido el lenguaje de programación
(Python), sin que tengas que buscarlo ni instalarlo por separado.

**3. Cierra y vuelve a abrir PowerShell** para que reconozca lo instalado.

**4. Entra a la carpeta:**
```
cd D:\NFL
```

**5. Instala todas las piezas del programa** (descarga ~190 componentes automáticamente,
5-10 minutos la primera vez):
```
uv sync
```

**6. Crea la base de datos vacía:**
```
uv run nflp db init
```

**7. Descarga TODO el historial (16 temporadas), lo procesa, entrena los modelos y
genera las primeras predicciones — un solo comando, 20-30 minutos:**
```
uv run nflp pipeline full --seasons 2011:2026
```

<details>
<summary>📋 Si prefieres verlo paso a paso en vez de un solo comando</summary>

```
uv run nflp data ingest --seasons 2011:2026      # calendario, lesiones, rosters (5-10 min)
uv run nflp stats build --seasons 2011:2026      # estadísticas de jugadores y equipos
uv run nflp data weather --seasons 2011:2026     # clima histórico (10-15 min)
uv run nflp features build                       # datos internos de partidos
uv run nflp features build-players                # datos internos de jugadores
uv run nflp model train --target all              # entrena los modelos de partido
uv run nflp model train-props                     # entrena los modelos de jugador
uv run nflp monitor check                         # pone los modelos en funcionamiento
uv run nflp predict --season 2026                 # genera las primeras predicciones
```

</details>

**8. Enciende el programa:**
```
uv run nflp dashboard
```

A partir de aquí, esa computadora funciona igual que cualquier otra: Parte 2 para
abrirlo, Parte 3 para actualizarla cada semana con un solo comando.

---

## Tarjeta resumen (para imprimir o guardar)

```
ABRIR EL PROGRAMA
  1. Abrir PowerShell
  2. cd D:\NFL
  3. uv run nflp dashboard
  4. Ir a http://localhost:8501 en el navegador

ACTUALIZAR CADA SEMANA (dentro de D:\NFL, en PowerShell)
  uv run nflp pipeline weekly

VER ESTADISTICAS DE UN JUGADOR CONCRETO (opcional)
  uv run nflp predict --target receiving_yards --season 2026 --week 3
  uv run nflp predict --target rushing_yards --season 2026 --week 3
```

---

## ¿Algo no funcionó?

| Lo que ves | Qué significa | Qué hacer |
|---|---|---|
| "uv no se reconoce como un comando" | Falta instalar la herramienta base | Repite el Paso 2 de la Parte 5 |
| La página del navegador no carga | El programa no está encendido | Repite la Parte 2 |
| "Port 8501 is not available" | Ya hay una ventana con el programa abierto | Ve directo a `http://localhost:8501`, no hace falta abrirlo de nuevo |
| Algún comando tarda "demasiado" | Es normal la primera vez (descarga datos de internet) | Espera; si pasan más de 20 minutos, avisa a soporte técnico |
| Aparece texto en rojo | Puede ser solo un aviso, no siempre es un error grave | Si el comando termina y no se traba, generalmente está bien |
| `uv run nflp pipeline weekly` no encuentra partidos nuevos | Todavía no se jugó la semana | Es normal a principio de semana; los resultados aparecen después de que se jueguen |
| "Sin proyecciones" en la pestaña de jugadores | No has corrido la actualización de esa semana | Corre `uv run nflp pipeline weekly` (Parte 3) |

Si nada de esto resuelve el problema, guarda una foto de lo que aparece en la pantalla y
consulta con quien te entregó el proyecto — con esa foto pueden diagnosticarlo rápido.

Para entender **qué es** este programa y **por qué** se construyó así, lee
[`QUE_ES_ESTO.md`](QUE_ES_ESTO.md).
