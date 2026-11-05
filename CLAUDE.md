# NFL-Predictor — instrucciones para Claude

## Agentes dedicados

Cada cambio se hace con el agente que corresponde (definidos en `.claude/agents/`):

| Tipo de trabajo | Agente |
|---|---|
### Agentes del sistema de predicción (uno por módulo de `src/nflp/agents/`)

| Módulo | Agente |
|---|---|
| Ingesta: calendario, equipos, rosters, lesiones, clima (`agents/data`) | `data-agent` |
| Estadísticas históricas por jugador y equipo (`agents/stats`) | `stats-agent` |
| Feature store y auditoría anti-leakage (`agents/features`) | `features-agent` |
| Entrenamiento, calibración, props, SHAP (`agents/modeling`) | `modeling-agent` |
| Walk-forward y baselines (`agents/backtesting`) | `backtesting-agent` |
| Drift, degradación, promoción y rollback (`agents/monitoring`) | `monitoring-agent` |
| Registro de modelos y gate de promoción (`src/nflp/registry`) | `registry-agent` |
| Flows de Prefect, comando semanal y tarea programada (`orchestrator`) | `orchestrator-agent` |

### Agentes de aplicación y entrega

| Tipo de trabajo | Agente |
|---|---|
| Frontend Next.js (`web/src/app`, `web/src/components`) | `web-frontend` |
| Datos entre pipeline y web: sync a Neon, migraciones, rutas `/api` | `api-data` |
| Commit, push a GitHub, deploy a Vercel | `release-deploy` |
| Verificación antes de commit o deploy (solo lectura) | `qa-verifier` |
| Revisión de seguridad antes de desplegar código que toca datos, API o secretos | `security-reviewer` (en `~/.claude/agents`) |

Orden típico para un cambio: agente de dominio → `qa-verifier` → `release-deploy`.
Si la tarea cruza dominios (por ejemplo, una columna nueva que va de `features-agent` a `modeling-agent` y de ahí a `api-data` y `web-frontend`), divídela entre los agentes que correspondan en ese orden, uno a la vez.

## Reglas que no se negocian

- Nunca escribir cadenas de conexión, contraseñas o tokens en archivos del repo ni en mensajes de commit. Solo en variables de entorno o en `web/.env.local` (ignorado por git).
- La web usa un rol de Postgres de solo lectura. Las rutas `/api` solo ejecutan SELECT con parámetros enlazados.
- Validación siempre temporal (walk-forward). Sin splits aleatorios. Sin features que usen el resultado del partido o líneas de mercado.
- Los commits van solo a nombre del dueño del repo: sin línea `Co-Authored-By` ni ninguna otra atribución a Claude o a agentes, en mensajes de commit ni en PRs.
- Antes de desplegar a producción: `npm run build` limpio y verificación en vivo con curl.
