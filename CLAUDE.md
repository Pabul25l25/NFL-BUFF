# NFL-Predictor — instrucciones para Claude

## Agentes dedicados

Cada cambio se hace con el agente que corresponde (definidos en `.claude/agents/`):

| Tipo de trabajo | Agente |
|---|---|
| Pipeline Python: ingesta, features, modelos, promoción, predicciones (`src/nflp`) | `pipeline-modeling` |
| Frontend Next.js (`web/src/app`, `web/src/components`) | `web-frontend` |
| Datos entre pipeline y web: sync a Neon, migraciones, rutas `/api` | `api-data` |
| Commit, push a GitHub, deploy a Vercel | `release-deploy` |
| Verificación antes de commit o deploy (solo lectura) | `qa-verifier` |
| Revisión de seguridad antes de desplegar código que toca datos, API o secretos | `security-reviewer` |

Orden típico para un cambio: agente de dominio → `qa-verifier` → `release-deploy`.
Si la tarea cruza dominios (por ejemplo, una columna nueva que va del pipeline a la página), divídela entre los agentes que correspondan en ese orden.

## Reglas que no se negocian

- Nunca escribir cadenas de conexión, contraseñas o tokens en archivos del repo ni en mensajes de commit. Solo en variables de entorno o en `web/.env.local` (ignorado por git).
- La web usa un rol de Postgres de solo lectura. Las rutas `/api` solo ejecutan SELECT con parámetros enlazados.
- Validación siempre temporal (walk-forward). Sin splits aleatorios. Sin features que usen el resultado del partido o líneas de mercado.
- Los commits terminan con `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Antes de desplegar a producción: `npm run build` limpio y verificación en vivo con curl.
