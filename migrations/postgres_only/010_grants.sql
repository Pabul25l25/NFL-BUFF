-- SOLO Postgres/Neon (DuckDB no soporta GRANT). Ejecutar como el rol dueño,
-- DESPUES de aplicar la migración 010. No lo aplica `nflp db init`.
-- Idempotente. Si el rol tiene ALTER DEFAULT PRIVILEGES para tablas nuevas,
-- este fichero es redundante pero inofensivo.
GRANT SELECT ON pub_predictions_props, pub_parlay_game_corr,
                pub_parlay_game_legs, pub_parlay_calibration
    TO nflp_web_ro;
