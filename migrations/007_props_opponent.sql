-- ---------------------------------------------------------------------------
-- 007 — Rival y localía en pub_predictions_props.
--
-- Para partidos aún no jugados `player_stats_weekly` no tiene filas, así que
-- `team` salía NULL y la web mostraba el rival equivocado. Ahora el publish
-- toma equipo/rival/localía de `features_player` y los guarda aquí.
-- ---------------------------------------------------------------------------

ALTER TABLE pub_predictions_props ADD COLUMN IF NOT EXISTS opponent_team VARCHAR;
ALTER TABLE pub_predictions_props ADD COLUMN IF NOT EXISTS is_home BOOLEAN;
