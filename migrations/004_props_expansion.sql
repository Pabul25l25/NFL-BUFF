-- ---------------------------------------------------------------------------
-- 004 — Expansión de props: recepciones, touchdowns, yardas de pase, sacks.
--
-- Los datos crudos (receptions, passing_yards, passing_tds, receiving_tds,
-- rushing_tds, sacks_suffered) ya existían en player_stats_weekly desde la
-- ingesta original — nflverse los publica junto al resto de PLAYER_METRICS.
-- Lo que faltaba era el lugar donde guardar sus medias móviles ya desplazadas
-- (el mismo patrón que receiving_yards_r5 / rushing_yards_r5).
-- ---------------------------------------------------------------------------

ALTER TABLE features_player ADD COLUMN IF NOT EXISTS receptions_r3 DOUBLE PRECISION;

ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_yards_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_yards_r3 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_tds_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_tds_r3 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_epa_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_attempts_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS completions_r5 DOUBLE PRECISION;

ALTER TABLE features_player ADD COLUMN IF NOT EXISTS receiving_tds_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS receiving_tds_r3 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS rushing_tds_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS rushing_tds_r3 DOUBLE PRECISION;

-- sacks_suffered: veces que el propio jugador (normalmente el QB) fue capturado.
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS sacks_suffered_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS sacks_suffered_r3 DOUBLE PRECISION;

-- Contexto adicional para el QB: cuánto presiona típicamente la defensa rival.
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS opp_sack_rate_def_r5 DOUBLE PRECISION;
