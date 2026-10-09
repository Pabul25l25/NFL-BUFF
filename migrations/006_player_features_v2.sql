-- ---------------------------------------------------------------------------
-- 006 — Feature store de jugador v2 (feature_set_version = v2).
--
-- Todas las columnas nuevas respetan el contrato as-of (ver
-- src/nflp/agents/features/player_form.py y tests/test_player_features.py):
-- o salen de primitivas shifted_* (el partido objetivo nunca entra), o de
-- información publicada antes de as_of_ts (parte de lesiones con
-- date_modified < as_of_ts, róster semanal sin el estado INA del día de partido).
-- ---------------------------------------------------------------------------

-- ---- uso, ventana corta (antes target_share_r3 se calculaba y se descartaba) ----
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS target_share_r3 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS air_yards_share_r3 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS carry_share_r3 DOUBLE PRECISION;

-- ---- medias exponenciales desplazadas (vida media 3 partidos) ----
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS target_share_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS air_yards_share_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS carry_share_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS snap_pct_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS receiving_yards_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS rushing_yards_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS receptions_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS targets_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_yards_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS carries_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_attempts_ewm DOUBLE PRECISION;

-- ---- defensa rival por posición (media desplazada de 5 partidos de la defensa) ----
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS opp_rec_yds_allowed_pos_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS opp_receptions_allowed_pos_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS opp_targets_allowed_pos_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS opp_rush_yds_allowed_rb_r5 DOUBLE PRECISION;

-- ---- QB titular esperado del equipo ----
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS team_qb_epa_per_db_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS team_qb_passing_yards_r5 DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS team_qb_passing_yards_ewm DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS team_qb_changed DOUBLE PRECISION;

-- ---- oportunidad vacante y rol ----
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS vacated_target_share DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS vacated_carry_share DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS depth_rank_proxy DOUBLE PRECISION;

-- ---- uso con prior para poca muestra ----
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS target_share_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS carry_share_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS snap_pct_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS targets_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS receptions_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS receiving_yards_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS rushing_yards_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS carries_blend DOUBLE PRECISION;
ALTER TABLE features_player ADD COLUMN IF NOT EXISTS passing_yards_blend DOUBLE PRECISION;
