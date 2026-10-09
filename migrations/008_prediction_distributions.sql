-- ---------------------------------------------------------------------------
-- 008 — Distribución por jugador y contexto del partido en `predictions`.
--
-- Los props v2 sirven una distribución que depende del jugador:
--   * continuos (yardas, recepciones): rejilla de 19 cuantiles p05..p95,
--     monótona y calibrada de forma conformal;
--   * conteos (touchdowns, sacks): Poisson con parámetro λ.
-- `lower_80`, `upper_80` y `point_estimate` se mantienen como hasta ahora
-- para la web. Equipo, rival y localía se guardan con la predicción para que
-- un parlay del mismo partido pueda agrupar jugadores sin volver a
-- `features_player`. En predicciones de partido estas columnas quedan NULL.
-- ---------------------------------------------------------------------------

-- JSON: lista de 19 valores [p05, p10, ..., p95].
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS quantiles VARCHAR;
-- λ de la Poisson (solo props de conteo con distribución Poisson).
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS poisson_lambda DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS team VARCHAR;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS opponent_team VARCHAR;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS is_home BOOLEAN;
