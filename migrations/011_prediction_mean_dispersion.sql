-- ---------------------------------------------------------------------------
-- 011 — Media de la distribución servida y dispersión de la binomial negativa.
--
-- * mean_estimate: valor esperado de la distribución que se sirve para el
--   prop. En continuos es la media implícita de la rejilla p05..p95 (integral
--   de la función cuantil con colas extrapoladas linealmente en escala normal y
--   el mismo suelo que la rejilla); en Poisson es λ y en binomial negativa μ.
--   No sustituye a point_estimate (mediana L1 en continuos).
-- * nb_dispersion: tamaño r de la binomial negativa (Var = μ + μ²/r), ajustado
--   por máxima verosimilitud en la temporada de calibración. NULL si la
--   distribución servida no es binomial negativa.
--
-- Se añaden en `predictions` (local) y en su copia publicada
-- `pub_predictions_props` (la que lee la web).
-- ---------------------------------------------------------------------------

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS mean_estimate DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS nb_dispersion DOUBLE PRECISION;

ALTER TABLE pub_predictions_props ADD COLUMN IF NOT EXISTS mean_estimate DOUBLE PRECISION;
ALTER TABLE pub_predictions_props ADD COLUMN IF NOT EXISTS nb_dispersion DOUBLE PRECISION;
