-- ---------------------------------------------------------------------------
-- 010 — Publicación de parlays y distribuciones de props para la web.
--
-- (009 crea en la base local las tablas parlay_game_corr / parlay_game_legs /
-- parlay_calibration; aquí van sus copias aplanadas `pub_*` que lee la web.)
-- Los permisos de lectura para el rol de la web no van aquí porque DuckDB no
-- soporta GRANT: ver migrations/postgres_only/010_grants.sql.
-- ---------------------------------------------------------------------------

-- Distribución por jugador (ver 008): 19 cuantiles p05..p95 en JSON, o lambda.
ALTER TABLE pub_predictions_props ADD COLUMN IF NOT EXISTS quantiles VARCHAR;
ALTER TABLE pub_predictions_props ADD COLUMN IF NOT EXISTS poisson_lambda DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS pub_parlay_game_corr (
    game_id    VARCHAR NOT NULL,
    model_id   VARCHAR,
    legs       VARCHAR,   -- JSON: lista de nombres de piernas
    matrix     VARCHAR,   -- JSON: matriz de correlación 2D
    as_of_ts   TIMESTAMP,
    PRIMARY KEY (game_id)
);

CREATE TABLE IF NOT EXISTS pub_parlay_game_legs (
    game_id        VARCHAR NOT NULL,
    model_id       VARCHAR,
    p_home_win     DOUBLE PRECISION,
    margin_mu      DOUBLE PRECISION,
    margin_sigma   DOUBLE PRECISION,
    total_mu       DOUBLE PRECISION,
    total_sigma    DOUBLE PRECISION,
    as_of_ts       TIMESTAMP,
    PRIMARY KEY (game_id)
);

-- Solo se publica la última calibración; la web toma la más reciente.
CREATE TABLE IF NOT EXISTS pub_parlay_calibration (
    model_id              VARCHAR NOT NULL,
    ece                   DOUBLE PRECISION,
    logloss               DOUBLE PRECISION,
    ece_independent       DOUBLE PRECISION,
    logloss_independent   DOUBLE PRECISION,
    n                     INTEGER,
    created_at            TIMESTAMP,
    PRIMARY KEY (model_id)
);
