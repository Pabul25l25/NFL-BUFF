-- ---------------------------------------------------------------------------
-- 009 — Modelo de correlación para parlays del mismo partido (cópula gaussiana).
--
-- `parlay_game_corr`: por partido, la lista de piernas ("p:<gsis_id>:<target>"
-- para cada prop proyectado de ambos equipos, más "g:margin" y "g:total") y
-- la matriz de correlación latente (semidefinida positiva, diagonal 1) que las
-- une. Si el modelo correlado no supera a la independencia en el backtest de
-- calibración, se guarda la identidad (y así consta en parlay_calibration).
--
-- `parlay_game_legs`: piernas de partido como normales. margin = local − visitante.
--
-- `parlay_calibration`: un registro por modelo de parlay, con la calibración
-- walk-forward de parlays sintéticos (cópula vs independencia).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS parlay_game_corr (
    game_id   VARCHAR NOT NULL,
    model_id  VARCHAR NOT NULL,
    legs      VARCHAR,          -- JSON: lista de nombres de piernas
    matrix    VARCHAR,          -- JSON: matriz 2D, 3 decimales
    as_of_ts  TIMESTAMP,
    PRIMARY KEY (game_id, model_id)
);

CREATE TABLE IF NOT EXISTS parlay_game_legs (
    game_id       VARCHAR NOT NULL,
    model_id      VARCHAR NOT NULL,
    p_home_win    DOUBLE PRECISION,
    margin_mu     DOUBLE PRECISION,
    margin_sigma  DOUBLE PRECISION,
    total_mu      DOUBLE PRECISION,
    total_sigma   DOUBLE PRECISION,
    as_of_ts      TIMESTAMP,
    PRIMARY KEY (game_id, model_id)
);

CREATE TABLE IF NOT EXISTS parlay_calibration (
    model_id             VARCHAR NOT NULL,
    ece                  DOUBLE PRECISION,
    logloss              DOUBLE PRECISION,
    ece_independent      DOUBLE PRECISION,
    logloss_independent  DOUBLE PRECISION,
    n                    INTEGER,
    created_at           TIMESTAMP,
    PRIMARY KEY (model_id)
);
