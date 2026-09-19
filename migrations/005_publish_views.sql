-- ---------------------------------------------------------------------------
-- 005 — Tablas de publicación para el frontend web (Vercel).
--
-- Vercel es serverless y no puede leer data/nflp.duckdb (disco local del
-- usuario). El mecanismo de sincronización (src/nflp/sync/publish.py) copia un
-- subconjunto de datos a una base Postgres alcanzable desde internet (Neon).
--
-- Estas dos tablas son APLANADAS a propósito: contienen el resultado ya
-- unido de `predictions` + `games`/`players` + `model_registry`, siguiendo
-- exactamente las mismas consultas que ya existen en
-- `src/nflp/dashboard/data.py` (`load_predictions`, `load_prop_predictions`).
-- Así el SQL "de negocio" (qué se une con qué) sigue viviendo en un solo
-- lugar, y las API routes en TypeScript solo hacen un SELECT simple, sin
-- joins, sobre una tabla ya lista para servir.
--
-- En la base LOCAL (DuckDB) estas tablas se crean pero quedan vacías: nunca
-- se les escribe nada ahí, solo en la copia de Neon. Es el mismo patrón ya
-- usado por el resto del esquema (portable, sin sintaxis específica de motor).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS pub_predictions_games (
    prediction_id     VARCHAR NOT NULL,
    game_id           VARCHAR NOT NULL,
    model_id          VARCHAR,
    target            VARCHAR NOT NULL,   -- winner | spread | total
    point_estimate    DOUBLE PRECISION,
    probability       DOUBLE PRECISION,
    pred_std          DOUBLE PRECISION,
    lower_80          DOUBLE PRECISION,
    upper_80          DOUBLE PRECISION,
    confidence_level  VARCHAR,
    shap_values       VARCHAR,            -- JSON
    explanation_es    VARCHAR,
    predicted_at      TIMESTAMP,
    as_of_ts          TIMESTAMP,
    season            INTEGER,
    week              INTEGER,
    season_type       VARCHAR,
    gameday           DATE,
    kickoff_utc       TIMESTAMP,
    home_team         VARCHAR,
    away_team         VARCHAR,
    home_score        INTEGER,
    away_score        INTEGER,
    actual_margin     INTEGER,
    actual_total      INTEGER,
    roof              VARCHAR,
    surface           VARCHAR,
    stadium           VARCHAR,
    model_version     INTEGER,
    model_family      VARCHAR,
    temperature_c     DOUBLE PRECISION,
    wind_kph          DOUBLE PRECISION,
    precipitation_mm  DOUBLE PRECISION,
    PRIMARY KEY (game_id, target)
);

CREATE TABLE IF NOT EXISTS pub_predictions_props (
    game_id           VARCHAR NOT NULL,
    gsis_id           VARCHAR NOT NULL,
    model_id          VARCHAR,
    target            VARCHAR NOT NULL,
    point_estimate    DOUBLE PRECISION,
    lower_80          DOUBLE PRECISION,
    upper_80          DOUBLE PRECISION,
    explanation_es    VARCHAR,
    shap_values       VARCHAR,
    predicted_at      TIMESTAMP,
    as_of_ts          TIMESTAMP,
    season            INTEGER,
    week              INTEGER,
    home_team         VARCHAR,
    away_team         VARCHAR,
    kickoff_utc       TIMESTAMP,
    player_name       VARCHAR,
    position          VARCHAR,
    team              VARCHAR,
    -- estadísticas reales, una por cada prop soportado (NULL si aún no se jugó
    -- o no aplica a ese jugador). El API en TypeScript elige la columna
    -- correcta según `target`, igual que `_PROP_ACTUAL_COLUMN` en el dashboard.
    receiving_yards   DOUBLE PRECISION,
    rushing_yards     DOUBLE PRECISION,
    receptions        DOUBLE PRECISION,
    receiving_tds     DOUBLE PRECISION,
    rushing_tds       DOUBLE PRECISION,
    passing_yards     DOUBLE PRECISION,
    passing_tds       DOUBLE PRECISION,
    sacks_suffered    DOUBLE PRECISION,
    PRIMARY KEY (game_id, gsis_id, target)
);

-- Cuándo se sincronizó por última vez cada tabla publicada. La UI la usa para
-- mostrar "datos de hace X horas" — importante porque, a diferencia del
-- Streamlit local, esta copia ya no se actualiza en vivo.
CREATE TABLE IF NOT EXISTS sync_meta (
    table_name   VARCHAR NOT NULL,
    synced_at    TIMESTAMP NOT NULL,
    row_count    INTEGER,
    PRIMARY KEY (table_name)
);

CREATE INDEX IF NOT EXISTS idx_pub_games_season_week ON pub_predictions_games (season, week);
CREATE INDEX IF NOT EXISTS idx_pub_props_season_week ON pub_predictions_props (season, week);
