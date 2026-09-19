-- ---------------------------------------------------------------------------
-- 001 — Tablas núcleo: referencia y hechos.
--
-- SQL deliberadamente portable entre DuckDB y PostgreSQL:
-- se usan VARCHAR / INTEGER / DOUBLE PRECISION / BOOLEAN / DATE / TIMESTAMP,
-- válidos en ambos motores. Nada de tipos ni funciones específicas de un motor.
-- ---------------------------------------------------------------------------

-- ============================ TABLAS DE REFERENCIA ==========================

CREATE TABLE IF NOT EXISTS teams (
    team_abbr        VARCHAR NOT NULL,
    team_name        VARCHAR,
    team_conf        VARCHAR,
    team_division    VARCHAR,
    team_color       VARCHAR,
    team_color2      VARCHAR,
    team_logo_espn   VARCHAR,
    PRIMARY KEY (team_abbr)
);

-- nflverse no publica coordenadas de estadio: esta tabla se siembra desde
-- src/nflp/agents/data/stadiums.py. Es imprescindible para el clima
-- (Open-Meteo pide lat/lon), la distancia de viaje y el efecto altitud.
CREATE TABLE IF NOT EXISTS stadiums (
    stadium_id       VARCHAR NOT NULL,
    stadium_name     VARCHAR,
    team_abbr        VARCHAR,
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    altitude_m       DOUBLE PRECISION,
    roof_type        VARCHAR,          -- outdoors | dome | closed | open
    surface          VARCHAR,          -- grass | fieldturf | astroturf | ...
    timezone         VARCHAR,
    PRIMARY KEY (stadium_id)
);

-- ============================== TABLAS DE HECHOS =============================

-- Nota anti-leakage: home_score, away_score y result son RESULTADOS.
-- Están en la denylist de features (ver src/nflp/agents/features/contracts.py).
CREATE TABLE IF NOT EXISTS games (
    game_id          VARCHAR NOT NULL,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    season_type      VARCHAR NOT NULL,     -- REG | POST (separados, sección 2)
    gameday          DATE,
    gametime         VARCHAR,
    kickoff_utc      TIMESTAMP,
    weekday          VARCHAR,
    home_team        VARCHAR NOT NULL,
    away_team        VARCHAR NOT NULL,
    stadium_id       VARCHAR,
    stadium          VARCHAR,
    roof             VARCHAR,
    surface          VARCHAR,
    div_game         BOOLEAN,
    neutral_site     BOOLEAN,
    home_rest        INTEGER,
    away_rest        INTEGER,
    home_qb_id       VARCHAR,
    away_qb_id       VARCHAR,
    home_qb_name     VARCHAR,
    away_qb_name     VARCHAR,
    home_coach       VARCHAR,
    away_coach       VARCHAR,
    referee          VARCHAR,
    -- resultados (nunca features)
    home_score       INTEGER,
    away_score       INTEGER,
    result           INTEGER,              -- home_score - away_score
    total            INTEGER,              -- home_score + away_score
    overtime         BOOLEAN,
    PRIMARY KEY (game_id)
);

CREATE TABLE IF NOT EXISTS players (
    gsis_id          VARCHAR NOT NULL,
    display_name     VARCHAR,
    position         VARCHAR,
    position_group   VARCHAR,
    birth_date       DATE,
    height           DOUBLE PRECISION,
    weight           DOUBLE PRECISION,
    college          VARCHAR,
    rookie_season    INTEGER,
    draft_round      INTEGER,
    draft_pick       INTEGER,
    headshot_url     VARCHAR,
    PRIMARY KEY (gsis_id)
);

CREATE TABLE IF NOT EXISTS rosters_weekly (
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    season_type      VARCHAR,
    team             VARCHAR NOT NULL,
    gsis_id          VARCHAR NOT NULL,
    position         VARCHAR,
    depth_chart_pos  VARCHAR,
    jersey_number    INTEGER,
    status           VARCHAR,
    years_exp        INTEGER,
    PRIMARY KEY (season, week, team, gsis_id)
);

-- date_modified es la clave del snapshot as-of: permite reconstruir qué se
-- sabía del parte de lesiones ANTES del kickoff, sin leakage.
CREATE TABLE IF NOT EXISTS injuries (
    season                   INTEGER NOT NULL,
    week                     INTEGER NOT NULL,
    season_type              VARCHAR,
    team                     VARCHAR NOT NULL,
    gsis_id                  VARCHAR NOT NULL,
    position                 VARCHAR,
    full_name                VARCHAR,
    report_primary_injury    VARCHAR,
    report_secondary_injury  VARCHAR,
    report_status            VARCHAR,   -- Out | Doubtful | Questionable | NULL
    practice_primary_injury  VARCHAR,
    practice_status          VARCHAR,
    date_modified            TIMESTAMP,
    PRIMARY KEY (season, week, team, gsis_id)
);

CREATE TABLE IF NOT EXISTS player_stats_weekly (
    player_id                VARCHAR NOT NULL,
    season                   INTEGER NOT NULL,
    week                     INTEGER NOT NULL,
    season_type              VARCHAR NOT NULL,
    game_id                  VARCHAR,
    team                     VARCHAR,
    opponent_team            VARCHAR,
    player_name              VARCHAR,
    position                 VARCHAR,
    position_group           VARCHAR,
    -- pase
    completions              DOUBLE PRECISION,
    attempts                 DOUBLE PRECISION,
    passing_yards            DOUBLE PRECISION,
    passing_tds              DOUBLE PRECISION,
    passing_interceptions    DOUBLE PRECISION,
    sacks_suffered           DOUBLE PRECISION,
    passing_air_yards        DOUBLE PRECISION,
    passing_first_downs      DOUBLE PRECISION,
    passing_epa              DOUBLE PRECISION,
    passing_cpoe             DOUBLE PRECISION,
    pacr                     DOUBLE PRECISION,
    -- carrera
    carries                  DOUBLE PRECISION,
    rushing_yards            DOUBLE PRECISION,
    rushing_tds              DOUBLE PRECISION,
    rushing_first_downs      DOUBLE PRECISION,
    rushing_epa              DOUBLE PRECISION,
    rushing_fumbles_lost     DOUBLE PRECISION,
    -- recepción
    receptions               DOUBLE PRECISION,
    targets                  DOUBLE PRECISION,
    receiving_yards          DOUBLE PRECISION,
    receiving_tds            DOUBLE PRECISION,
    receiving_air_yards      DOUBLE PRECISION,
    receiving_yards_after_catch DOUBLE PRECISION,
    receiving_first_downs    DOUBLE PRECISION,
    receiving_epa            DOUBLE PRECISION,
    target_share             DOUBLE PRECISION,
    air_yards_share          DOUBLE PRECISION,
    wopr                     DOUBLE PRECISION,
    racr                     DOUBLE PRECISION,
    fantasy_points_ppr       DOUBLE PRECISION,
    PRIMARY KEY (player_id, season, week, season_type)
);

CREATE TABLE IF NOT EXISTS team_stats_weekly (
    team                     VARCHAR NOT NULL,
    season                   INTEGER NOT NULL,
    week                     INTEGER NOT NULL,
    season_type              VARCHAR NOT NULL,
    game_id                  VARCHAR,
    opponent_team            VARCHAR,
    -- ofensiva
    completions              DOUBLE PRECISION,
    attempts                 DOUBLE PRECISION,
    passing_yards            DOUBLE PRECISION,
    passing_tds              DOUBLE PRECISION,
    passing_interceptions    DOUBLE PRECISION,
    passing_epa              DOUBLE PRECISION,
    passing_cpoe             DOUBLE PRECISION,
    passing_first_downs      DOUBLE PRECISION,
    passing_air_yards        DOUBLE PRECISION,
    sacks_suffered           DOUBLE PRECISION,
    carries                  DOUBLE PRECISION,
    rushing_yards            DOUBLE PRECISION,
    rushing_tds              DOUBLE PRECISION,
    rushing_epa              DOUBLE PRECISION,
    rushing_first_downs      DOUBLE PRECISION,
    receiving_yards          DOUBLE PRECISION,
    receiving_epa            DOUBLE PRECISION,
    -- defensiva
    def_sacks                DOUBLE PRECISION,
    def_qb_hits              DOUBLE PRECISION,
    def_interceptions        DOUBLE PRECISION,
    def_pass_defended        DOUBLE PRECISION,
    def_tackles_for_loss     DOUBLE PRECISION,
    def_fumbles_forced       DOUBLE PRECISION,
    def_tds                  DOUBLE PRECISION,
    -- balón suelto / especiales
    fumbles_lost             DOUBLE PRECISION,
    special_teams_tds        DOUBLE PRECISION,
    PRIMARY KEY (team, season, week, season_type)
);

CREATE TABLE IF NOT EXISTS snap_counts (
    game_id          VARCHAR NOT NULL,
    pfr_player_id    VARCHAR NOT NULL,
    player           VARCHAR,
    position         VARCHAR,
    team             VARCHAR,
    season           INTEGER,
    week             INTEGER,
    season_type      VARCHAR,
    offense_snaps    DOUBLE PRECISION,
    offense_pct      DOUBLE PRECISION,
    defense_snaps    DOUBLE PRECISION,
    defense_pct      DOUBLE PRECISION,
    st_snaps         DOUBLE PRECISION,
    st_pct           DOUBLE PRECISION,
    PRIMARY KEY (game_id, pfr_player_id)
);

-- Solo se puebla para estadios sin techo cerrado (sección 2 del prompt).
-- 'source' distingue pronóstico (partido futuro) de archivo (histórico).
CREATE TABLE IF NOT EXISTS weather (
    game_id          VARCHAR NOT NULL,
    temperature_c    DOUBLE PRECISION,
    wind_kph         DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION,
    humidity_pct     DOUBLE PRECISION,
    snowfall_cm      DOUBLE PRECISION,
    source           VARCHAR,          -- forecast | archive | indoor
    as_of_ts         TIMESTAMP,
    PRIMARY KEY (game_id)
);

-- ===================== MERCADO — AISLADO A PROPÓSITO ========================
-- Esta tabla existe SOLO como benchmark del Backtesting Agent.
-- El Feature Engineering Agent no la lee nunca: el aislamiento es estructural,
-- no una convención. Ver docs/anti-leakage.md.
CREATE TABLE IF NOT EXISTS market_lines (
    game_id            VARCHAR NOT NULL,
    spread_line        DOUBLE PRECISION,   -- positivo = local favorito
    total_line         DOUBLE PRECISION,
    home_moneyline     DOUBLE PRECISION,
    away_moneyline     DOUBLE PRECISION,
    home_spread_odds   DOUBLE PRECISION,
    away_spread_odds   DOUBLE PRECISION,
    over_odds          DOUBLE PRECISION,
    under_odds         DOUBLE PRECISION,
    PRIMARY KEY (game_id)
);

-- =========================== AUDITORÍA DE INGESTA ===========================

CREATE TABLE IF NOT EXISTS ingestion_log (
    ingestion_id     VARCHAR NOT NULL,
    source           VARCHAR NOT NULL,   -- nflverse | open-meteo | static
    asset            VARCHAR NOT NULL,
    season           INTEGER,
    rows_ingested    INTEGER,
    etag             VARCHAR,
    url              VARCHAR,
    status           VARCHAR,            -- ok | cached | error
    message          VARCHAR,
    fetched_at       TIMESTAMP NOT NULL,
    PRIMARY KEY (ingestion_id)
);

-- =============================== ÍNDICES ====================================

CREATE INDEX IF NOT EXISTS idx_games_season_week   ON games (season, week);
CREATE INDEX IF NOT EXISTS idx_games_kickoff       ON games (kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_games_home          ON games (home_team, season);
CREATE INDEX IF NOT EXISTS idx_games_away          ON games (away_team, season);
CREATE INDEX IF NOT EXISTS idx_pstats_season_week  ON player_stats_weekly (season, week);
CREATE INDEX IF NOT EXISTS idx_pstats_player       ON player_stats_weekly (player_id, season);
CREATE INDEX IF NOT EXISTS idx_tstats_season_week  ON team_stats_weekly (season, week);
CREATE INDEX IF NOT EXISTS idx_tstats_team         ON team_stats_weekly (team, season);
CREATE INDEX IF NOT EXISTS idx_injuries_lookup     ON injuries (season, week, team);
CREATE INDEX IF NOT EXISTS idx_snaps_season_week   ON snap_counts (season, week);
