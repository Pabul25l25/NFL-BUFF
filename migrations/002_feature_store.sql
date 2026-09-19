-- ---------------------------------------------------------------------------
-- 002 — Feature store.
--
-- Invariante central del proyecto: TODA fila de feature lleva `as_of_ts`, el
-- instante hasta el que se permitió mirar datos (kickoff menos un margen).
-- Cualquier feature construida con información posterior a `as_of_ts` es un
-- leak, y hay pruebas que lo verifican (tests/features/test_leakage.py).
-- ---------------------------------------------------------------------------

-- Features a nivel partido: alimenta los modelos de ganador, spread y total.
-- Formato ancho y explícito (no EAV): el número de features es estable y así
-- el entrenamiento lee una sola tabla sin pivotar.
CREATE TABLE IF NOT EXISTS features_game (
    game_id                      VARCHAR NOT NULL,
    feature_set_version          VARCHAR NOT NULL,
    season                       INTEGER NOT NULL,
    week                         INTEGER NOT NULL,
    season_type                  VARCHAR,
    home_team                    VARCHAR,
    away_team                    VARCHAR,
    as_of_ts                     TIMESTAMP NOT NULL,
    feature_hash                 VARCHAR,

    -- ---- contexto del partido (conocido al publicarse el calendario) ----
    is_divisional                BOOLEAN,
    is_neutral_site              BOOLEAN,
    is_postseason                BOOLEAN,
    week_phase                   VARCHAR,      -- early | mid | late | playoffs
    roof_is_closed               BOOLEAN,
    surface_is_turf              BOOLEAN,
    altitude_m                   DOUBLE PRECISION,
    is_high_altitude             BOOLEAN,

    -- ---- descanso y viaje ----
    home_rest_days               DOUBLE PRECISION,
    away_rest_days               DOUBLE PRECISION,
    rest_advantage               DOUBLE PRECISION,
    home_short_week              BOOLEAN,
    away_short_week              BOOLEAN,
    home_off_bye                 BOOLEAN,
    away_off_bye                 BOOLEAN,
    away_travel_km               DOUBLE PRECISION,
    away_tz_shift_hours          DOUBLE PRECISION,

    -- ---- clima (NULL en estadios cerrados) ----
    temperature_c                DOUBLE PRECISION,
    wind_kph                     DOUBLE PRECISION,
    precipitation_mm             DOUBLE PRECISION,
    humidity_pct                 DOUBLE PRECISION,
    is_adverse_weather           BOOLEAN,
    weather_source               VARCHAR,

    -- ---- forma del equipo local (medias móviles, solo semanas anteriores) ----
    home_epa_off_r3              DOUBLE PRECISION,
    home_epa_off_r5              DOUBLE PRECISION,
    home_epa_off_r8              DOUBLE PRECISION,
    home_epa_def_r3              DOUBLE PRECISION,
    home_epa_def_r5              DOUBLE PRECISION,
    home_epa_def_r8              DOUBLE PRECISION,
    home_pass_epa_off_r5         DOUBLE PRECISION,
    home_rush_epa_off_r5         DOUBLE PRECISION,
    home_pass_epa_def_r5         DOUBLE PRECISION,
    home_rush_epa_def_r5         DOUBLE PRECISION,
    home_epa_off_ewm             DOUBLE PRECISION,
    home_epa_def_ewm             DOUBLE PRECISION,
    home_points_for_r5           DOUBLE PRECISION,
    home_points_against_r5       DOUBLE PRECISION,
    home_margin_r5               DOUBLE PRECISION,
    home_cpoe_r5                 DOUBLE PRECISION,
    home_sack_rate_off_r5        DOUBLE PRECISION,
    home_sack_rate_def_r5        DOUBLE PRECISION,
    home_turnover_margin_r5      DOUBLE PRECISION,
    home_pass_rate_r5            DOUBLE PRECISION,

    -- ---- forma del equipo visitante ----
    away_epa_off_r3              DOUBLE PRECISION,
    away_epa_off_r5              DOUBLE PRECISION,
    away_epa_off_r8              DOUBLE PRECISION,
    away_epa_def_r3              DOUBLE PRECISION,
    away_epa_def_r5              DOUBLE PRECISION,
    away_epa_def_r8              DOUBLE PRECISION,
    away_pass_epa_off_r5         DOUBLE PRECISION,
    away_rush_epa_off_r5         DOUBLE PRECISION,
    away_pass_epa_def_r5         DOUBLE PRECISION,
    away_rush_epa_def_r5         DOUBLE PRECISION,
    away_epa_off_ewm             DOUBLE PRECISION,
    away_epa_def_ewm             DOUBLE PRECISION,
    away_points_for_r5           DOUBLE PRECISION,
    away_points_against_r5       DOUBLE PRECISION,
    away_margin_r5               DOUBLE PRECISION,
    away_cpoe_r5                 DOUBLE PRECISION,
    away_sack_rate_off_r5        DOUBLE PRECISION,
    away_sack_rate_def_r5        DOUBLE PRECISION,
    away_turnover_margin_r5      DOUBLE PRECISION,
    away_pass_rate_r5            DOUBLE PRECISION,

    -- ---- splits local/visitante (histórico propio de cada condición) ----
    home_epa_off_at_home_r8      DOUBLE PRECISION,
    away_epa_off_at_away_r8      DOUBLE PRECISION,

    -- ---- diferenciales (matchup) ----
    epa_off_diff                 DOUBLE PRECISION,
    epa_def_diff                 DOUBLE PRECISION,
    net_epa_diff                 DOUBLE PRECISION,
    margin_diff_r5               DOUBLE PRECISION,
    pass_matchup_diff            DOUBLE PRECISION,   -- pase local vs def. pase visitante
    rush_matchup_diff            DOUBLE PRECISION,

    -- ---- Elo (baseline y a la vez feature legítima) ----
    home_elo_pre                 DOUBLE PRECISION,
    away_elo_pre                 DOUBLE PRECISION,
    elo_diff                     DOUBLE PRECISION,

    -- ---- disponibilidad por lesiones (snapshot as-of) ----
    home_injury_index            DOUBLE PRECISION,
    away_injury_index            DOUBLE PRECISION,
    home_qb_questionable         BOOLEAN,
    away_qb_questionable         BOOLEAN,
    home_starters_out            DOUBLE PRECISION,
    away_starters_out            DOUBLE PRECISION,
    injury_index_diff            DOUBLE PRECISION,

    -- ---- clúster de estilo (agente no supervisado; feature auxiliar) ----
    home_style_cluster           INTEGER,
    away_style_cluster           INTEGER,
    style_mismatch               BOOLEAN,

    PRIMARY KEY (game_id, feature_set_version)
);

-- Features a nivel jugador-partido: alimenta los modelos de props.
CREATE TABLE IF NOT EXISTS features_player (
    game_id                      VARCHAR NOT NULL,
    gsis_id                      VARCHAR NOT NULL,
    feature_set_version          VARCHAR NOT NULL,
    season                       INTEGER NOT NULL,
    week                         INTEGER NOT NULL,
    season_type                  VARCHAR,
    team                         VARCHAR,
    opponent_team                VARCHAR,
    position                     VARCHAR,
    is_home                      BOOLEAN,
    as_of_ts                     TIMESTAMP NOT NULL,
    feature_hash                 VARCHAR,

    -- ---- proyección de uso (el driver dominante de los props) ----
    snap_pct_r5                  DOUBLE PRECISION,
    snap_pct_r3                  DOUBLE PRECISION,
    target_share_r5              DOUBLE PRECISION,
    air_yards_share_r5           DOUBLE PRECISION,
    wopr_r5                      DOUBLE PRECISION,
    carry_share_r5               DOUBLE PRECISION,
    routes_proxy_r5              DOUBLE PRECISION,
    games_played_prior           DOUBLE PRECISION,
    is_low_sample                BOOLEAN,     -- dispara la rama semi-supervisada

    -- ---- producción reciente ----
    receiving_yards_r3           DOUBLE PRECISION,
    receiving_yards_r5           DOUBLE PRECISION,
    receptions_r5                DOUBLE PRECISION,
    targets_r5                   DOUBLE PRECISION,
    receiving_epa_r5             DOUBLE PRECISION,
    racr_r5                      DOUBLE PRECISION,
    yac_r5                       DOUBLE PRECISION,
    rushing_yards_r3             DOUBLE PRECISION,
    rushing_yards_r5             DOUBLE PRECISION,
    carries_r5                   DOUBLE PRECISION,
    rushing_epa_r5               DOUBLE PRECISION,
    ypc_r5                       DOUBLE PRECISION,

    -- ---- contexto de equipo y rival ----
    team_pass_rate_r5            DOUBLE PRECISION,
    team_plays_r5                DOUBLE PRECISION,
    team_epa_off_r5              DOUBLE PRECISION,
    opp_pass_epa_def_r5          DOUBLE PRECISION,
    opp_rush_epa_def_r5          DOUBLE PRECISION,
    opp_points_allowed_r5        DOUBLE PRECISION,

    -- ---- estado físico y entorno ----
    injury_status_code           INTEGER,     -- 0 sano, 1 questionable, 2 doubtful, 3 out
    practice_status_code         INTEGER,
    rest_days                    DOUBLE PRECISION,
    wind_kph                     DOUBLE PRECISION,
    temperature_c                DOUBLE PRECISION,
    is_adverse_weather           BOOLEAN,
    roof_is_closed               BOOLEAN,

    PRIMARY KEY (game_id, gsis_id, feature_set_version)
);

-- Elo calculado de forma incremental y estrictamente cronológica.
-- Se materializa para que el backtest no lo recalcule en cada iteración.
CREATE TABLE IF NOT EXISTS elo_ratings (
    game_id          VARCHAR NOT NULL,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    team             VARCHAR NOT NULL,
    elo_pre          DOUBLE PRECISION,
    elo_post         DOUBLE PRECISION,
    is_home          BOOLEAN,
    PRIMARY KEY (game_id, team)
);

-- Perfiles de estilo de equipo por temporada-semana (K-means, no supervisado).
CREATE TABLE IF NOT EXISTS team_style_clusters (
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    team             VARCHAR NOT NULL,
    cluster_id       INTEGER,
    model_version    VARCHAR,
    PRIMARY KEY (season, week, team)
);

CREATE INDEX IF NOT EXISTS idx_fgame_season_week  ON features_game (season, week);
CREATE INDEX IF NOT EXISTS idx_fgame_asof         ON features_game (as_of_ts);
CREATE INDEX IF NOT EXISTS idx_fplayer_season_wk  ON features_player (season, week);
CREATE INDEX IF NOT EXISTS idx_fplayer_player     ON features_player (gsis_id, season);
CREATE INDEX IF NOT EXISTS idx_elo_team           ON elo_ratings (team, season, week);
