-- ---------------------------------------------------------------------------
-- 003 — Registro de modelos, predicciones, backtesting y monitoreo.
--
-- Cubre las secciones 5 (reentrenamiento, rollback, shadow) y 7 (trazabilidad:
-- qué modelo generó qué número y cuándo) del prompt.
-- ---------------------------------------------------------------------------

-- Espejo consultable del registro de MLflow. MLflow es la fuente de verdad de
-- los artefactos; esta tabla existe para que el dashboard y el gate de
-- promoción consulten estado sin depender del servidor de MLflow.
CREATE TABLE IF NOT EXISTS model_registry (
    model_id             VARCHAR NOT NULL,   -- uuid
    target               VARCHAR NOT NULL,   -- winner | spread | total | receiving_yards | ...
    family               VARCHAR NOT NULL,   -- lightgbm | xgboost | catboost | elo | naive
    version              INTEGER NOT NULL,   -- incremental por target
    feature_set_version  VARCHAR,
    mlflow_run_id        VARCHAR,
    artifact_path        VARCHAR,
    feature_names        VARCHAR,            -- JSON serializado
    hyperparameters      VARCHAR,            -- JSON serializado
    dataset_hash         VARCHAR,            -- reproducibilidad del backtest
    train_seasons        VARCHAR,
    n_train_rows         INTEGER,
    n_val_rows           INTEGER,
    -- métricas (la métrica primaria depende del target)
    metric_name          VARCHAR,
    train_metric         DOUBLE PRECISION,
    val_metric           DOUBLE PRECISION,
    test_metric          DOUBLE PRECISION,
    train_val_gap        DOUBLE PRECISION,   -- deterioro relativo train->val
    metrics_json         VARCHAR,            -- todas las métricas, JSON
    -- ciclo de vida
    status               VARCHAR NOT NULL,   -- rejected | shadow | production | archived
    promotion_blocked_reason VARCHAR,
    created_at           TIMESTAMP NOT NULL,
    promoted_at          TIMESTAMP,
    archived_at          TIMESTAMP,
    PRIMARY KEY (model_id)
);

-- Trazabilidad completa: cada número mostrado en el dashboard apunta al modelo
-- exacto que lo generó y al instante en que se generó.
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id        VARCHAR NOT NULL,
    game_id              VARCHAR NOT NULL,
    model_id             VARCHAR NOT NULL,
    target               VARCHAR NOT NULL,
    gsis_id              VARCHAR,            -- solo para props
    -- salida del modelo
    point_estimate       DOUBLE PRECISION,   -- margen, total o yardas proyectadas
    probability          DOUBLE PRECISION,   -- P(gana local) o P(over) según target
    pred_std             DOUBLE PRECISION,   -- desviación de la distribución predictiva
    lower_80             DOUBLE PRECISION,
    upper_80             DOUBLE PRECISION,
    confidence_level     VARCHAR,            -- alta | media | baja
    -- explicabilidad (sección 7)
    shap_values          VARCHAR,            -- JSON: {feature: valor_shap}
    shap_base_value      DOUBLE PRECISION,
    explanation_es       VARCHAR,            -- justificación en lenguaje natural
    -- metadatos
    is_shadow            BOOLEAN,            -- predicción de un modelo en shadow
    as_of_ts             TIMESTAMP,
    predicted_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (prediction_id)
);

-- Resultados del walk-forward, desagregados por segmento (sección 4).
CREATE TABLE IF NOT EXISTS backtest_results (
    result_id            VARCHAR NOT NULL,
    run_id               VARCHAR NOT NULL,   -- agrupa una corrida completa
    model_label          VARCHAR NOT NULL,   -- nombre del modelo o baseline
    target               VARCHAR NOT NULL,
    segment_type         VARCHAR NOT NULL,   -- overall | season | team | weather | week_phase
    segment_value        VARCHAR,
    n_games              INTEGER,
    -- métricas probabilísticas
    log_loss             DOUBLE PRECISION,
    brier_score          DOUBLE PRECISION,
    ece                  DOUBLE PRECISION,   -- expected calibration error
    accuracy             DOUBLE PRECISION,
    -- métricas de regresión
    mae                  DOUBLE PRECISION,
    rmse                 DOUBLE PRECISION,
    crps                 DOUBLE PRECISION,
    -- métricas específicas de apuestas
    ats_win_pct          DOUBLE PRECISION,   -- % aciertos contra el spread
    beats_market         BOOLEAN,
    created_at           TIMESTAMP NOT NULL,
    PRIMARY KEY (result_id)
);

-- Puntos de la curva de calibración, para graficar sin recalcular.
CREATE TABLE IF NOT EXISTS calibration_bins (
    run_id               VARCHAR NOT NULL,
    model_label          VARCHAR NOT NULL,
    target               VARCHAR NOT NULL,
    bin_index            INTEGER NOT NULL,
    bin_lower            DOUBLE PRECISION,
    bin_upper            DOUBLE PRECISION,
    mean_predicted       DOUBLE PRECISION,
    observed_frequency   DOUBLE PRECISION,
    n_samples            INTEGER,
    PRIMARY KEY (run_id, model_label, target, bin_index)
);

-- Drift por feature: alimenta el disparador de reentrenamiento (sección 5).
CREATE TABLE IF NOT EXISTS drift_metrics (
    drift_id             VARCHAR NOT NULL,
    checked_at           TIMESTAMP NOT NULL,
    season               INTEGER,
    week                 INTEGER,
    feature_name         VARCHAR NOT NULL,
    psi                  DOUBLE PRECISION,   -- population stability index
    kl_divergence        DOUBLE PRECISION,
    reference_mean       DOUBLE PRECISION,
    current_mean         DOUBLE PRECISION,
    is_drifted           BOOLEAN,
    PRIMARY KEY (drift_id)
);

-- Bitácora de decisiones del Retraining/Monitoring Agent.
-- Deja constancia auditable de por qué se promovió o bloqueó cada modelo.
CREATE TABLE IF NOT EXISTS retraining_log (
    event_id             VARCHAR NOT NULL,
    occurred_at          TIMESTAMP NOT NULL,
    target               VARCHAR,
    action               VARCHAR,   -- retrain | promote | block | rollback | shadow_start
    model_id             VARCHAR,
    previous_model_id    VARCHAR,
    reason               VARCHAR,
    details_json         VARCHAR,
    PRIMARY KEY (event_id)
);

-- Rendimiento del modelo en producción semana a semana (vista de degradación).
CREATE TABLE IF NOT EXISTS production_performance (
    perf_id              VARCHAR NOT NULL,
    model_id             VARCHAR NOT NULL,
    target               VARCHAR NOT NULL,
    season               INTEGER NOT NULL,
    week                 INTEGER NOT NULL,
    n_predictions        INTEGER,
    log_loss             DOUBLE PRECISION,
    brier_score          DOUBLE PRECISION,
    accuracy             DOUBLE PRECISION,
    mae                  DOUBLE PRECISION,
    computed_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (perf_id)
);

CREATE INDEX IF NOT EXISTS idx_registry_target   ON model_registry (target, status);
CREATE INDEX IF NOT EXISTS idx_registry_version  ON model_registry (target, version);
CREATE INDEX IF NOT EXISTS idx_pred_game         ON predictions (game_id, target);
CREATE INDEX IF NOT EXISTS idx_pred_model        ON predictions (model_id);
CREATE INDEX IF NOT EXISTS idx_backtest_run      ON backtest_results (run_id, target);
CREATE INDEX IF NOT EXISTS idx_drift_week        ON drift_metrics (season, week);
CREATE INDEX IF NOT EXISTS idx_prodperf_lookup   ON production_performance (target, season, week);
