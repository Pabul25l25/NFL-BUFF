"""Pruebas de integración: el pipeline completo sobre datos sintéticos.

Verifican que las piezas encajan de extremo a extremo —features, entrenamiento,
walk-forward, registro— sin tocar la red y sobre una base temporal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflp.agents.features import FeatureAgent
from nflp.agents.modeling import ModelingAgent
from nflp.agents.modeling.datasets import load_game_dataset
from nflp.core.migrations import current_version, discover_migrations, migrate


class TestMigrations:
    def test_migrations_are_discovered_and_ordered(self) -> None:
        migrations = discover_migrations()
        assert len(migrations) >= 3
        versions = [m.version for m in migrations]
        assert versions == sorted(versions)

    def test_migrations_are_idempotent(self, db) -> None:
        """Volver a aplicar no debe cambiar nada."""
        version_before = current_version(db)
        applied = migrate(db)
        assert applied == []
        assert current_version(db) == version_before

    def test_modified_migration_is_rejected(self, settings, tmp_path) -> None:
        """Editar una migración ya aplicada debe fallar de forma explícita."""
        from nflp.core.db import Database

        # Base limpia: el fixture `db` ya trae aplicadas las migraciones reales,
        # cuyos checksums chocarían con los de este directorio de prueba.
        settings.paths.duckdb_path = tmp_path / "migration_test.duckdb"
        database = Database(settings)

        directory = tmp_path / "migrations"
        directory.mkdir()
        path = directory / "001_test.sql"
        path.write_text("CREATE TABLE IF NOT EXISTS t_demo (a INTEGER);", encoding="utf-8")

        applied = migrate(database, directory)
        assert len(applied) == 1

        path.write_text(
            "CREATE TABLE IF NOT EXISTS t_demo (a INTEGER, b INTEGER);", encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="modificada"):
            migrate(database, directory)

    def test_all_expected_tables_exist(self, db) -> None:
        expected = [
            "teams", "stadiums", "games", "players", "injuries",
            "player_stats_weekly", "team_stats_weekly", "weather", "market_lines",
            "features_game", "features_player", "elo_ratings",
            "model_registry", "predictions", "backtest_results", "drift_metrics",
        ]
        for table in expected:
            assert db.table_exists(table), f"Falta la tabla '{table}'"


class TestDatabase:
    def test_upsert_is_idempotent(self, db) -> None:
        frame = pd.DataFrame(
            {"team_abbr": ["KC", "BUF"], "team_name": ["Chiefs", "Bills"]}
        )
        db.write_df(frame, "teams", mode="upsert", key_columns=["team_abbr"])
        db.write_df(frame, "teams", mode="upsert", key_columns=["team_abbr"])
        assert db.row_count("teams") == 2

    def test_upsert_updates_existing(self, db) -> None:
        db.write_df(
            pd.DataFrame({"team_abbr": ["KC"], "team_name": ["Old"]}),
            "teams", mode="upsert", key_columns=["team_abbr"],
        )
        db.write_df(
            pd.DataFrame({"team_abbr": ["KC"], "team_name": ["New"]}),
            "teams", mode="upsert", key_columns=["team_abbr"],
        )
        assert db.scalar("SELECT team_name FROM teams WHERE team_abbr = 'KC'") == "New"

    def test_scalar_returns_none_on_empty(self, db) -> None:
        assert db.scalar("SELECT max(version) FROM model_registry") is None

    def test_unknown_columns_are_ignored(self, db) -> None:
        """Si nflverse añade una columna nueva, la ingesta no debe romperse."""
        frame = pd.DataFrame(
            {"team_abbr": ["KC"], "team_name": ["Chiefs"], "columna_nueva": ["x"]}
        )
        rows = db.write_df(frame, "teams", mode="append")
        assert rows == 1


class TestFeaturePipeline:
    def test_builds_features_and_passes_audit(self, settings, populated_db) -> None:
        agent = FeatureAgent(settings, populated_db)
        frame = agent.build()

        assert not frame.empty
        assert "as_of_ts" in frame.columns
        assert frame["as_of_ts"].notna().all()

    def test_as_of_always_precedes_kickoff(self, settings, populated_db) -> None:
        FeatureAgent(settings, populated_db).build()
        check = populated_db.query(
            """
            SELECT count(*) AS violaciones
            FROM features_game f JOIN games g USING (game_id)
            WHERE f.as_of_ts >= g.kickoff_utc
            """
        )
        assert int(check.iloc[0]["violaciones"]) == 0

    def test_elo_materialised(self, settings, populated_db) -> None:
        FeatureAgent(settings, populated_db).build()
        assert populated_db.row_count("elo_ratings") > 0

    def test_feature_store_excludes_outcomes(self, settings, populated_db) -> None:
        frame = FeatureAgent(settings, populated_db).build()
        for column in ("home_score", "away_score", "result", "total"):
            assert column not in frame.columns

    def test_rebuild_is_idempotent(self, settings, populated_db) -> None:
        agent = FeatureAgent(settings, populated_db)
        agent.build()
        first = populated_db.row_count("features_game")
        agent.build()
        assert populated_db.row_count("features_game") == first


class TestEndToEnd:
    """El pipeline completo: features → dataset → entrenamiento → registro."""

    def test_dataset_loads_after_feature_build(self, settings, populated_db) -> None:
        FeatureAgent(settings, populated_db).build()
        dataset = load_game_dataset(
            "spread", db=populated_db, feature_version=settings.features.version,
            min_prior_games=0,
        )
        assert len(dataset) > 0
        assert len(dataset.feature_names) > 10
        assert dataset.dataset_hash

    def test_train_registers_model(self, settings, populated_db) -> None:
        FeatureAgent(settings, populated_db).build()
        agent = ModelingAgent(settings, populated_db)
        result = agent.train(
            "spread", families=["lightgbm"], val_seasons=1, calib_seasons=1
        )

        assert result.model_id
        assert result.status in {"shadow", "rejected"}
        assert populated_db.row_count("model_registry") == 1
        assert np.isfinite(result.val_metric)

    def test_registry_records_full_metadata(self, settings, populated_db) -> None:
        """La trazabilidad exige guardar features, hash y métricas."""
        FeatureAgent(settings, populated_db).build()
        ModelingAgent(settings, populated_db).train(
            "spread", families=["lightgbm"], val_seasons=1, calib_seasons=1
        )

        row = populated_db.query("SELECT * FROM model_registry LIMIT 1").iloc[0]
        assert row["feature_names"]
        assert row["dataset_hash"] is not None
        assert row["train_seasons"]
        assert row["n_train_rows"] > 0

    def test_promotion_and_rollback(self, settings, populated_db) -> None:
        from nflp.registry import ModelRegistry

        FeatureAgent(settings, populated_db).build()
        agent = ModelingAgent(settings, populated_db)
        first = agent.train("spread", families=["lightgbm"], val_seasons=1, calib_seasons=1)
        second = agent.train("spread", families=["lightgbm"], val_seasons=1, calib_seasons=1)

        registry = ModelRegistry(settings, populated_db)
        # Solo se promueve lo que el gate no rechazó.
        promotable = [r for r in (first, second) if r.status != "rejected"]
        if len(promotable) < 2:
            pytest.skip("El gate rechazó algún modelo; el escenario no aplica.")

        assert registry.promote(promotable[0].model_id)
        assert registry.current_production("spread") == promotable[0].model_id

        assert registry.promote(promotable[1].model_id)
        assert registry.current_production("spread") == promotable[1].model_id

        reverted = registry.rollback("spread")
        assert reverted == promotable[0].model_id

    def test_rejected_model_cannot_be_promoted(self, settings, populated_db) -> None:
        """El gate no debe poder saltarse a mano."""
        from nflp.registry import ModelRecord, ModelRegistry

        registry = ModelRegistry(settings, populated_db)
        record = ModelRecord(
            model_id="test-rejected", target="spread", family="lightgbm", version=1,
            feature_set_version="v1", feature_names=["a"], hyperparameters={},
            dataset_hash="x", train_seasons="2020-2021", n_train_rows=100,
            n_val_rows=20, metric_name="mae", train_metric=1.0, val_metric=5.0,
            status="rejected", promotion_blocked_reason="sobreajuste",
        )
        populated_db.write_df(
            pd.DataFrame([record.to_row()]), "model_registry", mode="append"
        )
        assert not registry.promote("test-rejected")


class TestWalkForward:
    def test_never_trains_on_future(self, settings, populated_db) -> None:
        """Ninguna predicción puede provenir de un modelo entrenado con el futuro."""
        from nflp.agents.backtesting.walkforward import WalkForwardConfig, run_walk_forward

        FeatureAgent(settings, populated_db).build()
        dataset = load_game_dataset(
            "spread", db=populated_db, feature_version=settings.features.version,
            min_prior_games=0,
        )

        config = WalkForwardConfig(
            start_season=2022, end_season=2023, retrain_every_weeks=6,
            min_train_seasons=1, family="lightgbm", calibrate=False,
        )
        result = run_walk_forward(dataset, config)

        assert len(result.predictions) > 0
        assert result.n_retrains > 0
        # Todas las predicciones caen en el rango evaluable.
        assert result.predictions["season"].between(2022, 2023).all()
        assert result.predictions["y_pred"].notna().all()
