"""Configuración tipada del proyecto.

Carga `config/settings.yaml` y permite sobrescribir cualquier valor mediante
variables de entorno con prefijo ``NFLP_`` (anidadas con ``__``).

Ejemplo::

    NFLP_BACKTEST__START_SEASON=2018
    NFLP_DATABASE__BACKEND=postgresql
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def project_root() -> Path:
    """Raíz del proyecto (el directorio que contiene ``pyproject.toml``)."""
    # src/nflp/core/config.py -> src/nflp/core -> src/nflp -> src -> raíz
    return Path(__file__).resolve().parents[3]


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("data/artifacts")
    reports_dir: Path = Path("data/reports")
    duckdb_path: Path = Path("data/nflp.duckdb")

    def resolve(self, root: Path) -> PathsConfig:
        """Convierte las rutas relativas en absolutas respecto a `root`."""
        resolved = {
            name: (root / value if not value.is_absolute() else value)
            for name, value in self.model_dump().items()
        }
        return PathsConfig(**resolved)

    def ensure_dirs(self) -> None:
        """Crea los directorios de datos si no existen."""
        for directory in (self.data_dir, self.raw_dir, self.artifacts_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    backend: Literal["duckdb", "postgresql"] = "duckdb"
    postgres_url: str = "postgresql+psycopg://nflp:nflp@localhost:5432/nflp"


class IngestionConfig(BaseModel):
    start_season: int = 2011
    end_season: int = 2026
    cache_enabled: bool = True
    request_delay_seconds: float = 0.5
    max_retries: int = 5
    timeout_seconds: int = 60

    @property
    def seasons(self) -> list[int]:
        return list(range(self.start_season, self.end_season + 1))


class WeatherConfig(BaseModel):
    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    batch_size: int = 10
    request_delay_seconds: float = 1.0


class FeaturesConfig(BaseModel):
    version: str = "v1"
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 5, 8])
    ewm_halflife: float = 4.0
    as_of_offset_hours: float = 1.0
    style_clusters: int = 6
    min_prior_games: int = 3


class ModelingConfig(BaseModel):
    random_seed: int = 1729
    targets: list[str] = Field(default_factory=lambda: ["winner", "spread", "total"])
    families: list[str] = Field(default_factory=lambda: ["lightgbm", "xgboost", "catboost"])
    optuna_trials: int = 40
    calibration: Literal["isotonic", "sigmoid"] = "isotonic"


class PropsConfig(BaseModel):
    targets: list[str] = Field(
        default_factory=lambda: [
            "receiving_yards", "rushing_yards", "receptions",
            "receiving_tds", "rushing_tds", "passing_yards", "passing_tds",
            "sacks_suffered",
        ]
    )
    min_projected_snaps: int = 10
    usage_window: int = 5


class BacktestConfig(BaseModel):
    start_season: int = 2015
    end_season: int = 2025
    retrain_every_weeks: int = 4
    min_train_seasons: int = 3


class MonitoringConfig(BaseModel):
    max_train_val_gap: float = 0.15
    psi_threshold: float = 0.2
    shadow_weeks: int = 2
    rollback_degradation: float = 0.10


class MLflowConfig(BaseModel):
    # SQLite: el store de ficheros quedó en modo mantenimiento en MLflow 3.
    tracking_uri: str = "sqlite:///data/mlflow.db"
    experiment_name: str = "nfl-predictor"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rich_tracebacks: bool = True

    @field_validator("level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class Settings(BaseModel):
    """Configuración completa del proyecto."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    props: PropsConfig = Field(default_factory=PropsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _apply_env_overrides(data: dict[str, Any], prefix: str = "NFLP_") -> dict[str, Any]:
    """Aplica variables de entorno ``NFLP_SECCION__CLAVE`` sobre el dict cargado."""
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path = env_key[len(prefix) :].lower().split("__")
        cursor: dict[str, Any] = data
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        # YAML nos da conversión de tipos gratis (int, float, bool, listas).
        try:
            cursor[path[-1]] = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            cursor[path[-1]] = raw_value
    return data


def load_settings(config_path: Path | None = None) -> Settings:
    """Carga la configuración desde YAML + entorno y resuelve rutas absolutas."""
    root = project_root()
    path = config_path or root / "config" / "settings.yaml"

    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    data = _apply_env_overrides(data)
    settings = Settings(**data)
    # Las rutas del YAML son relativas a la raíz del proyecto.
    settings.paths = settings.paths.resolve(root)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Configuración cacheada para todo el proceso."""
    return load_settings()


def reset_settings_cache() -> None:
    """Invalida la caché (útil en pruebas que manipulan el entorno)."""
    get_settings.cache_clear()
