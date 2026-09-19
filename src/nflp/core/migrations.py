"""Runner de migraciones versionadas.

Decisión de diseño: en lugar de Alembic usamos ficheros SQL numerados aplicados
por este runner. Motivo: `duckdb-engine` tiene huecos conocidos con el
autogenerate y los tipos de Alembic, mientras que el SQL plano que escribimos es
válido tanto en DuckDB como en PostgreSQL. Se mantiene la propiedad pedida por el
prompt —migraciones versionadas, ordenadas e idempotentes— sin la fragilidad.

Cada fichero vive en ``migrations/`` con el patrón ``NNN_nombre.sql``.
La tabla ``schema_migrations`` registra las ya aplicadas.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nflp.core.config import project_root
from nflp.core.db import Database
from nflp.core.logging import get_logger

log = get_logger(__name__)

_FILENAME_RE = re.compile(r"^(\d+)_(.+)\.sql$")

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER      NOT NULL,
    name        VARCHAR      NOT NULL,
    checksum    VARCHAR      NOT NULL,
    applied_at  TIMESTAMP    NOT NULL,
    PRIMARY KEY (version)
);
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()[:16]


def discover_migrations(directory: Path | None = None) -> list[Migration]:
    """Lee y ordena las migraciones disponibles en disco."""
    directory = directory or project_root() / "migrations"
    if not directory.exists():
        return []

    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            log.warning("Fichero de migración ignorado (nombre inválido): %s", path.name)
            continue
        migrations.append(
            Migration(
                version=int(match.group(1)),
                name=match.group(2),
                path=path,
                sql=path.read_text(encoding="utf-8"),
            )
        )

    versions = [m.version for m in migrations]
    duplicates = {v for v in versions if versions.count(v) > 1}
    if duplicates:
        raise RuntimeError(f"Versiones de migración duplicadas: {sorted(duplicates)}")

    return sorted(migrations, key=lambda m: m.version)


def applied_versions(db: Database) -> dict[int, str]:
    """Versiones ya aplicadas -> checksum registrado."""
    db.execute_script(_BOOTSTRAP)
    df = db.query("SELECT version, checksum FROM schema_migrations ORDER BY version")
    return dict(zip(df["version"].tolist(), df["checksum"].tolist(), strict=True))


def migrate(db: Database, directory: Path | None = None) -> list[Migration]:
    """Aplica las migraciones pendientes. Devuelve las aplicadas en esta llamada."""
    migrations = discover_migrations(directory)
    if not migrations:
        log.warning("No se encontraron ficheros de migración.")
        return []

    already = applied_versions(db)
    pending: list[Migration] = []

    for migration in migrations:
        recorded = already.get(migration.version)
        if recorded is None:
            pending.append(migration)
        elif recorded != migration.checksum:
            # Una migración ya aplicada cambió de contenido: es un error de
            # proceso, no algo que debamos arreglar en silencio.
            raise RuntimeError(
                f"La migración {migration.version:03d}_{migration.name} fue modificada "
                f"después de aplicarse (checksum {recorded} -> {migration.checksum}). "
                "Crea una migración nueva en vez de editar una existente."
            )

    if not pending:
        log.info("Base de datos al día: %d migraciones aplicadas.", len(already))
        return []

    for migration in pending:
        log.info("Aplicando migración %03d_%s ...", migration.version, migration.name)
        db.execute_script(migration.sql)
        db.execute(
            "INSERT INTO schema_migrations (version, name, checksum, applied_at) "
            "VALUES (?, ?, ?, ?)"
            if db.backend == "duckdb"
            else "INSERT INTO schema_migrations (version, name, checksum, applied_at) "
            "VALUES (:v, :n, :c, :a)",
            [migration.version, migration.name, migration.checksum, datetime.now(UTC)]
            if db.backend == "duckdb"
            else {
                "v": migration.version,
                "n": migration.name,
                "c": migration.checksum,
                "a": datetime.now(UTC),
            },
        )

    log.info("Aplicadas %d migraciones nuevas.", len(pending))
    return pending


def current_version(db: Database) -> int:
    """Versión más alta aplicada (0 si la base está vacía)."""
    versions = applied_versions(db)
    return max(versions) if versions else 0
