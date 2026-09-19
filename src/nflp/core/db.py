"""Capa de acceso a datos.

Soporta dos backends con el mismo SQL:

* **DuckDB** (por defecto) — analítica local, sin servidor, lectura directa de parquet.
* **PostgreSQL** — vía SQLAlchemy, para el despliegue con Docker Compose.

El objetivo es que los agentes escriban SQL una sola vez. Por eso se evita
deliberadamente cualquier sintaxis específica de un motor en las migraciones y
consultas; las diferencias inevitables se aíslan aquí.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from nflp.core.config import Settings, get_settings
from nflp.core.logging import get_logger

log = get_logger(__name__)

# DuckDB no admite dos conexiones de escritura al mismo fichero desde procesos
# distintos; dentro del proceso serializamos las escrituras con este lock.
_WRITE_LOCK = threading.RLock()


class Database:
    """Fachada única de acceso a la base de datos."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.database.backend

    # ------------------------------------------------------------------ conexión

    @contextmanager
    def connect(self, read_only: bool = False) -> Iterator[Any]:
        """Devuelve una conexión al backend configurado.

        Para DuckDB entrega un `duckdb.DuckDBPyConnection`; para PostgreSQL, una
        conexión SQLAlchemy. Ambas exponen lo que necesitan los agentes a través
        de los métodos de esta clase.
        """
        if self.backend == "duckdb":
            import duckdb

            path = self.settings.paths.duckdb_path
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(path), read_only=read_only)
            try:
                # httpfs permite leer parquet remoto de nflverse sin descargar a disco.
                conn.execute("INSTALL httpfs; LOAD httpfs;")
            except Exception:  # noqa: BLE001 - la extensión es opcional
                log.debug("Extensión httpfs no disponible; se usará descarga a disco.")
            try:
                yield conn
            finally:
                conn.close()
        else:
            from sqlalchemy import create_engine

            engine = create_engine(self.settings.database.postgres_url)
            conn = engine.connect()
            try:
                yield conn
            finally:
                conn.close()
                engine.dispose()

    # ------------------------------------------------------------------ lectura

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Ejecuta una consulta y devuelve un DataFrame."""
        with self.connect(read_only=False) as conn:
            if self.backend == "duckdb":
                result = conn.execute(sql, list(params) if params else None)
                return result.fetch_df()
            from sqlalchemy import text

            return pd.read_sql(text(sql), conn, params=params)

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """Devuelve el primer valor de la primera fila, o None.

        Normaliza los nulos de pandas (``pd.NA``, ``NaT``, ``NaN``) a ``None``:
        de lo contrario un `SELECT max(...)` sobre una tabla vacía devuelve
        ``pd.NA`` y cualquier evaluación booleana posterior falla.
        """
        df = self.query(sql, params)
        if df.empty:
            return None
        value = df.iloc[0, 0]
        return None if pd.isna(value) else value

    def table_exists(self, table: str) -> bool:
        sql = (
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema NOT IN ('pg_catalog','information_schema')"
        )
        if self.backend != "duckdb":
            sql = sql.replace("?", "%(t)s")
            return bool(self.scalar(sql, {"t": table}))  # type: ignore[arg-type]
        return bool(self.scalar(sql, [table]))

    def row_count(self, table: str) -> int:
        if not self.table_exists(table):
            return 0
        return int(self.scalar(f"SELECT count(*) FROM {table}") or 0)

    # ------------------------------------------------------------------ escritura

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        """Ejecuta una sentencia sin resultado."""
        with _WRITE_LOCK, self.connect() as conn:
            if self.backend == "duckdb":
                conn.execute(sql, list(params) if params else None)
            else:
                from sqlalchemy import text

                conn.execute(text(sql), params or {})
                conn.commit()

    def execute_script(self, sql: str) -> None:
        """Ejecuta un script con varias sentencias separadas por ``;``."""
        with _WRITE_LOCK, self.connect() as conn:
            if self.backend == "duckdb":
                conn.execute(sql)
            else:
                from sqlalchemy import text

                for statement in _split_statements(sql):
                    conn.execute(text(statement))
                conn.commit()

    def write_df(
        self,
        df: pd.DataFrame,
        table: str,
        *,
        mode: str = "append",
        key_columns: Sequence[str] | None = None,
    ) -> int:
        """Escribe un DataFrame en una tabla existente.

        Args:
            mode: ``append`` añade filas; ``replace`` vacía la tabla primero;
                ``upsert`` borra las filas cuya clave coincide y luego inserta
                (semántica idempotente, necesaria para reingestas semanales).
            key_columns: columnas clave, obligatorias cuando ``mode='upsert'``.

        Returns:
            Número de filas escritas.
        """
        if df.empty:
            log.debug("Sin filas que escribir en %s.", table)
            return 0

        if mode == "upsert" and not key_columns:
            raise ValueError("El modo 'upsert' requiere key_columns.")

        # Alineamos el DataFrame al esquema real de la tabla: solo columnas
        # existentes y en el orden correcto. Esto evita fallos cuando nflverse
        # añade columnas nuevas aguas arriba.
        table_columns = self.get_columns(table)
        if table_columns:
            usable = [c for c in table_columns if c in df.columns]
            missing = [c for c in df.columns if c not in table_columns]
            if missing:
                log.debug("Columnas ignoradas para %s: %s", table, ", ".join(sorted(missing)[:10]))
            df = df[usable]

        with _WRITE_LOCK, self.connect() as conn:
            if self.backend == "duckdb":
                conn.register("_staging_df", df)
                try:
                    if mode == "replace":
                        conn.execute(f"DELETE FROM {table}")
                    elif mode == "upsert":
                        keys = ", ".join(key_columns)  # type: ignore[arg-type]
                        conn.execute(
                            f"DELETE FROM {table} WHERE ({keys}) IN "
                            f"(SELECT {keys} FROM _staging_df)"
                        )
                    cols = ", ".join(df.columns)
                    conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _staging_df")
                finally:
                    conn.unregister("_staging_df")
            else:
                from sqlalchemy import text

                if mode == "replace":
                    conn.execute(text(f"DELETE FROM {table}"))
                elif mode == "upsert":
                    keys = list(key_columns)  # type: ignore[arg-type]
                    tuples = df[keys].drop_duplicates().itertuples(index=False, name=None)
                    clause = " OR ".join(
                        "(" + " AND ".join(f"{k} = :{k}_{i}" for k in keys) + ")"
                        for i, _ in enumerate(df[keys].drop_duplicates().itertuples())
                    )
                    params = {
                        f"{k}_{i}": v
                        for i, row in enumerate(tuples)
                        for k, v in zip(keys, row, strict=True)
                    }
                    if clause:
                        conn.execute(text(f"DELETE FROM {table} WHERE {clause}"), params)
                df.to_sql(table, conn, if_exists="append", index=False, method="multi")
                conn.commit()

        log.debug("Escritas %d filas en %s (modo=%s).", len(df), table, mode)
        return len(df)

    def get_columns(self, table: str) -> list[str]:
        """Columnas de una tabla, o lista vacía si no existe."""
        if not self.table_exists(table):
            return []
        df = self.query(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{table}' ORDER BY ordinal_position"
        )
        return df["column_name"].tolist()


def _split_statements(script: str) -> list[str]:
    """Divide un script SQL en sentencias, ignorando ``;`` dentro de literales."""
    statements: list[str] = []
    buffer: list[str] = []
    in_string = False
    for char in script:
        if char == "'":
            in_string = not in_string
        if char == ";" and not in_string:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def get_db(settings: Settings | None = None) -> Database:
    """Atajo para obtener la fachada de base de datos."""
    return Database(settings)


def duckdb_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).paths.duckdb_path
