"""Capa de acceso a datos.

Soporta dos backends con el mismo SQL:

* **DuckDB** (por defecto) — analítica local, sin servidor, lectura directa de parquet.
* **PostgreSQL** — vía SQLAlchemy, para el despliegue con Docker Compose.

El objetivo es que los agentes escriban SQL una sola vez. Por eso se evita
deliberadamente cualquier sintaxis específica de un motor en las migraciones y
consultas; las diferencias inevitables se aíslan aquí.
"""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from nflp.core.config import Settings, get_settings
from nflp.core.logging import get_logger


def _normalize_postgres_url(url: str) -> str:
    """Fuerza el dialecto ``psycopg`` (v3) en cualquier cadena de conexión.

    El proyecto depende de `psycopg[binary]` (v3), pero los proveedores de
    Postgres (Neon incluido) entregan cadenas con el prefijo genérico
    ``postgresql://``, que SQLAlchemy resuelve por defecto al dialecto
    `psycopg2` — no instalado aquí, y produce un `ModuleNotFoundError` confuso.
    Se normaliza el prefijo para que cualquier cadena copiada tal cual del
    proveedor funcione sin que haya que acordarse de añadir `+psycopg`.
    """
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):  # alias corto que usan algunos proveedores
        return "postgresql+psycopg://" + url[len("postgres://") :]
    return url


# PostgreSQL admite como máximo 65.535 parámetros por sentencia. `df.to_sql`
# con `method="multi"` los agota fácil en tablas anchas o con muchas filas
# (``ncols`` × ``chunksize`` filas por lote) si no se limita explícitamente.
_POSTGRES_MAX_PARAMS = 65_535


def _safe_chunksize(n_columns: int) -> int:
    """Filas por lote que mantienen el envío bien por debajo del límite."""
    return max(1, (_POSTGRES_MAX_PARAMS // 2) // max(n_columns, 1))

log = get_logger(__name__)

# Identificadores SQL (tablas/columnas) que se interpolan en el texto de una
# sentencia: nunca pueden enlazarse como parámetro, así que se validan contra
# esta expresión antes de usarlos.
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def validate_identifier(name: str) -> str:
    """Devuelve ``name`` si es un identificador SQL seguro; si no, ValueError."""
    if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Identificador SQL no válido: {name!r}")
    return name


def _qmark_to_named(sql: str, n_params: int) -> str:
    """Reescribe los ``?`` posicionales como ``:_p0``, ``:_p1``... para `text()`.

    El SQL del proyecto usa ``?`` (estilo nativo de DuckDB) como marcador
    portable. SQLAlchemy `text()` solo entiende parámetros nombrados, así que
    para Postgres se traducen aquí, ignorando los ``?`` dentro de literales
    entre comillas simples o de comentarios ``-- ...``.
    """
    out: list[str] = []
    index = 0
    in_string = False
    in_comment = False
    for i, char in enumerate(sql):
        if in_comment:
            out.append(char)
            if char == "\n":
                in_comment = False
        elif char == "'":
            in_string = not in_string
            out.append(char)
        elif not in_string and char == "-" and sql[i + 1 : i + 2] == "-":
            in_comment = True
            out.append(char)
        elif char == "?" and not in_string:
            out.append(f":_p{index}")
            index += 1
        else:
            out.append(char)
    if index != n_params:
        raise ValueError(
            f"La sentencia tiene {index} marcadores '?' pero se pasaron {n_params} parámetros."
        )
    return "".join(out)


def _bind_postgres(sql: str, params: Any) -> tuple[str, dict[str, Any]]:
    """Normaliza (sql, params) al estilo nombrado que espera SQLAlchemy `text()`."""
    if not params:
        return sql, {}
    if isinstance(params, dict):
        return sql, dict(params)
    values = list(params)
    return _qmark_to_named(sql, len(values)), {f"_p{i}": v for i, v in enumerate(values)}


def _bind_duckdb(sql: str, params: Any) -> tuple[str, Any]:
    """Normaliza (sql, params) para DuckDB: ``?`` + lista, o ``$nombre`` + dict."""
    if not params:
        return sql, None
    if isinstance(params, dict):
        # Acepta el mismo estilo `:nombre` que Postgres y lo traduce a `$nombre`
        # (sin tocar los casts `::tipo`).
        for key in params:
            sql = re.sub(rf"(?<![:\w]):{re.escape(key)}\b", f"${key}", sql)
        return sql, dict(params)
    return sql, list(params)


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

            engine = create_engine(_normalize_postgres_url(self.settings.database.postgres_url))
            conn = engine.connect()
            try:
                yield conn
            finally:
                conn.close()
                engine.dispose()

    # ------------------------------------------------------------------ lectura

    def query(
        self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Ejecuta una consulta y devuelve un DataFrame.

        ``params`` se enlaza siempre como parámetro, nunca se interpola. El
        estilo portable es ``?`` posicional con una lista (funciona igual en
        DuckDB y en Postgres); también se acepta ``:nombre`` con un dict.
        """
        with self.connect(read_only=False) as conn:
            if self.backend == "duckdb":
                bound_sql, bound = _bind_duckdb(sql, params)
                result = conn.execute(bound_sql, bound)
                return result.fetch_df()
            from sqlalchemy import text

            bound_sql, bound = _bind_postgres(sql, params)
            return pd.read_sql(text(bound_sql), conn, params=bound)

    def scalar(
        self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None
    ) -> Any:
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
        # `?` se traduce al estilo nombrado de SQLAlchemy (`:_p0`) para Postgres
        # en `_bind_postgres`; nunca se escribe `%(t)s` de psycopg a mano.
        sql = (
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema NOT IN ('pg_catalog','information_schema')"
        )
        return bool(self.scalar(sql, [table]))

    def row_count(self, table: str) -> int:
        if not self.table_exists(table):
            return 0
        return int(self.scalar(f"SELECT count(*) FROM {validate_identifier(table)}") or 0)

    # ------------------------------------------------------------------ escritura

    def execute(
        self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None
    ) -> None:
        """Ejecuta una sentencia sin resultado (mismos estilos de parámetro que `query`)."""
        with _WRITE_LOCK, self.connect() as conn:
            if self.backend == "duckdb":
                bound_sql, bound = _bind_duckdb(sql, params)
                conn.execute(bound_sql, bound)
            else:
                from sqlalchemy import text

                bound_sql, bound = _bind_postgres(sql, params)
                conn.execute(text(bound_sql), bound)
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

        # Tabla y claves se interpolan en el SQL: validarlas como identificadores.
        validate_identifier(table)
        for key in key_columns or ():
            validate_identifier(key)

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
                    # Tabla temporal + DELETE por subconsulta: el mismo patrón
                    # que ya usa el backend DuckDB (`register` + `IN (SELECT
                    # ... FROM _staging_df)`). La versión anterior armaba una
                    # cláusula OR con un parámetro nombrado POR FILA — con
                    # tablas de miles de filas (`games`, `player_stats_weekly`)
                    # eso supera el límite de 65.535 parámetros por sentencia
                    # de Postgres y la conexión simplemente se cae a mitad de
                    # envío. Nunca se notó contra DuckDB porque ese backend usa
                    # una ruta completamente distinta.
                    keys = list(key_columns)  # type: ignore[arg-type]
                    key_frame = df[keys].drop_duplicates()
                    staging = f"_upsert_keys_{uuid.uuid4().hex[:8]}"
                    key_frame.to_sql(
                        staging, conn, if_exists="replace", index=False,
                        method="multi", chunksize=_safe_chunksize(len(keys)),
                    )
                    keys_sql = ", ".join(keys)
                    try:
                        conn.execute(
                            text(
                                f"DELETE FROM {table} WHERE ({keys_sql}) IN "
                                f"(SELECT {keys_sql} FROM {staging})"
                            )
                        )
                    finally:
                        # Se limpia pase lo que pase: dejar una tabla de
                        # apoyo huérfana en Neon es peor que un DELETE fallido
                        # y visible en el log.
                        conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))
                df.to_sql(
                    table, conn, if_exists="append", index=False,
                    method="multi", chunksize=_safe_chunksize(len(df.columns)),
                )
                conn.commit()

        log.debug("Escritas %d filas en %s (modo=%s).", len(df), table, mode)
        return len(df)

    def get_columns(self, table: str) -> list[str]:
        """Columnas de una tabla, o lista vacía si no existe."""
        if not self.table_exists(table):
            return []
        df = self.query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        )
        return df["column_name"].tolist()


def _split_statements(script: str) -> list[str]:
    """Divide un script SQL en sentencias, ignorando ``;`` dentro de literales
    o de comentarios ``-- ...``.

    **Bug real que motivó el segundo caso**: varios comentarios de las
    migraciones contienen un punto y coma dentro del propio texto (p. ej.
    "no supervisado; feature auxiliar"). Sin reconocer el comentario, ese `;`
    se interpretaba como fin de sentencia y partía un `CREATE TABLE` a la
    mitad — invisible contra DuckDB (que ejecuta el script entero de una vez,
    sin pasar por esta función) y solo se manifestó al aplicar las mismas
    migraciones contra Postgres/Neon por primera vez.
    """
    statements: list[str] = []
    buffer: list[str] = []
    in_string = False
    in_comment = False
    chars = script
    i = 0
    while i < len(chars):
        char = chars[i]
        if in_comment:
            buffer.append(char)
            if char == "\n":
                in_comment = False
            i += 1
            continue
        if char == "'":
            in_string = not in_string
            buffer.append(char)
            i += 1
            continue
        if not in_string and char == "-" and i + 1 < len(chars) and chars[i + 1] == "-":
            in_comment = True
            buffer.append(char)
            i += 1
            continue
        if char == ";" and not in_string:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            i += 1
            continue
        buffer.append(char)
        i += 1
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def get_db(settings: Settings | None = None) -> Database:
    """Atajo para obtener la fachada de base de datos."""
    return Database(settings)


def duckdb_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).paths.duckdb_path
