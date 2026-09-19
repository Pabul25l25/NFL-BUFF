"""Logging centralizado (sección 1 del prompt: logging centralizado del orquestador).

Los mensajes se emiten en español. El formato es legible en consola vía Rich y
se puede volcar a fichero para las corridas del pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False

# Consola compartida: el dashboard y la CLI escriben por el mismo canal.
console = Console(stderr=True)


def setup_logging(
    level: str = "INFO",
    *,
    rich_tracebacks: bool = True,
    log_file: Path | None = None,
) -> None:
    """Configura el logging raíz. Idempotente: llamarlo dos veces no duplica handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=rich_tracebacks,
            show_path=False,
            omit_repeated_times=False,
            markup=False,
        )
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Silenciamos librerías ruidosas que no aportan a la traza del pipeline.
    for noisy in ("httpx", "httpcore", "urllib3", "matplotlib", "numexpr", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Logger con el prefijo del proyecto."""
    return logging.getLogger(name if name.startswith("nflp") else f"nflp.{name}")
