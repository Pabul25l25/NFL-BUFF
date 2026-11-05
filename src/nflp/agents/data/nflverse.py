"""Cliente de descarga de nflverse.

nflverse publica sus datasets como *release assets* de GitHub en formato parquet:

    https://github.com/nflverse/nflverse-data/releases/download/<tag>/<asset>.parquet

Se descarga directamente en vez de usar ``nfl_data_py`` porque esa librería está
desactualizada y fija versiones antiguas de pandas. El acceso directo además nos
da control sobre la caché y los reintentos.

Caché: cada asset se guarda en ``data/raw/`` junto a un fichero ``.meta.json`` con
su ETag. En la siguiente ejecución se manda ``If-None-Match``; si el servidor
responde 304, se reutiliza el fichero local sin volver a descargar.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nflp.core.config import Settings, get_settings
from nflp.core.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

# Assets sin dimensión temporal: un único fichero para todo el histórico.
STATIC_ASSETS: dict[str, str] = {
    "games": "schedules/games",
    "teams": "teams/teams_colors_logos",
    "players": "players/players",
    "draft_picks": "draft_picks/draft_picks",
}

# Assets particionados por temporada: ``<tag>/<prefijo>_<season>``.
SEASONAL_ASSETS: dict[str, str] = {
    "player_stats": "stats_player/stats_player_week",
    "team_stats": "stats_team/stats_team_week",
    "injuries": "injuries/injuries",
    "snap_counts": "snap_counts/snap_counts",
    "weekly_rosters": "weekly_rosters/roster_weekly",
    "depth_charts": "depth_charts/depth_charts",
}


class NflverseError(RuntimeError):
    """Fallo irrecuperable al obtener un asset de nflverse."""


@dataclass
class FetchResult:
    """Resultado de una descarga, para la bitácora de ingesta."""

    asset: str
    url: str
    path: Path
    etag: str | None
    from_cache: bool
    rows: int


class NflverseClient:
    """Descarga y cachea assets parquet de nflverse."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = self.settings.paths.raw_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0

    # ------------------------------------------------------------------ URLs

    @staticmethod
    def static_url(asset: str) -> str:
        if asset not in STATIC_ASSETS:
            raise NflverseError(f"Asset estático desconocido: {asset}")
        return f"{BASE_URL}/{STATIC_ASSETS[asset]}.parquet"

    @staticmethod
    def seasonal_url(asset: str, season: int) -> str:
        if asset not in SEASONAL_ASSETS:
            raise NflverseError(f"Asset por temporada desconocido: {asset}")
        return f"{BASE_URL}/{SEASONAL_ASSETS[asset]}_{season}.parquet"

    # ------------------------------------------------------------------ caché

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        name = url.rsplit("/", 1)[-1]
        return self.cache_dir / name, self.cache_dir / f"{name}.meta.json"

    def _read_meta(self, meta_path: Path) -> dict[str, str]:
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _throttle(self) -> None:
        """Espaciado mínimo entre peticiones (cortesía con GitHub)."""
        delay = self.settings.ingestion.request_delay_seconds
        elapsed = time.monotonic() - self._last_request
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request = time.monotonic()

    # ------------------------------------------------------------------ fetch

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _download(self, url: str, etag: str | None) -> tuple[bytes | None, str | None]:
        """Descarga el asset. Devuelve ``(None, etag)`` si el servidor responde 304."""
        self._throttle()
        headers = {"User-Agent": "nfl-predictor/0.1"}
        if etag:
            headers["If-None-Match"] = etag

        with httpx.Client(
            timeout=self.settings.ingestion.timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = client.get(url, headers=headers)

        if response.status_code == 304:
            return None, etag
        if response.status_code == 404:
            # Habitual: una temporada aún no publicada para ese asset.
            raise NflverseError(f"Asset no encontrado (404): {url}")
        response.raise_for_status()
        return response.content, response.headers.get("ETag")

    def fetch(self, url: str, *, force: bool = False) -> FetchResult:
        """Obtiene un asset, usando la caché local cuando es posible."""
        data_path, meta_path = self._cache_paths(url)
        meta = self._read_meta(meta_path)
        cached_etag = meta.get("etag") if self.settings.ingestion.cache_enabled else None

        use_cache = (
            not force
            and self.settings.ingestion.cache_enabled
            and data_path.exists()
        )

        content: bytes | None = None
        etag = cached_etag

        if use_cache:
            content, etag = self._download(url, cached_etag)
            if content is None:
                log.debug("Caché válida (304): %s", data_path.name)
        else:
            content, etag = self._download(url, None)

        if content is not None:
            data_path.write_bytes(content)
            meta_path.write_text(
                json.dumps({"etag": etag, "url": url}, ensure_ascii=False),
                encoding="utf-8",
            )

        frame = pd.read_parquet(data_path)
        return FetchResult(
            asset=data_path.stem,
            url=url,
            path=data_path,
            etag=etag,
            from_cache=content is None,
            rows=len(frame),
        )

    # ------------------------------------------------------------------ API

    def load_static(self, asset: str, *, force: bool = False) -> pd.DataFrame:
        """Carga un asset sin partición temporal (calendario, equipos, jugadores)."""
        result = self.fetch(self.static_url(asset), force=force)
        log.info(
            "nflverse %-14s %7d filas %s",
            asset,
            result.rows,
            "(caché)" if result.from_cache else "(descargado)",
        )
        return pd.read_parquet(result.path)

    def load_seasonal(
        self,
        asset: str,
        seasons: list[int],
        *,
        force: bool = False,
        skip_missing: bool = True,
    ) -> pd.DataFrame:
        """Carga y concatena un asset particionado por temporada.

        Args:
            skip_missing: si una temporada no está publicada (404), se omite con
                un aviso en vez de abortar toda la ingesta. Es el caso normal de
                la temporada en curso para assets que aún no se han generado.
        """
        frames: list[pd.DataFrame] = []
        for season in seasons:
            url = self.seasonal_url(asset, season)
            try:
                result = self.fetch(url, force=force)
            except NflverseError as exc:
                if skip_missing:
                    log.warning("nflverse %s %d no disponible: %s", asset, season, exc)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                if skip_missing and exc.response.status_code == 404:
                    log.warning("nflverse %s %d no publicado todavía.", asset, season)
                    continue
                raise

            frame = pd.read_parquet(result.path)
            frames.append(frame)
            log.debug(
                "nflverse %s %d: %d filas %s",
                asset,
                season,
                result.rows,
                "(caché)" if result.from_cache else "(descargado)",
            )

        if not frames:
            log.warning("nflverse %s: ninguna temporada disponible.", asset)
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        log.info(
            "nflverse %-14s %7d filas (%d temporadas)",
            asset,
            len(combined),
            len(frames),
        )
        return combined
