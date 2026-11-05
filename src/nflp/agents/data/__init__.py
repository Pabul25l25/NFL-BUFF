"""Data Agent — ingesta y normalización de datos.

Responsabilidad única: dejar en la base de datos una copia limpia, normalizada y
auditable de calendario, equipos, estadios, rosters, lesiones y clima.
"""

from nflp.agents.data.agent import DataAgent, IngestionReport
from nflp.agents.data.nflverse import NflverseClient
from nflp.agents.data.stadiums import haversine_km, stadiums_frame

__all__ = [
    "DataAgent",
    "IngestionReport",
    "NflverseClient",
    "haversine_km",
    "stadiums_frame",
]
