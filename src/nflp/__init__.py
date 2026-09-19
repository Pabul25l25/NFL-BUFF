"""NFL-Predictor — plataforma de predicción NFL con foco en corrección metodológica.

Arquitectura multi-agente (ver docs/arquitectura.md):

* ``agents.data``        — ingesta y normalización (calendario, estadios, lesiones, clima)
* ``agents.stats``       — estadísticas históricas de jugador y equipo
* ``agents.features``    — construcción de features con contrato as-of anti-leakage
* ``agents.modeling``    — entrenamiento, calibración y versionado
* ``agents.backtesting`` — validación walk-forward y baselines
* ``agents.monitoring``  — drift, gate de promoción, shadow deployment

Cada agente corre de forma independiente por CLI (``nflp <agente> <comando>``) o
como tarea dentro de los flows de Prefect (``nflp.orchestrator``).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
