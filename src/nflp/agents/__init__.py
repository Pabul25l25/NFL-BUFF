"""Los seis agentes del sistema.

Cada subpaquete expone una clase ``*Agent`` con un método ``run()`` y un grupo de
comandos propio en la CLI, de modo que puede ejecutarse aislado o dentro del
pipeline orquestado por Prefect.
"""

__all__ = ["backtesting", "data", "features", "modeling", "monitoring", "stats"]
