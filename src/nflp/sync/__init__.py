"""Publicación de un subconjunto de datos hacia la nube (Neon/Postgres).

Existe por una única razón: el frontend nuevo en React se despliega en
Vercel, que es *serverless* y no tiene forma de leer
``D:\\NFL\\data\\nflp.duckdb`` (disco local del usuario, red doméstica). El
pipeline de Python no cambia en nada — sigue calculando todo igual, en la PC,
con DuckDB — y al final simplemente publica una copia de lo necesario para
mostrar en pantalla.
"""

from nflp.sync.publish import publish_all

__all__ = ["publish_all"]
