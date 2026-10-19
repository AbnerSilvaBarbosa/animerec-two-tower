import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import config as C


def ingest_postgres(animes, E: np.ndarray):
    dim = E.shape[1]
    with psycopg.connect(C.DATABASE_URL, connect_timeout=5, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS animes")
        conn.execute(
            f"""CREATE TABLE animes (id int PRIMARY KEY, name text, genre text, type text,
                episodes real, rating real, members int, embedding vector({dim}))"""
        )
        rows = [
            (int(a.anime_id), a.name, a.genre, a.type, float(a.episodes), float(a.rating), int(a.members), E[i])
            for i, a in enumerate(animes.itertuples())
        ]
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO animes VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        conn.execute("CREATE INDEX animes_emb_hnsw ON animes USING hnsw (embedding vector_cosine_ops)")


def search_postgres(vec: np.ndarray, exclude: list[int], type_: str | None, min_rating: float, k: int):
    with psycopg.connect(C.DATABASE_URL, connect_timeout=3) as conn:
        register_vector(conn)
        rows = conn.execute(
            """SELECT id, name, genre, type, episodes, rating, members, 1 - (embedding <=> %(v)s) AS score
               FROM animes
               WHERE id <> ALL(%(ex)s) AND rating >= %(mr)s AND (%(t)s::text IS NULL OR type = %(t)s)
               ORDER BY embedding <=> %(v)s LIMIT %(k)s""",
            {"v": vec, "ex": exclude, "mr": min_rating, "t": type_, "k": k},
        ).fetchall()
    return [dict(zip(["id", "name", "genre", "type", "episodes", "rating", "members", "score"], r)) for r in rows]
