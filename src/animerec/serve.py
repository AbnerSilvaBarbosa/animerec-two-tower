"""Inferência: monta o vetor do usuário (torre do usuário) e busca os animes mais próximos."""
import json

import keras
import numpy as np
import pandas as pd

from . import config as C
from . import modeling  # noqa: F401  (registra camadas)
from .store import search_postgres


class Recommender:
    def __init__(self):
        a = C.ARTIFACTS
        self.user_tower = keras.models.load_model(a / "user_tower.keras")
        meta = json.loads((a / "meta.json").read_text())
        self.vocab = meta["vocab"]
        self.tok = {t: i for i, t in enumerate(self.vocab)}
        self.G = np.load(a / "genre_matrix.npy")
        self.catalog = pd.read_csv(a / "catalog.csv")
        self.pos = pd.Index(self.catalog["anime_id"])
        emb = a / "item_embeddings.npy"
        self.E = np.load(emb) if emb.exists() else None
        self.use_pg = False
        try:
            import psycopg
            with psycopg.connect(C.DATABASE_URL, connect_timeout=2) as c:
                self.use_pg = c.execute("SELECT to_regclass('animes')").fetchone()[0] is not None
        except Exception:
            self.use_pg = False

    def embedding_map(self):
        if self.E is None:
            raise RuntimeError("Índice não gerado. Rode a etapa 'Vetorizar' primeiro.")
        if getattr(self, "_map", None) is None:
            from .viz import EmbeddingMap
            self._map = EmbeddingMap(self.E, self.catalog)
        return self._map

    @property
    def genres(self) -> list[str]:
        return [t for t in self.vocab if t not in ("", "[UNK]")]

    def user_vector(self, genres, liked_ids, avg_rating, n_watched):
        parts = []
        idx = self.pos.get_indexer([i for i in liked_ids if i in self.pos])
        if len(idx):
            parts.append(self.G[idx].mean(0))
        gv = np.zeros(len(self.vocab), "float32")
        for g in genres:
            if g in self.tok:
                gv[self.tok[g]] = 1.0
        if gv.any():
            parts.append(gv)
        if not parts:
            raise ValueError("Informe ao menos um gênero ou um anime que você gostou.")
        prof = np.mean(parts, axis=0, dtype="float32")[None, :]
        x = {"genre_profile": prof, "avg_rating": np.array([[avg_rating]], "float32"),
             "n_hist": np.array([[max(n_watched, len(liked_ids))]], "float32")}
        return self.user_tower.predict(x, verbose=0)[0]

    def recommend(self, genres, liked_ids, avg_rating=7.5, n_watched=20, type_=None, min_rating=0.0, k=12, exclude=()):
        vec = self.user_vector(genres, liked_ids, avg_rating, n_watched)
        self.last_vec = vec  # exposto p/ a API desenhar o vetor do usuário no mapa
        ex = list({*liked_ids, *exclude})
        if self.use_pg:
            try:
                return "postgres", search_postgres(vec, ex, type_, min_rating, k)
            except Exception:
                self.use_pg = False
        if self.E is None:
            raise RuntimeError("Índice não gerado. Rode a etapa 'Indexar' no front.")
        c = self.catalog
        s = self.E @ vec
        ok = ~c["anime_id"].isin(ex) & (c["rating"] >= min_rating)
        if type_:
            ok &= c["type"] == type_
        s = np.where(ok, s, -np.inf)
        top = np.argsort(-s)[:k]
        out = c.iloc[top].rename(columns={"anime_id": "id"}).assign(score=s[top])
        return "numpy", out[["id", "name", "genre", "type", "episodes", "rating", "members", "score"]].to_dict("records")

    def search(self, q: str, limit=10):
        m = self.catalog[self.catalog["name"].str.contains(q, case=False, regex=False, na=False)]
        m = m.sort_values("members", ascending=False).head(limit)
        return [{"id": int(r.anime_id), "name": r.name, "type": r.type, "genre": r.genre} for r in m.itertuples()]
