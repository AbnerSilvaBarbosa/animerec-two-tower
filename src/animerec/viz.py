"""Dados para as visualizações do front: distribuição do catálogo e mapa 2D (PCA) dos embeddings."""
import numpy as np
import pandas as pd


def overview(catalog: pd.DataFrame) -> dict:
    toks = catalog["genres"].fillna("").str.split().explode()
    toks = toks[~toks.isin(["", "none"])]
    top = toks.value_counts().head(14)
    types = catalog["type"].fillna("Unknown").value_counts()
    rating_bins = np.histogram(catalog["rating"].dropna(), bins=np.arange(1, 11.01, 0.5))[0]
    return {
        "genres": [{"name": k, "n": int(v)} for k, v in top.items()],
        "types": [{"name": k, "n": int(v)} for k, v in types.items()],
        "anime_rating_hist": rating_bins.tolist(),
        "n_animes": len(catalog),
        "top": [
            {"name": r.name_, "members": int(r.members), "rating": float(r.rating)}
            for r in catalog.nlargest(6, "members").rename(columns={"name": "name_"}).itertuples()
        ],
    }


class EmbeddingMap:
    """PCA 2D dos embeddings dos animes. Guarda média/componentes para projetar também o vetor do usuário."""

    def __init__(self, E: np.ndarray, catalog: pd.DataFrame):
        self.mean = E.mean(0)
        _, _, vt = np.linalg.svd(E - self.mean, full_matrices=False)
        self.comp = vt[:2].T
        xy = (E - self.mean) @ self.comp
        self.lo, self.hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
        self.xy = self._scale(xy)

        first = catalog["genres"].fillna("").str.split().str[0].fillna("none")
        self.legend = [g for g in first.value_counts().index if g != "none"][:8]
        self.g = np.array([self.legend.index(g) if g in self.legend else len(self.legend) for g in first])
        mem = np.log1p(catalog["members"].to_numpy("float64"))
        self.size = ((mem - mem.min()) / (mem.max() - mem.min() + 1e-9)).round(3)
        self.catalog = catalog

    def _scale(self, xy):
        return np.clip(2 * (xy - self.lo) / (self.hi - self.lo + 1e-9) - 1, -1.15, 1.15)

    def project(self, v: np.ndarray) -> list[float]:
        return self._scale(((v - self.mean) @ self.comp)[None, :])[0].round(4).tolist()

    def payload(self) -> dict:
        c = self.catalog
        return {
            "id": c["anime_id"].astype(int).tolist(),
            "name": c["name"].tolist(),
            "x": self.xy[:, 0].round(3).tolist(),
            "y": self.xy[:, 1].round(3).tolist(),
            "g": self.g.tolist(),
            "s": self.size.tolist(),
            "legend": self.legend + ["outros"],
        }
