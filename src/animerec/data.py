"""Passos 1 e 2 das notas: extrair dados brutos e fazer o split ANTES de qualquer cálculo."""
import html
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


def extract() -> bool:
    """Descompacta o archive.zip se os CSVs ainda não existirem. Retorna True se extraiu."""
    if C.ANIME_CSV.exists() and C.RATING_CSV.exists():
        return False
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(C.ZIP_PATH) as z:
        z.extractall(C.RAW_DIR)
    return True


def tokenize_genres(s) -> str:
    """'Slice of Life, Sci-Fi' -> 'slice_of_life sci-fi' (um token por gênero, serializável no grafo)."""
    if not isinstance(s, str) or not s.strip():
        return "none"
    return " ".join(g.strip().lower().replace(" ", "_") for g in s.split(","))


def load_animes() -> pd.DataFrame:
    df = pd.read_csv(C.ANIME_CSV)
    df["name"] = df["name"].map(html.unescape)
    df["genre"] = df["genre"].fillna("")
    df["genres"] = df["genre"].map(tokenize_genres)
    df["type"] = df["type"].fillna("Unknown")
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce").fillna(0).astype("float32")
    df["rating"] = df["rating"].fillna(df["rating"].median()).astype("float32")
    df["members"] = df["members"].astype("float32")
    return df.reset_index(drop=True)


@dataclass
class Splits:
    animes: pd.DataFrame
    ratings: pd.DataFrame  # user_id, anime_id, rating, train_user, history
    stats: dict


def make_splits(n_users: int, seed: int = 42) -> Splits:
    """Split por USUÁRIO (treino/teste) e, dentro de cada usuário, histórico (perfil) x alvo (exemplos).

    Nada é calculado (médias, vocabulários) antes deste ponto -> sem data leakage.
    """
    rng = np.random.default_rng(seed)
    animes = load_animes()

    raw = pd.read_csv(C.RATING_CSV, dtype={"user_id": "int32", "anime_id": "int32", "rating": "int8"})
    n_raw = len(raw)
    n_unrated = int((raw["rating"] < 0).sum())  # -1 = assistiu mas não avaliou
    r = raw[(raw["rating"] >= 0) & raw["anime_id"].isin(animes["anime_id"])]
    del raw

    counts = r.groupby("user_id").size()
    eligible = counts[counts >= C.MIN_RATINGS].index.to_numpy()
    users = rng.permutation(eligible)[:n_users]
    cut = int(len(users) * (1 - C.TEST_USER_FRACTION))
    train_users = users[:cut]

    r = r[r["user_id"].isin(users)].copy()
    r["train_user"] = r["user_id"].isin(train_users)
    r["history"] = rng.random(len(r)) < C.HISTORY_FRACTION

    tr, te = r[r["train_user"]], r[~r["train_user"]]
    stats = {
        "animes": len(animes),
        "ratings_raw": n_raw,
        "ratings_unrated_dropped": n_unrated,
        "eligible_users": len(eligible),
        "users_sampled": len(users),
        "train_users": len(train_users),
        "test_users": len(users) - len(train_users),
        "train_examples": int((~tr["history"]).sum()),
        "test_examples": int((~te["history"]).sum()),
        "train_history_rows": int(tr["history"].sum()),
        "test_history_rows": int(te["history"].sum()),
        "liked_ratio_train": round(float(((tr["rating"] >= C.LIKE_THRESHOLD)[~tr["history"]]).mean()), 4),
        "rating_hist": np.bincount(r["rating"].to_numpy(), minlength=11).tolist(),
    }
    return Splits(animes=animes, ratings=r.reset_index(drop=True), stats=stats)
