"""Passos 3 e 4 das notas: adaptar camadas só com o treino e colocar o pré-processamento DENTRO da rede.

Arquitetura Two-Tower: torre do usuário (perfil de gêneros + nota média + volume) e torre do anime
(gêneros, tipo, episódios, popularidade, nota). O score é o produto escalar dos vetores normalizados.
"""
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from scipy.sparse import csr_matrix

from . import config as C

ITEM_STR = ["genres", "type"]
ITEM_NUM = ["episodes", "members", "rating"]


def item_inputs(animes: pd.DataFrame) -> dict:
    d = {c: tf.constant(animes[c].tolist(), shape=(len(animes), 1)) for c in ITEM_STR}
    d.update({c: animes[c].to_numpy("float32").reshape(-1, 1) for c in ITEM_NUM})
    return d


def adapt_item_layers(train_animes: pd.DataFrame) -> dict:
    """Aprende vocabulário/médias SÓ com os animes vistos pelos usuários de treino."""
    tv = layers.TextVectorization(standardize="lower", split="whitespace", output_mode="multi_hot", name="genres_tv")
    tv.adapt(tf.constant(train_animes["genres"].tolist()))
    type_lookup = layers.StringLookup(output_mode="one_hot", name="type_lookup")
    type_lookup.adapt(tf.constant(train_animes["type"].tolist()))
    norms = {}
    for c in ITEM_NUM:
        x = train_animes[c].to_numpy("float32").reshape(-1, 1)
        if c in ("episodes", "members"):  # escala exponencial -> log1p + z-score
            x = np.log1p(x)
        norms[c] = layers.Normalization(name=f"norm_{c}")
        norms[c].adapt(x)
    return {"tv": tv, "type": type_lookup, "norms": norms}


def genre_matrix(tv, animes: pd.DataFrame) -> np.ndarray:
    """Multi-hot de gêneros de cada anime, no mesmo espaço do vocabulário aprendido."""
    return tv(tf.constant(animes["genres"].tolist())).numpy().astype("float32")


def user_profiles(history: pd.DataFrame, G: np.ndarray, anime_pos: pd.Index, user_ids: np.ndarray):
    """Perfil = média do multi-hot de gêneros dos animes que o usuário gostou no histórico."""
    uidx = pd.Index(user_ids)
    h = history[history["rating"] >= C.LIKE_THRESHOLD]
    rows, cols = uidx.get_indexer(h["user_id"]), anime_pos.get_indexer(h["anime_id"])
    H = csr_matrix((np.ones(len(h), "float32"), (rows, cols)), shape=(len(uidx), G.shape[0]))
    n_liked = np.asarray(H.sum(1)).ravel()
    profile = np.asarray(H @ G) / np.maximum(n_liked, 1)[:, None]
    g = history.groupby("user_id")["rating"]
    avg = g.mean().reindex(uidx).fillna(7.0).to_numpy("float32")
    n = g.size().reindex(uidx).fillna(0).to_numpy("float32")
    return profile.astype("float32"), avg.reshape(-1, 1), n.reshape(-1, 1)


def build_towers(item_layers: dict, vocab_size: int, train_avg: np.ndarray, train_n: np.ndarray, emb_dim=C.EMBEDDING_DIM):
    # --- torre do anime ---
    i_in = {
        "genres": keras.Input((1,), dtype="string", name="genres"),
        "type": keras.Input((1,), dtype="string", name="type"),
        **{c: keras.Input((1,), dtype="float32", name=c) for c in ITEM_NUM},
    }
    feats = [layers.Flatten()(item_layers["tv"](i_in["genres"])), item_layers["type"](i_in["type"])]
    for c in ITEM_NUM:
        x = keras.ops.log1p(i_in[c]) if c in ("episodes", "members") else i_in[c]
        feats.append(item_layers["norms"][c](x))
    x = layers.Dense(128, activation="relu")(layers.concatenate(feats))
    x = layers.Dense(emb_dim)(x)
    v = layers.UnitNormalization(name="item_embedding")(x)
    item_tower = keras.Model(i_in, v, name="item_tower")

    # --- torre do usuário ---
    norm_avg = layers.Normalization(name="norm_avg_rating")
    norm_avg.adapt(train_avg)
    norm_n = layers.Normalization(name="norm_n_hist")
    norm_n.adapt(np.log1p(train_n))
    u_in = {
        "genre_profile": keras.Input((vocab_size,), name="genre_profile"),
        "avg_rating": keras.Input((1,), name="avg_rating"),
        "n_hist": keras.Input((1,), name="n_hist"),
    }
    feats = [u_in["genre_profile"], norm_avg(u_in["avg_rating"]), norm_n(keras.ops.log1p(u_in["n_hist"]))]
    x = layers.Dense(128, activation="relu")(layers.concatenate(feats))
    x = layers.Dense(emb_dim)(x)
    u = layers.UnitNormalization(name="user_embedding")(x)
    user_tower = keras.Model(u_in, u, name="user_tower")

    # --- modelo completo: P(gostar) = sigmoid(a * cos(u, v) + b) ---
    dot = layers.Dot(axes=1)([u, v])
    # a >= 0 (NonNeg): garante que "mais parecido" => "mais provável de gostar"; sem isso o treino pode
    # achar a < 0 e inverter o ranking do top-K (cos alto => P(gostar) baixo).
    out = layers.Dense(
        1, activation="sigmoid", name="p_like",
        kernel_initializer=keras.initializers.Constant(3.0), kernel_constraint=keras.constraints.NonNeg(),
    )(dot)
    model = keras.Model({**u_in, **i_in}, out, name="two_tower")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model, user_tower, item_tower
