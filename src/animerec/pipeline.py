"""Orquestra as etapas que o front dispara: preparar -> treinar -> testar -> indexar."""
import json
import threading
import time
import traceback

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config as C
from . import data, modeling
from .store import ingest_postgres

STAGES = ["prepare", "train", "evaluate", "index"]


class Pipeline:
    def __init__(self, on_ready=None):
        self.lock = threading.Lock()
        self.on_ready = on_ready  # chamado quando artefatos novos ficam disponíveis p/ servir
        self.status = {s: "pending" for s in STAGES}
        self.running = None
        self.logs: list[str] = []
        self.progress = 0.0
        self.stats: dict = {}
        self.history: list[dict] = []
        self.metrics: dict = {}
        self.index_info: dict = {}
        self.splits = None
        self.ctx: dict = {}
        self._restore()

    # ---------- infra ----------
    def log(self, msg: str):
        self.logs.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        self.logs = self.logs[-300:]

    def _restore(self):
        """Se já existem artefatos de uma execução anterior, mostra no front."""
        meta = C.ARTIFACTS / "pipeline.json"
        if meta.exists():
            d = json.loads(meta.read_text())
            self.stats, self.history = d.get("stats", {}), d.get("history", [])
            self.metrics, self.index_info = d.get("metrics", {}), d.get("index_info", {})
            self.status.update(d.get("status", {}))

    def _persist(self):
        C.ARTIFACTS.mkdir(exist_ok=True)
        (C.ARTIFACTS / "pipeline.json").write_text(json.dumps({
            "stats": self.stats, "history": self.history, "metrics": self.metrics,
            "index_info": self.index_info, "status": self.status,
        }))

    def snapshot(self) -> dict:
        return {
            "status": self.status, "running": self.running, "progress": self.progress, "logs": self.logs[-80:],
            "stats": self.stats, "history": self.history, "metrics": self.metrics, "index_info": self.index_info,
        }

    def start(self, stage: str, opts: dict) -> bool:
        if stage not in STAGES or not self.lock.acquire(blocking=False):
            return False
        self.running = stage
        threading.Thread(target=self._run, args=(stage, opts), daemon=True).start()
        return True

    def _run(self, stage, opts):
        self.status[stage], self.progress = "running", 0.0
        try:
            getattr(self, f"stage_{stage}")(**opts)
            self.status[stage] = "done"
            # reexecutar uma etapa invalida as seguintes
            if stage in ("prepare", "train"):
                for s in STAGES[STAGES.index(stage) + 1:]:
                    self.status[s] = "pending"
            self.progress = 1.0
        except Exception as e:  # mostra o erro no front
            self.status[stage] = "error"
            self.log(f"ERRO em {stage}: {e}")
            self.log(traceback.format_exc(limit=3))
        finally:
            self._persist()
            self.running = None
            self.lock.release()

    # ---------- etapa 1: dados + split ----------
    def stage_prepare(self, n_users: int = 15000, seed: int = 42):
        self.log("Descompactando archive.zip (se necessário)...")
        if data.extract():
            self.log("CSVs extraídos para data/raw/")
        self.log(f"Lendo rating.csv e fazendo o split por usuário (amostra de {n_users} usuários)...")
        self.splits = data.make_splits(n_users, seed)
        self.stats = self.splits.stats
        self.history, self.metrics, self.index_info, self.ctx = [], {}, {}, {}
        self.log(f"Split OK: {self.stats['train_examples']} exemplos de treino / {self.stats['test_examples']} de teste.")

    # ---------- etapa 2: adapt + treino ----------
    def stage_train(self, epochs: int = 5, batch_size: int = 1024):
        if self.splits is None:
            raise RuntimeError("Rode a etapa 'Dados e split' primeiro.")
        sp = self.splits
        animes, r = sp.animes, sp.ratings
        anime_pos = pd.Index(animes["anime_id"])
        tr, te = r[r["train_user"]], r[~r["train_user"]]

        self.log("Adaptando camadas SÓ com os animes vistos pelos usuários de treino (.adapt)...")
        seen = animes[animes["anime_id"].isin(tr["anime_id"].unique())]
        il = modeling.adapt_item_layers(seen)
        G = modeling.genre_matrix(il["tv"], animes)
        vocab = il["tv"].get_vocabulary()
        self.log(f"Vocabulário de gêneros: {len(vocab)} tokens | tipos: {il['type'].vocabulary_size()}")

        def build(part: pd.DataFrame):
            users = part["user_id"].unique()
            prof, avg, n = modeling.user_profiles(part[part["history"]], G, anime_pos, users)
            tgt = part[~part["history"]]
            ui = pd.Index(users).get_indexer(tgt["user_id"])
            items = modeling.item_inputs(animes.iloc[anime_pos.get_indexer(tgt["anime_id"])])
            x = {"genre_profile": prof[ui], "avg_rating": avg[ui], "n_hist": n[ui], **items}
            y = (tgt["rating"] >= C.LIKE_THRESHOLD).to_numpy("float32")
            return x, y, (users, prof, avg, n)

        x_tr, y_tr, (u_tr, p_tr, a_tr, n_tr) = build(tr)
        x_te, y_te, test_pack = build(te)
        model, user_tower, item_tower = modeling.build_towers(il, len(vocab), a_tr, n_tr)

        def to_ds(x, y, shuffle):
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                ds = ds.shuffle(100_000, seed=42)
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        steps = int(np.ceil(len(y_tr) / batch_size))
        outer = self

        class UI(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                outer.progress = (self.epoch + batch / steps) / epochs

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch

            def on_epoch_end(self, epoch, logs=None):
                outer.history.append({"epoch": epoch + 1, **{k: round(float(v), 4) for k, v in logs.items()}})
                outer.log(f"epoch {epoch + 1}/{epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in logs.items()))

        self.history = []
        self.log(f"Treinando {len(y_tr)} exemplos, {epochs} épocas, batch {batch_size}...")
        model.fit(to_ds(x_tr, y_tr, True), validation_data=to_ds(x_te, y_te, False), epochs=epochs, verbose=0, callbacks=[UI()])

        C.ARTIFACTS.mkdir(exist_ok=True)
        model.save(C.ARTIFACTS / "two_tower.keras")
        user_tower.save(C.ARTIFACTS / "user_tower.keras")
        item_tower.save(C.ARTIFACTS / "item_tower.keras")
        (C.ARTIFACTS / "meta.json").write_text(json.dumps({"vocab": vocab, "like_threshold": C.LIKE_THRESHOLD}))
        np.save(C.ARTIFACTS / "genre_matrix.npy", G)
        animes.to_csv(C.ARTIFACTS / "catalog.csv", index=False)
        self.ctx = {"model": model, "user_tower": user_tower, "item_tower": item_tower, "G": G,
                    "x_te": x_te, "y_te": y_te, "test_pack": test_pack}
        self.log("Modelo salvo em artifacts/ (two_tower, user_tower, item_tower).")

    # ---------- etapa 3: teste ----------
    def stage_evaluate(self, k: int = 20):
        if not self.ctx:
            raise RuntimeError("Treine o modelo primeiro (nesta sessão).")
        c, sp = self.ctx, self.splits
        self.log("Avaliando no conjunto de teste (usuários que o modelo nunca viu)...")
        loss, acc, auc = c["model"].evaluate(
            tf.data.Dataset.from_tensor_slices((c["x_te"], c["y_te"])).batch(2048), verbose=0)

        animes, r = sp.animes, sp.ratings
        anime_pos = pd.Index(animes["anime_id"])
        E = c["item_tower"].predict(modeling.item_inputs(animes), batch_size=2048, verbose=0)
        users, prof, avg, n = c["test_pack"]
        U = c["user_tower"].predict({"genre_profile": prof, "avg_rating": avg, "n_hist": n}, batch_size=2048, verbose=0)
        scores = U @ E.T
        pop = animes["members"].to_numpy()

        te = r[~r["train_user"]]
        upos = pd.Index(users)
        seen = te[te["history"]]
        scores[upos.get_indexer(seen["user_id"]), anime_pos.get_indexer(seen["anime_id"])] = -np.inf
        rel = te[(~te["history"]) & (te["rating"] >= C.LIKE_THRESHOLD)]
        rel_by_user = rel.groupby("user_id")["anime_id"].apply(lambda s: set(anime_pos.get_indexer(s)))
        mask_seen = np.zeros_like(scores, dtype=bool)
        mask_seen[upos.get_indexer(seen["user_id"]), anime_pos.get_indexer(seen["anime_id"])] = True

        top_model = np.argpartition(-scores, k, axis=1)[:, :k]
        pop_scores = np.where(mask_seen, -np.inf, pop[None, :])
        top_pop = np.argpartition(-pop_scores, k, axis=1)[:, :k]
        rec_m, rec_p = [], []
        for uid, items in rel_by_user.items():
            i = upos.get_loc(uid)
            rec_m.append(len(items & set(top_model[i])) / len(items))
            rec_p.append(len(items & set(top_pop[i])) / len(items))
        self.metrics = {
            "test_loss": round(float(loss), 4), "test_accuracy": round(float(acc), 4), "test_auc": round(float(auc), 4),
            "k": k, "recall_at_k_model": round(float(np.mean(rec_m)), 4), "recall_at_k_popularity": round(float(np.mean(rec_p)), 4),
            "users_evaluated": len(rec_m),
        }
        self.log(f"AUC={auc:.4f} acc={acc:.4f} | recall@{k}: modelo={self.metrics['recall_at_k_model']} vs popularidade={self.metrics['recall_at_k_popularity']}")

    # ---------- etapa 4: vetorizar no Postgres ----------
    def stage_index(self):
        if not self.ctx and not (C.ARTIFACTS / "item_tower.keras").exists():
            raise RuntimeError("Treine o modelo primeiro.")
        sp = self.splits
        animes = sp.animes if sp is not None else pd.read_csv(C.ARTIFACTS / "catalog.csv")
        item_tower = self.ctx.get("item_tower") or keras.models.load_model(C.ARTIFACTS / "item_tower.keras")
        self.log("Gerando embeddings dos animes com a torre de itens...")
        E = item_tower.predict(modeling.item_inputs(animes), batch_size=2048, verbose=0).astype("float32")
        np.save(C.ARTIFACTS / "item_embeddings.npy", E)
        self.log(f"{E.shape[0]} vetores de dimensão {E.shape[1]}. Gravando no Postgres (pgvector)...")
        try:
            ingest_postgres(animes, E)
            self.index_info = {"postgres": True, "rows": int(E.shape[0]), "dim": int(E.shape[1])}
            self.log("Postgres: tabela 'animes' + índice HNSW (cosseno) criados.")
        except Exception as e:
            self.index_info = {"postgres": False, "rows": int(E.shape[0]), "dim": int(E.shape[1]), "error": str(e)[:200]}
            self.log(f"Postgres indisponível ({e}). Recomendação vai usar numpy em memória.")
        if self.on_ready:
            self.on_ready()
