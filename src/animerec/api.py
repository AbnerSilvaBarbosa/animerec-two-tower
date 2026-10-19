from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config as C
from .pipeline import Pipeline
from .serve import Recommender

state = {"rec": None}


def load_recommender():
    try:
        state["rec"] = Recommender()
    except Exception:  # artefatos ainda não existem
        state["rec"] = None


pipeline = Pipeline(on_ready=load_recommender)


@asynccontextmanager
async def lifespan(_):
    load_recommender()
    yield


app = FastAPI(title="AnimeRec", lifespan=lifespan)


class PrepareOpts(BaseModel):
    n_users: int = Field(15000, ge=500, le=200000)
    seed: int = 42


class TrainOpts(BaseModel):
    epochs: int = Field(5, ge=1, le=50)
    batch_size: int = Field(1024, ge=32, le=8192)


class RecommendReq(BaseModel):
    genres: list[str] = []
    liked_ids: list[int] = []
    avg_rating: float = Field(7.5, ge=1, le=10)
    n_watched: int = Field(20, ge=1)
    type: str | None = None
    min_rating: float = 0.0
    k: int = Field(12, ge=1, le=50)


@app.get("/api/pipeline")
def get_pipeline():
    return {**pipeline.snapshot(), "model_ready": state["rec"] is not None}


@app.post("/api/pipeline/{stage}", status_code=202)
def run_stage(stage: str, body: dict | None = None):
    opts = body or {}
    if stage == "prepare":
        opts = PrepareOpts(**opts).model_dump()
    elif stage == "train":
        opts = TrainOpts(**opts).model_dump()
    else:
        opts = {}
    if not pipeline.start(stage, opts):
        raise HTTPException(409, "Etapa inválida ou já existe uma etapa em execução.")
    return {"started": stage}


@app.get("/api/meta")
def meta():
    rec = state["rec"]
    if rec is None:
        raise HTTPException(503, "Modelo ainda não treinado.")
    types = sorted(rec.catalog["type"].dropna().unique().tolist())
    return {"genres": rec.genres, "types": types, "backend": "postgres" if rec.use_pg else "numpy"}


@app.get("/api/animes/search")
def search(q: str):
    if state["rec"] is None:
        raise HTTPException(503, "Modelo ainda não treinado.")
    return state["rec"].search(q) if len(q) >= 2 else []


@app.post("/api/recommend")
def recommend(req: RecommendReq):
    rec = state["rec"]
    if rec is None:
        raise HTTPException(503, "Modelo ainda não treinado.")
    try:
        backend, items = rec.recommend(req.genres, req.liked_ids, req.avg_rating, req.n_watched, req.type, req.min_rating, req.k)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))
    vec = rec.last_vec
    out = {"backend": backend, "items": items, "user_vec": vec.round(3).tolist()}
    if rec.E is not None:
        out["user_xy"] = rec.embedding_map().project(vec)
        rows = rec.pos.get_indexer([i["id"] for i in items])
        for it, r in zip(items, rows):
            it["vec"] = rec.E[r].round(2).tolist()
    return out


@app.get("/api/viz/overview")
def viz_overview():
    from . import data, viz
    if state["rec"] is not None:
        return viz.overview(state["rec"].catalog)
    if C.ANIME_CSV.exists():
        return viz.overview(data.load_animes())
    raise HTTPException(404, "Catálogo ainda não disponível. Rode a etapa 1.")


@app.get("/api/viz/embedding")
def viz_embedding():
    rec = state["rec"]
    if rec is None or rec.E is None:
        raise HTTPException(503, "Treine e vetorize o modelo antes (etapas 2–4).")
    return rec.embedding_map().payload()


STATIC = C.ROOT / "src" / "animerec" / "static"
app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/learn")
def learn():
    return FileResponse(STATIC / "learn.html")
