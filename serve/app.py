"""FastAPI backend + static frontend for NeuroTrace-DAG."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from serve.engine import get_engine

WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="NeuroTrace-DAG", version="2.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")


class Event(BaseModel):
    id: int | None = None
    name: str | None = None
    correct: int = 1


class RecommendIn(BaseModel):
    history: list[Event] = Field(min_length=2)
    k: int = 8


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    engine = get_engine()
    return {"ok": True, "concepts": len(engine.names), "edges": engine.graph.number_of_edges()}


@app.get("/api/metrics")
def metrics():
    return get_engine().metrics


@app.get("/api/scenarios")
def scenarios():
    return get_engine().scenarios()


@app.get("/api/search")
def search(q: str = "", limit: int = 12):
    return get_engine().search(q, limit=limit)


@app.post("/api/recommend")
def recommend(body: RecommendIn):
    try:
        return get_engine().recommend([e.model_dump() for e in body.history], k=body.k)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
