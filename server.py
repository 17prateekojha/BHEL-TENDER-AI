"""Local API used by the VS Code extension.  Run: uvicorn server:app --port 8000"""
from fastapi import FastAPI
from pydantic import BaseModel

import rag

app = FastAPI(title="Document RAG Assistant")


class Query(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok", "docs_dir": str(rag.DOCS_DIR)}


@app.post("/ask")
def ask(q: Query):
    return rag.answer(q.query)


@app.post("/reindex")
def reindex():
    return rag.build_index()
