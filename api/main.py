"""FastAPI backend for the React front end.

It exposes the same edupulse core the Streamlit app uses, so both front
ends share one document index and one set of prompts.

Run locally:
    uvicorn api.main:app --reload --port 8000

If the React app has been built (frontend/dist), it is served from "/" too,
so frontend and API can be deployed as a single service.
"""

import json
import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from edupulse import __version__
from edupulse.config import MissingAPIKeyError, get_api_key, get_settings
from edupulse.embeddings import Embedder, OpenAIEmbedder
from edupulse.ingest import index_pdf
from edupulse.llm import LLM_ERRORS, friendly_error, get_client, stream_chat
from edupulse.pdf_parser import PDFParseError
from edupulse.presets import find_by_name, load_presets
from edupulse.prompts import build_chat_messages, build_system_prompt
from edupulse.rag_engine import answer_stream, source_to_dict
from edupulse.study_tools import (
    FlashcardDeck,
    Quiz,
    generate_flashcards,
    generate_quiz,
    sample_context,
)
from edupulse.vector_store import DocumentInfo, VectorStore

MAX_UPLOAD_MB = 25
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
MAX_CONTEXT_MESSAGES = 10

app = FastAPI(title="EduPulse AI API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSONAS, TONES = load_presets()


# --- dependencies -----------------------------------------------------------------


@lru_cache
def get_store() -> VectorStore:
    return VectorStore.persistent(get_settings().chroma_dir)


@lru_cache(maxsize=8)
def _ai_for_key(api_key: str) -> tuple[OpenAI, OpenAIEmbedder]:
    client = get_client(api_key)
    return client, OpenAIEmbedder(client)


def get_ai(
    x_openai_key: Annotated[str | None, Header()] = None,
) -> tuple[OpenAI, Embedder]:
    """OpenAI client + embedder. The front end may send its own key in X-OpenAI-Key."""
    try:
        return _ai_for_key(get_api_key(x_openai_key))
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


Store = Annotated[VectorStore, Depends(get_store)]
AI = Annotated[tuple[OpenAI, Embedder], Depends(get_ai)]


def require_document(store: VectorStore, doc_id: str) -> DocumentInfo:
    info = store.get_document(doc_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Document not found. Upload it again.")
    return info


def llm_error(exc: Exception) -> HTTPException:
    status = 401 if isinstance(exc, MissingAPIKeyError) else 502
    return HTTPException(status_code=status, detail=friendly_error(exc))


# --- request / response models ---------------------------------------------------


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    doc_id: str | None = None
    persona: str = PERSONAS[0].name
    tone: str = TONES[0].name


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class GenerateRequest(BaseModel):
    topic: str | None = None
    n: int = Field(default=5, ge=1, le=20)


class UploadResponse(BaseModel):
    document: DocumentInfo
    cached: bool


# --- routes ----------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/presets")
def presets() -> dict:
    return {
        "personas": [{"name": p.name, "description": p.description} for p in PERSONAS],
        "tones": [t.name for t in TONES],
    }


@app.get("/api/documents")
def list_documents(store: Store) -> list[DocumentInfo]:
    return store.list_documents()


@app.post("/api/documents")
async def upload_document(
    store: Store, ai: AI, file: Annotated[UploadFile, File()]
) -> UploadResponse:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDFs up to {MAX_UPLOAD_MB} MB please.")
    _, embedder = ai
    try:
        info, cached = index_pdf(data, file.filename or "document.pdf", store, embedder)
    except PDFParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLM_ERRORS as exc:
        raise llm_error(exc) from exc
    return UploadResponse(document=info, cached=cached)


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, store: Store) -> None:
    require_document(store, doc_id)
    store.delete_document(doc_id)


@app.post("/api/documents/{doc_id}/search")
def search(doc_id: str, body: SearchRequest, store: Store, ai: AI) -> list[dict]:
    require_document(store, doc_id)
    _, embedder = ai
    try:
        results = store.search(doc_id, embedder.embed_query(body.query), k=body.k)
    except LLM_ERRORS as exc:
        raise llm_error(exc) from exc
    return [source_to_dict(r) for r in results]


@app.post("/api/chat")
def chat(body: ChatRequest, store: Store, ai: AI) -> StreamingResponse:
    """Stream the answer as NDJSON: a "sources" event, "token" events, then "done"."""
    client, embedder = ai
    persona = find_by_name(PERSONAS, body.persona) or PERSONAS[0]
    tone = find_by_name(TONES, body.tone) or TONES[0]
    history = [m.model_dump() for m in body.messages]
    system_prompt = build_system_prompt(persona, tone)

    try:
        if body.doc_id:
            require_document(store, body.doc_id)
            results, tokens = answer_stream(
                client,
                store,
                embedder,
                body.doc_id,
                system_prompt,
                history,
                max_messages=MAX_CONTEXT_MESSAGES,
            )
        else:
            results = []
            messages = build_chat_messages(system_prompt, history, MAX_CONTEXT_MESSAGES)
            tokens = stream_chat(client, messages)
    except LLM_ERRORS as exc:
        raise llm_error(exc) from exc

    def events() -> Iterator[str]:
        sources = [source_to_dict(r) for r in results]
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        try:
            for token in tokens:
                yield json.dumps({"type": "token", "text": token}) + "\n"
        except LLM_ERRORS as exc:
            # the response has already started, so report the error in-band
            yield json.dumps({"type": "error", "message": friendly_error(exc)}) + "\n"
            return
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/documents/{doc_id}/quiz")
def quiz(doc_id: str, body: GenerateRequest, store: Store, ai: AI) -> Quiz:
    require_document(store, doc_id)
    client, embedder = ai
    try:
        chunks = sample_context(store, embedder, doc_id, body.topic)
        return generate_quiz(client, chunks, n=body.n)
    except LLM_ERRORS as exc:
        raise llm_error(exc) from exc


@app.post("/api/documents/{doc_id}/flashcards")
def flashcards(doc_id: str, body: GenerateRequest, store: Store, ai: AI) -> FlashcardDeck:
    require_document(store, doc_id)
    client, embedder = ai
    try:
        chunks = sample_context(store, embedder, doc_id, body.topic)
        return generate_flashcards(client, chunks, n=body.n)
    except LLM_ERRORS as exc:
        raise llm_error(exc) from exc


# Mounted last so every /api route above takes precedence.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
