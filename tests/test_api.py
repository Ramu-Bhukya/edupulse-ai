import json
from pathlib import Path

import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_ai, get_store
from edupulse.vector_store import VectorStore
from tests.conftest import FakeOpenAI, KeywordEmbedder

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "cell_biology_notes.pdf"


@pytest.fixture
def fake_ai():
    return FakeOpenAI("Mitochondria make ATP [1]."), KeywordEmbedder()


@pytest.fixture
def client(tmp_path, fake_ai):
    store = VectorStore.persistent(str(tmp_path / "chroma"))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_ai] = lambda: fake_ai
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, data=None, name="cells.pdf"):
    files = {"file": (name, data or SAMPLE.read_bytes(), "application/pdf")}
    return client.post("/api/documents", files=files)


def read_events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_health_and_presets(client):
    assert client.get("/api/health").json()["status"] == "ok"
    presets = client.get("/api/presets").json()
    assert "Medical Tutor" in [p["name"] for p in presets["personas"]]
    assert "Socratic" in presets["tones"]


def test_upload_list_and_delete(client):
    first = upload(client)
    assert first.status_code == 200
    doc = first.json()["document"]
    assert doc["page_count"] == 3
    assert first.json()["cached"] is False

    assert upload(client).json()["cached"] is True
    assert [d["doc_id"] for d in client.get("/api/documents").json()] == [doc["doc_id"]]

    assert client.delete(f"/api/documents/{doc['doc_id']}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{doc['doc_id']}").status_code == 404


def test_upload_rejects_non_pdf(client):
    response = upload(client, data=b"hello", name="notes.txt")
    assert response.status_code == 422
    assert "valid PDF" in response.json()["detail"]


def test_search(client):
    doc_id = upload(client).json()["document"]["doc_id"]
    query = {"query": "What does the Golgi apparatus do?", "k": 2}
    response = client.post(f"/api/documents/{doc_id}/search", json=query)
    results = response.json()
    assert len(results) == 2
    assert "Golgi" in results[0]["text"]
    assert {"page_label", "similarity", "index"} <= results[0].keys()


def test_search_unknown_document(client):
    response = client.post("/api/documents/nope/search", json={"query": "x"})
    assert response.status_code == 404


def test_chat_with_document_streams_sources_then_tokens(client, fake_ai):
    doc_id = upload(client).json()["document"]["doc_id"]
    body = {
        "doc_id": doc_id,
        "persona": "Medical Tutor",
        "messages": [{"role": "user", "content": "What do mitochondria do?"}],
    }
    response = client.post("/api/chat", json=body)
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = read_events(response)
    assert events[0]["type"] == "sources" and events[0]["sources"]
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == "Mitochondria make ATP [1]."
    assert events[-1] == {"type": "done"}

    system = fake_ai[0].chat.completions.calls[-1]["messages"][0]["content"]
    assert "medical tutor" in system.lower()
    assert "<context>" in system


def test_chat_without_document(client, fake_ai):
    body = {"messages": [{"role": "user", "content": "Hi"}]}
    events = read_events(client.post("/api/chat", json=body))
    assert events[0] == {"type": "sources", "sources": []}
    assert "<context>" not in fake_ai[0].chat.completions.calls[-1]["messages"][0]["content"]


def test_chat_validation(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    bad_role = {"messages": [{"role": "system", "content": "x"}]}
    assert client.post("/api/chat", json=bad_role).status_code == 422


def test_chat_error_mid_stream_is_reported(client, fake_ai):
    class Broken:
        def create(self, **kwargs):
            request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
            raise openai.RateLimitError(
                "slow down", response=httpx2.Response(429, request=request), body=None
            )

    fake_ai[0].chat.completions = Broken()
    body = {"messages": [{"role": "user", "content": "Hi"}]}
    events = read_events(client.post("/api/chat", json=body))
    assert events[-1]["type"] == "error"
    assert "Too many requests" in events[-1]["message"]


def test_quiz_and_flashcards(client, fake_ai):
    doc_id = upload(client).json()["document"]["doc_id"]
    completions = fake_ai[0].chat.completions
    completions.parsed = {
        "questions": [
            {
                "question": "What makes ATP?",
                "options": ["Mitochondria", "Ribosome", "Golgi", "Nucleus"],
                "correct_index": 0,
                "explanation": "See p. 1.",
            }
        ]
    }
    quiz = client.post(f"/api/documents/{doc_id}/quiz", json={"n": 5}).json()
    q = quiz["questions"][0]
    assert q["options"][q["correct_index"]] == "Mitochondria"

    completions.parsed = {"cards": [{"term": "ATP", "definition": "Energy currency."}]}
    deck = client.post(
        f"/api/documents/{doc_id}/flashcards", json={"topic": "energy", "n": 10}
    ).json()
    assert deck == {"cards": [{"term": "ATP", "definition": "Energy currency."}]}


def test_missing_key_is_401(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app.dependency_overrides[get_store] = lambda: VectorStore.persistent(str(tmp_path / "c"))
    try:
        response = TestClient(app).post(
            "/api/chat", json={"messages": [{"role": "user", "content": "Hi"}]}
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 401
    assert "API key" in response.json()["detail"]
