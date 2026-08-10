import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import rag


class DummyDoc:
    def __init__(self, content, source):
        self.page_content = content
        self.metadata = {"source_pdf": source}


@pytest.fixture(autouse=True)
def mock_rag(monkeypatch):
    def fake_answer_question(question, persist_dir):
        return {"answer": "This is a mocked answer.", "sources": ["sample.pdf"]}

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)


client = TestClient(app)


def test_ask_endpoint_returns_answer():
    response = client.post("/ask", json={"question": "What is test?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mocked answer."
    assert data["sources"] == ["sample.pdf"]


def test_ask_endpoint_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400
    data = response.json()
    assert "Question must not be empty" in data["detail"]
