import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import rag
from ingest.ingest import ingest_documents


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


app = FastAPI(title="Chat With My Documents")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"


@app.post("/ingest")
def ingest():
    try:
        count = ingest_documents(DATA_DIR, DB_DIR)
        return {"message": f"Ingested {count} chunks."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    try:
        result = rag.answer_question(request.question, DB_DIR)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No documents ingested yet. Run /ingest first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
