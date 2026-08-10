# Chat With My Documents

A beginner-friendly Retrieval-Augmented Generation (RAG) project using Python, FastAPI, LangChain, OpenAI, and Chroma.

## Project structure

- `app/`
  - `main.py` — FastAPI app with `/ingest` and `/ask`
  - `rag.py` — retrieval + generation pipeline using LangChain
- `ingest/`
  - `ingest.py` — loads PDFs, splits text, generates embeddings, and writes to Chroma
- `data/` — place your PDF files here
- `tests/` — pytest tests for basic chunking and API behavior
- `.env.example` — example environment variables
- `requirements.txt` / `pyproject.toml` — dependency definitions

## Setup

1. Create a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set your OpenAI key:

```bash
copy .env.example .env
```

4. Put your PDF files into the `data/` folder.

## Ingestion

The ingestion script:

- loads PDFs from `data/`
- splits them into chunks for better retrieval
- embeds each chunk with OpenAI embeddings
- stores vectors locally in Chroma (`db/`)

Run ingestion with:

```bash
python ingest\ingest.py
```

Or call the API endpoint:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

## Why chunking matters

We use `CHUNK_SIZE=500` and `CHUNK_OVERLAP=100`.

- `chunk_size=500` keeps each chunk big enough to preserve meaning.
- `chunk_overlap=100` ensures important sentences are not split across chunks.

## Run the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

### Example `/ask` request

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about our product?"}'
```

Response:

```json
{
  "answer": "...",
  "sources": ["example.pdf"]
}
```

## Tests

Run tests with:

```bash
pytest
```

## Notes

- The app uses OpenAI for both embeddings and text generation.
- The code is intentionally simple so you can learn how ingestion, retrieval, and generation fit together.
- If you want to swap providers later, replace the OpenAI-specific LangChain classes in `app/rag.py` and `ingest/ingest.py`.
