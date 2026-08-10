import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pypdf import PdfReader


def load_env():
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required. Set it in .env or as an environment variable."
        )
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return {
        "openai_api_key": openai_api_key,
        "embedding_model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "chunk_size": int(os.environ.get("CHUNK_SIZE", 500)),
        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", 100)),
    }


def load_pdfs(data_dir: Path):
    """Load all PDF files from the data directory using pypdf."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    documents = []
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)

        page_content = "\n\n".join(text).strip()
        if not page_content:
            continue

        documents.append(Document(page_content=page_content, metadata={"source_pdf": pdf_path.name}))
    return documents


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    """
    Split documents into text chunks.

    We choose a chunk size of 500 tokens (approx 400-500 words) so each piece is
    large enough to keep context, but small enough to make retrieval accurate.
    Overlap of 100 tokens helps avoid cutting important sentences at the boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def create_chroma_store(documents, embedding_model: str, persist_dir: Path):
    """Create a Chroma vector store from documents."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=str(persist_dir),
        collection_name="documents",
    )


def ingest_documents(data_dir: Path, persist_dir: Path):
    config = load_env()
    documents = load_pdfs(data_dir)
    chunks = split_documents(documents, config["chunk_size"], config["chunk_overlap"])
    client = create_chroma_store(chunks, config["embedding_model"], persist_dir)
    client.persist()
    return len(chunks)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    persist_dir = Path(__file__).resolve().parents[1] / "db"

    try:
        count = ingest_documents(data_dir, persist_dir)
        print(f"Ingested {count} chunks into Chroma at {persist_dir}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
