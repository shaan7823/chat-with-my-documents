from pathlib import Path

from ingest.ingest import split_documents
from langchain.schema import Document


def test_split_documents_creates_chunks():
    documents = [Document(page_content="This is a test document. " * 100, metadata={"source_pdf": "sample.pdf"})]
    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all("sample.pdf" in chunk.metadata.get("source_pdf", "") for chunk in chunks)
    assert all(len(chunk.page_content) <= 120 for chunk in chunks)


def test_split_documents_overlap_keeps_context():
    text = "Sentence one. Sentence two. Sentence three. " * 20
    documents = [Document(page_content=text, metadata={"source_pdf": "sample.pdf"})]
    chunks = split_documents(documents, chunk_size=100, chunk_overlap=30)

    assert len(chunks) > 1
    assert "Sentence" in chunks[0].page_content
    assert "Sentence" in chunks[1].page_content
    assert chunks[0].metadata["source_pdf"] == "sample.pdf"
