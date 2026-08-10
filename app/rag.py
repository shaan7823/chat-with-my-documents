import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using only the provided document excerpts.
If the answer is not in the excerpts, say you don't know instead of guessing.

Context:
{context}

Question: {question}

Answer clearly and cite the document names using brackets, for example [document.pdf]."""


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
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "embedding_model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "top_k": int(os.environ.get("TOP_K", 3)),
    }


def load_vector_store(persist_dir: Path, embedding_model: str):
    if not persist_dir.exists():
        raise FileNotFoundError(f"Vector store directory not found: {persist_dir}")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name="documents",
        embedding_function=embeddings,
    )


def build_context(chunks):
    """Create a single context string from retrieved chunks."""
    context_parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source_pdf", "unknown source")
        text = chunk.page_content.strip()
        context_parts.append(f"Source: {source}\n{text}")
    return "\n\n".join(context_parts)


def answer_question(question: str, persist_dir: Path):
    config = load_env()
    if not question or not question.strip():
        raise ValueError("Question must not be empty")

    client = load_vector_store(persist_dir, config["embedding_model"])
    retriever = client.as_retriever(search_kwargs={"k": config["top_k"]})
    relevant_docs = retriever.get_relevant_documents(question)

    if not relevant_docs:
        return {"answer": "No relevant documents found. Try ingesting PDFs or asking a simpler question.", "sources": []}

    context = build_context(relevant_docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatOpenAI(model_name=config["openai_model"], temperature=0)
    response = llm.predict(prompt)

    sources = []
    for chunk in relevant_docs:
        source = chunk.metadata.get("source_pdf")
        if source and source not in sources:
            sources.append(source)

    return {"answer": response, "sources": sources}
