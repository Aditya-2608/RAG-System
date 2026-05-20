import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# Embeddings (required to include API key on Streamlit Cloud)
def _embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )


# Build vectorstore (fully in-memory, safest for cloud)
def build_vectorstore(docs, persist: bool = False):
    embed = _embeddings()

    # Defensive check: avoid empty docs error
    docs = [d for d in docs if d.page_content.strip()]

    if len(docs) == 0:
        raise ValueError("No valid text extracted from the uploaded PDF.")

    store = InMemoryVectorStore.from_documents(
        docs,
        embedding=embed
    )

    return store


def load_vectorstore():
    return None  # In-memory vectorstore cannot be reloaded

