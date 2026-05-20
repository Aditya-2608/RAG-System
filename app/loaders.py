from langchain_community.document_loaders import PyPDFLoader
from typing import List

def load_pdf(path: str) -> List:
    loader = PyPDFLoader(path)
    docs = loader.load()  # returns LangChain Document objects with page metadata
    # Ensure source filename metadata exists
    for d in docs:
        d.metadata = d.metadata or {}
    return docs
