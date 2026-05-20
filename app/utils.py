from typing import List
from langchain_core.documents import Document 

def doc_sources(docs: List[Document]):
    sources = []
    for d in docs:
        md = d.metadata or {}
        src = md.get("source_file") or md.get("source") or "Unknown"
        page = md.get("page") or md.get("page_number") or "Unknown"
        sources.append((str(src), page))
    return sources
