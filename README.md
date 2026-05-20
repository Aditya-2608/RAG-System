# AI-Powered Document Search (RAG System)

This project is a Retrieval-Augmented Generation (RAG) application that enables users to extract insights from PDF documents using Google Gemini, LangChain, and an in-memory vector store. Users can upload one or more documents and interact with them through question answering, summarization, and multi-document comparison — all processed in real time.

Built with Streamlit, this system showcases how Large Language Models (LLMs) and vector-based retrieval can be combined to create an intuitive and powerful document analysis tool.


---

## Overview

The application processes uploaded PDFs by extracting text, splitting it into meaningful chunks, generating embeddings, and storing them temporarily in an in-memory vector store.
Users can then:

Ask questions about any uploaded document

Generate concise summaries

Compare multiple PDFs side-by-side


The system is designed for correctness, transparency, and a clean user experience.

---

## Key Features

### PDF Upload & Processing
- Upload one or multiple PDFs.
- Automatic text extraction and preparation for downstream tasks.

### Intelligent Chunking
- Documents are split into smaller segments to improve retrieval accuracy.
- Helps reduce hallucination and ensures precise answer grounding.

### In-Memory Vector Search
- Embeddings are stored in RAM using LangChain’s InMemoryVectorStore.
- Ensures fast retrieval without requiring a persistent database.

### Retrieval-Augmented Question Answering
- Retrieves only the most relevant chunks using similarity search.
- Google Gemini provides factual, context-grounded answers.

### Dynamic Summarization
- Automatically chooses 5–15 bullet points depending on document length.
- Ensures summaries stay clean, structured, and to the point.

### Multi-Document Comparison
- Compares any number of uploaded documents, generating:
  - Individual document overviews
  - Similarities across documents
  - Differences between documents
  - A concluding summary

### Source Transparency
- Every answer includes the exact PDF(s) used in retrieval.

### Database Reset System
- Safely resets the vector database, cached embeddings, and session state.
- Ensures no stale data or outdated chunks remain.

---

## Technology Stack

- Python
- Streamlit (UI)
- LangChain (RAG orchestration)
- Google Generative AI Embeddings
- Google Gemini 2.5 Flash (LLM)
- PDF parsing utilities

---

## How It Works

1. User uploads one or more PDFs.
2. Text is extracted and chunked into meaningful segments.
3. Embeddings are generated using Google’s embedding model.
4. Chunks are stored in an in-memory vectorstore.
5. For Q&A:
  - The retriever fetches the most relevant content.
  - Gemini generates the answer strictly from the retrieved context.
6. For summaries or comparison:
  - The system processes all content and produces structured output.
  - This architecture ensures fast, transparent, and reliable document analysis without external database dependencies.  

---

## Use Cases

- Research paper analysis
- Policy document review
- Corporate report examination
- General-purpose PDF insight extraction
- Education and academic summarization
- Legal, healthcare, or financial document comparison

---

## Author

**Aditi Arya**  

---

*Feel free to contribute, raise issues, or suggest improvements!*
