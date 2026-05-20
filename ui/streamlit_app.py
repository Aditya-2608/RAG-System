import os
from pathlib import Path
import uuid
import streamlit as st
from dotenv import load_dotenv
import sys

# Ensure project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.loaders import load_pdf
from app.text_splitter import split_documents
from app.vectorstore import build_vectorstore
from app.chains import (
    make_conversational_chain,
    make_summarizer_chain,
    make_compare_chain
)
from app.config import (
    UPLOAD_DIR,
    GEMINI_MODEL_NAME
)

load_dotenv()

# PAGE + HEADER

st.set_page_config(page_title="RAG System", layout="wide")
st.title("🚀 AI-Powered Document Search ")


# 🔥 ----------------------
# REMOVED SIDEBAR COMPLETELY
# 🔥 ----------------------


# FILE UPLOAD

uploaded = st.file_uploader("📄 Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    st.markdown("### Processing Uploaded Files...")

    docs_all = []
    progress = st.progress(0)

    for i, f in enumerate(uploaded):
        out_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{f.name}"
        with open(out_path, "wb") as fh:
            fh.write(f.getbuffer())

        raw_docs = load_pdf(str(out_path))
        for d in raw_docs:
            d.metadata["source_file"] = f.name

        docs_all.extend(raw_docs)
        progress.progress((i + 1) / len(uploaded))

    st.success(" PDF(s) loaded successfully!")

    
    # CHUNKING
    
    st.markdown("### Splitting into chunks...")
    chunks = split_documents(docs_all)
    st.success(f" Created {len(chunks)} chunks")

    
    # BUILD VECTORSTORE
    
    st.markdown("### Building Vector Database...")
    store = build_vectorstore(chunks, persist=False)
    st.success("Vectorstore built!")

    # Create RAG components
    retriever = store.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": 2}     
    )

    conv_chain = make_conversational_chain(retriever, model_name=GEMINI_MODEL_NAME)
    summarizer_chain = make_summarizer_chain(model_name=GEMINI_MODEL_NAME)
    compare_chain = make_compare_chain(model_name=GEMINI_MODEL_NAME)

    
    # Q&A SECTION
    
    st.markdown("## 💬 Ask Questions")
    question = st.text_input("Your question:")

    if question:
        with st.spinner("Thinking..."):
            result = conv_chain.invoke({"input": question})

        st.markdown("### Answer")
        st.write(result["answer"])

        if "context" in result:
            st.markdown("### 📚 Sources Used")

            unique_files = []
            for doc in result["context"]:
                if doc.metadata["source_file"] not in unique_files and doc.page_content.strip() in result["answer"]:
                    unique_files.append(doc.metadata["source_file"])

            if not unique_files:
                unique_files = list({d.metadata["source_file"] for d in result["context"]})

            for i, file in enumerate(unique_files, 1):
                st.write(f"{i}. {file}")

    
    # QUICK ACTIONS
    
    st.markdown("---")
    st.markdown("## ⚡ Quick Actions")

    col1, col2 = st.columns(2)

    # SUMMARY
    with col1:
        if st.button("📝 Summarize"):

            full_text = "\n\n".join([c.page_content for c in chunks])
            total_words = len(full_text.split())

            if total_words < 500:
                bullet_count = 5
            elif total_words < 1500:
                bullet_count = 7
            elif total_words < 3000:
                bullet_count = 10
            elif total_words < 5000:
                bullet_count = 12
            else:
                bullet_count = 15

            summary = summarizer_chain.run({
                "doc": full_text,
                "bullet_count": bullet_count
            })

            st.markdown(f"### 📝 Summary ({bullet_count} bullets)")
            st.write(summary)

    # MULTI-DOCUMENT COMPARISON
    with col2:
        if st.button("📋 Compare Documents") and len(docs_all) >= 2:

            grouped = {}
            for d in docs_all:
                fname = d.metadata.get("source_file", "Unknown")
                grouped.setdefault(fname, []).append(d)

            if len(grouped) < 2:
                st.error("Need at least two different documents to compare.")
            else:
                file_summaries = []

                for fname, docs in grouped.items():
                    text = "\n\n".join([d.page_content for d in docs])
                    file_summaries.append(f"### Document: {fname}\n\n{text}\n\n")

                combined_text = "\n\n".join(file_summaries)

                comp = compare_chain.run({
                    "all_docs": combined_text,
                    "filenames": ", ".join(grouped.keys())
                })

                st.markdown("### 📋 Comparison")
                st.write(comp)

else:
    st.info(" Upload PDFs above.")

# Signature
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; margin-top:
30px;'>
        Developed by ✨ <i>Aditi Arya</i> ✨
    </div>
    """,
    unsafe_allow_html=True
)    