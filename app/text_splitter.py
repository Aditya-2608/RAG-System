from langchain_text_splitters import RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "? ", "! "]
)

def split_documents(docs):
    return SPLITTER.split_documents(docs)
