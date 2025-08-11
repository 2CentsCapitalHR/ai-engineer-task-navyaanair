"""
RAG indexing with LangChain + Ollama Embeddings + Chroma
"""

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader, JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ---------------- CONFIG ----------------
DATA_DIR = "AllReferences"  # your folder
PERSIST_DIR = "./chroma_langchain"
EMBED_MODEL = "nomic-embed-text"  # Ollama model for embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ---------------- LOADING ----------------
def load_all_documents(data_dir):
    docs = []
    
    # PDFs
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs.extend(pdf_loader.load())

    # DOCX
    docx_loader = DirectoryLoader(
        data_dir,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs.extend(docx_loader.load())

    # JSON
    json_loader = DirectoryLoader(
        data_dir,
        glob="**/*.json",
        loader_cls=lambda path: JSONLoader(
            file_path=path,
            jq_schema=".content",  # adjust this depending on JSON structure
            text_content=False
        )
    )
    docs.extend(json_loader.load())

    return docs

# ---------------- CHUNKING ----------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

# ---------------- EMBEDDING + STORAGE ----------------
def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    return vectorstore

# ---------------- QUERYING ----------------
def query_vectorstore(query, top_k=5):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    results = vectorstore.similarity_search(query, k=top_k)
    return results

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_index", action="store_true", help="Index all documents")
    parser.add_argument("--query", type=str, help="Run a query")
    args = parser.parse_args()

    if args.do_index:
        print("Loading documents...")
        documents = load_all_documents(DATA_DIR)
        print(f"Loaded {len(documents)} documents.")
        
        print("Chunking documents...")
        chunks = chunk_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        
        print("Creating vectorstore...")
        create_vectorstore(chunks)
        print("Indexing complete.")

    if args.query:
        results = query_vectorstore(args.query)
        for i, r in enumerate(results, start=1):
            print(f"\n[{i}] {r.metadata}")
            print(r.page_content[:500], "...")
