import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
DB_DIR = "./data/chroma_db"
PAPERS_DIR = "./data/papers"

def ingest_documents():
    print(f"Scanning directory {PAPERS_DIR} for PDFs...")
    loader = PyPDFDirectoryLoader(PAPERS_DIR)
    docs = loader.load()
    
    if not docs:
        print("No PDFs found!")
        return

    print(f"Loaded {len(docs)} pages. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    total_chunks = len(chunks)
    print(f"Total chunks to process: {total_chunks}")

    # Start fresh or load existing
    batch_size = 25  # Very safe batch size
    
    print(f"Ingesting first batch of {batch_size}...")
    vectorstore = Chroma.from_documents(
        documents=chunks[:batch_size], 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )

    # Loop through the rest
    for i in range(batch_size, total_chunks, batch_size):
        print(f"--- Completed {i}/{total_chunks} chunks ---")
        print(f"Sleeping for 30 seconds to reset API quota...")
        time.sleep(30) 
        
        batch = chunks[i : i + batch_size]
        print(f"Ingesting next {len(batch)} chunks...")
        vectorstore.add_documents(batch)

    print("SUCCESS: All papers ingested into ChromaDB.")

def get_vectorstore():
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

if __name__ == "__main__":
    ingest_documents()


def clear_database():
    # 1. Clear the vector data inside Chroma without deleting the locked folder
    try:
        db = get_vectorstore()
        # This empties the data but keeps the connection alive
        db.delete_collection()
        print("--- VECTOR DATA CLEARED ---")
    except Exception as e:
        print(f"Error clearing vector data: {e}")

    # 2. Clear the actual PDF files so they aren't re-ingested next time
    papers_dir = "./data/papers"
    if os.path.exists(papers_dir):
        for filename in os.listdir(papers_dir):
            file_path = os.path.join(papers_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting PDF file {file_path}: {e}")
        print("--- SOURCE PAPERS CLEARED ---")