Technical Content Engine 
An autonomous Multi-Agent RAG pipeline for Academic Literature Reviews

This project is a high-performance Technical Content Engine built with LangGraph, ChromaDB, and Gemini 3.8 Flash. It moves beyond standard RAG by implementing an agentic workflow where multiple specialized AI agents collaborate, critique, and revise technical drafts based on local research papers.
The Agentic Workflow:

Unlike traditional "one-shot" RAG systems, this engine employs a cyclic multi-agent architecture:

    Researcher Agent: Queries a local vector database (ChromaDB) to extract precise technical facts, metrics, and conclusions from ingested academic PDFs.

    Technical Writer Agent: Takes the retrieved context and synthesizes it into a professional, two-paragraph technical summary.

    Reviewer Agent: Acts as the quality gate. It evaluates the draft for accuracy, length, and tone. If the draft is insufficient, it provides specific feedback and routes the state back to the Writer for revision.

    Self-Correction Loop: The system autonomously iterates up to 3 times until the Reviewer provides approval, ensuring high-fidelity output.

Key Features:

    Agentic RAG: Fact-grounded generation that ignores pre-trained biases in favor of provided document context.

    PDF Ingestion Pipeline: Automated parsing and chunking of dense academic papers into searchable vector embeddings.

    FastAPI Backend: A robust API layer to handle generation requests.

    Streamlit UI: A clean, user-friendly interface to interact with the agent collective and track revision metrics.

Tech Stack:

    Orchestration: LangGraph

    LLM: Google Gemini 3.8 Flash

    Vector Database: ChromaDB

    Embeddings: Google Generative AI (gemini-embedding-001)

    Frameworks: FastAPI, Streamlit, LangChain

    Environment: Conda / Python 3.11+