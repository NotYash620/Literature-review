from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from app.engine import content_engine, GraphState
from app.retriever import ingest_documents, clear_database

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ingest")
async def trigger_ingestion(file: UploadFile = File(...)):
    try:
        # Create directory if it doesn't exist
        os.makedirs("./data/papers", exist_ok=True)
        file_path = f"./data/papers/{file.filename}"
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger the RAG ingestion logic
        ingest_documents()
        return {"message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_content(request: QueryRequest):
    initial_state: GraphState = {
        "topic": request.query, 
        "context": "",
        "draft": "",
        "feedback": "",
        "is_approved": False,
        "revision_count": 0
    }
    
    try:
        result = content_engine.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Returning more info for the UI "Agent Thoughts"
    return {
        "topic": result["topic"],
        "total_revisions": result["revision_count"],
        "final_draft": result["draft"],
        "research_context": result.get("context", ""),
        "last_feedback": result.get("feedback", "")
    }

@app.post("/clear")
async def clear_kb():
    try:
        clear_database()
        return {"message": "Knowledge base cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))