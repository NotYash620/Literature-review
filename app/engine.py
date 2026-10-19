from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Import our custom database loader
from app.retriever import get_vectorstore

load_dotenv()

# 1. Expand State to hold the retrieved RAG context
class GraphState(TypedDict):
    topic: str
    context: str  
    draft: str
    feedback: str
    is_approved: bool
    revision_count: int

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

class ReviewResult(BaseModel):
    is_approved: bool = Field(description="True if the draft is perfect, accurate, and exactly two paragraphs. False otherwise.")
    feedback: str = Field(description="Constructive feedback on how to improve the draft. Leave empty if approved.")

# 2. NEW: The Researcher Node
def research_node(state: GraphState):
    print("--- RESEARCH AGENT SEARCHING KNOWLEDGE BASE ---")
    
    # Dynamically connect to the vector database so collection changes/clears are always picked up
    try:
        db = get_vectorstore()
        retriever = db.as_retriever(search_kwargs={"k": 2})  # Fetch top 2 most relevant chunks
        docs = retriever.invoke(state["topic"])
        context = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
    except Exception as e:
        print(f"    -> Retrieval error or database empty: {e}")
        context = ""
    
    if context:
        print("    -> Relevant documents found and injected.")
    else:
        print("    -> No relevant documents found in local database.")
        
    return {"context": context}

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)

# 3. UPDATED: The Writer Node (Now uses context)
def technical_writer_node(state: GraphState):
    current_revisions = state.get("revision_count", 0)
    print(f"--- WRITER AGENT DRAFTING (Attempt {current_revisions + 1}) ---")
    
    if state.get("feedback"):
        # Revision Prompt
        prompt = f"""
        Topic: {state['topic']}
        Background Facts: {state['context']}
        Previous Draft: {state['draft']}
        Reviewer Feedback: {state['feedback']}
        
        Please revise the draft based strictly on the reviewer's feedback. 
        Ensure it remains a professional, two-paragraph technical summary.
        Use the provided Background Facts if applicable.
        """
    else:
        # First-time Draft Prompt (Injecting the RAG Context)
        prompt = f"""
        Write a short, professional, two-paragraph technical summary about: {state['topic']}
        
        Base your summary heavily on the following verified facts:
        {state['context']}
        """
    
    response = llm.invoke(prompt)
    draft_text = extract_text(response.content)
    return {"draft": draft_text, "revision_count": current_revisions + 1}

# 4. The Reviewer Node (Remains the same)
def reviewer_node(state: GraphState):
    print("--- REVIEWER AGENT CHECKING DRAFT ---")
    prompt = f"""
    Evaluate this technical draft about '{state['topic']}'. 
    It must be a highly professional, accurate, and be exactly two paragraphs. 
    If it fails any of these criteria, reject it and provide specific feedback.
    
    Draft to review:
    {state['draft']}
    """
    
    structured_llm = llm.with_structured_output(ReviewResult)
    review: ReviewResult = structured_llm.invoke(prompt)
    
    print(f"    -> Approved: {review.is_approved}")
    if not review.is_approved:
        print(f"    -> Feedback: {review.feedback}")
        
    return {"is_approved": review.is_approved, "feedback": review.feedback}

# 5. The Routing Logic (Remains the same)
def should_publish(state: GraphState):
    if state.get("revision_count", 0) >= 3:
        print("--- MAX REVISIONS REACHED. FORCING PUBLISH. ---")
        return END
    if state.get("is_approved"):
        print("--- DRAFT APPROVED. PUBLISHING. ---")
        return END
    
    print("--- DRAFT REJECTED. ROUTING BACK TO WRITER. ---")
    return "writer"

# 6. UPDATED: Build the Complete Engine Graph
workflow = StateGraph(GraphState)

# Add all three agents
workflow.add_node("researcher", research_node)
workflow.add_node("writer", technical_writer_node)
workflow.add_node("reviewer", reviewer_node)

# Connect the flow: START -> researcher -> writer -> reviewer
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")

# Conditional logic for revisions
workflow.add_conditional_edges("reviewer", should_publish)

# Compile
content_engine = workflow.compile()