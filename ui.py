import streamlit as st
import requests

st.set_page_config(page_title="Tech Content Engine", page_icon="🤖", layout="wide")

st.title("Multi-Agent Technical Content Engine")

# --- SIDEBAR: File Upload ---
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    
    if st.button("Ingest Paper"):
        if uploaded_file:
            with st.spinner("Breaking down PDF and embedding into ChromaDB..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post("http://127.0.0.1:8000/ingest", files=files)
                    if response.status_code == 200:
                        st.success(f"Learned: {uploaded_file.name}")
                    else:
                        st.error(f"Ingestion failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI server at http://127.0.0.1:8000. Please start the backend.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Upload a file first!")
    
    st.divider()
    st.info("The agents will only use facts found in the uploaded documents.")

# --- MAIN: Query & Generation ---
query = st.text_area("Research Topic", placeholder="e.g. Compare SHAP and LIME stability metrics...")

if st.button("Start Agent Collaboration"):
    if not query.strip():
        st.warning("Enter a topic!")
    else:
        with st.spinner("Researcher is searching, Writer is drafting..."):
            try:
                res = requests.post("http://127.0.0.1:8000/generate", json={"query": query})
                if res.status_code == 200:
                    data = res.json()
                    
                    # Show the Final Result
                    st.subheader("Final Peer-Reviewed Draft")
                    draft = data["final_draft"]
                    if isinstance(draft, list):
                        draft = "\n\n".join(
                            part["text"] if isinstance(part, dict) and "text" in part else str(part)
                            for part in draft
                        )
                    st.markdown(draft)
                    
                    # --- NEW: AGENT THOUGHTS (Phase 1 Upgrade) ---
                    st.divider()
                    st.subheader("Agent Collaboration Logs")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("Researcher: Retrieved Chunks"):
                            st.write(data["research_context"] if data["research_context"] else "No direct facts found.")
                    
                    with col2:
                        with st.expander("Reviewer: Final Critique"):
                            st.metric("Total Revisions", data["total_revisions"])
                            st.write(data["last_feedback"] if data.get("last_feedback") else "Approved on first attempt.")
                else:
                    st.error(f"Backend error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server at http://127.0.0.1:8000. Please start the backend.")
            except Exception as e:
                st.error(f"Error: {e}")


if st.sidebar.button("🗑️ Clear Knowledge Base"):
    try:
        res = requests.post("http://127.0.0.1:8000/clear")
        if res.status_code == 200:
            st.sidebar.success("Database wiped! You can now ingest a fresh paper.")
        else:
            st.sidebar.error(f"Failed to clear database: {res.text}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("Could not connect to FastAPI server at http://127.0.0.1:8000. Please start the backend.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")