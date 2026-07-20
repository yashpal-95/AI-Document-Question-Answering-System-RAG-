import os
import shutil
import uvicorn
import chromadb
import PyPDF2
import pytesseract
import uuid

from typing import TypedDict, List
from datetime import datetime
from pdf2image import convert_from_path
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# ML & LangGraph Imports
from sentence_transformers import SentenceTransformer, CrossEncoder

from langgraph.graph import StateGraph, END

# Import the database manager from database.py (Assumed existing)
from database import db_manager

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

def call_ollama(prompt: str):

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json().get("response", "").strip()

    except Exception as e:
        print("OLLAMA ERROR:", e)
        return ""
    
# --- STATE DEFINITION FOR LANGGRAPH ---
class RAGState(TypedDict):  
    user_id: str
    question: str
    retrieved_docs: List[str]  # Raw docs from Chroma
    reranked_context: str      # Filtered & sorted text for the LLM
    answer: str
    status: str                # To track if context was found

# --- GLOBAL MODEL STORAGE ---
models = {}

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- STARTUP: Connecting to DB & Loading Models ---")
    db_manager.connect()

    models["embed"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    models["reranker"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("--- Models Loaded Successfully ---")
    yield

    print("--- SHUTDOWN: Disconnecting ---")
    db_manager.disconnect()
    models.clear()
    
app = FastAPI(title="Multi-User OCR RAG API (with Reranking & LangGraph)", lifespan=lifespan)

# --- INITIALIZATION ---
UPLOAD_DIR = "./uploaded_pdfs"
CHROMA_PATH = "./chromadb"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="pdf_chunks")

# --- SCHEMAS ---
class QuestionRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = 5 # Increased default to allow reranker to filter down

class DeleteRequest(BaseModel):
    user_id: str

class AnswerResponse(BaseModel):
    user_id: str
    question: str
    answer: str
    hallucination_percent: float
    accuracy_score: float

# --- HELPER LOGIC ---
def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]



def llm_evaluate_answer(question, answer, context):

    if not answer or not context:
        return 0.0, 100.0

    prompt = f"""
You are an expert RAG evaluator.

Evaluate whether the answer is supported ONLY by the context.

Return STRICT JSON only:

{{
  "accuracy_score": <integer 0-100>,
  "hallucination_score": <integer 0-100>
}}

Question:
{question}

Context:
{context}

Answer:
{answer}
"""

    result = call_ollama(prompt)

    print("EVALUATOR RAW OUTPUT:", result)

    import re
    import json
    try:
        json_match = re.search(r"\{.*\}", result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            accuracy = float(parsed.get("accuracy_score", 0))
            hallucination = float(parsed.get("hallucination_score", 100))
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print("JSON Parsing Failed:", e)
        accuracy = 0.0
        hallucination = 100.0
    return accuracy, hallucination
    
    
# --- LANGGRAPH NODES ---

def retrieve_node(state: RAGState):
    """
    Step 1: Fetch raw documents from ChromaDB based on embedding similarity.
    """
    question = state["question"]
    user_id = state["user_id"]
    
    # Encode question
    q_emb = models["embed"].encode(question).tolist()
    
    # Query Chroma (fetch slightly more docs, e.g., top 10, to let reranker filter best 3)
    results = collection.query(
        query_embeddings=[q_emb], 
        n_results=10, 
        where={"user_id": user_id}
    )
    
    docs = results['documents'][0] if results['documents'] else []
    
    if not docs:
        return {"status": "no_docs", "retrieved_docs": []}
    
    return {"retrieved_docs": docs, "status": "docs_found"}

def rerank_node(state: RAGState):
    """
    Step 2: Use Cross-Encoder to re-score documents specifically against the question.
    """
    if state["status"] == "no_docs":
        return {"reranked_context": ""}

    question = state["question"]
    docs = state["retrieved_docs"]
    reranker = models["reranker"]

    # Prepare pairs [ [query, doc1], [query, doc2] ... ]
    pairs = [[question, doc] for doc in docs]
    
    # Get scores (higher is better)
    scores = reranker.predict(pairs)
    
    # Combine docs with scores and sort desc
    scored_docs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    
    # Take top 3 highest quality chunks
    top_docs = [doc for doc, score in scored_docs[:3]]
    
    # Join them for the LLM Context
    context_text = " ".join(top_docs)
    
    return {"reranked_context": context_text}

def generate_node(state: RAGState):

    if state["status"] == "no_docs" or not state["reranked_context"]:
        return {"answer": "No relevant context found."}

    context = state["reranked_context"]
    question = state["question"]

    prompt = f"""
You are an expert assistant.

Answer strictly using ONLY the context below.
If answer not present, say NOT_AVAILABLE.

Context:
{context}

Question:
{question}

Answer:
"""

    result = call_ollama(prompt)
    answer = result.strip()

    if not answer:
        answer = "No relevant context found in the provided document."

    if "NOT_AVAILABLE" in answer.upper():
        answer = "No relevant context found in the provided document."

    return {"answer": answer}


# --- BUILD THE GRAPH ---
def build_graph():
    workflow = StateGraph(RAGState)
    
    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    
    # Add Edges (Linear Flow)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Initialize the compiled graph
rag_app = build_graph()

# --- ENDPOINTS ---

@app.post("/upload")
async def upload_pdf(user_id: str, file: UploadFile = File(...)):
    ALLOWED_CONTENT_TYPE = "application/pdf"
    filename = file.filename.lower()
    if file.content_type != ALLOWED_CONTENT_TYPE or not filename.endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are acceptable.")

    user_dir = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # --- OCR & INDEXING LOGIC ---
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 20:
                    text += page_text + "\n"
        
        # OCR Fallback
        if len(text.strip()) < 50:
            images = convert_from_path(file_path)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"

        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            chunk_id = f"{user_id}_{file.filename}_{i}_{uuid.uuid4().hex[:8]}"
            # Use the global embed model
            emb = models["embed"].encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                metadatas=[{"source": file.filename, "user_id": user_id}],
                ids=[chunk_id],
                embeddings=[emb]
            )
            
        await db_manager.save_file_metadata(user_id, file.filename)
        return {"message": f"Successfully indexed {file.filename} for User: {user_id}"}
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # 1. Validation
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    user_context = await db_manager.get_user_context(request.user_id)
    if user_context is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # 2. Invoke LangGraph
    # We pass the initial state
    inputs = {
        "user_id": request.user_id, 
        "question": request.question,
        "retrieved_docs": [],
        "reranked_context": "",
        "answer": "",
        "status": "start"
    }
    
    # Run the graph
    result_state = rag_app.invoke(inputs)
    
    final_answer = result_state["answer"]
    final_context = result_state["reranked_context"]

    # 3. Calculate Metrics
    # Note: We calculate metrics here in the endpoint, not the graph, 
    # to keep the graph focused on generation logic.
    accuracy_scr, hallucination_pct = llm_evaluate_answer(
    request.question,
    final_answer,
    final_context
)

    await db_manager.save_chat_log(request.user_id, request.question, final_answer)
    
    return {
        "user_id": request.user_id, 
        "question": request.question, 
        "answer": final_answer,
        "hallucination_percent": hallucination_pct,
        "accuracy_score": accuracy_scr
    }

@app.delete("/delete_user_data")
async def delete_user_data(request: DeleteRequest):
    logs_deleted = await db_manager.delete_all_user_data(request.user_id)
    if logs_deleted is None:
        return {"status": "error", "message": f"User '{request.user_id}' doesn't exist."}

    try:
        collection.delete(where={"user_id": request.user_id})
        user_dir = os.path.join(UPLOAD_DIR, request.user_id)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        return {"status": "success", "message": f"All data for {request.user_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



import motor.motor_asyncio
from datetime import datetime
from typing import Optional
import uuid

# MongoDB Configuration
MONGO_DETAILS = "mongodb://localhost:27017/"

class MongoDBManager:
    def __init__(self):
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db = None
        self.chat_collection = None
        self.files_collection = None # New collection for file tracking

    def connect(self):
        """Initializes the MongoDB client."""
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
        self.db = self.client["rag_database"]
        self.chat_collection = self.db.get_collection("chat_history")
        self.files_collection = self.db.get_collection('user_files')
        print("✅ Connected to MongoDB.")

    def disconnect(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            print("❌ Disconnected from MongoDB.")
    
    async def save_file_metadata(self, user_id: str, filename: str):
        """Tracks the latest file indexed for the user"""    
        await self.files_collection.update_one(
            {"user_id": user_id},
            {"$set": {"last_file": filename, "timestamp": datetime.utcnow()}},
            upsert=True
        )
        
    async def get_user_context(self, user_id: str):
        """Returns metadata about the user's last upload"""
        return await self.files_collection.find_one({"user_id": user_id})    
        
    async def save_chat_log(self, user_id: str, question: str, answer: str):
        """Saves a Q&A interaction with a unique session ID."""
        document = {
            "log_id": str(uuid.uuid4()),
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow()
        }
        await self.chat_collection.insert_one(document)

    async def delete_all_user_data(self, user_id: str):
        """Checks for user existence and removes all data."""
        
        # FIX: Assign these to two different variables
        user_in_chats = await self.chat_collection.find_one({"user_id": user_id})
        user_in_files = await self.files_collection.find_one({'user_id': user_id})
        
        # Now both variables are defined for this check
        if not user_in_chats and not user_in_files:
            return None
        
        res_chat = await self.chat_collection.delete_many({"user_id": user_id})
        await self.files_collection.delete_one({"user_id": user_id})
        
        return res_chat.deleted_count

# Singleton instance
db_manager = MongoDBManager()