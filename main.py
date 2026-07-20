# import os
# import shutil
# import uvicorn
# import chromadb
# import PyPDF2
# import uuid
# import httpx
# import logging
# import json

# from typing import TypedDict, List
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from pydantic import BaseModel
# from contextlib import asynccontextmanager

# from sentence_transformers import SentenceTransformer, CrossEncoder
# from langgraph.graph import StateGraph, END
# from database import db_manager

# # ---------------- LOGGING ----------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ---------------- OLLAMA CONFIG ----------------
# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3.2:3b"

# async def call_ollama(prompt: str):
#     async with httpx.AsyncClient(timeout=300.0) as client:
#         try:
#             response = await client.post(
#                 OLLAMA_URL,
#                 json={
#                     "model": OLLAMA_MODEL,
#                     "prompt": prompt,
#                     "stream": False,
#                     "format": "json",
#                     "options": {"temperature": 0.1}
#                 }
#             )
#             response.raise_for_status()
#             return response.json().get("response", "").strip()
#         except Exception as e:
#             logger.error(f"OLLAMA ERROR: {repr(e)}")
#             return ""

# # ---------------- STATE ----------------
# class RAGState(TypedDict):
#     user_id: str
#     question: str
#     retrieved_docs: List[str]
#     reranked_context: str
#     answer: str
#     accuracy_score: float
#     hallucination_score: float
#     evaluation_summary: str   # ← NEW: carry summary through state
#     status: str

# models = {}

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("🚀 Starting System...")
#     db_manager.connect()
#     models["embed"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     models["reranker"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     yield
#     db_manager.disconnect()
#     models.clear()
#     logger.info("🛑 System Shutdown")

# app = FastAPI(title="RAG System with Internal LLM Evaluation", lifespan=lifespan)

# UPLOAD_DIR = "./uploaded_pdfs"
# CHROMA_PATH = "./chromadb"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(CHROMA_PATH, exist_ok=True)

# client = chromadb.PersistentClient(path=CHROMA_PATH)
# collection = client.get_or_create_collection(name="pdf_chunks")

# # ---------------- HELPERS ----------------
# def split_text(text, chunk_size=500, overlap=50):
#     words = text.split()
#     return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# # ---------------- SCORE LABEL HELPER ----------------
# def get_hallucination_label(score: float) -> str:
#     if score == 0:
#         return "✅ None"
#     elif score <= 30:
#         return "🟡 Low"
#     elif score <= 70:
#         return "🟠 Moderate"
#     elif score <= 99:
#         return "🔴 High"
#     else:
#         return "🚨 Complete"

# def get_accuracy_label(score: float) -> str:
#     if score >= 90:
#         return "✅ Excellent"
#     elif score >= 70:
#         return "🟡 Good"
#     elif score >= 40:
#         return "🟠 Partial"
#     elif score > 0:
#         return "🔴 Weak"
#     else:
#         return "🚨 None"

# # ---------------- SCHEMAS ----------------
# class QuestionRequest(BaseModel):
#     user_id: str
#     question: str

# class DeleteRequest(BaseModel):
#     user_id: str

# # ↓ UPDATED: richer response schema
# class AnswerResponse(BaseModel):
#     user_id: str
#     question: str
#     answer: str
#     scores: dict           # accuracy + hallucination scores + labels + summary

# # ---------------- RAG NODES ----------------

# async def retrieve_node(state: RAGState):
#     q_emb = models["embed"].encode(state["question"]).tolist()

#     results = collection.query(
#         query_embeddings=[q_emb],
#         n_results=10,
#         where={"user_id": state["user_id"]}
#     )

#     docs = results["documents"][0] if results["documents"] else []

#     if not docs:
#         return {"status": "no_docs", "retrieved_docs": []}

#     return {"retrieved_docs": docs, "status": "docs_found"}

# async def rerank_node(state: RAGState):
#     if state["status"] == "no_docs":
#         return {"reranked_context": ""}

#     pairs = [[state["question"], doc] for doc in state["retrieved_docs"]]
#     scores = models["reranker"].predict(pairs)

#     scored_docs = sorted(zip(state["retrieved_docs"], scores),
#                          key=lambda x: x[1], reverse=True)

#     top_docs = [doc for doc, _ in scored_docs[:2]]
#     return {"reranked_context": " ".join(top_docs)}

# async def generate_node(state: RAGState):

#     if state["status"] == "no_docs" or not state["reranked_context"]:
#         return {
#             "answer": "No relevant context found in your documents.",
#             "accuracy_score": 0.0,
#             "hallucination_score": 0.0,
#             "evaluation_summary": "No context available to evaluate."
#         }

#     context = state["reranked_context"]
#     question = state["question"]

#     # ----------- GENERATION PROMPT -----------
#     gen_prompt = f"""
# You are a precise AI assistant.

# Answer the question using ONLY the provided context.
# If the answer is not found in the context, respond exactly:
# "Not found in context"

# Context:
# {context}

# Question:
# {question}

# Return JSON:
# {{
#   "answer": "<answer>"
# }}
# """

#     raw_answer = await call_ollama(gen_prompt)
#     answer_text = "Not found in context"
#     try:
#         parsed = json.loads(raw_answer)
#         raw_parsed_answer = parsed.get("answer", "Not found in context")
#         if isinstance(raw_parsed_answer, list):
#             answer_text = ", ".join(str(item) for item in raw_parsed_answer)
#         elif isinstance(raw_parsed_answer, str):
#             answer_text = raw_parsed_answer
#         else:
#             answer_text = str(raw_parsed_answer)
#     except Exception:
#         answer_text = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)

#     # ----------- EVALUATION PROMPT -----------
#     eval_prompt = f"""
# You are an internal evaluation model for a Retrieval-Augmented Generation (RAG) system.

# Your ONLY job is to evaluate whether the Generated Answer is supported by the provided Context Chunks.

# STRICT RULES:
# - Do NOT use any outside or prior knowledge.
# - Do NOT assume or infer missing information.
# - Base your evaluation SOLELY on what is explicitly stated in the Context Chunks.

# =========================================================

# SCORING DEFINITIONS

# Total Accuracy Score (0-100):
#   100     -> Every factual claim in the answer is fully and explicitly supported by Context.
#   70-99   -> Mostly supported; very minor phrasing differences only.
#   40-69   -> Partially supported; some claims are missing or ambiguous in Context.
#   1-39    -> Weakly supported; most claims are not found in Context.
#   0       -> Completely unsupported; no claims match Context.

# Total Hallucination Score (0-100):
#   0       -> No hallucination. Every piece of information is grounded in Context.
#   1-30    -> Minor unsupported wording, but core facts are grounded.
#   31-70   -> Some unsupported or fabricated claims present.
#   71-99   -> Mostly invented or unsupported information.
#   100     -> Completely fabricated; nothing matches Context.

# IMPORTANT ENFORCEMENT RULES:
# - If ANY factual claim is not found in Context, Hallucination Score MUST be > 30.
# - If MOST claims are unsupported, Hallucination Score MUST be > 70.

# =========================================================

# EVALUATION PROCESS:

# Step 1 -> Break the Generated Answer into individual factual claims.
# Step 2 -> For each claim, check if it is explicitly present in the Context Chunks.
# Step 3 -> Count supported vs unsupported claims.
# Step 4 -> Assign scores strictly following definitions above.
# Step 5 -> Write a brief evaluation summary explaining your reasoning.

# =========================================================

# User Question:
# {question}

# Context Chunks:
# {context}

# Generated Answer:
# {answer_text}

# Return ONLY valid JSON (no extra text):
# {{
#   "total_accuracy_score": 0,
#   "total_hallucination_score": 0,
#   "evaluation_summary": "string"
# }}
# """

#     raw_eval = await call_ollama(eval_prompt)

#     acc = 0.0
#     hall = 100.0
#     summary = "Evaluation could not be parsed."

#     try:
#         parsed_eval = json.loads(raw_eval)
#         acc = float(parsed_eval.get("total_accuracy_score", 0))
#         hall = float(parsed_eval.get("total_hallucination_score", 100))
#         summary = parsed_eval.get("evaluation_summary", summary)
#     except:
#         logger.warning("⚠ Evaluation JSON parsing failed")

#     return {
#         "answer": answer_text,
#         "accuracy_score": acc,
#         "hallucination_score": hall,
#         "evaluation_summary": summary
#     }

# # ---------------- GRAPH ----------------
# def build_graph():
#     workflow = StateGraph(RAGState)
#     workflow.add_node("retrieve", retrieve_node)
#     workflow.add_node("rerank", rerank_node)
#     workflow.add_node("generate", generate_node)

#     workflow.set_entry_point("retrieve")
#     workflow.add_edge("retrieve", "rerank")
#     workflow.add_edge("rerank", "generate")
#     workflow.add_edge("generate", END)

#     return workflow.compile()

# rag_app = build_graph()

# # ---------------- ENDPOINTS ----------------

# @app.post("/upload")
# async def upload_pdf(user_id: str, file: UploadFile = File(...)):
#     user_dir = os.path.join(UPLOAD_DIR, user_id)
#     os.makedirs(user_dir, exist_ok=True)

#     file_path = os.path.join(user_dir, file.filename)

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     text = ""
#     with open(file_path, "rb") as f:
#         reader = PyPDF2.PdfReader(f)
#         for page in reader.pages:
#             extracted = page.extract_text()
#             if extracted:
#                 text += extracted + "\n"

#     chunks = split_text(text)

#     for i, chunk in enumerate(chunks):
#         chunk_id = f"{user_id}_{file.filename}_{i}_{uuid.uuid4().hex[:8]}"
#         emb = models["embed"].encode(chunk).tolist()
#         collection.add(
#             documents=[chunk],
#             metadatas=[{"source": file.filename, "user_id": user_id}],
#             ids=[chunk_id],
#             embeddings=[emb]
#         )

#     await db_manager.save_file_metadata(user_id, file.filename)

#     return {"message": f"Indexed {len(chunks)} chunks for {user_id}"}


# @app.post("/ask", response_model=AnswerResponse)
# async def ask_question(request: QuestionRequest):

#     inputs = {
#         "user_id": request.user_id,
#         "question": request.question,
#         "retrieved_docs": [],
#         "reranked_context": "",
#         "answer": "",
#         "accuracy_score": 0.0,
#         "hallucination_score": 0.0,
#         "evaluation_summary": "",
#         "status": "start"
#     }

#     result_state = await rag_app.ainvoke(inputs)

#     acc = result_state["accuracy_score"]
#     hall = result_state["hallucination_score"]

#     # Log scores clearly to server console
#     logger.info("=" * 50)
#     logger.info(f"📊 EVALUATION SCORES FOR USER: {request.user_id}")
#     logger.info(f"   Question       : {request.question}")
#     logger.info(f"   Accuracy Score : {acc}/100  → {get_accuracy_label(acc)}")
#     logger.info(f"   Hallucination  : {hall}/100 → {get_hallucination_label(hall)}")
#     logger.info(f"   Summary        : {result_state['evaluation_summary']}")
#     logger.info("=" * 50)

#     await db_manager.save_chat_log(
#         request.user_id,
#         request.question,
#         result_state["answer"]
#     )

#     return {
#         "user_id": request.user_id,
#         "question": request.question,
#         "answer": result_state["answer"],
#         # ↓ All score info grouped under one clean key
#         "scores": {
#             "total_accuracy_score": acc,
#             "accuracy_label": get_accuracy_label(acc),
#             "total_hallucination_score": hall,
#             "hallucination_label": get_hallucination_label(hall),
#             "evaluation_summary": result_state["evaluation_summary"]
#         }
#     }


# @app.delete("/delete_user_data")
# async def delete_user_data(request: DeleteRequest):
#     await db_manager.delete_all_user_data(request.user_id)
#     collection.delete(where={"user_id": request.user_id})
#     return {"status": "success"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)




import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, CrossEncoder

# ✅ Flat imports
from settings import EMBED_MODEL, RERANKER_MODEL
from rag_pipeline import build_rag_graph
from routes import router, init_router
from database import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RAG System...")
    db_manager.connect()

    models["embed"]    = SentenceTransformer(EMBED_MODEL)
    models["reranker"] = CrossEncoder(RERANKER_MODEL)

    rag_app = build_rag_graph(models)
    init_router(rag_app, models)

    logger.info("✅ Models loaded and graph compiled")
    yield

    db_manager.disconnect()
    models.clear()
    logger.info("🛑 System shutdown complete")


app = FastAPI(
    title="RAG System with Internal LLM Evaluation",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)