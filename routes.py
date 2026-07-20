# ============================================================
#  routes.py  –  All FastAPI endpoint definitions
# ============================================================

import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from schemas import (
    DeleteRequest, DeletePDFRequest,
    AnswerResponse, PDFListResponse, MessageResponse, StatusResponse,
)
from ingestion import save_and_index_pdf, list_user_pdfs, delete_pdf, delete_all_user_data, get_chromadb_stats, _nuke_orphans
from scoring import get_accuracy_label, get_hallucination_label
from database import db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

_rag_app    = None
_models_ref = None


def init_router(rag_app, models: dict):
    global _rag_app, _models_ref
    _rag_app    = rag_app
    _models_ref = models


# ── Request schemas ───────────────────────────────────────────

class CreateUserRequest(BaseModel):
    display_name: str

class LoginRequest(BaseModel):
    user_id: str

class QuestionRequest(BaseModel):
    user_id:      str
    question:     str
    selected_pdf: Optional[str] = None   # None = all docs, filename = specific PDF


# ── User Management ───────────────────────────────────────────

@router.get("/users")
async def list_users():
    users = await db_manager.list_all_users()
    return {"users": users}


@router.post("/users/create")
async def create_user(request: CreateUserRequest):
    if not request.display_name.strip():
        raise HTTPException(status_code=400, detail="Display name cannot be empty")
    user = await db_manager.create_user(request.display_name)
    return {"status": "created", "user": user}


@router.post("/users/login")
async def login_user(request: LoginRequest):
    user = await db_manager.login_user(request.user_id.strip())
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{request.user_id}' does not exist")
    return {"status": "logged_in", "user": user}


@router.get("/users/{user_id}")
async def get_user(user_id: str):
    user = await db_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' does not exist")
    return user


@router.delete("/users/{user_id}/delete_all", response_model=StatusResponse)
async def admin_delete_user(user_id: str):
    exists = await db_manager.user_exists(user_id)
    if not exists:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' does not exist")
    delete_all_user_data(user_id)
    await db_manager.delete_all_user_data(user_id)
    return {"status": f"All data for '{user_id}' permanently deleted"}


# ── PDF Management ────────────────────────────────────────────

@router.post("/upload", response_model=MessageResponse)
async def upload_pdf(user_id: str, file: UploadFile = File(...)):
    # ── Validate PDF only ─────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.filename}'. Only PDF files (.pdf) are accepted. JPG, PNG and other formats are not supported."
        )
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Only PDF files are accepted."
        )

    exists = await db_manager.user_exists(user_id)
    if not exists:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found. Please register first.")

    chunks = save_and_index_pdf(
        user_id=user_id,
        filename=file.filename,
        file_obj=file.file,
        embed_model=_models_ref["embed"],
    )
    await db_manager.save_file_metadata(user_id, file.filename)
    return {"message": f"Indexed {chunks} chunks from '{file.filename}'"}


@router.get("/pdfs/{user_id}", response_model=PDFListResponse)
async def list_pdfs(user_id: str):
    files = list_user_pdfs(user_id)
    return {"user_id": user_id, "files": files}


@router.delete("/pdfs/delete_one", response_model=StatusResponse)
async def delete_one_pdf(request: DeletePDFRequest):
    success = delete_pdf(request.user_id, request.filename)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    await db_manager.delete_file_metadata(request.user_id, request.filename)
    return {"status": f"'{request.filename}' deleted from disk, ChromaDB and MongoDB"}


# ── Chat History ──────────────────────────────────────────────

@router.get("/chat_history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    logs = await db_manager.get_chat_history(user_id, limit=limit)
    return {"user_id": user_id, "history": logs}


@router.delete("/chat_history/delete", response_model=StatusResponse)
async def delete_chat_history(request: DeleteRequest):
    count = await db_manager.delete_chat_history(request.user_id)
    return {"status": f"Deleted {count} chat log(s) for '{request.user_id}'"}


# ── Delete Everything ─────────────────────────────────────────

@router.delete("/delete_user_data", response_model=StatusResponse)
async def delete_all_user_data_endpoint(request: DeleteRequest):
    delete_all_user_data(request.user_id)
    await db_manager.delete_all_user_data(request.user_id)
    return {"status": f"All data deleted for '{request.user_id}'"}


# ── Debug / Admin ─────────────────────────────────────────────

@router.get("/debug/chromadb/{user_id}")
async def debug_chromadb(user_id: str):
    """
    Show what is currently stored in ChromaDB for a user.
    Returns chunk counts per file — useful to verify deletions worked.
    """
    stats = get_chromadb_stats(user_id)
    return {
        "user_id":      user_id,
        "total_chunks": stats["total_chunks"],
        "files":        stats["files"],   # {filename: chunk_count}
        "message":      "These are the embeddings currently stored in ChromaDB for this user."
    }


@router.delete("/debug/chromadb/nuke_orphans", response_model=StatusResponse)
async def nuke_orphan_chunks():
    """
    Admin: scan entire ChromaDB and delete any chunks whose
    source file no longer exists on disk.
    """
    count = _nuke_orphans()
    return {"status": f"Deleted {count} orphan chunk(s) from ChromaDB"}


# ── Q&A ───────────────────────────────────────────────────────────

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # ── Guard: user must exist ────────────────────────────────
    exists = await db_manager.user_exists(request.user_id)
    if not exists:
        raise HTTPException(status_code=404, detail=f"User '{request.user_id}' not found.")

    # ── Guard: user must have at least one PDF ────────────────
    user_pdfs = list_user_pdfs(request.user_id)
    if not user_pdfs:
        raise HTTPException(
            status_code=400,
            detail="No PDFs uploaded yet. Please upload a PDF document first before asking questions."
        )

    # ── Guard: selected PDF must exist on disk ───────────────
    if request.selected_pdf and request.selected_pdf not in user_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{request.selected_pdf}' not found. Please select a valid document."
        )

    # ── Guard: selected PDF must have chunks in ChromaDB ─────
    # Catches PDFs that exist on disk but failed to index —
    # returns immediately without hitting Ollama (prevents timeout).
    if request.selected_pdf:
        from ingestion import collection as chroma_collection
        try:
            check = chroma_collection.get(
                where={
                    "$and": [
                        {"user_id": {"$eq": request.user_id}},
                        {"source":  {"$eq": request.selected_pdf}},
                    ]
                },
                include=["metadatas"],
            )
            if not check["ids"]:
                return {
                    "user_id":     request.user_id,
                    "question":    request.question,
                    "answer":      f"No indexed content found for '{request.selected_pdf}'. The PDF may not have been indexed correctly. Try re-uploading it.",
                    "queried_pdf": request.selected_pdf,
                    "scores": {
                        "total_accuracy_score":      0.0,
                        "accuracy_label":            "🚨 None",
                        "total_hallucination_score": 0.0,
                        "hallucination_label":       "✅ Minimal",
                        "evaluation_summary":        "No content available to evaluate.",
                    },
                }
        except Exception as e:
            logger.warning(f"ChromaDB pre-check error: {e}")

    queried_pdf = request.selected_pdf or "All Documents"
    logger.info(f"📄 Query target: {queried_pdf} | user={request.user_id}")

    inputs = {
        "user_id":             request.user_id,
        "question":            request.question,
        "selected_pdf":        request.selected_pdf,   # passed into RAG graph
        "retrieved_docs":      [],
        "reranked_context":    "",
        "answer":              "",
        "accuracy_score":      0.0,
        "hallucination_score": 0.0,
        "evaluation_summary":  "",
        "status":              "start",
    }

    result = await _rag_app.ainvoke(inputs)
    acc  = result["accuracy_score"]
    hall = result["hallucination_score"]

    logger.info("=" * 52)
    logger.info(f"📊 EVALUATION  user={request.user_id} | pdf={queried_pdf}")
    logger.info(f"   Accuracy   : {acc}/100  → {get_accuracy_label(acc)}")
    logger.info(f"   Hallucinate: {hall}/100 → {get_hallucination_label(hall)}")
    logger.info("=" * 52)

    await db_manager.save_chat_log(request.user_id, request.question, result["answer"])

    return {
        "user_id":     request.user_id,
        "question":    request.question,
        "answer":      result["answer"],
        "queried_pdf": queried_pdf,
        "scores": {
            "total_accuracy_score":      acc,
            "accuracy_label":            get_accuracy_label(acc),
            "total_hallucination_score": hall,
            "hallucination_label":       get_hallucination_label(hall),
            "evaluation_summary":        result["evaluation_summary"],
        },
    }
