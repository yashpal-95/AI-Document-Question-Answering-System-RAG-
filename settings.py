# ============================================================
#  config/settings.py  –  Central configuration
# ============================================================

import os

# ── Ollama ──────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# ── Paths ───────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploaded_pdfs")
CHROMA_PATH = os.path.join(BASE_DIR, "chromadb")

os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ── Embedding / Reranker models ──────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval ────────────────────────────────────────────────
TOP_K_RETRIEVE = 10
TOP_K_RERANK   = 2
