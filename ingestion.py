# ============================================================
#  ingestion.py  –  PDF parsing, chunking, indexing, deletion
# ============================================================

import os
import shutil
import uuid
import logging

import PyPDF2
import chromadb

from settings import UPLOAD_DIR, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = _chroma_client.get_or_create_collection(name="pdf_chunks")


# ── Folder variant resolver ───────────────────────────────────

def _get_variants(user_id: str) -> list[str]:
    """Return all possible user_id folder/chromadb variants."""
    uid = user_id.strip()
    variants = {uid}
    if uid.startswith("user_"):
        variants.add(uid[len("user_"):])
    else:
        variants.add(f"user_{uid}")
    seen, unique = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _user_pdf_dir(user_id: str) -> str:
    """
    Creates and returns:  <UPLOAD_DIR>/<user_id>/
    This is where all uploads for this user are stored.
    """
    path = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(path, exist_ok=True)
    logger.info(f"📁 User folder ready: {path}")
    return path


# ── Text helpers ─────────────────────────────────────────────

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]


def _extract_text(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# ── ChromaDB debug helpers ────────────────────────────────────

def get_all_chunks_for_user(user_id: str) -> list[dict]:
    """
    Returns all ChromaDB chunks for a user across all variants.
    Used for debugging / admin visibility.
    """
    all_chunks = []
    for variant in _get_variants(user_id):
        try:
            results = collection.get(
                where={"user_id": variant},
                include=["metadatas", "documents"],
            )
            for doc_id, meta, doc in zip(
                results["ids"], results["metadatas"], results["documents"]
            ):
                all_chunks.append({
                    "id":       doc_id,
                    "user_id":  meta.get("user_id"),
                    "source":   meta.get("source"),
                    "variant":  variant,
                    "preview":  doc[:80] + "..." if len(doc) > 80 else doc,
                })
        except Exception as e:
            logger.warning(f"get_all_chunks_for_user warning [{variant}]: {e}")
    return all_chunks


def get_chromadb_stats(user_id: str) -> dict:
    """Summary stats of what's stored in ChromaDB for a user."""
    chunks = get_all_chunks_for_user(user_id)
    files  = {}
    for c in chunks:
        src = c.get("source", "unknown")
        files[src] = files.get(src, 0) + 1
    return {
        "user_id":      user_id,
        "total_chunks": len(chunks),
        "files":        files,   # {filename: chunk_count}
    }


# ── Duplicate prevention ──────────────────────────────────────

def _file_already_indexed(user_id: str, filename: str) -> bool:
    """
    Check if a file is already indexed in ChromaDB to prevent duplicates.
    Checks all user_id variants.
    """
    for variant in _get_variants(user_id):
        try:
            results = collection.get(where={
                "$and": [
                    {"user_id": {"$eq": variant}},
                    {"source":  {"$eq": filename}},
                ]
            })
            if results["ids"]:
                logger.info(f"⚠️  Already indexed: {filename} under variant={variant} ({len(results['ids'])} chunks)")
                return True
        except Exception as e:
            logger.warning(f"Duplicate check warning [{variant}]: {e}")
    return False


def _delete_existing_chunks(user_id: str, filename: str):
    """Delete all existing ChromaDB chunks for a file before re-indexing."""
    for variant in _get_variants(user_id):
        try:
            results = collection.get(where={
                "$and": [
                    {"user_id": {"$eq": variant}},
                    {"source":  {"$eq": filename}},
                ]
            })
            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"🗑️  Pre-cleared {len(results['ids'])} chunks for {filename} [variant={variant}]")
        except Exception as e:
            logger.warning(f"Pre-clear warning [{variant}]: {e}")


# ── Public API ───────────────────────────────────────────────

def save_and_index_pdf(user_id: str, filename: str, file_obj, embed_model) -> int:
    """
    1. Creates uploaded_pdfs/<user_id>/ folder automatically
    2. Saves PDF to disk
    3. Clears any existing chunks (prevents duplicates)
    4. Extracts text, chunks, embeds, stores in ChromaDB
    Returns chunk count.
    """
    # ── Create user folder on disk ────────────────────────────
    user_dir  = _user_pdf_dir(user_id)          # uploaded_pdfs/<user_id>/
    file_path = os.path.join(user_dir, filename)

    # ── Save file ─────────────────────────────────────────────
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file_obj, buf)
    logger.info(f"💾 Saved PDF: {file_path}")

    # ── Prevent duplicates: clear old chunks if re-uploading ──
    _delete_existing_chunks(user_id, filename)

    # ── Extract, chunk, embed ─────────────────────────────────
    text   = _extract_text(file_path)
    chunks = split_text(text)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{user_id}_{filename}_{i}_{uuid.uuid4().hex[:8]}"
        emb = embed_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            metadatas=[{"source": filename, "user_id": user_id}],
            ids=[chunk_id],
            embeddings=[emb],
        )

    # ── Log what's now in ChromaDB for this user ──────────────
    stats = get_chromadb_stats(user_id)
    logger.info(f"✅ Indexed {len(chunks)} chunks | user={user_id} | file={filename}")
    logger.info(f"   ChromaDB total: {stats['total_chunks']} chunks across {list(stats['files'].keys())}")
    return len(chunks)


def list_user_pdfs(user_id: str) -> list[str]:
    """
    List all PDFs for a user — checks ALL folder variants.
    """
    found = []
    for variant in _get_variants(user_id):
        folder = os.path.join(UPLOAD_DIR, variant)
        if os.path.isdir(folder):
            pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
            found.extend(pdfs)
            if pdfs:
                logger.info(f"  Found {len(pdfs)} PDFs in {folder}")
    return list(dict.fromkeys(found))


def delete_pdf(user_id: str, filename: str) -> bool:
    """
    Full delete of a single PDF:
    ✅ Disk (all variant folders)
    ✅ ChromaDB chunks (all variants)
    ✅ Orphan cleanup
    """
    logger.info(f"🗑️  delete_pdf START — user={user_id}, file={filename}")
    variants = _get_variants(user_id)

    # ── Step 1: Delete from disk ──────────────────────────────
    deleted_from_disk = False
    for variant in variants:
        file_path = os.path.join(UPLOAD_DIR, variant, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            logger.info(f"  ✅ Disk: deleted {file_path}")
            deleted_from_disk = True
            # Remove empty folder
            folder = os.path.dirname(file_path)
            if os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
                logger.info(f"  🗂️  Removed empty folder: {folder}")
            break

    if not deleted_from_disk:
        logger.warning(f"  ❌ File not found on disk for user={user_id}")
        return False

    # ── Step 2: Delete ChromaDB chunks (all variants) ─────────
    total_deleted = 0
    for variant in variants:
        try:
            results = collection.get(where={
                "$and": [
                    {"user_id": {"$eq": variant}},
                    {"source":  {"$eq": filename}},
                ]
            })
            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"  ✅ ChromaDB: removed {len(results['ids'])} chunks [variant={variant}]")
                total_deleted += len(results["ids"])
        except Exception as e:
            logger.warning(f"  ChromaDB delete warning [{variant}]: {e}")

    # ── Step 3: Nuke orphans ──────────────────────────────────
    orphans = _nuke_orphans()

    logger.info(f"✅ delete_pdf DONE | file={filename} | chunks={total_deleted} | orphans={orphans}")
    return True


def delete_all_user_data(user_id: str):
    """
    Full wipe:
    ✅ All disk folders (all variants)
    ✅ All ChromaDB chunks (all variants)
    ✅ Orphan cleanup
    """
    logger.info(f"🗑️  delete_all_user_data START — user={user_id}")
    variants = _get_variants(user_id)

    # ── Step 1: Remove all disk folders ──────────────────────
    for variant in variants:
        folder = os.path.join(UPLOAD_DIR, variant)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            logger.info(f"  ✅ Disk: removed folder {folder}")

    # ── Step 2: Remove ChromaDB chunks (all variants) ─────────
    for variant in variants:
        try:
            results = collection.get(where={"user_id": {"$eq": variant}})
            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"  ✅ ChromaDB: removed {len(results['ids'])} chunks [variant={variant}]")
        except Exception as e:
            logger.warning(f"  ChromaDB delete warning [{variant}]: {e}")

    # ── Step 3: Nuke orphans ──────────────────────────────────
    orphans = _nuke_orphans()

    logger.info(f"✅ delete_all_user_data DONE | user={user_id} | orphans={orphans}")


def _nuke_orphans() -> int:
    """
    Scan ALL ChromaDB entries and remove any whose source file
    no longer exists on disk. Catches any leftover embeddings.
    Returns count of deleted orphan chunks.
    """
    try:
        all_results = collection.get(include=["metadatas"])
        orphan_ids  = []

        for doc_id, meta in zip(all_results["ids"], all_results["metadatas"]):
            uid      = meta.get("user_id", "")
            filename = meta.get("source",  "")
            file_exists = any(
                os.path.isfile(os.path.join(UPLOAD_DIR, v, filename))
                for v in _get_variants(uid)
            )
            if not file_exists:
                orphan_ids.append(doc_id)

        if orphan_ids:
            collection.delete(ids=orphan_ids)
            logger.info(f"🧹 Nuked {len(orphan_ids)} orphan ChromaDB chunks")
        else:
            logger.info("🧹 ChromaDB clean — no orphan chunks found")

        return len(orphan_ids)
    except Exception as e:
        logger.warning(f"Orphan nuke warning: {e}")
        return 0
