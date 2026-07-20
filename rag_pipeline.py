# ============================================================
#  rag_pipeline.py  –  LangGraph RAG nodes + graph
# ============================================================

import json
import logging
from typing import TypedDict, List, Optional
from functools import partial

from langgraph.graph import StateGraph, END
from settings import TOP_K_RETRIEVE, TOP_K_RERANK
from llm import call_ollama
from ingestion import collection

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────

class RAGState(TypedDict):
    user_id:             str
    question:            str
    selected_pdf:        Optional[str]   # None = all docs, filename = one PDF
    retrieved_docs:      List[str]
    reranked_context:    str
    answer:              str
    accuracy_score:      float
    hallucination_score: float
    evaluation_summary:  str
    status:              str


# ── Nodes ────────────────────────────────────────────────────

async def retrieve_node(state: RAGState, *, models: dict) -> dict:
    q_emb        = models["embed"].encode(state["question"]).tolist()
    user_id      = state["user_id"]
    selected_pdf = state.get("selected_pdf")   # None if not set

    # ── Step 1: Check if selected PDF has ANY chunks indexed ──
    # Do this BEFORE the vector query to catch empty PDFs immediately
    # and avoid sending anything to Ollama (which causes timeout).
    if selected_pdf:
        try:
            check = collection.get(
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"source":  {"$eq": selected_pdf}},
                    ]
                },
                include=["metadatas"],
            )
            if not check["ids"]:
                logger.warning(f"⚠️  No chunks found for PDF='{selected_pdf}' user={user_id} — returning no_docs immediately")
                return {"status": "no_docs", "retrieved_docs": []}
        except Exception as e:
            logger.warning(f"Pre-check warning: {e}")
            return {"status": "no_docs", "retrieved_docs": []}

    # ── Step 2: Build ChromaDB filter ────────────────────────
    if selected_pdf:
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"source":  {"$eq": selected_pdf}},
            ]
        }
        logger.info(f"🔍 Retrieving from PDF: {selected_pdf}")
    else:
        where_filter = {"user_id": {"$eq": user_id}}
        logger.info(f"🔍 Retrieving from all docs — user={user_id}")

    # ── Step 3: Vector similarity search ─────────────────────
    try:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K_RETRIEVE,
            where=where_filter,
        )
        docs = results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.warning(f"ChromaDB query error: {e}")
        docs = []

    if not docs:
        return {"status": "no_docs", "retrieved_docs": []}
    return {"retrieved_docs": docs, "status": "docs_found"}


async def rerank_node(state: RAGState, *, models: dict) -> dict:
    if state["status"] == "no_docs":
        return {"reranked_context": ""}

    pairs  = [[state["question"], doc] for doc in state["retrieved_docs"]]
    scores = models["reranker"].predict(pairs)

    sorted_docs = sorted(
        zip(state["retrieved_docs"], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    top_docs = [doc for doc, _ in sorted_docs[:TOP_K_RERANK]]
    return {"reranked_context": " ".join(top_docs)}


async def generate_node(state: RAGState) -> dict:
    if state["status"] == "no_docs" or not state["reranked_context"]:
        selected = state.get("selected_pdf")
        msg = (
            f"No relevant content found in '{selected}'. "
            "Try selecting a different PDF or use 'All Documents'."
            if selected
            else "No relevant context found. Please upload a PDF and try again."
        )
        return {
            "answer":              msg,
            "accuracy_score":      0.0,
            "hallucination_score": 0.0,
            "evaluation_summary":  "No context available to evaluate.",
        }

    context  = state["reranked_context"]
    question = state["question"]

    # ── Generation ───────────────────────────────────────────
    gen_prompt = f"""
You are a precise AI assistant.

Answer the question using ONLY the provided context.
If the answer is not found in the context, respond exactly:
"Not found in context"

Context:
{context}

Question:
{question}

Return JSON:
{{
  "answer": "<your answer as a plain string>"
}}
"""
    raw_answer  = await call_ollama(gen_prompt)
    answer_text = _parse_answer(raw_answer)

    # ── Evaluation ───────────────────────────────────────────
    eval_prompt    = _build_eval_prompt(question, context, answer_text)
    raw_eval       = await call_ollama(eval_prompt)
    acc, hall, summary = _parse_eval(raw_eval)

    return {
        "answer":              answer_text,
        "accuracy_score":      acc,
        "hallucination_score": hall,
        "evaluation_summary":  summary,
    }


# ── Prompt builders ───────────────────────────────────────────

def _build_eval_prompt(question: str, context: str, answer_text: str) -> str:
    return f"""
You are an expert evaluator. Read the Question, Context, and Generated Answer carefully.

Your job:
1. Decide what percentage of the Generated Answer is accurately supported by the Context.
2. Decide what percentage of the Generated Answer contains information NOT found in the Context (hallucination).
3. Write one sentence explaining your reasoning.

Be precise. Do not round to 0 or 100 unless truly warranted.
Think carefully before assigning scores.

Question: {question}

Context: {context}

Generated Answer: {answer_text}

Return ONLY this JSON with your exact assessed percentages:
{{
  "total_accuracy_score": <integer 0-100>,
  "total_hallucination_score": <integer 0-100>,
  "evaluation_summary": "<one sentence explanation>"
}}
"""


# ── Parsers ───────────────────────────────────────────────────

def _parse_answer(raw: str) -> str:
    try:
        parsed = json.loads(raw)
        value  = parsed.get("answer", "Not found in context")
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)
    except Exception:
        return raw.strip() if isinstance(raw, str) else str(raw)


def _parse_eval(raw: str) -> tuple[float, float, str]:
    default_summary = "Evaluation could not be parsed."
    try:
        parsed  = json.loads(raw)
        acc     = float(parsed.get("total_accuracy_score",      0))
        hall    = float(parsed.get("total_hallucination_score", 100))
        summary = parsed.get("evaluation_summary", default_summary)
        return acc, hall, summary
    except Exception:
        logger.warning("⚠ Evaluation JSON parsing failed")
        return 0.0, 100.0, default_summary


# ── Graph factory ─────────────────────────────────────────────

def build_rag_graph(models: dict):
    """
    Compile and return the LangGraph RAG pipeline.
    models dict must contain 'embed' and 'reranker' keys.
    """
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", partial(retrieve_node, models=models))
    workflow.add_node("rerank",   partial(rerank_node,   models=models))
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank",   "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
