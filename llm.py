# ============================================================
#  services/llm.py  –  Ollama LLM calls
# ============================================================

import httpx
import logging
from settings import OLLAMA_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


async def call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the raw response string."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model":   OLLAMA_MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "format":  "json",
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"OLLAMA ERROR: {repr(e)}")
            return ""
