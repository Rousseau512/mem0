import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.openai import OpenAILLM
from mem0.memory.utils import extract_json
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)


class MemoryCategories(BaseModel):
    categories: List[str]


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _has_local_codex_auth() -> bool:
    return (
        Path("~/.pi/agent/auth.json").expanduser().exists()
        or Path("~/.codex/auth.json").expanduser().exists()
    )


@lru_cache(maxsize=1)
def _fallback_openai_llm() -> OpenAILLM:
    """Create a lazy fallback LLM for categorization.

    OpenMemory used to instantiate the raw OpenAI client at import time, which
    bypassed Mem0's Codex OAuth support and crashed when only OAuth credentials
    were configured. Keep this as a fallback for direct categorization calls;
    normal OpenMemory requests use the configured Mem0 memory client's LLM.
    """
    use_codex_oauth = (
        _env_truthy("OPENAI_USE_CODEX_OAUTH")
        or _env_truthy("CODEX_OAUTH")
        or (not os.getenv("OPENAI_API_KEY") and _has_local_codex_auth())
    )
    model = os.getenv("OPENMEMORY_CATEGORIZATION_MODEL") or os.getenv("LLM_MODEL")
    if not model:
        model = "gpt-5.5" if use_codex_oauth else "gpt-4o-mini"

    return OpenAILLM(
        OpenAIConfig(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            use_codex_oauth=use_codex_oauth or None,
            codex_auth_file=os.getenv("OPENAI_CODEX_AUTH_FILE") or os.getenv("CODEX_AUTH_FILE"),
        )
    )


def _get_categorization_llm():
    """Return the configured Mem0 LLM so categorization follows Mem0 auth.

    Importing get_memory_client at module import time would create a circular
    import with app.models. Import lazily inside the function instead.
    """
    try:
        from app.utils.memory import get_memory_client

        memory_client = get_memory_client()
        if memory_client and getattr(memory_client, "llm", None):
            return memory_client.llm
    except Exception as exc:
        logger.warning("Falling back to Codex OAuth categorization LLM: %s", exc)

    return _fallback_openai_llm()


def _parse_categories_response(response: Any) -> MemoryCategories:
    if isinstance(response, MemoryCategories):
        return response
    if isinstance(response, dict):
        payload = response
    else:
        payload = json.loads(extract_json(str(response)))
    return MemoryCategories.model_validate(payload)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory},
        ]

        llm = _get_categorization_llm()
        response = llm.generate_response(
            messages=messages,
            response_format={"type": "json_object"},
        )
        parsed = _parse_categories_response(response)
        return [cat.strip().lower() for cat in parsed.categories if cat.strip()]

    except Exception as e:
        logger.error("[ERROR] Failed to get categories: %s", e)
        raise
