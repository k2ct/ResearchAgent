"""
Unified LLM client for the ResearchAgent LLM Enhancement Layer.

Provides a single source of truth for LLM configuration and invocation.
All enhancers use this client; individual modules no longer read env vars directly.

Design:
- ``is_llm_enhancement_enabled()`` gates ALL LLM features.
- ``get_chat_llm()`` returns a ChatOpenAI or None.
- ``invoke_llm_with_fallback()`` never raises — it always returns a result dict.
- API keys are NEVER printed.
"""

import os
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

# Load .env at import time so callers don't need to do it themselves.
load_dotenv()


# ── Configuration Gating ──────────────────────────────────────────────


def is_llm_enhancement_enabled() -> bool:
    """
    Check whether the LLM Enhancement Layer is globally enabled.

    Reads ``ENABLE_LLM_ENHANCEMENT`` from environment.
    Accepts: true / 1 / yes / y (case-insensitive).

    Default is ``false`` — all modules fall back to rule-based output.
    """
    value = os.getenv("ENABLE_LLM_ENHANCEMENT", "false").strip().lower()
    return value in ("1", "true", "yes", "y")


def get_llm_config() -> Dict[str, Any]:
    """
    Read LLM configuration from environment variables.

    Returns a dict with keys:
    - api_key: str (mask in logs — DO NOT print)
    - base_url: str
    - model: str
    - temperature: float
    - max_input_chars: int
    - enabled: bool
    """
    return {
        "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "base_url": os.getenv("OPENAI_BASE_URL", "").strip() or None,
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        "max_input_chars": int(os.getenv("LLM_MAX_INPUT_CHARS", "12000")),
        "enabled": is_llm_enhancement_enabled(),
    }


def get_chat_llm() -> Optional[ChatOpenAI]:
    """
    Create a ChatOpenAI instance from environment config.

    Returns None if:
    - ``ENABLE_LLM_ENHANCEMENT`` is false
    - ``OPENAI_API_KEY`` is missing or empty
    """
    if not is_llm_enhancement_enabled():
        return None

    config = get_llm_config()
    api_key = config["api_key"]

    if not api_key:
        return None

    kwargs: Dict[str, Any] = {
        "model": config["model"],
        "api_key": api_key,
        "temperature": config["temperature"],
    }

    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]

    return ChatOpenAI(**kwargs)


# ── Text Utilities ────────────────────────────────────────────────────


def safe_truncate_text(text: str, max_chars: Optional[int] = None) -> str:
    """
    Truncate text to *max_chars*, appending a truncation notice.

    If *max_chars* is None, reads ``LLM_MAX_INPUT_CHARS`` from env (default 12000).
    """
    if max_chars is None:
        max_chars = int(os.getenv("LLM_MAX_INPUT_CHARS", "12000"))

    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n[... {len(text) - max_chars} characters truncated ...]\n\n"
        + text[-half:]
    )


# ── Core Invocation ───────────────────────────────────────────────────


def invoke_llm_with_fallback(
    messages: List[BaseMessage],
    fallback_text: str,
    feature_name: str,
) -> Dict[str, Any]:
    """
    Invoke the LLM and return a standardised result dict.

    **This function NEVER raises.** If anything goes wrong (LLM disabled,
    no API key, network error, timeout), it returns the *fallback_text*
    with ``used_llm=False``.

    Args:
        messages: List of LangChain messages (SystemMessage, HumanMessage).
        fallback_text: Text to return when LLM is unavailable.
        feature_name: Human-readable label for error messages / logging.

    Returns::

        {
            "ok": bool,
            "used_llm": bool,
            "text": str,
            "error": str,
            "feature_name": str,
        }
    """
    result: Dict[str, Any] = {
        "ok": False,
        "used_llm": False,
        "text": fallback_text,
        "error": "",
        "feature_name": feature_name,
    }

    # Gate 1: global switch
    if not is_llm_enhancement_enabled():
        result["error"] = "LLM enhancement is disabled (ENABLE_LLM_ENHANCEMENT=false)."
        return result

    # Gate 2: LLM instance
    llm = get_chat_llm()
    if llm is None:
        result["error"] = "OPENAI_API_KEY is not configured."
        return result

    # Gate 3: invoke
    try:
        # Truncate message content to stay within context window
        truncated_messages: List[BaseMessage] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                content = safe_truncate_text(content)
            # Rebuild message (preserving type)
            truncated_messages.append(msg.__class__(content=content))

        response = llm.invoke(truncated_messages)
        result["ok"] = True
        result["used_llm"] = True
        result["text"] = str(response.content) if response.content else fallback_text
        result["error"] = ""

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        # On any failure, return fallback (already set)

    return result
