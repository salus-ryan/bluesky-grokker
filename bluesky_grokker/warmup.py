"""Warm-up module – verifies LLM and embedding provider connectivity on startup."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    WARMUP_ENABLED,
)

log = logging.getLogger(__name__)

WARMUP_PROMPT = [
    {"role": "system", "content": "You are a health-check assistant. Reply with exactly: OK"},
    {"role": "user", "content": "ping"},
]


async def warmup_llm() -> Dict[str, Any]:
    """Send a minimal completion request to verify the LLM provider is reachable."""
    from agent import llm_complete

    log.info("Warming up LLM provider=%s model=%s …", LLM_PROVIDER, LLM_MODEL)
    t0 = time.monotonic()
    try:
        response = await llm_complete(WARMUP_PROMPT, max_tokens=8)
        latency = time.monotonic() - t0
        log.info("LLM warm-up OK  latency=%.2fs  response=%r", latency, response[:60])
        return {"provider": LLM_PROVIDER, "model": LLM_MODEL, "ok": True, "latency_s": round(latency, 3), "response": response[:60]}
    except Exception as exc:
        latency = time.monotonic() - t0
        log.error("LLM warm-up FAILED  provider=%s  error=%s  (%.2fs)", LLM_PROVIDER, exc, latency)
        return {"provider": LLM_PROVIDER, "model": LLM_MODEL, "ok": False, "latency_s": round(latency, 3), "error": str(exc)}


async def warmup_embedding() -> Dict[str, Any]:
    """Send a minimal embedding request to verify the embedding provider works."""
    from models.embedding import get_embedding, _resolve_embed_provider

    provider = _resolve_embed_provider()
    log.info("Warming up embedding provider=%s …", provider)
    t0 = time.monotonic()
    try:
        vec = await get_embedding("warm-up test")
        latency = time.monotonic() - t0
        dim = len(vec)
        log.info("Embedding warm-up OK  provider=%s  dim=%d  latency=%.2fs", provider, dim, latency)
        return {"provider": provider, "ok": True, "dim": dim, "latency_s": round(latency, 3)}
    except Exception as exc:
        latency = time.monotonic() - t0
        log.error("Embedding warm-up FAILED  provider=%s  error=%s  (%.2fs)", provider, exc, latency)
        return {"provider": provider, "ok": False, "latency_s": round(latency, 3), "error": str(exc)}


async def run_warmup() -> Dict[str, Any]:
    """Run all warm-up checks. Returns a summary dict."""
    if not WARMUP_ENABLED:
        log.info("Warm-up disabled (WARMUP_ENABLED=false)")
        return {"skipped": True}

    log.info("═══ Running warm-up checks ═══")
    t0 = time.monotonic()

    llm_result, embed_result = await asyncio.gather(
        warmup_llm(),
        warmup_embedding(),
        return_exceptions=True,
    )

    # Handle cases where gather returned an exception object
    if isinstance(llm_result, BaseException):
        llm_result = {"ok": False, "error": str(llm_result)}
    if isinstance(embed_result, BaseException):
        embed_result = {"ok": False, "error": str(embed_result)}

    total = time.monotonic() - t0
    summary = {
        "llm": llm_result,
        "embedding": embed_result,
        "all_ok": llm_result.get("ok", False) and embed_result.get("ok", False),
        "total_s": round(total, 3),
    }

    if summary["all_ok"]:
        log.info("═══ Warm-up complete  ALL OK  (%.2fs) ═══", total)
    else:
        log.warning("═══ Warm-up complete  SOME FAILURES  (%.2fs) ═══", total)
        if not llm_result.get("ok"):
            log.warning("  LLM:       %s", llm_result.get("error", "unknown"))
        if not embed_result.get("ok"):
            log.warning("  Embedding: %s", embed_result.get("error", "unknown"))

    return summary
