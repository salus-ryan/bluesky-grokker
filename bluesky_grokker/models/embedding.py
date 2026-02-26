"""Embedding model abstraction – supports OpenAI, Google, OpenRouter, and
Ollama providers.  Anthropic does not offer an embedding API, so when the
embedding provider is set to 'anthropic' we fall back to OpenAI (if a key
is available) or Ollama."""

from __future__ import annotations

import asyncio
import logging
from typing import List

import numpy as np

from config import (
    ANTHROPIC_API_KEY,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

log = logging.getLogger(__name__)


def _resolve_embed_provider() -> str:
    """Return the effective embedding provider."""
    provider = (EMBEDDING_PROVIDER or LLM_PROVIDER).lower()
    # Anthropic has no embedding API – pick the best available fallback
    if provider == "anthropic":
        if OPENAI_API_KEY:
            log.info("Anthropic has no embedding API; falling back to OpenAI for embeddings")
            return "openai"
        log.info("Anthropic has no embedding API; falling back to Ollama for embeddings")
        return "ollama"
    return provider


async def get_embedding(text: str) -> List[float]:
    """Return an embedding vector for *text*."""
    provider = _resolve_embed_provider()
    if provider == "openai":
        return await _openai_embed(text)
    if provider == "google":
        return await _google_embed(text)
    if provider == "openrouter":
        return await _openrouter_embed(text)
    return await _ollama_embed(text)


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of texts."""
    provider = _resolve_embed_provider()
    if provider == "openai":
        return await _openai_embed_batch(texts)
    if provider == "google":
        return await _google_embed_batch(texts)
    if provider == "openrouter":
        return await _openrouter_embed_batch(texts)
    # Ollama – no native batch endpoint
    return [await _ollama_embed(t) for t in texts]


# ── OpenAI ───────────────────────────────────────────────────────────────────

async def _openai_embed(text: str) -> List[float]:
    import openai

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


async def _openai_embed_batch(texts: List[str]) -> List[List[float]]:
    import openai

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


# ── Google (Generative AI) ───────────────────────────────────────────────────

async def _google_embed(text: str) -> List[float]:
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: genai.embed_content(
            model=EMBEDDING_MODEL or "models/text-embedding-004",
            content=text,
        ),
    )
    return _fit_vector(result["embedding"])


async def _google_embed_batch(texts: List[str]) -> List[List[float]]:
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: genai.embed_content(
            model=EMBEDDING_MODEL or "models/text-embedding-004",
            content=texts,
        ),
    )
    return [_fit_vector(v) for v in result["embedding"]]


# ── OpenRouter (OpenAI-compatible) ───────────────────────────────────────────

async def _openrouter_embed(text: str) -> List[float]:
    import openai

    client = openai.AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    resp = await client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return resp.data[0].embedding


async def _openrouter_embed_batch(texts: List[str]) -> List[List[float]]:
    import openai

    client = openai.AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    resp = await client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


# ── Ollama / local ───────────────────────────────────────────────────────────

async def _ollama_embed(text: str) -> List[float]:
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        vec = resp.json()["embedding"]
    return _fit_vector(vec)


# ── Utilities ────────────────────────────────────────────────────────────────

def _fit_vector(vec: List[float]) -> List[float]:
    """Pad or truncate *vec* to the configured EMBEDDING_DIM."""
    if len(vec) < EMBEDDING_DIM:
        vec = vec + [0.0] * (EMBEDDING_DIM - len(vec))
    return vec[:EMBEDDING_DIM]


def zero_vector() -> List[float]:
    """Return a zero vector of the configured dimension."""
    return [0.0] * EMBEDDING_DIM
