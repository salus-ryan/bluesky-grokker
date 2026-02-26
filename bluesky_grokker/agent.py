"""Agent Layer – autonomous Bluesky agent that monitors discourse, reasons
about it via an LLM, and can post / reply on the network."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from bluesky_client import BlueskyClient
from config import (
    AGENT_ENABLED,
    AGENT_LOOP_INTERVAL,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SWARM_ENABLED,
)
import query as query_engine
import storage

log = logging.getLogger(__name__)


# ── LLM abstraction ─────────────────────────────────────────────────────────

async def llm_complete(
    messages: List[Dict[str, str]],
    model: str = LLM_MODEL,
    max_tokens: int = 512,
) -> str:
    """Send a chat-completion request to the configured LLM provider."""
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        return await _openai_complete(messages, model, max_tokens)
    if provider == "anthropic":
        return await _anthropic_complete(messages, model, max_tokens)
    if provider == "google":
        return await _google_complete(messages, model, max_tokens)
    if provider == "openrouter":
        return await _openrouter_complete(messages, model, max_tokens)
    return await _ollama_complete(messages, model, max_tokens)


# ── OpenAI ───────────────────────────────────────────────────────────────────

async def _openai_complete(
    messages: List[Dict[str, str]], model: str, max_tokens: int
) -> str:
    import openai

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.7
    )
    return resp.choices[0].message.content or ""


# ── Anthropic ────────────────────────────────────────────────────────────────

async def _anthropic_complete(
    messages: List[Dict[str, str]], model: str, max_tokens: int
) -> str:
    import anthropic

    # Anthropic separates system prompt from messages
    system_text = ""
    user_messages: List[Dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            user_messages.append(m)

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=model or "claude-sonnet-4-20250514",
        system=system_text.strip(),
        messages=user_messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    # resp.content is a list of content blocks
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


# ── Google (Generative AI) ───────────────────────────────────────────────────

async def _google_complete(
    messages: List[Dict[str, str]], model: str, max_tokens: int
) -> str:
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    genai_model = genai.GenerativeModel(model or "gemini-1.5-flash")

    # Convert messages to a single prompt string for the Gemini API
    prompt_parts: List[str] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            prompt_parts.append(f"[System]: {m['content']}")
        elif role == "user":
            prompt_parts.append(f"[User]: {m['content']}")
        elif role == "assistant":
            prompt_parts.append(f"[Assistant]: {m['content']}")
    prompt_parts.append("[Assistant]:")

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: genai_model.generate_content(
            "\n\n".join(prompt_parts),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            ),
        ),
    )
    return result.text or ""


# ── OpenRouter (OpenAI-compatible) ───────────────────────────────────────────

async def _openrouter_complete(
    messages: List[Dict[str, str]], model: str, max_tokens: int
) -> str:
    import openai

    client = openai.AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.7
    )
    return resp.choices[0].message.content or ""


# ── Ollama / local ───────────────────────────────────────────────────────────

async def _ollama_complete(
    messages: List[Dict[str, str]], model: str, max_tokens: int
) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


# ── Swarm-powered completion ─────────────────────────────────────────────────

async def swarm_complete(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
) -> str:
    """Use the multi-model swarm pipeline for reasoning.

    Falls back to single-model llm_complete if the swarm fails.
    """
    from swarm.pipeline import SwarmPipeline

    # Extract the user query from messages
    query = ""
    context = ""
    for m in messages:
        if m["role"] == "user":
            query = m["content"]
        elif m["role"] == "system":
            context = m["content"]

    if not query:
        return await llm_complete(messages, max_tokens=max_tokens)

    try:
        pipeline = SwarmPipeline()
        result = await pipeline.run(query, context=context)
        answer = result.get("answer", "")
        if answer:
            log.info(
                "Swarm answer  confidence=%.2f  source=%s  providers=%s",
                result.get("confidence", 0),
                result.get("source", "?"),
                result.get("best_provider", "?"),
            )
            return answer
    except Exception as exc:
        log.warning("Swarm pipeline failed, falling back to single-model: %s", exc)

    return await llm_complete(messages, max_tokens=max_tokens)


# ── Context assembly ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Bluesky-Grokker, an intelligent assistant that lives natively
on the Bluesky social network.  You observe the global discourse, understand
context and nuance, and can answer questions about what's happening on Bluesky.

Guidelines:
- Be concise and insightful.
- Reference real posts and trends when available.
- Never fabricate post content.  If you lack data, say so.
- Keep responses under 280 characters when replying on Bluesky.
"""


async def build_context(notification: Dict[str, Any]) -> str:
    """Assemble relevant context for a notification the agent is responding to."""
    parts: List[str] = []

    # The triggering post text
    text = notification.get("text") or ""
    if text:
        parts.append(f"User @{notification.get('author_handle', '?')} wrote:\n\"{text}\"")

    # Semantic search for related posts
    if text:
        try:
            related = await query_engine.semantic_search(text, limit=5)
            if related:
                parts.append("\nRelated recent posts:")
                for r in related:
                    parts.append(f"  - [{r.get('author_did', '?')}]: \"{r.get('text', '')[:120]}\"")
        except Exception as exc:
            log.debug("context semantic search failed: %s", exc)

    # Trending topics
    try:
        trending = await query_engine.get_trending_topics(limit=5)
        if trending:
            tags = ", ".join(f"#{t['topic']}({t['count']})" for t in trending)
            parts.append(f"\nTrending: {tags}")
    except Exception as exc:
        log.debug("context trending failed: %s", exc)

    return "\n".join(parts)


# ── Agent decision loop ─────────────────────────────────────────────────────

class GrokkerAgent:
    """The autonomous Bluesky-Grokker agent."""

    def __init__(self, bsky_client: BlueskyClient) -> None:
        self._bsky = bsky_client
        self._running = False
        self._seen_uris: set[str] = set()
        self._stats = {"processed": 0, "replied": 0, "posted": 0, "errors": 0}

    async def start(self) -> None:
        """Main agent loop – polls notifications and decides how to act."""
        if not AGENT_ENABLED:
            log.info("Agent is disabled (AGENT_ENABLED=false). Skipping.")
            return

        self._running = True
        log.info("Agent loop starting  interval=%ss", AGENT_LOOP_INTERVAL)

        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                log.error("Agent tick error: %s", exc)
                self._stats["errors"] += 1

            await asyncio.sleep(AGENT_LOOP_INTERVAL)

    def stop(self) -> None:
        self._running = False
        log.info("Agent stopped  stats=%s", self._stats)

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # ── tick ─────────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        notifications = await asyncio.get_event_loop().run_in_executor(
            None, self._bsky.get_notifications, 25
        )

        for notif in notifications:
            uri = notif.get("uri", "")
            if uri in self._seen_uris:
                continue
            self._seen_uris.add(uri)

            # Only respond to mentions / replies
            reason = notif.get("reason", "")
            if reason not in ("mention", "reply"):
                continue

            text = notif.get("text", "")
            if not text:
                continue

            self._stats["processed"] += 1
            await self._respond_to(notif)

    async def _respond_to(self, notif: Dict[str, Any]) -> None:
        """Assemble context, call LLM, and publish the reply."""
        context = await build_context(notif)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        try:
            if SWARM_ENABLED:
                reply_text = await swarm_complete(messages, max_tokens=280)
            else:
                reply_text = await llm_complete(messages, max_tokens=280)
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            self._stats["errors"] += 1
            return

        # Trim to Bluesky limit (300 graphemes)
        reply_text = reply_text[:300]

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._bsky.reply_to_post(
                    text=reply_text,
                    parent_uri=notif["uri"],
                    parent_cid=notif["cid"],
                ),
            )
            self._stats["replied"] += 1
            log.info("Replied to %s: %.60s…", notif["uri"], reply_text)
        except Exception as exc:
            log.error("Reply failed: %s", exc)
            self._stats["errors"] += 1

    # ── autonomous posting ───────────────────────────────────────────────

    async def post_observation(self) -> Optional[str]:
        """Generate and publish an autonomous observation about current discourse."""
        try:
            trending = await query_engine.get_trending_topics(limit=10)
            recent = await storage.get_recent_posts(limit=20)
        except Exception as exc:
            log.error("Cannot gather context for observation: %s", exc)
            return None

        context_parts = ["Current Bluesky discourse snapshot:"]
        if trending:
            tags = ", ".join(f"#{t['topic']}({t['count']})" for t in trending)
            context_parts.append(f"Trending: {tags}")
        if recent:
            context_parts.append("Recent posts:")
            for p in recent[:10]:
                context_parts.append(f"  - \"{p.get('text', '')[:100]}\"")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Based on the following discourse snapshot, compose a single "
                    "insightful observation to post on Bluesky (max 280 chars).\n\n"
                    + "\n".join(context_parts)
                ),
            },
        ]

        try:
            if SWARM_ENABLED:
                text = await swarm_complete(messages, max_tokens=280)
            else:
                text = await llm_complete(messages, max_tokens=280)
            text = text.strip()[:300]
        except Exception as exc:
            log.error("LLM observation failed: %s", exc)
            return None

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._bsky.create_post(text)
            )
            self._stats["posted"] += 1
            log.info("Posted observation: %.80s…  uri=%s", text, result["uri"])
            return result["uri"]
        except Exception as exc:
            log.error("Observation post failed: %s", exc)
            return None
