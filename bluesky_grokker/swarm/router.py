"""ModelRouter – fans out queries to multiple LLM providers in parallel,
normalises responses into a common schema ready for distillation.

Inspired by elevate-foundry/braille model-router.js but implemented as
async Python using the provider backends already in agent.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SWARM_MAX_RETRIES,
    SWARM_STRATEGY,
    SWARM_TIMEOUT_MS,
)

log = logging.getLogger(__name__)


# ── Provider definitions ─────────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    name: str
    model: str
    api_key: str
    enabled: bool = True
    temperature: float = 0.3
    max_tokens: int = 2048


def _available_providers() -> List[ProviderConfig]:
    """Build the list of providers that have API keys configured."""
    providers: List[ProviderConfig] = []

    if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-key-here":
        providers.append(ProviderConfig(name="openai", model="gpt-4o", api_key=OPENAI_API_KEY))

    if ANTHROPIC_API_KEY:
        providers.append(ProviderConfig(name="anthropic", model="claude-sonnet-4-20250514", api_key=ANTHROPIC_API_KEY))

    if GOOGLE_API_KEY:
        providers.append(ProviderConfig(name="google", model="gemini-1.5-flash", api_key=GOOGLE_API_KEY))

    if OPENROUTER_API_KEY:
        providers.append(ProviderConfig(name="openrouter", model=LLM_MODEL, api_key=OPENROUTER_API_KEY))

    return providers


# ── Normalised response ──────────────────────────────────────────────────────

@dataclass
class ProviderResponse:
    provider: str
    model: str
    content: str = ""
    reasoning: str = ""
    answer: str = ""
    confidence: float = 0.5
    latency_s: float = 0.0
    error: Optional[str] = None


@dataclass
class RouterResult:
    query: str
    responses: List[ProviderResponse] = field(default_factory=list)
    aggregated: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# ── Router ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise reasoning engine. Respond with valid JSON containing: "
    '"reasoning" (step-by-step chain of thought), '
    '"answer" (concise conclusion), '
    '"confidence" (float 0-1).'
)


class ModelRouter:
    """Query multiple LLM providers in parallel and aggregate results."""

    def __init__(
        self,
        providers: Optional[List[ProviderConfig]] = None,
        strategy: str = SWARM_STRATEGY,
        timeout_ms: int = SWARM_TIMEOUT_MS,
        max_retries: int = SWARM_MAX_RETRIES,
    ) -> None:
        self.providers = providers or _available_providers()
        self.strategy = strategy
        self.timeout_s = timeout_ms / 1000.0
        self.max_retries = max_retries

    # ── public ────────────────────────────────────────────────────────────

    async def query(self, query: str, context: str = "") -> RouterResult:
        """Fan out to all providers and return normalised, aggregated result."""
        t0 = time.monotonic()

        tasks = [
            self._call_provider(p, query, context)
            for p in self.providers
            if p.enabled
        ]

        if not tasks:
            log.warning("ModelRouter: no providers available")
            return RouterResult(
                query=query,
                aggregated={"answer": "", "confidence": 0, "consensus": False},
                meta={"provider_count": 0},
            )

        if self.strategy == "fastest":
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            responses = [t.result() for t in done if not t.cancelled()]
        else:
            responses = await asyncio.gather(*tasks, return_exceptions=False)

        total_s = time.monotonic() - t0
        aggregated = self._aggregate(responses)

        return RouterResult(
            query=query,
            responses=responses,
            aggregated=aggregated,
            meta={
                "strategy": self.strategy,
                "provider_count": len(self.providers),
                "response_count": len(responses),
                "total_latency_s": round(total_s, 3),
            },
        )

    # ── per-provider call ─────────────────────────────────────────────────

    async def _call_provider(
        self, provider: ProviderConfig, query: str, context: str
    ) -> ProviderResponse:
        """Call a single provider with retry logic."""
        import json as _json

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{query}\n\nContext:\n{context}" if context else query},
        ]

        for attempt in range(1 + self.max_retries):
            t0 = time.monotonic()
            try:
                raw = await asyncio.wait_for(
                    self._dispatch(provider, messages),
                    timeout=self.timeout_s,
                )
                latency = time.monotonic() - t0
                return self._normalise(provider, raw, latency)

            except asyncio.TimeoutError:
                log.warning("Swarm timeout  provider=%s  attempt=%d", provider.name, attempt)
            except Exception as exc:
                log.warning("Swarm error  provider=%s  attempt=%d  err=%s", provider.name, attempt, exc)

        return ProviderResponse(
            provider=provider.name,
            model=provider.model,
            error=f"All {1 + self.max_retries} attempts failed",
        )

    async def _dispatch(self, provider: ProviderConfig, messages: List[Dict[str, str]]) -> str:
        """Route to the right provider backend and return raw text."""
        name = provider.name.lower()

        if name == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=provider.api_key)
            resp = await client.chat.completions.create(
                model=provider.model, messages=messages,
                max_tokens=provider.max_tokens, temperature=provider.temperature,
            )
            return resp.choices[0].message.content or ""

        if name == "anthropic":
            import anthropic
            system_text = ""
            user_msgs: List[Dict[str, str]] = []
            for m in messages:
                if m["role"] == "system":
                    system_text += m["content"] + "\n"
                else:
                    user_msgs.append(m)
            client = anthropic.AsyncAnthropic(api_key=provider.api_key)
            resp = await client.messages.create(
                model=provider.model, system=system_text.strip(),
                messages=user_msgs, max_tokens=provider.max_tokens,
                temperature=provider.temperature,
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))

        if name == "google":
            import google.generativeai as genai
            genai.configure(api_key=provider.api_key)
            model = genai.GenerativeModel(provider.model)
            prompt = "\n\n".join(f"[{m['role']}]: {m['content']}" for m in messages)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=provider.max_tokens,
                        temperature=provider.temperature,
                    ),
                ),
            )
            return result.text or ""

        if name == "openrouter":
            import openai
            client = openai.AsyncOpenAI(api_key=provider.api_key, base_url=OPENROUTER_BASE_URL)
            resp = await client.chat.completions.create(
                model=provider.model, messages=messages,
                max_tokens=provider.max_tokens, temperature=provider.temperature,
            )
            return resp.choices[0].message.content or ""

        # Ollama fallback
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": provider.model, "messages": messages, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")

    # ── normalisation ─────────────────────────────────────────────────────

    def _normalise(self, provider: ProviderConfig, raw: str, latency: float) -> ProviderResponse:
        import json as _json

        reasoning = ""
        answer = raw
        confidence = 0.5

        # Try to parse structured JSON from the model
        try:
            parsed = _json.loads(raw)
            reasoning = parsed.get("reasoning", "")
            answer = parsed.get("answer", raw)
            conf = parsed.get("confidence", 0.5)
            confidence = max(0.0, min(1.0, float(conf)))
        except (ValueError, TypeError, KeyError):
            pass

        return ProviderResponse(
            provider=provider.name,
            model=provider.model,
            content=raw,
            reasoning=reasoning,
            answer=answer,
            confidence=confidence,
            latency_s=round(latency, 3),
        )

    # ── aggregation ───────────────────────────────────────────────────────

    def _aggregate(self, responses: List[ProviderResponse]) -> Dict[str, Any]:
        successful = [r for r in responses if r.error is None]
        if not successful:
            return {"answer": "", "confidence": 0, "consensus": False, "best_provider": None}

        best = max(successful, key=lambda r: r.confidence)
        avg_conf = sum(r.confidence for r in successful) / len(successful)

        # Simple consensus: check if answers roughly agree
        answers = [r.answer.lower().strip()[:100] for r in successful]
        unique = set(answers)
        consensus = len(unique) == 1 or len(unique) <= len(successful) // 2

        return {
            "answer": best.answer,
            "confidence": round(avg_conf, 3),
            "best_provider": best.provider,
            "consensus": consensus,
            "provider_count": len(successful),
        }
