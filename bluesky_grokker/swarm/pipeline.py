"""SwarmPipeline – end-to-end multi-model reasoning orchestrator.

Ported from elevate-foundry/braille pipeline.js (DistillationPipeline).

Wires together:
  [Query] → ModelRouter → Distiller → SemanticCodec → [Answer + Knowledge]

Also maintains a lightweight local knowledge base (edge-runtime style)
for caching distilled conclusions and concept lookups.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from swarm.codec import SemanticCodec
from swarm.distiller import Distiller
from swarm.router import ModelRouter, RouterResult

log = logging.getLogger(__name__)


class KnowledgeBase:
    """Lightweight local knowledge store inspired by braille EdgeRuntime.

    Caches distilled concepts, relations, and conclusions for fast
    re-use without hitting upstream LLMs.
    """

    def __init__(self, max_concepts: int = 2000) -> None:
        self.max_concepts = max_concepts
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.relations: List[Dict[str, str]] = []
        self.conclusions: Dict[str, Dict[str, Any]] = {}

    def ingest(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a distilled thought structure."""
        added = 0

        for concept in thought.get("concepts", []):
            name = concept.get("name", str(concept))
            if name in self.concepts:
                existing = self.concepts[name]
                existing["frequency"] = existing.get("frequency", 0) + concept.get("frequency", 1)
                existing["confidence"] = max(
                    existing.get("confidence", 0), concept.get("confidence", 0.5)
                )
            else:
                self.concepts[name] = {
                    "frequency": concept.get("frequency", 1),
                    "confidence": concept.get("confidence", 0.5),
                    "provider_count": concept.get("provider_count", 1),
                    "types": concept.get("types", []),
                }
                added += 1

        for rel in thought.get("relations", []):
            self.relations.append(rel)

        # Cache conclusion
        query = thought.get("query", "")
        conclusion = thought.get("conclusion", {})
        if query and conclusion.get("answer"):
            key = self._hash_query(query)
            self.conclusions[key] = {
                "query": query,
                "answer": conclusion["answer"],
                "confidence": conclusion.get("confidence", 0.5),
                "consensus": conclusion.get("consensus", False),
                "timestamp": time.time(),
            }

        self._enforce_budget()

        return {
            "concepts_added": added,
            "total_concepts": len(self.concepts),
            "total_relations": len(self.relations),
            "total_conclusions": len(self.conclusions),
        }

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we already have a cached conclusion for this query."""
        key = self._hash_query(query)
        return self.conclusions.get(key)

    def _hash_query(self, query: str) -> str:
        import hashlib
        normalised = query.lower().strip()
        return hashlib.md5(normalised.encode()).hexdigest()[:12]

    def _enforce_budget(self) -> None:
        if len(self.concepts) <= self.max_concepts:
            return
        # Evict lowest-value concepts
        scored = sorted(
            self.concepts.items(),
            key=lambda kv: kv[1].get("frequency", 0) * kv[1].get("confidence", 0),
        )
        to_remove = len(self.concepts) - self.max_concepts
        for name, _ in scored[:to_remove]:
            del self.concepts[name]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "concepts": len(self.concepts),
            "relations": len(self.relations),
            "conclusions": len(self.conclusions),
        }


class SwarmPipeline:
    """Orchestrates multi-model reasoning with distillation and caching.

    Usage:
        pipeline = SwarmPipeline()
        result = await pipeline.run("What are the trending topics on Bluesky?")
        print(result["answer"])
    """

    def __init__(
        self,
        router: ModelRouter | None = None,
        distiller: Distiller | None = None,
        codec: SemanticCodec | None = None,
        use_cache: bool = True,
    ) -> None:
        self.codec = codec or SemanticCodec()
        self.router = router or ModelRouter()
        self.distiller = distiller or Distiller(codec=self.codec)
        self.kb = KnowledgeBase() if use_cache else None

        self._log: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────

    async def run(self, query: str, context: str = "") -> Dict[str, Any]:
        """Run the full swarm pipeline for a query."""
        t0 = time.monotonic()
        entry: Dict[str, Any] = {"query": query, "stages": {}}

        # Check local cache first
        if self.kb:
            cached = self.kb.lookup(query)
            if cached:
                entry["stages"]["cache"] = {"hit": True}
                entry["total_s"] = round(time.monotonic() - t0, 3)
                self._log.append(entry)
                log.info("Swarm cache hit  query=%s…", query[:50])
                return {
                    "answer": cached["answer"],
                    "confidence": cached["confidence"],
                    "consensus": cached.get("consensus", False),
                    "source": "cache",
                    "pipeline": entry,
                }

        # Stage 1: Route to multiple providers
        router_result = await self.router.query(query, context)
        entry["stages"]["router"] = {
            "latency_s": router_result.meta.get("total_latency_s", 0),
            "providers": router_result.meta.get("provider_count", 0),
            "responses": router_result.meta.get("response_count", 0),
        }

        # Stage 2: Distill into MOTL thought structure
        distilled = self.distiller.distill(router_result)
        entry["stages"]["distiller"] = {
            "latency_ms": distilled["meta"]["latency_ms"],
            "concepts": distilled["meta"]["concept_count"],
            "merged_concepts": distilled["meta"]["merged_concept_count"],
            "compression_ratio": distilled["meta"]["compression_ratio"],
        }

        # Stage 3: Ingest into local knowledge base
        if self.kb:
            ingest_result = self.kb.ingest(distilled["thought"])
            entry["stages"]["knowledge_base"] = ingest_result

        entry["total_s"] = round(time.monotonic() - t0, 3)
        self._log.append(entry)

        answer = router_result.aggregated.get("answer", "")
        confidence = router_result.aggregated.get("confidence", 0)

        log.info(
            "Swarm complete  providers=%d  concepts=%d  confidence=%.2f  time=%.2fs",
            entry["stages"]["router"]["providers"],
            distilled["meta"]["merged_concept_count"],
            confidence,
            entry["total_s"],
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "consensus": router_result.aggregated.get("consensus", False),
            "best_provider": router_result.aggregated.get("best_provider"),
            "source": "swarm",
            "distilled": {
                "concept_count": distilled["meta"]["merged_concept_count"],
                "compression_ratio": distilled["meta"]["compression_ratio"],
                "encoded_size": distilled["encoded"]["size"],
            },
            "pipeline": entry,
        }

    async def run_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Run multiple queries sequentially, building up the knowledge base."""
        results = []
        for q in queries:
            results.append(await self.run(q))
        return results

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pipeline": {
                "total_runs": len(self._log),
                "avg_latency_s": (
                    sum(e.get("total_s", 0) for e in self._log) / len(self._log)
                    if self._log
                    else 0
                ),
            },
            "distiller": self.distiller.get_stats(),
            "knowledge_base": self.kb.get_stats() if self.kb else None,
        }
