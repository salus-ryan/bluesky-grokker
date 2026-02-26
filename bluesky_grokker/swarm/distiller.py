"""Distiller – merges multi-model responses into a consensus thought structure.

Ported from elevate-foundry/braille distiller.js.

Pipeline:  ModelRouter → **Distiller** → SemanticCodec → compressed knowledge

Stages:
  1. Parse reasoning chains from each provider response
  2. Build concept graph across all providers
  3. Merge overlapping concepts (boost confidence for multi-provider agreement)
  4. Build a unified thought structure
  5. Encode via SemanticCodec
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from swarm.codec import SemanticCodec
from swarm.router import ProviderResponse, RouterResult

log = logging.getLogger(__name__)


# ── Concept pattern matching ─────────────────────────────────────────────────

CONCEPT_PATTERNS: Dict[str, re.Pattern] = {
    "ENTITY": re.compile(r"\b(entity|object|item|instance|thing|element)\b", re.I),
    "ACTION": re.compile(r"\b(action|do|perform|execute|run|process|compute)\b", re.I),
    "PROPERTY": re.compile(r"\b(property|attribute|field|characteristic|feature)\b", re.I),
    "STATE": re.compile(r"\b(state|status|condition|phase|mode)\b", re.I),
    "RELATION": re.compile(r"\b(relation|relationship|connection|link|association)\b", re.I),
    "CAUSE": re.compile(r"\b(cause|because|since|due to|reason)\b", re.I),
    "EFFECT": re.compile(r"\b(effect|result|outcome|consequence|impact)\b", re.I),
    "QUANTITY": re.compile(r"\b(quantity|count|number|amount|total|sum)\b", re.I),
    "QUALITY": re.compile(r"\b(quality|good|bad|better|worse|optimal)\b", re.I),
    "CONTAINS": re.compile(r"\b(contains|includes|has|comprises|consists)\b", re.I),
    "EQUALS": re.compile(r"\b(equals|is|same as|identical|equivalent)\b", re.I),
}

STEP_CLASSIFIERS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(analyze|examine|consider|look at)\b", re.I), "ANALYSIS"),
    (re.compile(r"\b(therefore|thus|conclude|result)\b", re.I), "CONCLUSION"),
    (re.compile(r"\b(assume|if|suppose|given)\b", re.I), "ASSUMPTION"),
    (re.compile(r"\b(compare|versus|contrast|differ)\b", re.I), "COMPARISON"),
    (re.compile(r"\b(synthesize|combine|merge|integrate)\b", re.I), "SYNTHESIS"),
]


def _extract_concepts_from_text(text: str) -> List[str]:
    """Extract semantic concepts from a reasoning step."""
    concepts: List[str] = []
    # Quoted terms
    for match in re.finditer(r'"([^"]+)"', text):
        concepts.append(match.group(1))
    # Pattern-matched concepts
    for concept_name, pattern in CONCEPT_PATTERNS.items():
        if pattern.search(text):
            concepts.append(concept_name)
    # Capitalised noun-like tokens
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        concepts.append(match.group().lower())
    return concepts


def _classify_step(text: str) -> str:
    for pattern, label in STEP_CLASSIFIERS:
        if pattern.search(text):
            return label
    return "REASONING"


# ── Distiller ─────────────────────────────────────────────────────────────────

class Distiller:
    """Distil multi-provider reasoning into a single compressed thought."""

    def __init__(
        self,
        merge_threshold: float = 0.7,
        codec: SemanticCodec | None = None,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.codec = codec or SemanticCodec()

        self._stats = {
            "distillations": 0,
            "total_concepts_extracted": 0,
            "total_concepts_merged": 0,
            "avg_compression_ratio": 0.0,
        }

    # ── Public API ────────────────────────────────────────────────────────

    def distill(self, router_result: RouterResult) -> Dict[str, Any]:
        """Full distillation pipeline: responses → thought → encoded packet."""
        t0 = time.monotonic()

        # Stage 1: extract reasoning steps per provider
        reasoning_steps = self._extract_reasoning_steps(router_result)

        # Stage 2: build concept graph
        concept_graph = self._build_concept_graph(reasoning_steps)

        # Stage 3: merge overlapping concepts across providers
        merged_graph = self._merge_concept_graph(concept_graph)

        # Stage 4: build thought structure
        thought = self._build_thought(merged_graph, router_result)

        # Stage 5: encode with SemanticCodec (MOTL)
        encoded = self.codec.encode_thought(thought)

        latency_ms = (time.monotonic() - t0) * 1000
        self._update_stats(concept_graph, merged_graph, encoded)

        return {
            "thought": thought,
            "encoded": encoded,
            "meta": {
                "latency_ms": round(latency_ms, 2),
                "reasoning_step_count": sum(len(rs["steps"]) for rs in reasoning_steps),
                "concept_count": len(concept_graph["concepts"]),
                "merged_concept_count": len(merged_graph["concepts"]),
                "compression_ratio": encoded.get("compression_ratio", 0),
                "providers": [r.provider for r in router_result.responses if r.error is None],
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "codec": self.codec.get_stats()}

    # ── Stage 1: Extract reasoning steps ──────────────────────────────────

    def _extract_reasoning_steps(self, router_result: RouterResult) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []

        for resp in router_result.responses:
            if resp.error:
                continue

            raw = resp.reasoning or resp.content or ""
            parsed = self._parse_reasoning(raw, resp.provider)
            steps.append({
                "provider": resp.provider,
                "confidence": resp.confidence,
                "steps": parsed,
                "answer": resp.answer,
            })

        return steps

    def _parse_reasoning(self, text: str, provider: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        lines = re.split(r"\n|(?:Step \d+:)|(?:- )", text)
        lines = [l.strip() for l in lines if l.strip()]

        return [
            {
                "index": i,
                "provider": provider,
                "text": line,
                "concepts": _extract_concepts_from_text(line),
                "type": _classify_step(line),
            }
            for i, line in enumerate(lines)
        ]

    # ── Stage 2: Build concept graph ──────────────────────────────────────

    def _build_concept_graph(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        concepts: Dict[str, Dict[str, Any]] = {}
        relations: List[Dict[str, str]] = []

        for chain in reasoning_steps:
            prev_concepts: List[str] = []

            for step in chain["steps"]:
                for concept in step["concepts"]:
                    if concept not in concepts:
                        concepts[concept] = {
                            "frequency": 0,
                            "providers": set(),
                            "types": set(),
                            "confidence": 0.0,
                        }
                    info = concepts[concept]
                    info["frequency"] += 1
                    info["providers"].add(chain["provider"])
                    info["types"].add(step["type"])
                    info["confidence"] = max(info["confidence"], chain["confidence"])

                    for prev in prev_concepts:
                        if prev != concept:
                            relations.append({
                                "source": prev,
                                "target": concept,
                                "type": "FOLLOWS",
                                "provider": chain["provider"],
                            })

                prev_concepts = step["concepts"]

        return {"concepts": concepts, "relations": relations}

    # ── Stage 3: Merge concept graph ──────────────────────────────────────

    def _merge_concept_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Dict[str, Any]] = {}

        for concept, info in graph["concepts"].items():
            pc = len(info["providers"])
            boosted = min(1.0, info["confidence"] * (1 + (pc - 1) * 0.15))
            merged[concept] = {
                "frequency": info["frequency"],
                "provider_count": pc,
                "confidence": boosted,
                "types": list(info["types"]),
            }

        # De-duplicate relations
        seen: Set[str] = set()
        merged_rels: List[Dict[str, str]] = []
        for rel in graph["relations"]:
            key = f"{rel['source']}→{rel['type']}→{rel['target']}"
            if key not in seen:
                seen.add(key)
                merged_rels.append(rel)

        return {"concepts": merged, "relations": merged_rels}

    # ── Stage 4: Build thought structure ──────────────────────────────────

    def _build_thought(self, merged: Dict[str, Any], router_result: RouterResult) -> Dict[str, Any]:
        sorted_concepts = sorted(
            merged["concepts"].items(),
            key=lambda kv: kv[1]["confidence"] * kv[1]["frequency"],
            reverse=True,
        )

        return {
            "type": "DISTILLED_REASONING",
            "query": router_result.query,
            "concepts": [
                {
                    "name": name,
                    "confidence": info["confidence"],
                    "frequency": info["frequency"],
                    "provider_count": info["provider_count"],
                    "types": info["types"],
                }
                for name, info in sorted_concepts
            ],
            "relations": [
                {"source": r["source"], "target": r["target"], "type": r["type"]}
                for r in merged["relations"]
            ],
            "conclusion": {
                "answer": router_result.aggregated.get("answer", ""),
                "confidence": router_result.aggregated.get("confidence", 0),
                "consensus": router_result.aggregated.get("consensus", False),
                "best_provider": router_result.aggregated.get("best_provider"),
            },
        }

    # ── Stats ─────────────────────────────────────────────────────────────

    def _update_stats(
        self,
        raw_graph: Dict[str, Any],
        merged_graph: Dict[str, Any],
        encoded: Dict[str, Any],
    ) -> None:
        self._stats["distillations"] += 1
        self._stats["total_concepts_extracted"] += len(raw_graph["concepts"])
        self._stats["total_concepts_merged"] += len(merged_graph["concepts"])

        ratio = encoded.get("compression_ratio", 1.0)
        n = self._stats["distillations"]
        self._stats["avg_compression_ratio"] = (
            self._stats["avg_compression_ratio"] * (n - 1) + ratio
        ) / n
