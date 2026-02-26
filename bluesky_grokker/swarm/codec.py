"""SemanticCodec – MOTL-inspired variable-bit-depth concept encoding.

Ported from elevate-foundry/braille semantic-codec.js / motl-protocol.js.

Encodes semantic concepts into compact bit-string representations using
frequency-adaptive variable-length coding.  Higher-frequency concepts get
shorter encodings, similar to Huffman coding but with reinforcement-learning
based optimisation for streaming contexts.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


@dataclass
class ConceptInfo:
    frequency: int = 0
    confidence: float = 0.0
    provider_count: int = 0
    types: List[str] = field(default_factory=list)


class SemanticCodec:
    """Variable-bit-depth encoder for semantic concepts.

    Maintains encoding tables at multiple bit depths (1-8 bits) and
    dynamically reassigns concepts to shorter codes as they become
    more frequent in the conversation stream.
    """

    def __init__(
        self,
        initial_bit_depth: int = 3,
        adaptive: bool = True,
        context_window: int = 1000,
        rl_enabled: bool = True,
    ) -> None:
        self.initial_bit_depth = initial_bit_depth
        self.adaptive = adaptive
        self.context_window = context_window
        self.rl_enabled = rl_enabled

        # Fixed encoding tables (like MOTL's coreOperations / commonRelations)
        self._core_ops: Dict[str, str] = {"AND": "0", "OR": "1"}
        self._common_relations: Dict[str, str] = {
            "EQUALS": "00", "CONTAINS": "01", "GREATER_THAN": "10", "LESS_THAN": "11",
        }
        self._fundamental: Dict[str, str] = {
            "ENTITY": "000", "ACTION": "001", "PROPERTY": "010", "STATE": "011",
            "EVENT": "100", "RELATION": "101", "QUANTITY": "110", "QUALITY": "111",
        }

        # Dynamic concepts (learned at runtime)
        self._dynamic: Dict[str, str] = {}

        # Context tracking
        self._concept_freq: Dict[str, int] = defaultdict(int)
        self._recent: List[str] = []
        self._relation_graph: Dict[str, Set[str]] = defaultdict(set)

        # Performance metrics
        self.metrics: Dict[str, float] = {
            "compression_ratio": 0.0,
            "processing_ms": 0.0,
            "adaptation_rate": 0.0,
        }

        # RL state
        if rl_enabled:
            self._rl_lr = 0.1
            self._rl_gamma = 0.9
            self._rl_epsilon = 0.2
            self._rl_values: Dict[str, float] = {}

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_concept(self, concept: str) -> Optional[str]:
        """Look up the bit encoding for a concept across all tables."""
        for table in (self._core_ops, self._common_relations, self._fundamental, self._dynamic):
            if concept in table:
                return table[concept]
        return None

    def encode_concept_with_fallback(self, concept: str) -> str:
        """Encode a concept, falling back to literal 8-bit-per-char encoding."""
        bits = self.encode_concept(concept)
        if bits is not None:
            return bits
        # Literal fallback: prefix marker + 8 bits per character
        literal = "".join(format(ord(c), "08b") for c in concept)
        return "1111" + literal

    def has_fixed_encoding(self, concept: str) -> bool:
        return concept in self._core_ops or concept in self._common_relations or concept in self._fundamental

    # ── Encoding table management ─────────────────────────────────────────

    def assign_encodings(self, concepts: List[str], bit_depth: int) -> None:
        """Assign bit encodings to a list of concepts at a given bit depth."""
        max_enc = 2 ** bit_depth
        for i, concept in enumerate(concepts[:max_enc]):
            self._dynamic[concept] = format(i, f"0{bit_depth}b")

    def reassign_bit_depths(self, sorted_concepts: List[str]) -> None:
        """Reassign encodings based on frequency-sorted concepts.

        Most frequent → shortest codes.
        """
        self._dynamic.clear()
        idx = 0
        tiers = [(1, 2), (2, 4), (3, 8), (4, 16)]

        for bits, max_count in tiers:
            count = min(max_count, len(sorted_concepts) - idx)
            if count <= 0:
                break
            self.assign_encodings(sorted_concepts[idx : idx + count], bits)
            idx += count

        # Remaining concepts get 8-bit depth
        if idx < len(sorted_concepts):
            self.assign_encodings(sorted_concepts[idx:], 8)

    # ── Context tracking ──────────────────────────────────────────────────

    def update_context(self, concepts: List[str]) -> None:
        """Track concept frequency and optionally re-optimise encodings."""
        for c in concepts:
            self._concept_freq[c] += 1

        self._recent = (concepts + self._recent)[: self.context_window]

        if self.adaptive:
            sorted_concepts = sorted(
                (c for c in self._concept_freq if not self.has_fixed_encoding(c)),
                key=lambda c: self._concept_freq[c],
                reverse=True,
            )
            self.reassign_bit_depths(sorted_concepts)

    def update_relations(self, relations: List[Dict[str, str]]) -> None:
        """Track relation graph for concept connectivity."""
        for rel in relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src and tgt:
                self._relation_graph[src].add(tgt)

    # ── RL optimisation ───────────────────────────────────────────────────

    def update_rl(self, compression_ratio: float, processing_ms: float) -> None:
        """Update the RL value function based on encoding performance."""
        if not self.rl_enabled:
            return

        reward = compression_ratio - (processing_ms / 1000.0)

        for concept, bits in self._dynamic.items():
            key = f"{concept}:{len(bits)}"
            current = self._rl_values.get(key, 0.0)
            self._rl_values[key] = current + self._rl_lr * (reward - current)

        self._rl_epsilon *= 0.999

    # ── Thought structure encoding (MOTL-style) ──────────────────────────

    def encode_thought(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Encode a thought structure into a compact bit representation."""
        import time
        t0 = time.monotonic()

        concepts = self._extract_concepts(thought)
        relations = self._extract_relations(thought)

        # Update context
        self.update_context(concepts)
        self.update_relations(relations)

        # Build bit string
        header = "101010"  # Version marker
        concept_bits = "".join(self.encode_concept_with_fallback(c) for c in concepts)
        relation_bits = ""
        for rel in relations:
            relation_bits += self.encode_concept_with_fallback(rel.get("type", "RELATED"))
            relation_bits += self.encode_concept_with_fallback(rel.get("source", ""))
            relation_bits += self.encode_concept_with_fallback(rel.get("target", ""))

        bit_string = header + concept_bits + relation_bits
        size = len(bit_string)

        # Compression ratio
        original_size = len(str(thought)) * 8
        ratio = original_size / max(size, 1)

        elapsed_ms = (time.monotonic() - t0) * 1000
        self.metrics["compression_ratio"] = ratio
        self.metrics["processing_ms"] = elapsed_ms

        if self.rl_enabled:
            self.update_rl(ratio, elapsed_ms)

        return {
            "encoded": bit_string,
            "size": size,
            "compression_ratio": round(ratio, 3),
            "concept_count": len(concepts),
            "relation_count": len(relations),
        }

    # ── Concept extraction ────────────────────────────────────────────────

    def _extract_concepts(self, obj: Any, depth: int = 0) -> List[str]:
        """Recursively extract concept strings from a thought structure."""
        if depth > 10:
            return []
        concepts: List[str] = []
        if isinstance(obj, dict):
            for key, val in obj.items():
                concepts.append(str(key))
                concepts.extend(self._extract_concepts(val, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                concepts.extend(self._extract_concepts(item, depth + 1))
        else:
            concepts.append(str(obj))
        return concepts

    def _extract_relations(self, obj: Any) -> List[Dict[str, str]]:
        """Extract relations from the thought structure."""
        if not isinstance(obj, dict):
            return []
        relations: List[Dict[str, str]] = []
        for key, val in obj.items():
            if isinstance(val, dict):
                relations.append({"type": "CONTAINS", "source": str(key), "target": str(key)})
            elif val is not None:
                relations.append({"type": "EQUALS", "source": str(key), "target": str(val)})
        return relations

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "encoding_tables": {
                "core_ops": len(self._core_ops),
                "common_relations": len(self._common_relations),
                "fundamental": len(self._fundamental),
                "dynamic": len(self._dynamic),
                "total": len(self._core_ops) + len(self._common_relations) + len(self._fundamental) + len(self._dynamic),
            },
            "context": {
                "recent_count": len(self._recent),
                "unique_concepts": len(self._concept_freq),
                "relation_graph_size": len(self._relation_graph),
            },
            "metrics": dict(self.metrics),
            "rl": {
                "epsilon": round(self._rl_epsilon, 4),
                "value_fn_size": len(self._rl_values),
            } if self.rl_enabled else None,
        }
