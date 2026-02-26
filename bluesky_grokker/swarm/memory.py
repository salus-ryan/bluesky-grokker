"""BrailleMemory – a continuously-learning braille-native concept model.

This is not a static LLM.  It is a living weight distribution over a semantic
concept graph encoded in Z₂⁸ braille space.  Every firehose window, every
like, every reply shifts the weights.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                  BrailleMemory                      │
  │                                                     │
  │  concept_weights : {concept → float}                │
  │    ↑ updated by every firehose ingest               │
  │    ↑ boosted by likes/reposts on related concepts   │
  │    ↑ decayed over time (exponential half-life)      │
  │                                                     │
  │  relation_graph  : {(src, type, tgt) → float}       │
  │    ↑ edges accumulate from swarm extraction          │
  │    ↑ weights = co-occurrence × provider agreement    │
  │                                                     │
  │  codec           : TieredConceptCodec (rebuilt each  │
  │                    epoch from current weights)       │
  │                                                     │
  │  epoch_history   : [{timestamp, top_concepts,        │
  │                      drift_score, stats}]           │
  │                                                     │
  │  interaction_log : [{type, concept, delta, ts}]     │
  │                                                     │
  └─────────────────────────────────────────────────────┘

The codec tier assignments are a FUNCTION of the weights — frequent/heavy
concepts get short codes, fading concepts get long codes or fall off entirely.
This means the braille encoding itself drifts with the discourse.

Persistence: the entire state serializes to JSON and can be saved to a Modal
Volume, SQLite, or flat file between runs.  On next boot, the memory loads
its prior state and continues learning.

Response generation: BrailleMemory can "think" by traversing its weighted
concept graph (activation spreading), producing a braille-native thought
that gets decoded to English only as a final render step.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

BRAILLE_BASE = 0x2800


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class ConceptNode:
    """A node in the braille concept graph."""
    weight: float = 0.0
    frequency: int = 0
    providers: Set[str] = field(default_factory=set)
    last_seen: float = 0.0       # epoch timestamp
    categories: Set[str] = field(default_factory=set)  # topics, entities, etc.
    interaction_boosts: float = 0.0  # accumulated from likes/reposts

    @property
    def effective_weight(self) -> float:
        return self.weight + self.interaction_boosts

    def to_dict(self) -> dict:
        return {
            "w": round(self.weight, 4),
            "f": self.frequency,
            "p": list(self.providers),
            "ts": round(self.last_seen, 2),
            "cat": list(self.categories),
            "ib": round(self.interaction_boosts, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptNode":
        return cls(
            weight=d.get("w", 0.0),
            frequency=d.get("f", 0),
            providers=set(d.get("p", [])),
            last_seen=d.get("ts", 0.0),
            categories=set(d.get("cat", [])),
            interaction_boosts=d.get("ib", 0.0),
        )


@dataclass
class RelationEdge:
    """A weighted edge: (src) →[type]→ (tgt)."""
    weight: float = 0.0
    count: int = 0
    providers: Set[str] = field(default_factory=set)
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "w": round(self.weight, 4),
            "n": self.count,
            "p": list(self.providers),
            "ts": round(self.last_seen, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationEdge":
        return cls(
            weight=d.get("w", 0.0),
            count=d.get("n", 0),
            providers=set(d.get("p", [])),
            last_seen=d.get("ts", 0.0),
        )


@dataclass
class Epoch:
    """Snapshot of one learning epoch (one firehose window)."""
    timestamp: float
    n_posts: int
    n_concepts_ingested: int
    n_relations_ingested: int
    top_concepts: List[str]
    drift_score: float
    avg_bits: float
    stats: Dict[str, Any] = field(default_factory=dict)


# ── BrailleMemory ────────────────────────────────────────────────────────────

class BrailleMemory:
    """A living, braille-native concept model.

    The weights are not static.  Every ingest() call shifts them.
    The codec is rebuilt from the current weight distribution.
    The braille encoding of any concept can change between epochs.
    """

    # Interaction signal strengths
    SIGNAL_POST     = 1.0
    SIGNAL_REPLY    = 1.5    # replies carry more intent than posts
    SIGNAL_LIKE     = 0.3
    SIGNAL_REPOST   = 0.8
    SIGNAL_FOLLOW   = 0.1    # weak concept signal, strong social signal

    # Decay
    HALF_LIFE_SECS  = 3600.0   # concept weight halves every hour
    DECAY_FLOOR     = 0.01     # minimum weight before pruning

    # Graph limits
    MAX_CONCEPTS    = 4096
    MAX_RELATIONS   = 8192
    MAX_EPOCHS      = 1000

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relations: Dict[str, RelationEdge] = {}  # key = "src→type→tgt"
        self.epochs: List[Epoch] = []
        self.interaction_log: List[dict] = []
        self._birth_ts: float = time.time()
        self._total_posts_seen: int = 0
        self._total_interactions: int = 0

    # ── Core: ingest concepts from a swarm extraction ────────────────────

    def ingest_extraction(
        self,
        concepts_by_category: Dict[str, List[str]],
        relations: List[dict],
        provider: str = "unknown",
        n_posts: int = 0,
    ) -> dict:
        """Feed one model's extraction into memory.

        concepts_by_category: {"topics": [...], "entities": [...], ...}
        relations: [{"src": ..., "type": ..., "tgt": ...}, ...]
        """
        now = time.time()
        n_concepts = 0
        n_relations = 0

        # Ingest concepts
        for category, concept_list in concepts_by_category.items():
            for concept in concept_list:
                if not concept:
                    continue
                concept_lower = concept.lower().strip()
                node = self.concepts.get(concept_lower)
                if node is None:
                    node = ConceptNode()
                    self.concepts[concept_lower] = node

                node.frequency += 1
                node.weight += self.SIGNAL_POST
                node.providers.add(provider)
                node.last_seen = now
                node.categories.add(category)
                n_concepts += 1

        # Ingest relations
        for rel in relations:
            src = (rel.get("src") or "").lower().strip()
            rtype = (rel.get("type") or "").upper().strip()
            tgt = (rel.get("tgt") or "").lower().strip()
            if not (src and rtype and tgt):
                continue

            key = f"{src}→{rtype}→{tgt}"
            edge = self.relations.get(key)
            if edge is None:
                edge = RelationEdge()
                self.relations[key] = edge

            edge.count += 1
            edge.weight += self.SIGNAL_POST
            edge.providers.add(provider)
            edge.last_seen = now
            n_relations += 1

        self._total_posts_seen += n_posts

        return {"concepts_ingested": n_concepts, "relations_ingested": n_relations}

    # ── Interaction feedback ─────────────────────────────────────────────

    def record_interaction(
        self,
        interaction_type: str,
        concepts: List[str],
        source_did: str = "",
    ) -> None:
        """Record a like/repost/reply/follow that boosts related concepts."""
        signal = {
            "like": self.SIGNAL_LIKE,
            "repost": self.SIGNAL_REPOST,
            "reply": self.SIGNAL_REPLY,
            "follow": self.SIGNAL_FOLLOW,
        }.get(interaction_type, 0.1)

        now = time.time()
        for concept in concepts:
            concept_lower = concept.lower().strip()
            node = self.concepts.get(concept_lower)
            if node is not None:
                node.interaction_boosts += signal
                node.last_seen = now

        self._total_interactions += 1
        self.interaction_log.append({
            "type": interaction_type,
            "concepts": concepts[:5],
            "signal": signal,
            "ts": now,
        })
        # Keep interaction log bounded
        if len(self.interaction_log) > 10000:
            self.interaction_log = self.interaction_log[-5000:]

    # ── Decay: the forgetting curve ──────────────────────────────────────

    def apply_decay(self) -> dict:
        """Apply exponential time-decay to all concept and relation weights.

        Concepts that haven't been seen recently fade.  This is how the
        model 'forgets' — not by deleting, but by letting weights approach
        zero until they're pruned.
        """
        now = time.time()
        ln2 = math.log(2)
        decay_lambda = ln2 / self.HALF_LIFE_SECS

        pruned_concepts = 0
        pruned_relations = 0

        # Decay concept weights
        to_prune = []
        for concept, node in self.concepts.items():
            dt = now - node.last_seen
            if dt > 0:
                factor = math.exp(-decay_lambda * dt)
                node.weight *= factor
                node.interaction_boosts *= factor
            if node.effective_weight < self.DECAY_FLOOR and node.frequency < 3:
                to_prune.append(concept)

        for c in to_prune:
            del self.concepts[c]
            pruned_concepts += 1

        # Decay relation weights
        rel_prune = []
        for key, edge in self.relations.items():
            dt = now - edge.last_seen
            if dt > 0:
                edge.weight *= math.exp(-decay_lambda * dt)
            if edge.weight < self.DECAY_FLOOR and edge.count < 2:
                rel_prune.append(key)

        for k in rel_prune:
            del self.relations[k]
            pruned_relations += 1

        return {
            "pruned_concepts": pruned_concepts,
            "pruned_relations": pruned_relations,
            "remaining_concepts": len(self.concepts),
            "remaining_relations": len(self.relations),
        }

    # ── Epoch: snapshot + rebuild codec ──────────────────────────────────

    def close_epoch(self, n_posts: int = 0) -> Epoch:
        """Close the current learning epoch.

        1. Apply decay
        2. Compute drift from last epoch
        3. Record snapshot
        4. Prune if over capacity
        """
        decay_stats = self.apply_decay()

        # Rank concepts by effective weight
        ranked = sorted(
            self.concepts.items(),
            key=lambda x: x[1].effective_weight,
            reverse=True,
        )
        top_concepts = [c for c, _ in ranked[:20]]

        # Drift score: how much did the top-20 change since last epoch?
        drift = 0.0
        if self.epochs:
            prev_top = set(self.epochs[-1].top_concepts)
            curr_top = set(top_concepts)
            if prev_top or curr_top:
                drift = 1.0 - len(prev_top & curr_top) / max(len(prev_top | curr_top), 1)

        # Compute average bits/concept from the weight distribution
        avg_bits = self._estimate_avg_bits(ranked)

        epoch = Epoch(
            timestamp=time.time(),
            n_posts=n_posts,
            n_concepts_ingested=sum(n.frequency for n in self.concepts.values()),
            n_relations_ingested=sum(e.count for e in self.relations.values()),
            top_concepts=top_concepts,
            drift_score=round(drift, 4),
            avg_bits=round(avg_bits, 2),
            stats=decay_stats,
        )
        self.epochs.append(epoch)

        # Bound epoch history
        if len(self.epochs) > self.MAX_EPOCHS:
            self.epochs = self.epochs[-self.MAX_EPOCHS:]

        # Prune to capacity
        self._prune_to_capacity()

        return epoch

    def _estimate_avg_bits(self, ranked: list) -> float:
        """Estimate avg bits/concept under the tiered codec for this distribution."""
        # Tier layout mirrors TieredConceptCodec
        tiers = [(3, 2), (4, 4), (5, 8), (7, 16)]  # (bits, slots)
        long_bits = 11

        total_weight = sum(n.effective_weight for _, n in ranked)
        if total_weight == 0:
            return 8.0

        avg = 0.0
        idx = 0
        for bits, slots in tiers:
            for _ in range(slots):
                if idx >= len(ranked):
                    break
                w = ranked[idx][1].effective_weight / total_weight
                avg += bits * w
                idx += 1

        # Remaining concepts at long tier
        for i in range(idx, len(ranked)):
            w = ranked[i][1].effective_weight / total_weight
            avg += long_bits * w

        return avg

    def _prune_to_capacity(self) -> None:
        """Drop lowest-weight concepts/relations if over capacity."""
        if len(self.concepts) > self.MAX_CONCEPTS:
            ranked = sorted(
                self.concepts.items(),
                key=lambda x: x[1].effective_weight,
            )
            for concept, _ in ranked[:len(self.concepts) - self.MAX_CONCEPTS]:
                del self.concepts[concept]

        if len(self.relations) > self.MAX_RELATIONS:
            ranked = sorted(
                self.relations.items(),
                key=lambda x: x[1].weight,
            )
            for key, _ in ranked[:len(self.relations) - self.MAX_RELATIONS]:
                del self.relations[key]

    # ── Thinking: activation spreading in braille space ──────────────────

    def think(self, seed_concepts: List[str], depth: int = 2, top_k: int = 15) -> dict:
        """Spread activation from seed concepts through the relation graph.

        This is how the model 'reasons' — it starts with seed concepts,
        follows weighted edges, and returns the activated subgraph.
        The result is a braille-native thought: a weighted concept set
        with relation paths, ready for encoding or English decode.
        """
        activated: Dict[str, float] = {}
        paths: List[str] = []

        # Seed activation
        for concept in seed_concepts:
            c = concept.lower().strip()
            node = self.concepts.get(c)
            if node:
                activated[c] = node.effective_weight
            else:
                activated[c] = 0.1  # weak activation for unknown seeds

        # Spread through relation graph
        frontier = list(activated.keys())
        for d in range(depth):
            next_frontier = []
            damping = 1.0 / (d + 2)  # activation decays with depth

            for src in frontier:
                for key, edge in self.relations.items():
                    parts = key.split("→")
                    if len(parts) != 3:
                        continue
                    rel_src, rel_type, rel_tgt = parts

                    target = None
                    if rel_src == src:
                        target = rel_tgt
                    elif rel_tgt == src:
                        target = rel_src

                    if target and target not in activated:
                        spread_weight = edge.weight * damping
                        node = self.concepts.get(target)
                        if node:
                            spread_weight *= node.effective_weight
                        activated[target] = spread_weight
                        next_frontier.append(target)
                        paths.append(f"{src} →[{rel_type}]→ {target} (w={spread_weight:.2f})")

            frontier = next_frontier

        # Rank by activation
        ranked = sorted(activated.items(), key=lambda x: -x[1])[:top_k]

        # Encode the thought as braille
        thought_braille = self._encode_thought(ranked)

        return {
            "activated_concepts": ranked,
            "paths": paths[:20],
            "braille": thought_braille,
            "seed": seed_concepts,
            "depth": depth,
        }

    def _encode_thought(self, ranked_concepts: List[Tuple[str, float]]) -> str:
        """Encode an activated concept set into braille cells.

        Uses the current weight distribution to assign codes — so the
        braille representation of a thought changes as the memory evolves.
        """
        # Build a mini codec from current concept weights
        all_weighted = sorted(
            self.concepts.items(),
            key=lambda x: x[1].effective_weight,
            reverse=True,
        )
        # Map top concepts to braille indices
        concept_to_idx = {}
        for i, (c, _) in enumerate(all_weighted[:254]):
            concept_to_idx[c] = i

        braille = ""
        for concept, _ in ranked_concepts:
            idx = concept_to_idx.get(concept)
            if idx is not None:
                braille += chr(BRAILLE_BASE + idx)
            else:
                # Literal: hash to braille cell
                h = sum(b for b in concept.encode("utf-8")) & 0xFF
                braille += chr(BRAILLE_BASE + h)

        return braille

    # ── Foundational knowledge: the system's own architecture thesis ─────

    def seed_architecture_thesis(self) -> dict:
        """Inject foundational concepts about the system's own architecture.

        These represent the theoretical basis for external memory as a model
        size reduction strategy.  They become persistent weighted nodes that
        the model can activate during think() — so it knows its own theory.

        Only seeds concepts that don't already exist (idempotent).
        """
        now = time.time()
        provider = "architecture-thesis"
        seeded_concepts = 0
        seeded_relations = 0

        # Foundational concept nodes
        thesis_concepts = {
            "meta": [
                "external memory", "parameter efficiency", "reasoning capacity",
                "knowledge storage", "concept graph", "activation spreading",
                "temporal decay", "interaction feedback", "epoch drift",
                "linear scaling", "braille encoding", "tiered codec",
                "continuous learning", "model size reduction",
                "structured retrieval", "weight distribution",
            ],
        }

        for category, concepts in thesis_concepts.items():
            for concept in concepts:
                c = concept.lower().strip()
                if c not in self.concepts:
                    node = ConceptNode()
                    node.weight = 2.0  # moderate foundational weight
                    node.frequency = 1
                    node.providers.add(provider)
                    node.last_seen = now
                    node.categories.add(category)
                    self.concepts[c] = node
                    seeded_concepts += 1

        # Foundational relation edges
        thesis_relations = [
            {"src": "external memory", "type": "REDUCES", "tgt": "parameter requirements"},
            {"src": "concept graph", "type": "REPLACES", "tgt": "knowledge storage"},
            {"src": "activation spreading", "type": "ENABLES", "tgt": "reasoning capacity"},
            {"src": "temporal decay", "type": "IMPROVES", "tgt": "parameter efficiency"},
            {"src": "interaction feedback", "type": "DRIVES", "tgt": "concept graph"},
            {"src": "braille encoding", "type": "COMPRESSES", "tgt": "knowledge storage"},
            {"src": "linear scaling", "type": "CONTRASTS", "tgt": "parameter scaling"},
            {"src": "continuous learning", "type": "ENABLES", "tgt": "model size reduction"},
            {"src": "structured retrieval", "type": "IMPROVES", "tgt": "parameter efficiency"},
            {"src": "epoch drift", "type": "MEASURES", "tgt": "continuous learning"},
        ]

        for rel in thesis_relations:
            src = rel["src"].lower().strip()
            rtype = rel["type"].upper().strip()
            tgt = rel["tgt"].lower().strip()
            key = f"{src}→{rtype}→{tgt}"

            if key not in self.relations:
                edge = RelationEdge()
                edge.weight = 2.0  # moderate foundational weight
                edge.count = 1
                edge.providers.add(provider)
                edge.last_seen = now
                self.relations[key] = edge
                seeded_relations += 1

                # Ensure target concepts exist too
                if tgt not in self.concepts:
                    node = ConceptNode()
                    node.weight = 1.5
                    node.frequency = 1
                    node.providers.add(provider)
                    node.last_seen = now
                    node.categories.add("meta")
                    self.concepts[tgt] = node
                    seeded_concepts += 1

        return {
            "seeded_concepts": seeded_concepts,
            "seeded_relations": seeded_relations,
            "total_concepts": len(self.concepts),
            "total_relations": len(self.relations),
        }

    # ── Query: what does the memory know right now? ──────────────────────

    def top_concepts(self, n: int = 20) -> List[Tuple[str, float]]:
        """Return the top N concepts by effective weight."""
        ranked = sorted(
            self.concepts.items(),
            key=lambda x: x[1].effective_weight,
            reverse=True,
        )
        return [(c, round(node.effective_weight, 4)) for c, node in ranked[:n]]

    def top_relations(self, n: int = 20) -> List[Tuple[str, float]]:
        """Return the top N relation edges by weight."""
        ranked = sorted(
            self.relations.items(),
            key=lambda x: x[1].weight,
            reverse=True,
        )
        return [(key, round(edge.weight, 4)) for key, edge in ranked[:n]]

    def age_seconds(self) -> float:
        return time.time() - self._birth_ts

    def summary(self) -> dict:
        """Quick snapshot of memory state."""
        top = self.top_concepts(10)
        return {
            "age_s": round(self.age_seconds()),
            "total_concepts": len(self.concepts),
            "total_relations": len(self.relations),
            "total_posts_seen": self._total_posts_seen,
            "total_interactions": self._total_interactions,
            "epochs": len(self.epochs),
            "drift": self.epochs[-1].drift_score if self.epochs else 0.0,
            "avg_bits": self.epochs[-1].avg_bits if self.epochs else 8.0,
            "top_10": top,
            "top_10_braille": self._encode_thought(top),
        }

    # ── Serialization ────────────────────────────────────────────────────

    def save(self) -> str:
        """Serialize the entire memory to a JSON string."""
        data = {
            "v": 1,
            "birth": self._birth_ts,
            "posts_seen": self._total_posts_seen,
            "interactions": self._total_interactions,
            "concepts": {c: n.to_dict() for c, n in self.concepts.items()},
            "relations": {k: e.to_dict() for k, e in self.relations.items()},
            "epochs": [
                {
                    "ts": e.timestamp,
                    "posts": e.n_posts,
                    "concepts": e.n_concepts_ingested,
                    "relations": e.n_relations_ingested,
                    "top": e.top_concepts[:10],
                    "drift": e.drift_score,
                    "avg_bits": e.avg_bits,
                }
                for e in self.epochs[-100:]  # keep last 100 epochs
            ],
        }
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def load(cls, data: str) -> "BrailleMemory":
        """Deserialize from a JSON string."""
        d = json.loads(data)
        mem = cls()
        mem._birth_ts = d.get("birth", time.time())
        mem._total_posts_seen = d.get("posts_seen", 0)
        mem._total_interactions = d.get("interactions", 0)

        for concept, nd in d.get("concepts", {}).items():
            mem.concepts[concept] = ConceptNode.from_dict(nd)

        for key, ed in d.get("relations", {}).items():
            mem.relations[key] = RelationEdge.from_dict(ed)

        for ep in d.get("epochs", []):
            mem.epochs.append(Epoch(
                timestamp=ep["ts"],
                n_posts=ep.get("posts", 0),
                n_concepts_ingested=ep.get("concepts", 0),
                n_relations_ingested=ep.get("relations", 0),
                top_concepts=ep.get("top", []),
                drift_score=ep.get("drift", 0.0),
                avg_bits=ep.get("avg_bits", 8.0),
            ))

        return mem

    def save_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.save())
        log.info("BrailleMemory saved to %s (%d concepts, %d relations)",
                 path, len(self.concepts), len(self.relations))

    @classmethod
    def load_from_file(cls, path: str) -> "BrailleMemory":
        with open(path, "r") as f:
            return cls.load(f.read())
