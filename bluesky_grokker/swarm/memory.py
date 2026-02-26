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
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

BRAILLE_BASE = 0x2800

# ── Non-semantic filter: tokens that should never occupy short tiers ──────────

# URL-like patterns, TLDs, boilerplate that compress well but carry no meaning
_NONSEMANTIC_EXACT = frozenset({
    "com", "www", "https", "http", "org", "net", "io", "co", "edu", "gov",
    "bsky", "bsky.social", "bsky.app", "twitter", "x.com",
    "please", "share", "follow", "like", "repost", "subscribe",
    "click", "link", "url", "amp", "ref", "utm",
    "it's", "i'm", "don't", "can't", "won't", "that's", "we're",
    "the", "and", "but", "for", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "are", "has", "his", "how",
    "its", "may", "new", "now", "old", "see", "way", "who", "did",
    "get", "let", "say", "she", "too", "use", "more", "just", "also",
    "right", "really", "thing", "things", "people", "someone", "something",
    "going", "getting", "want", "need", "think", "know", "make",
    "good", "bad", "much", "very", "still", "even", "back", "well",
    "yeah", "yes", "no", "ok", "okay", "lol", "lmao", "omg",
    "mixed", "discussing",  # cluster fallback filler
})

_NONSEMANTIC_PATTERNS = [
    re.compile(r"^https?://", re.I),
    re.compile(r"^www\.", re.I),
    re.compile(r"^@[a-z0-9]", re.I),      # @handles
    re.compile(r"\.(com|org|net|io|app|social|xyz)$", re.I),
    re.compile(r"^[a-f0-9]{8,}$"),          # hex hashes
    re.compile(r"^\d+$"),                    # pure numbers
]

# Tier penalty: nonsemantic concepts get their weight multiplied by this
# so they can still exist in memory but won't steal short codes
NONSEMANTIC_WEIGHT_PENALTY = 0.1


def is_nonsemantic(concept: str) -> bool:
    """Return True if a concept is non-semantic boilerplate."""
    c = concept.lower().strip()
    if c in _NONSEMANTIC_EXACT:
        return True
    if len(c) <= 2:
        return True
    for pat in _NONSEMANTIC_PATTERNS:
        if pat.search(c):
            return True
    return False


# ── Concept canonicalization ──────────────────────────────────────────────────

# Hand-built alias map: maps variant → canonical form
_ALIAS_MAP = {
    # Social media
    "social media": "social media",
    "social": "social media",
    "digital sharing": "social media",
    "sharing online": "social media",
    "social networking": "social media",
    "social network": "social media",
    # Politics
    "us politics": "american politics",
    "u.s. politics": "american politics",
    "american politics": "american politics",
    "usa politics": "american politics",
    "us government": "american politics",
    "uk economy": "british economy",
    "uk economy decline": "british economy",
    "british economy": "british economy",
    "uk politics": "british politics",
    "british politics": "british politics",
    # Technology
    "ai": "artificial intelligence",
    "a.i.": "artificial intelligence",
    "machine learning": "artificial intelligence",
    "ml": "artificial intelligence",
    "deep learning": "artificial intelligence",
    "tech": "technology",
    "tech industry": "technology",
    "technology industry": "technology",
    # Crypto
    "crypto": "cryptocurrency",
    "bitcoin": "cryptocurrency",
    "btc": "cryptocurrency",
    "ethereum": "cryptocurrency",
    "eth": "cryptocurrency",
    # Climate
    "climate change": "climate change",
    "global warming": "climate change",
    "climate crisis": "climate change",
    "climate emergency": "climate change",
    # Media / content
    "video game": "gaming",
    "video games": "gaming",
    "videogames": "gaming",
    "gaming": "gaming",
    "games": "gaming",
    "donate": "donations",
    "donation": "donations",
    "donations": "donations",
    "donating": "donations",
    # Emotions (collapse near-synonyms)
    "frustrated": "frustration",
    "frustrating": "frustration",
    "frustration": "frustration",
    "angry": "anger",
    "furious": "anger",
    "excited": "excitement",
    "exciting": "excitement",
    "excitement": "excitement",
    "happy": "happiness",
    "happiness": "happiness",
    "joyful": "happiness",
    # Economy
    "economy": "economy",
    "economic": "economy",
    "economics": "economy",
    "financial": "finance",
    "finance": "finance",
    "stock market": "finance",
    "stocks": "finance",
    # Healthcare
    "health care": "healthcare",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "health": "healthcare",
}

# Regex-based normalizations applied in order
_CANON_TRANSFORMS = [
    (re.compile(r"[''`]"), "'"),         # normalize quotes
    (re.compile(r"[""„]"), '"'),         # normalize double quotes
    (re.compile(r"\s+"), " "),           # collapse whitespace
    (re.compile(r"[^\w\s\-'./]"), ""),   # strip most punctuation
]


def canonicalize_concept(concept: str) -> str:
    """Normalize a concept string to its canonical form.

    1. lowercase + strip
    2. apply regex normalizations
    3. check alias map
    4. strip trailing 's' for simple depluralization (if >4 chars)
    """
    c = concept.lower().strip()
    for pattern, repl in _CANON_TRANSFORMS:
        c = pattern.sub(repl, c)
    c = c.strip()

    # Check alias map
    if c in _ALIAS_MAP:
        return _ALIAS_MAP[c]

    # Simple depluralization (avoids needing nltk)
    if len(c) > 4 and c.endswith("s") and not c.endswith("ss"):
        singular = c[:-1]
        if singular in _ALIAS_MAP:
            return _ALIAS_MAP[singular]

    return c


def find_nearest_memory_concept(concept: str, memory_concepts: Dict[str, Any],
                                 threshold: float = 0.8) -> Optional[str]:
    """Find the best matching existing memory concept for a new concept.

    Uses character-level similarity (Jaccard on character trigrams).
    Returns the matching memory concept key if similarity >= threshold,
    otherwise None.
    """
    if not memory_concepts:
        return None

    c = concept.lower().strip()
    if c in memory_concepts:
        return c

    # Build trigrams for the input
    c_trigrams = set()
    padded = f"  {c}  "
    for i in range(len(padded) - 2):
        c_trigrams.add(padded[i:i+3])

    if not c_trigrams:
        return None

    best_match = None
    best_sim = 0.0

    for mem_concept in memory_concepts:
        if abs(len(mem_concept) - len(c)) > max(len(c), len(mem_concept)) * 0.5:
            continue  # skip wildly different lengths

        m_padded = f"  {mem_concept}  "
        m_trigrams = set()
        for i in range(len(m_padded) - 2):
            m_trigrams.add(m_padded[i:i+3])

        if not m_trigrams:
            continue

        # Jaccard similarity on trigrams
        intersection = len(c_trigrams & m_trigrams)
        union = len(c_trigrams | m_trigrams)
        sim = intersection / union if union > 0 else 0.0

        if sim > best_sim and sim >= threshold:
            best_sim = sim
            best_match = mem_concept

    return best_match


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
    nonsemantic: bool = False  # stop-concept: excluded from codebook/seeds/drift
    tier_hint: int = -1  # last codec tier index (0-4), -1 = unassigned; for hysteresis

    @property
    def effective_weight(self) -> float:
        return self.weight + self.interaction_boosts

    @property
    def semantic_weight(self) -> float:
        """Weight for codebook/consensus purposes — zero if nonsemantic."""
        if self.nonsemantic:
            return 0.0
        return self.weight + self.interaction_boosts

    def to_dict(self) -> dict:
        d = {
            "w": round(self.weight, 4),
            "f": self.frequency,
            "p": list(self.providers),
            "ts": round(self.last_seen, 2),
            "cat": list(self.categories),
            "ib": round(self.interaction_boosts, 4),
        }
        if self.nonsemantic:
            d["ns"] = True
        if self.tier_hint >= 0:
            d["th"] = self.tier_hint
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptNode":
        return cls(
            weight=d.get("w", 0.0),
            frequency=d.get("f", 0),
            providers=set(d.get("p", [])),
            last_seen=d.get("ts", 0.0),
            categories=set(d.get("cat", [])),
            interaction_boosts=d.get("ib", 0.0),
            nonsemantic=d.get("ns", False),
            tier_hint=d.get("th", -1),
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

        n_filtered = 0
        n_canonicalized = 0

        # Ingest concepts
        for category, concept_list in concepts_by_category.items():
            for concept in concept_list:
                if not concept:
                    continue

                # Canonicalize: normalize + alias merge
                canonical = canonicalize_concept(concept)
                if canonical != concept.lower().strip():
                    n_canonicalized += 1

                # Memory-space mapping: match to existing memory concept if close
                mem_match = find_nearest_memory_concept(canonical, self.concepts)
                if mem_match is not None and mem_match != canonical:
                    canonical = mem_match
                    n_canonicalized += 1

                # Non-semantic filter: penalize weight AND mark the node
                signal = self.SIGNAL_POST
                _is_nonsem = is_nonsemantic(canonical)
                if _is_nonsem:
                    signal *= NONSEMANTIC_WEIGHT_PENALTY
                    n_filtered += 1

                node = self.concepts.get(canonical)
                if node is None:
                    node = ConceptNode()
                    self.concepts[canonical] = node

                node.frequency += 1
                node.weight += signal
                node.providers.add(provider)
                node.last_seen = now
                node.categories.add(category)
                if _is_nonsem:
                    node.nonsemantic = True
                n_concepts += 1

        # Ingest relations (with canonicalized endpoints)
        # Skip relations where either endpoint is nonsemantic
        for rel in relations:
            src = canonicalize_concept(rel.get("src") or "")
            rtype = (rel.get("type") or "").upper().strip()
            tgt = canonicalize_concept(rel.get("tgt") or "")
            if not (src and rtype and tgt):
                continue
            if is_nonsemantic(src) or is_nonsemantic(tgt):
                continue

            # Map to memory-space concepts
            src_match = find_nearest_memory_concept(src, self.concepts)
            if src_match:
                src = src_match
            tgt_match = find_nearest_memory_concept(tgt, self.concepts)
            if tgt_match:
                tgt = tgt_match

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

        return {
            "concepts_ingested": n_concepts,
            "relations_ingested": n_relations,
            "nonsemantic_penalized": n_filtered,
            "canonicalized": n_canonicalized,
        }

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

        # Rank concepts by effective weight — exclude nonsemantic from top-k
        ranked = sorted(
            self.concepts.items(),
            key=lambda x: x[1].effective_weight,
            reverse=True,
        )
        # Drift top-k uses only semantic concepts
        semantic_ranked = [(c, n) for c, n in ranked if not n.nonsemantic]
        top_concepts = [c for c, _ in semantic_ranked[:20]]

        # Drift score: how much did the top-20 change since last epoch?
        drift = 0.0
        if self.epochs:
            prev_top = set(self.epochs[-1].top_concepts)
            curr_top = set(top_concepts)
            if prev_top or curr_top:
                drift = 1.0 - len(prev_top & curr_top) / max(len(prev_top | curr_top), 1)

        # Compute average bits/concept from semantic weight distribution only
        avg_bits = self._estimate_avg_bits(semantic_ranked)

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
        """Estimate expected bits per concept *reference* under the tiered codec.

        This weights by usage frequency (effective_weight), not per unique
        concept.  E[L] = Σ P(c) · L(c), where P(c) = w(c) / Σw.
        This correctly reflects real payload cost.
        """
        # Tier layout mirrors TieredConceptCodec
        tiers = [(3, 2), (4, 4), (5, 8), (7, 16)]  # (bits, slots)
        long_bits = 11

        total_weight = sum(n.effective_weight for _, n in ranked)
        if total_weight == 0:
            return 8.0

        # Assign tier bits to concepts in rank order
        concept_bits: list[tuple[float, int]] = []  # (weight, bits)
        idx = 0
        for bits, slots in tiers:
            for _ in range(slots):
                if idx >= len(ranked):
                    break
                concept_bits.append((ranked[idx][1].effective_weight, bits))
                idx += 1

        # Remaining concepts at long tier
        for i in range(idx, len(ranked)):
            concept_bits.append((ranked[i][1].effective_weight, long_bits))

        # E[L] = Σ P(c) · L(c)
        avg = sum(w * b for w, b in concept_bits) / total_weight
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

    # ── Motif detection: repeated subgraph patterns ──────────────────────

    def detect_motifs(self, min_count: int = 2, top_k: int = 20) -> List[Dict]:
        """Detect repeated structural patterns in the relation graph.

        Motifs are reusable subgraph patterns that appear frequently:
        - Star motifs: a hub concept with multiple edges of the same type
        - Relation-type patterns: (type, degree) pairs that recur
        - Chain motifs: A→B→C paths that repeat

        Returns a list of motifs sorted by frequency, each with:
        - pattern: string description of the motif
        - count: how many times it appears
        - hub: the central concept (for star motifs)
        - edges: the relation keys involved
        """
        motifs: List[Dict] = []

        # --- Star motifs: concepts with N+ edges of the same type ---
        # Group edges by (src, type) and (tgt, type)
        src_type_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        tgt_type_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for key, edge in self.relations.items():
            parts = key.split("→")
            if len(parts) != 3:
                continue
            src, rtype, tgt = parts
            src_type_groups[(src, rtype)].append(key)
            tgt_type_groups[(tgt, rtype)].append(key)

        for (hub, rtype), edges in src_type_groups.items():
            if len(edges) >= min_count:
                total_weight = sum(
                    self.relations[k].weight for k in edges if k in self.relations
                )
                motifs.append({
                    "pattern": f"star:{hub}→[{rtype}]→*",
                    "type": "star_out",
                    "count": len(edges),
                    "hub": hub,
                    "relation_type": rtype,
                    "edges": edges,
                    "total_weight": total_weight,
                })

        for (hub, rtype), edges in tgt_type_groups.items():
            if len(edges) >= min_count:
                total_weight = sum(
                    self.relations[k].weight for k in edges if k in self.relations
                )
                motifs.append({
                    "pattern": f"star:*→[{rtype}]→{hub}",
                    "type": "star_in",
                    "count": len(edges),
                    "hub": hub,
                    "relation_type": rtype,
                    "edges": edges,
                    "total_weight": total_weight,
                })

        # --- Relation-type frequency (global structural pattern) ---
        rtype_counts: Counter = Counter()
        for key in self.relations:
            parts = key.split("→")
            if len(parts) == 3:
                rtype_counts[parts[1]] += 1

        for rtype, count in rtype_counts.most_common(10):
            if count >= min_count:
                motifs.append({
                    "pattern": f"rtype:{rtype}",
                    "type": "relation_type",
                    "count": count,
                    "hub": None,
                    "relation_type": rtype,
                    "edges": [],
                    "total_weight": 0,
                })

        # Sort by count × weight for star motifs, count for type motifs
        motifs.sort(key=lambda m: (m["count"] * (m.get("total_weight", 0) + 1)), reverse=True)
        return motifs[:top_k]

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

        # Seed activation — skip nonsemantic concepts
        for concept in seed_concepts:
            c = concept.lower().strip()
            node = self.concepts.get(c)
            if node and node.nonsemantic:
                continue  # stop-concepts don't seed activation
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

    # ── Stop-concept housekeeping ───────────────────────────────────────

    def mark_nonsemantic_concepts(self) -> int:
        """Retroactively scan and mark all nonsemantic concepts.

        Called on load to clean up legacy data from before the filter existed.
        Returns number of concepts newly marked.
        """
        marked = 0
        for concept, node in self.concepts.items():
            if not node.nonsemantic and is_nonsemantic(concept):
                node.nonsemantic = True
                marked += 1
        return marked

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

    def top_concepts(self, n: int = 20, include_nonsemantic: bool = False) -> List[Tuple[str, float]]:
        """Return the top N concepts by effective weight (semantic only by default)."""
        ranked = sorted(
            ((c, node) for c, node in self.concepts.items()
             if include_nonsemantic or not node.nonsemantic),
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
