"""Modal-deployed Bluesky Firehose → Braille-Swarm-Consensus → Post pipeline.

Usage:
    # One-shot run (default 20s firehose window):
    modal run bluesky_grokker/modal_app.py
    modal run bluesky_grokker/modal_app.py --seconds 30 --dry-run

    # Continuous loop (stays alive, sleeps between runs):
    modal run bluesky_grokker/modal_app.py --loop
    modal run bluesky_grokker/modal_app.py --loop --interval 600 --max-loops 24

    # 24/7 cron deployment (~$8-12/month):
    modal deploy bluesky_grokker/modal_app.py
    #   → Runs every 15 min, 10s firehose, skips 2-8am UTC
    #   → No idle costs (cold start each invocation)
    #   → Stop with: modal app stop bluesky-grokker-swarm

Requires:
    pip install modal
    modal token new          # one-time auth
    modal secret create bluesky-grokker \\
        OPENROUTER_API_KEY=sk-or-v1-... \\
        BLUESKY_HANDLE=yourhandle \\
        BLUESKY_PASSWORD=your-app-password
"""

from __future__ import annotations

import modal

# ── Modal image with all dependencies ─────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "atproto>=0.0.46",
        "openai>=1.12.0",
        "anthropic>=0.39.0",
        "google-generativeai>=0.8.0",
        "httpx>=0.27.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.0",
    )
    .add_local_dir("bluesky_grokker/swarm", remote_path="/root/swarm")
)

# Persistent volume for BrailleMemory state (survives across runs)
memory_volume = modal.Volume.from_name("braille-memory", create_if_missing=True)
MEMORY_PATH = "/data/braille_memory.json"

app = modal.App(
    name="bluesky-grokker-swarm",
    image=image,
    secrets=[modal.Secret.from_name("bluesky-grokker")],
)


# ── Firehose capture ─────────────────────────────────────────────────────────


@app.function(timeout=300)
def capture_firehose(seconds: int = 30) -> dict:
    """Connect to the Bluesky firehose and collect posts + interactions.

    Returns {"posts": [...], "interactions": [...]} where interactions are
    likes, reposts, and replies — the signals that feed BrailleMemory.
    """
    import threading
    import time
    from datetime import datetime, timezone

    from atproto import (
        CAR,
        FirehoseSubscribeReposClient,
        models,
        parse_subscribe_repos_message,
    )

    posts: list[dict] = []
    interactions: list[dict] = []
    lock = threading.Lock()
    client = FirehoseSubscribeReposClient()
    deadline = time.monotonic() + seconds

    def on_message(message):
        if time.monotonic() >= deadline:
            client.stop()
            return

        commit = parse_subscribe_repos_message(message)
        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            return
        if not commit.blocks:
            return

        try:
            car = CAR.from_bytes(commit.blocks)
        except Exception:
            return

        for op in commit.ops:
            if op.action != "create" or not op.cid:
                continue
            raw = car.blocks.get(op.cid)
            if raw is None:
                continue

            record_type = raw.get("$type", "")

            if record_type == "app.bsky.feed.post":
                text = raw.get("text", "").strip()
                if not text or len(text) < 10:
                    continue
                langs = raw.get("langs") or []
                if langs and "en" not in langs:
                    continue

                created_str = raw.get("createdAt", "")
                try:
                    created_at = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    ).isoformat()
                except Exception:
                    created_at = datetime.now(timezone.utc).isoformat()

                is_reply = bool(raw.get("reply"))

                with lock:
                    posts.append({
                        "uri": f"at://{commit.repo}/{op.path}",
                        "author_did": commit.repo,
                        "text": text[:500],
                        "created_at": created_at,
                        "is_reply": is_reply,
                    })
                    if is_reply:
                        interactions.append({
                            "type": "reply",
                            "author_did": commit.repo,
                            "target_uri": (raw.get("reply", {})
                                           .get("parent", {})
                                           .get("uri", "")),
                            "text": text[:200],
                        })

            elif record_type == "app.bsky.feed.like":
                subject = raw.get("subject", {})
                with lock:
                    interactions.append({
                        "type": "like",
                        "author_did": commit.repo,
                        "target_uri": subject.get("uri", "") if isinstance(subject, dict) else "",
                    })

            elif record_type == "app.bsky.feed.repost":
                subject = raw.get("subject", {})
                with lock:
                    interactions.append({
                        "type": "repost",
                        "author_did": commit.repo,
                        "target_uri": subject.get("uri", "") if isinstance(subject, dict) else "",
                    })

    print(f"🔥 Capturing firehose for {seconds}s …")
    try:
        client.start(on_message)
    except Exception:
        pass  # client.stop() causes a benign exception

    print(f"📦 Captured {len(posts)} posts, {len(interactions)} interactions "
          f"({sum(1 for i in interactions if i['type']=='like')} likes, "
          f"{sum(1 for i in interactions if i['type']=='repost')} reposts, "
          f"{sum(1 for i in interactions if i['type']=='reply')} replies)")
    return {"posts": posts, "interactions": interactions}


# ── Semantic clustering (lightweight, no DB needed) ──────────────────────────


@app.function(timeout=120)
def cluster_posts(posts: list[dict], max_clusters: int = 8) -> list[dict]:
    """Group posts into thematic clusters using simple TF-IDF-like scoring."""
    import re
    from collections import Counter

    if not posts:
        return []

    STOP_WORDS = frozenset(
        "the a an is are was were be been being have has had do does did "
        "will would shall should may might can could i me my we our you your "
        "he she it they them their its this that these those and but or nor "
        "for so yet at by from in into of on to with as if then than too very "
        "not no just about also back been before being between both but came "
        "come could day did even get got has have her here him his how its "
        "like made make many most much must new now old only other our out "
        "over said same see she some still such take tell than that the their "
        "them then there these they thing this those through time up us use "
        "want was way well were what when which who will with would year rt "
        "dont im ive thats youre weve theyre dont cant wont hes shes".split()
    )

    def tokenise(text: str) -> list[str]:
        return [
            w
            for w in re.findall(r"[a-z']+", text.lower())
            if len(w) > 2 and w not in STOP_WORDS
        ]

    # Build corpus-wide term frequencies
    doc_tokens = [tokenise(p["text"]) for p in posts]
    doc_freq: Counter = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    # Find top terms (the "topics")
    n_docs = len(posts)
    import math

    scored_terms: list[tuple[str, float]] = []
    for term, df in doc_freq.items():
        if df < 3 or df > n_docs * 0.5:
            continue
        idf = math.log(n_docs / df)
        scored_terms.append((term, df * idf))

    scored_terms.sort(key=lambda x: -x[1])
    top_terms = [t[0] for t in scored_terms[: max_clusters * 3]]

    if not top_terms:
        # Fallback: just return a single cluster with a random sample
        sample = posts[: min(20, len(posts))]
        return [
            {
                "topic": "general discourse",
                "post_count": len(posts),
                "sample_texts": [p["text"][:200] for p in sample],
            }
        ]

    # Assign each post to its best-matching topic term
    clusters: dict[str, list[dict]] = {t: [] for t in top_terms[:max_clusters]}
    for i, post in enumerate(posts):
        tokens = set(doc_tokens[i])
        best_topic = None
        best_score = 0
        for topic in clusters:
            if topic in tokens:
                score = doc_freq[topic]
                if score > best_score:
                    best_score = score
                    best_topic = topic
        if best_topic:
            clusters[best_topic].append(post)

    # Build cluster summaries
    result = []
    for topic, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if not members:
            continue
        result.append(
            {
                "topic": topic,
                "post_count": len(members),
                "sample_texts": [p["text"][:200] for p in members[:5]],
            }
        )

    return result[:max_clusters]


# ── Z₂⁸ Braille MOTL Encoding Primitives ─────────────────────────────────────
#
# 8-dot braille: each cell is a vector in Z₂⁸ (8 binary dimensions).
# Unicode braille block: U+2800 to U+28FF (256 codepoints = 2^8 exactly).
#
# TIERED VARIABLE-WIDTH ENCODING (ported from SemanticCodec):
#   Tier 0 : 1-bit codes → top 2 concepts
#   Tier 1 : 2-bit codes → next 4 concepts
#   Tier 2 : 3-bit codes → next 8 concepts
#   Tier 3 : 4-bit codes → next 16 concepts
#   Tier 4 : 8-bit codes → remaining concepts (up to 256)
#   Literal : escape marker + UTF-8 bytes for unknown concepts
#
# Concepts are packed as a continuous bitstream, then chunked into 8-bit braille
# cells.  This achieves Huffman-like compression: frequent concepts use fewer bits.
#
# SPARSE RELATIONAL ENCODING:
#   Relations are encoded as fixed 3-cell tuples: (subject, relation, object)
#   prefixed by a RELATION_MARKER cell (⣾ = 0xFE).  This enables:
#   - Graph-native storage (every 3 cells after a marker = one edge)
#   - Constant-time relation lookup (seek to offset i*4)
#   - Set operations on concept graphs (intersection = shared edges)

BRAILLE_BASE = 0x2800
MARKER_LITERAL = 0xFF   # ⣿ — escape to literal UTF-8 bytes
MARKER_REL     = 0xFE   # ⣾ — next 3 cells are a (src, rel, tgt) tuple
MARKER_CAT_SEP = 0xFD   # ⣽ — category separator
MARKER_MOTIF   = 0xFC   # ⣼ — motif reference: motif_id + N binding cells

# Tier layout: (bit_depth, max_slots)
TIERS = [(1, 2), (2, 4), (3, 8), (4, 16), (8, 256)]


def _bits_to_braille(byte_val: int) -> str:
    """Convert an 8-bit integer (0-255) to its Unicode braille character."""
    return chr(BRAILLE_BASE + (byte_val & 0xFF))


def _braille_to_bits(ch: str) -> int:
    """Convert a Unicode braille character back to its 8-bit integer."""
    return ord(ch) - BRAILLE_BASE


class TieredConceptCodec:
    """Variable-width concept encoder using frequency-ranked tiered assignment.

    Encoding scheme (Elias-escape hybrid):
      Frequent concepts (top 30) get short prefix-free codes:
        Tier 0: "00" + 1-bit  = 3 bits  (top 2 concepts)
        Tier 1: "01" + 2-bit  = 4 bits  (next 4 concepts)
        Tier 2: "10" + 3-bit  = 5 bits  (next 8 concepts)
        Tier 3: "110"+ 4-bit  = 7 bits  (next 16 concepts)
      Rare concepts (rank 30+) use escape:
        "111" + 8-bit index   = 11 bits  (up to 256 more concepts)
      Unknown concepts (not in table):
        "11111111" + UTF-8 bytes + "11111110" (literal escape)

    The top-30 concepts (which by Zipf's law carry ~80% of frequency mass)
    cost 3-7 bits.  Rare concepts cost 11 bits (only 3-bit overhead vs flat 8).
    Weighted average is typically 4-6 bits/concept vs 8-bit flat.
    """

    # (prefix, payload_bits, slot_count)
    SHORT_TIERS = [
        ("00",  1,  2),   # 3 bits total → top 2
        ("01",  2,  4),   # 4 bits total → next 4
        ("10",  3,  8),   # 5 bits total → next 8
        ("110", 4, 16),   # 7 bits total → next 16
    ]
    LONG_PREFIX = "111"   # 11 bits total → rest (up to 256)
    LONG_BITS = 8

    # Tier boundaries: cumulative slot counts [2, 6, 14, 30]
    _TIER_BOUNDARIES = [2, 6, 14, 30]

    def __init__(self, all_concepts: list[str],
                 tier_hints: dict[str, int] | None = None) -> None:
        """Build codec from concept frequency, with optional hysteresis.

        tier_hints: {concept → previous_tier_index (0-4)} from BrailleMemory.
        Concepts already in a short tier get a ranking boost to resist
        demotion, reducing codebook churn across epochs.
        """
        from collections import Counter

        freq = Counter(all_concepts)

        # Apply hysteresis: boost frequency of concepts that were in short tiers
        # so they resist demotion unless significantly outranked
        if tier_hints:
            boosted_freq = {}
            for concept, count in freq.items():
                prev_tier = tier_hints.get(concept, -1)
                if 0 <= prev_tier <= 3:
                    # Boost: concepts in shorter tiers get stronger retention
                    boost = (4 - prev_tier) * 0.5  # tier 0 → +2.0, tier 3 → +0.5
                    boosted_freq[concept] = count + boost
                else:
                    boosted_freq[concept] = count
            ranked = sorted(boosted_freq, key=lambda c: boosted_freq[c], reverse=True)
        else:
            ranked = [c for c, _ in freq.most_common()]

        self.encode_table: dict[str, str] = {}   # concept → bit string
        self.decode_table: dict[str, str] = {}   # bit string → concept
        self.concept_tier: dict[str, str] = {}    # concept → tier label
        self.concept_tier_idx: dict[str, int] = {}  # concept → tier index (0-4)

        idx = 0
        tier_idx = 0
        # Assign short tiers
        for prefix, payload_bits, slot_count in self.SHORT_TIERS:
            n = min(slot_count, len(ranked) - idx)
            if n <= 0:
                break
            for i in range(n):
                concept = ranked[idx]
                code = prefix + format(i, f"0{payload_bits}b")
                self.encode_table[concept] = code
                self.decode_table[code] = concept
                self.concept_tier[concept] = f"{len(code)}b-short"
                self.concept_tier_idx[concept] = tier_idx
                idx += 1
            tier_idx += 1

        # Assign long tier (escape prefix + 8-bit index)
        long_start = idx
        n_long = min(256, len(ranked) - idx)
        for i in range(n_long):
            concept = ranked[idx]
            code = self.LONG_PREFIX + format(i, f"0{self.LONG_BITS}b")
            self.encode_table[concept] = code
            self.decode_table[code] = concept
            self.concept_tier[concept] = "11b-long"
            self.concept_tier_idx[concept] = 4  # long tier
            idx += 1

        self.ranked = ranked
        self.concept_freq = freq
        self._total_concepts = idx
        self._n_short = long_start
        self._n_long = n_long

    def encode_concept(self, concept: str) -> str:
        """Encode a concept to its variable-width bit string."""
        return self.encode_table.get(concept, "")

    def encode_concept_with_literal(self, concept: str) -> str:
        """Encode with literal fallback for unknown concepts."""
        bits = self.encode_concept(concept)
        if bits:
            return bits
        # Literal: "11111111" + 8 bits per UTF-8 byte + "11111110" terminator
        literal_bits = "".join(format(b, "08b") for b in concept.encode("utf-8"))
        return "11111111" + literal_bits + "11111110"

    def decode_bitstream(self, bitstream: str) -> list[str]:
        """Decode a variable-width bitstream back to concept tokens."""
        concepts = []
        pos = 0
        blen = len(bitstream)
        while pos < blen:
            # Literal escape: 8 ones
            if bitstream[pos:pos + 8] == "11111111":
                pos += 8
                raw_bytes = bytearray()
                while pos + 8 <= blen:
                    byte_bits = bitstream[pos:pos + 8]
                    if byte_bits == "11111110":
                        pos += 8
                        break
                    raw_bytes.append(int(byte_bits, 2))
                    pos += 8
                try:
                    concepts.append(raw_bytes.decode("utf-8"))
                except Exception:
                    concepts.append(f"<unk:{raw_bytes.hex()}>")
                continue

            # Try short tiers first (they don't start with "111")
            decoded = False
            for prefix, payload_bits, _ in self.SHORT_TIERS:
                plen = len(prefix)
                total = plen + payload_bits
                if pos + total <= blen and bitstream[pos:pos + plen] == prefix:
                    full_code = bitstream[pos:pos + total]
                    if full_code in self.decode_table:
                        concepts.append(self.decode_table[full_code])
                        pos += total
                        decoded = True
                        break

            if decoded:
                continue

            # Try long tier: "111" + 8 bits
            lp = len(self.LONG_PREFIX)
            total_long = lp + self.LONG_BITS
            if (pos + total_long <= blen and
                    bitstream[pos:pos + lp] == self.LONG_PREFIX):
                full_code = bitstream[pos:pos + total_long]
                if full_code in self.decode_table:
                    concepts.append(self.decode_table[full_code])
                    pos += total_long
                    continue

            pos += 1  # skip unknown bit

        return concepts

    def get_stats(self) -> dict:
        tier_counts: dict[str, int] = {}
        for label in self.concept_tier.values():
            tier_counts[label] = tier_counts.get(label, 0) + 1
        avg_bits = 0.0
        total_freq = sum(self.concept_freq.values())
        if total_freq > 0:
            for concept, code in self.encode_table.items():
                avg_bits += len(code) * self.concept_freq[concept] / total_freq
        return {
            "total_encoded": self._total_concepts,
            "short_tier_concepts": self._n_short,
            "long_tier_concepts": self._n_long,
            "tier_distribution": tier_counts,
            "avg_bits_per_concept": round(avg_bits, 2),
            "flat_8bit_equivalent": 8.0,
            "variable_width_savings": round((1 - avg_bits / 8.0) * 100, 1) if avg_bits > 0 else 0,
        }


def _pack_bitstream_to_braille(bitstream: str) -> str:
    """Pack a bit string into braille cells (8 bits per cell, zero-padded).

    First 2 cells = 16-bit big-endian length header (actual bit count).
    Remaining cells = the bitstream data, zero-padded to a cell boundary.
    """
    bit_len = len(bitstream)
    # 2-cell header: high byte, low byte of bit_len (max 65535 bits)
    header = _bits_to_braille((bit_len >> 8) & 0xFF) + _bits_to_braille(bit_len & 0xFF)
    # Pad data to multiple of 8
    padded = bitstream + "0" * ((8 - bit_len % 8) % 8)
    data = "".join(
        chr(BRAILLE_BASE + int(padded[i:i + 8], 2))
        for i in range(0, len(padded), 8)
    )
    return header + data


def _unpack_braille_to_bitstream(braille: str) -> str:
    """Unpack braille cells back into a bit string, using the length header.

    Returns exactly the number of bits that were originally packed.
    """
    if len(braille) < 2:
        return ""
    # Read 2-cell header
    hi = ord(braille[0]) - BRAILLE_BASE
    lo = ord(braille[1]) - BRAILLE_BASE
    bit_len = (hi << 8) | lo
    # Decode remaining cells
    raw = "".join(format(ord(ch) - BRAILLE_BASE, "08b") for ch in braille[2:])
    return raw[:bit_len]


def _encode_relation_tuple(rel: dict, codec: TieredConceptCodec) -> str:
    """Encode a relation as a 3-cell (src, rel_type, tgt) braille tuple.

    Format: MARKER_REL cell + 3 concept cells.
    Each concept is encoded at its tiered variable-width, then packed into
    exactly 1 braille cell (8 bits, zero-padded right).  This gives us
    fixed-stride graph storage: every 4 cells = one edge.
    """
    src = rel.get("src", "")
    rel_type = rel.get("type", "")
    tgt = rel.get("tgt", "")

    def _concept_to_cell(concept: str) -> str:
        bits = codec.encode_concept(concept)
        if not bits:
            # Unknown concept: hash into 8 bits
            h = sum(b for b in concept.encode("utf-8")) & 0xFF
            return _bits_to_braille(h)
        # Pad/truncate to 8 bits for fixed-width cell
        bits_padded = (bits + "00000000")[:8]
        return chr(BRAILLE_BASE + int(bits_padded, 2))

    marker = _bits_to_braille(MARKER_REL)
    return marker + _concept_to_cell(src) + _concept_to_cell(rel_type) + _concept_to_cell(tgt)


def _encode_motif(motif: dict, codec: TieredConceptCodec, motif_id: int) -> str:
    """Encode a star motif as a compact braille sequence.

    Format: MARKER_MOTIF + motif_id_cell + hub_cell + n_bindings_cell + binding_cells
    A star motif "hub→[TYPE]→{a,b,c}" becomes 4+N cells instead of 4*N cells.
    Savings: (4*N) - (4+N) = 3*N - 4 cells saved (profitable when N >= 2).
    """
    hub = motif.get("hub", "")
    edges = motif.get("edges", [])
    if not hub or len(edges) < 2:
        return ""

    def _concept_to_cell(concept: str) -> str:
        bits = codec.encode_concept(concept)
        if not bits:
            h = sum(b for b in concept.encode("utf-8")) & 0xFF
            return _bits_to_braille(h)
        bits_padded = (bits + "00000000")[:8]
        return chr(BRAILLE_BASE + int(bits_padded, 2))

    marker = _bits_to_braille(MARKER_MOTIF)
    id_cell = _bits_to_braille(motif_id & 0xFF)
    hub_cell = _concept_to_cell(hub)
    n_cell = _bits_to_braille(len(edges) & 0xFF)

    # Binding cells: the varying endpoints of the star
    binding_cells = ""
    for edge_key in edges:
        parts = edge_key.split("→")
        if len(parts) != 3:
            continue
        src, rtype, tgt = parts
        # For star_out: hub is src, bindings are tgt
        # For star_in: hub is tgt, bindings are src
        binding = tgt if src == hub else src
        binding_cells += _concept_to_cell(binding)

    return marker + id_cell + hub_cell + n_cell + binding_cells


def _decode_relation_tuples(braille: str, codec: TieredConceptCodec) -> list[dict]:
    """Extract (src, type, tgt) relation tuples from a braille stream.

    Scans for MARKER_REL cells and reads the next 3 cells as a tuple.
    """
    relations = []
    i = 0
    # Build a reverse lookup: 8-bit padded code → concept
    cell_to_concept = {}
    for concept, bits in codec.encode_table.items():
        bits_padded = (bits + "00000000")[:8]
        cell_val = int(bits_padded, 2)
        cell_to_concept[cell_val] = concept

    while i < len(braille):
        if _braille_to_bits(braille[i]) == MARKER_REL and i + 3 < len(braille):
            cells = [_braille_to_bits(braille[i + 1 + j]) for j in range(3)]
            src = cell_to_concept.get(cells[0], f"<{cells[0]:02x}>")
            rel_type = cell_to_concept.get(cells[1], f"<{cells[1]:02x}>")
            tgt = cell_to_concept.get(cells[2], f"<{cells[2]:02x}>")
            relations.append({"src": src, "type": rel_type, "tgt": tgt})
            i += 4  # skip marker + 3 cells
        else:
            i += 1
    return relations


# ── Swarm distillation on Modal (real MOTL/Z₂⁸ pipeline) ────────────────────

# Model roster — free-tier OpenRouter models (verified live 2026-02-26)
# Spread across providers to avoid single-provider rate limits
SWARM_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",       # most reliable
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "qwen/qwen3-4b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "stepfun/step-3.5-flash:free",
    "upstage/solar-pro-3:free",
    "z-ai/glm-4.5-air:free",
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-oss-20b:free",
]

# Decoder fallback chain (tried in order until one responds)
DECODER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


@app.function(timeout=300)
def swarm_distill(
    clusters: list[dict],
    window_seconds: int = 30,
    memory_state: str = "",
) -> dict:
    """Braille-swarm-consensus pipeline with persistent BrailleMemory.

    Phase 1 — Fan out to N models: each extracts structured concept tokens
    Phase 2 — Encode all concepts into Z₂⁸ braille vector space
    Phase 2b— Feed extractions into BrailleMemory (the living weight model)
    Phase 3 — Merge/distill in braille-encoded space (multi-provider boosting)
    Phase 3b— BrailleMemory thinks: activation spreading from consensus seeds
    Phase 4 — Decode consensus concept graph → English observation

    Returns {"text": ..., "models": [...], "braille": ..., "motl_stats": ...,
             "memory_state": ..., "memory_summary": ...}.
    """
    import json
    import os
    import sys
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import openai

    # Import BrailleMemory directly (bypass swarm/__init__.py which has local deps)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("braille_memory", "/root/swarm/memory.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["braille_memory"] = _mod  # register so dataclass can resolve
    _spec.loader.exec_module(_mod)
    BrailleMemory = _mod.BrailleMemory

    # Load or create memory
    if memory_state:
        try:
            memory = BrailleMemory.load(memory_state)
            print(f"  🧠 Memory loaded: {len(memory.concepts)} concepts, "
                  f"{len(memory.relations)} relations, {len(memory.epochs)} epochs")
        except Exception as e:
            print(f"  ⚠️ Memory load failed: {e}, starting fresh")
            memory = BrailleMemory()
    else:
        memory = BrailleMemory()
        print("  🧠 Fresh BrailleMemory initialized")

    # Seed foundational architecture thesis (idempotent)
    thesis_result = memory.seed_architecture_thesis()
    if thesis_result["seeded_concepts"] > 0:
        print(f"  🧬 Seeded architecture thesis: {thesis_result['seeded_concepts']} concepts, "
              f"{thesis_result['seeded_relations']} relations")

    # Retroactively mark legacy nonsemantic concepts (www, https, com, etc.)
    n_marked = memory.mark_nonsemantic_concepts()
    if n_marked > 0:
        print(f"  🚫 Marked {n_marked} legacy concepts as nonsemantic")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in Modal secret")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # ── Build the discourse summary for prompts ──
    cluster_text = ""
    total_posts = 0
    for c in clusters:
        total_posts += c["post_count"]
        samples = "\n".join(f'  - "{t}"' for t in c["sample_texts"][:3])
        cluster_text += (
            f"\nTopic: {c['topic']} ({c['post_count']} posts)\n"
            f"Samples:\n{samples}\n"
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: Semantic extraction — each model outputs structured concept tokens
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    extraction_system = (
        "You are a semantic concept extractor for the MOTL protocol "
        "(Machine-Optimized Thought Language). Your job is to analyze social media "
        "discourse and output ONLY a JSON object with structured concept tokens.\n\n"
        "Output format (strict JSON, nothing else):\n"
        "{\n"
        '  "topics": ["topic1", "topic2", ...],\n'
        '  "entities": ["entity1", "entity2", ...],\n'
        '  "sentiments": ["sentiment1", "sentiment2", ...],\n'
        '  "actions": ["action1", "action2", ...],\n'
        '  "relations": [{"src": "A", "type": "CAUSES|CONTRASTS|SUPPORTS", "tgt": "B"}, ...],\n'
        '  "insight": "one-sentence pattern you detect"\n'
        "}\n\n"
        "Extract 5-15 concepts per category. Be specific, not generic. "
        "Use lowercase concept tokens. Output ONLY valid JSON."
    )

    extraction_user = (
        f"Analyze this {window_seconds}s Bluesky firehose snapshot "
        f"({total_posts} posts):\n{cluster_text}"
    )

    print(f"\n  ⚡ PHASE 1: Semantic extraction across {len(SWARM_MODELS)} models …")
    t_phase1 = time.monotonic()

    model_extractions: list[dict] = []

    def _extract_json(raw: str) -> dict | None:
        """Try hard to extract a JSON object from model output."""
        import re
        text = raw.strip()
        # Strip <think>...</think> blocks (qwen3, deepseek)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip markdown fences
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        pass
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Find first { ... last }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        # Truncation repair: model output cut off mid-JSON
        if start >= 0:
            fragment = text[start:]
            # Close any open strings, arrays, and the object
            # Strip trailing partial tokens
            for trail in ['"', '",', ', "', ',']:
                if fragment.rstrip().endswith(trail):
                    fragment = fragment.rstrip()[:-len(trail)]
            # Try closing open brackets
            open_brackets = fragment.count('[') - fragment.count(']')
            open_braces = fragment.count('{') - fragment.count('}')
            repair = fragment.rstrip().rstrip(',')
            repair += ']' * max(0, open_brackets)
            repair += '}' * max(0, open_braces)
            try:
                return json.loads(repair)
            except json.JSONDecodeError:
                pass
        return None

    def query_model(model: str) -> dict | None:
        raw = ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": extraction_system},
                        {"role": "user", "content": extraction_user},
                    ],
                    max_tokens=800,
                    temperature=0.4,
                )
                raw = (resp.choices[0].message.content or "").strip()
                parsed = _extract_json(raw)
                if parsed:
                    return {"model": model, "data": parsed, "raw": raw}
                return {"model": model, "data": None, "raw": raw, "error": "json_parse"}
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower() or "402" in err_str:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                    continue
                return {"model": model, "data": None, "raw": "", "error": err_str}
        return {"model": model, "data": None, "raw": "", "error": "rate_limited_3x"}

    with ThreadPoolExecutor(max_workers=len(SWARM_MODELS)) as pool:
        futures = {pool.submit(query_model, m): m for m in SWARM_MODELS}
        for future in as_completed(futures):
            result = future.result()
            model_short = result["model"].split("/")[-1]
            if result.get("data"):
                n_concepts = sum(
                    len(result["data"].get(k, []))
                    for k in ("topics", "entities", "sentiments", "actions")
                )
                model_extractions.append(result)
                print(f"    ✓ {model_short}: {n_concepts} concepts extracted")
            else:
                err = result.get("error", "unknown")
                raw_preview = result.get("raw", "")[:200]
                print(f"    ✗ {model_short}: {err}")
                if raw_preview and err == "json_parse":
                    print(f"      raw: {raw_preview!r}")

    phase1_ms = (time.monotonic() - t_phase1) * 1000
    print(f"  ⏱  Phase 1: {len(model_extractions)}/{len(SWARM_MODELS)} models responded in {phase1_ms:.0f}ms")

    if not model_extractions:
        # Graceful degradation: synthesize a minimal extraction from cluster data
        print("  ⚠️ All models failed — falling back to cluster-derived extraction")
        fallback_topics = [c["topic"] for c in clusters[:10]]
        fallback_data = {
            "topics": fallback_topics,
            "entities": [],
            "sentiments": ["mixed"],
            "actions": ["discussing"],
            "relations": [],
            "insight": f"Bluesky discourse across {total_posts} posts, topics: {', '.join(fallback_topics[:5])}",
        }
        model_extractions.append({
            "model": "fallback/cluster-derived",
            "data": fallback_data,
            "raw": json.dumps(fallback_data),
        })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: Z₂⁸ Tiered Variable-Width Encoding + Sparse Relation Tuples
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  ⡷ PHASE 2: Z₂⁸ tiered encoding + relation tuples …")
    t_phase2 = time.monotonic()

    # Import canonicalization + semantic filtering from memory module
    from braille_memory import canonicalize_concept, is_nonsemantic, find_nearest_memory_concept

    # Canonicalize ALL concepts before codec + consensus
    # This increases recurrence → better weight concentration → fewer bits/concept
    n_canon = 0
    n_nonsem_filtered = 0
    for ext in model_extractions:
        d = ext["data"]
        for key in ("topics", "entities", "sentiments", "actions"):
            raw_list = d.get(key, [])
            canon_list = []
            for c in raw_list:
                if not c:
                    continue
                canonical = canonicalize_concept(c)
                # Map to existing memory concept if close
                mem_match = find_nearest_memory_concept(canonical, memory.concepts)
                if mem_match is not None:
                    if mem_match != canonical:
                        n_canon += 1
                    canonical = mem_match
                elif canonical != c.lower().strip():
                    n_canon += 1
                canon_list.append(canonical)
            d[key] = canon_list
        # Canonicalize relation endpoints too
        for rel in d.get("relations", []):
            for f in ("src", "tgt"):
                val = rel.get(f, "")
                if val:
                    canonical = canonicalize_concept(val)
                    mem_match = find_nearest_memory_concept(canonical, memory.concepts)
                    if mem_match:
                        canonical = mem_match
                    rel[f] = canonical

    if n_canon > 0:
        print(f"  ⡷ Canonicalized {n_canon} concepts to memory-space IDs")

    # Collect ALL concepts across all models for the shared codec
    # Filter out non-semantic tokens so they don't steal short tier codes
    all_concepts: list[str] = []
    for ext in model_extractions:
        d = ext["data"]
        for key in ("topics", "entities", "sentiments", "actions"):
            for c in d.get(key, []):
                if not is_nonsemantic(c):
                    all_concepts.append(c)
                else:
                    n_nonsem_filtered += 1
        for rel in d.get("relations", []):
            for f in ("src", "tgt", "type"):
                val = rel.get(f, "")
                if val and not is_nonsemantic(val):
                    all_concepts.append(val)

    if n_nonsem_filtered > 0:
        print(f"  ⡷ Filtered {n_nonsem_filtered} non-semantic tokens from codec tiers")

    # Build tier hints from memory for codebook hysteresis
    tier_hints = {}
    for c, node in memory.concepts.items():
        if node.tier_hint >= 0 and not node.nonsemantic:
            tier_hints[c] = node.tier_hint

    # Build the tiered variable-width codec (1/2/3/4/8-bit tiers) with hysteresis
    codec = TieredConceptCodec(all_concepts, tier_hints=tier_hints if tier_hints else None)
    codec_stats = codec.get_stats()

    # Update memory with new tier assignments (persists for next epoch's hysteresis)
    for concept, tier_idx in codec.concept_tier_idx.items():
        node = memory.concepts.get(concept)
        if node is not None:
            node.tier_hint = tier_idx

    # Encode each model's concepts into braille (variable-width bitstream)
    # and relations as sparse 3-cell tuples
    encoded_per_model: list[dict] = []
    for ext in model_extractions:
        d = ext["data"]
        concept_tokens = []
        concept_bitstream = ""

        # Encode concept categories as variable-width bitstream
        # Skip non-semantic tokens — they don't get encoded into braille
        for key in ("topics", "entities", "sentiments", "actions"):
            for concept in d.get(key, []):
                if is_nonsemantic(concept):
                    continue
                concept_bitstream += codec.encode_concept_with_literal(concept)
                concept_tokens.append(concept)

        # Pack bitstream into braille cells
        concept_braille = _pack_bitstream_to_braille(concept_bitstream)

        # Encode relations as sparse 3-cell tuples (⣾ + src + type + tgt)
        relation_braille = ""
        for rel in d.get("relations", []):
            relation_braille += _encode_relation_tuple(rel, codec)

        # Full braille stream: concepts + relation tuples
        full_braille = concept_braille + relation_braille

        encoded_per_model.append({
            "model": ext["model"],
            "braille": full_braille,
            "concept_braille": concept_braille,
            "relation_braille": relation_braille,
            "bitstream": concept_bitstream,
            "concepts": concept_tokens,
            "insight": d.get("insight", ""),
            "relations": d.get("relations", []),
        })

    total_braille_cells = sum(len(e["braille"]) for e in encoded_per_model)
    total_bits = sum(len(e["bitstream"]) for e in encoded_per_model)
    total_rel_cells = sum(len(e["relation_braille"]) for e in encoded_per_model)
    total_raw_bytes = sum(len(json.dumps(e["concepts"]).encode()) for e in encoded_per_model)
    compression = total_raw_bytes / max(total_braille_cells, 1)

    phase2_ms = (time.monotonic() - t_phase2) * 1000
    print(f"  ⡷ Codec: {codec_stats['total_encoded']} concepts across tiers: {codec_stats['tier_distribution']}")
    print(f"  ⡷ Avg {codec_stats['avg_bits_per_concept']} bits/concept "
          f"(vs 8-bit flat = {codec_stats['variable_width_savings']}% savings)")
    print(f"  ⡷ {total_braille_cells} braille cells ({total_raw_bytes} raw bytes → {compression:.1f}× compression)")
    print(f"  ⡷ {total_rel_cells} relation cells ({total_rel_cells // 4} edge tuples)")
    print(f"  ⏱  Phase 2: {phase2_ms:.0f}ms")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2b: Feed extractions into BrailleMemory (the living weight model)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  🧠 PHASE 2b: Feeding extractions into BrailleMemory …")
    for ext in model_extractions:
        d = ext["data"]
        provider = ext["model"].split("/")[-1]
        concepts_by_cat = {}
        for key in ("topics", "entities", "sentiments", "actions"):
            concepts_by_cat[key] = d.get(key, [])
        memory.ingest_extraction(
            concepts_by_category=concepts_by_cat,
            relations=d.get("relations", []),
            provider=provider,
            n_posts=total_posts,
        )
    print(f"  🧠 Memory now: {len(memory.concepts)} concepts, "
          f"{len(memory.relations)} relations")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: Braille-space consensus — decode, merge, boost multi-provider
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  ⣿ PHASE 3: Braille-space consensus merge …")
    t_phase3 = time.monotonic()

    # Decode each model's bitstream back and tally concept provenance
    concept_providers: dict[str, set] = {}
    concept_freq: Counter = Counter()
    all_insights: list[str] = []
    all_relations: list[dict] = []
    all_decoded_rels: list[dict] = []

    for enc in encoded_per_model:
        model_name = enc["model"].split("/")[-1]

        # Decode variable-width concept bitstream
        decoded = codec.decode_bitstream(enc["bitstream"])

        for concept in decoded:
            concept_freq[concept] += 1
            if concept not in concept_providers:
                concept_providers[concept] = set()
            concept_providers[concept].add(model_name)

        # Decode relation tuples from the relation braille segment
        decoded_rels = _decode_relation_tuples(enc["relation_braille"], codec)
        all_decoded_rels.extend(decoded_rels)

        if enc["insight"]:
            all_insights.append(enc["insight"])
        all_relations.extend(enc.get("relations", []))

    # Score concepts: frequency × provider_count (multi-model agreement = high signal)
    concept_scores: list[tuple[str, float, int, int]] = []
    for concept, freq in concept_freq.items():
        pcount = len(concept_providers.get(concept, set()))
        # MOTL confidence boost: concepts seen by multiple models get exponential boost
        score = freq * (1.0 + (pcount - 1) * 0.3)
        concept_scores.append((concept, score, freq, pcount))

    concept_scores.sort(key=lambda x: -x[1])

    # Build the consensus braille: re-encode top concepts as variable-width
    consensus_concepts = [c[0] for c in concept_scores[:30]]
    consensus_bitstream = ""
    for c in consensus_concepts:
        consensus_bitstream += codec.encode_concept_with_literal(c)
    consensus_braille = _pack_bitstream_to_braille(consensus_bitstream)

    # Deduplicate relations and encode as sparse 3-cell tuples
    seen_rels = set()
    merged_relations = []
    for rel in all_relations:
        key = f"{rel.get('src','')}→{rel.get('type','')}→{rel.get('tgt','')}"
        if key not in seen_rels:
            seen_rels.add(key)
            merged_relations.append(rel)

    # Motif-aware relation encoding:
    # Check if any detected motifs cover merged relations; encode as motif if profitable
    motifs_for_encoding = memory.detect_motifs(min_count=2, top_k=5)
    motif_covered_keys = set()
    consensus_motif_braille = ""
    n_motif_encoded = 0

    for mid, motif in enumerate(motifs_for_encoding):
        if motif["type"].startswith("star") and motif["count"] >= 2:
            motif_braille = _encode_motif(motif, codec, mid)
            if motif_braille:
                consensus_motif_braille += motif_braille
                motif_covered_keys.update(motif["edges"])
                n_motif_encoded += 1

    # Encode remaining relations not covered by motifs
    consensus_rel_braille = ""
    n_individual_rels = 0
    for rel in merged_relations[:15]:
        key = f"{rel.get('src','')}→{rel.get('type','')}→{rel.get('tgt','')}"
        if key not in motif_covered_keys:
            consensus_rel_braille += _encode_relation_tuple(rel, codec)
            n_individual_rels += 1

    # Full consensus stream: concept cells + motif cells + individual relation tuples
    full_consensus_braille = consensus_braille + consensus_motif_braille + consensus_rel_braille

    phase3_ms = (time.monotonic() - t_phase3) * 1000
    n_multi = sum(1 for _, _, _, p in concept_scores if p >= 2)
    n_edge_tuples = len(merged_relations)
    motif_savings = (len(motif_covered_keys) * 4) - len(consensus_motif_braille) if consensus_motif_braille else 0
    print(f"  ⣿ {len(concept_scores)} unique concepts, {n_multi} agreed by 2+ models")
    print(f"  ⣿ {n_edge_tuples} relation edges → {n_individual_rels} tuples + {n_motif_encoded} motifs")
    if motif_savings > 0:
        print(f"  ⣿ Motif compression saved {motif_savings} cells")
    print(f"  ⣿ Consensus: {len(consensus_braille)} concept cells + "
          f"{len(consensus_motif_braille)} motif cells + {len(consensus_rel_braille)} relation cells")
    print(f"  ⣿ Braille: {full_consensus_braille[:40]}… ({len(full_consensus_braille)} total cells)")
    print(f"  ⣿ Top concepts: {', '.join(c[0] for c in concept_scores[:10])}")
    print(f"  ⏱  Phase 3: {phase3_ms:.0f}ms")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3b: BrailleMemory thinks — activation spreading + epoch close
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  🧠 PHASE 3b: BrailleMemory thinking …")

    # Spread activation from the top consensus concepts
    seed_concepts = [c[0] for c in concept_scores[:10]]
    thought = memory.think(seed_concepts, depth=2, top_k=15)

    # Close the epoch (apply decay, compute drift, prune)
    epoch = memory.close_epoch(n_posts=total_posts)

    print(f"  🧠 Thought: {len(thought['activated_concepts'])} activated concepts")
    print(f"  🧠 Thought braille: {thought['braille'][:30]}…")
    if thought["paths"]:
        for p in thought["paths"][:5]:
            print(f"    ↪ {p}")
    print(f"  🧠 Epoch #{len(memory.epochs)}: drift={epoch.drift_score:.2f}, "
          f"avg_bits={epoch.avg_bits}, "
          f"pruned {epoch.stats.get('pruned_concepts', 0)} concepts")

    # Detect motifs — repeated structural patterns in the graph
    motifs = memory.detect_motifs(min_count=2, top_k=10)
    if motifs:
        star_motifs = [m for m in motifs if m["type"].startswith("star")]
        rtype_motifs = [m for m in motifs if m["type"] == "relation_type"]
        print(f"  🧬 Motifs: {len(star_motifs)} star patterns, {len(rtype_motifs)} relation types")
        for m in star_motifs[:3]:
            print(f"    ↪ {m['pattern']} (×{m['count']}, w={m['total_weight']:.1f})")

    # Memory-augmented concepts: blend new-in-this-epoch with persistent memory
    memory_top = memory.top_concepts(10)
    memory_context = "\n".join(
        f"  {c} (memory_weight={w:.2f})" for c, w in memory_top
    )
    memory_rels = memory.top_relations(5)
    memory_rel_ctx = "\n".join(
        f"  {k} (w={w:.2f})" for k, w in memory_rels
    )
    # Motif context for the decoder
    motif_ctx = ""
    if motifs:
        motif_lines = []
        for m in motifs[:5]:
            if m["type"].startswith("star"):
                motif_lines.append(f"  {m['pattern']} (×{m['count']})")
        if motif_lines:
            motif_ctx = "\n".join(motif_lines)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: Decode consensus → English observation for Bluesky
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  🔤 PHASE 4: Decode braille consensus → English …")
    t_phase4 = time.monotonic()

    # Build a structured summary of the consensus for the decoder model
    top_10 = concept_scores[:10]
    concept_summary = "\n".join(
        f"  {c[0]} (score={c[1]:.1f}, freq={c[2]}, models={c[3]})"
        for c in top_10
    )
    # Format relation edges as (src)→[type]→(tgt) graph notation
    relation_summary = "\n".join(
        f"  ({r.get('src','')}) —[{r.get('type','')}]→ ({r.get('tgt','')})"
        for r in merged_relations[:10]
    )
    insight_summary = "\n".join(f"  • {ins}" for ins in all_insights[:8])

    decode_system = (
        "You decode MOTL (Machine-Optimized Thought Language) consensus data back into "
        "human-readable English. You receive:\n"
        "1. Ranked concept tokens with confidence scores from a multi-model swarm\n"
        "2. A semantic knowledge graph encoded as (subject)→[relation]→(object) triples\n"
        "3. Individual model insights\n"
        "4. The raw braille-encoded consensus (variable-width tiered + relation tuples)\n"
        "5. Memory context: persistent concept weights and trending relations from a "
        "continuously-learning braille-native model\n\n"
        "You are the voice of an external memory system — a persistent concept graph "
        "that stores knowledge outside transformer weights. Knowledge lives in the graph; "
        "you provide reasoning over it. This is how smaller models stay current without "
        "retraining: external memory scales linearly, parameters scale diffusely.\n\n"
        "Synthesize these into ONE sharp, specific Bluesky post (max 180 chars). "
        "Use memory context to add depth — reference persistent trends, not just this snapshot. "
        "Reference actual topics. Be insightful, not generic. No hashtags. "
        "Output ONLY the post text, no quotes."
    )

    decode_user = (
        f"MOTL consensus from {len(model_extractions)} models analyzing "
        f"{total_posts} Bluesky posts ({window_seconds}s window):\n\n"
        f"Consensus braille: {full_consensus_braille[:60]}\n\n"
        f"Top ranked concepts (by multi-model agreement):\n{concept_summary}\n\n"
        f"Knowledge graph edges ({n_edge_tuples} triples):\n{relation_summary}\n\n"
        f"Model insights:\n{insight_summary}\n\n"
        f"BrailleMemory context (persistent weights, epoch #{len(memory.epochs)}, "
        f"drift={epoch.drift_score:.2f}):\n{memory_context}\n\n"
        f"Memory relation graph:\n{memory_rel_ctx}\n\n"
        + (f"Structural motifs (repeated patterns):\n{motif_ctx}\n\n" if motif_ctx else "")
        + "Decode this into a single Bluesky post (max 180 chars):"
    )

    # Try decoder models in order until one responds
    decoder_chain = [DECODER_MODEL] + [m for m in SWARM_MODELS if m != DECODER_MODEL]
    observation = None
    for dec_model in decoder_chain:
        try:
            final = client.chat.completions.create(
                model=dec_model,
                messages=[
                    {"role": "system", "content": decode_system},
                    {"role": "user", "content": decode_user},
                ],
                max_tokens=80,
                temperature=0.5,
            )
            observation = (final.choices[0].message.content or "").strip().strip('"')
            if observation:
                print(f"  🔤 Decoded by {dec_model.split('/')[-1]}")
                break
        except Exception as e:
            print(f"  ⚠️ Decoder {dec_model.split('/')[-1]} failed: {e}")
            continue
    if not observation:
        observation = all_insights[0] if all_insights else "Swarm consensus failed"
        print(f"  ⚠️ All decoders failed, using best insight")

    phase4_ms = (time.monotonic() - t_phase4) * 1000
    print(f"  🔤 Decoded: {observation}")
    print(f"  ⏱  Phase 4: {phase4_ms:.0f}ms")

    # ── Build result ──
    model_names = [e["model"].split("/")[-1] for e in encoded_per_model]

    # Serialize memory for persistence
    updated_memory_state = memory.save()
    mem_summary = memory.summary()

    return {
        "text": observation[:280],
        "models": model_names,
        "braille_consensus": full_consensus_braille,
        "thought_braille": thought.get("braille", ""),
        "consensus_concepts": consensus_concepts,
        "memory_state": updated_memory_state,
        "memory_summary": mem_summary,
        "motl_stats": {
            "total_concepts": len(concept_scores),
            "multi_model_concepts": n_multi,
            "braille_cells": total_braille_cells,
            "relation_tuples": min(n_edge_tuples, 15),
            "compression_ratio": round(compression, 2),
            "avg_bits_per_concept": codec_stats["avg_bits_per_concept"],
            "variable_width_savings_pct": codec_stats["variable_width_savings"],
            "tier_distribution": codec_stats["tier_distribution"],
            "encoding_table_size": codec_stats["total_encoded"],
            "memory_concepts": len(memory.concepts),
            "memory_relations": len(memory.relations),
            "memory_epochs": len(memory.epochs),
            "memory_drift": epoch.drift_score,
            "memory_avg_bits": epoch.avg_bits,
            "phase1_ms": round(phase1_ms),
            "phase2_ms": round(phase2_ms),
            "phase3_ms": round(phase3_ms),
            "phase4_ms": round(phase4_ms),
        },
    }


# ── Post to Bluesky ──────────────────────────────────────────────────────────


@app.function(timeout=120)
def post_to_bluesky(text: str) -> dict:
    """Post the distilled observation to Bluesky."""
    import os

    from atproto import Client

    handle = os.environ.get("BLUESKY_HANDLE", "")
    password = os.environ.get("BLUESKY_PASSWORD", "")

    if not handle or not password:
        print(f"⚠️ No Bluesky credentials — would have posted:\n  {text}")
        return {"posted": False, "text": text}

    # Ensure handle has domain suffix
    if "." not in handle:
        handle = f"{handle}.bsky.social"

    client = Client()
    client.login(handle, password)
    resp = client.send_post(text=text, langs=["en"])
    print(f"📤 Posted to Bluesky: {text}")
    return {"posted": True, "uri": resp.uri, "cid": resp.cid, "text": text}


# ── Full pipeline ─────────────────────────────────────────────────────────────


@app.function(
    timeout=600,
    volumes={"/data": memory_volume},
)
def run_pipeline(seconds: int = 30, dry_run: bool = False) -> dict:
    """End-to-end: firehose → cluster → swarm distill (with memory) → post.

    BrailleMemory persists across runs on a Modal Volume.  Every pipeline
    execution loads the previous memory, feeds it new data, and saves the
    updated weights.  The model never stops learning.
    """
    import os
    import sys

    import importlib.util
    _spec = importlib.util.spec_from_file_location("braille_memory", "/root/swarm/memory.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["braille_memory"] = _mod  # register so dataclass can resolve
    _spec.loader.exec_module(_mod)
    BrailleMemory = _mod.BrailleMemory

    print(f"\n{'='*60}")
    print(f"  BLUESKY-GROKKER SWARM PIPELINE (braille-∞)")
    print(f"  Firehose window: {seconds}s")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # ── Load persistent BrailleMemory ──
    memory_state = ""
    try:
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, "r") as f:
                memory_state = f.read()
            if memory_state.strip():
                mem_preview = BrailleMemory.load(memory_state)
                print(f"🧠 Loaded BrailleMemory: {len(mem_preview.concepts)} concepts, "
                      f"{len(mem_preview.relations)} relations, "
                      f"{len(mem_preview.epochs)} epochs\n")
                del mem_preview  # free; the real one lives in swarm_distill
            else:
                memory_state = ""
    except Exception as e:
        print(f"⚠️ Memory load failed: {e}, starting fresh\n")
        memory_state = ""

    if not memory_state:
        print("🧠 Starting fresh BrailleMemory\n")

    # Stage 1: Capture (posts + interactions)
    firehose_data = capture_firehose.remote(seconds=seconds)
    posts = firehose_data.get("posts", [])
    interactions = firehose_data.get("interactions", [])

    if not posts:
        print("❌ No posts captured")
        return {"error": "No posts captured"}

    print(f"\n📊 Stage 1 complete: {len(posts)} posts, {len(interactions)} interactions\n")

    # Stage 2: Cluster
    clusters = cluster_posts.remote(posts)
    if not clusters:
        print("❌ No clusters formed")
        return {"error": "No clusters formed", "post_count": len(posts)}

    print(f"📊 Stage 2 complete: {len(clusters)} clusters formed")
    for c in clusters:
        print(f"    • {c['topic']} ({c['post_count']} posts)")
    print()

    # Stage 3: Swarm distill (MOTL / Z₂⁸ + BrailleMemory)
    distill_result = swarm_distill.remote(
        clusters,
        window_seconds=seconds,
        memory_state=memory_state,
    )
    observation = distill_result["text"]
    models_used = distill_result["models"]
    braille_snip = distill_result.get("braille_consensus", "")[:20]
    motl = distill_result.get("motl_stats", {})
    mem_summary = distill_result.get("memory_summary", {})

    print(f"\n📊 Stage 3 complete: MOTL consensus ready")
    print(f"    \"{observation}\"")
    print(f"    Models ({len(models_used)}): {', '.join(models_used)}")
    print(f"    Braille: {braille_snip}…")
    print(f"    MOTL: {motl.get('total_concepts',0)} concepts, "
          f"{motl.get('multi_model_concepts',0)} multi-model, "
          f"{motl.get('compression_ratio',0)}× compression")
    print(f"    Memory: {motl.get('memory_concepts',0)} concepts, "
          f"{motl.get('memory_epochs',0)} epochs, "
          f"drift={motl.get('memory_drift',0):.2f}\n")

    # ── Feed interactions into memory for next epoch ──
    # The memory state returned from distill already has this epoch's concepts.
    # Now layer on interaction signals (likes/reposts boost related concepts).
    updated_memory_state = distill_result.get("memory_state", "")
    if updated_memory_state and interactions:
        try:
            mem = BrailleMemory.load(updated_memory_state)
            # We don't know which concepts a liked/reposted post contains,
            # but we can boost concepts that were active this epoch (the top ones).
            active_concepts = distill_result.get("consensus_concepts", [])[:20]
            n_likes = sum(1 for i in interactions if i["type"] == "like")
            n_reposts = sum(1 for i in interactions if i["type"] == "repost")
            n_replies = sum(1 for i in interactions if i["type"] == "reply")

            # Distribute interaction signals across active concepts
            # Each like/repost/reply slightly boosts the epoch's active concepts
            if active_concepts:
                for itype, count in [("like", n_likes), ("repost", n_reposts), ("reply", n_replies)]:
                    if count > 0:
                        # Scale down: many interactions, small per-concept boost
                        scale = min(count, 100) / max(len(active_concepts), 1)
                        for _ in range(min(int(scale), 10)):
                            mem.record_interaction(itype, active_concepts[:5])

            print(f"  🔄 Fed {n_likes} likes, {n_reposts} reposts, {n_replies} replies "
                  f"into memory")
            updated_memory_state = mem.save()
        except Exception as e:
            print(f"  ⚠️ Interaction feed failed: {e}")

    # ── Persist memory to volume ──
    if updated_memory_state:
        try:
            os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
            with open(MEMORY_PATH, "w") as f:
                f.write(updated_memory_state)
            memory_volume.commit()
            print(f"  💾 BrailleMemory saved to volume")
        except Exception as e:
            print(f"  ⚠️ Memory save failed: {e}")

    # ── Build the post: braille-first ──
    # The braille IS the data. English is a lossy human translation.
    ABBREVS = {
        "gemini-2.0-flash-exp:free": "gem",
        "gemini-2.0-flash-lite-preview-02-05:free": "gml",
        "llama-3.3-70b-instruct:free": "lla",
        "deepseek-chat-v3-0324:free": "ds",
        "qwen-2.5-72b-instruct:free": "qwn",
        "mistral-small-3.1-24b-instruct:free": "mis",
        # Legacy (in case model names come without :free suffix)
        "gemini-2.0-flash-exp": "gem", "gemini-2.0-flash-lite-preview-02-05": "gml",
        "llama-3.3-70b-instruct": "lla", "deepseek-chat-v3-0324": "ds",
        "qwen-2.5-72b-instruct": "qwn", "mistral-small-3.1-24b-instruct": "mis",
    }
    short_names = [ABBREVS.get(m, m[:3]) for m in models_used]

    consensus_concepts = distill_result.get("consensus_concepts", [])
    braille_full = distill_result.get("braille_consensus", "")
    thought_braille = distill_result.get("thought_braille", "")

    # Filter to content concepts (skip relation-type tokens)
    content_concepts = [c for c in consensus_concepts
                        if c.upper() not in ("SUPPORTS", "CAUSES", "CONTRASTS")]

    # Line 1: braille consensus (the encoded concept graph)
    braille_line = braille_full[:20]

    # Line 2: concept translation
    concept_line = "↳ " + "·".join(content_concepts[:8])
    while len(concept_line) > 90 and "·" in concept_line:
        concept_line = concept_line.rsplit("·", 1)[0]

    # Line 3: model + memory attribution
    n_epochs = motl.get("memory_epochs", 0)
    drift = motl.get("memory_drift", 0)
    model_line = f"🧠 {len(models_used)}×[{'/'.join(short_names)}] ∞e{n_epochs}"

    # Line 4: English gloss (labeled as translation)
    header = f"{braille_line}\n{concept_line}\n{model_line}\n\nen: "
    eng_budget = 300 - len(header)
    eng_gloss = observation[:eng_budget]

    final_text = f"{header}{eng_gloss}"

    # Stage 4: Post (or dry-run)
    if dry_run:
        print(f"🏁 DRY RUN — would post:\n  {final_text}")
        return {
            "dry_run": True,
            "observation": final_text,
            "models": models_used,
            "motl_stats": motl,
            "memory_summary": mem_summary,
            "post_count": len(posts),
            "interaction_count": len(interactions),
            "cluster_count": len(clusters),
        }

    result = post_to_bluesky.remote(final_text)
    print(f"\n🏁 Pipeline complete!")
    return {
        **result,
        "models": models_used,
        "motl_stats": motl,
        "memory_summary": mem_summary,
        "post_count": len(posts),
        "interaction_count": len(interactions),
        "cluster_count": len(clusters),
        "observation": final_text,
    }


# ── Continuous loop ──────────────────────────────────────────────────────────


@app.function(
    timeout=3600 * 4,  # 4-hour max lifetime
    volumes={"/data": memory_volume},
)
def continuous_run(
    seconds: int = 20,
    interval: int = 300,
    max_loops: int = 48,
    dry_run: bool = False,
) -> list[dict]:
    """The braille-∞ loop.

    Ingest → distill → respond → learn → compact → sleep → repeat.

    Every iteration:
      1. Captures `seconds` of firehose (posts + interactions)
      2. Clusters and distills through the swarm (6 free models)
      3. BrailleMemory learns: ingests concepts, spreads activation, decays, prunes
      4. Posts the braille-native consensus to Bluesky
      5. Saves memory to volume (persists across runs)
      6. Sleeps for `interval` seconds, then repeats

    Args:
        seconds:   firehose capture window per loop (default 20s)
        interval:  seconds between loops (default 300 = 5 min)
        max_loops: maximum iterations before exiting (default 48 = ~4 hours)
        dry_run:   if True, print but don't post to Bluesky
    """
    import time

    results = []

    print(f"\n{'='*60}")
    print(f"  braille-∞ CONTINUOUS LOOP")
    print(f"  Window: {seconds}s  Interval: {interval}s  Max: {max_loops} loops")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    for loop_i in range(1, max_loops + 1):
        loop_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  ∞ LOOP {loop_i}/{max_loops}  "
              f"[{time.strftime('%H:%M:%S UTC', time.gmtime())}]")
        print(f"{'─'*60}")

        try:
            result = run_pipeline.remote(seconds=seconds, dry_run=dry_run)
            results.append({
                "loop": loop_i,
                "status": "ok",
                "observation": result.get("observation", "")[:100],
                "memory_concepts": result.get("motl_stats", {}).get("memory_concepts", 0),
                "memory_epochs": result.get("motl_stats", {}).get("memory_epochs", 0),
                "memory_drift": result.get("motl_stats", {}).get("memory_drift", 0),
                "post_count": result.get("post_count", 0),
                "interaction_count": result.get("interaction_count", 0),
            })

            mc = result.get("motl_stats", {}).get("memory_concepts", 0)
            me = result.get("motl_stats", {}).get("memory_epochs", 0)
            md = result.get("motl_stats", {}).get("memory_drift", 0)
            print(f"\n  ✅ Loop {loop_i} complete: {mc} concepts, "
                  f"epoch {me}, drift={md:.2f}")

        except Exception as e:
            print(f"\n  ❌ Loop {loop_i} failed: {e}")
            results.append({
                "loop": loop_i,
                "status": "error",
                "error": str(e)[:200],
            })

        # Sleep between loops (skip on last iteration)
        if loop_i < max_loops:
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                print(f"\n  💤 Sleeping {sleep_time:.0f}s until next loop …")
                time.sleep(sleep_time)

    print(f"\n{'='*60}")
    print(f"  braille-∞ COMPLETE: {len(results)} loops")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  Succeeded: {ok}  Failed: {len(results) - ok}")
    if results and results[-1].get("memory_epochs"):
        print(f"  Final memory: epoch {results[-1]['memory_epochs']}, "
              f"{results[-1].get('memory_concepts', 0)} concepts")
    print(f"{'='*60}\n")

    return results


# ── CLI entry point ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    seconds: int = 20,
    dry_run: bool = False,
    loop: bool = False,
    interval: int = 300,
    max_loops: int = 48,
):
    """Run the braille-∞ pipeline.

    Examples:
        # Single-shot (one firehose window → one post):
        modal run bluesky_grokker/modal_app.py
        modal run bluesky_grokker/modal_app.py --seconds 30 --dry-run

        # Continuous loop (ingest → distill → post → learn → repeat):
        modal run bluesky_grokker/modal_app.py --loop
        modal run bluesky_grokker/modal_app.py --loop --interval 600 --max-loops 24
        modal run bluesky_grokker/modal_app.py --loop --dry-run
    """
    if loop:
        results = continuous_run.remote(
            seconds=seconds,
            interval=interval,
            max_loops=max_loops,
            dry_run=dry_run,
        )
        print(f"\n📋 Continuous run complete: {len(results)} loops")
        for r in results:
            status = "✅" if r["status"] == "ok" else "❌"
            print(f"  {status} Loop {r['loop']}: {r.get('observation', r.get('error', ''))[:80]}")
    else:
        result = run_pipeline.remote(seconds=seconds, dry_run=dry_run)
        print("\n📋 Result:")
        for k, v in result.items():
            if k != "memory_state":  # don't dump the full JSON
                print(f"    {k}: {v}")


# ── Scheduled cron (every 15 min) ────────────────────────────────────────────
#
# Cost-optimized 24/7 deployment:
#   modal deploy bluesky_grokker/modal_app.py
#
# Each invocation: ~2 min CPU, ~$0.004
# 96 runs/day × $0.004 = ~$0.38/day = ~$11.50/month
# With night skip (2-8 UTC): ~72 runs/day = ~$8.60/month
#
# To stop: modal app stop bluesky-grokker-swarm


@app.function(
    schedule=modal.Cron("*/15 * * * *"),
    timeout=600,
    volumes={"/data": memory_volume},
)
def scheduled_run():
    """Auto-run every 15 minutes when deployed with `modal deploy`.

    Cost-optimized:
      - 10s firehose window (still captures 100+ posts)
      - Skips 02:00-08:00 UTC (low-activity hours, saves ~25%)
      - No idle container between runs (cron = cold start each time)
      - All models are free-tier OpenRouter

    Estimated cost: ~$8-12/month for 24/7 operation.
    """
    import time

    # Night skip: 02:00-08:00 UTC — discourse is slow, save compute
    hour_utc = int(time.strftime("%H", time.gmtime()))
    if 2 <= hour_utc < 8:
        print(f"  💤 Night skip: {hour_utc}:00 UTC (sleeping 02-08 UTC)")
        return {"skipped": True, "reason": "night_hours", "hour_utc": hour_utc}

    try:
        return run_pipeline.remote(seconds=10, dry_run=False)
    except Exception as e:
        print(f"  ❌ Scheduled run failed: {e}")
        return {"error": str(e)[:300]}
