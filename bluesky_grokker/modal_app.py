"""Modal-deployed Bluesky Firehose → Braille-Swarm-Consensus → Post pipeline.

Usage:
    # One-shot run (default 30s firehose window):
    modal run modal_app.py

    # Custom window:
    modal run modal_app.py --seconds 60

    # Deploy as a cron (every 15 minutes):
    modal deploy modal_app.py

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
)

app = modal.App(
    name="bluesky-grokker-swarm",
    image=image,
    secrets=[modal.Secret.from_name("bluesky-grokker")],
)

# ── Firehose capture ─────────────────────────────────────────────────────────


@app.function(timeout=300)
def capture_firehose(seconds: int = 30) -> list[dict]:
    """Connect to the Bluesky firehose and collect posts for `seconds`."""
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

            if raw.get("$type") == "app.bsky.feed.post":
                text = raw.get("text", "").strip()
                if not text or len(text) < 10:
                    continue
                langs = raw.get("langs") or []
                # Only English posts (or untagged)
                if langs and "en" not in langs:
                    continue

                created_str = raw.get("createdAt", "")
                try:
                    created_at = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    ).isoformat()
                except Exception:
                    created_at = datetime.now(timezone.utc).isoformat()

                with lock:
                    posts.append(
                        {
                            "uri": f"at://{commit.repo}/{op.path}",
                            "author_did": commit.repo,
                            "text": text[:500],
                            "created_at": created_at,
                        }
                    )

    print(f"🔥 Capturing firehose for {seconds}s …")
    # Run the blocking firehose client in the main thread
    # It will self-stop when the deadline is reached
    try:
        client.start(on_message)
    except Exception:
        pass  # client.stop() causes a benign exception

    print(f"📦 Captured {len(posts)} posts")
    return posts


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

    def __init__(self, all_concepts: list[str]) -> None:
        from collections import Counter

        freq = Counter(all_concepts)
        ranked = [c for c, _ in freq.most_common()]

        self.encode_table: dict[str, str] = {}   # concept → bit string
        self.decode_table: dict[str, str] = {}   # bit string → concept
        self.concept_tier: dict[str, str] = {}    # concept → tier label

        idx = 0
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
                idx += 1

        # Assign long tier (escape prefix + 8-bit index)
        long_start = idx
        n_long = min(256, len(ranked) - idx)
        for i in range(n_long):
            concept = ranked[idx]
            code = self.LONG_PREFIX + format(i, f"0{self.LONG_BITS}b")
            self.encode_table[concept] = code
            self.decode_table[code] = concept
            self.concept_tier[concept] = "11b-long"
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

# Expanded model roster — 8 diverse models for richer consensus
SWARM_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-001",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2411",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat-v3-0324",
]


@app.function(timeout=300)
def swarm_distill(clusters: list[dict], window_seconds: int = 30) -> dict:
    """Braille-swarm-consensus pipeline.

    Phase 1 — Fan out to N models: each extracts structured concept tokens
    Phase 2 — Encode all concepts into Z₂⁸ braille vector space
    Phase 3 — Merge/distill in braille-encoded space (multi-provider boosting)
    Phase 4 — Decode consensus concept graph → English observation

    Returns {"text": ..., "models": [...], "braille": ..., "motl_stats": ...}.
    """
    import json
    import os
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import openai

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

    def query_model(model: str) -> dict | None:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": extraction_system},
                    {"role": "user", "content": extraction_user},
                ],
                max_tokens=500,
                temperature=0.4,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            return {"model": model, "data": parsed, "raw": raw}
        except json.JSONDecodeError:
            # Try to salvage partial JSON
            return {"model": model, "data": None, "raw": raw, "error": "json_parse"}
        except Exception as e:
            return {"model": model, "data": None, "raw": "", "error": str(e)}

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
                print(f"    ✗ {model_short}: {err}")

    phase1_ms = (time.monotonic() - t_phase1) * 1000
    print(f"  ⏱  Phase 1: {len(model_extractions)}/{len(SWARM_MODELS)} models responded in {phase1_ms:.0f}ms")

    if not model_extractions:
        raise RuntimeError("All model extractions failed")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: Z₂⁸ Tiered Variable-Width Encoding + Sparse Relation Tuples
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  ⡷ PHASE 2: Z₂⁸ tiered encoding + relation tuples …")
    t_phase2 = time.monotonic()

    # Collect ALL concepts across all models for the shared codec
    all_concepts: list[str] = []
    for ext in model_extractions:
        d = ext["data"]
        for key in ("topics", "entities", "sentiments", "actions"):
            all_concepts.extend(d.get(key, []))
        for rel in d.get("relations", []):
            for f in ("src", "tgt", "type"):
                val = rel.get(f, "")
                if val:
                    all_concepts.append(val)

    # Build the tiered variable-width codec (1/2/3/4/8-bit tiers)
    codec = TieredConceptCodec(all_concepts)
    codec_stats = codec.get_stats()

    # Encode each model's concepts into braille (variable-width bitstream)
    # and relations as sparse 3-cell tuples
    encoded_per_model: list[dict] = []
    for ext in model_extractions:
        d = ext["data"]
        concept_tokens = []
        concept_bitstream = ""

        # Encode concept categories as variable-width bitstream
        for key in ("topics", "entities", "sentiments", "actions"):
            for concept in d.get(key, []):
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

    # Encode merged relations as consensus relation braille
    consensus_rel_braille = ""
    for rel in merged_relations[:15]:
        consensus_rel_braille += _encode_relation_tuple(rel, codec)

    # Full consensus stream: concept cells + relation tuples
    full_consensus_braille = consensus_braille + consensus_rel_braille

    phase3_ms = (time.monotonic() - t_phase3) * 1000
    n_multi = sum(1 for _, _, _, p in concept_scores if p >= 2)
    n_edge_tuples = len(merged_relations)
    print(f"  ⣿ {len(concept_scores)} unique concepts, {n_multi} agreed by 2+ models")
    print(f"  ⣿ {n_edge_tuples} relation edges → {min(n_edge_tuples, 15)} consensus tuples")
    print(f"  ⣿ Consensus: {len(consensus_braille)} concept cells + {len(consensus_rel_braille)} relation cells")
    print(f"  ⣿ Braille: {full_consensus_braille[:40]}… ({len(full_consensus_braille)} total cells)")
    print(f"  ⣿ Top concepts: {', '.join(c[0] for c in concept_scores[:10])}")
    print(f"  ⏱  Phase 3: {phase3_ms:.0f}ms")

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
        "4. The raw braille-encoded consensus (variable-width tiered + relation tuples)\n\n"
        "Synthesize these into ONE sharp, specific Bluesky post (max 180 chars). "
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
        "Decode this into a single Bluesky post (max 180 chars):"
    )

    try:
        final = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": decode_system},
                {"role": "user", "content": decode_user},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        observation = (final.choices[0].message.content or "").strip().strip('"')
    except Exception as e:
        print(f"  ⚠️ GPT-4o decode failed: {e}, falling back to best insight")
        observation = all_insights[0] if all_insights else "Swarm consensus failed"

    phase4_ms = (time.monotonic() - t_phase4) * 1000
    print(f"  🔤 Decoded: {observation}")
    print(f"  ⏱  Phase 4: {phase4_ms:.0f}ms")

    # ── Build result ──
    model_names = [e["model"].split("/")[-1] for e in encoded_per_model]

    return {
        "text": observation[:280],
        "models": model_names,
        "braille_consensus": full_consensus_braille,
        "consensus_concepts": consensus_concepts,
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
            "phase1_ms": round(phase1_ms),
            "phase2_ms": round(phase2_ms),
            "phase3_ms": round(phase3_ms),
            "phase4_ms": round(phase4_ms),
        },
    }


# ── Post to Bluesky ──────────────────────────────────────────────────────────


@app.function(timeout=30)
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


@app.function(timeout=600)
def run_pipeline(seconds: int = 30, dry_run: bool = False) -> dict:
    """End-to-end: firehose capture → cluster → swarm distill → post."""
    print(f"\n{'='*60}")
    print(f"  BLUESKY-GROKKER SWARM PIPELINE")
    print(f"  Firehose window: {seconds}s")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # Stage 1: Capture
    posts = capture_firehose.remote(seconds=seconds)
    if not posts:
        print("❌ No posts captured")
        return {"error": "No posts captured"}

    print(f"\n📊 Stage 1 complete: {len(posts)} posts captured\n")

    # Stage 2: Cluster
    clusters = cluster_posts.remote(posts)
    if not clusters:
        print("❌ No clusters formed")
        return {"error": "No clusters formed", "post_count": len(posts)}

    print(f"📊 Stage 2 complete: {len(clusters)} clusters formed")
    for c in clusters:
        print(f"    • {c['topic']} ({c['post_count']} posts)")
    print()

    # Stage 3: Swarm distill (MOTL / Z₂⁸ braille pipeline)
    distill_result = swarm_distill.remote(clusters, window_seconds=seconds)
    observation = distill_result["text"]
    models_used = distill_result["models"]
    braille_snip = distill_result.get("braille_consensus", "")[:20]
    motl = distill_result.get("motl_stats", {})

    print(f"\n📊 Stage 3 complete: MOTL consensus ready")
    print(f"    \"{observation}\"")
    print(f"    Models ({len(models_used)}): {', '.join(models_used)}")
    print(f"    Braille: {braille_snip}…")
    print(f"    MOTL: {motl.get('total_concepts',0)} concepts, "
          f"{motl.get('multi_model_concepts',0)} multi-model, "
          f"{motl.get('compression_ratio',0)}× compression\n")

    # Build the final post: observation + braille + translation + models
    # Strategy: build footer first (fixed budget), give rest to observation
    ABBREVS = {
        "gpt-4o": "4o", "gpt-4o-mini": "4om",
        "claude-3.5-sonnet": "son", "claude-3-haiku": "hai",
        "gemini-2.0-flash-001": "gem", "gemini-2.0-flash-lite-001": "gml",
        "llama-3.3-70b-instruct": "lla", "mistral-large-2411": "mis",
        "qwen-2.5-72b-instruct": "qwn", "deepseek-chat-v3-0324": "ds",
    }
    short_names = [ABBREVS.get(m, m[:3]) for m in models_used]
    model_line = f"🧠 {len(models_used)}×[{'/'.join(short_names)}]"

    consensus_concepts = distill_result.get("consensus_concepts", [])
    braille_full = distill_result.get("braille_consensus", "")

    # Use top 8 concepts for the braille+translation lines
    n_show = min(8, len(braille_full), len(consensus_concepts))
    braille_line = braille_full[:n_show]
    # Build concept translation, trimming to fit
    concept_tokens = consensus_concepts[:n_show]
    # Filter out MOTL relation types (SUPPORTS/CAUSES/CONTRASTS) — keep content concepts
    content_concepts = [c for c in concept_tokens if c.upper() not in ("SUPPORTS", "CAUSES", "CONTRASTS")]
    if not content_concepts:
        content_concepts = concept_tokens[:n_show]
    concept_line = "·".join(content_concepts)
    # Cap translation line at 80 chars
    while len(concept_line) > 80 and "·" in concept_line:
        concept_line = concept_line.rsplit("·", 1)[0]

    footer = f"\n\n⡷ {braille_line}\n↳ {concept_line}\n{model_line}"
    obs_budget = 300 - len(footer)
    final_text = f"{observation[:obs_budget]}{footer}"

    # Stage 4: Post (or dry-run)
    if dry_run:
        print(f"🏁 DRY RUN — would post:\n  {final_text}")
        return {
            "dry_run": True,
            "observation": final_text,
            "models": models_used,
            "motl_stats": motl,
            "post_count": len(posts),
            "cluster_count": len(clusters),
        }

    result = post_to_bluesky.remote(final_text)
    print(f"\n🏁 Pipeline complete!")
    return {
        **result,
        "models": models_used,
        "motl_stats": motl,
        "post_count": len(posts),
        "cluster_count": len(clusters),
        "observation": final_text,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(seconds: int = 30, dry_run: bool = False):
    """Run the full pipeline from the command line.

    Examples:
        modal run modal_app.py                    # 30s window, posts to Bluesky
        modal run modal_app.py --seconds 60       # 60s window
        modal run modal_app.py --dry-run           # don't post, just print
    """
    result = run_pipeline.remote(seconds=seconds, dry_run=dry_run)
    print("\n📋 Result:")
    for k, v in result.items():
        print(f"    {k}: {v}")


# ── Scheduled cron (every 15 min) ────────────────────────────────────────────


@app.function(
    schedule=modal.Cron("*/15 * * * *"),
    timeout=600,
)
def scheduled_run():
    """Auto-run every 15 minutes when deployed with `modal deploy`."""
    return run_pipeline.remote(seconds=30, dry_run=False)
