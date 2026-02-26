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

# 8-dot braille: each cell is a vector in Z₂⁸ (8 binary dimensions).
# Unicode braille block: U+2800 to U+28FF (256 codepoints = 2^8 exactly).
# We map semantic concepts → braille cells using frequency-adaptive coding.

BRAILLE_BASE = 0x2800  # ⠀ (empty braille)


def _bits_to_braille(bits: int) -> str:
    """Convert an 8-bit integer (0-255) to its Unicode braille character."""
    return chr(BRAILLE_BASE + (bits & 0xFF))


def _braille_to_bits(ch: str) -> int:
    """Convert a Unicode braille character back to its 8-bit integer."""
    return ord(ch) - BRAILLE_BASE


def _encode_concept_to_braille(concept: str, concept_table: dict) -> str:
    """Encode a concept string into braille cells using the shared table.

    Known concepts get a 1-cell encoding (8 bits, frequency-ranked).
    Unknown concepts get a literal multi-cell encoding (1 byte per char).
    """
    if concept in concept_table:
        return _bits_to_braille(concept_table[concept])
    # Literal fallback: marker cell (0xFF = ⣿) + one braille cell per byte
    return _bits_to_braille(0xFF) + "".join(
        _bits_to_braille(b) for b in concept.encode("utf-8")
    )


def _decode_braille_to_concept(braille: str, reverse_table: dict) -> list[str]:
    """Decode a braille string back into concept tokens."""
    concepts = []
    i = 0
    while i < len(braille):
        bits = _braille_to_bits(braille[i])
        if bits == 0xFF:
            # Literal: read until next known cell or end
            i += 1
            raw_bytes = bytearray()
            while i < len(braille):
                b = _braille_to_bits(braille[i])
                if b == 0xFF or b in reverse_table:
                    break
                raw_bytes.append(b)
                i += 1
            try:
                concepts.append(raw_bytes.decode("utf-8"))
            except Exception:
                concepts.append(f"<unk:{raw_bytes.hex()}>")
        elif bits in reverse_table:
            concepts.append(reverse_table[bits])
            i += 1
        else:
            i += 1
    return concepts


def _build_concept_table(all_concepts: list[str]) -> dict[str, int]:
    """Build a frequency-ranked concept → 8-bit-index table (MOTL adaptive).

    Most frequent concepts get the lowest indices (shortest in variable-depth
    schemes; here all are 1 braille cell but rank still matters for merge
    confidence boosting).
    """
    from collections import Counter
    freq = Counter(all_concepts)
    # Reserve 0xFF for literal marker, 0xFE for separator
    ranked = [c for c, _ in freq.most_common(254)]
    return {concept: idx for idx, concept in enumerate(ranked)}


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
    # PHASE 2: Z₂⁸ Braille Encoding — encode all concepts into braille space
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  ⡷ PHASE 2: Z₂⁸ braille encoding …")
    t_phase2 = time.monotonic()

    # Collect ALL concepts across all models for the shared encoding table
    all_concepts: list[str] = []
    for ext in model_extractions:
        d = ext["data"]
        for key in ("topics", "entities", "sentiments", "actions"):
            all_concepts.extend(d.get(key, []))
        for rel in d.get("relations", []):
            all_concepts.extend([rel.get("src", ""), rel.get("tgt", ""), rel.get("type", "")])

    # Build the adaptive MOTL table (frequency-ranked → braille index)
    concept_table = _build_concept_table(all_concepts)
    reverse_table = {v: k for k, v in concept_table.items()}

    # Encode each model's concepts into braille
    encoded_per_model: list[dict] = []
    for ext in model_extractions:
        d = ext["data"]
        braille_stream = ""
        concept_tokens = []

        for key in ("topics", "entities", "sentiments", "actions"):
            for concept in d.get(key, []):
                braille_stream += _encode_concept_to_braille(concept, concept_table)
                concept_tokens.append(concept)
            braille_stream += _bits_to_braille(0xFE)  # category separator

        # Encode relations
        for rel in d.get("relations", []):
            for field in ("src", "type", "tgt"):
                val = rel.get(field, "")
                if val:
                    braille_stream += _encode_concept_to_braille(val, concept_table)

        encoded_per_model.append({
            "model": ext["model"],
            "braille": braille_stream,
            "concepts": concept_tokens,
            "insight": d.get("insight", ""),
            "relations": d.get("relations", []),
        })

    total_braille_cells = sum(len(e["braille"]) for e in encoded_per_model)
    total_raw_bytes = sum(len(json.dumps(e["concepts"]).encode()) for e in encoded_per_model)
    compression = total_raw_bytes / max(total_braille_cells, 1)

    phase2_ms = (time.monotonic() - t_phase2) * 1000
    print(f"  ⡷ Encoded {len(concept_table)} unique concepts into Z₂⁸ space")
    print(f"  ⡷ {total_braille_cells} braille cells ({total_raw_bytes} raw bytes → {compression:.1f}× compression)")
    print(f"  ⏱  Phase 2: {phase2_ms:.0f}ms")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: Braille-space consensus — merge concepts, boost multi-provider
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n  ⣿ PHASE 3: Braille-space consensus merge …")
    t_phase3 = time.monotonic()

    # Decode each model's braille back and tally concept provenance
    concept_providers: dict[str, set] = {}
    concept_freq: Counter = Counter()
    all_insights: list[str] = []
    all_relations: list[dict] = []

    for enc in encoded_per_model:
        model_name = enc["model"].split("/")[-1]
        # Decode this model's braille stream back to concepts
        decoded = _decode_braille_to_concept(enc["braille"], reverse_table)

        for concept in decoded:
            concept_freq[concept] += 1
            if concept not in concept_providers:
                concept_providers[concept] = set()
            concept_providers[concept].add(model_name)

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

    # Build the consensus braille string (top concepts only, re-encoded)
    consensus_concepts = [c[0] for c in concept_scores[:30]]
    consensus_braille = "".join(
        _encode_concept_to_braille(c, concept_table) for c in consensus_concepts
    )

    # Deduplicate relations
    seen_rels = set()
    merged_relations = []
    for rel in all_relations:
        key = f"{rel.get('src','')}→{rel.get('type','')}→{rel.get('tgt','')}"
        if key not in seen_rels:
            seen_rels.add(key)
            merged_relations.append(rel)

    phase3_ms = (time.monotonic() - t_phase3) * 1000
    n_multi = sum(1 for _, _, _, p in concept_scores if p >= 2)
    print(f"  ⣿ {len(concept_scores)} unique concepts, {n_multi} agreed by 2+ models")
    print(f"  ⣿ Consensus braille: {consensus_braille[:40]}… ({len(consensus_braille)} cells)")
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
    relation_summary = "\n".join(
        f"  {r.get('src','')} —[{r.get('type','')}]→ {r.get('tgt','')}"
        for r in merged_relations[:10]
    )
    insight_summary = "\n".join(f"  • {ins}" for ins in all_insights[:8])

    decode_system = (
        "You decode MOTL (Machine-Optimized Thought Language) consensus data back into "
        "human-readable English. You receive:\n"
        "1. Ranked concept tokens with confidence scores from a multi-model swarm\n"
        "2. Semantic relations between concepts\n"
        "3. Individual model insights\n"
        "4. The raw braille-encoded consensus string\n\n"
        "Synthesize these into ONE sharp, specific Bluesky post (max 180 chars). "
        "Reference actual topics. Be insightful, not generic. No hashtags. "
        "Output ONLY the post text, no quotes."
    )

    decode_user = (
        f"MOTL consensus from {len(model_extractions)} models analyzing "
        f"{total_posts} Bluesky posts ({window_seconds}s window):\n\n"
        f"Consensus braille: {consensus_braille[:60]}\n\n"
        f"Top ranked concepts (by multi-model agreement):\n{concept_summary}\n\n"
        f"Key relations:\n{relation_summary}\n\n"
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
        "braille_consensus": consensus_braille,
        "consensus_concepts": consensus_concepts,
        "motl_stats": {
            "total_concepts": len(concept_scores),
            "multi_model_concepts": n_multi,
            "braille_cells": total_braille_cells,
            "compression_ratio": round(compression, 2),
            "encoding_table_size": len(concept_table),
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
