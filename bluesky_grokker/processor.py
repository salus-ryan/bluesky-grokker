"""Semantic Processor – normalises text, computes embeddings, extracts entities,
and writes everything to the storage layer."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from config import BATCH_SIZE, MAX_CONCURRENT_EMBEDS
from firehose import FirehoseIngestor, FirehosePost, FirehoseRelationship
from models.embedding import get_embeddings_batch
import storage

log = logging.getLogger(__name__)

# ── Text normalisation helpers ───────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@[\w.-]+\.[\w]+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lower-case, collapse whitespace, strip surrounding space."""
    t = text.strip()
    t = _WHITESPACE_RE.sub(" ", t)
    return t


def extract_urls(text: str) -> List[str]:
    return _URL_RE.findall(text)


def extract_mentions(text: str) -> List[str]:
    return _MENTION_RE.findall(text)


def extract_hashtags(text: str) -> List[str]:
    return _HASHTAG_RE.findall(text)


# ── Processor ────────────────────────────────────────────────────────────────

class SemanticProcessor:
    """Consumes events from the firehose ingestor, enriches them, and persists."""

    def __init__(self, ingestor: FirehoseIngestor) -> None:
        self._ingestor = ingestor
        self._running = False
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDS)
        self._stats = {"processed": 0, "embedded": 0, "errors": 0}

    async def start(self) -> None:
        self._running = True
        log.info("Semantic processor starting")
        await asyncio.gather(
            self._process_posts(),
            self._process_relationships(),
        )

    def stop(self) -> None:
        self._running = False
        log.info("Semantic processor stopped  stats=%s", self._stats)

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # ── post processing ──────────────────────────────────────────────────

    async def _process_posts(self) -> None:
        batch: List[FirehosePost] = []

        async for post in self._ingestor.posts():
            if not self._running:
                break
            batch.append(post)
            if len(batch) >= BATCH_SIZE:
                await self._flush_batch(batch)
                batch = []

        # flush remaining
        if batch:
            await self._flush_batch(batch)

    async def _flush_batch(self, batch: List[FirehosePost]) -> None:
        # 1. Upsert authors & store posts
        for p in batch:
            try:
                await storage.upsert_author(p.author_did)
                await storage.store_post(self._post_to_dict(p))
                self._stats["processed"] += 1
            except Exception as exc:
                log.debug("store error: %s", exc)
                self._stats["errors"] += 1

        # 2. Compute embeddings for posts with meaningful text
        embeddable = [p for p in batch if p.text and len(p.text.strip()) > 10]
        if not embeddable:
            return

        texts = [normalise_text(p.text) for p in embeddable]
        try:
            vectors = await get_embeddings_batch(texts)
            for post, vec in zip(embeddable, vectors):
                await storage.store_embedding(post.uri, vec)
                self._stats["embedded"] += 1
        except Exception as exc:
            log.warning("Embedding batch failed: %s", exc)
            self._stats["errors"] += 1

    # ── relationship processing ──────────────────────────────────────────

    async def _process_relationships(self) -> None:
        async for rel in self._ingestor.relationships():
            if not self._running:
                break
            try:
                await storage.store_relationship(
                    rel_type=rel.type,
                    subject_did=rel.subject_did,
                    object_uri=rel.object_uri,
                    object_did=rel.object_did,
                    meta=rel.meta,
                )
            except Exception as exc:
                log.debug("relationship store error: %s", exc)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _post_to_dict(p: FirehosePost) -> Dict[str, Any]:
        urls = extract_urls(p.text)
        mentions = extract_mentions(p.text)
        hashtags = extract_hashtags(p.text)

        return {
            "uri": p.uri,
            "cid": p.cid,
            "author_did": p.author_did,
            "text": p.text,
            "created_at": p.created_at,
            "reply_parent": p.reply_parent,
            "reply_root": p.reply_root,
            "embed_type": p.embed_type,
            "langs": p.langs,
            "meta": {
                "urls": urls,
                "mentions": mentions,
                "hashtags": hashtags,
            },
        }
