"""Firehose Ingestor – connects to com.atproto.sync.subscribeRepos and emits
normalised event objects for downstream processing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from atproto import (
    CAR,
    AtUri,
    FirehoseSubscribeReposClient,
    firehose_models,
    models,
    parse_subscribe_repos_message,
)

from config import FIREHOSE_URL

log = logging.getLogger(__name__)


# ── Normalised event types ───────────────────────────────────────────────────

@dataclass
class FirehosePost:
    uri: str
    cid: str
    author_did: str
    text: str
    created_at: datetime
    reply_parent: Optional[str] = None
    reply_root: Optional[str] = None
    embed_type: Optional[str] = None
    langs: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FirehoseRelationship:
    type: str  # "like", "repost", "follow"
    subject_did: str
    object_uri: Optional[str] = None
    object_did: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    meta: Dict[str, Any] = field(default_factory=dict)


# ── Firehose consumer ───────────────────────────────────────────────────────

class FirehoseIngestor:
    """Subscribes to the Bluesky firehose and yields normalised events."""

    def __init__(self) -> None:
        self._client = FirehoseSubscribeReposClient()
        self._post_queue: asyncio.Queue[FirehosePost] = asyncio.Queue(maxsize=10_000)
        self._rel_queue: asyncio.Queue[FirehoseRelationship] = asyncio.Queue(maxsize=10_000)
        self._running = False
        self._stats = {"commits": 0, "posts": 0, "likes": 0, "reposts": 0, "follows": 0}

    # ── public API ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start consuming the firehose in a background task."""
        self._running = True
        log.info("Firehose ingestor starting – connecting to %s", FIREHOSE_URL)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._run_sync)

    def stop(self) -> None:
        self._running = False
        self._client.stop()
        log.info("Firehose ingestor stopped  stats=%s", self._stats)

    async def posts(self) -> AsyncIterator[FirehosePost]:
        """Async iterator that yields posts as they arrive."""
        while self._running or not self._post_queue.empty():
            try:
                post = await asyncio.wait_for(self._post_queue.get(), timeout=1.0)
                yield post
            except asyncio.TimeoutError:
                continue

    async def relationships(self) -> AsyncIterator[FirehoseRelationship]:
        while self._running or not self._rel_queue.empty():
            try:
                rel = await asyncio.wait_for(self._rel_queue.get(), timeout=1.0)
                yield rel
            except asyncio.TimeoutError:
                continue

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # ── internal ─────────────────────────────────────────────────────────

    def _run_sync(self) -> None:
        """Blocking call that the atproto SDK requires; we run it in an executor."""
        self._client.start(self._on_message)

    def _on_message(self, message: firehose_models.MessageFrame) -> None:
        """Callback executed for every firehose frame."""
        if not self._running:
            self._client.stop()
            return

        commit = parse_subscribe_repos_message(message)
        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            return

        self._stats["commits"] += 1

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

            uri = f"at://{commit.repo}/{op.path}"
            self._route_record(uri, str(op.cid), commit.repo, op.path, raw)

    def _route_record(
        self, uri: str, cid: str, repo: str, path: str, record: dict
    ) -> None:
        record_type = record.get("$type", "")

        if record_type == "app.bsky.feed.post":
            self._handle_post(uri, cid, repo, record)
        elif record_type == "app.bsky.feed.like":
            self._handle_like(uri, repo, record)
        elif record_type == "app.bsky.feed.repost":
            self._handle_repost(uri, repo, record)
        elif record_type == "app.bsky.graph.follow":
            self._handle_follow(uri, repo, record)

    # ── record handlers ──────────────────────────────────────────────────

    def _handle_post(self, uri: str, cid: str, repo: str, record: dict) -> None:
        reply_parent = None
        reply_root = None
        reply = record.get("reply")
        if reply:
            parent = reply.get("parent", {})
            root = reply.get("root", {})
            reply_parent = parent.get("uri") if isinstance(parent, dict) else None
            reply_root = root.get("uri") if isinstance(root, dict) else None

        embed_type = None
        embed = record.get("embed")
        if embed:
            embed_type = embed.get("$type")

        created_str = record.get("createdAt", "")
        try:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except Exception:
            created_at = datetime.now(timezone.utc)

        post = FirehosePost(
            uri=uri,
            cid=cid,
            author_did=repo,
            text=record.get("text", ""),
            created_at=created_at,
            reply_parent=reply_parent,
            reply_root=reply_root,
            embed_type=embed_type,
            langs=record.get("langs") or [],
        )

        try:
            self._post_queue.put_nowait(post)
            self._stats["posts"] += 1
        except asyncio.QueueFull:
            pass  # drop under back-pressure

    def _handle_like(self, uri: str, repo: str, record: dict) -> None:
        subject = record.get("subject", {})
        rel = FirehoseRelationship(
            type="like",
            subject_did=repo,
            object_uri=subject.get("uri") if isinstance(subject, dict) else None,
        )
        try:
            self._rel_queue.put_nowait(rel)
            self._stats["likes"] += 1
        except asyncio.QueueFull:
            pass

    def _handle_repost(self, uri: str, repo: str, record: dict) -> None:
        subject = record.get("subject", {})
        rel = FirehoseRelationship(
            type="repost",
            subject_did=repo,
            object_uri=subject.get("uri") if isinstance(subject, dict) else None,
        )
        try:
            self._rel_queue.put_nowait(rel)
            self._stats["reposts"] += 1
        except asyncio.QueueFull:
            pass

    def _handle_follow(self, uri: str, repo: str, record: dict) -> None:
        rel = FirehoseRelationship(
            type="follow",
            subject_did=repo,
            object_did=record.get("subject"),
        )
        try:
            self._rel_queue.put_nowait(rel)
            self._stats["follows"] += 1
        except asyncio.QueueFull:
            pass
