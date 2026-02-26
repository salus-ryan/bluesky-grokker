"""Storage layer – PostgreSQL + pgvector + Redis hot cache."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
import redis.asyncio as aioredis

from config import DATABASE_URL, EMBEDDING_DIM, REDIS_URL

log = logging.getLogger(__name__)

# ── Module-level connection handles (set by init_storage) ────────────────────
_pool: Optional[asyncpg.Pool] = None
_redis: Optional[aioredis.Redis] = None


# ── Lifecycle ────────────────────────────────────────────────────────────────

async def init_storage() -> None:
    """Create connection pool and ensure schema exists."""
    global _pool, _redis
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=4, max_size=20)
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await _ensure_schema()
    log.info("Storage layer initialised")


async def close_storage() -> None:
    global _pool, _redis
    if _pool:
        await _pool.close()
    if _redis:
        await _redis.aclose()


async def _ensure_schema() -> None:
    """Create tables and extensions if they don't exist."""
    async with _pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                did         TEXT PRIMARY KEY,
                handle      TEXT,
                display_name TEXT,
                first_seen  TIMESTAMPTZ DEFAULT now(),
                post_count  INTEGER DEFAULT 0,
                meta        JSONB DEFAULT '{}'::jsonb
            );
        """)
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS posts (
                uri         TEXT PRIMARY KEY,
                cid         TEXT NOT NULL,
                author_did  TEXT NOT NULL REFERENCES authors(did) ON DELETE CASCADE,
                text        TEXT,
                created_at  TIMESTAMPTZ NOT NULL,
                indexed_at  TIMESTAMPTZ DEFAULT now(),
                reply_parent TEXT,
                reply_root  TEXT,
                embed_type  TEXT,
                langs       TEXT[],
                meta        JSONB DEFAULT '{{}}'::jsonb,
                embedding   vector({EMBEDDING_DIM})
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id          BIGSERIAL PRIMARY KEY,
                type        TEXT NOT NULL,
                subject_did TEXT NOT NULL,
                object_uri  TEXT,
                object_did  TEXT,
                created_at  TIMESTAMPTZ DEFAULT now(),
                meta        JSONB DEFAULT '{}'::jsonb
            );
        """)
        # Indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_posts_author ON posts (author_did);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_posts_created ON posts (created_at DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_posts_reply_root ON posts (reply_root);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (type);"
        )
    log.info("Schema verified")


# ── Authors ──────────────────────────────────────────────────────────────────

async def upsert_author(did: str, handle: str = "", display_name: str = "") -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO authors (did, handle, display_name)
            VALUES ($1, $2, $3)
            ON CONFLICT (did) DO UPDATE
                SET handle = COALESCE(NULLIF($2, ''), authors.handle),
                    display_name = COALESCE(NULLIF($3, ''), authors.display_name),
                    post_count = authors.post_count + 1
            """,
            did,
            handle,
            display_name,
        )


# ── Posts ────────────────────────────────────────────────────────────────────

async def store_post(post: Dict[str, Any]) -> None:
    """Insert a post record. Upserts on conflict."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO posts (uri, cid, author_did, text, created_at,
                               reply_parent, reply_root, embed_type, langs, meta)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (uri) DO NOTHING
            """,
            post["uri"],
            post["cid"],
            post["author_did"],
            post.get("text", ""),
            post.get("created_at", datetime.now(timezone.utc)),
            post.get("reply_parent"),
            post.get("reply_root"),
            post.get("embed_type"),
            post.get("langs"),
            json.dumps(post.get("meta", {})),
        )
    # Hot cache
    if _redis:
        await _redis.setex(
            f"post:{post['uri']}", 300, json.dumps(post, default=str)
        )


async def store_embedding(uri: str, embedding: List[float]) -> None:
    """Attach an embedding vector to an existing post."""
    vec_literal = "[" + ",".join(str(v) for v in embedding) + "]"
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE posts SET embedding = $1::vector WHERE uri = $2",
            vec_literal,
            uri,
        )


async def get_post(uri: str) -> Optional[Dict[str, Any]]:
    # Try cache first
    if _redis:
        cached = await _redis.get(f"post:{uri}")
        if cached:
            return json.loads(cached)
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM posts WHERE uri = $1", uri)
    if row:
        return dict(row)
    return None


# ── Relationships ────────────────────────────────────────────────────────────

async def store_relationship(
    rel_type: str,
    subject_did: str,
    object_uri: str | None = None,
    object_did: str | None = None,
    meta: dict | None = None,
) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO relationships (type, subject_did, object_uri, object_did, meta)
            VALUES ($1, $2, $3, $4, $5)
            """,
            rel_type,
            subject_did,
            object_uri,
            object_did,
            json.dumps(meta or {}),
        )


# ── Semantic search ──────────────────────────────────────────────────────────

async def semantic_search(
    embedding: List[float], limit: int = 20
) -> List[Dict[str, Any]]:
    """Find the most similar posts by cosine distance."""
    vec_literal = "[" + ",".join(str(v) for v in embedding) + "]"
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT uri, cid, author_did, text, created_at,
                   embedding <=> $1::vector AS distance
            FROM posts
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vec_literal,
            limit,
        )
    return [dict(r) for r in rows]


# ── Thread reconstruction ────────────────────────────────────────────────────

async def get_thread(root_uri: str) -> List[Dict[str, Any]]:
    """Return all posts in a thread, ordered by creation time."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM posts
            WHERE reply_root = $1 OR uri = $1
            ORDER BY created_at
            """,
            root_uri,
        )
    return [dict(r) for r in rows]


# ── Author profile ──────────────────────────────────────────────────────────

async def get_author(did: str) -> Optional[Dict[str, Any]]:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM authors WHERE did = $1", did)
    return dict(row) if row else None


async def get_author_recent_posts(did: str, limit: int = 20) -> List[Dict[str, Any]]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM posts WHERE author_did = $1 ORDER BY created_at DESC LIMIT $2",
            did,
            limit,
        )
    return [dict(r) for r in rows]


# ── Trending ─────────────────────────────────────────────────────────────────

async def get_recent_posts(limit: int = 100) -> List[Dict[str, Any]]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM posts ORDER BY created_at DESC LIMIT $1", limit
        )
    return [dict(r) for r in rows]


# ── Stats ────────────────────────────────────────────────────────────────────

async def get_post_count() -> int:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT count(*) AS cnt FROM posts")
    return row["cnt"]
