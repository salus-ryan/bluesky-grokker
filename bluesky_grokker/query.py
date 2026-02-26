"""Semantic Query Engine – vector similarity search, thread reconstruction,
author profiling, and trending topic detection."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from models.embedding import get_embedding
from processor import extract_hashtags
import storage

log = logging.getLogger(__name__)


async def semantic_search(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Embed *query_text* and return the most semantically similar posts."""
    vec = await get_embedding(query_text)
    results = await storage.semantic_search(vec, limit=limit)
    return results


async def get_thread(thread_root_uri: str) -> List[Dict[str, Any]]:
    """Reconstruct a full thread starting from *thread_root_uri*."""
    return await storage.get_thread(thread_root_uri)


async def get_author_profile(did: str) -> Dict[str, Any]:
    """Return author info together with recent posts."""
    author = await storage.get_author(did)
    posts = await storage.get_author_recent_posts(did, limit=30)
    return {
        "author": author,
        "recent_posts": posts,
        "post_count": len(posts),
    }


async def get_trending_topics(limit: int = 20) -> List[Dict[str, Any]]:
    """Scan recent posts for the most common hashtags."""
    recent = await storage.get_recent_posts(limit=500)
    counter: Counter[str] = Counter()
    for post in recent:
        text = post.get("text", "")
        for tag in extract_hashtags(text):
            counter[tag.lower()] += 1

    return [
        {"topic": tag, "count": count}
        for tag, count in counter.most_common(limit)
    ]


async def discourse_evolution(
    query_text: str, windows: int = 5, window_size: int = 50
) -> List[Dict[str, Any]]:
    """Return snapshots of how discourse around *query_text* has shifted over
    the most recent *windows* × *window_size* posts.  Each window contains the
    top matches and a summary of prevalent hashtags."""
    recent = await storage.get_recent_posts(limit=windows * window_size)

    snapshots: List[Dict[str, Any]] = []
    for i in range(0, len(recent), window_size):
        chunk = recent[i : i + window_size]
        counter: Counter[str] = Counter()
        for post in chunk:
            for tag in extract_hashtags(post.get("text", "")):
                counter[tag.lower()] += 1
        snapshots.append(
            {
                "window": i // window_size,
                "post_count": len(chunk),
                "top_hashtags": counter.most_common(10),
                "time_range": {
                    "start": str(chunk[-1].get("created_at", "")) if chunk else None,
                    "end": str(chunk[0].get("created_at", "")) if chunk else None,
                },
            }
        )
    return snapshots
