"""Bluesky Posting Interface – login, post, reply, and read notifications
using the official AT Protocol endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from atproto import Client, models

from config import BLUESKY_HANDLE, BLUESKY_PASSWORD

log = logging.getLogger(__name__)


class BlueskyClient:
    """Thin async-friendly wrapper around the atproto SDK Client."""

    def __init__(
        self,
        handle: str = BLUESKY_HANDLE,
        password: str = BLUESKY_PASSWORD,
    ) -> None:
        self._handle = handle
        self._password = password
        self._client: Optional[Client] = None
        self._did: Optional[str] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def login(self) -> str:
        """Authenticate and return our DID."""
        self._client = Client()
        profile = self._client.login(self._handle, self._password)
        self._did = profile.did
        log.info("Logged in as %s (%s)", self._handle, self._did)
        return self._did

    @property
    def did(self) -> Optional[str]:
        return self._did

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("BlueskyClient not logged in – call login() first")
        return self._client

    # ── posting ──────────────────────────────────────────────────────────

    def create_post(self, text: str, langs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new top-level post and return the response dict."""
        resp = self.client.send_post(text=text, langs=langs or ["en"])
        log.info("Posted: %.60s…  uri=%s", text, resp.uri)
        return {"uri": resp.uri, "cid": resp.cid}

    def reply_to_post(
        self,
        text: str,
        parent_uri: str,
        parent_cid: str,
        root_uri: Optional[str] = None,
        root_cid: Optional[str] = None,
        langs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Reply to an existing post."""
        root_ref = models.AppBskyFeedPost.ReplyRef(
            parent=models.ComAtprotoRepoStrongRef.Main(
                uri=parent_uri, cid=parent_cid
            ),
            root=models.ComAtprotoRepoStrongRef.Main(
                uri=root_uri or parent_uri,
                cid=root_cid or parent_cid,
            ),
        )
        resp = self.client.send_post(
            text=text,
            reply_to=root_ref,
            langs=langs or ["en"],
        )
        log.info("Replied: %.60s…  uri=%s", text, resp.uri)
        return {"uri": resp.uri, "cid": resp.cid}

    # ── reading ──────────────────────────────────────────────────────────

    def get_notifications(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch recent notifications for the logged-in account."""
        resp = self.client.app.bsky.notification.list_notifications(
            params={"limit": limit}
        )
        notifications: List[Dict[str, Any]] = []
        for n in resp.notifications:
            notifications.append(
                {
                    "uri": n.uri,
                    "cid": n.cid,
                    "author_did": n.author.did,
                    "author_handle": n.author.handle,
                    "reason": n.reason,
                    "text": getattr(n.record, "text", None),
                    "indexed_at": str(n.indexed_at),
                    "is_read": n.is_read,
                }
            )
        return notifications

    def get_post_thread(self, uri: str, depth: int = 6) -> Dict[str, Any]:
        """Fetch a post thread from the Bluesky API."""
        resp = self.client.app.bsky.feed.get_post_thread(
            params={"uri": uri, "depth": depth}
        )
        return self._thread_to_dict(resp.thread)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _thread_to_dict(thread: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if hasattr(thread, "post"):
            p = thread.post
            result["post"] = {
                "uri": p.uri,
                "cid": p.cid,
                "author_did": p.author.did,
                "author_handle": p.author.handle,
                "text": getattr(p.record, "text", ""),
                "created_at": str(getattr(p.record, "created_at", "")),
                "like_count": getattr(p, "like_count", 0),
                "reply_count": getattr(p, "reply_count", 0),
                "repost_count": getattr(p, "repost_count", 0),
            }
        if hasattr(thread, "replies") and thread.replies:
            result["replies"] = [
                BlueskyClient._thread_to_dict(r) for r in thread.replies
            ]
        return result
