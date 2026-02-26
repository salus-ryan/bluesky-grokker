"""Main Runtime – boots firehose ingestor, semantic processor, and agent loop
as concurrent async tasks."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import List

from config import AGENT_ENABLED, BLUESKY_HANDLE, BLUESKY_PASSWORD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-24s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("grokker")


async def main() -> None:
    # ── Late imports so config / logging are ready ───────────────────────
    import storage
    from firehose import FirehoseIngestor
    from processor import SemanticProcessor
    from agent import GrokkerAgent
    from bluesky_client import BlueskyClient

    # 1. Storage
    log.info("Initialising storage…")
    await storage.init_storage()

    # 1b. Warm-up LLM + embedding providers
    from warmup import run_warmup
    warmup_result = await run_warmup()
    if not warmup_result.get("skipped") and not warmup_result.get("all_ok"):
        log.warning("Some warm-up checks failed – Grokker will start but may have issues")

    # 2. Firehose
    ingestor = FirehoseIngestor()

    # 3. Processor
    processor = SemanticProcessor(ingestor)

    # 4. Agent (optional)
    agent: GrokkerAgent | None = None
    if AGENT_ENABLED and BLUESKY_HANDLE and BLUESKY_PASSWORD:
        bsky_client = BlueskyClient()
        await asyncio.get_event_loop().run_in_executor(None, bsky_client.login)
        agent = GrokkerAgent(bsky_client)
        log.info("Agent enabled – will respond and post autonomously")
    else:
        log.info("Agent disabled – running in ingest-only mode")

    # ── Graceful shutdown ────────────────────────────────────────────────
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        log.info("Shutdown signal received")
        shutdown_event.set()
        ingestor.stop()
        processor.stop()
        if agent:
            agent.stop()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # ── Launch tasks ─────────────────────────────────────────────────────
    tasks: List[asyncio.Task] = [
        asyncio.create_task(ingestor.start(), name="firehose"),
        asyncio.create_task(processor.start(), name="processor"),
    ]
    if agent:
        tasks.append(asyncio.create_task(agent.start(), name="agent"))

    # Periodic stats reporter
    async def _report_stats() -> None:
        while not shutdown_event.is_set():
            await asyncio.sleep(30)
            post_count = 0
            try:
                post_count = await storage.get_post_count()
            except Exception:
                pass
            log.info(
                "stats  firehose=%s  processor=%s  db_posts=%d%s",
                ingestor.stats,
                processor.stats,
                post_count,
                f"  agent={agent.stats}" if agent else "",
            )

    tasks.append(asyncio.create_task(_report_stats(), name="stats"))

    log.info("Bluesky-Grokker is live 🚀")

    # Wait for shutdown
    await shutdown_event.wait()

    # Cancel running tasks
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await storage.close_storage()
    log.info("Bluesky-Grokker shut down cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
