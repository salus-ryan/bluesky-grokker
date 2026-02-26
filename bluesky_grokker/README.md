# Bluesky-Grokker

**A real-time intelligence system built natively on the Bluesky AT Protocol.**

Bluesky-Grokker continuously ingests the global Bluesky firehose, constructs a live semantic model of discourse, and exposes APIs and agent interfaces for querying, reasoning, and acting on that model.

## Architecture

| Component | File | Purpose |
|-----------|------|---------|
| Firehose Ingestor | `firehose.py` | Subscribes to `com.atproto.sync.subscribeRepos`, parses posts/likes/reposts/follows |
| Semantic Processor | `processor.py` | Normalises text, computes embeddings, extracts entities, writes to DB |
| Storage Layer | `storage.py` | PostgreSQL + pgvector + Redis hot cache |
| Query Engine | `query.py` | Vector similarity search, thread reconstruction, trending topics |
| Bluesky Client | `bluesky_client.py` | Login, post, reply, read notifications via AT Protocol |
| Agent Layer | `agent.py` | Autonomous agent – monitors discourse, reasons via LLM, posts/replies |
| Embedding Model | `models/embedding.py` | OpenAI or Ollama embedding abstraction |
| Main Runtime | `main.py` | Async orchestrator for all subsystems |

## Prerequisites

- **Python 3.11+**
- **PostgreSQL** with the [pgvector](https://github.com/pgvector/pgvector) extension
- **Redis**
- An **OpenAI API key** (or a local **Ollama** instance for embeddings/LLM)

## Quick Start

### 1. Install dependencies

```bash
cd bluesky_grokker
pip install -r requirements.txt
```

### 2. Set up PostgreSQL

```sql
CREATE DATABASE bluesky_grokker;
CREATE USER grokker WITH PASSWORD 'grokker';
GRANT ALL PRIVILEGES ON DATABASE bluesky_grokker TO grokker;
-- pgvector extension is created automatically on first run
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 4. Run

```bash
python main.py
```

## Configuration

All configuration is via environment variables (or a `.env` file). See `config.py` for the full list:

| Variable | Default | Description |
|----------|---------|-------------|
| `BLUESKY_HANDLE` | | Your Bluesky handle (e.g. `user.bsky.social`) |
| `BLUESKY_PASSWORD` | | App password |
| `DATABASE_URL` | `postgresql://grokker:grokker@localhost:5432/bluesky_grokker` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `OPENAI_API_KEY` | | OpenAI API key (if using OpenAI provider) |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model name for chat completions |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `AGENT_ENABLED` | `false` | Set to `true` to enable autonomous agent |
| `AGENT_LOOP_INTERVAL` | `30` | Seconds between agent polling cycles |

## Modes of Operation

### Ingest-only (default)

With `AGENT_ENABLED=false`, the system ingests the firehose, computes embeddings, and populates the database. You can query it programmatically via `query.py`.

### Agent mode

With `AGENT_ENABLED=true` and valid Bluesky credentials, the agent will:

1. Poll notifications for mentions/replies
2. Assemble semantic context from the database
3. Generate responses via the configured LLM
4. Post replies back to Bluesky

## Querying (Programmatic)

```python
import asyncio
from query import semantic_search, get_trending_topics, get_thread
from storage import init_storage

async def demo():
    await init_storage()
    results = await semantic_search("decentralized social media")
    for r in results:
        print(r["text"][:100], r["distance"])

asyncio.run(demo())
```

## Performance Targets

- **Ingestion**: 100+ posts/sec
- **Ingestion latency**: < 2 seconds from post to indexed embedding
- **Semantic query**: < 500 ms response time

## Roadmap

### Phase 1 (MVP) ✅
- Firehose ingestion
- Post storage + embeddings
- Semantic search
- Autonomous agent posting

### Phase 2
- Trend detection
- Influence modelling
- Topic clustering
- Conversation prediction

### Phase 3
- Multi-agent reasoning
- Persistent discourse ontology
- Autonomous posting strategies

## License

MIT
