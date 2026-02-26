import os
from dotenv import load_dotenv

load_dotenv()

# Bluesky credentials
BLUESKY_HANDLE: str = os.getenv("BLUESKY_HANDLE", "")
BLUESKY_PASSWORD: str = os.getenv("BLUESKY_PASSWORD", "")

# Database
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://grokker:grokker@localhost:5432/bluesky_grokker"
)

# Redis
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# API keys (set whichever provider you use)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# Embedding config
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "")  # defaults to LLM_PROVIDER
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

# LLM provider: "openai" | "anthropic" | "google" | "openrouter" | "ollama"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Firehose
FIREHOSE_URL: str = os.getenv(
    "FIREHOSE_URL", "wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos"
)

# Agent behaviour
AGENT_LOOP_INTERVAL: int = int(os.getenv("AGENT_LOOP_INTERVAL", "30"))
AGENT_ENABLED: bool = os.getenv("AGENT_ENABLED", "false").lower() == "true"

# Processing
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))
MAX_CONCURRENT_EMBEDS: int = int(os.getenv("MAX_CONCURRENT_EMBEDS", "10"))

# Swarm / multi-model reasoning
SWARM_ENABLED: bool = os.getenv("SWARM_ENABLED", "false").lower() == "true"
SWARM_STRATEGY: str = os.getenv("SWARM_STRATEGY", "all")  # "all" | "fastest" | "consensus"
SWARM_TIMEOUT_MS: int = int(os.getenv("SWARM_TIMEOUT_MS", "30000"))
SWARM_MAX_RETRIES: int = int(os.getenv("SWARM_MAX_RETRIES", "1"))
SWARM_MERGE_THRESHOLD: float = float(os.getenv("SWARM_MERGE_THRESHOLD", "0.7"))

# Warm-up
WARMUP_ENABLED: bool = os.getenv("WARMUP_ENABLED", "true").lower() == "true"
