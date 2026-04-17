"""Configuration management for ToolBlind."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Global configuration loaded from environment variables and .env file."""

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    cache_dir: str = ".cache/toolblind"
    results_dir: str = "data/results"
    tasks_dir: str = "data/tasks"
    log_level: str = "INFO"
    max_retries: int = 5
    retry_base_delay: float = 1.0
    judge_model: str = "claude-sonnet-4-20250514"
    default_sample_size: Optional[int] = None
    seed: int = 42

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables and .env file."""
        load_dotenv()

        sample_size_str = os.getenv("TOOLBLIND_DEFAULT_SAMPLE_SIZE", "")
        sample_size = int(sample_size_str) if sample_size_str.strip() else None

        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            cache_dir=os.getenv("TOOLBLIND_CACHE_DIR", ".cache/toolblind"),
            results_dir=os.getenv("TOOLBLIND_RESULTS_DIR", "data/results"),
            tasks_dir=os.getenv("TOOLBLIND_TASKS_DIR", "data/tasks"),
            log_level=os.getenv("TOOLBLIND_LOG_LEVEL", "INFO"),
            max_retries=int(os.getenv("TOOLBLIND_MAX_RETRIES", "5")),
            retry_base_delay=float(os.getenv("TOOLBLIND_RETRY_BASE_DELAY", "1.0")),
            judge_model=os.getenv("TOOLBLIND_JUDGE_MODEL", "claude-sonnet-4-20250514"),
            default_sample_size=sample_size,
            seed=int(os.getenv("TOOLBLIND_SEED", "42")),
        )


_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Override the global config (useful for testing)."""
    global _config
    _config = config
