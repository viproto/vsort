"""Configuration management for VSort.

Handles loading, saving, and reading the config file stored at
~/.vsort/config.json. Provides typed access to all settings.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".vsort"
CONFIG_FILE = CONFIG_DIR / "config.json"
MANIFESTS_DIR = CONFIG_DIR / "manifests"

# Legacy path — used for one-time migration from the old name
_LEGACY_CONFIG_DIR = Path.home() / ".ai-file-sorter"

DEFAULT_PORT = 8081
DEFAULT_HOST = "127.0.0.1"
DEFAULT_TEXT_SIZE_LIMIT = 50 * 1024  # 50 KB
DEFAULT_SAMPLE_CHARS = 2000  # chars of text content to send to SLM

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
LLAMA_CPP_RELEASE_TAG = "b5060"  # recent stable tag; updated at runtime if needed
MMPROJ_FILENAME = "mmproj-F16.gguf"

# ── Model variants ──────────────────────────────────────────────────────

MODEL_VARIANTS = {
    "E2B": {
        "repo": "unsloth/gemma-4-E2B-it-GGUF",
        "label": "Gemma 4 2B (fast, less accurate)",
    },
    "E4B": {
        "repo": "unsloth/gemma-4-E4B-it-GGUF",
        "label": "Gemma 4 4B (slower, more accurate)",
    },
}

# ── Quantization options ───────────────────────────────────────────────

QUANTIZATION_OPTIONS = {
    "UD-IQ2_M": {"label": "Ultra-compact IQ2_M (~2GB E2B, ~3.4GB E4B)", "tier": "smallest"},
    "UD-IQ3_XXS": {"label": "Very compact IQ3_XXS (~2.3GB E2B, ~3.5GB E4B)", "tier": "tiny"},
    "IQ4_XS": {"label": "Compact IQ4_XS (~2.8GB E2B, ~4.5GB E4B)", "tier": "small"},
    "Q4_K_M": {"label": "Balanced Q4_K_M (~3GB E2B, ~4.7GB E4B) — recommended", "tier": "balanced"},
    "Q5_K_M": {"label": "Higher quality Q5_K_M (~3.2GB E2B, ~5.2GB E4B)", "tier": "good"},
    "Q6_K": {"label": "High quality Q6_K (~4.3GB E2B, ~6.7GB E4B)", "tier": "high"},
    "Q8_0": {"label": "Near-lossless Q8_0 (~4.8GB E2B, ~7.8GB E4B)", "tier": "best"},
    "BF16": {"label": "Uncompressed BF16 (~8.9GB E2B, ~14.4GB E4B)", "tier": "lossless"},
}


def get_model_repo_id(model_variant: str = "E2B") -> str:
    """Return the HuggingFace repo ID for the given model variant."""
    return MODEL_VARIANTS[model_variant]["repo"]


def get_model_filename_glob(model_variant: str = "E2B", quantization: str = "UD-IQ2_M") -> str:
    """Return a glob pattern for the model filename.

    Pattern: gemma-4-{variant}-it-{quant}.gguf
    """
    return f"gemma-4-{model_variant}-it-{quantization}.gguf"


# Backward-compatible aliases (used by setup.py imports)
MODEL_REPO_ID = MODEL_VARIANTS["E2B"]["repo"]
MODEL_FILENAME_GLOB = get_model_filename_glob()

# Sorting modes: control how much file content is sent to the SLM
SORT_MODES = {
    "rapid": "Only filenames, extensions, and sizes — fastest, least accurate",
    "normal": "Filenames + truncated content for small text files (default)",
    "full": "Full content for text files (up to 200KB) — slower, most accurate",
}

# ── Sort strategies (steering) ─────────────────────────────────────────

SORT_STRATEGIES = {
    "auto": "Let the model decide how to group (default)",
    "content": "Group by content, topic, or theme — ignore dates/metadata. "
              "Use this when you want 'Vacation-Photos' not 'June-15'",
    "date": "Group by time periods (month, season, year). "
            "Use this for chronological organization like '2025-June'",
}

# Full-mode limits
FULL_TEXT_SIZE_LIMIT = 200 * 1024  # 200 KB
FULL_SAMPLE_CHARS = 8000  # more chars in full mode

# Default exclusion patterns (skipped during sorting)
DEFAULT_EXCLUSIONS = [
    "*.tmp", "*.bak", "*.part", "*.crdownload",
    "desktop.ini", "Thumbs.db", ".DS_Store",
]


@dataclass
class SortDirectory:
    """A directory the user wants sorted."""
    path: str
    schedule: str = "one-time"  # "one-time" or "periodic"
    schedule_interval: str = "daily"  # used when schedule=="periodic"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    initialized: bool = False
    setup_complete: bool = False
    directories: List[Dict[str, str]] = field(default_factory=list)
    llama_server_path: Optional[str] = None
    model_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    text_size_limit: int = DEFAULT_TEXT_SIZE_LIMIT
    sample_chars: int = DEFAULT_SAMPLE_CHARS
    llama_cpp_tag: str = LLAMA_CPP_RELEASE_TAG
    ctx_size: int = 8192  # context window for llama-server
    sort_mode: str = "normal"  # "rapid", "normal", or "full"
    sort_strategy: str = "auto"  # "auto", "content", or "date" — steering for the SLM
    model_variant: str = "E2B"  # "E2B" or "E4B"
    quantization: str = "UD-IQ2_M"  # quantization format for the GGUF model
    yolo: bool = False  # YOLO mode: max context, no batching
    think: bool = False  # Show model reasoning during sorting
    dry_run: bool = False  # Preview categorization without moving files
    rename: bool = False  # Ask SLM to suggest better filenames
    exclusions: List[str] = field(default_factory=list)  # glob patterns to skip

    # ── helpers ──────────────────────────────────────────────────────

    def get_sort_directories(self) -> List[SortDirectory]:
        """Convert raw dicts into typed SortDirectory objects."""
        return [SortDirectory(**d) for d in self.directories]

    def set_sort_directories(self, dirs: List[SortDirectory]) -> None:
        """Store typed SortDirectory objects as plain dicts."""
        self.directories = [asdict(d) for d in dirs]

    def effective_text_size_limit(self) -> int:
        """Return the text size limit for the current sort mode."""
        if self.sort_mode == "full":
            return FULL_TEXT_SIZE_LIMIT
        return self.text_size_limit

    def effective_sample_chars(self) -> int:
        """Return the sample chars limit for the current sort mode."""
        if self.sort_mode == "full":
            return FULL_SAMPLE_CHARS
        return self.sample_chars

    # ── persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        """Write current config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
            tmp.replace(CONFIG_FILE)
            logger.debug("Config saved to %s", CONFIG_FILE)
        except OSError as exc:
            logger.error("Failed to save config: %s", exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    # ── class methods ────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "AppConfig":
        """Load config from disk, returning defaults if not yet created.

        Also performs a one-time migration from the legacy ~/.ai-file-sorter
        directory if the new ~/.vsort directory doesn't exist yet.
        """
        # One-time migration from old config location
        if not CONFIG_DIR.exists() and _LEGACY_CONFIG_DIR.exists():
            try:
                import shutil
                shutil.move(str(_LEGACY_CONFIG_DIR), str(CONFIG_DIR))
                logger.info("Migrated config from %s to %s", _LEGACY_CONFIG_DIR, CONFIG_DIR)
            except OSError as exc:
                logger.warning("Could not migrate config directory: %s", exc)

        if CONFIG_FILE.exists():
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Config file corrupt (%s); using defaults.", exc)
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Return True if a config file already exists on disk."""
        return CONFIG_FILE.exists()

    @classmethod
    def config_dir(cls) -> Path:
        """Return the config directory path, creating it if needed."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return CONFIG_DIR
