"""
Central configuration for the medallion pipeline.

Loads secrets from .env, defines every folder path the pipeline reads/writes,
and figures out which LLM provider to use. Import from this module instead of
reading os.environ or building paths by hand anywhere else in the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root (if present) into the
# environment. Real environment variables set outside .env are not overridden.
load_dotenv()

# --------------------------------------------------------------------------
# Folder paths
# --------------------------------------------------------------------------

# BASE_DIR is the project root: two levels up from this file
# (core/config.py -> core/ -> project root).
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

REPORTS_DIR = DATA_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"

# Every directory the pipeline needs to exist before it runs.
ALL_DIRS = [
    DATA_DIR,
    LANDING_DIR,
    PROFILES_DIR,
    STTM_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    REPORTS_DIR,
    AUDIT_DIR,
    CHROMA_DIR,
]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# API keys and model settings
# --------------------------------------------------------------------------

# Anthropic (Claude) - primary/default provider for this project.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Google Gemini - optional provider.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Groq - optional provider.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenAI - optional provider.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# GitHub Models - optional provider. GITHUB_MODEL uses the GA endpoint's
# publisher-prefixed naming (e.g. "openai/gpt-4.1-mini"), so GITHUB_BASE_URL
# must point at the matching GA inference endpoint (models.github.ai) --
# NOT the older preview endpoint (models.inference.ai.azure.com), which used
# bare model names and doesn't recognize publisher-prefixed IDs.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini")
GITHUB_BASE_URL = "https://models.github.ai/inference"

# --------------------------------------------------------------------------
# LLM provider auto-detection
# --------------------------------------------------------------------------


def _is_real_value(value: str) -> bool:
    """
    True if value looks like an actual secret rather than an unfilled
    .env.example placeholder (e.g. "your_anthropic_api_key_here"). Every
    placeholder in .env.example follows that "your_..._here" pattern, so a
    key left untouched is still a non-empty (truthy) Python string --
    without this check, _detect_llm_provider() below would treat an
    unfilled placeholder as a real, configured key.
    """
    if not value:
        return False
    stripped = value.strip()
    return bool(stripped) and not (stripped.startswith("your_") and stripped.endswith("_here"))


def _detect_llm_provider() -> str:
    """
    Decide which LLM provider to use.

    1. If LLM_PROVIDER is explicitly set in the environment, use that.
    2. Otherwise, pick the first provider (in preference order) that has a
       REAL API key configured (not just an unfilled .env.example
       placeholder): anthropic > github. These are the two providers
       core.llm.make_llm() actually implements -- there's no point
       detecting a provider (e.g. a filled-in GROQ_API_KEY) that the
       pipeline can't actually build a client for.
    3. If nothing is configured, fall back to "anthropic" (the project
       default) so downstream code has a consistent value to check against.
    """
    explicit_provider = os.getenv("LLM_PROVIDER")
    if explicit_provider:
        return explicit_provider.strip().lower()

    if _is_real_value(ANTHROPIC_API_KEY):
        return "anthropic"
    if _is_real_value(GITHUB_TOKEN):
        return "github"

    return "anthropic"


LLM_PROVIDER = _detect_llm_provider()
