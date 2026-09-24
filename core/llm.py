"""
Shared LLM client factory.

Every agent needs a chat model, but which provider that actually means
depends on which API key is available (see core.config's provider
auto-detection): either a native Anthropic key, or a GitHub Copilot/Models
token -- those are the two this project actually needs to support.
Centralizing the provider branching here means adding a fix (or, if ever
needed, a new provider) only has to happen in one place instead of being
duplicated across every agent file.
"""

from core.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GITHUB_BASE_URL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    LLM_PROVIDER,
)


def make_llm(temperature: float = 0, caller: str = None):
    """
    Build a chat model client for whichever provider core.config resolved
    to (LLM_PROVIDER). Every agent calls this the same way.

    Args:
        temperature: sampling temperature to pass to the provider.
        caller: optional label (e.g. "bronze agent") included in the error
            message if LLM_PROVIDER is unsupported, so a failure is
            traceable to which agent hit it without needing a full
            traceback.
    """
    provider = LLM_PROVIDER

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY, temperature=temperature
        )

    if provider == "github":
        # GitHub Models -- including tokens issued via a GitHub Copilot
        # subscription -- exposes an OpenAI-compatible chat completions API
        # at an Azure AI Inference endpoint. That means it's called via the
        # OpenAI SDK (langchain_openai) with a custom base_url, not a
        # GitHub-specific client library.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=GITHUB_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_BASE_URL,
            temperature=temperature,
        )

    context = f" (requested by {caller})" if caller else ""
    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}{context}. "
        "Supported values: anthropic, github."
    )
