"""
Shared LLM client factory.

Every agent needs a chat model, but which provider that actually means
depends on which API key is available (see core.config's provider
auto-detection) -- it might be Anthropic, or it might be a GitHub
Copilot/Models token, or something else entirely. Centralizing the
provider branching here means adding support for a new provider, or fixing
how an existing one is called, only has to happen in one place instead of
being duplicated across every agent file.
"""

from core.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_MODEL,
    GITHUB_BASE_URL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def make_llm(temperature: float = 0):
    """
    Build a chat model client for whichever provider core.config resolved
    to (LLM_PROVIDER). Every agent calls this the same way.
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

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=temperature)
        # openai/gpt-oss-* models on Groq are reasoning models that emit
        # hidden chain-of-thought in their response by default. Every agent
        # in this codebase parses its final message as plain text or JSON,
        # so leaked reasoning would break that -- ask Groq to hide it.
        if GROQ_MODEL and "gpt-oss" in GROQ_MODEL:
            llm = llm.bind(reasoning_format="hidden")
        return llm

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=temperature
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}. "
        "Supported values: anthropic, github, openai, groq, gemini."
    )
