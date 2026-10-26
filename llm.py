import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Check the providers' model lists; free models can change.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_name": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_name": "GEMINI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_name": "OPENROUTER_API_KEY",
    },
}

PROVIDER_TIMEOUT_SECONDS = 30.0

MODELS = {
    "cheap": [
        {"provider": "groq", "model": "openai/gpt-oss-20b"},
        {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
    ],
    "powerful": [
        {"provider": "groq", "model": "openai/gpt-oss-120b"},
        {"provider": "gemini", "model": "gemini-2.5-pro"},
        {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free"},
    ],
}


class LLMUnavailableError(Exception):
    """Raised when no configured provider can answer the prompt."""


def provider_for_model(model: str) -> str:
    for choices in MODELS.values():
        for choice in choices:
            if choice["model"] == model:
                return choice["provider"]
    raise ValueError(f"Model is not configured: {model}")


def generate(prompt: str, tier: str) -> tuple[str, str, int, int]:
    """Try each configured provider for the selected tier in order."""
    if tier not in MODELS:
        raise ValueError(f"Unknown model tier: {tier}")

    configured = []
    failures = []

    for choice in MODELS[tier]:
        provider = choice["provider"]
        model = choice["model"]
        provider_config = PROVIDERS[provider]
        api_key = os.getenv(provider_config["key_name"])

        if not api_key:
            continue

        configured.append(f"{provider}/{model}")

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=provider_config["base_url"],
                timeout=PROVIDER_TIMEOUT_SECONDS,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content
            usage = response.usage
            if answer is None or usage is None:
                raise RuntimeError("The provider returned no answer or token usage.")

            return (
                answer,
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
            )
        except Exception as error:
            print(f"Warning: {provider}/{model} failed: {error}")
            failures.append(f"{provider}/{model} ({type(error).__name__})")

    if not configured:
        raise LLMUnavailableError(
            f"No API keys are configured for the {tier} tier. "
            "Add at least one provider key to .env."
        )

    raise LLMUnavailableError(
        f"All configured {tier} models failed: {', '.join(failures)}."
    )
