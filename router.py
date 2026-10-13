import re

# Approximate paid-list prices in US dollars per 1,000 tokens.
# These are estimates; update them when provider prices change.
PRICES = {
    "openai/gpt-oss-20b": {"input": 0.00010, "output": 0.00050},
    "gemini-3.5-flash-lite": {"input": 0.00010, "output": 0.00040},
    "openai/gpt-oss-120b": {"input": 0.00015, "output": 0.00060},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01000},
    "nvidia/nemotron-3.5-lightning:free": {
        "input": 0.00020,
        "output": 0.00060,
    },
}

CHEAP_TASKS = ("translate", "summarize", "classify", "fix grammar", "define")


def choose_tier(prompt: str) -> str:
    words = prompt.split()
    normalized = re.sub(r"\s+", " ", prompt.lower()).strip()
    if len(words) < 20 and any(task in normalized for task in CHEAP_TASKS):
        return "cheap"
    return "powerful"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES[model]
    cost = (
        input_tokens * price["input"] + output_tokens * price["output"]
    ) / 1000
    return round(cost, 8)
