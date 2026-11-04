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

CHEAP_TASKS = (
    "translate",
    "summarize",
    "summarise",
    "classify",
    "fix grammar",
    "fix the grammar",
    "define",
)
POLITE_PREFIXES = ("please", "kindly", "can you", "could you", "would you")

# The task must be the prompt's leading verb, as a whole word:
# "Please translate hello" matches; "Why is this hard to translate?" and
# "Why is x undefined?" do not.
CHEAP_TASK_PATTERN = re.compile(
    r"^(?:(?:" + "|".join(map(re.escape, POLITE_PREFIXES)) + r")\s+)*"
    r"(?:" + "|".join(map(re.escape, CHEAP_TASKS)) + r")\b"
)


def choose_tier(prompt: str) -> str:
    words = prompt.split()
    normalized = re.sub(r"\s+", " ", prompt.lower())
    normalized = re.sub(r"^\W+", "", normalized)
    if len(words) < 20 and CHEAP_TASK_PATTERN.match(normalized):
        return "cheap"
    return "powerful"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES[model]
    cost = (
        input_tokens * price["input"] + output_tokens * price["output"]
    ) / 1000
    return round(cost, 8)
