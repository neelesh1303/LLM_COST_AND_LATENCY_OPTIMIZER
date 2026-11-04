import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from time import monotonic

import numpy as np
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.85
MAX_CACHE_ENTRIES = 500
CACHE_TTL_SECONDS = 24 * 60 * 60

# Answers to these prompts go stale quickly, so they are never cached.
TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|now|right now|current|currently|"
    r"latest|recent|this (?:week|month|year))\b"
)
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Double quotes, backticks, or single quotes that are not apostrophes.
QUOTED = re.compile(r"\"([^\"]+)\"|`([^`]+)`|(?<!\w)'([^']+)'(?!\w)")
NEGATIONS = {"not", "no", "never", "none", "nothing", "without", "cannot"}
# Words sharing a stem after these prefixes are treated as opposites:
# "enable"/"disable", "install"/"uninstall", "increase"/"decrease".
OPPOSITE_PREFIXES = ("dis", "un", "in", "im", "non", "de", "anti", "en")
MIN_STEM_LENGTH = 4


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def normalize(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.lower()).strip().rstrip("?.! ")


def is_time_sensitive(prompt: str) -> bool:
    return bool(TIME_SENSITIVE.search(prompt.lower()))


@dataclass(frozen=True)
class PromptFeatures:
    """Parts of a prompt that must agree before a similar answer is reused."""

    key: str
    words: frozenset[str]
    details: tuple[str, ...]


def extract_features(prompt: str) -> PromptFeatures:
    key = normalize(prompt)
    words = WORD.findall(key)
    negation_count = sum(
        1 for word in words if word in NEGATIONS or word.endswith("n't")
    )
    details = [f"number:{number}" for number in NUMBER.findall(key)]
    details.append(f"negations:{negation_count}")
    for match in QUOTED.finditer(key):
        quoted = next(group for group in match.groups() if group is not None)
        details.append(f"quoted:{quoted.strip()}")
    # Text after a colon is usually the input to transform ("Fix grammar: ...").
    if ":" in key:
        details.append(f"payload:{key.split(':', 1)[1].strip()}")
    return PromptFeatures(key, frozenset(words), tuple(details))


def stems(word: str) -> set[str]:
    candidates = {word}
    candidates.update(
        word[len(prefix):]
        for prefix in OPPOSITE_PREFIXES
        if word.startswith(prefix)
    )
    return {stem for stem in candidates if len(stem) >= MIN_STEM_LENGTH}


def has_opposite_words(first: PromptFeatures, second: PromptFeatures) -> bool:
    only_first = first.words - second.words
    only_second = second.words - first.words
    return any(
        stems(a) & stems(b) for a in only_first for b in only_second
    )


def is_safe_match(first: PromptFeatures, second: PromptFeatures) -> bool:
    return first.details == second.details and not has_opposite_words(
        first, second
    )


@dataclass
class CacheEntry:
    features: PromptFeatures
    embedding: np.ndarray
    answer: str
    provider: str
    model_used: str
    input_tokens: int
    output_tokens: int
    created_at: float


class SemanticCache:
    """In-memory LRU cache: exact matches first, then guarded semantic matches."""

    def __init__(
        self,
        max_entries: int = MAX_CACHE_ENTRIES,
        ttl_seconds: float = CACHE_TTL_SECONDS,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = Lock()

    def find(
        self, prompt: str
    ) -> tuple[CacheEntry | None, np.ndarray | None]:
        """Return a cached entry, and the prompt's embedding for a later add.

        The embedding is None when the prompt must not be cached.
        """
        if is_time_sensitive(prompt):
            return None, None

        features = extract_features(prompt)
        with self.lock:
            self._remove_expired()
            entry = self.entries.get(features.key)
            if entry:
                self.entries.move_to_end(features.key)
                return entry, entry.embedding

        # Encode outside the lock; it is the slow step.
        embedding = get_model().encode(
            prompt,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        with self.lock:
            if not self.entries:
                return None, embedding

            entries = list(self.entries.values())
            similarities = np.stack([e.embedding for e in entries]) @ embedding
            # Check candidates from most to least similar.
            for index in np.argsort(similarities)[::-1]:
                if similarities[index] < SIMILARITY_THRESHOLD:
                    break
                entry = entries[index]
                if is_safe_match(features, entry.features):
                    self.entries.move_to_end(entry.features.key)
                    return entry, embedding
        return None, embedding

    def add(
        self,
        prompt: str,
        embedding: np.ndarray | None,
        answer: str,
        provider: str,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if embedding is None:
            return

        features = extract_features(prompt)
        with self.lock:
            self.entries[features.key] = CacheEntry(
                features=features,
                embedding=embedding,
                answer=answer,
                provider=provider,
                model_used=model_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                created_at=monotonic(),
            )
            self.entries.move_to_end(features.key)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)

    def _remove_expired(self) -> None:
        now = monotonic()
        expired = [
            key
            for key, entry in self.entries.items()
            if now - entry.created_at > self.ttl_seconds
        ]
        for key in expired:
            del self.entries[key]


semantic_cache = SemanticCache()
