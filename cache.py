from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.85
MAX_CACHE_ENTRIES = 500


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class CacheEntry:
    embedding: np.ndarray
    answer: str
    provider: str
    model_used: str
    input_tokens: int
    output_tokens: int


class SemanticCache:
    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []

    def find(
        self, prompt: str
    ) -> tuple[CacheEntry | None, np.ndarray]:
        embedding = get_model().encode(
            prompt,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if not self.entries:
            return None, embedding

        similarities = np.array(
            [entry.embedding @ embedding for entry in self.entries]
        )
        best_index = int(np.argmax(similarities))
        if similarities[best_index] >= SIMILARITY_THRESHOLD:
            return self.entries[best_index], embedding
        return None, embedding

    def add(
        self,
        embedding: np.ndarray,
        answer: str,
        provider: str,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.entries.append(
            CacheEntry(
                embedding=embedding,
                answer=answer,
                provider=provider,
                model_used=model_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)


semantic_cache = SemanticCache()
