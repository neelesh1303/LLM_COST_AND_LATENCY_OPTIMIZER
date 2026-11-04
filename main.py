from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cache import get_model, semantic_cache
from db import get_totals, init_db, log_request
from llm import LLMUnavailableError, MODELS, generate, provider_for_model
from router import PRICES, calculate_cost, choose_tier

app = FastAPI(title="LLM Cost Optimizer Gateway")
MAX_PROMPT_CHARS = 4000

# Use the most expensive configured powerful model as the comparison baseline.
POWERFUL_REFERENCE_MODEL = max(
    (choice["model"] for choice in MODELS["powerful"]),
    key=lambda model: PRICES[model]["input"] + PRICES[model]["output"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)


@app.on_event("startup")
def start_app() -> None:
    init_db()
    # Load the embedding model now so the first /chat call is not slow.
    get_model()


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty.")

    started = perf_counter()
    cached, embedding = semantic_cache.find(prompt)

    if cached:
        answer = cached.answer
        provider = cached.provider
        model_used = cached.model_used
        input_tokens = cached.input_tokens
        output_tokens = cached.output_tokens
        cache_hit = True
        cost = 0.0
    else:
        tier = choose_tier(prompt)
        try:
            answer, model_used, input_tokens, output_tokens = generate(
                prompt, tier
            )
        except LLMUnavailableError as error:
            latency_ms = round((perf_counter() - started) * 1000, 2)
            log_request(
                prompt=prompt,
                provider="unavailable",
                model_used="unavailable",
                cache_hit=False,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=latency_ms,
                success=False,
            )
            raise HTTPException(status_code=503, detail=str(error)) from error

        provider = provider_for_model(model_used)
        cache_hit = False
        cost = calculate_cost(model_used, input_tokens, output_tokens)
        semantic_cache.add(
            prompt,
            embedding,
            answer,
            provider,
            model_used,
            input_tokens,
            output_tokens,
        )

    latency_ms = round((perf_counter() - started) * 1000, 2)
    log_request(
        prompt=prompt,
        provider=provider,
        model_used=model_used,
        cache_hit=cache_hit,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        latency_ms=latency_ms,
    )

    return {
        "answer": answer,
        "model_used": model_used,
        "provider": provider,
        "cache_hit": cache_hit,
        "latency_ms": latency_ms,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
        "cost_usd": cost,
    }


@app.get("/stats")
def stats() -> dict:
    (
        total,
        cache_hits,
        notional_cost,
        input_tokens,
        output_tokens,
        failed_requests,
        average_latency_ms,
    ) = get_totals()
    powerful_cost = calculate_cost(
        POWERFUL_REFERENCE_MODEL,
        input_tokens,
        output_tokens,
    )
    savings_percent = (
        round((1 - notional_cost / powerful_cost) * 100, 2)
        if powerful_cost
        else 0.0
    )

    return {
        "total_requests": total,
        "failed_requests": failed_requests,
        "average_latency_ms": round(average_latency_ms, 2),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
        "total_notional_cost_usd": notional_cost,
        "cost_if_all_requests_used_powerful_model_usd": powerful_cost,
        "savings_percent": savings_percent,
    }
