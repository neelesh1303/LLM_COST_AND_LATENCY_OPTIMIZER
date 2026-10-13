# LLM Cost Optimizer Gateway

A small FastAPI app that checks an in-memory semantic cache, chooses a model tier, calls a real provider, and records requests in SQLite.

## 1. Set up

Use Python 3.11 or newer. In PowerShell, run these commands from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run this once in that PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open `.env` and add one or more keys. Keep `.env` private; it is excluded by `.gitignore`.

- Groq: create a key at <https://console.groq.com/keys>.
- Gemini: create a key at <https://aistudio.google.com/app/apikey>.
- OpenRouter: create a key at <https://openrouter.ai/settings/keys>.

Only one provider key is required. A provider with no key is skipped.

The first `/chat` call downloads `all-MiniLM-L6-v2` for the semantic cache, so that call needs internet access.

## 2. Models and fallback

The model list is in `llm.py`. The starting model names are:

- Cheap: Groq `openai/gpt-oss-20b`; Gemini `gemini-3.5-flash-lite`.
- Powerful: Groq `openai/gpt-oss-120b`; Gemini `gemini-2.5-pro`; OpenRouter `nvidia/nemotron-3.5-lightning:free`.

Model availability changes. Before using the app, check the current lists at:

- <https://console.groq.com/docs/models>
- <https://ai.google.dev/gemini-api/docs/models>
- <https://openrouter.ai/models> (filter by **Free**)

The gateway tries configured models in tier order. If all fail, `/chat` returns HTTP 503; it never makes up an answer.

## 3. Cost estimate

Free-tier calls cost **$0** (subject to each provider's free quota). `cost_usd` is a **notional cost**: it shows what the usage might cost on paid plans. It is not a bill.

Prices in `router.py` are approximate paid-list prices per 1,000 tokens. Update the `PRICES` dictionary there when provider pricing changes. `/stats` compares the logged token usage with the most expensive model in the configured powerful tier. For cache hits, it uses the saved answer's original token usage for this comparison; the cache hit itself costs $0.

## 4. Run and test

Start the API:

```powershell
uvicorn main:app --reload
```

In a second PowerShell window, test the endpoints:

```powershell
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Translate hello to French"}'
curl.exe http://127.0.0.1:8000/stats
```

`/chat` returns `answer`, `model_used`, `provider`, `cache_hit`, `latency_ms`, `tokens`, and `cost_usd`. Cache-hit token counts describe the saved answer's original generation; the cache hit's notional cost is still zero.

## 5. Simple routing and cache

A prompt with fewer than 20 words that contains `translate`, `summarize`, `classify`, `fix grammar`, or `define` uses the cheap tier. Other prompts use the powerful tier.

The semantic cache is in memory, so it clears when the app restarts. Similarity of at least `0.85` reuses an answer. If the threshold is too low, unrelated prompts may share an answer; if it is too high, paraphrases are less likely to hit the cache.

## Mini demo

Run these six commands while the server is running:

```powershell
# 1. Short translation -> cheap tier
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Translate hello to French"}'

# 2. Complex request -> powerful tier
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Compare how solar and wind power affect electricity-grid reliability, and explain the tradeoffs with examples."}'

# 3. Repeat command 1 -> cache hit
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Translate hello to French"}'

# 4. Similar wording -> likely cache hit
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Translate the word hello into French"}'

# 5. Another short task -> cheap tier
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Define photosynthesis"}'

# 6. Request usage and notional savings
curl.exe http://127.0.0.1:8000/stats
```

Demo points:

1. Simple tasks are sent to a lower-cost model; other prompts use the powerful tier.
2. Repeated or sufficiently similar prompts reuse an answer without another provider call.
3. The providers' free tiers cost $0 for the demo; the displayed cost and savings are notional paid-plan estimates.
