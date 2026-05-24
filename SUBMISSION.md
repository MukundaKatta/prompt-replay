# prompt-replay: Hermes Agent Challenge submission

Track: Hermes Agent Challenge (dev.to)
Author: Mukunda Katta
Repo: https://github.com/MukundaKatta/prompt-replay
License: MIT

---

## Record once, replay anywhere: A/B testing prompts and models without a notebook

Every team that ships LLM features hits the same wall on day 90.

You picked a model. You tuned a prompt. Production is stable. Then a faster, cheaper model lands and you want to switch. Or you change one sentence in the system prompt and want to know which of last week's 400 user prompts now answer differently. Or a provider deprecates a model and you have a week to decide what to migrate to.

The standard answer is "spin up a Jupyter notebook, paste in some prompts, eyeball the diff." That works for the first 5 prompts. It does not work when you have 400 of them, and it does not catch the cases where the new model is 95 percent the same and 5 percent silently wrong.

`prompt-replay` is a small Python library for the obvious flow: record real prompts during a normal run, then later replay every one of them through any other provider or model and get a structured comparison report. Zero runtime dependencies. Python 3.10+.

## Two phases, one store

The whole library is built around two phases and one persistent store between them.

Phase one is recording. You wrap your provider call with a decorator. Every invocation gets captured as a `RecordedEntry` containing the full request shape (model, messages, tools, temperature, max_tokens, every kwarg) and the response your provider returned. The default store is `JsonlStore`, which is one JSON object per line, append-only, crash-safe. Use `InMemoryStore` in tests.

```python
from prompt_replay import Recorder, JsonlStore

recorder = Recorder(store=JsonlStore("prompts.jsonl"))

@recorder.capture
def call(model, messages, **kw):
    return provider.messages.create(model=model, messages=messages, **kw)

call("claude-sonnet-4-7", [{"role": "user", "content": "weather in NYC"}])
```

That is the entire setup. The decorator binds positional args to their parameter names via `inspect.signature`, so the captured request always has `request["model"]`, even when you called it positionally. Async functions work the same way.

Phase two is replay. You point a `Replayer` at the same store, pass a runner that knows how to send a request dict back to the provider, and optionally pass an override that swaps one field.

```python
from prompt_replay import Replayer, JsonlStore

replayer = Replayer(JsonlStore("prompts.jsonl"))

report = replayer.replay(
    override={"model": "claude-haiku-4-5"},
    runner=lambda req: provider.messages.create(**req),
)

print(report.as_table())
```

The override accepts a static dict (shallow merge, with `None` deleting a key) or a callable for full control. The runner is just a callable. Bring your own provider client.

## Three diff modes, no opinion about embeddings

Comparing LLM outputs is the hard part of any A/B test. The library ships three diff strategies and lets you pick per replay.

`exact` is byte equality after pulling text out of common SDK shapes. It knows about Anthropic's `content[].text` and OpenAI's `choices[].message.content`. Good for cases where you genuinely want the same string back.

`json_diff` flattens nested responses into dotted paths and reports added, removed, and changed leaves. If your provider returns structured output, this is the mode you want, because it tells you exactly which key drifted.

`semantic` does cosine similarity on a vector representation of the response text, with a caller-supplied embedder. The library does not bundle an embedder. That is a deliberate choice. It would either drag in `sentence-transformers` and a 300MB model file, or it would lock you into one provider's embedding API. Instead the embedder is a plain `Callable[[str], list[float]]`. Pass whatever you already use.

```python
from prompt_replay import Replayer
from my_app import embed   # any callable str -> list[float]

replayer = Replayer(store, mode="semantic", embedder=embed, threshold=0.9)
```

## What the report tells you

A `ReplayReport` has two pieces. `by_prompt` is the per-entry detail (original, replayed, diff). `stats` is the aggregate: total, matched, drifted, errored, avg_score.

The aggregate is what you act on. "Average score 0.92, 8 of 400 drifted, 0 errors" means the model swap is probably safe and you have 8 specific prompts to eyeball before shipping. "94 errored" means the new model rejects your tool schema and you should not ship.

`as_table()` renders both pieces as an ASCII table for terminal review. The library does not write HTML, does not start a server, does not phone home.

## Where this fits in a stack

`prompt-replay` lives next to a small family of agent-side libraries I have shipped to PyPI and crates.io under the @MukundaKatta name. The ones it shares a neighborhood with:

- `agentsnap`: snapshot tests for tool-call traces. Asserts shape. Good for "lock this in, fail loudly if it changes."
- `cachebench`: prompt cache observability. Hit ratio and miss-aware retry.
- `agenttrace`: cost and latency aggregation across calls.
- `llm-message-hash`: canonical hashing of LLM requests for cache keys.

`prompt-replay` is the one you reach for when the question is "if I change X, how do the outputs change?" not "is the shape correct?" or "how much did this cost?" Each library is a single small thing, zero dependencies, designed to compose.

## Testing and shape

The repo has 50 tests covering record, replay, all three diff modes, both stores, async runners, sync runners, error handling, override semantics, and an end-to-end record-then-replay flow. They run in 0.06 seconds because there are no real network calls anywhere.

Coverage on the public surface is high because the surface is small. Recorder. Replayer. JsonlStore. InMemoryStore. Three diff functions. A report dataclass. That is the API. If you want more, the `Sink` protocol lets you write your own store backend.

## Why I am submitting this to Hermes

Hermes is about agents that compose well. A prompt-replay loop is a compounding tool: every time you record a real production run, you build up evidence you can later use to decide a model swap or a prompt change without guessing. The library does one thing, does it without dependencies, and lives in a family of similar small libraries that the agent ecosystem can pick up piecemeal.

If you build an agent today and add `Recorder` to its provider call, in a month you have an unforced regression set. In two months you can re-score that set against any new model. That is the kind of small choice that compounds.

## Try it

```bash
pip install prompt-replay
```

Repo: https://github.com/MukundaKatta/prompt-replay
PyPI: https://pypi.org/project/prompt-replay/
License: MIT

Examples in the repo:
- `examples/record_then_replay.py`: record three prompts under one model, replay under another, print the diff table.
- `examples/ab_test_models.py`: compare two candidate runners against a baseline using the semantic diff mode.

Feedback welcome on the GitHub repo.
