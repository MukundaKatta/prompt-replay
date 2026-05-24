# prompt-replay

[![PyPI](https://img.shields.io/badge/pypi-prompt--replay-blue)](https://pypi.org/project/prompt-replay/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

Record LLM prompts during one run, replay them later against any provider or model to compare outputs. Useful for A/B testing prompt changes, regression testing after a model upgrade, and evaluating a new model against a frozen prompt set.

Built for the moment when you upgrade `claude-sonnet-4-7` to `claude-haiku-4-5` and want to know which 12 of your 400 production prompts now answer differently, without spinning up a notebook from scratch.

## Install

```bash
pip install prompt-replay
```

Zero runtime dependencies. Python 3.10+.

## 60-second quickstart

```python
from prompt_replay import Recorder, Replayer, JsonlStore

# Phase 1: record during a normal app run.
recorder = Recorder(store=JsonlStore("prompts.jsonl"))

@recorder.capture
def call(model, messages, **kw):
    return real_provider.messages.create(model=model, messages=messages, **kw)

call("claude-sonnet-4-7", [{"role": "user", "content": "weather in NYC"}])
call("claude-sonnet-4-7", [{"role": "user", "content": "joke about cats"}])

# Phase 2: later, replay every prompt against a different model.
replayer = Replayer(JsonlStore("prompts.jsonl"))

report = replayer.replay(
    override={"model": "claude-haiku-4-5"},
    runner=lambda req: real_provider.messages.create(**req),
)

print(report.as_table())
# prompt_id     mode   matched  score  note
# ------------  -----  -------  -----  ----
# 8a1c...       exact  yes      1.00
# 2f04...       exact  no       0.00
#
# total=2 matched=1 drifted=1 errored=0 avg_score=0.50
```

That is the whole loop. Same store powers both phases. No state to manage.

## What it does well

- **Full request capture.** Model, messages, tools, max_tokens, temperature, every kwarg the wrapped function received. Replays land on the same shape.
- **Single-field swaps.** Override `{"model": "..."}` or `{"temperature": 0.0}` and leave everything else identical. Useful for clean A/Bs.
- **Three diff modes.** `exact` for byte equality, `json_diff` for deep-key diffs when both sides parse as JSON, `semantic` for cosine similarity with a caller-supplied embedder.
- **Sync and async.** `@recorder.capture` and `Replayer.replay_async` work on both. `replay_async` has a `concurrency` cap.
- **Pluggable stores.** `JsonlStore` for append-only files, `InMemoryStore` for tests, or any object with `write(entry)` and `read_all()`.
- **Errors are first-class.** Captured during record (with the original exception type and message) and during replay (as a report-level `errored` count).

## When to use this vs sibling libs

| Library | Job |
|---|---|
| `prompt-replay` | **Record prompts now, replay them later** against a different model or provider for A/B + regression testing. |
| [`agentsnap`](https://pypi.org/project/agentsnap/) | Snapshot tool-call traces for unit tests. Asserts shape, not cross-model behavior. |
| [`cachebench`](https://pypi.org/project/cachebench/) | Observability for prompt caches: hit ratio, miss-aware retry. Not about replaying anything. |
| [`agenttrace`](https://github.com/MukundaKatta/agenttrace) | Cost and latency aggregation across LLM calls. Different axis. |
| [`llm-message-hash`](https://crates.io/crates/llm-message-hash) | Canonical hashing of LLM requests for cache keys. Single-call concern. |

If you want "did this prompt set get better or worse when I changed the model", that is this library. If you want "lock the shape of one tool call so a regression breaks the test", reach for `agentsnap`.

## When this is not what you want

- **You need full deterministic snapshot tests.** Use `agentsnap`. It is built for the snapshot workflow (record, commit, fail loudly on diff). `prompt-replay` is built for "run, score, decide".
- **You only care about latency or cost regressions.** `agenttrace` is the right shape. This library does not track timing.
- **You want a managed eval platform UI.** Use something like Braintrust, Langfuse, or Promptfoo. This is a small library, not a hosted product.

## API surface

```python
from prompt_replay import (
    Recorder,           # capture decorator + manual record()
    Replayer,           # replay over a Sink, returns a ReplayReport
    ReplayRunner,       # optional wrapper around the (request) -> response callable
    JsonlStore,         # append-only file Sink
    InMemoryStore,      # RAM Sink, ideal for tests
    RecordedEntry,      # one (request, response) pair
    Sink,               # Protocol: write(entry) + read_all()
    ReplayReport,       # by_prompt + stats + as_table()
    ReplayEntry,        # one replay outcome (original, replayed, diff)
    ReplayStats,        # total/matched/drifted/errored/avg_score
    diff,               # dispatcher: exact / semantic / json_diff
    exact_diff,
    semantic_diff,
    json_diff,
    DiffResult,
    DiffMode,
    Embedder,           # callable type alias for str -> list[float]
)
```

### `Recorder(store, *, capture_errors=True, metadata=None)`

Wrap any callable that talks to a provider. The decorator binds positional args to their parameter names via `inspect.signature`, so `call("claude-sonnet-4-7", messages)` records `request["model"]` correctly.

```python
recorder = Recorder(store=JsonlStore("prompts.jsonl"), metadata={"env": "prod"})

@recorder.capture
def call(model, messages, **kw): ...
```

When the wrapped call raises, the recorder writes an entry with `error="<Exception>: <message>"` then re-raises. Set `capture_errors=False` to skip writing errored entries.

For non-decorator flows, `recorder.record(request=..., response=...)` writes a single entry by hand.

### `Replayer(store, *, mode="exact", embedder=None, threshold=0.85, skip_errored=True)`

Walks the store, builds a (possibly-overridden) request for each entry, hands it to your `runner`, and diffs the result.

```python
replayer = Replayer(store, mode="json_diff")
report = replayer.replay(runner=lambda req: provider.call(**req))
```

For async runners:

```python
report = await replayer.replay_async(runner=async_runner, concurrency=8)
```

Overrides can be a dict (`{"model": "..."}` replaces, `{"tools": None}` deletes) or a callable `(request) -> request` for full control.

### Diff modes

```python
exact_diff(original, replayed)
json_diff(original, replayed)
semantic_diff(original, replayed, embedder=my_embedder, threshold=0.9)
```

`exact_diff` first extracts text from common SDK shapes (Anthropic `content[].text`, OpenAI `choices[].message.content`), then compares strings. `json_diff` flattens nested dicts/lists into dotted paths and reports added / removed / changed leaves. `semantic_diff` short-circuits on byte equality, then falls back to cosine similarity in your embedder's vector space.

### Reports

```python
report.by_prompt           # list[ReplayEntry]
report.stats.total         # int
report.stats.matched       # int
report.stats.drifted       # int
report.stats.errored       # int
report.stats.avg_score     # float across non-errored entries
print(report.as_table())   # ASCII table for terminal review
```

## Async example

```python
import asyncio
from prompt_replay import Recorder, Replayer, JsonlStore

recorder = Recorder(store=JsonlStore("prompts.jsonl"))

@recorder.capture
async def call(model, messages, **kw):
    return await provider.messages.create(model=model, messages=messages, **kw)

async def main():
    await call("claude-sonnet-4-7", [{"role": "user", "content": "hi"}])
    replayer = Replayer(JsonlStore("prompts.jsonl"))
    report = await replayer.replay_async(
        runner=lambda req: provider.messages.create(**req),
        override={"model": "claude-haiku-4-5"},
        concurrency=4,
    )
    print(report.as_table())

asyncio.run(main())
```

## Sibling libs

Part of the @MukundaKatta agent-stack family on PyPI and crates.io. Other Python libs in the family: agentsnap, agentguard, agentvet, agentcast, agentfit, agenttrace, cachebench, llmfleet, agenttap, agentleash, birddog, recruitertriage, driftvane, bedrock-kit, token-budget-py.

## License

MIT. See [LICENSE](LICENSE).
