"""Record a small batch of prompts, then replay them with a different model.

Run from the repo root after `pip install -e .`:

    python examples/record_then_replay.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from prompt_replay import JsonlStore, Recorder, Replayer


def fake_provider_call(model: str, messages: list, **_: object) -> dict:
    """Stand-in for a real provider. Each model returns a slightly different text."""
    user = messages[0]["content"]
    if model.startswith("claude-sonnet"):
        return {"content": [{"type": "text", "text": f"Sonnet says: {user}"}]}
    return {"content": [{"type": "text", "text": f"Haiku says: {user}"}]}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "prompts.jsonl"

        # Phase 1: record during the normal run.
        recorder = Recorder(store=JsonlStore(store_path))

        @recorder.capture
        def call(model, messages, **kw):
            return fake_provider_call(model, messages, **kw)

        for prompt in ("What is the weather?", "Tell me a joke.", "Summarize Hamlet."):
            call("claude-sonnet-4-7", [{"role": "user", "content": prompt}])

        # Phase 2: replay against haiku and see what changed.
        store = JsonlStore(store_path)
        replayer = Replayer(store)

        report = replayer.replay(
            override={"model": "claude-haiku-4-5"},
            runner=lambda req: fake_provider_call(**req),
        )

        print(report.as_table())


if __name__ == "__main__":
    main()
