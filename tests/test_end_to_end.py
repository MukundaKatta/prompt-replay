"""End-to-end: record a small batch, swap the model, replay, check stats."""

from prompt_replay import (
    JsonlStore,
    Recorder,
    Replayer,
)


# A toy provider with two "models". sonnet returns full sentences, haiku returns
# truncated ones. We use this contrast to drive matched/drifted counts.
def _fake_provider(model, messages, **kw):
    user_text = messages[0]["content"]
    if model == "claude-sonnet-4-7":
        return {"content": [{"type": "text", "text": f"Long answer for: {user_text}"}]}
    if model == "claude-haiku-4-5":
        return {"content": [{"type": "text", "text": f"Short: {user_text}"}]}
    return {"content": [{"type": "text", "text": "unknown model"}]}


def test_record_then_replay_with_model_swap(tmp_path):
    path = tmp_path / "prompts.jsonl"
    recorder = Recorder(store=JsonlStore(path))

    @recorder.capture
    def call(model, messages, **kw):
        return _fake_provider(model, messages, **kw)

    prompts = ["weather in NYC", "stock price of NVDA", "joke about cats"]
    for p in prompts:
        call("claude-sonnet-4-7", [{"role": "user", "content": p}])

    # Replay with the same model: every entry should match exactly.
    store_for_replay = JsonlStore(path)
    replayer = Replayer(store_for_replay)
    same_report = replayer.replay(
        runner=lambda req: _fake_provider(**req),
    )
    assert same_report.stats.total == 3
    assert same_report.stats.matched == 3
    assert same_report.stats.drifted == 0

    # Replay with haiku: every entry should drift because the text differs.
    swap_report = replayer.replay(
        runner=lambda req: _fake_provider(**req),
        override={"model": "claude-haiku-4-5"},
    )
    assert swap_report.stats.total == 3
    assert swap_report.stats.matched == 0
    assert swap_report.stats.drifted == 3
    # Each replayed request was actually rewritten.
    assert all(
        e.replayed_request["model"] == "claude-haiku-4-5" for e in swap_report.by_prompt
    )


def test_replay_with_json_diff_mode(tmp_path):
    path = tmp_path / "prompts.jsonl"
    recorder = Recorder(store=JsonlStore(path))

    @recorder.capture
    def call(model, payload):
        return {"summary": payload["topic"], "score": 0.9}

    call("gpt-5.4", {"topic": "alpha"})
    call("gpt-5.4", {"topic": "beta"})

    store = JsonlStore(path)
    replayer = Replayer(store, mode="json_diff")
    # Replay returns a slightly different score, summary stays the same.
    report = replayer.replay(
        runner=lambda req: {"summary": req["payload"]["topic"], "score": 0.7},
    )
    assert report.stats.total == 2
    assert report.stats.drifted == 2
    for entry in report.by_prompt:
        assert "score" in entry.diff.details["changed"]


def test_full_table_rendering_does_not_crash(tmp_path):
    path = tmp_path / "prompts.jsonl"
    recorder = Recorder(store=JsonlStore(path))

    @recorder.capture
    def call(model, messages):
        return {"content": [{"type": "text", "text": "ok"}]}

    call("claude-sonnet-4-7", [{"role": "user", "content": "hi"}])
    replayer = Replayer(JsonlStore(path))
    report = replayer.replay(runner=lambda req: {"content": [{"type": "text", "text": "ok"}]})
    text = report.as_table()
    assert "prompt_id" in text
    assert "matched=1" in text
