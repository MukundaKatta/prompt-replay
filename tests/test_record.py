import pytest

from prompt_replay import InMemoryStore, Recorder


def test_capture_records_sync_call():
    store = InMemoryStore()
    recorder = Recorder(store=store)

    @recorder.capture
    def call(model, messages, max_tokens=256):
        return {"content": [{"type": "text", "text": "hello"}]}

    result = call("claude-sonnet-4-7", [{"role": "user", "content": "hi"}])
    assert result["content"][0]["text"] == "hello"

    entries = list(store.read_all())
    assert len(entries) == 1
    assert entries[0].request["model"] == "claude-sonnet-4-7"
    assert entries[0].request["max_tokens"] == 256
    assert entries[0].response["content"][0]["text"] == "hello"


def test_capture_records_error_then_reraises():
    store = InMemoryStore()
    recorder = Recorder(store=store)

    @recorder.capture
    def call(prompt):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        call("anything")

    entries = list(store.read_all())
    assert len(entries) == 1
    assert entries[0].error == "RuntimeError: boom"
    assert entries[0].response is None


def test_capture_can_skip_errors():
    store = InMemoryStore()
    recorder = Recorder(store=store, capture_errors=False)

    @recorder.capture
    def call(prompt):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        call("x")

    assert len(store) == 0


def test_record_manual_entry():
    store = InMemoryStore()
    recorder = Recorder(store=store, metadata={"env": "test"})
    entry = recorder.record(
        request={"model": "haiku", "messages": []},
        response="ok",
    )
    assert entry.metadata == {"env": "test"}
    assert list(store.read_all())[0].response == "ok"


async def test_capture_records_async_call():
    store = InMemoryStore()
    recorder = Recorder(store=store)

    @recorder.capture
    async def call(model, messages):
        return {"text": "async-ok"}

    out = await call("claude-haiku-4-5", [{"role": "user", "content": "hi"}])
    assert out == {"text": "async-ok"}

    entries = list(store.read_all())
    assert len(entries) == 1
    assert entries[0].request["model"] == "claude-haiku-4-5"
    assert entries[0].response == {"text": "async-ok"}


def test_capture_handles_varargs_kwargs():
    store = InMemoryStore()
    recorder = Recorder(store=store)

    @recorder.capture
    def call(*args, **kwargs):
        return {"ok": True}

    call("a", "b", model="x", temperature=0.2)
    entry = list(store.read_all())[0]
    # Falls through to the args/kwargs shape because binding does not match
    # a normal signature.
    assert entry.request["args"] == ["a", "b"]
    assert entry.request["kwargs"]["model"] == "x"
    assert entry.request["kwargs"]["temperature"] == 0.2


def test_metadata_is_copied_per_entry():
    store = InMemoryStore()
    meta: dict = {"trial": 1}
    recorder = Recorder(store=store, metadata=meta)

    @recorder.capture
    def call(p):
        return "ok"

    call("a")
    meta["trial"] = 999
    call("b")
    entries = list(store.read_all())
    assert entries[0].metadata == {"trial": 1}
    # Second entry uses whatever metadata snapshot the recorder copied at init.
    # The recorder copies once at init, so trial stays at 1 on the recorder side.
    assert entries[1].metadata == {"trial": 1}
