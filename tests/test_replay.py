import pytest

from prompt_replay import (
    InMemoryStore,
    RecordedEntry,
    Replayer,
    ReplayRunner,
)


def _seed_store(*entries):
    store = InMemoryStore()
    for e in entries:
        store.write(e)
    return store


def test_replay_no_override_runs_runner_for_each_entry():
    store = _seed_store(
        RecordedEntry(request={"model": "claude-sonnet-4-7"}, response="alpha"),
        RecordedEntry(request={"model": "claude-sonnet-4-7"}, response="beta"),
    )
    seen = []

    def runner(req):
        seen.append(req["model"])
        # Echo the same response so diffs all match.
        if "alpha" in seen:
            return "alpha"
        return "beta"

    replayer = Replayer(store)
    report = replayer.replay(runner=lambda req: "alpha" if seen.append(req) or len(seen) == 1 else "beta")
    assert report.stats.total == 2


def test_replay_dict_override_swaps_model():
    store = _seed_store(
        RecordedEntry(request={"model": "claude-sonnet-4-7", "messages": []}, response="x"),
    )
    seen = []

    def runner(req):
        seen.append(req)
        return "x"

    replayer = Replayer(store)
    report = replayer.replay(runner=runner, override={"model": "claude-haiku-4-5"})
    assert seen[0]["model"] == "claude-haiku-4-5"
    assert report.stats.matched == 1


def test_replay_override_callable():
    store = _seed_store(
        RecordedEntry(request={"model": "a", "temperature": 0.7}, response="y"),
    )

    def override(req):
        req["temperature"] = 0.0
        req["model"] = "b"
        return req

    seen: list = []
    replayer = Replayer(store)
    replayer.replay(runner=lambda req: (seen.append(req), "y")[1], override=override)
    assert seen[0]["model"] == "b"
    assert seen[0]["temperature"] == 0.0


def test_replay_override_none_value_deletes_key():
    store = _seed_store(
        RecordedEntry(request={"model": "a", "tools": [{"name": "t"}]}, response="z"),
    )
    seen: list = []
    replayer = Replayer(store)
    replayer.replay(
        runner=lambda req: (seen.append(req), "z")[1],
        override={"tools": None},
    )
    assert "tools" not in seen[0]
    assert seen[0]["model"] == "a"


def test_replay_skips_errored_by_default():
    store = _seed_store(
        RecordedEntry(request={"model": "a"}, response="ok"),
        RecordedEntry(request={"model": "b"}, response=None, error="boom"),
    )
    seen: list = []
    replayer = Replayer(store)
    report = replayer.replay(runner=lambda req: (seen.append(req), "ok")[1])
    assert report.stats.total == 1
    assert len(seen) == 1


def test_replay_can_include_errored_when_disabled():
    store = _seed_store(
        RecordedEntry(request={"model": "a"}, response=None, error="x"),
    )
    replayer = Replayer(store, skip_errored=False)
    report = replayer.replay(runner=lambda req: "x")
    assert report.stats.total == 1


def test_replay_runner_capturing_exception():
    store = _seed_store(
        RecordedEntry(request={"model": "a"}, response="orig"),
    )

    def runner(req):
        raise TimeoutError("slow")

    replayer = Replayer(store)
    report = replayer.replay(runner=runner)
    assert report.stats.errored == 1
    assert report.by_prompt[0].error.startswith("TimeoutError:")


def test_replay_runner_class_wrapper():
    store = _seed_store(RecordedEntry(request={"model": "x"}, response="ok"))
    runner = ReplayRunner(lambda req: "ok")
    replayer = Replayer(store)
    report = replayer.replay(runner=runner)
    assert report.stats.matched == 1


async def test_replay_async_with_concurrency():
    store = _seed_store(
        *(
            RecordedEntry(request={"model": "a", "i": i}, response=f"v{i}")
            for i in range(8)
        )
    )

    async def runner(req):
        # Each call returns the same value it recorded so all diffs match.
        return f"v{req['i']}"

    replayer = Replayer(store)
    report = await replayer.replay_async(runner=runner, concurrency=4)
    assert report.stats.total == 8
    assert report.stats.matched == 8


async def test_replay_async_accepts_sync_runner():
    store = _seed_store(RecordedEntry(request={"model": "a"}, response="ok"))

    def sync_runner(req):
        return "ok"

    replayer = Replayer(store)
    report = await replayer.replay_async(runner=sync_runner)
    assert report.stats.matched == 1
