import json
from pathlib import Path

from prompt_replay import JsonlStore, RecordedEntry


def test_jsonl_creates_parent_dir(tmp_path):
    nested = tmp_path / "nested" / "deep" / "prompts.jsonl"
    JsonlStore(nested)
    assert nested.parent.exists()
    assert nested.exists()


def test_jsonl_write_and_read(tmp_path):
    path = tmp_path / "prompts.jsonl"
    store = JsonlStore(path)
    store.write(RecordedEntry(request={"model": "claude-sonnet"}, response="ok"))
    store.write(
        RecordedEntry(request={"model": "gpt-5"}, response={"choices": []})
    )
    entries = list(store.read_all())
    assert len(entries) == 2
    assert entries[0].request == {"model": "claude-sonnet"}
    assert entries[1].response == {"choices": []}


def test_jsonl_one_entry_per_line(tmp_path):
    path = tmp_path / "p.jsonl"
    store = JsonlStore(path)
    for i in range(5):
        store.write(RecordedEntry(request={"i": i}, response=str(i)))
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    # Each line is valid JSON on its own.
    for line in lines:
        payload = json.loads(line)
        assert "id" in payload and "request" in payload


def test_jsonl_empty_file_iterates_to_nothing(tmp_path):
    path = tmp_path / "empty.jsonl"
    store = JsonlStore(path)
    assert list(store.read_all()) == []


def test_jsonl_append_only_across_instances(tmp_path):
    path = tmp_path / "stream.jsonl"
    s1 = JsonlStore(path)
    s1.write(RecordedEntry(request={"i": 1}, response="a"))
    s2 = JsonlStore(path)
    s2.write(RecordedEntry(request={"i": 2}, response="b"))
    # Reading from a third instance shows both entries.
    s3 = JsonlStore(path)
    out = list(s3.read_all())
    assert len(out) == 2
    assert [e.request["i"] for e in out] == [1, 2]
