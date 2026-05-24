"""Stores for recorded prompts.

A `Sink` is any object that can `write(entry)` and iterate via `read_all()`.
Bundled implementations: `InMemoryStore` (RAM, default for tests) and
`JsonlStore` (one JSON object per line, append-only file).
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


def _new_id() -> str:
    """Short unique id for a recorded prompt entry."""
    return uuid.uuid4().hex[:12]


@dataclass
class RecordedEntry:
    """One captured request + response pair.

    `request` is the full kwargs dict that was sent to the provider call,
    plus a "model" field when one was provided positionally or via kwargs.
    `response` is whatever the wrapped function returned, coerced to a
    JSON-serializable shape (dict, list, str, number, bool, None).
    """

    id: str = field(default_factory=_new_id)
    request: dict[str, Any] = field(default_factory=dict)
    response: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordedEntry":
        return cls(
            id=data.get("id") or _new_id(),
            request=dict(data.get("request") or {}),
            response=data.get("response"),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )


class Sink(Protocol):
    """Anything you can write entries to and iterate over later."""

    def write(self, entry: RecordedEntry) -> None: ...
    def read_all(self) -> Iterable[RecordedEntry]: ...


class InMemoryStore:
    """RAM-backed store. Useful for tests and single-process runs."""

    def __init__(self, entries: Iterable[RecordedEntry] | None = None) -> None:
        self._entries: list[RecordedEntry] = list(entries or [])
        self._lock = threading.Lock()

    def write(self, entry: RecordedEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def read_all(self) -> Iterator[RecordedEntry]:
        # Snapshot so concurrent writes do not surprise the iterator.
        with self._lock:
            snapshot = list(self._entries)
        return iter(snapshot)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class JsonlStore:
    """Append-only JSON Lines file store.

    Each line is one `RecordedEntry.to_dict()` payload. Writes are
    line-flushed so a crash in the middle of a run still leaves all
    completed entries on disk.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Touch the file so `read_all` on a fresh store does not raise.
        if not self._path.exists():
            self._path.touch()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, entry: RecordedEntry) -> None:
        line = json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    def read_all(self) -> Iterator[RecordedEntry]:
        if not self._path.exists():
            return iter(())
        with self._path.open("r", encoding="utf-8") as fh:
            lines = [line for line in fh.readlines() if line.strip()]
        return (RecordedEntry.from_dict(json.loads(line)) for line in lines)
