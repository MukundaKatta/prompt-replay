"""Replayer: run recorded prompts against any provider and compare outputs."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

from .diff import DiffMode, Embedder, diff as run_diff
from .report import ReplayEntry, ReplayReport
from .store import RecordedEntry, Sink

# A runner consumes the (possibly overridden) request dict and returns a
# response. Async runners are supported and resolved by `replay_async`.
SyncRunner = Callable[[dict[str, Any]], Any]
AsyncRunner = Callable[[dict[str, Any]], Awaitable[Any]]


def _apply_override(
    base: dict[str, Any],
    override: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build the replay request from the recorded request + an override.

    The override can be a static dict (shallow merge, with `None` values
    deleting keys) or a callable that returns a fresh request dict.
    """
    if override is None:
        return dict(base)
    if callable(override):
        return dict(override(dict(base)))
    out = dict(base)
    for k, v in override.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out


class Replayer:
    """Drive recorded entries back through a provider and diff the results."""

    def __init__(
        self,
        store: Sink,
        *,
        mode: DiffMode = "exact",
        embedder: Embedder | None = None,
        threshold: float = 0.85,
        skip_errored: bool = True,
    ) -> None:
        self._store = store
        self._mode = mode
        self._embedder = embedder
        self._threshold = threshold
        self._skip_errored = skip_errored

    @property
    def store(self) -> Sink:
        return self._store

    def replay(
        self,
        *,
        runner: SyncRunner,
        override: dict[str, Any]
        | Callable[[dict[str, Any]], dict[str, Any]]
        | None = None,
        mode: DiffMode | None = None,
        embedder: Embedder | None = None,
        threshold: float | None = None,
    ) -> ReplayReport:
        """Replay every recorded entry through `runner` and build a report."""
        report = ReplayReport()
        mode_to_use = mode or self._mode
        embedder_to_use = embedder if embedder is not None else self._embedder
        threshold_to_use = (
            threshold if threshold is not None else self._threshold
        )
        for original_entry in self._store.read_all():
            if self._skip_errored and original_entry.error is not None:
                continue
            entry = self._replay_one(
                original_entry,
                runner=runner,
                override=override,
                mode=mode_to_use,
                embedder=embedder_to_use,
                threshold=threshold_to_use,
            )
            report.add(entry)
        return report

    async def replay_async(
        self,
        *,
        runner: AsyncRunner | SyncRunner,
        override: dict[str, Any]
        | Callable[[dict[str, Any]], dict[str, Any]]
        | None = None,
        mode: DiffMode | None = None,
        embedder: Embedder | None = None,
        threshold: float | None = None,
        concurrency: int = 1,
    ) -> ReplayReport:
        """Async variant. `concurrency` caps in-flight runner calls."""
        report = ReplayReport()
        mode_to_use = mode or self._mode
        embedder_to_use = embedder if embedder is not None else self._embedder
        threshold_to_use = (
            threshold if threshold is not None else self._threshold
        )
        sem = asyncio.Semaphore(max(concurrency, 1))

        async def run_one(original_entry: RecordedEntry) -> ReplayEntry:
            replay_request = _apply_override(original_entry.request, override)
            async with sem:
                try:
                    if inspect.iscoroutinefunction(runner):
                        result = await runner(replay_request)
                    else:
                        # Allow callers to pass a sync runner too.
                        result = await asyncio.to_thread(
                            runner, replay_request  # type: ignore[arg-type]
                        )
                except Exception as exc:
                    return ReplayEntry(
                        prompt_id=original_entry.id,
                        request=dict(original_entry.request),
                        replayed_request=replay_request,
                        original=original_entry.response,
                        replayed=None,
                        diff=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            diff_result = run_diff(
                original_entry.response,
                result,
                mode=mode_to_use,
                embedder=embedder_to_use,
                threshold=threshold_to_use,
            )
            return ReplayEntry(
                prompt_id=original_entry.id,
                request=dict(original_entry.request),
                replayed_request=replay_request,
                original=original_entry.response,
                replayed=result,
                diff=diff_result,
            )

        candidates = [
            e
            for e in self._store.read_all()
            if not (self._skip_errored and e.error is not None)
        ]
        results = await asyncio.gather(*(run_one(e) for e in candidates))
        for r in results:
            report.add(r)
        return report

    def _replay_one(
        self,
        original_entry: RecordedEntry,
        *,
        runner: SyncRunner,
        override: dict[str, Any]
        | Callable[[dict[str, Any]], dict[str, Any]]
        | None,
        mode: DiffMode,
        embedder: Embedder | None,
        threshold: float,
    ) -> ReplayEntry:
        replay_request = _apply_override(original_entry.request, override)
        try:
            result = runner(replay_request)
        except Exception as exc:
            return ReplayEntry(
                prompt_id=original_entry.id,
                request=dict(original_entry.request),
                replayed_request=replay_request,
                original=original_entry.response,
                replayed=None,
                diff=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        diff_result = run_diff(
            original_entry.response,
            result,
            mode=mode,
            embedder=embedder,
            threshold=threshold,
        )
        return ReplayEntry(
            prompt_id=original_entry.id,
            request=dict(original_entry.request),
            replayed_request=replay_request,
            original=original_entry.response,
            replayed=result,
            diff=diff_result,
        )


class ReplayRunner:
    """Thin re-export for callers who prefer importing a Runner type symbol.

    Kept around so example code like `from prompt_replay import ReplayRunner`
    matches the README quickstart even if the runner is just a callable.
    """

    def __init__(self, fn: SyncRunner) -> None:
        self._fn = fn

    def __call__(self, request: dict[str, Any]) -> Any:
        return self._fn(request)


__all__ = ["AsyncRunner", "Replayer", "ReplayRunner", "SyncRunner"]
