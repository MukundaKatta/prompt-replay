"""Recorder: capture LLM provider calls during a normal run."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import is_dataclass
from typing import Any, Awaitable, Callable, TypeVar

from .store import RecordedEntry, Sink

F = TypeVar("F", bound=Callable[..., Any])


def _coerce_to_jsonable(value: Any) -> Any:
    """Best-effort coercion of a provider response into a JSON-friendly shape.

    The default flow handles dicts, lists, primitives, dataclasses, and any
    object with `model_dump()` (Pydantic) or `to_dict()` (SDK message types).
    Anything else falls through to its `str()` form so a recorded entry is
    never lost just because the response is an exotic SDK object.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_to_jsonable(v) for v in value]
    # Pydantic v2 first, then dataclasses, then anything with .to_dict().
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _coerce_to_jsonable(model_dump())
        except Exception:
            pass
    if is_dataclass(value):
        from dataclasses import asdict

        try:
            return _coerce_to_jsonable(asdict(value))
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _coerce_to_jsonable(to_dict())
        except Exception:
            pass
    return str(value)


class Recorder:
    """Wrap a provider call so every invocation is recorded to a `Sink`.

    Usage:

        recorder = Recorder(store=JsonlStore("prompts.jsonl"))

        @recorder.capture
        def call(model, messages, **kw):
            return provider.messages.create(model=model, messages=messages, **kw)

    The captured `request` dict contains every kwarg the wrapped function
    received, plus positional arguments bound to their parameter names via
    `inspect.signature`. The captured `response` is whatever the function
    returned, coerced into a JSON-friendly form.
    """

    def __init__(
        self,
        store: Sink,
        *,
        capture_errors: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._store = store
        self._capture_errors = capture_errors
        # A shallow copy so the caller can mutate their dict later without
        # affecting future entries.
        self._metadata = dict(metadata or {})

    @property
    def store(self) -> Sink:
        return self._store

    def capture(self, fn: F) -> F:
        """Decorator: record every call to `fn` as a `RecordedEntry`."""
        sig = inspect.signature(fn)

        if inspect.iscoroutinefunction(fn):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                request = _bind_request(sig, args, kwargs)
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    if self._capture_errors:
                        self._store.write(
                            RecordedEntry(
                                request=request,
                                response=None,
                                error=f"{type(exc).__name__}: {exc}",
                                metadata=dict(self._metadata),
                            )
                        )
                    raise
                self._store.write(
                    RecordedEntry(
                        request=request,
                        response=_coerce_to_jsonable(result),
                        metadata=dict(self._metadata),
                    )
                )
                return result

            return async_wrapper  # type: ignore[return-value]

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _bind_request(sig, args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                if self._capture_errors:
                    self._store.write(
                        RecordedEntry(
                            request=request,
                            response=None,
                            error=f"{type(exc).__name__}: {exc}",
                            metadata=dict(self._metadata),
                        )
                    )
                raise
            self._store.write(
                RecordedEntry(
                    request=request,
                    response=_coerce_to_jsonable(result),
                    metadata=dict(self._metadata),
                )
            )
            return result

        return sync_wrapper  # type: ignore[return-value]

    def record(
        self,
        *,
        request: dict[str, Any],
        response: Any,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecordedEntry:
        """Manual escape hatch: record an entry without a decorator."""
        merged_meta = dict(self._metadata)
        if metadata:
            merged_meta.update(metadata)
        entry = RecordedEntry(
            request=dict(request),
            response=_coerce_to_jsonable(response),
            error=error,
            metadata=merged_meta,
        )
        self._store.write(entry)
        return entry


def _bind_request(
    sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Bind positional + keyword args to a flat dict keyed by parameter name.

    Falls back to `args`/`kwargs` keys if the signature does not match (for
    example if the wrapped function uses bare `*args`/`**kwargs`).
    """
    try:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {k: _coerce_to_jsonable(v) for k, v in bound.arguments.items()}
    except TypeError:
        return {
            "args": _coerce_to_jsonable(list(args)),
            "kwargs": _coerce_to_jsonable(dict(kwargs)),
        }


__all__ = ["Recorder"]
