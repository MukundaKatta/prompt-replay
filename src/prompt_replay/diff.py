"""Diff strategies for comparing original and replayed responses.

Each strategy returns a `DiffResult` describing whether the two outputs
matched and (optionally) a structured summary of what differed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

DiffMode = Literal["exact", "semantic", "json_diff"]


@dataclass
class DiffResult:
    """Outcome of comparing original vs replayed response."""

    mode: DiffMode
    matched: bool
    # Score is mode-specific. Exact: 1.0 if match else 0.0.
    # Semantic: cosine similarity in [-1.0, 1.0]. JSON: ratio of unchanged keys.
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def _extract_text(value: Any) -> str:
    """Best-effort string extraction from a response.

    Plain string returns as-is. Otherwise walks common shapes used by
    Anthropic, OpenAI, and similar SDKs: `{"content": [{"text": "..."}]}` and
    `{"choices": [{"message": {"content": "..."}}]}`.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Anthropic-style: {"content": [{"type": "text", "text": "..."}]}
        content = value.get("content")
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        chunks.append(txt)
                elif isinstance(item, str):
                    chunks.append(item)
            if chunks:
                return "\n".join(chunks)
        if isinstance(content, str):
            return content
        # OpenAI-style: {"choices": [{"message": {"content": "..."}}]}
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    msg_content = msg.get("content")
                    if isinstance(msg_content, str):
                        return msg_content
                text = first.get("text")
                if isinstance(text, str):
                    return text
        # Fallback: a "text" field at the root.
        if isinstance(value.get("text"), str):
            return value["text"]
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def exact_diff(original: Any, replayed: Any) -> DiffResult:
    """String-equality diff after normalizing each side to text."""
    a = _extract_text(original)
    b = _extract_text(replayed)
    matched = a == b
    return DiffResult(
        mode="exact",
        matched=matched,
        score=1.0 if matched else 0.0,
        details={"original_len": len(a), "replayed_len": len(b)},
    )


class Embedder(Protocol):
    """Anything that turns a string into a vector of floats."""

    def __call__(self, text: str) -> list[float]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def semantic_diff(
    original: Any,
    replayed: Any,
    *,
    embedder: Embedder,
    threshold: float = 0.85,
) -> DiffResult:
    """Cosine-similarity diff using a caller-provided embedder."""
    a = _extract_text(original)
    b = _extract_text(replayed)
    if a == b:
        return DiffResult(
            mode="semantic",
            matched=True,
            score=1.0,
            details={"reason": "byte-equal", "threshold": threshold},
        )
    vec_a = embedder(a)
    vec_b = embedder(b)
    score = _cosine(vec_a, vec_b)
    return DiffResult(
        mode="semantic",
        matched=score >= threshold,
        score=score,
        details={"threshold": threshold},
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict/list into dotted-path leaves."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            out[prefix or "."] = {}
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, path))
    elif isinstance(value, list):
        if not value:
            out[prefix or "."] = []
        for i, item in enumerate(value):
            path = f"{prefix}[{i}]"
            out.update(_flatten(item, path))
    else:
        out[prefix or "."] = value
    return out


def _try_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


def json_diff(original: Any, replayed: Any) -> DiffResult:
    """Deep-key diff when both responses parse as JSON.

    The matcher reports added, removed, and changed leaves. The score is the
    fraction of leaves that are present and equal on both sides.
    """
    a = _try_parse_json(original)
    b = _try_parse_json(replayed)
    if a is None and isinstance(original, (dict, list)):
        a = original
    if b is None and isinstance(replayed, (dict, list)):
        b = replayed
    if not isinstance(a, (dict, list)) or not isinstance(b, (dict, list)):
        return DiffResult(
            mode="json_diff",
            matched=False,
            score=0.0,
            details={"reason": "not_json_on_one_or_both_sides"},
        )
    flat_a = _flatten(a)
    flat_b = _flatten(b)
    keys_a = set(flat_a.keys())
    keys_b = set(flat_b.keys())
    added = sorted(keys_b - keys_a)
    removed = sorted(keys_a - keys_b)
    changed = sorted(k for k in keys_a & keys_b if flat_a[k] != flat_b[k])
    total = len(keys_a | keys_b) or 1
    unchanged = total - len(added) - len(removed) - len(changed)
    score = unchanged / total
    return DiffResult(
        mode="json_diff",
        matched=not added and not removed and not changed,
        score=score,
        details={
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
            "total_keys": total,
        },
    )


def diff(
    original: Any,
    replayed: Any,
    *,
    mode: DiffMode = "exact",
    embedder: Embedder | None = None,
    threshold: float = 0.85,
) -> DiffResult:
    """Convenience entry point that dispatches on `mode`."""
    if mode == "exact":
        return exact_diff(original, replayed)
    if mode == "semantic":
        if embedder is None:
            raise ValueError("semantic diff requires an embedder=callable")
        return semantic_diff(
            original, replayed, embedder=embedder, threshold=threshold
        )
    if mode == "json_diff":
        return json_diff(original, replayed)
    raise ValueError(f"unknown diff mode: {mode!r}")


__all__ = [
    "DiffMode",
    "DiffResult",
    "Embedder",
    "diff",
    "exact_diff",
    "json_diff",
    "semantic_diff",
]
