"""Structured replay report types and table rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .diff import DiffResult


@dataclass
class ReplayEntry:
    """One prompt's worth of replay outcome."""

    prompt_id: str
    request: dict[str, Any]
    replayed_request: dict[str, Any]
    original: Any
    replayed: Any
    diff: DiffResult | None = None
    error: str | None = None

    @property
    def matched(self) -> bool:
        return self.diff is not None and self.diff.matched and self.error is None


@dataclass
class ReplayStats:
    """Aggregate counts across a replay run."""

    total: int = 0
    matched: int = 0
    drifted: int = 0
    errored: int = 0
    # Average score across non-error entries with a diff attached.
    avg_score: float = 0.0


@dataclass
class ReplayReport:
    """Container for per-entry results plus aggregate stats."""

    by_prompt: list[ReplayEntry] = field(default_factory=list)
    stats: ReplayStats = field(default_factory=ReplayStats)

    def add(self, entry: ReplayEntry) -> None:
        self.by_prompt.append(entry)
        self._recompute_stats()

    def _recompute_stats(self) -> None:
        total = len(self.by_prompt)
        errored = sum(1 for e in self.by_prompt if e.error is not None)
        matched = sum(1 for e in self.by_prompt if e.matched)
        drifted = total - matched - errored
        scored = [
            e.diff.score
            for e in self.by_prompt
            if e.diff is not None and e.error is None
        ]
        avg = sum(scored) / len(scored) if scored else 0.0
        self.stats = ReplayStats(
            total=total,
            matched=matched,
            drifted=drifted,
            errored=errored,
            avg_score=avg,
        )

    def as_table(self) -> str:
        """Render a small ASCII table good enough for terminal review.

        Columns: prompt_id (12), mode (10), matched (7), score (6), note (rest).
        """
        rows: list[tuple[str, str, str, str, str]] = []
        header = ("prompt_id", "mode", "matched", "score", "note")
        rows.append(header)
        for entry in self.by_prompt:
            mode = entry.diff.mode if entry.diff else "-"
            matched = "yes" if entry.matched else "no"
            score = f"{entry.diff.score:.2f}" if entry.diff else "-"
            note = entry.error or _short_note(entry)
            rows.append((entry.prompt_id, mode, matched, score, note))

        # Compute column widths from data so the table grows for long ids.
        widths = [max(len(r[i]) for r in rows) for i in range(5)]

        def fmt(row: tuple[str, str, str, str, str]) -> str:
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))

        sep = "  ".join("-" * w for w in widths)
        lines = [fmt(header), sep] + [fmt(r) for r in rows[1:]]
        stats = self.stats
        summary = (
            f"total={stats.total} matched={stats.matched} "
            f"drifted={stats.drifted} errored={stats.errored} "
            f"avg_score={stats.avg_score:.2f}"
        )
        lines.append("")
        lines.append(summary)
        return "\n".join(lines)


def _short_note(entry: ReplayEntry) -> str:
    """A one-line hint about what is interesting in this entry."""
    if entry.diff is None:
        return "no-diff"
    if entry.diff.mode == "json_diff":
        details = entry.diff.details
        changed = details.get("changed") or []
        added = details.get("added") or []
        removed = details.get("removed") or []
        bits = []
        if changed:
            bits.append(f"changed={len(changed)}")
        if added:
            bits.append(f"added={len(added)}")
        if removed:
            bits.append(f"removed={len(removed)}")
        return ",".join(bits) or "json-equal"
    if entry.diff.mode == "semantic":
        return f"thr={entry.diff.details.get('threshold')}"
    return ""


__all__ = ["ReplayEntry", "ReplayReport", "ReplayStats"]
