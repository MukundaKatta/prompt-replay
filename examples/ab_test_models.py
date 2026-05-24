"""A/B-test two model swaps against a recorded baseline.

This shows the common workflow:

  1. Record prod prompts to a JSONL store.
  2. Replay them through each candidate model.
  3. Compare the matched/drifted/avg_score stats per candidate.

The score is mode-specific. Here we use a semantic diff with a toy
character-histogram embedder so the example has no external dependency.
Swap in a real embedder (sentence-transformers, OpenAI, etc.) for prod.
"""

from __future__ import annotations

from prompt_replay import InMemoryStore, RecordedEntry, Replayer


def char_histogram(text: str) -> list[float]:
    """Toy embedder. Char counts plus a length bucket."""
    vec = [0.0] * 27
    for ch in text.lower():
        idx = ord(ch) - ord("a")
        if 0 <= idx < 26:
            vec[idx] += 1.0
    vec[26] = float(len(text)) / 100.0
    return vec


def candidate_a(req: dict) -> str:
    return f"Candidate A reply for {req['prompt']}"


def candidate_b(req: dict) -> str:
    return f"Different style response about {req['prompt']}"


def main() -> None:
    # Pretend these came from a real recording session.
    baseline = InMemoryStore(
        [
            RecordedEntry(
                request={"model": "baseline", "prompt": p},
                response=f"Candidate A reply for {p}",
            )
            for p in ("weather", "stocks", "jokes")
        ]
    )

    replayer = Replayer(baseline, mode="semantic", embedder=char_histogram, threshold=0.9)

    for name, runner in (("candidate_a", candidate_a), ("candidate_b", candidate_b)):
        report = replayer.replay(runner=runner)
        print(f"=== {name} ===")
        print(report.as_table())
        print()


if __name__ == "__main__":
    main()
