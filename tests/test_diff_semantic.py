import pytest

from prompt_replay import semantic_diff, diff


def fake_embedder(text: str) -> list[float]:
    # Tiny deterministic embedder: bag of characters keyed by 26 lowercase letters
    # plus a length bucket. Good enough for unit-test cosine sims.
    vec = [0.0] * 27
    for ch in text.lower():
        idx = ord(ch) - ord("a")
        if 0 <= idx < 26:
            vec[idx] += 1.0
    vec[26] = float(len(text)) / 100.0
    return vec


def test_semantic_byte_equal_short_circuits():
    r = semantic_diff("hello", "hello", embedder=fake_embedder, threshold=0.9)
    assert r.matched is True
    assert r.score == 1.0
    assert r.details["reason"] == "byte-equal"


def test_semantic_similar_passes_threshold():
    r = semantic_diff(
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumped over the lazy dogs",
        embedder=fake_embedder,
        threshold=0.85,
    )
    assert r.matched is True
    assert r.score >= 0.85


def test_semantic_dissimilar_fails_threshold():
    r = semantic_diff(
        "hello world",
        "completely unrelated string about apples",
        embedder=fake_embedder,
        threshold=0.95,
    )
    assert r.matched is False
    assert r.score < 0.95


def test_diff_dispatch_semantic_requires_embedder():
    with pytest.raises(ValueError):
        diff("a", "b", mode="semantic")


def test_diff_dispatch_semantic_with_embedder():
    r = diff("hi", "hi", mode="semantic", embedder=fake_embedder)
    assert r.matched is True
