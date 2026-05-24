from prompt_replay import exact_diff


def test_exact_matches_identical_strings():
    r = exact_diff("hello world", "hello world")
    assert r.matched is True
    assert r.score == 1.0
    assert r.mode == "exact"


def test_exact_detects_diff_strings():
    r = exact_diff("hello", "hi")
    assert r.matched is False
    assert r.score == 0.0
    assert r.details["original_len"] == 5
    assert r.details["replayed_len"] == 2


def test_exact_extracts_anthropic_shape():
    a = {"content": [{"type": "text", "text": "hi"}]}
    b = {"content": [{"type": "text", "text": "hi"}]}
    r = exact_diff(a, b)
    assert r.matched is True


def test_exact_extracts_openai_shape():
    a = {"choices": [{"message": {"content": "yo"}}]}
    b = {"choices": [{"message": {"content": "yo"}}]}
    r = exact_diff(a, b)
    assert r.matched is True


def test_exact_unequal_anthropic_text():
    a = {"content": [{"type": "text", "text": "alpha"}]}
    b = {"content": [{"type": "text", "text": "beta"}]}
    r = exact_diff(a, b)
    assert r.matched is False
