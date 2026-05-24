from prompt_replay import json_diff


def test_json_equal_dicts():
    r = json_diff({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert r.matched is True
    assert r.score == 1.0
    assert r.details["changed"] == []


def test_json_changed_leaf():
    r = json_diff({"a": 1, "b": 2}, {"a": 1, "b": 3})
    assert r.matched is False
    assert "b" in r.details["changed"]
    assert r.score < 1.0


def test_json_added_and_removed_keys():
    r = json_diff({"a": 1, "b": 2}, {"a": 1, "c": 9})
    assert "c" in r.details["added"]
    assert "b" in r.details["removed"]
    assert r.matched is False


def test_json_nested_dotted_paths():
    a = {"meta": {"model": "claude-sonnet-4-7", "temp": 0.7}}
    b = {"meta": {"model": "claude-haiku-4-5", "temp": 0.7}}
    r = json_diff(a, b)
    assert "meta.model" in r.details["changed"]
    assert "meta.temp" not in r.details["changed"]


def test_json_list_paths():
    a = {"items": [{"v": 1}, {"v": 2}]}
    b = {"items": [{"v": 1}, {"v": 99}]}
    r = json_diff(a, b)
    assert any("items[1].v" in c for c in r.details["changed"])


def test_json_parses_string_inputs():
    r = json_diff('{"a": 1}', '{"a": 2}')
    assert r.matched is False
    assert "a" in r.details["changed"]


def test_json_returns_unmatched_when_not_json():
    r = json_diff("plain text", "still plain text")
    assert r.matched is False
    assert r.details["reason"] == "not_json_on_one_or_both_sides"
