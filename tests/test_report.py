from prompt_replay import DiffResult, ReplayEntry, ReplayReport


def _entry(prompt_id, matched=True, score=1.0, error=None, mode="exact"):
    diff = None if error else DiffResult(mode=mode, matched=matched, score=score)
    return ReplayEntry(
        prompt_id=prompt_id,
        request={"model": "x"},
        replayed_request={"model": "y"},
        original="a",
        replayed="a" if matched else "b",
        diff=diff,
        error=error,
    )


def test_report_aggregates_match_drift_error():
    r = ReplayReport()
    r.add(_entry("p1", matched=True, score=1.0))
    r.add(_entry("p2", matched=False, score=0.4))
    r.add(_entry("p3", error="TimeoutError: slow"))
    assert r.stats.total == 3
    assert r.stats.matched == 1
    assert r.stats.drifted == 1
    assert r.stats.errored == 1


def test_report_avg_score_excludes_errors():
    r = ReplayReport()
    r.add(_entry("p1", matched=True, score=1.0))
    r.add(_entry("p2", matched=False, score=0.5))
    r.add(_entry("p3", error="x"))
    # avg over (1.0 + 0.5) / 2
    assert abs(r.stats.avg_score - 0.75) < 1e-9


def test_report_as_table_includes_summary_line():
    r = ReplayReport()
    r.add(_entry("p1", matched=True, score=1.0))
    r.add(_entry("p2", matched=False, score=0.3))
    text = r.as_table()
    assert "prompt_id" in text
    assert "p1" in text and "p2" in text
    assert "total=2" in text
    assert "matched=1" in text


def test_replay_entry_matched_property():
    e_ok = _entry("p1", matched=True, score=1.0)
    e_err = _entry("p2", error="x")
    assert e_ok.matched is True
    assert e_err.matched is False
