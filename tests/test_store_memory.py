from prompt_replay import InMemoryStore, RecordedEntry


def test_in_memory_round_trip():
    store = InMemoryStore()
    e1 = RecordedEntry(request={"model": "a"}, response="hi")
    e2 = RecordedEntry(request={"model": "b"}, response={"text": "yo"})
    store.write(e1)
    store.write(e2)
    out = list(store.read_all())
    assert len(out) == 2
    assert out[0].request == {"model": "a"}
    assert out[1].response == {"text": "yo"}


def test_in_memory_len_grows():
    store = InMemoryStore()
    assert len(store) == 0
    store.write(RecordedEntry())
    store.write(RecordedEntry())
    store.write(RecordedEntry())
    assert len(store) == 3


def test_in_memory_seed_entries():
    seed = [RecordedEntry(request={"k": 1}), RecordedEntry(request={"k": 2})]
    store = InMemoryStore(seed)
    assert len(store) == 2
    ids = [e.request["k"] for e in store.read_all()]
    assert ids == [1, 2]


def test_iter_is_snapshot_safe():
    store = InMemoryStore()
    for i in range(3):
        store.write(RecordedEntry(request={"i": i}))
    it = store.read_all()
    store.write(RecordedEntry(request={"i": 99}))
    seen = [e.request["i"] for e in it]
    # Snapshot semantics: iterator does not include entries added after read_all.
    assert seen == [0, 1, 2]
