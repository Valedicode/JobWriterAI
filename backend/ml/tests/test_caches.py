"""The teacher run's caches return exactly what was stored and fetch only what is missing.

Run from backend/:  python -m pytest ml/tests
"""
from ml.labeling.caches import EmbeddingCache, ExtractionCache, Store


def _fetcher(calls):
    def fetch(texts, model="m"):
        calls.append(list(texts))
        return [[float(len(t)), 0.1 + len(t) / 3] for t in texts]
    return fetch


def test_embeddings_are_fetched_once_and_returned_in_order(tmp_path):
    calls = []
    first = EmbeddingCache(Store(tmp_path / "e.sqlite", "embeddings")).wrap(_fetcher(calls))(
        ["python", "go", "python"], model="m")
    # A second run opens the same file: only the new text is fetched.
    again = EmbeddingCache(Store(tmp_path / "e.sqlite", "embeddings")).wrap(_fetcher(calls))(
        ["go", "sql"], model="m")
    assert calls == [["python", "go"], ["sql"]]
    assert first == [[6.0, 0.1 + 6 / 3], [2.0, 0.1 + 2 / 3], [6.0, 0.1 + 6 / 3]]  # bit-identical floats
    assert again == [[2.0, 0.1 + 2 / 3], [3.0, 0.1 + 3 / 3]]


def test_embeddings_of_another_model_are_fetched_again():
    calls = []
    cached = EmbeddingCache(Store(None, "embeddings")).wrap(_fetcher(calls))
    cached(["go"], model="a")
    cached(["go"], model="b")
    assert calls == [["go"], ["go"]]


def test_extractions_are_kept_per_teacher_version_and_kind():
    store = Store(None, "extractions")
    ExtractionCache(store, "teacher-v1").put("resume", "text", {"cv": {"skills": ["Go"]}})
    assert ExtractionCache(store, "teacher-v1").get("resume", "text") == {"cv": {"skills": ["Go"]}}
    assert ExtractionCache(store, "teacher-v2").get("resume", "text") is None
    assert ExtractionCache(store, "teacher-v1").get("job", "text") is None


def test_many_keys_are_read_in_chunks():
    store = Store(None, "t")
    store.put_many({f"k{i}": bytes([i % 256]) for i in range(1200)})
    found = store.get_many([f"k{i}" for i in range(1200)] + ["missing"])
    assert len(found) == 1200 and found["k1199"] == bytes([1199 % 256])
