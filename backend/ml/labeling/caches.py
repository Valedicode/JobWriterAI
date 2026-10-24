"""On-disk caches for the teacher run, in SQLite so parallel workers and a restarted run share them.

- Extractions are keyed by the teacher version and the text's sha256. Each distinct resume and job
  is extracted once (the train split has 642 resumes and 280 jobs behind its 6,241 rows), and every
  row with the same resume sees the same extraction.
- Embeddings are keyed by model and text. The scorer embeds skill names one request at a time, and
  the same names recur across most pairs.

A cache belongs to one teacher version: extractions from another version are never reused, because
the version is part of the key.
"""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

_CHUNK = 500  # stays under SQLite's limit on query parameters


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Store:
    """A thread-safe key -> bytes table in a SQLite file, or in memory when ``path`` is None."""

    def __init__(self, path: Path | None, table: str):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._table = table
        self._lock = threading.Lock()
        with self._lock:
            if path is not None:
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
            self._db.commit()

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        with self._lock:
            for i in range(0, len(keys), _CHUNK):
                chunk = keys[i:i + _CHUNK]
                marks = ",".join("?" * len(chunk))
                found.update(self._db.execute(f"SELECT key, value FROM {self._table} WHERE key IN ({marks})", chunk))
        return found

    def put_many(self, items: dict[str, bytes]) -> None:
        with self._lock:
            self._db.executemany(f"INSERT OR REPLACE INTO {self._table} VALUES (?, ?)", items.items())
            self._db.commit()


class ExtractionCache:
    """Extraction results (JSON) by kind ("resume" or "job"), teacher version and text."""

    def __init__(self, store: Store, version: str):
        self.store, self.version = store, version

    def _key(self, kind: str, text: str) -> str:
        return f"{kind}:{self.version}:{sha256(text)}"

    def get(self, kind: str, text: str) -> dict | None:
        key = self._key(kind, text)
        raw = self.store.get_many([key]).get(key)
        return json.loads(raw) if raw is not None else None

    def put(self, kind: str, text: str, value: dict) -> None:
        self.store.put_many({self._key(kind, text): json.dumps(value).encode("utf-8")})


class EmbeddingCache:
    """Embedding vectors by model and text, stored as float64 so a cached vector is bit-identical."""

    def __init__(self, store: Store):
        self.store = store

    def wrap(self, fetch):
        """``fetch(texts, model=...)`` (the scorer's ``get_embeddings``), answered from the cache
        first. The texts not cached yet are fetched in one call, in their original order."""
        def cached(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
            keys = [f"{model}:{sha256(t)}" for t in texts]
            found = self.store.get_many(list(dict.fromkeys(keys)))
            missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
            if missing:
                vectors = fetch(missing, model=model)
                new = {f"{model}:{sha256(t)}": np.asarray(v, dtype=np.float64).tobytes()
                       for t, v in zip(missing, vectors)}
                self.store.put_many(new)
                found.update(new)
            return [np.frombuffer(found[k], dtype=np.float64).tolist() for k in keys]
        return cached
