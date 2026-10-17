"""
Where generated documents (tailored CVs, cover letters) are kept.

Every file belongs to one session and lives under ``data/sessions/<id>/``.
The download URL carries the session id, a random UUID4, so it works as an
unguessable capability until user accounts exist (plan item 14). Before this,
PDFs were stored as ``data/<name>_cv_tailored.pdf``: guessable from the
applicant's name, shared between sessions with the same name, and served to
anyone who asked.

``FileStore`` is the interface the rest of the app uses; ``LocalFileStore``
keeps files on local disk. An object-storage implementation (Supabase
Storage / S3) can replace it later without touching the graph nodes.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional, Protocol

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,150}\.pdf$")


def is_session_id(value: str) -> bool:
    return bool(_SESSION_ID_RE.match(value or ""))


UNSAFE_FILENAME_MESSAGE = (
    "The output filename must end in .pdf and use only letters, digits, '.', '_' and '-' "
    "(no folders), e.g. 'ada_lovelace_cv.pdf'."
)


def is_safe_filename(value: str) -> bool:
    """A plain PDF name: no folder part, so a write can't leave its session
    folder, and every name written can also be downloaded."""
    return bool(_FILENAME_RE.match(value or "")) and ".." not in value


class FileStore(Protocol):
    def session_dir(self, session_id: str) -> Path:
        """Directory generated files for this session are written to (created on demand)."""

    def path_for(self, session_id: str, filename: str) -> Optional[Path]:
        """The stored file, or None if the ids are invalid or nothing is stored there."""

    def download_url(self, session_id: str, filename: str) -> str:
        """URL the frontend uses to fetch the file."""

    def delete_session(self, session_id: str) -> None:
        """Remove every file of the session."""

    def expire(self, ttl: timedelta, is_live: Callable[[str], bool]) -> list[str]:
        """Remove sessions untouched for ``ttl`` that are no longer live; return their ids."""


class LocalFileStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _dir(self, session_id: str) -> Path:
        if not is_session_id(session_id):
            raise ValueError(f"not a session id: {session_id!r}")
        return self.root / session_id

    def session_dir(self, session_id: str) -> Path:
        directory = self._dir(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def path_for(self, session_id: str, filename: str) -> Optional[Path]:
        if not is_session_id(session_id) or not is_safe_filename(filename):
            return None
        path = (self.root / session_id / filename).resolve()
        if path.parent != (self.root / session_id).resolve() or not path.is_file():
            return None
        return path

    def download_url(self, session_id: str, filename: str) -> str:
        return f"/api/files/{session_id}/{filename}"

    def delete_session(self, session_id: str) -> None:
        if is_session_id(session_id):
            shutil.rmtree(self.root / session_id, ignore_errors=True)

    def expire(self, ttl: timedelta, is_live: Callable[[str], bool]) -> list[str]:
        if not self.root.is_dir():
            return []
        cutoff = time.time() - ttl.total_seconds()
        removed: list[str] = []
        for directory in self.root.iterdir():
            session_id = directory.name
            if not directory.is_dir() or not is_session_id(session_id):
                continue
            newest = max((p.stat().st_mtime for p in directory.iterdir()), default=directory.stat().st_mtime)
            # A file generated long ago may belong to a session that is still
            # active; only sessions that are gone lose their files.
            if newest < cutoff and not is_live(session_id):
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(session_id)
        return removed


_store: Optional[FileStore] = None
_store_lock = threading.Lock()


def get_file_store() -> FileStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = LocalFileStore(DATA_DIR / "sessions")
    return _store


def configure_file_store(store: Optional[FileStore]) -> None:
    """Swap the store (tests; later an object-storage implementation). None resets it."""
    global _store
    with _store_lock:
        _store = store
