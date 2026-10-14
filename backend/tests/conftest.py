"""Shared test setup."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.file_store import LocalFileStore, configure_file_store


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch):
    """
    Keep every test on the in-memory checkpointer, even when backend/.env
    sets DATABASE_URL for local runs. The Postgres integration test connects
    through TEST_DATABASE_URL explicitly instead.
    """
    monkeypatch.setenv("DATABASE_URL", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _temporary_file_store(tmp_path):
    """Generated files go to a per-test folder, never to backend/data."""
    store = LocalFileStore(tmp_path / "sessions")
    configure_file_store(store)
    yield store
    configure_file_store(None)
