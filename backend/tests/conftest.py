"""Shared test setup."""

from __future__ import annotations

import pytest

from app.config import get_settings


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
