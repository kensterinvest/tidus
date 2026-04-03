"""Shared pytest fixtures for Tidus test suite."""

import pytest
from httpx import ASGITransport, AsyncClient

from tidus.main import app
from tidus.settings import get_settings


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
async def client():
    """Async HTTPX client wired to the FastAPI app (no live server needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
