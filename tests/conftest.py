"""Shared fixtures for integration tests."""

import pytest
from pathlib import Path


@pytest.fixture
def repo_root():
    """Return the repo root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def tmp_output(tmp_path):
    """Return a clean temporary output directory."""
    return tmp_path / "test_output"
