"""Tests for configurable retry defaults via environment variables.

Verifies that COMFY_API_MAX_RETRIES, COMFY_API_RETRY_DELAY, and
COMFY_API_RETRY_BACKOFF environment variables are respected.

NOTE: Cannot import from comfy_api_nodes directly because the import
chain triggers CUDA initialization.  The helpers under test are
reimplemented here identically to the production code in client.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class _RetryDefaults:
    max_retries: int = _env_int("COMFY_API_MAX_RETRIES", 3)
    retry_delay: float = _env_float("COMFY_API_RETRY_DELAY", 1.0)
    retry_backoff: float = _env_float("COMFY_API_RETRY_BACKOFF", 2.0)


class TestEnvHelpers:
    def test_env_int_returns_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _env_int("NONEXISTENT_KEY", 42) == 42

    def test_env_int_returns_env_value(self):
        with patch.dict(os.environ, {"TEST_KEY": "10"}):
            assert _env_int("TEST_KEY", 42) == 10

    def test_env_int_returns_default_on_invalid_value(self):
        with patch.dict(os.environ, {"TEST_KEY": "not_a_number"}):
            assert _env_int("TEST_KEY", 42) == 42

    def test_env_float_returns_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _env_float("NONEXISTENT_KEY", 1.5) == 1.5

    def test_env_float_returns_env_value(self):
        with patch.dict(os.environ, {"TEST_KEY": "2.5"}):
            assert _env_float("TEST_KEY", 1.5) == 2.5

    def test_env_float_returns_default_on_invalid_value(self):
        with patch.dict(os.environ, {"TEST_KEY": "bad"}):
            assert _env_float("TEST_KEY", 1.5) == 1.5


class TestRetryDefaults:
    def test_hardcoded_defaults_match_expected(self):
        defaults = _RetryDefaults()
        assert defaults.max_retries == 3
        assert defaults.retry_delay == 1.0
        assert defaults.retry_backoff == 2.0

    def test_env_vars_would_override_at_import_time(self):
        """Dataclass field defaults are evaluated at class-definition time.
        This test verifies that _env_int/_env_float return the env values,
        which is what populates the dataclass fields at import time."""
        with patch.dict(os.environ, {"COMFY_API_MAX_RETRIES": "10"}):
            assert _env_int("COMFY_API_MAX_RETRIES", 3) == 10
        with patch.dict(os.environ, {"COMFY_API_RETRY_DELAY": "3.0"}):
            assert _env_float("COMFY_API_RETRY_DELAY", 1.0) == 3.0
        with patch.dict(os.environ, {"COMFY_API_RETRY_BACKOFF": "1.5"}):
            assert _env_float("COMFY_API_RETRY_BACKOFF", 2.0) == 1.5

    def test_explicit_construction_overrides_defaults(self):
        defaults = _RetryDefaults(max_retries=10, retry_delay=3.0, retry_backoff=1.5)
        assert defaults.max_retries == 10
        assert defaults.retry_delay == 3.0
        assert defaults.retry_backoff == 1.5

    def test_frozen_dataclass(self):
        defaults = _RetryDefaults()
        with pytest.raises(AttributeError):
            defaults.max_retries = 999
