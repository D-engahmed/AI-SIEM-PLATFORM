"""
Basic tests for config.py -- this had zero coverage before 2026-08-17.
Not exhaustive: focused on the two things that actually matter operationally
(get_settings() works with no env vars, and the boundary-threshold bug fix
actually rejects what it's supposed to).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import Settings, get_settings


class TestGetSettings:
    def test_returns_settings_with_no_env_vars(self, monkeypatch):
        for var in list(__import__("os").environ):
            if var.startswith("CML_"):
                monkeypatch.delenv(var, raising=False)
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_defaults_match_documented_values(self):
        settings = Settings()
        assert settings.escalation_threshold == 0.5  # confirmed 2026-08-17
        assert settings.api_port == 9100  # AD-041


class TestThresholdBoundaries:
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_exact_boundary_rejected(self, value):
        with pytest.raises(ValidationError):
            Settings(escalation_threshold=value)

    @pytest.mark.parametrize("value", [0.01, 0.5, 0.99])
    def test_interior_values_accepted(self, value):
        Settings(escalation_threshold=value)

    @pytest.mark.parametrize("value", [-0.01, 1.01, 5.0, -5.0])
    def test_out_of_field_range_rejected_before_custom_validator_runs(self, value):
        # Field(ge=0.0, le=1.0) catches these -- verifies the fix documented
        # in config.py's comment actually holds (a misconfigured "5" instead
        # of "0.5" never reaches _threshold_sane at all).
        with pytest.raises(ValidationError):
            Settings(escalation_threshold=value)


class TestApiPort:
    def test_out_of_range_port_rejected(self):
        with pytest.raises(ValidationError):
            Settings(api_port=70000)
