"""Tests for cjm_capability_whisper.capability — pure-compute Whisper tool.

Projected from the capability notebook's test cells at the c25780e8 flip
(hermetic: initialize is lazy config-apply, no model download; the transcribe
path is exercised by the e2e harnesses)."""
from dataclasses import fields

import pytest

from cjm_capability_whisper.capability import (WhisperCapabilityConfig,
                                               WhisperLocalCapability)
from cjm_substrate.core.capability import ToolCapability
from cjm_substrate.utils.validation import SCHEMA_ENUM, dict_to_config


def test_pure_compute_surface():
    capability = WhisperLocalCapability()
    assert isinstance(capability, ToolCapability)
    assert capability.config_class.__name__ == "WhisperCapabilityConfig"
    assert capability.version
    # native-surface model: pure-compute transcribe replaces the fused execute
    assert hasattr(capability, "transcribe") and not hasattr(capability, "execute")
    assert not hasattr(capability, "supported_formats")


def test_model_enum_and_validation():
    model_field = next(f for f in fields(WhisperCapabilityConfig) if f.name == "model")
    models = model_field.metadata.get(SCHEMA_ENUM, [])
    assert "tiny" in models

    cfg = dict_to_config(WhisperCapabilityConfig, {"model": "tiny"}, validate=True)
    assert cfg.model == "tiny"
    with pytest.raises(ValueError):
        dict_to_config(WhisperCapabilityConfig, {"model": "invalid"}, validate=True)
    with pytest.raises(ValueError):
        dict_to_config(WhisperCapabilityConfig, {"model": "tiny", "temperature": 1.5},
                       validate=True)


def test_initialize_and_reinitialize_lazy():
    capability = WhisperLocalCapability()
    capability.initialize({"model": "tiny", "device": "cpu"})
    assert capability.get_current_config()["model"] == "tiny"
    # idempotent re-init with the same config, then a config change — both are
    # lazy config-applies here (no model was loaded); the unload path is e2e
    capability.initialize({"model": "tiny", "device": "cpu"})
    capability.initialize({"model": "base", "device": "cpu"})
    assert capability.get_current_config()["model"] == "base"


def test_config_schema_for_ui():
    schema = WhisperLocalCapability().get_config_schema()
    assert schema["name"] == "WhisperCapabilityConfig"
    assert len(schema["properties"]) == len(fields(WhisperCapabilityConfig))
    assert schema["properties"]["model"].get("enum")
