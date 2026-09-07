"""Unit tests for NeuroStage configuration loading and merging."""
from __future__ import annotations

from pathlib import Path

import pytest

from neurostage.config import Config, _deep_merge, _set_nested, load_config


def test_deep_merge_basic():
    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    override = {"a": 2, "nested": {"y": 99, "z": 30}, "b": 3}
    merged = _deep_merge(base, override)

    assert merged["a"] == 2
    assert merged["b"] == 3
    assert merged["nested"]["x"] == 10
    assert merged["nested"]["y"] == 99
    assert merged["nested"]["z"] == 30


def test_set_nested_casting():
    cfg = {}
    _set_nested(cfg, "model.transformer.n_layers", "4")
    assert cfg["model"]["transformer"]["n_layers"] == 4

    _set_nested(cfg, "training.lr", "0.0005")
    assert cfg["training"]["lr"] == 0.0005

    _set_nested(cfg, "training.use_class_weights", "true")
    assert cfg["training"]["use_class_weights"] is True

    _set_nested(cfg, "training.use_class_weights", "false")
    assert cfg["training"]["use_class_weights"] is False

    _set_nested(cfg, "experiment.name", "ablation_run")
    assert cfg["experiment"]["name"] == "ablation_run"


def test_config_attribute_access():
    c = Config({"experiment": {"name": "test_exp", "seed": 42}})
    assert c.experiment.name == "test_exp"
    assert c.experiment.seed == 42

    with pytest.raises(AttributeError):
        _ = c.nonexistent_key


def test_load_base_config():
    base_path = Path("configs/base.yaml")
    if not base_path.exists():
        pytest.skip("configs/base.yaml not found")

    cfg = load_config(str(base_path), "training.epochs=1")
    assert cfg.training.epochs == 1
    assert cfg.model.n_classes == 5
