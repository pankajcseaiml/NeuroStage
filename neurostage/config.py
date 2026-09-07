"""Configuration loader: merges base.yaml with any override YAML and CLI key=value pairs."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set cfg[a][b][c] = value from 'a.b.c=value' style CLI override."""
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Auto-cast: try int, float, bool, then str
    raw = str(value)
    if raw.lower() in ("true", "false"):
        d[keys[-1]] = raw.lower() == "true"
    else:
        for cast in (int, float):
            try:
                d[keys[-1]] = cast(raw)
                return
            except ValueError:
                pass
        d[keys[-1]] = raw


class Config(dict):
    """A dict subclass that supports attribute access: cfg.experiment.name."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")
        if isinstance(val, dict):
            return Config(val)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(base: str | Path = "configs/base.yaml", *overrides: str) -> Config:
    """
    Load and merge configuration.

    Parameters
    ----------
    base : path to the base YAML config.
    *overrides : extra YAML paths OR dotted CLI key=value strings.

    Returns
    -------
    Config : a deeply-merged dict with attribute access.

    Examples
    --------
    >>> cfg = load_config("configs/base.yaml", "configs/ablation_cnn_only.yaml", "training.lr=5e-4")
    """
    with open(base) as f:
        cfg: dict = yaml.safe_load(f)

    for override in overrides:
        if "=" in override and not Path(override.split("=")[0]).exists():
            # dotted key=value CLI override
            key, val = override.split("=", 1)
            _set_nested(cfg, key.strip(), val.strip())
        else:
            # YAML file override
            with open(override) as f:
                extra = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, extra)

    return Config(cfg)


def config_from_argv(base: str | Path = "configs/base.yaml") -> Config:
    """Convenience: load_config with sys.argv[1:] as overrides (for CLI scripts)."""
    return load_config(base, *sys.argv[1:])
