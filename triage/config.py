"""YAML config loading with dotted command-line overrides.

Example:  load_config("configs/default.yaml", ["train.epochs=2", "model.backbone=tiny"])
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _parse_value(text: str) -> Any:
    # YAML parsing gives us ints, floats, bools, lists and null for free.
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)
