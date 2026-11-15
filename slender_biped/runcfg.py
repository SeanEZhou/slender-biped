"""Rebuild a run's EnvConfig from its config.json (round 13), with overrides."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .mjx_env import EnvConfig


def _tup(x):
    return tuple(_tup(v) for v in x) if isinstance(x, list) else x


def env_config(run_dir: str | Path, **over) -> EnvConfig:
    d = json.loads((Path(run_dir) / "config.json").read_text())["env"]
    base = EnvConfig()
    kw = {}
    for f in dataclasses.fields(base):
        if f.name in d:
            v = d[f.name]
            kw[f.name] = _tup(v) if isinstance(getattr(base, f.name), tuple) else v
    kw.update(over)
    return dataclasses.replace(base, **kw)
