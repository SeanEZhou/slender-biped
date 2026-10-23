"""Config loading: geometry variants and actuator assignments."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
MODEL_DIR = ROOT / "models"

JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
SIDES = {"L": 1.0, "R": -1.0}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def load_geometry(variant: str = "baseline", foot: str | None = None, ankle: bool | None = None,
                  actuator_set: str | None = None, overrides: dict | None = None) -> dict:
    g = load_yaml("geometry.yaml")
    if variant not in g["variants"]:
        raise KeyError(f"unknown geometry variant {variant!r}; have {list(g['variants'])}")
    cfg = deep_merge(g["base"], g["variants"][variant])
    if foot is not None:
        cfg["foot"] = foot
    if ankle is not None:
        cfg["ankle"] = ankle
    if actuator_set is not None:
        cfg["actuator_set"] = actuator_set
    cfg = deep_merge(cfg, overrides or {})
    cfg["variant"] = variant
    return cfg


def load_actuators() -> dict:
    return load_yaml("actuators.yaml")


def joint_servos(cfg: dict, act: dict | None = None) -> dict[str, dict]:
    """Map joint type -> servo spec dict (with 'model' key) for the cfg's actuator set."""
    act = act or load_actuators()
    assign = act["assignments"][cfg["actuator_set"]]
    out = {}
    for j in JOINTS:
        if j == "ankle" and not cfg["ankle"]:
            continue
        spec = dict(act["catalog"][assign[j]])
        spec["model"] = assign[j]
        out[j] = spec
    return out


def model_name(cfg: dict) -> str:
    dof = 10 if cfg["ankle"] else 8
    return f"{cfg['variant']}_{cfg['foot']}_{dof}dof"
