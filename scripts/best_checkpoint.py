"""Declared checkpoint-selection rule: highest in-training (GPU, deterministic, separate seeds) eval reward.

    .venv/bin/python scripts/best_checkpoint.py <run>   -> prints params_XXXXM.pkl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1]
rows = [json.loads(x) for x in (ROOT / "runs" / run / "progress.jsonl").read_text().splitlines() if x.strip()]
# Amended 2026-10-23 (before any CPU result of the affected run was seen): untrained step-0 checkpoints are
# excluded. On flat feet an untrained, near-zero-action policy holds pose passively and can score highest.
rows = [r for r in rows if r["step"] > 0] or rows
best = max(rows, key=lambda r: r.get("eval/episode_reward", float("-inf")))
avail = sorted(p.name for p in (ROOT / "runs" / run).glob("params_*M.pkl") if p.name != "params_0000M.pkl")
name = f"params_{int(best['step'] / 1e6):04d}M.pkl"
if name not in avail:  # progress step and checkpoint name are written at the same eval
    name = min(avail, key=lambda n: abs(int(n[7:11]) - best["step"] / 1e6))
print(name)
