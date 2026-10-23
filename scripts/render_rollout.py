"""Render a policy rollout to mp4 on CPU (llvmpipe). Seeds are given explicitly, never cherry-picked:
the default renders seeds 10000..10002 (the first three evaluation seeds).

    .venv/bin/python scripts/render_rollout.py <run> --test walk [--seeds 10000 10001] [--dr]
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# CPU-only: never touch the GPU from evaluation (training owns it). Params unpickling imports JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slender_biped.render import Renderer, write_video  # noqa: E402  (sets GL env first)
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, run_episode  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import TESTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--test", default="stand")
    ap.add_argument("--seeds", type=int, nargs="*", default=[10000, 10001, 10002])
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--cam", default="track")
    a = ap.parse_args()
    cfg, run_dir = load_run(a.run)
    t = TESTS[a.test]
    out = ROOT / "assets" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        ec = EvalConfig.from_train(cfg["env"], dr=a.dr, push_vel=t.get("push", 0.0))
        env = CpuBiped(cfg["env"]["model"], ec, seed=seed)
        r = Renderer(env.m, 640, 480)
        r.set_groups(shells=True)
        res, frames = run_episode(env, NumpyPolicy(run_dir / a.params), a.seconds or t["seconds"], t["sched"],
                                  renderer=r, cam=a.cam, heading_hold=t.get("heading_hold", False))
        r.close()
        path = out / f"{a.run}__{a.test}{'_dr' if a.dr else ''}_seed{seed}.mp4"
        write_video(path, frames, fps=int(1 / (2 * ec.ctrl_dt)))
        print(path, f"survived {res['survived_s']:.1f}s fell={res['fell']} touchdowns={res['touchdowns']} "
                    f"forward={res['forward_m']:.2f}m")


if __name__ == "__main__":
    main()
