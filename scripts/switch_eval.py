"""Walk -> stop -> walk with two policies: a walker while the command is non-zero, a quiet stander at zero (round 4).

    .venv/bin/python scripts/switch_eval.py <walk_run> <walk_params> <stand_run> <stand_params> [--n 50] [--delay 0.3]

The walker (reference-residual actions) hands over to the stander (absolute actions) `delay` s after the command
drops to zero (with --double-support: at the next double-support window of the walker's gait clock), and takes back control as soon as it rises. Both share the observation layout; the environment
uses the walker's config (e.g. tip_condim); round-5 extra walker inputs sit in front of the base layout. Schedule: 0.15 m/s for 6 s, zero for 6 s, 0.15 m/s for 6 s.
Success = no fall, >= 0.3 m covered in each walking phase and no lift-off in the last 3 s
of the stop phase.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("walk_run"); ap.add_argument("walk_params"); ap.add_argument("stand_run"); ap.add_argument("stand_params")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--delay", type=float, default=0.3)
ap.add_argument("--double-support", action="store_true",
                help="hand over only in a double-support window of the walker's reference clock (no contact sensing)")
a = ap.parse_args()
wcfg, wrd = load_run(a.walk_run)
_, srd = load_run(a.stand_run)
walker, stander = NumpyPolicy(wrd / a.walk_params), NumpyPolicy(srd / a.stand_params)
sched = [(0.0, 0.15), (6.0, 0.0), (12.0, 0.15)]
ok, falls, quiet, lifts, moved, vr, vm = 0, 0, 0, [], [], [], []
for s in range(a.n):
    env = CpuBiped(wcfg["env"]["model"], EvalConfig.from_train(wcfg["env"]), seed=10000 + s)
    obs = env.reset((0.15, 0.0, 0.0))
    yaw0 = yaw_of(env.d.qpos[3:7])
    dt = env.cfg.ctrl_dt
    zero_t, xs, fell, lo, vmax, vs2, standing = None, {}, False, 0, 0.0, [], False
    low, _ = env.tip_low(); con_prev = low < 0.0015
    for k in range(int(18.0 / dt)):
        t = k * dt
        vx = [v for t0, v in sched if t >= t0][-1]
        env.cmd[0] = vx
        err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
        env.cmd[2] = float(np.clip(err, -0.3, 0.3)) if vx > 0 else 0.0
        obs[-5:-2] = env.cmd
        zero_t = (zero_t if zero_t is not None else t) if vx == 0 else None
        if zero_t is None:
            standing = False
        elif not standing and t - zero_t >= a.delay:
            if a.double_support:
                from slender_biped.ref_gait import lookup
                standing = bool(np.all(lookup(env.ref, 0.15, env.phase)[2]))
            else:
                standing = True
        env.cfg.ref_residual = not standing
        # a round-5 walker has extra inputs in front of the shared base layout; the stander reads only the base
        obs, _ = env.step(stander(obs[-len(stander.mean):]) if standing else walker(obs))
        for t0 in (0.0, 6.0, 12.0, 18.0 - dt):
            if abs(t - t0) < dt / 2:
                xs[t0] = env.d.qpos[0:2].copy()
        low, _ = env.tip_low(); con = low < 0.0015
        if 9.0 <= t < 12.0:
            lo += int(np.sum(con_prev & ~con))
            vmax = max(vmax, float(np.hypot(*env.d.qvel[0:2])))
            vs2.append(float(np.sum(env.d.qvel[0:2] ** 2)))
        con_prev = con
        if env.fallen():
            fell = True
            break
    falls += fell
    if not fell:
        fwd = np.array([np.cos(yaw0), np.sin(yaw0)])
        m1, m2 = float((xs[6.0] - xs[0.0]) @ fwd), float((xs[18.0 - dt] - xs[12.0]) @ fwd)
        vrms = float(np.sqrt(np.mean(vs2))); vr.append(vrms); vm.append(vmax)
        q = lo == 0
        quiet += q; lifts.append(lo); moved.append((m1, m2))
        ok += q and m1 >= 0.3 and m2 >= 0.3
n = a.n
print(f"{a.walk_run}+{a.stand_run}: success {ok/n:.0%} | falls {falls/n:.0%} | quiet stop {quiet/n:.0%} | "
      f"stop lift-offs median {np.median(lifts) if lifts else float('nan'):.0f} | walk 1/2 median "
      f"{np.median([m[0] for m in moved]) if moved else 0:.2f}/{np.median([m[1] for m in moved]) if moved else 0:.2f} m")
print(f"  stop phase (last 3 s): base speed RMS median {np.median(vr):.3f} m/s, peak median {np.median(vm):.3f} m/s")
