"""Evaluate stand controllers on the servo model fitted to the hardware (results/servo_fit.json).

    .venv-hw/bin/python scripts/eval_fitted.py hw_stand_push2 hw_stand_fit5v hw_stand_fit7v pd [--n 40]

For each controller (an RL run in runs/, or "pd" = scripts/pd_balance.py gains from results/pd_balance.json), runs
--n episodes of --secs on slender_biped/evaluation.CpuBiped in four conditions:
    5V        fitted servo model, 5 V strength/speed (0.68), nominal (no randomization)
    5V_dr     same + randomization (servo profile x0.6-1.5, dead time, voltage 0.60-0.75, masses, friction, gains)
    5V_push   5V_dr + random 0.3 m/s shoves every 2-4 s
    7V_push   7.4 V version of 5V_push (voltage 0.90-1.05)
Reports the fraction of episodes still standing at the end and the mean time before a fall.
Writes results/eval_fitted.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "hardware/leg_p0/software"))
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

FIT = dict(prof_acc=(7.0, 35.0), kp=8.0, reset_joint_noise=0.02, reset_tilt=0.02)   # hardware-like start (~1 deg)
CONDS = {
    "5V": dict(dr=False, latency_sub=5, strength_scale=0.68, speed_scale=0.68, push_vel=0.0),
    "5V_dr": dict(dr=True, dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1), dr_volt=(0.6, 0.75), push_vel=0.0),
    "5V_push": dict(dr=True, dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1), dr_volt=(0.6, 0.75), push_vel=0.3),
    "7V_push": dict(dr=True, dr_prof_acc=(0.6, 1.6), dr_delay_steps=(0, 1), dr_volt=(0.9, 1.05), push_vel=0.3),
    "5V_push005": dict(dr=True, dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1), dr_volt=(0.6, 0.75), push_vel=0.05),
    "5V_push01": dict(dr=True, dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1), dr_volt=(0.6, 0.75), push_vel=0.1),
    "7V": dict(dr=True, dr_prof_acc=(0.6, 1.6), dr_delay_steps=(0, 1), dr_volt=(0.9, 1.05), push_vel=0.0),
    "7V_push005": dict(dr=True, dr_prof_acc=(0.6, 1.6), dr_delay_steps=(0, 1), dr_volt=(0.9, 1.05), push_vel=0.05),
    "7V_push01": dict(dr=True, dr_prof_acc=(0.6, 1.6), dr_delay_steps=(0, 1), dr_volt=(0.9, 1.05), push_vel=0.1),
    "7V_push02": dict(dr=True, dr_prof_acc=(0.6, 1.6), dr_delay_steps=(0, 1), dr_volt=(0.9, 1.05), push_vel=0.2),
    # what-if: hips with the knees' fast motion profile (35 rad/s^2), i.e. if the hip firmware ramp were lifted
    "5V_push01_fasthip": dict(dr=True, prof_acc=(35.0, 35.0), dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1),
                              dr_volt=(0.6, 0.75), push_vel=0.1),
    "5V_fasthip_dr": dict(dr=True, prof_acc=(35.0, 35.0), dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1),
                          dr_volt=(0.6, 0.75), push_vel=0.0),
    "5V_push005_fasthip": dict(dr=True, prof_acc=(35.0, 35.0), dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1),
                               dr_volt=(0.6, 0.75), push_vel=0.05),
    "orig_servo_push01": dict(dr=True, prof_acc=(0.0, 0.0), push_vel=0.1),   # the idealized training servo, 7.4 V
}
ENV_BASE = None
RAMP_S = 1.0                                     # policy_stand.py run --ramp default


def make_ctrl(name):
    if name == "pd":
        import pd_balance as pdb
        gains = dict(json.loads((ROOT / "results/pd_balance.json").read_text())["volt_0.68_hw"]["gains"], ki_p=-0.5)
        state = {}                                     # self-trim integrator (one controller per episode)
        return (lambda obs: pdb.controller(obs, gains, state) / pdb.ACTION_SCALE), None
    import policy_stand as ps
    pol = ps.Policy(name)
    return pol, pol.env


def episode(args):
    name, cond, seed, secs = args
    ctrl, env_cfg = make_ctrl(name)
    base = env_cfg or json.loads((ROOT / "runs/hw_stand_push2/config.json").read_text())["env"]
    over = dict(FIT, **CONDS[cond])          # a condition's own prof_acc overrides the fitted one
    cfg = EvalConfig.from_train(base, **over)
    env = CpuBiped("hw_p0_8dof", cfg, seed=seed)
    obs = env.reset()
    n = int(secs / cfg.ctrl_dt)
    for k in range(n):
        if env_cfg is None:                     # balancer: no blend needed
            obs, _ = env.step(ctrl(obs))
        else:                                   # RL policy: blended in over RAMP_S like the robot runner, which feeds
            a = np.clip(ctrl(obs), -1, 1)       # back the un-blended action as the previous action
            obs, _ = env.step(min(1.0, k * cfg.ctrl_dt / RAMP_S) * a)
            obs[22:30] = a
        if env.fallen():
            return k * cfg.ctrl_dt
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrls", nargs="+")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    a = ap.parse_args()
    out = ROOT / "results/eval_fitted.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    with mp.Pool(a.procs) as pool:
        for c in a.ctrls:
            res.setdefault(c, {})
            for cond in a.conds:
                t = np.array(pool.map(episode, [(c, cond, 2000 + s, a.secs) for s in range(a.n)]))
                res[c][cond] = dict(stand=float(np.mean(t >= a.secs - 1e-6)), mean_time=float(t.mean()), n=a.n,
                                    secs=a.secs)
                print(f"{c:16s} {cond:8s} standing at {a.secs:.0f} s: {100 * res[c][cond]['stand']:5.1f} %   "
                      f"mean time {t.mean():5.1f} s", flush=True)
            out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
