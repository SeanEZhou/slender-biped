"""Simulated tap test (measuring.md E7) and weight shift (E6) under the stance hold, to fit the give.

    .venv/bin/python scripts/tap_sim.py --model hw_ank3_b150_m2_10dof --flex 'hip_roll,10' 'hip_pitch,10' --series --tc 0.005

Tap: a 2-sample sideways (or fore-aft) push at the pelvis's centre of mass (the lumped upper body), then a damped sine
fitted to the pelvis gyro (frequency, damping ratio). Robot, full body: roll 4.8-5.7 Hz, zeta 0.14-0.19; legs only:
no ring. Shift: E6's ramp (0.5 deg/s of hip roll); robot, full body: tips at 3-5 deg.
"""
import argparse, os, sys, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "hardware/leg_p0/software"))
import numpy as np
from scipy.optimize import curve_fit
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run

fit_f = lambda tt, a, z, fr, ph, c: a * np.exp(-z * 2 * np.pi * fr * tt) * np.cos(2 * np.pi * fr * tt + ph) + c

def make(model, flex, series, tc, extra):
    cfg, _ = load_run(ROOT / "runs" / "fb2_lean")
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, dr_volt=(0.68, 0.68), dr_delay_steps=(0, 0), dr_kp=(1.0, 1.0),
                kp_hip_roll=2.5, hidden_off=(("knee", -0.045),), drag_back=0.0, hidden_flex=flex, noise_qd=0.0,
                flex_series=series, flex_tc=tc)
    over.update(extra)
    env = CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=1)
    return env

def tap(env, axis, force=3.0):
    env.reset((0, 0, 0)); env.ref = None
    W = []
    for i in range(110):
        if i == 50: env.d.xfrc_applied[1, 1 - axis] = force * (1 if axis == 0 else -1)
        if i == 52: env.d.xfrc_applied[1, :] = 0.0
        env.step(np.zeros(env.nj)); W.append(np.degrees(env.d.qvel[3 + axis]))
    x = np.array(W[53:93]); tt = np.arange(len(x)) * 0.02
    best = None
    for f0 in (2, 4, 6, 9, 13, 18):
        try:
            p, _ = curve_fit(fit_f, tt, x, p0=[x[0], 0.2, f0, 0, 0], bounds=([-500, 0, 0.5, -7, -50], [500, 1.5, 24, 7, 50]), maxfev=20000)
            r = np.sum((fit_f(tt, *p) - x) ** 2)
            if best is None or r < best[1]: best = (p, r)
        except Exception: pass
    p, r = best
    return dict(f=p[2], z=p[1], peak=np.abs(W[50:56]).max(), left=np.sqrt(np.mean(x[10:30] ** 2)) / max(np.abs(x[:5]).max(), 1e-6), fallen=env.fallen())

def shift(env, rate=0.5, s_max=9.5):
    import exercise_poses as ep
    env.reset((0, 0, 0)); env.ref = None
    q_nom = env.q_nom.copy(); tr = ep.ShiftRamp("L", rate, s_max)
    from slender_biped.evaluation import quat_rot_t
    roll0 = None
    for i in range(int(tr.duration / 0.02)):
        q, _, _ = tr(i * 0.02)
        a = np.clip((np.asarray(q) - q_nom) / env.cfg.action_scale, -1, 1)
        env.step(a)
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0])); roll = math.degrees(math.atan2(g[1], -g[2]))
        if roll0 is None: roll0 = roll
        if abs(roll - roll0) > 8 or env.fallen():
            return math.degrees(abs(tr.s(i * 0.02))), roll - roll0
    return None, roll - roll0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hw_ank3_b150_m2_10dof")
    ap.add_argument("--flex", nargs="*", default=["hip_roll,3.5"])
    ap.add_argument("--series", action="store_true"); ap.add_argument("--tc", type=float, default=0.01)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-shift", action="store_true")
    a = ap.parse_args()
    import ast
    flex = tuple(tuple([x.split(",")[0]] + [float(v) for v in x.split(",")[1:]]) for x in a.flex)
    extra = {k: ast.literal_eval(v) for k, v in (x.split("=", 1) for x in a.set)}
    env = make(a.model, flex, a.series, a.tc, extra)
    r, p = tap(env, 0), tap(env, 1)
    sh = (None, 0) if a.no_shift else shift(env)
    print(f"{a.model} flex {a.flex} series {a.series} tc {a.tc} {' '.join(a.set)} | roll tap {r['f']:4.1f} Hz z {r['z']:.2f} left {100*r['left']:3.0f}% "
          f"| pitch tap {p['f']:4.1f} Hz z {p['z']:.2f} left {100*p['left']:3.0f}% | shift: tips at {('%.1f deg' % sh[0]) if sh[0] else 'never'} (pelvis roll {sh[1]:+.1f})", flush=True)

if __name__ == "__main__":
    main()
