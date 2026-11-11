"""How safe are the steps? (2026-11-09: "if the simulation is already on the verge of tipping it'd never work in real life")

    .venv/bin/python scripts/step_quality.py <run> [--params params_0020M.pkl] [--vx 0.008] [--n 8] [--seconds 40]
                                              [--floors 0.004,0.1,0.3,0.8] [--roll-bias 2]

Per floor, starting from standing, the robot's IMU filter in the loop, servo / mass randomization per seed:

* steps: a foot leaves the floor (> 3 mm), stays up >= 0.1 s and lands >= 1 cm further forward; 6+ = walks with six
  steps or more; falls
* touchdown: the foot's downward speed in the last 20 ms before it lands (mm/s, median / max over all landings)
* after touchdown (0.6 s): peak body tilt (deg) and peak roll/pitch rate (deg/s), median / worst over landings
* tilt: the worst body tilt over the walk (deg); slide: board travel while loaded (mm per step)
"""
import argparse, math, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of, quat_rot_t  # noqa: E402


def episode(env, pol, vx, seconds):
    obs = env.reset((vx, 0.0, 0.0))
    p0 = env.d.qpos[0:2].copy()
    dt = env.cfg.ctrl_dt
    yaw0 = yaw_of(env.d.qpos[3:7]); fwd = np.array([np.cos(yaw0), np.sin(yaw0)])
    up = np.zeros(2); peak = np.zeros(2); x_off = np.zeros(2); steps = 0; fell = False
    low_hist = [[], []]
    td_v, after = [], []          # touchdown speeds; (start index) of post-touchdown windows
    tilt, rate = [], []
    slide, c_prev, loaded_prev = 0.0, None, np.ones(2, bool)
    import mujoco
    names = [mujoco.mj_id2name(env.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(env.m.ngeom)]
    side = np.array([1 if n.startswith("L_") else 2 if n.startswith("R_") else 0 for n in names])
    foot = np.array([("_tip" in n or "_padcol" in n) for n in names])
    self_hit = 0.0
    tid = env.tip_gid
    scuff_t, scuff_n, scuff_prev = 0.0, 0, np.zeros(2, bool)
    td_tilt, lift_pk = [], []
    rates = []
    for k in range(int(seconds / dt)):
        err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
        env.cmd[2] = float(np.clip(err, -0.3, 0.3)); obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        tilt.append(math.degrees(math.asin(min(1.0, math.hypot(g[0], g[1])))))
        rate.append(math.degrees(np.linalg.norm(env.d.qvel[3:5])))
        low, c = env.tip_low()
        for j in range(env.d.ncon):                      # one foot touching the other foot or the other shin
            g1, g2 = env.d.contact[j].geom1, env.d.contact[j].geom2
            if side[g1] and side[g2] and side[g1] != side[g2] and (foot[g1] or foot[g2]):
                self_hit += dt
                break
        loaded = low < 0.0015
        rates.append(np.degrees(env.d.qvel[3:5]).copy())
        R = env.d.geom_xmat[tid].reshape(2, 3, 3)
        pad_pitch = np.degrees(np.arcsin(np.clip(R[:, 2, 0], -1, 1)))   # toe-up positive
        pad_roll = np.degrees(np.arcsin(np.clip(R[:, 2, 1], -1, 1)))
        if c_prev is not None:                           # scuff: a foot moving > 3 cm/s sideways/forward within 2 mm of
            vxy = np.linalg.norm((c - c_prev)[:, :2], axis=1) / dt   # the floor, not planted (planted = slide)
            sc = (low < 0.002) & (vxy > 0.03) & ~(loaded & loaded_prev)
            scuff_t += dt * float(np.sum(sc)); scuff_n += int(np.sum(sc & ~scuff_prev)); scuff_prev = sc
        if c_prev is not None:
            slide += float(np.sum(np.linalg.norm((c - c_prev)[:, :2], axis=1) * (loaded & loaded_prev)))
        c_prev, loaded_prev = c.copy(), loaded
        for i in range(2):
            low_hist[i].append(low[i])
            if low[i] > 0.003 and up[i] == 0:
                x_off[i] = c[i, :2] @ fwd
            if low[i] > 0.0015:
                up[i] += dt; peak[i] = max(peak[i], low[i])
            else:
                if up[i] >= 0.1 and peak[i] > 0.003:
                    if c[i, :2] @ fwd - x_off[i] >= 0.01:
                        steps += 1
                    td_tilt.append((abs(pad_pitch[i]), abs(pad_roll[i]), pad_pitch[i]))
                    lift_pk.append(peak[i])
                    h = low_hist[i]
                    td_v.append(max(0.0, (h[-2] - h[-1]) / dt if len(h) > 1 else 0.0))
                    after.append(k)
                up[i] = 0.0; peak[i] = 0.0
        if env.fallen():
            fell = True
            break
    tilt, rate = np.array(tilt), np.array(rate)
    w = int(0.6 / dt)
    post = [(tilt[s:s + w].max(), rate[s:s + w].max()) for s in after if s + 1 < len(tilt)]
    rates = np.array(rates) if rates else np.zeros((1, 2))
    return dict(fwd=float((env.d.qpos[0:2] - p0) @ fwd),rms_roll=float(np.sqrt(np.mean(rates[:, 0] ** 2))), rms_pitch=float(np.sqrt(np.mean(rates[:, 1] ** 2))),
                scuff_n=scuff_n, scuff_t=scuff_t, td_tilt=td_tilt, lift_pk=lift_pk, steps=steps, fell=fell, td_v=td_v, post=post, tilt=float(tilt.max()) if len(tilt) else 0.0,
                slide=slide / max(steps, 1), self_hit=self_hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--params")
    ap.add_argument("--vx", type=float, default=0.008)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--floors", default="0.004,0.1,0.3,0.8")
    ap.add_argument("--roll-bias", type=float, default=0.0)
    ap.add_argument("--model", help="evaluate on another model (e.g. another pad)")
    ap.add_argument("--zero", action="store_true", help="zero residual: the run's reference alone (open loop)")
    a = ap.parse_args()
    cfg, rd = load_run(a.run)
    params = a.params or sorted(rd.glob("params_*M.pkl"))[-1].name
    pol = NumpyPolicy(rd / params)
    if a.zero:
        n_act = len(load_run(a.run)[0]["env"].get("model") and __import__("json").loads((ROOT / "models" / f"{cfg['env']['model']}.json").read_text())["joints"])
        pol = lambda obs: np.zeros(n_act)  # noqa: E731
        params = "zero residual (reference only)"
    model = a.model or cfg["env"]["model"]
    print(f"{a.run} {params} on {model} vx {a.vx}, {a.seconds:.0f} s" + (f", roll bias {a.roll_bias:+.1f} deg" if a.roll_bias else ""))
    for tc in (float(x) for x in a.floors.split(",")):
        over = dict(floor_tc=tc, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45,
                    imu_tilt_bias=(np.radians(a.roll_bias), 0.0), dr_imu_tilt=0.0)
        res = [episode(CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=70000 + s), pol, a.vx, a.seconds)
               for s in range(a.n)]
        st = np.array([r["steps"] for r in res]); fl = sum(r["fell"] for r in res)
        tv = 1000 * np.array(sum((r["td_v"] for r in res), [])) if any(r["td_v"] for r in res) else np.zeros(1)
        post = np.array(sum((r["post"] for r in res), [])) if any(r["post"] for r in res) else np.zeros((1, 2))
        print(f"  floor {tc:5.3f}: 6+ steps {sum(st >= 6)}/{a.n}  steps {np.median(st):.0f} ({st.min()}-{st.max()})  falls {fl}/{a.n} | "
              f"touchdown {np.median(tv):3.0f}/{tv.max():3.0f} mm/s | after: tilt {np.median(post[:, 0]):4.1f}/{post[:, 0].max():4.1f} deg, "
              f"rate {np.median(post[:, 1]):3.0f}/{post[:, 1].max():3.0f} deg/s | worst tilt {max(r['tilt'] for r in res):4.1f} | "
              f"slide {1000 * np.median([r['slide'] for r in res]):4.1f} mm/step | feet touching {sum(r['self_hit'] > 0 for r in res)}/{a.n} walks "
              f"({max(r['self_hit'] for r in res):.2f} s max)", flush=True)
        tt = np.array(sum((r["td_tilt"] for r in res), [])) if any(r["td_tilt"] for r in res) else np.zeros((1, 3))
        lp = 1000 * np.array(sum((r["lift_pk"] for r in res), [])) if any(r["lift_pk"] for r in res) else np.zeros(1)
        nst = max(1, int(st.sum()))
        print(f"      wobble rms roll {np.median([r['rms_roll'] for r in res]):4.1f} pitch {np.median([r['rms_pitch'] for r in res]):4.1f} deg/s | "
              f"scuffs {sum(r['scuff_n'] for r in res) / nst:4.2f}/step ({np.median([r['scuff_t'] for r in res]):.2f} s/walk) | "
              f"touchdown tilt pitch {np.median(tt[:, 0]):4.1f}/{tt[:, 0].max():4.1f} (toe-down {np.mean(tt[:, 2] < -2):.0%}) roll "
              f"{np.median(tt[:, 1]):4.1f}/{tt[:, 1].max():4.1f} deg | lift {np.median(lp):4.1f} mm (min {lp.min():4.1f}) | "
              f"forward {100 * np.median([r['fwd'] for r in res]):.0f} cm in {a.seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
