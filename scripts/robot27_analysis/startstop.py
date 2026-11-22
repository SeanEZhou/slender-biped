"""Round 27: start, stop and restart in the GPU 'sim robot' (gait_eval's robot settings, action low-pass 0.7).
Each walker starts standing (reset pose), the command goes to --vx at once (slewed as in training), back to 0 at
--stop s and to --vx again at --go s, ending at --T s.

Reports: falls per segment; the first two steps after the start and after the restart (swing foot ahead of the stance
foot dx / foot spacing dy, mm, 0.12 s after the scheduled touchdown, as the robot's logs are read) and the hip-roll
spread L - R at the first lift-off (the weight shift; the table's is -13.8 deg for a right step); how long the clock
took to stop after the command went to 0 (walk_state policies), the feet's stagger and body tilt while standing.

    scripts/gpu_python.sh scripts/robot27_analysis/startstop.py <run> [--vx 0.025] [--set k=json ...]
"""
import argparse, functools, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "robot18_analysis"))
import jax, jax.numpy as jp, numpy as np, mujoco
from brax.envs.wrappers import training as bw
from push_eval import ROBOT, load_policy
from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn
from slender_biped.runcfg import env_config
import swing_clear as sc

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--vx", type=float, default=0.025)
ap.add_argument("--stop", type=float, default=8.0); ap.add_argument("--go", type=float, default=13.0); ap.add_argument("--T", type=float, default=19.0)
ap.add_argument("--n", type=int, default=32); ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
W, p, d = load_policy(a.run)
over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=0.7,
            cmd_resample_s=(1000.0, 1000.0), reset_hard_prob=0.0, reset_joint_noise=0.02, reset_tilt=0.02,
            cmd_vx=(a.vx, a.vx), cmd_zero_prob=0.0, cmd_wz=(0.0, 0.0), sup=False)
over.update(ROBOT)
for kv in a.set:
    k, v = kv.split("=", 1); x = json.loads(v)
    over[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
cfg = env_config(d, **over); env = SlenderBipedEnv(cfg); N = a.n; nst = int(a.T / cfg.ctrl_dt); nj = env.nj
ks, kg = int(a.stop / cfg.ctrl_dt), int(a.go / cfg.ctrl_dt)
wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(randomization_fn, rng=jax.random.split(jax.random.PRNGKey(1), N)))
st = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), N))
goal = st.info["cmd_goal"]
st = st.replace(info={**st.info, "phase": jp.zeros_like(st.info["phase"])})   # right foot first, as the robot runner
def body(c, k):
    st, dead = c
    cg = jp.where((k >= ks) & (k < kg), jp.zeros_like(goal), goal)
    st = st.replace(info={**st.info, "cmd_goal": cg})
    st2 = wenv.step(st, W(st.obs["state"])); dead = dead | (st2.done > 0); ps = st2.pipeline_state
    return (st2, dead), (ps.qpos[:, 3:7], st2.info["phase"], ps.qpos[:, 7:7 + nj] + st2.info.get("flex", jp.zeros((N, nj))),
                         dead, st2.info["walking"])
(st, dead), (quat, ph, q, dd, wk) = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(nst)))((st, jp.zeros(N, bool)))
quat, ph, q, dd, wk = map(np.asarray, (quat, ph, q, dd, wk))
w, x, y, z = [quat[..., i] for i in range(4)]
G = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
tilt = np.degrees(np.arccos(np.clip(-G[..., 2], -1, 1)))
tab, zz, ev = sc.table(a.run.split(":")[0]); stn = zz["stance"][-1]; n = len(stn)
m, dm, tid = sc._m, sc._d, sc._tid
def feet(qq, g):
    dm.qpos[:] = 0; dm.qpos[2] = 0.3; dm.qpos[3:7] = sc.gap_fit._quat_from_g(g); dm.qpos[7:] = qq; mujoco.mj_kinematics(m, dm)
    return dm.geom_xpos[tid[0]].copy(), dm.geom_xpos[tid[1]].copy()
J = list(env.joints); iL, iR = J.index("L_hip_roll"), J.index("R_hip_roll")
def first_steps(k, i0, i1):
    out, spread = [], None
    for i in range(i0, i1 - 7):
        if dd[i + 6, k] or len(out) >= 2: break
        p0 = int((ph[i, k] / 2 / np.pi % 1) * n) % n; p1 = int((ph[i + 1, k] / 2 / np.pi % 1) * n) % n
        for s_ in (0, 1):
            if spread is None and stn[p0][s_] >= .5 and stn[p1][s_] < .5:
                spread = (s_, np.degrees(q[i, k, iL] - q[i, k, iR]))
            if stn[p0][s_] < .5 and stn[p1][s_] >= .5 and len(out) < 2:
                L, R = feet(q[i + 6, k], G[i + 6, k]); sw, sd = (L, R) if s_ == 0 else (R, L)
                out.append(("LR"[s_], (sw[0] - sd[0]) * 1000, abs(L[1] - R[1]) * 1000))
    return out, spread
def seg_report(name, i0, i1):
    S = [first_steps(k, i0, i1) for k in range(N) if not dd[i0, k]]
    msg = []
    for j in range(2):
        r = [s[0][j] for s in S if len(s[0]) > j]
        if r:
            msg.append(f"step {j + 1}: dx {np.median([x[1] for x in r]):+.0f} (p10 {np.percentile([x[1] for x in r], 10):+.0f}) / dy {np.median([x[2] for x in r]):.0f} mm")
    sp = [(-1 if s[1][0] == 1 else 1) * s[1][1] for s in S if s[1] is not None]   # toward the stance foot, + = shifted
    msg.append(f"shift at 1st lift-off {np.median(sp):+.1f} deg (p10 {np.percentile(sp, 10):+.1f}; table +13.8)" if sp else "no lift-off")
    print(f"  {name}: " + ", ".join(msg))
up = lambda i: int((~dd[min(i, nst - 1)]).sum())
print(f"startstop {a.run} ({p.name}) vx {a.vx} stop {a.stop}s go {a.go}s T {a.T}s n {N} {' '.join(a.set)}")
print(f"  up: after 3 s {up(150)}, at the stop command {up(ks)}, at the restart {up(kg)}, at the end {up(nst - 1)} of {N}")
seg_report("start", 0, ks)
alive = [k for k in range(N) if not dd[kg - 1, k]]
if cfg.walk_state:
    t_stop = [(np.argmax(~wk[ks:kg, k]) * cfg.ctrl_dt) if (~wk[ks:kg, k]).any() else np.nan for k in alive]
    print(f"  stop: clock stopped {np.nanmedian(t_stop):.2f} s after the command (p90 {np.nanpercentile(t_stop, 90):.2f}), "
          f"never {int(np.isnan(t_stop).sum())} of {len(alive)}")
stag, tl = [], []
for k in alive:
    L, R = feet(q[kg - 1, k], G[kg - 1, k]); stag.append(abs(L[0] - R[0]) * 1000); tl.append(tilt[ks + 150:kg, k].std())
if alive:
    print(f"  standing (before the restart): feet stagger |dx| p50 {np.median(stag):.0f} p90 {np.percentile(stag, 90):.0f} mm, "
          f"tilt wobble std p50 {np.median(tl):.2f} deg")
seg_report("restart", kg, nst)
