"""Round 29: turn in place, then walk on, in the GPU 'sim robot' (gait_eval's robot settings, action low-pass 0.7).
Each walker starts standing; the command walks at --vx until --t1, turns in place at +-wz (half the walkers left, half
right; vx 0) until --t2, then walks at --vx again until --T. Commands ramp as in training. Heading-command policies
(heading_cmd, z0's lineage): the turn is a new target heading --deg away (the policy sees the corrective yaw rate);
otherwise the turn command is the rate --wz.

Reports per direction: up at the end of each segment; the yaw turned during the turn segment (and until the next walk
has settled, +1.5 s) against the commanded; the heading error at the end; yaw drift while walking; forward travel in
the last walking segment (along the new heading); the hip-yaw range used.

    scripts/gpu_python.sh scripts/robot29/turn_eval.py <run> [--wz 0.15] [--zero] [--set k=json ...]
--zero plays the reference open loop (zero action): a check of the turning reference itself.
"""
import argparse, functools, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jax, jax.numpy as jp, numpy as np
from brax.envs.wrappers import training as bw
from push_eval import ROBOT, load_policy
from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn
from slender_biped.runcfg import env_config

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--vx", type=float, default=0.025); ap.add_argument("--wz", type=float, default=0.15)
ap.add_argument("--deg", type=float, default=60.0)
ap.add_argument("--t1", type=float, default=5.0); ap.add_argument("--t2", type=float, default=15.0); ap.add_argument("--T", type=float, default=22.0)
ap.add_argument("--n", type=int, default=32); ap.add_argument("--zero", action="store_true"); ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
W, p, d = load_policy(a.run)
over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=0.7,
            cmd_resample_s=(1000.0, 1000.0), reset_hard_prob=0.0, reset_joint_noise=0.02, reset_tilt=0.02,
            cmd_vx=(a.vx, a.vx), cmd_zero_prob=0.0, cmd_wz=(0.0, 0.0), sup=False)
over.update(ROBOT)
for kv in a.set:
    k, v = kv.split("=", 1); x = json.loads(v)
    over[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
over["turn_prob"] = 1e-6   # the turn command handling needs turn_prob > 0; never sampled (no reset turns)
cfg = env_config(d, **over)
env = SlenderBipedEnv(cfg); N = a.n; nst = int(a.T / cfg.ctrl_dt); nj = env.nj
k1, k2 = int(a.t1 / cfg.ctrl_dt), int(a.t2 / cfg.ctrl_dt)
wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(randomization_fn, rng=jax.random.split(jax.random.PRNGKey(1), N)))
st = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), N))
sgn = jp.where(jp.arange(N) < N // 2, 1.0, -1.0)
g_walk = jp.tile(jp.array([a.vx, 0.0, 0.0]), (N, 1))
g_turn = jp.stack([jp.zeros(N), jp.zeros(N), sgn * a.wz], -1)
st = st.replace(info={**st.info, "phase": jp.zeros_like(st.info["phase"]), "cmd": jp.zeros((N, 3)), "cmd_goal": g_walk})
pol = (lambda o: jp.zeros(o.shape[:-1] + (nj,))) if a.zero else W
iy = [list(env.joints).index(j) for j in env.joints if j.endswith("hip_yaw")]
HC = bool(cfg.heading_cmd)
if HC:
    g_turn = jp.zeros((N, 3))
def body(c, k):
    st, dead = c
    cg = jp.where((k >= k1) & (k < k2), g_turn, g_walk)
    hd_ = st.info["heading"] + jp.where(k == k1, sgn * np.radians(a.deg), 0.0) if HC else st.info["heading"]
    st = st.replace(info={**st.info, "cmd_goal": cg, "heading": hd_})
    st2 = wenv.step(st, pol(st.obs["state"])); dead = dead | (st2.done > 0); ps = st2.pipeline_state
    return (st2, dead), (ps.qpos[:, 0:2], ps.qpos[:, 3:7], ps.qpos[:, 7:7 + nj][:, jp.array(iy)], dead, st2.info["heading"])
(st, dead), (xy, quat, qy, dd, hd) = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(nst)))((st, jp.zeros(N, bool)))
xy, quat, qy, dd, hd = map(np.asarray, (xy, quat, qy, dd, hd))
w, x, y, z = [quat[..., i] for i in range(4)]
yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), axis=0)
dt = cfg.ctrl_dt
tag = f"{a.run}{' ZERO ACTION' if a.zero else ''} vx {a.vx} {f'heading {a.deg:g} deg' if HC else f'wz {a.wz}'} | walk 0-{a.t1:g} s, turn to {a.t2:g} s, walk to {a.T:g} s | n {N}"
print(f"turn {tag}  turn_yaw {cfg.turn_yaw}")
want = a.deg if HC else np.degrees(a.wz * (a.t2 - a.t1))
for name, sel in (("left", np.arange(N) < N // 2), ("right", np.arange(N) >= N // 2)):
    s_ = 1 if name == "left" else -1
    up = [100 * np.mean(~dd[kk - 1][sel]) for kk in (k1, k2, nst)]
    ok = sel & ~dd[-1]
    ok_t = sel & ~dd[k2 - 1]
    e2 = min(k2 + int(1.5 / dt), nst - 1)
    turned = np.degrees(yaw[e2] - yaw[k1]) * s_
    drift = np.degrees(yaw[k1 - 1] - yaw[int(1.0 / dt)])
    herr = np.degrees((hd[-1] - yaw[-1] + np.pi) % (2 * np.pi) - np.pi)
    hdg = yaw[-1]; dxy = xy[-1] - xy[e2]
    fwd = (np.cos(hdg) * dxy[:, 0] + np.sin(hdg) * dxy[:, 1]) * 100
    msg = f"  {name:5s} up {up[0]:3.0f} / {up[1]:3.0f} / {up[2]:3.0f} %  (walk / turn / walk)"
    if ok_t.any():
        msg += (f" | turned {np.median(turned[ok_t]):5.1f} deg (p10 {np.percentile(turned[ok_t], 10):5.1f}, p90 "
                f"{np.percentile(turned[ok_t], 90):5.1f}; asked {want:.0f})")
    if ok.any():
        msg += (f" | then walked {np.median(fwd[ok]):4.1f} cm in {a.T - a.t2 - 1.5:.1f} s, heading error at the end "
                f"{np.median(np.abs(herr[ok])):4.1f} deg | yaw drift in the first walk {np.median(drift[sel & ~dd[k1 - 1]]):+5.1f} deg")
    print(msg)
    yr = np.degrees(qy[k1:k2][:, sel])
    print(f"        hip yaw during the turn: p5 {np.percentile(yr, 5):+5.1f} p95 {np.percentile(yr, 95):+5.1f} deg")
