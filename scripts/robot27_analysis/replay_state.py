"""Round 27: put the GPU 'sim robot' into a logged robot state and let the policy carry on.

From the log at time --t: the joint angles and speeds, the body tilt (IMU gravity) and spin (gyro), the clock phase,
the last two actions and the last servo targets. The body's linear velocity isn't logged: it is estimated from the FK,
assuming the lower foot stands still. Yaw is 0, and the lower foot is put on the floor. --n randomized sim robots
(gait_eval's robot settings) start from the same state.

Reports, every 0.1 s: the body roll (deg, + = right; median and p10..p90 over the walkers) against the robot's log, and
how many walkers fell (tilt > 30 deg), plus the first right / left lift-off in the sim against the log.

    scripts/gpu_python.sh scripts/robot27_analysis/replay_state.py <log.csv> --t 6.88 [--run z7_anc2:params_0012M.pkl]
        [--gate] [--set k=json ...]
"""
import argparse, functools, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "robot18_analysis")); sys.path.insert(0, str(ROOT / "scripts" / "robot25_analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "robot27_analysis"))
import jax, jax.numpy as jp, numpy as np
from mujoco import mjx
from brax.envs.wrappers import training as bw
from push_eval import ROBOT, load_policy
from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn
from slender_biped.runcfg import env_config
import limp
from limp import feet, s

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("--t", type=float, nargs="+", required=True)
ap.add_argument("--run", default=None); ap.add_argument("--n", type=int, default=32); ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--gate", action="store_true", help="the settle gate on (as trained); default off, as on the robot")
ap.add_argument("--vx", type=float, default=0.02); ap.add_argument("--v0", action="store_true", help="base linear velocity 0"); ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
run = a.run or Path(a.log).name[6:-20]
W, p, d = load_policy(run)
over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=0.7,
            cmd_resample_s=(1000.0, 1000.0), reset_hard_prob=0.0, reset_joint_noise=0.0, reset_tilt=0.0,
            cmd_vx=(a.vx, a.vx), cmd_zero_prob=0.0, cmd_wz=(0.0, 0.0), sup=False, cmd_vx_slew=0.0)
over.update(ROBOT)
if not a.gate:
    over["settle_gate_u"] = ()
for kv in a.set:
    k, v = kv.split("=", 1); x = json.loads(v)
    over[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
cfg = env_config(d, **over); env = SlenderBipedEnv(cfg); N = a.n; nj = env.nj; dt = cfg.ctrl_dt
J = list(env.joints); assert J == list(s.J), (J, s.J)

h, A = limp.load(a.log); col = lambda k: A[:, h.index(k)]
tl = col("t"); G = A[:, [h.index(k) for k in ("gx", "gy", "gz")]]
Q = A[:, [h.index("q_" + j) for j in J]]; QD = A[:, [h.index("qd_" + j) for j in J]]
ACT = A[:, [h.index("a_" + j) for j in J]]; CMD = A[:, [h.index("cmd_" + j) for j in J]]
WB = A[:, [h.index(k) for k in ("wx", "wy", "wz")]]; PH = col("phase")
roll_log = np.degrees(np.arctan2(-G[:, 1], -G[:, 2]))
tab, zz, ev = s.table(run.split(":")[0]); stn = np.array(zz["stance"][-1]); nst_ = len(stn)
sched = lambda ph: stn[int((ph / 2 / np.pi % 1) * nst_) % nst_]

def quat_from_g(g):
    """Body orientation (yaw 0) whose world-down vector in the body frame is g."""
    g = g / np.linalg.norm(g)
    roll = np.arctan2(-g[1], -g[2]); pitch = np.arctan2(g[0], np.hypot(g[1], g[2]))
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])

def build(i):
    """The logged state at tick i: qpos (z set later), qvel, info overrides."""
    F = lambda k: np.array(feet(Q[k], G[k])) - np.array([0, 0, 0.3])          # feet rel. base, yaw-free world frame
    Fi = F(i); low = int(np.argmin(Fi[:, 2]))
    dF = (F(min(i + 2, len(tl) - 1))[low] - F(max(i - 2, 0))[low]) / (tl[min(i + 2, len(tl) - 1)] - tl[max(i - 2, 0)])
    v_base = 0 * dF if a.v0 else -dF                                             # the lower foot stands still
    return dict(quat=quat_from_g(G[i]), q=Q[i], qd=QD[i], w=WB[i], v=v_base, phase=PH[i], g=G[i] / np.linalg.norm(G[i]),
                last=ACT[i - 1], last2=ACT[i - 2], tgt=CMD[i - 1], low="LR"[low])

wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(randomization_fn, rng=jax.random.split(jax.random.PRNGKey(1), N)))
reset = jax.jit(wenv.reset)
nst = int(a.T / dt)

def place(st, S):
    info = st.info
    q_off = info["q_off"]                                    # (N, nj): the encoders read qpos + q_off
    qj = jp.asarray(S["q"])[None] - q_off
    qpos = st.pipeline_state.qpos.at[:, 0:2].set(0.0).at[:, 3:7].set(jp.asarray(S["quat"])[None]).at[:, 7:7 + nj].set(qj)
    qvel = jp.zeros_like(st.pipeline_state.qvel).at[:, 0:3].set(jp.asarray(S["v"])[None]).at[:, 3:6].set(jp.asarray(S["w"])[None])
    qvel = qvel.at[:, 6:6 + nj].set(jp.asarray(S["qd"])[None])
    def one(data, qp, qv):
        dd = mjx.kinematics(env.sys, data.replace(qpos=qp, qvel=qv))
        lo, _ = env._tip_low(dd)
        qp = qp.at[2].add(-jp.min(lo) + 0.0005)
        return mjx.forward(env.sys, dd.replace(qpos=qp))
    data = jax.vmap(one)(st.pipeline_state, qpos, qvel)
    tgt = jp.broadcast_to(jp.asarray(S["tgt"]), (N, nj))
    upd = dict(phase=jp.full((N,), S["phase"]), walking=jp.ones(N, bool),
               cmd=jp.broadcast_to(jp.array([a.vx, 0.0, 0.0]), (N, 3)), cmd_goal=jp.broadcast_to(jp.array([a.vx, 0.0, 0.0]), (N, 3)),
               heading=jp.zeros(N), last_act=jp.broadcast_to(jp.asarray(S["last"]), (N, nj)),
               last_act2=jp.broadcast_to(jp.asarray(S["last2"]), (N, nj)), prev_cmd=tgt, q_tgt=tgt,
               cmd_buf=jp.broadcast_to(tgt[:, None, :], info["cmd_buf"].shape), g_est=jp.broadcast_to(jp.asarray(S["g"]), (N, 3)),
               flex=jp.zeros((N, nj)), flex_v=jp.zeros((N, nj)), w_f=jp.broadcast_to(jp.asarray(S["w"]), (N, 3)),
               wp_f=jp.full((N,), S["w"][1]), v_prev=data.qvel[:, 0:3], anc_st=jp.ones((N, 2), bool))
    info = {**info, **{k: v for k, v in upd.items() if k in info}}
    obs = jax.vmap(lambda dd, inf, la: env._obs(dd, inf, la, jp.zeros(nj), inf["rng"])[0])(data, info, info["last_act"])
    return st.replace(pipeline_state=data, obs=obs, info=info)

def body(c, k):
    st, dead = c
    st2 = wenv.step(st, W(st.obs["state"])); dead = dead | (st2.done > 0); ps = st2.pipeline_state
    return (st2, dead), (ps.qpos[:, 3:7], st2.info["phase"], dead, ps.qpos[:, 7:7 + nj])
roll_out = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(nst)))

print(f"replay {Path(a.log).name} policy {run} ({p.name}) gate {'on' if a.gate else 'off'} n {N} {' '.join(a.set)}")
for t0 in a.t:
    i = int(np.argmin(np.abs(tl - t0))); S = build(i)
    st = place(reset(jax.random.split(jax.random.PRNGKey(0), N)), S)
    (stf, dead), (quat, ph, dd, qj) = roll_out((st, jp.zeros(N, bool)))
    quat, ph, dd = map(np.asarray, (quat, ph, dd))
    w, x, y, z = [quat[..., m] for m in range(4)]
    g = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    roll = np.degrees(np.arctan2(-g[..., 1], -g[..., 2])); tilt = np.degrees(np.arccos(np.clip(-g[..., 2], -1, 1)))
    fell = (tilt > 30).any(0) | dd[-1]
    side = np.where(fell, np.sign(roll[np.argmax(tilt > 30, 0), np.arange(N)]), 0)
    u0 = S["phase"] / 2 / np.pi % 1
    print(f"\n t {tl[i]:.2f} s  u {u0:.2f}  roll {roll_log[i]:+.1f}  gyro x {np.degrees(S['w'][0]):+.0f} deg/s  "
          f"v_base est {S['v'][0]*1000:+.0f},{S['v'][1]*1000:+.0f},{S['v'][2]*1000:+.0f} mm/s (lower foot {S['low']})")
    print(f"   fell {int(fell.sum())}/{N} (left {int((side < 0).sum())}, right {int((side > 0).sum())})")
    # lift-offs in the log and in the sim (scheduled ones; the sim's clock is the policy's own when the gate holds)
    def lifts(phs):
        out = []
        for k in range(1, len(phs)):
            a0, a1 = sched(phs[k - 1]), sched(phs[k])
            for sd in (0, 1):
                if a0[sd] >= .5 and a1[sd] < .5: out.append(f"{'LR'[sd]}@{k * dt:.2f}")
        return out
    ll = lifts(PH[i:i + nst]); sl = lifts(ph[:, 0])
    print(f"   scheduled lift-offs after t: robot {' '.join(ll[:3])} | sim walker 0 {' '.join(sl[:3])}")
    print("   time   robot roll | sim roll p50 (p10..p90)  up")
    for k in range(0, nst, 5):
        j = i + 1 + k; rl = f"{roll_log[j]:+6.1f}" if j < len(tl) else "   end"
        up = ~(tilt[:k + 1] > 30).any(0)
        print(f"   +{(k + 1) * dt:4.2f}  {rl}     | {np.median(roll[k]):+6.1f} ({np.percentile(roll[k], 10):+6.1f}..{np.percentile(roll[k], 90):+6.1f})  {int(up.sum()):2d}")
