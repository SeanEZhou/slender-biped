"""Round 28: robot log states in the GPU 'sim robot'.

A log tick gives the joint angles and speeds, the IMU gravity estimate and gyro, the clock phase, the policy's last
two (low-passed) actions and the last servo targets. The body's linear velocity is not logged: `fit_velocity` picks it
by shooting, from a grid of candidates the one whose first 0.2 s in the (nominal) sim best follows the logged gravity
(roll and pitch). Yaw is 0, and the lower foot is put on the floor.
"""
import functools, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p_ in (ROOT, ROOT / "scripts", ROOT / "scripts" / "robot18_analysis", ROOT / "scripts" / "robot25_analysis"):
    sys.path.insert(0, str(p_))
import csv, json
import jax, jax.numpy as jp, numpy as np
from mujoco import mjx
from brax.envs.wrappers import training as bw
from push_eval import ROBOT, load_policy
from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn
from slender_biped.runcfg import env_config

def load_log(f):
    r = list(csv.reader(open(f))); h = r[0]
    A = np.array([[float(v) if v not in ('True', 'False') else float(v == 'True') for v in x] for x in r[1:] if len(x) == len(h)])
    return h, A

def which_params(run, f, env):
    """The checkpoint that ran in log f: the one whose actions, from the logged inputs, match the logged actions."""
    d = ROOT / "runs" / run; pk = sorted(d.glob("params_*M.pkl")) or [d / "params.pkl"]
    L = Log(f, env)
    idx = np.where(np.abs(np.diff(L.ph)) > 1e-6)[0]; idx = idx[(idx > 2)][20:400:5]
    if len(idx) < 5:
        return pk[-1].name, np.nan
    best = None
    for p in pk:
        W, _, _ = load_policy(f"{run}:{p.name}")
        e = float(np.median(np.abs(np.clip(np.asarray(W(L.obs(idx))), -1, 1) - L.raw(idx))))
        if best is None or e < best[1]:
            best = (p.name, e)
    return best

class Log:
    def __init__(self, f, env):
        self.f = f; h, A = load_log(f); self.h, self.A = h, A
        c = lambda k: A[:, h.index(k)]
        J = list(env.joints); self.J = J
        self.t = c('t'); self.G = A[:, [h.index(k) for k in ('gx', 'gy', 'gz')]]
        self.W = A[:, [h.index(k) for k in ('wx', 'wy', 'wz')]]
        self.Q = A[:, [h.index('q_' + j) for j in J]]; self.QD = A[:, [h.index('qd_' + j) for j in J]]
        self.ACT = A[:, [h.index('a_' + j) for j in J]]; self.CMD = A[:, [h.index('cmd_' + j) for j in J]]
        self.ph = c('phase'); self.vx = c('vx_cmd') if 'vx_cmd' in h else None
        self.walking = c('walking') if 'walking' in h else None
        self.q_nom = np.asarray(env.q_nom); self.lpf = 0.7
        self.roll = np.degrees(np.arctan2(-self.G[:, 1], -self.G[:, 2])); self.pitch = np.degrees(np.arctan2(self.G[:, 0], -self.G[:, 2]))
    def vcmd(self, i, default=0.02):
        if self.vx is not None:
            return float(self.vx[i])
        return default if np.abs(self.ph[i] - self.ph[max(i - 1, 0)]) > 1e-6 else 0.0
    def obs(self, idx):
        vx = np.array([self.vcmd(i) for i in idx])
        return np.concatenate([self.G[idx], self.W[idx], self.Q[idx] - self.q_nom, 0.1 * self.QD[idx], self.ACT[idx - 1],
                               np.stack([vx, 0 * vx, 0 * vx], 1), np.stack([np.sin(self.ph[idx]), np.cos(self.ph[idx])], 1)], 1)
    def raw(self, idx):
        return (self.ACT[idx] - self.lpf * self.ACT[idx - 1]) / (1 - self.lpf)
    def state(self, i):
        g = self.G[i] / np.linalg.norm(self.G[i])
        return dict(quat=quat_from_g(g), q=self.Q[i], qd=self.QD[i], w=self.W[i], phase=float(self.ph[i]), g=g,
                    last=self.ACT[i - 1], last2=self.ACT[i - 2], tgt=self.CMD[i - 1], vx=self.vcmd(i))

def quat_from_g(g):
    g = g / np.linalg.norm(g)
    roll = np.arctan2(-g[1], -g[2]); pitch = np.arctan2(g[0], np.hypot(g[1], g[2]))
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])

def robot_cfg(d, gate=False, **kw):
    over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=0.7,
                cmd_resample_s=(1000.0, 1000.0), reset_hard_prob=0.0, reset_joint_noise=0.0, reset_tilt=0.0,
                cmd_vx=(0.02, 0.02), cmd_zero_prob=0.0, cmd_wz=(0.0, 0.0), sup=False, cmd_vx_slew=0.0)
    over.update(ROBOT)
    if not gate:
        over["settle_gate_u"] = ()
    over.update(kw)
    return env_config(d, **over)

class Sim:
    """Batched sim robots started from log states. `dr`: domain-randomized walkers (else nominal)."""
    def __init__(self, env, W, N, T, dr=True, seed=1):
        self.env, self.W, self.N, self.nst = env, W, N, int(round(T / env.cfg.ctrl_dt))
        if dr:
            self.wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(randomization_fn, rng=jax.random.split(jax.random.PRNGKey(seed), N)))
            self._reset, self._step = self.wenv.reset, self.wenv.step
        else:
            self._reset, self._step = jax.vmap(env.reset), jax.vmap(env.step)
        self.reset = jax.jit(self._reset)
        nj = env.nj
        def body(c, k):
            st, dead = c
            st2 = self._step(st, W(st.obs["state"])); dead = dead | (st2.done > 0)
            return (st2, dead), (st2.pipeline_state.qpos[:, 3:7], st2.info["g_est"], st2.info["phase"], dead)
        self.roll_out = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(self.nst)))
        self.place = jax.jit(self._place)

    def _place(self, st, S, V):
        """S: dict of log-state arrays (broadcast), V: (N, 3) base linear velocity (world)."""
        env, N, nj = self.env, self.N, self.env.nj
        info = st.info
        qj = S["q"][None] - info["q_off"]
        qpos = st.pipeline_state.qpos.at[:, 0:2].set(0.0).at[:, 3:7].set(S["quat"][None]).at[:, 7:7 + nj].set(qj)
        qvel = jp.zeros_like(st.pipeline_state.qvel).at[:, 0:3].set(V).at[:, 3:6].set(S["w"][None]).at[:, 6:6 + nj].set(S["qd"][None])
        def one(data, qp, qv):
            dd = mjx.kinematics(env.sys, data.replace(qpos=qp, qvel=qv))
            lo, _ = env._tip_low(dd)
            qp = qp.at[2].add(-jp.min(lo) + 0.0005)
            return mjx.forward(env.sys, dd.replace(qpos=qp))
        data = jax.vmap(one)(st.pipeline_state, qpos, qvel)
        tgt = jp.broadcast_to(S["tgt"], (N, nj)); cmd = jp.broadcast_to(jp.stack([S["vx"], 0.0, 0.0]), (N, 3))
        upd = dict(phase=jp.full((N,), S["phase"]), walking=jp.full((N,), S["vx"] > 1e-3), cmd=cmd, cmd_goal=cmd,
                   heading=jp.zeros(N), last_act=jp.broadcast_to(S["last"], (N, nj)), last_act2=jp.broadcast_to(S["last2"], (N, nj)),
                   prev_cmd=tgt, q_tgt=tgt, cmd_buf=jp.broadcast_to(tgt[:, None, :], info["cmd_buf"].shape),
                   g_est=jp.broadcast_to(S["g"], (N, 3)), flex=jp.zeros((N, nj)), flex_v=jp.zeros((N, nj)),
                   w_f=jp.broadcast_to(S["w"], (N, 3)), wp_f=jp.full((N,), S["w"][1]), v_prev=data.qvel[:, 0:3],
                   anc_st=jp.ones((N, 2), bool))
        info = {**info, **{k: v for k, v in upd.items() if k in info}}
        obs = jax.vmap(lambda dd, inf: env._obs(dd, inf, inf["last_act"], jp.zeros(nj), inf["rng"])[0])(data, info)
        return st.replace(pipeline_state=data, obs=obs, info=info)

    def run(self, S, V, seed=0):
        S = {k: jp.asarray(v) for k, v in S.items()}
        st = self.place(self.reset(jax.random.split(jax.random.PRNGKey(seed), self.N)), S, jp.asarray(V))
        (_, dead), (quat, gest, ph, dd) = self.roll_out((st, jp.zeros(self.N, bool)))
        quat, gest, ph, dd = map(np.asarray, (quat, gest, ph, dd))
        w, x, y, z = [quat[..., m] for m in range(4)]
        g = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
        return dict(roll=np.degrees(np.arctan2(-g[..., 1], -g[..., 2])), pitch=np.degrees(np.arctan2(g[..., 0], -g[..., 2])),
                    tilt=np.degrees(np.arccos(np.clip(-g[..., 2], -1, 1))), groll=np.degrees(np.arctan2(-gest[..., 1], -gest[..., 2])),
                    gpitch=np.degrees(np.arctan2(gest[..., 0], -gest[..., 2])), phase=ph, dead=dd)

GRID = np.array([(vx, vy) for vx in np.linspace(-0.25, 0.25, 11) for vy in np.linspace(-0.3, 0.3, 13)])

def fit_velocity(fit_sim, L, i, K=10):
    """Shooting: the base velocity (vx, vy; vz 0) whose first K ticks follow the logged gravity best."""
    S = L.state(i); n = fit_sim.N
    V = np.zeros((n, 3)); V[:len(GRID), :2] = GRID[:n]
    R = fit_sim.run(S, V)
    k = min(K, fit_sim.nst, len(L.t) - i - 2)
    lr, lp = L.roll[i + 1:i + 1 + k], L.pitch[i + 1:i + 1 + k]
    err = np.mean((R["groll"][:k, :len(GRID)] - lr[:, None]) ** 2 + (R["gpitch"][:k, :len(GRID)] - lp[:, None]) ** 2, 0)
    b = int(np.argmin(err))
    return np.array([GRID[b][0], GRID[b][1], 0.0]), float(np.sqrt(err[b])), float(np.sqrt(err[np.argmin(np.abs(GRID).sum(1))]))
