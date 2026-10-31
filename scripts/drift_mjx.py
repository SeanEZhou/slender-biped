"""Sideways drift measured in the MJX training simulator (sim-to-sim check for scripts/drift_eval.py).

    .venv/bin/python scripts/drift_mjx.py <run> --params params_XXXXM.pkl [--n 32] [--seconds 20]

Same command as drift_eval (0.15 m/s, heading held by the env's heading loop), training randomization on, no command
resampling. Reports lateral offset when passing 2 m forward and the on-line share (|offset| <= 0.2 m).
"""
import argparse
import dataclasses
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402

from slender_biped.mjx_env import EnvConfig, SlenderBipedEnv, randomization_fn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--params", default="params.pkl")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--vx", type=float, default=0.15)
a = ap.parse_args()
rd = ROOT / "runs" / a.run
cfg = json.loads((rd / "config.json").read_text())["env"]
fields = {f.name for f in dataclasses.fields(EnvConfig)}
cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in fields}
cfg["rw"] = {**EnvConfig().rw, **cfg["rw"]}
ec = dataclasses.replace(EnvConfig(**cfg), cmd_zero_prob=0.0, cmd_vx=(a.vx, a.vx), cmd_vy=(0.0, 0.0), cmd_wz=(0.0, 0.0),
                         cmd_resample_s=(1e3, 1e3), mirror_aug=False)
env = SlenderBipedEnv(ec)
norm, pol, _ = pickle.load(open(rd / a.params, "rb"))
mean, std = norm.mean["state"], norm.std["state"]
layers = [pol["params"][f"hidden_{i}"] for i in range(len(pol["params"]))]


def policy(o):
    x = (o - mean) / std
    for i, l in enumerate(layers):
        x = x @ l["kernel"] + l["bias"]
        if i < len(layers) - 1:
            x = jax.nn.swish(x)
    return jp.tanh(x[..., : x.shape[-1] // 2])


sys_v, in_axes = randomization_fn(env.sys, jax.random.PRNGKey(7), a.n) if cfg.get("dr") else (None, None)


def make(fn):
    if sys_v is None:
        return jax.vmap(fn)
    def wrapped(sysv, *args):
        env.sys = sysv
        return fn(*args)
    return jax.vmap(wrapped, in_axes=(in_axes, 0) + (0,) * 0)


if sys_v is not None:
    def r1(sysv, k):
        old, env.sys = env.sys, sysv
        s = env.reset(k); env.sys = old; return s
    def s1(sysv, s):
        old, env.sys = env.sys, sysv
        s = env.step(s, policy(s.obs["state"])); env.sys = old; return s
    reset = jax.jit(jax.vmap(r1, in_axes=(in_axes, 0)))
    step = jax.jit(jax.vmap(s1, in_axes=(in_axes, 0)))
    st = reset(sys_v, jax.random.split(jax.random.PRNGKey(0), a.n))
    stepf = lambda s: step(sys_v, s)  # noqa: E731
else:
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(lambda s: env.step(s, policy(s.obs["state"]))))
    st = reset(jax.random.split(jax.random.PRNGKey(0), a.n))
    stepf = step
q = np.array(st.pipeline_state.qpos)
p0, yaw0 = q[:, :2], np.arctan2(2 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]), 1 - 2 * (q[:, 5] ** 2 + q[:, 6] ** 2))
fwd, left = np.stack([np.cos(yaw0), np.sin(yaw0)], 1), np.stack([-np.sin(yaw0), np.cos(yaw0)], 1)
lat2, dead = np.full(a.n, np.nan), np.zeros(a.n, bool)
for k in range(int(a.seconds / ec.ctrl_dt)):
    st = stepf(st)
    dead |= np.array(st.done) > 0
    d = np.array(st.pipeline_state.qpos[:, :2]) - p0
    f, l = (d * fwd).sum(1), (d * left).sum(1)
    new = np.isnan(lat2) & (f >= 2.0) & ~dead
    lat2[new] = l[new]
ok = ~np.isnan(lat2)
print(f"[MJX] {a.run} {a.params}: fell {dead.mean():.0%} reach2m {ok.mean():.0%} on-line@2m {(np.abs(lat2[ok]) <= 0.2).sum() / a.n:.0%} "
      f"|lat|@2m med {np.median(np.abs(lat2[ok])) if ok.any() else float('nan'):.2f}  lat@2m mean {np.nanmean(lat2):+.2f}")
