"""Smoke test: jit the MJX env (with DR), step zero actions, print rates and metrics."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

from slender_biped.mjx_env import EnvConfig, SlenderBipedEnv, randomization_fn  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "baseline_A_8dof"
env = SlenderBipedEnv(EnvConfig(model=model, dr=True))
N = 1024
rng = jax.random.split(jax.random.PRNGKey(0), N)
sys_v, in_axes = randomization_fn(env.sys, rng)


def reset(sx, k):
    env.sys = sx
    return env.reset(k)


def step(sx, s, a):
    env.sys = sx
    return env.step(s, a)


r = jax.jit(jax.vmap(reset, in_axes=(in_axes, 0)))
st = jax.jit(jax.vmap(step, in_axes=(in_axes, 0, 0)))
t = time.time(); s = r(sys_v, rng); jax.block_until_ready(s.obs); print("reset compile", round(time.time() - t, 1))
a = jp.zeros((N, env.action_size))
t = time.time(); s = st(sys_v, s, a); jax.block_until_ready(s.obs); print("step compile", round(time.time() - t, 1))
t = time.time()
dones = 0
for i in range(100):
    s = st(sys_v, s, a)
    dones += float(s.done.mean())
jax.block_until_ready(s.obs)
print("steps/s", round(100 * N / (time.time() - t)), "obs", s.obs["state"].shape, s.obs["privileged_state"].shape)
print("mean done-rate/step", dones / 100, {k: round(float(v.mean()), 3) for k, v in s.metrics.items()})
