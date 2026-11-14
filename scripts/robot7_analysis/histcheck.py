import os, sys, numpy as np
os.environ["JAX_PLATFORMS"]="cpu"; sys.path.insert(0,".")
from pathlib import Path
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run
cfg, rd = load_run(Path("runs/c0_hist")); pol=NumpyPolicy(rd/"params.pkl")
env=CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], obs_noise=0.0, dr=False), seed=3)
obs=env.reset((0.015,0,0)); nh=cfg["env"]["obs_hist"]; hist=None; last_a=np.zeros(10); err=0
for k in range(150):
    # runner-style observation from the "sensor" values of the simulator's own frame
    fr=env._frame.copy(); frame=np.concatenate([fr[:26], last_a])
    if hist is None: hist=np.tile(frame,(nh,1))
    o=np.concatenate([hist.ravel(), frame, env.cmd, [np.sin(env.phase), np.cos(env.phase)]])
    hist=np.vstack([hist[1:], frame])
    err=max(err, np.abs(o-obs).max())
    a=np.clip(pol(obs),-1,1); obs,_=env.step(a); last_a=a
print("max |runner obs - sim obs| over 150 steps:", err, "size", len(o))
