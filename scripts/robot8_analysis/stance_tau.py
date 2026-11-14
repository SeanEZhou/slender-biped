"""Standing at the stance pose (round-7 sim, knee offset -0.04): servo torques on the hip pitches and knees per forward
shift of the pelvis centre of mass, against the robot's servo load before engage (round-7 evening logs:
hip pitch L/R -16..-48 / 0..+56, knee -80..-112 / +40..+104, units of the servo's load reading)."""
import math, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, eval6
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run, quat_rot_t
cfg, rd = load_run(ROOT / "runs" / "h4_t05")
for dx in (-0.01, 0.0, 0.01, 0.02, 0.03):
    over = dict(floor_tc=0.02, push_vel=0.0, dr_imu_tilt=0.0, dr_delay_steps=(0, 0), drag_back=0.0, reset_joint_noise=0.0)
    over.update(eval6.SIMS["r7"]); over["hidden_off"] = (("knee", -0.04),); over["com_dx"] = dx
    T, P = [], []
    for seed in range(6):
        env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=seed)
        obs = env.reset((0.0, 0, 0))
        Ti = []
        for i in range(100):
            obs, lg = env.step(np.zeros(env.nj))
            if i > 50: Ti.append(env.d.qfrc_actuator[6:6 + env.nj].copy())
        if env.fallen(): continue                      # (at the back tip point some draws fall)
        T += Ti; g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0])); P.append(math.degrees(math.atan2(-g[0], -g[2])))
    p = np.mean(P); t = np.mean(T, 0)
    print(f"com {100*dx:+4.1f} cm ({len(P)}/6 standing): pitch {p:+5.1f}  tau hipP L/R {t[2]:+.3f}/{t[7]:+.3f}  knee L/R {t[3]:+.3f}/{t[8]:+.3f} N*m  hip/knee {t[2]/t[3]:+.2f}")
