"""Standing at the stance pose (zero action, round-7 sim) per knee / hip-pitch offset: IMU pitch, knee - hip on the
encoders and pitch - (knee - hip). The robot before engage: +4.1..+7.5 deg, knee-hip ~17.2, pitch - (knee-hip) -12."""
import math, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, eval6
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run, quat_rot_t
cfg, rd = load_run(ROOT / "runs" / "h4_t05")
for ko, ho in [(0.0, 0), (-0.025, 0), (-0.05, 0), (-0.07, 0), (-0.09, 0), (0.03, 0), (0.0, 0.03), (0.0, -0.03)]:
    over = dict(floor_tc=0.02, push_vel=0.0, dr_imu_tilt=0.0, dr_delay_steps=(0, 0), drag_back=0.0, reset_joint_noise=0.0)
    over.update(eval6.SIMS["r7"]); over["hidden_off"] = (("knee", ko), ("hip_pitch", ho))
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=1)
    obs = env.reset((0.0, 0, 0))
    for i in range(100):
        obs, _ = env.step(np.zeros(env.nj))
        if env.fallen(): break
    g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0])); p = math.degrees(math.atan2(-g[0], -g[2]))
    q = np.degrees(env.d.qpos[7:7 + env.nj] + env.flex + env.q_off); kh = q[[3, 8]].mean() - q[[2, 7]].mean()
    print(f"knee {ko:+.3f} hip {ho:+.3f}: pitch {p:+5.1f} knee-hip {kh:5.1f} pitch-(knee-hip) {p-kh:+5.1f} fallen {env.fallen()}")
