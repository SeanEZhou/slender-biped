"""Gate C sanity clips on the FREE base (no rig, no support).

modes:
  limp   - servos commanded to hold their current reading but with duty forced to 0
           (back-EMF braking only): how does the structure collapse?
  hold   - naive servo position hold of the stand pose (no balance controller)
  random - random joint targets around the stand pose (smoothed), 2 Hz

    python scripts/sanity_sim.py baseline_B_10dof hold
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slender_biped.render import Renderer, stand_data, write_video  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from slender_biped.actuator import ServoBank, ServoParams  # noqa: E402
from slender_biped.config import load_actuators  # noqa: E402


def run(name, mode, seconds=4.0, seed=0, video=True):
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"{name}.xml"))
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text())
    act = load_actuators()
    specs = [act["catalog"][act["assignments"][meta["cfg"]["actuator_set"]][j.split("_", 1)[1]]] for j in meta["joints"]]
    bank = ServoBank(ServoParams.from_config(specs, act["servo_model"], m.opt.timestep), m.opt.timestep)
    d = stand_data(m)
    q0 = np.array(meta["nominal_q"])
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in meta["joints"]]
    qa = np.array([m.jnt_qposadr[i] for i in jid])
    va = np.array([m.jnt_dofadr[i] for i in jid])
    bank.reset(q0)
    rng = np.random.default_rng(seed)
    r = Renderer(m, 640, 480) if video else None
    if r:
        r.set_groups(shells=True)
    frames, log = [], {"t": [], "z": [], "tilt_deg": [], "tau_max": []}
    tgt, tgt_next = q0.copy(), q0.copy()
    fall_t = None
    for k in range(int(seconds / m.opt.timestep)):
        t = k * m.opt.timestep
        if mode == "random" and k % 500 == 0:
            tgt, tgt_next = tgt_next, q0 + rng.normal(0, 0.15, len(q0))
        cmd = q0 if mode != "random" else tgt + (tgt_next - tgt) * ((k % 500) / 500)
        bank.command(cmd)
        tau = bank.torque(d.qpos[qa], d.qvel[va])
        if mode == "limp":
            tau = np.clip(-bank.ts * d.qvel[va] / bank.w0, -bank.ts, bank.ts)  # duty 0: back-EMF braking only
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        up = d.xmat[1].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        if fall_t is None and (d.qpos[2] < 0.6 * meta["stand_base_z"] or tilt > 45):
            fall_t = t
        if k % 20 == 0:
            log["t"].append(t)
            log["z"].append(float(d.qpos[2]))
            log["tilt_deg"].append(tilt)
            log["tau_max"].append(float(np.max(np.abs(tau) / bank.ts)))
        if r and k % 33 == 0:
            frames.append(r.render(d, "track"))
    if r:
        out = ROOT / "assets" / "videos"
        out.mkdir(parents=True, exist_ok=True)
        write_video(out / f"sanity_{name}_{mode}.mp4", frames, fps=30)
        r.close()
    return {"model": name, "mode": mode, "fall_time_s": fall_t, "final_z": log["z"][-1],
            "max_tilt_deg": max(log["tilt_deg"]), "peak_tau_over_stall": max(log["tau_max"])}


if __name__ == "__main__":
    names = [sys.argv[1]] if len(sys.argv) > 1 else ["baseline_A_10dof", "baseline_B_10dof", "baseline_A_8dof", "baseline_B_8dof"]
    modes = [sys.argv[2]] if len(sys.argv) > 2 else ["limp", "hold", "random"]
    res = []
    for n in names:
        for mo in modes:
            r = run(n, mo, video=n.endswith("10dof"))
            res.append(r)
            print(r)
    (ROOT / "results" / "sanity_sim.json").write_text(json.dumps(res, indent=1))
