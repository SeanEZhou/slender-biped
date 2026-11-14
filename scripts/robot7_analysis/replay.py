import sys, csv, json, math, os, numpy as np
sys.path.insert(0, ".")
os.environ.setdefault("JAX_PLATFORMS","cpu")
from pathlib import Path
from slender_biped.evaluation import NumpyPolicy
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
def load(f):
    rows = list(csv.reader(open(f))); h = rows[0]; A = np.array(rows[1:], float)
    c = lambda ks: A[:, [h.index(k) for k in ks]]
    return A[:, 0], c(["gx","gy","gz"]), c(["wx","wy","wz"]), c([f"q_{n}" for n in J]), c([f"qd_{n}" for n in J]), c([f"a_{n}" for n in J])
HG=1.0
def build(run, f, vx=0.015):
    d=Path("runs")/run; pp=sorted(d.glob("params_*M.pkl")) or [d/"params.pkl"]; pol=NumpyPolicy(pp[-1]); env=json.load(open(d/"config.json"))["env"]; dt, hz = env["ctrl_dt"], env["gait_hz"]
    q_nom = np.array(json.load(open(f"models/{env['model']}.json"))["nominal_q"])
    t, g, w, q, qd, act = load(f)
    eng = int(np.argmax(np.any(act != 0, 1)))
    gate_u = tuple(env.get("settle_gate_u", ())); gw = float(env.get("settle_gate_w", .3)); gn = int(float(env.get("settle_gate_max", .5)) / dt)
    slew = float(env.get("cmd_vx_slew", 0)); cmd = np.array([0.0 if slew else vx, 0, 0]); phase = 0.0; held = 0; yaw = 0.0
    O = []; last = np.zeros(10)
    for i in range(eng, len(t)):
        if slew: cmd[0] += float(np.clip(vx - cmd[0], -slew*dt, slew*dt))
        cmd[2] = float(np.clip(-HG * yaw, -.3, .3))
        o = np.concatenate([g[i], w[i], q[i]-q_nom, 0.1*qd[i], last, cmd.copy(), [math.sin(phase), math.cos(phase)]])
        O.append(o); last = act[i]
        ph = (phase + 2*math.pi*hz*dt) % (2*math.pi)
        u0, u1 = phase/(2*math.pi), ph/(2*math.pi)
        cross = any(((u0-gu)%1) > ((u1-gu)%1) for gu in gate_u)
        hold = cross and float(w[i,0]**2+w[i,1]**2) > gw**2 and held < gn
        held = held+1 if hold else 0
        if not hold: phase = ph
        yaw += float(-(w[i] @ g[i])) * dt
    return pol, np.array(O), act[eng:], t[eng:]
