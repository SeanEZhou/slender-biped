"""Gate B torque / speed audit.

1. Quasi-static gravity torques: double and single support, crouch -> near-straight.
2. Swing-leg inverse dynamics for min-jerk swing trajectories at several step rates
   (torque AND speed, plotted against each candidate servo's torque-speed curve).
3. Stance transient scenarios: single-support static x {1.5, 2.0} (explicit assumption;
   to be replaced by measured RL-rollout torques in Gates D/E).
4. Crouch-to-stand drive through the servo model on an explicit vertical-slider test rig.

    python scripts/torque_audit.py [model_name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from slender_biped.actuator import ServoBank, ServoParams, available_torque  # noqa: E402
from slender_biped.config import load_actuators  # noqa: E402
from slender_biped.model_builder import armature  # noqa: E402
from slender_biped.statics import Statics, balance_single, stance_pose  # noqa: E402

PLOTS = ROOT / "results" / "plots"
JT = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]


def static_sweep(m, meta):
    st = Statics(m, meta["joints"])
    has_ankle = any(j.endswith("ankle") for j in meta["joints"])
    rows = []
    for kdeg in [10, 20, 32, 45, 60, 75, 90, 110]:
        k = np.radians(kdeg)
        q = stance_pose(k, has_ankle)
        st.set(q)
        tau, f, r = st.torques("LR")
        rows.append({"pose": f"double k={kdeg}", "knee_deg": kdeg, "support": "double",
                     "tau": dict(zip(meta["joints"], np.abs(tau))), "resid": r,
                     "com_minus_tips_x": float(st.com()[0] - 0.5 * (st.contact_pos("L")[0] + st.contact_pos("R")[0]))})
        qs, bal = balance_single(st, q, "L")
        tau, f, r = st.torques("L")
        rows.append({"pose": f"single k={kdeg}", "knee_deg": kdeg, "support": "single",
                     "tau": dict(zip(meta["joints"], np.abs(tau))), "resid": r, "balance_err_m": bal,
                     "stance_roll_deg": float(np.degrees(-qs["L_hip_roll"]))})
    return rows


def minjerk(t, T):
    s = np.clip(t / T, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def swing_trajectories(m, meta, step_hz_list=(1.0, 1.5, 2.0, 2.5, 3.0), stride=0.08, lift_knee=np.radians(30)):
    """Left leg swings from behind to in front (hip +/-), knee flex-extend, base held fixed.

    stride = foot travel relative to the hip during one swing (= step length).
    """
    m.dof_armature[:] = 0  # link dynamics only; reflected rotor inertia is added per servo set in evaluate()
    st = Statics(m, meta["joints"])
    has_ankle = any(j.endswith("ankle") for j in meta["joints"])
    out = {}
    k0 = np.radians(meta["cfg"]["nominal_pose_deg"]["knee"])
    leg = meta["cfg"]["thigh"]["length"] + meta["cfg"]["shin"]["length"]
    dh = np.arctan(stride / 2 / (0.9 * leg))
    for hz in step_hz_list:
        T = 1.0 / (2 * hz) * 0.85  # swing ~85% of a step period (short double support)
        dt = 0.002
        t = np.arange(0, T + dt, dt)
        h0 = k0 / 2
        hip = h0 - dh + 2 * dh * minjerk(t, T)
        knee = k0 + lift_knee * np.sin(np.pi * np.clip(t / T, 0, 1)) ** 2
        ank = np.full_like(t, k0 / 2)  # ankle simply holds (no need to keep the tip level in the air)
        traj = {"L_hip_pitch": hip, "L_knee": knee}
        if has_ankle:
            traj["L_ankle"] = ank
        vel = {k: np.gradient(v, dt) for k, v in traj.items()}
        acc = {k: np.gradient(v, dt) for k, v in vel.items()}
        rec = {k: {"tau": [], "qd": [], "qdd": acc[k].tolist()} for k in traj}
        base = stance_pose(k0, has_ankle)
        for i in range(len(t)):
            q = dict(base)
            q.update({k: v[i] for k, v in traj.items()})
            st.set(q, {k: v[i] for k, v in vel.items()}, {k: v[i] for k, v in acc.items()})
            tau, _, _ = st.torques("")
            td = dict(zip(meta["joints"], tau))
            for k in traj:
                rec[k]["tau"].append(float(td[k]))
                rec[k]["qd"].append(float(vel[k][i]))
        out[hz] = {"T_swing": T, **rec}
    return out


def rig_model(m_xml: str, z0: float):
    import re
    xml = m_xml.replace('<freejoint name="root"/>', '<joint name="rig_z" type="slide" axis="0 0 1" damping="0.5"/>')
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    return mujoco.MjModel.from_xml_string(xml)


def crouch_to_stand(meta, act, set_name, xml, seconds=2.5, knee_from=100.0):
    """Explicit TEST RIG: pelvis on a frictionless vertical slider (no pitch/roll/lateral DOF)."""
    m = rig_model(xml, meta["stand_base_z"])
    d = mujoco.MjData(m)
    specs = [act["catalog"][act["assignments"][set_name][j.split("_", 1)[1]]] for j in meta["joints"]]
    sp = ServoParams.from_config(specs, act["servo_model"], m.opt.timestep)
    bank = ServoBank(sp, m.opt.timestep)
    for i, s in enumerate(specs):
        m.actuator_ctrlrange[i] = [-s["stall_torque"], s["stall_torque"]]
        m.dof_armature[m.jnt_dofadr[m.actuator_trnid[i, 0]]] = armature(s, act["servo_model"])
    has_ankle = any(j.endswith("ankle") for j in meta["joints"])
    q_start = stance_pose(np.radians(knee_from), has_ankle)
    q_end = stance_pose(np.radians(meta["cfg"]["nominal_pose_deg"]["knee"]), has_ankle)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in meta["joints"]]
    qa = np.array([m.jnt_qposadr[i] for i in jid])
    va = np.array([m.jnt_dofadr[i] for i in jid])
    qs = np.array([q_start.get(j, 0.0) for j in meta["joints"]])
    qe = np.array([q_end.get(j, 0.0) for j in meta["joints"]])
    d.qpos[qa] = qs
    # drop the rig so the tips touch the floor
    mujoco.mj_kinematics(m, d)
    tips = [d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_contact")][2] for s in "LR"]
    d.qpos[0] = -min(tips) + 0.0005
    bank.reset(qs)
    log = {"t": [], "z": [], "tau": [], "qd": [], "duty_sat": [], "q": []}
    steps = int(seconds / m.opt.timestep)
    t_settle, t_move = 0.5, 1.2
    for k in range(steps):
        t = k * m.opt.timestep
        a = minjerk(t - t_settle, t_move)
        bank.command(qs + (qe - qs) * a)
        d.ctrl[:] = bank.torque(d.qpos[qa], d.qvel[va])
        mujoco.mj_step(m, d)
        if k % 5 == 0:
            log["t"].append(t)
            log["z"].append(float(d.qpos[0]))
            log["tau"].append(bank.last["tau"].tolist())
            log["qd"].append(d.qvel[va].tolist())
            log["duty_sat"].append(bank.last["duty_sat"].tolist())
            log["q"].append(d.qpos[qa].tolist())
    err = np.abs(np.array(log["q"][-1]) - qe)
    return {"final_track_err_deg": dict(zip(meta["joints"], np.degrees(err).round(2).tolist())),
            "max_abs_tau": dict(zip(meta["joints"], np.abs(np.array(log["tau"])).max(0).round(3).tolist())),
            "duty_sat_frac": dict(zip(meta["joints"], np.array(log["duty_sat"]).mean(0).round(3).tolist())),
            "pelvis_rise_m": float(log["z"][-1] - log["z"][0]), "log": log}


def servo_for(act, set_name, jt):
    return act["catalog"][act["assignments"][set_name][jt]]


def evaluate(static_rows, swing, act, sets, has_ankle):
    """Per actuator set & joint: peak demand / stall and / sustained cap; swing torque-speed feasibility."""
    der = act["derating_scenarios"]
    res = {}
    for sname in sets:
        r = {}
        for jt in JT:
            if jt == "ankle" and not has_ankle:
                continue
            sv = servo_for(act, sname, jt)
            key = f"L_{jt}"
            stat_double = max(row["tau"][key] for row in static_rows if row["support"] == "double" and row["knee_deg"] <= 90)
            stat_single = max(row["tau"][key] for row in static_rows if row["support"] == "single" and row["knee_deg"] <= 60)
            stat_single_nom = [row for row in static_rows if row["support"] == "single" and row["knee_deg"] == 32][0]["tau"][key]
            # swing: per step rate, fraction of the trajectory where demand exceeds the full-duty envelope
            sw = {}
            for hz, s in swing.items():
                if key not in s:
                    continue
                tau = np.array(s[key]["tau"]) + armature(sv, act["servo_model"]) * np.array(s[key]["qdd"])
                qd = np.array(s[key]["qd"])
                avail = available_torque(sv["stall_torque"], sv["noload_speed"], qd, np.sign(tau + 1e-12))
                viol = np.abs(tau) > avail
                sw[str(hz)] = {"feasible": bool(not viol.any()), "violation_frac": float(viol.mean()),
                               "peak_speed_over_noload": float(np.max(np.abs(qd)) / sv["noload_speed"]),
                               "peak_tau": float(np.max(np.abs(tau)))}
            feas = [float(h) for h, v in sw.items() if v["feasible"]]
            sust = {k: v * sv["stall_torque"] for k, v in der.items()}
            if sv.get("rated_torque"):
                sust["vendor_rated"] = sv["rated_torque"]
            r[jt] = {
                "servo": act["assignments"][sname][jt], "stall": sv["stall_torque"], "noload_speed": sv["noload_speed"],
                "static_single_stance_nominal": stat_single_nom,
                "static_single_stance_worst_k<=60": stat_single, "static_double_worst_k<=90": stat_double,
                "peak_ratio_single_x2_vs_stall": 2 * stat_single / sv["stall_torque"],
                "sustained_ratio_nominal_vs_caps": {k: stat_single_nom / v for k, v in sust.items()},
                "swing": sw,
                "max_feasible_step_rate": (max(feas) if feas and min(feas) == min(float(h) for h in sw) else 0.0) if sw else None,
            }
        res[sname] = r
    return res


def plot_static(static_rows, act, sets, path, joints):
    fig, axs = plt.subplots(1, len(joints), figsize=(3.4 * len(joints), 3.6), dpi=130)
    for ax, jt in zip(axs, joints):
        for sup, mk in (("double", "o-"), ("single", "s-")):
            rows = [r for r in static_rows if r["support"] == sup]
            ax.plot([r["knee_deg"] for r in rows], [r["tau"][f"L_{jt}"] for r in rows], mk, ms=3, label=f"{sup} support")
        colors = plt.cm.tab10.colors
        for i, s in enumerate(sets):
            sv = servo_for(act, s, jt)
            ax.axhline(sv["stall_torque"], color=colors[i + 2], ls="--", lw=1)
            ax.axhline(sv["stall_torque"] * act["derating_scenarios"]["nominal"], color=colors[i + 2], ls=":", lw=1)
            ax.text(112, sv["stall_torque"], f"{act['assignments'][s][jt]}", fontsize=6, color=colors[i + 2], ha="right", va="bottom")
        ax.set_title(jt)
        ax.set_xlabel("knee flexion [deg] (hip = ankle = knee/2)")
        ax.set_yscale("log")
        ax.set_ylim(0.005, 4)
    axs[0].set_ylabel("|static torque| [N m]  (-- stall, : 0.33 stall)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_swing(swing, act, sets, path, has_ankle):
    joints = ["hip_pitch", "knee"] + (["ankle"] if has_ankle else [])
    fig, axs = plt.subplots(1, len(joints), figsize=(4.4 * len(joints), 3.8), dpi=130)
    colors = plt.cm.tab10.colors
    for ax, jt in zip(axs, joints):
        arm = armature(servo_for(act, sets[0], jt), act["servo_model"])
        for i, (hz, s) in enumerate(swing.items()):
            tau = np.array(s[f"L_{jt}"]["tau"]) + arm * np.array(s[f"L_{jt}"]["qdd"])
            ax.plot(np.array(s[f"L_{jt}"]["qd"]), tau, lw=1.4, color=colors[i], label=f"{hz} steps/s")
        w = np.linspace(-14, 14, 200)
        for k, sname in enumerate(sets):
            sv = servo_for(act, sname, jt)
            ap = available_torque(sv["stall_torque"], sv["noload_speed"], w, 1.0)
            an = -available_torque(sv["stall_torque"], sv["noload_speed"], w, -1.0)
            ax.plot(w, ap, "--", color=colors[k + 4], lw=1, label=act["assignments"][sname][jt])
            ax.plot(w, an, "--", color=colors[k + 4], lw=1)
        ax.set_xlabel("joint speed [rad/s]")
        ax.set_ylabel("torque [N m]")
        ax.set_title(f"swing {jt}: demand (incl. {sets[0]} rotor inertia)\nvs full-duty envelopes", fontsize=9)
        ax.set_ylim(-1.6, 1.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rig(rig, meta, path, set_name):
    log = rig["log"]
    t = np.array(log["t"])
    tau = np.array(log["tau"])
    qd = np.array(log["qd"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5), dpi=130)
    for i, j in enumerate(meta["joints"][: len(meta["joints"]) // 2]):
        axs[0].plot(t, tau[:, i], label=j)
        axs[1].plot(t, qd[:, i], label=j)
    axs[2].plot(t, np.array(log["z"]) - log["z"][0])
    axs[0].set_title(f"servo torque (left leg) [{set_name}]")
    axs[1].set_title("joint speed [rad/s]")
    axs[2].set_title("pelvis rise on vertical rig [m]")
    axs[0].legend(fontsize=7)
    for a in axs:
        a.set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline_B_10dof"
    xml = (ROOT / "models" / f"{name}.xml").read_text()
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text())
    m = mujoco.MjModel.from_xml_string(xml)
    act = load_actuators()
    sets = list(act["assignments"])
    has_ankle = any(j.endswith("ankle") for j in meta["joints"])
    PLOTS.mkdir(parents=True, exist_ok=True)

    static_rows = static_sweep(m, meta)
    swing = swing_trajectories(mujoco.MjModel.from_xml_string(xml), meta)
    ev = evaluate(static_rows, swing, act, sets, has_ankle)
    rigs = {}
    for s in sets:
        r = crouch_to_stand(meta, act, s, xml)
        plot_rig(r, meta, PLOTS / f"rig_crouch_to_stand_{name}_{s}.png", s)
        r.pop("log")
        rigs[s] = r
    joints = [j for j in JT if has_ankle or j != "ankle"]
    plot_static(static_rows, act, ["baseline", "torque_feetech_12v", "xl330_only"], PLOTS / f"static_torque_{name}.png", joints)
    plot_swing(swing, act, ["baseline", "torque_feetech_12v", "slim_dynamixel", "xl330_only"], PLOTS / f"swing_torque_speed_{name}.png", has_ankle)
    out = {"model": name, "mass_kg": float(sum(m.body_mass)), "static": static_rows,
           "evaluation": ev, "rig_crouch_to_stand": rigs,
           "swing_params": {"step_length_m": 0.08, "knee_lift_deg": 30, "rates_steps_per_s": list(swing)},
           "notes": ["Static torques are exact for the rigid point-contact model; dynamic stance peaks are NOT "
                     "measured here (x1.5/x2 are assumptions). Swing torques are inverse dynamics with the pelvis "
                     "held still.",
                     "Sustained caps use derating factors of stall when vendor rated torque is unknown."]}
    (ROOT / "results" / f"torque_audit_{name}.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"== {name}  mass {out['mass_kg']*1000:.0f} g")
    for s in sets:
        print(f"-- {s}")
        for jt, e in ev[s].items():
            caps = e["sustained_ratio_nominal_vs_caps"]
            print(f"  {jt:9s} {e['servo']:18s} stance(nom)={e['static_single_stance_nominal']:.3f}Nm "
                  f"single(k<=60)={e['static_single_stance_worst_k<=60']:.3f} "
                  f"2x/stall={e['peak_ratio_single_x2_vs_stall']:.2f} "
                  f"nom/0.33stall={caps['nominal']:.2f} "
                  f"max feasible swing rate={e['max_feasible_step_rate']} steps/s")
        print(f"   rig: rise={rigs[s]['pelvis_rise_m']*1000:.1f} mm, track err={rigs[s]['final_track_err_deg']}, "
              f"duty_sat={rigs[s]['duty_sat_frac']}")


if __name__ == "__main__":
    main()
