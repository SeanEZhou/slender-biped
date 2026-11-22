"""Round 28: render the fall replays, to see where each sim start sits against the robot's own way into the fall.

Per fall, one video on one time axis (seconds from the fall onset), slowed down:
- column 1 'robot (log)': the robot's logged pose, replayed kinematically (logged joint angles, the IMU tilt, yaw 0,
  the lower foot on the floor). Only the pose is logged, so it doesn't move across the floor.
- the other columns: one of --n randomized sim walkers (one with the majority outcome) (the logged policy and gate) started from the log state at
  onset - dt for each dt in --deltas. Before its start a panel shows the log pose, greyed, with 'starts at ...'.
Rows: front view (roll) and side view (pitch). Plus a PNG per fall: the robot's roll and pitch against the sim walkers'
(walker 0 and the median of all) from each start.

    scripts/gpu_python.sh scripts/robot28/render_falls.py <log file> [...] [--deltas 1.6 0.8 0.4 0] [--out DIR]
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import robostate as rs
import jax, jax.numpy as jp
import mujoco
from PIL import Image, ImageDraw
from push_eval import load_policy
from slender_biped.mjx_env import SlenderBipedEnv

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("--deltas", type=float, nargs="+", default=[1.6, 0.8, 0.4, 0.0, -0.3])
ap.add_argument("--n", type=int, default=32); ap.add_argument("--walker", type=int, default=0)
ap.add_argument("--after", type=float, default=1.2, help="seconds shown after the onset (capped by the log end + 0.6)")
ap.add_argument("--slow", type=float, default=0.5); ap.add_argument("--out", required=True)
ap.add_argument("--W", type=int, default=300); ap.add_argument("--H", type=int, default=340)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
falls = {r["file"]: r for r in json.load(open(rs.ROOT / "results/robot28/falls.json"))}
D = rs.ROOT / "hardware/leg_p0/software"

def roll_pitch(quat):
    w, x, y, z = [quat[..., m] for m in range(4)]
    g = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return np.degrees(np.arctan2(-g[..., 1], -g[..., 2])), np.degrees(np.arctan2(g[..., 0], -g[..., 2]))

for name in a.files:
    name = Path(name).name
    r = falls[name]; run = r["run"]; f = str(D / name)
    _, _, d = load_policy(run)
    env_log = SlenderBipedEnv(rs.robot_cfg(d))
    pk, perr = rs.which_params(run, f, env_log); pname = f"{run}:{pk}"
    L = rs.Log(f, env_log)
    gate = bool("gate_held" in L.h and L.A[:, L.h.index("gate_held")].max() > 0)
    W, _, dp = load_policy(pname)
    env = SlenderBipedEnv(rs.robot_cfg(dp, gate=gate)); nj = env.nj
    t_on = r["t_fall"]; t_a = t_on - max(a.deltas) - 0.3; t_b = min(t_on + a.after, L.t[-1] + 0.6)
    T_sim = t_b - (t_on - max(a.deltas)) + 0.05
    T_sim = max(T_sim, 2.0)
    sim = rs.Sim(env, W, a.n, T_sim, dr=True); fsim = rs.Sim(env, W, len(rs.GRID), 0.2, dr=False)

    def body(c, k):
        st, dead = c
        st2 = sim._step(st, W(st.obs["state"])); dead = dead | (st2.done > 0)
        return (st2, dead), (st2.pipeline_state.qpos, dead)
    roll_q = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(sim.nst)))

    runs = []
    for dt in a.deltas:
        i = int(np.argmin(np.abs(L.t - (t_on - dt))))
        V, e_fit, _ = rs.fit_velocity(fsim, L, i)
        st = sim.place(sim.reset(jax.random.split(jax.random.PRNGKey(0), a.n)), {k: jp.asarray(v) for k, v in L.state(i).items()},
                       jp.asarray(np.tile(V, (a.n, 1))))
        q0 = np.asarray(st.pipeline_state.qpos)
        (_, _), (qp, dead) = roll_q((st, jp.zeros(a.n, bool)))
        qp = np.concatenate([q0[None], np.asarray(qp)], 0)                      # (nst+1, N, nq), tick 0 = the start
        rl, pt = roll_pitch(qp[..., 3:7]); tilt = np.hypot(rl, pt)
        fell = (tilt > 30).any(0)
        w_show = int(np.argmax(fell == (fell.mean() > 0.5)))                     # a walker with the majority outcome
        runs.append(dict(dt=dt, t0=float(L.t[i]), qp=qp, roll=rl, pitch=pt, fell=fell, V=V, e_fit=e_fit, w=w_show))
        print(f"{name[6:-4]} start onset-{dt:.1f}s (t {L.t[i]:.2f}): V {V[0]:+.2f},{V[1]:+.2f} fit {e_fit:.2f} deg | "
              f"{fell.mean()*100:.0f}% of {a.n} fall; shown: walker {w_show} ({'falls' if fell[w_show] else 'stays up'})", flush=True)

    # --- the robot's pose from the log (kinematic), on the CPU model
    vis = rs.ROOT / "models" / (env.cfg.model.replace("_train", "") + ".xml")     # the visual model (shells), same joints
    m = mujoco.MjModel.from_xml_path(str(vis)) if vis.exists() else env.mj_model; dd = mujoco.MjData(m)
    assert m.nq == env.mj_model.nq
    # the lower foot on the floor: use the env's own tip points on the MJX model for consistency
    from mujoco import mjx
    tip_low = jax.jit(lambda qq: env._tip_low(mjx.kinematics(env.sys, mjx.make_data(env.sys).replace(qpos=qq)))[0])
    def log_pose(i):
        q = np.zeros(m.nq); q[3:7] = rs.quat_from_g(L.G[i]); q[7:7 + nj] = L.Q[i]
        q[2] = -float(np.min(np.asarray(tip_low(jp.asarray(q))))) + 0.0005
        return q

    # --- frames
    R = mujoco.Renderer(m, a.H, a.W); opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = 1; opt.geomgroup[2] = 1
    def fcam(az):
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE; c.lookat[:] = (0.0, 0.0, 0.17)
        c.distance, c.azimuth, c.elevation = 0.95, az, -4.0
        return c
    cams = [fcam(180.0), fcam(90.0)]                                  # front (looking back at the robot), side (from its right)
    def shot(q, cam, grey=False):
        dd.qpos[:] = q; dd.qvel[:] = 0; mujoco.mj_forward(m, dd)
        R.update_scene(dd, camera=cam, scene_option=opt); img = R.render().copy()
        if grey:
            img = (0.45 * img + 0.55 * 255).astype(np.uint8)
        return img
    ctrl = env.cfg.ctrl_dt
    ticks = np.where((L.t >= t_a) & (L.t <= t_b))[0]
    t_axis = np.arange(t_a, t_b, ctrl)
    frames = []
    for tt in t_axis:
        il = int(np.argmin(np.abs(L.t - tt))); log_ok = abs(L.t[il] - tt) < 1.5 * ctrl
        ql = log_pose(il) if log_ok else None
        cols = [("robot (log)", ql, False, f"roll {L.roll[il]:+.0f}  pitch {L.pitch[il]:+.0f}" if log_ok else "log ended")]
        for R_ in runs:
            k = int(round((tt - R_["t0"]) / ctrl))
            lab = f"sim from onset{-R_['dt']:+.1f}s" if R_["dt"] else "sim from onset"
            if k < 0:
                cols.append((lab, ql, True, f"starts at {-R_['dt'] + 0.0:+.1f} s"))
            else:
                k = min(k, len(R_["qp"]) - 1); w_ = R_["w"]
                cols.append((lab, R_["qp"][k, w_], False, f"roll {R_['roll'][k, w_]:+.0f}  pitch {R_['pitch'][k, w_]:+.0f}"))
        rows = []
        for cam in cams:
            row = []
            for lab, q, grey, txt in cols:
                img = shot(q, cam, grey) if q is not None else np.full((a.H, a.W, 3), 200, np.uint8)
                row.append(img)
            rows.append(np.concatenate(row, 1))
        fr = Image.fromarray(np.concatenate(rows, 0)); dr_ = ImageDraw.Draw(fr)
        for c_, (lab, q, grey, txt) in enumerate(cols):
            dr_.text((c_ * a.W + 6, 4), lab, fill=(0, 0, 0)); dr_.text((c_ * a.W + 6, 16), txt, fill=(0, 0, 0))
            if c_ > 0:
                R_ = runs[c_ - 1]
                dr_.text((c_ * a.W + 6, a.H - 28), f"{R_['fell'].mean()*100:.0f}% of {a.n} walkers fall", fill=(160, 0, 0))
                dr_.text((c_ * a.W + 6, a.H - 16), f"walker {R_['w']} shown ({'falls' if R_['fell'][R_['w']] else 'stays up'})", fill=(0, 0, 0))
        dr_.text((6, a.H - 16), "front: robot's left = image right", fill=(0, 0, 0)); dr_.text((6, 2 * a.H - 16), "side: walking to the right", fill=(0, 0, 0))
        rel = tt - t_on
        fr2 = Image.new("RGB", (fr.width, 2 * a.H + 24), (255, 255, 255)); fr2.paste(fr, (0, 0)); dr2 = ImageDraw.Draw(fr2)
        dr2.text((8, 2 * a.H + 6), f"{name[6:-4]}  policy {pname}  gate {'on' if gate else 'off'}   t = {rel:+.2f} s from the fall onset"
                 f"   ({a.slow:g}x speed; the robot leaves its walking range at 0)", fill=(160, 0, 0) if rel >= 0 else (0, 0, 0))
        frames.append(np.asarray(fr2))
    R.close()
    import imageio.v2 as imageio
    vid = out / f"replay_{name[6:-4]}.mp4"
    with imageio.get_writer(vid, fps=int(round(a.slow / ctrl)), codec="libx264", quality=7, macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)

    # --- the roll/pitch plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sl = (L.t >= t_a) & (L.t <= t_b)
    for j, (key, ylab) in enumerate((("roll", "roll (deg, + right)"), ("pitch", "pitch (deg, + forward)"))):
        ax[j].plot(L.t[sl] - t_on, getattr(L, key)[sl], "k", lw=2.2, label="robot")
        for c_, R_ in enumerate(runs):
            tk = R_["t0"] - t_on + ctrl * np.arange(len(R_["qp"])); y = R_[key]
            col = f"C{c_}"
            ax[j].plot(tk, y[:, R_["w"]], col, lw=1.2, label=f"sim from onset{-R_['dt']:+.1f}s ({R_['fell'].mean()*100:.0f}% fall)")
            ax[j].plot(tk, np.median(y, 1), col, lw=0.8, ls="--")
            ax[j].axvline(R_["t0"] - t_on, color=col, lw=0.6, ls=":")
        ax[j].axvline(0, color="r", lw=0.8); ax[j].set_ylabel(ylab); ax[j].set_ylim(-35, 35); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=7, loc="lower left"); ax[1].set_xlabel("s from the fall onset (red line); dotted: the sim starts")
    ax[0].set_title(f"{name[6:-4]} ({pname}, gate {'on' if gate else 'off'}): solid = a walker with the majority outcome, dashed = median of {a.n}")
    fig.tight_layout(); fig.savefig(out / f"replay_{name[6:-4]}.png", dpi=110); plt.close(fig)
    print("wrote", vid, flush=True)
