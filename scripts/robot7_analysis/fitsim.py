"""Fit the simulator to the robot's wiggle (E8) responses."""
import sys, os, math, json, numpy as np
os.environ["JAX_PLATFORMS"]="cpu"; sys.path.insert(0,"."); sys.path.insert(0,"hardware/leg_p0/software")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run, quat_rot_t
import exercise_poses as ep, wig
src=open("hardware/leg_p0/software/exercises.py").read(); NS={"ep":ep,"np":np,"math":math}
exec(src[src.index("WIGGLE = {"):src.index("def cmd_wiggle")], NS)
CFG,_ = load_run(Path("runs/h4_t05"))
FREQS=(1,2,3,3.5,4,4.5,5,5.5,6,7)
BASE=dict(kp=8.0, volt=0.68, lat=6, delay=0, acc=(7.0,35.0,35.0), floor_tc=0.1, floor_damp=1.0, tc=0.0055, knee_off=-0.09,
          flex=(("hip_roll",10.0,0.15,2.5),("ankle_roll",10.0,0.15,2.5),("knee",15.0),("hip_pitch",15.0),("hip_yaw",10.0)), kp_roll=2.5,
          model="hw_ank3_b150_m2_10dof", extra={})
def make(P):
    over=dict(floor_tc=P["floor_tc"], floor_damp=P["floor_damp"], push_vel=0.0, imu_filter=True, dr=False, hidden_flex=tuple(tuple(x) for x in P["flex"]),
              flex_series=True, flex_tc=P["tc"], hidden_off=(("knee",P["knee_off"]),) if P["knee_off"] else (), drag_back=0.0, noise_qd=0.0, obs_noise=0.0,
              prof_acc=tuple(P["acc"]), kp=P["kp"], delay_steps=P["delay"], dr_delay_steps=(0, 4), reset_tilt=0.0, reset_joint_noise=0.0)
    over.update(P["extra"])
    env=CpuBiped(P["model"], EvalConfig.from_train(CFG["env"], **over), seed=1); env.reset((0,0,0)); env.ref=None
    s=env.servo; v=P["volt"]
    s["strength"]=np.full(env.nj, v); s["speed"]=np.full(env.nj, v); s["latency"]=P["lat"]; s["delay"]=P["delay"]
    s["kp"]=np.array([P["kp_roll"] if j.endswith("hip_roll") else 1.0 for j in env.joints]); s["acc"]=env.prof_acc.copy()
    return env
def wiggle(P, pattern, amp, lean=0.0, freqs=FREQS):
    env=make(P); tr=NS["Wiggle"](pattern, amp, list(freqs), lean_deg=lean); q_nom=env.q_nom.copy()
    T=[];W=[];Q=[];C=[];S=[];pitch=[]
    for i in range(int(tr.duration/0.02)):
        q,st,_=tr(i*0.02); a=np.clip((np.asarray(q)-q_nom)/env.cfg.action_scale,-1,1); env.step(a)
        T.append(i*0.02); W.append(env.d.qvel[3:6].copy()); Q.append(env.d.qpos[7:7+env.nj]+env.flex+env.q_off); C.append(env.cmd_buf[0].copy()); S.append(st)
        g=quat_rot_t(env.d.qpos[3:7],np.array([0,0,-1.0])); pitch.append(math.degrees(math.atan2(-g[0],-g[2])))
        if env.fallen(): return dict(fell=i*0.02, pitch0=float(np.mean(pitch[:50])))
    T,W,Q,C,S=map(np.array,(T,W,Q,C,S)); p=wig.PAT[pattern]; w=np.degrees(W); q=np.degrees(Q); cmd=np.degrees(C)
    out=dict(fell=None, pitch0=float(np.mean(pitch[100:140])), f=[], jr=[], jp=[], g=[], gp=[])
    ax={"roll":0,"pitch":1,"squat":1,"knees":1,"yaw":2}[pattern]
    for s in dict.fromkeys(S):
        if not s.endswith("Hz"): continue
        fr=float(s[:-2]); m=S==s; tm=T[m]; m&=(T>tm[0]+0.4)&(T<tm[-1]-0.3)
        uc=wig.lockin(T[m],cmd[m]@p/(p@p),fr); uq=wig.lockin(T[m],q[m]@p/(p@p),fr); gw=wig.lockin(T[m],w[m,ax],fr)/uc
        out["f"].append(fr); out["jr"].append(abs(uq/uc)); out["jp"].append(float(np.angle(uq/uc,1))); out["g"].append(abs(gw)); out["gp"].append(float(np.angle(gw,1)))
    return out
def robot(f):
    o=wig.analyse(f, quiet=True); h,A,st,meta=wig.load(f)
    g=A[:,[h.index(k) for k in ('gx','gy','gz')]]; pit=np.degrees(np.arctan2(-g[:,0],-g[:,2]))
    ax={"roll":0,"pitch":1,"squat":1,"knees":1,"yaw":2}[meta["pattern"]]
    return dict(f=[x[0] for x in o], jr=[x[2] for x in o], jp=[x[3] for x in o], g=[x[4][ax] for x in o], gp=[x[5][ax] for x in o], pitch0=float(pit[:50].mean()))
S="hardware/leg_p0/software/"
TARGETS={("roll",1.5,0.0):S+"ex_wiggle_roll_20261113_114550.csv", ("pitch",1.5,0.0):S+"ex_wiggle_pitch_20261113_114720.csv",
         ("squat",1.0,2.0):S+"ex_wiggle_squat_20261113_115925.csv", ("yaw",1.5,0.0):S+"ex_wiggle_yaw_20261113_114920.csv"}
def score(sim, rob):
    if sim["fell"] is not None: return 9.0
    lg=lambda a,b: np.mean(np.abs(np.log(np.maximum(a,0.05)/np.maximum(b,0.05))))
    return float(lg(np.array(sim["g"]),np.array(rob["g"])) + 0.5*lg(np.array(sim["jr"]),np.array(rob["jr"])))
def show(name, d):
    if d.get("fell") is not None: print(f"  {name:10s} FELL at {d['fell']:.2f} s (pitch0 {d['pitch0']:+.1f})"); return
    print(f"  {name:10s} pitch0 {d['pitch0']:+5.1f} joint "+" ".join(f"{x:4.2f}" for x in d["jr"])+" | gyro/deg "+" ".join(f"{x:5.1f}" for x in d["g"]))
if __name__=="__main__":
    import ast
    P=dict(BASE); [P.update({k:ast.literal_eval(v)}) for k,v in (x.split("=",1) for x in sys.argv[1:])]
    tot=0
    for (pat,amp,lean),f in TARGETS.items():
        r=robot(f); s=wiggle(P,pat,amp,lean); show("ROBOT "+pat,r); show("sim "+pat,s); sc=score(s,r); tot+=sc; print(f"  score {sc:.2f}")
    print("TOTAL", round(tot,2))
