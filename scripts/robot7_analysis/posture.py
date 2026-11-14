import sys, os, csv, numpy as np, math, glob
os.environ["JAX_PLATFORMS"]="cpu"; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
from pathlib import Path
J=["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
S="hardware/leg_p0/software/"
def robot(run):
    out=[]
    for f in sorted(glob.glob(S+f"stand_{run}_20261113_19*.csv")):
        r=list(csv.reader(open(f)));h=r[0];A=np.array(r[1:],float);t=A[:,0];act=A[:,[h.index(f"a_{n}") for n in J]]
        eng=int(np.argmax(np.any(act!=0,1))); s=(t>t[eng]+1)&(t<t[-1]-1)
        if s.sum()<50: continue
        q=np.degrees(A[s][:,[h.index(f"q_{n}") for n in J]]); g=A[s][:,[h.index(k) for k in ('gx','gy','gz')]]
        pit=np.degrees(np.arctan2(-g[:,0],-g[:,2])); a=act[s]
        out.append((q[:,[2,7]].mean(), q[:,[3,8]].mean(), pit.mean(), a[:,[2,7]].mean(), a[:,[3,8]].mean()))
    return np.array(out).mean(0) if out else None
def sim(run, lpf=0.0):
    import eval6
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of, quat_rot_t
    cfg, rd = load_run(Path("runs")/run); pol=NumpyPolicy(rd/"params.pkl")
    over=dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0, dr_volt=(0.68,0.68), dr_delay_steps=(0,1), prof_acc=(25.0,60.0,60.0), noise_qd=0.25, dr_kp=(0.75,1.25), kp_hip_roll=2.5, drag_back=0.0)
    over.update(eval6.SIMS["r7"]);
    if lpf: over["act_lpf"]=lpf
    R=[]
    for seed in range(3):
        env=CpuBiped("hw_ank3_b150_m2_10dof", EvalConfig.from_train(cfg["env"], **over), seed=81000+seed); obs=env.reset((0.015,0,0)); y0=yaw_of(env.d.qpos[3:7])
        Q=[];P=[];Aa=[]
        for i in range(750):
            env.cmd[2]=float(np.clip((y0-yaw_of(env.d.qpos[3:7])+np.pi)%(2*np.pi)-np.pi,-.3,.3)); obs[-5:-2]=env.cmd
            a=pol(obs); obs,_=env.step(a)
            if i>50: Q.append(np.degrees(env.d.qpos[7:17]+env.flex+env.q_off)); g=quat_rot_t(env.d.qpos[3:7],np.array([0,0,-1.0])); P.append(math.degrees(math.atan2(-g[0],-g[2]))); Aa.append(env.last_act)
            if env.fallen(): break
        if len(Q)<50: continue
        Q=np.array(Q);Aa=np.array(Aa); R.append((Q[:,[2,7]].mean(),Q[:,[3,8]].mean(),np.mean(P),Aa[:,[2,7]].mean(),Aa[:,[3,8]].mean()))
    return np.array(R).mean(0)
for run,lpf in [("r10_lcp_h4",0),("r13_alpf_b7",0),("r11_alpf_fwd",0)]:
    rb=robot(run); sm=sim(run,lpf)
    fmt=lambda x: f"hip {x[0]:5.1f} knee {x[1]:5.1f} pitch {x[2]:+5.1f} | act hip {x[3]:+.2f} knee {x[4]:+.2f}"
    print(f"{run:13s} ROBOT {fmt(rb)}\n{'':13s} SIM   {fmt(sm)}")
