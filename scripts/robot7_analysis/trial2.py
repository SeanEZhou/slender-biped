import sys, csv, json, math, os, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"."); os.environ["JAX_PLATFORMS"]="cpu"
import gap_fit, mujoco
from slender_biped.evaluation import NumpyPolicy
m,d,tid,_=gap_fit._fk_setup(gap_fit.MODEL)
J=["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
for f in sys.argv[1:]:
    run=os.path.basename(f)[6:-20]
    r=list(csv.reader(open(f)));h=r[0];A=np.array(r[1:],float);t=A[:,0]
    c=lambda ks: A[:,[h.index(k) for k in ks]]
    g=c(["gx","gy","gz"]);w=np.degrees(c(["wx","wy","wz"]));act=c([f"a_{n}" for n in J]);q=c([f"q_{n}" for n in J])
    roll=np.degrees(np.arctan2(g[:,1],-g[:,2]));pit=np.degrees(np.arctan2(-g[:,0],-g[:,2]))
    eng=int(np.argmax(np.any(act!=0,1))); s=t>t[eng]+0.5
    def bp(x):
        fr=np.fft.rfftfreq(len(x),0.02);X=np.fft.rfft(x-x.mean(0),axis=0);X[(fr<3.5)|(fr>=8)]=0;return np.fft.irfft(X,len(x),axis=0)
    sh=np.sqrt(np.mean(bp(w[s])**2,0)) if s.sum()>50 else [np.nan]*3
    dz=[]; zt=[]
    for i in range(len(t)):
        d.qpos[:]=0;d.qpos[2]=.3;d.qpos[3:7]=gap_fit._quat_from_g(g[i]);d.qpos[7:]=q[i];mujoco.mj_kinematics(m,d);dz.append(1000*(d.geom_xpos[tid[0],2]-d.geom_xpos[tid[1],2]))
    dz=np.array(dz); sw=np.abs(dz)>5; steps=int(np.sum(sw[1:]&~sw[:-1]))
    # infer the action low-pass from the logged (filtered) actions: a_t = k a_{t-1} + (1-k) raw_t, raw from the policy?
    k_est=np.nan
    try:
        pol=NumpyPolicy(sorted(__import__('pathlib').Path(f"runs/{run}").glob("params_*M.pkl"))[-1])
    except Exception: pol=None
    ps=pit[s]
    print(f"{os.path.basename(f)[6:-4]:30s} T {t[-1]:5.1f} steps {steps:3d} shake r/p/y {sh[0]:5.1f} {sh[1]:5.1f} {sh[2]:5.1f}  pitch walk mean {ps.mean() if len(ps) else np.nan:+5.1f} p10 {np.percentile(ps,10) if len(ps) else np.nan:+5.1f}  end roll {roll[-1]:+4.0f} pitch {pit[-1]:+4.0f}")
