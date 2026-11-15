"""Robot round 10 logs: roll and roll rate toward the stance side at each lift-off, for all steps and for the steps
that ended in a fall. Usage (repo root): .venv/bin/python scripts/robot10_analysis/liftoff.py hardware/leg_p0/software/stand_<run>_*.csv"""
import sys, csv, os, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import gap_fit, mujoco
m, d, tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
agg={}
for f in sorted(sys.argv[1:]):
    r=list(csv.reader(open(f))); h=r[0]; A=np.array(r[1:],float); t=A[:,0]
    c=lambda ks: A[:,[h.index(k) for k in ks]]
    g=c(["gx","gy","gz"]); w=np.degrees(c(["wx","wy","wz"])); q=c([f"q_{n}" for n in J]); a=c([f"a_{n}" for n in J]); cmd=np.degrees(c([f"cmd_{n}" for n in J])); qd=np.degrees(q)
    if not np.any(a!=0): continue
    eng=int(np.argmax(np.any(a!=0,1))); dt=np.median(np.diff(t))
    roll=np.degrees(np.arctan2(g[:,1],-g[:,2])); fp=-np.degrees(np.arctan2(-g[:,0],-g[:,2]))
    R=roll-roll[max(0,eng-25):eng+1].mean()
    rr=np.gradient(R,t)
    dz=np.zeros(len(t))
    for i in range(eng,len(t)):
        d.qpos[:]=0; d.qpos[2]=.3; d.qpos[3:7]=gap_fit._quat_from_g(g[i]); d.qpos[7:]=q[i]; mujoco.mj_kinematics(m,d)
        dz[i]=1000*(d.geom_xpos[tid[0],2]-d.geom_xpos[tid[1],2])
    bad=np.where((np.abs(R[eng:])>20)|(np.abs(fp[eng:]-4)>22))[0]; end=eng+bad[0] if len(bad) else len(t)-1
    ev=[]; up=False
    for i in range(eng,end):
        if not up and abs(dz[i])>6: up=True; s=i
        elif up and abs(dz[i])<3: up=False; ev.append((s,i))
    if up: ev.append((s,end))
    name=os.path.basename(f)[6:-20]
    for s,e in ev:
        sd=-np.sign(dz[s+1])  # + : roll direction toward stance (L up -> negative roll is stance)
        k=max(eng,s-int(0.1/dt))
        # stance hip roll: error q - cmd on stance side, during swing
        st = 6 if dz[s+1]>0 else 1  # L up -> stance R hip roll idx 6
        err=(qd[s:e+1,st]-cmd[s:e+1,st]); 
        agg.setdefault(name,[]).append((sd*R[s], sd*rr[k:s+1].mean(), sd*(R[s:e+1]).max(), np.abs(dz[s:e+1]).max(), t[e]-t[s], np.abs(err).max(), e==end))
print("policy     steps  roll@LO toward stance  roll rate@LO (deg/s)  max roll toward stance in swing   fell")
for n,X in agg.items():
    X=np.array(X); F=X[X[:,6]==1]; O=X[X[:,6]==0]
    print(f"{n:10s} {len(X):4d}   {np.median(O[:,0]):+5.1f}               {np.median(O[:,1]):+6.1f}  (p75 {np.percentile(O[:,1],75):+5.1f})       {np.median(O[:,2]):+5.1f} p90 {np.percentile(O[:,2],90):+5.1f}   | falling steps n={len(F)}: LO roll {np.median(F[:,0]) if len(F) else 0:+5.1f} rate {np.median(F[:,1]) if len(F) else 0:+6.1f}")
