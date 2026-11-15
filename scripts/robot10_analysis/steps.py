"""Robot round 10 logs: per trial, steps, swing time, double-support gap, swing peak (FK, includes body tilt:
compare ratios only), roll during swing, gyro RMS, pitch peak-to-peak per 2 s (the rock building up).
Usage (repo root): .venv/bin/python scripts/robot10_analysis/steps.py hardware/leg_p0/software/stand_<run>_*.csv"""
import sys, csv, os, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import gap_fit, mujoco
m, d, tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
agg={}
for f in sorted(sys.argv[1:]):
    r=list(csv.reader(open(f))); h=r[0]; A=np.array(r[1:],float); t=A[:,0]
    c=lambda ks: A[:,[h.index(k) for k in ks]]
    g=c(["gx","gy","gz"]); w=np.degrees(c(["wx","wy","wz"])); q=c([f"q_{n}" for n in J]); a=c([f"a_{n}" for n in J])
    if not np.any(a!=0): continue
    eng=int(np.argmax(np.any(a!=0,1))); dt=np.median(np.diff(t))
    roll=np.degrees(np.arctan2(g[:,1],-g[:,2])); fp=-np.degrees(np.arctan2(-g[:,0],-g[:,2]))
    r0=roll[max(0,eng-25):eng+1].mean(); R=roll-r0
    dz=np.zeros(len(t))
    for i in range(eng,len(t)):
        d.qpos[:]=0; d.qpos[2]=.3; d.qpos[3:7]=gap_fit._quat_from_g(g[i]); d.qpos[7:]=q[i]; mujoco.mj_kinematics(m,d)
        dz[i]=1000*(d.geom_xpos[tid[0],2]-d.geom_xpos[tid[1],2])
    bad=np.where((np.abs(R[eng:])>20)|(np.abs(fp[eng:]-4)>22))[0]; end=eng+bad[0] if len(bad) else len(t)-1
    ev=[]; up=False
    for i in range(eng,end):
        if not up and abs(dz[i])>6: up=True; s=i
        elif up and abs(dz[i])<3: up=False; ev.append((s,i))
    name=os.path.basename(f)[6:-20]
    S=[]
    for (s,e) in ev:
        side=np.sign(dz[s+1]) # + : left foot higher -> stance right
        pk=np.abs(dz[s:e]).max()
        # roll toward the swing side (falling into the lifted foot) = side*R ... sign: positive R = ? use magnitude signed by side
        rs=side*R[s:e]
        S.append((t[e]-t[s], pk, rs.max()-rs[0], rs.min()-rs[0], np.ptp(fp[s:e]), np.sqrt((w[s:e,:2]**2).mean())))
    S=np.array(S) if S else np.zeros((0,6))
    # gaps between steps (double support)
    gaps=[t[ev[k+1][0]]-t[ev[k][1]] for k in range(len(ev)-1)]
    # pitch rock growth: ptp of pitch per 2 s window
    win=int(2/dt); pr=[np.ptp(fp[i:i+win]) for i in range(eng+int(1.5/dt),end-win,win)]
    agg.setdefault(name,[]).append((S,gaps,t[end]-t[eng]))
    print(f"{name:10s} {os.path.basename(f)[-10:-4]} up {t[end]-t[eng]:5.1f}s steps {len(S):3d} swing {np.median(S[:,0]) if len(S) else 0:4.2f}s gap {np.median(gaps) if gaps else 0:4.2f}s  pk {np.median(S[:,1]) if len(S) else 0:4.1f}mm  roll+ {np.median(S[:,2]) if len(S) else 0:4.1f} roll- {np.median(S[:,3]) if len(S) else 0:+5.1f}  pitch ptp/step {np.median(S[:,4]) if len(S) else 0:4.1f}  gyroRMS {np.median(S[:,5]) if len(S) else 0:4.0f}  pitch ptp per 2s: {' '.join(f'{x:.0f}' for x in pr[:12])}")
print()
for n,L in agg.items():
    S=np.vstack([x[0] for x in L if len(x[0])]) if any(len(x[0]) for x in L) else np.zeros((0,6)); G=sum([x[1] for x in L],[])
    print(f"{n:10s} trials {len(L):2d} median up {np.median([x[2] for x in L]):5.1f}s  steps {len(S):4d}  swing pk {np.median(S[:,1]):4.1f}mm (p75 {np.percentile(S[:,1],75):4.1f})  roll into swing p90 {np.percentile(S[:,2],90):4.1f}  away p90 {np.percentile(-S[:,3],90):4.1f}  gap med {np.median(G) if G else 0:4.2f}s  gyro {np.median(S[:,5]):4.0f}")
