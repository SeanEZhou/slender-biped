"""Robot round 10 logs: how each fall ended (roll, pitch, which foot was up) and roll by swing side.
Usage (repo root): .venv/bin/python scripts/robot10_analysis/falls.py hardware/leg_p0/software/stand_<run>_*.csv"""
import sys, csv, os, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import gap_fit, mujoco
m, d, tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
agg={}
for f in sorted(sys.argv[1:]):
    r=list(csv.reader(open(f))); h=r[0]; A=np.array(r[1:],float); t=A[:,0]
    c=lambda ks: A[:,[h.index(k) for k in ks]]
    g=c(["gx","gy","gz"]); q=c([f"q_{n}" for n in J]); a=c([f"a_{n}" for n in J]); cmd=np.degrees(c([f"cmd_{n}" for n in J]))
    if not np.any(a!=0): continue
    eng=int(np.argmax(np.any(a!=0,1))); dt=np.median(np.diff(t))
    roll=np.degrees(np.arctan2(g[:,1],-g[:,2])); fp=-np.degrees(np.arctan2(-g[:,0],-g[:,2]))
    R=roll-roll[max(0,eng-25):eng+1].mean()
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
    # side=+1 : left foot up (stance right)
    X=[(np.sign(dz[s+1]), R[s], R[s:e+1][np.argmax(np.abs(R[s:e+1]))], np.abs(dz[s:e+1]).max(), t[e]-t[s], fp[s], np.ptp(cmd[s:e+1,[1,6]],0).max()) for s,e in ev]
    agg.setdefault(name,[]).extend(X)
    if len(bad):
        j=end; lastside=np.sign(dz[ev[-1][0]+1]) if ev else 0
        inswing = ev and ev[-1][1]>=end-int(0.15/dt)
        print(f"{name:10s} {os.path.basename(f)[-10:-4]} FALL at {t[j]-t[eng]:5.1f}s roll {R[j]:+5.1f} fwd {fp[j]:+5.1f}  last lifted foot {'L' if lastside>0 else 'R'}  during swing {bool(inswing)}  last swing pk {X[-1][3] if X else 0:4.1f}mm dur {X[-1][4] if X else 0:4.2f}s")
print()
print("policy     side  n   roll@liftoff  roll peak in swing  (deg, + = robot's left down?)   swing pk  dur  hipRoll cmd ptp")
for n,X in agg.items():
    X=np.array(X)
    for sd in (1,-1):
        Y=X[X[:,0]==sd]
        if len(Y): print(f"{n:10s} {'Lup' if sd>0 else 'Rup'} {len(Y):4d}   {np.median(Y[:,1]):+5.1f}         {np.median(Y[:,2]):+5.1f} (p90 |{np.percentile(np.abs(Y[:,2]),90):4.1f}|)                          {np.median(Y[:,3]):5.1f}  {np.median(Y[:,4]):4.2f}  {np.median(Y[:,6]):4.1f}")
