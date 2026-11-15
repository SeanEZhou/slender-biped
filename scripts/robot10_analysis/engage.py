"""Robot round 10 logs: pitch right after engaging (forward positive) and the first joint moves.
Usage (repo root): .venv/bin/python scripts/robot10_analysis/engage.py hardware/leg_p0/software/stand_<run>_*.csv"""
import sys, csv, os, numpy as np
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
print("policy            time   pre | fwd pitch at +0.25 +0.5 +0.75 +1.0 +1.5  max1.5 | hipP L/R  knee L/R (deg vs pre, at +0.5s) | fwd rate max")
for f in sorted(sys.argv[1:]):
    r=list(csv.reader(open(f))); h=r[0]; A=np.array(r[1:],float); t=A[:,0]
    c=lambda ks: A[:,[h.index(k) for k in ks]]
    g=c(["gx","gy","gz"]); w=c(["wx","wy","wz"]); a=c([f"a_{n}" for n in J]); cmd=np.degrees(c([f"cmd_{n}" for n in J]))
    if not np.any(a!=0): continue
    eng=int(np.argmax(np.any(a!=0,1))); dt=np.median(np.diff(t))
    fp=-np.degrees(np.arctan2(-g[:,0],-g[:,2]))   # forward positive
    pre=fp[max(0,eng-25):eng+1].mean()
    at=lambda s: fp[min(len(t)-1,eng+int(s/dt))]
    e15=min(len(t),eng+int(1.5/dt))
    i5=min(len(t)-1,eng+int(0.5/dt)); dc=cmd[i5]-cmd[eng-1]
    print(f"{os.path.basename(f)[6:-20]:16s} {os.path.basename(f)[-10:-4]} {pre:+5.1f} | {at(.25):+5.1f} {at(.5):+5.1f} {at(.75):+5.1f} {at(1):+5.1f} {at(1.5):+5.1f}  {fp[eng:e15].max():+5.1f} | {dc[2]:+5.1f}/{dc[7]:+5.1f} {dc[3]:+5.1f}/{dc[8]:+5.1f} | {np.degrees(-w[eng:e15,1]).max():+5.0f}")
