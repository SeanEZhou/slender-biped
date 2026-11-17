"""Round 19 (2026-11-17): gait size and rhythm on the robot, per run. Per swing: step length (swing board ahead of the stance
board 0.1 s after the scheduled touchdown, encoder FK + IMU attitude), stride (the swing board's travel relative to the
stance board, lift-off to touchdown), step time, speed (stride / 2 step times), body roll / pitch peak-to-peak per cycle,
hip-roll spread at lift-off, the landing knee's speed just before touchdown, and the gate's hold share.
Usage (repo root): .venv/bin/python scripts/robot19_analysis/gait_metrics.py <stand_*.csv> ...
"""
import sys, csv, math, numpy as np, mujoco
sys.path.insert(0,'scripts/robot18_analysis'); sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import swing_clear as s
J=s.J
def feet(q,g):
    s._d.qpos[:]=0; s._d.qpos[2]=0.3; s._d.qpos[3:7]=s.gap_fit._quat_from_g(g); s._d.qpos[7:]=q
    mujoco.mj_kinematics(s._m,s._d)
    return np.array([s._d.geom_xpos[k].copy() for k in s._tid])*1000  # L,R board centres mm, gravity frame (x fwd)
res={}
for f in sys.argv[1:]:
    r=list(csv.reader(open(f))); h=r[0]; A=np.array([[float(v) if v not in('True','False') else float(v=='True') for v in x] for x in r[1:] if len(x)==len(h)])
    col=lambda k:A[:,h.index(k)]
    run=f.split('/')[-1][6:-20]; tab,z,ev=s.table(run)
    t=col('t'); g=A[:,[h.index(k) for k in ('gx','gy','gz')]]; u=col('phase')/2/math.pi; held=col('gate_held')
    q=A[:,[h.index('q_'+n) for n in J]]; qd=A[:,[h.index('qd_'+n) for n in J]]
    roll=np.degrees(np.arctan2(-g[:,1],-g[:,2])); pitch=np.degrees(np.arctan2(g[:,0],-g[:,2]))
    eng=np.where(np.abs(np.diff(u))>1e-6)[0]; k0=eng[0]
    roll-=roll[k0-25:k0+1].mean(); pitch-=pitch[k0-25:k0+1].mean()
    bad=np.where((np.hypot(roll,pitch)>20)&(np.arange(len(t))>k0))[0]; b=bad[0] if len(bad) else len(t)-1
    evs=[]
    for i in range(k0,b):
        for (ue,kind,side) in ev:
            if u[i]!=u[i+1] and (u[i]-ue)%1>(u[i+1]-ue)%1: evs.append((i+1,kind,side))
    R=res.setdefault(run,dict(len=[],D=[],dt=[],spread=[],roll=[],held=[],kd=[],pitch=[],T=0,tds=0,ws=[]))
    R['T']+=t[b]-t[k0]; R['held'].append(held[k0:b].sum()); R['tds']+=sum(e[1]=='td' for e in evs)
    tds=[e for e in evs if e[1]=='td']
    for a,(i,kind,side) in enumerate(evs):
        if kind!='lo': continue
        nx=[e for e in evs[a+1:] if e[1]=='td' and e[2]==side]
        if not nx: continue
        j=nx[0][0]; fi=0 if side=='L' else 1
        jj=min(j+5,b-1)                       # 0.1 s after scheduled touchdown, foot planted
        P0=feet(q[i],g[i]); P1=feet(q[jj],g[jj])
        sep0=P0[fi,0]-P0[1-fi,0]; sep1=P1[fi,0]-P1[1-fi,0]
        R['len'].append(sep1); R['D'].append(sep1-sep0)
        R['spread'].append(abs(math.degrees(q[i,1]-q[i,6])))
        kk=slice(max(i,j-3),j); R['kd'].append(np.degrees(np.abs(qd[kk,3+5*fi])).max())
    for a in range(1,len(tds)): R['dt'].append(t[tds[a][0]]-t[tds[a-1][0]])
    # lean: p2p roll per cycle (between same-side touchdowns)
    sameR=[e[0] for e in tds if e[2]=='R']
    for a in range(1,len(sameR)): R['roll'].append(np.ptp(roll[sameR[a-1]:sameR[a]])); R['pitch'].append(np.ptp(pitch[sameR[a-1]:sameR[a]]))
print(f"{'run':10s} walk_s  tds | step len mm p50 (p10-p90) | stride D mm | s/step p50 | speed mm/s | roll p2p deg | pitch p2p | hip-roll spread @lo | knee deg/s pre-td | gate held %")
for run,R in res.items():
    a=lambda k:np.array(R[k])
    sp=np.median(a('D'))/2/np.median(a('dt'))
    print(f"{run:10s} {R['T']:6.0f} {R['tds']:4d} | {np.median(a('len')):5.0f} ({np.percentile(a('len'),10):4.0f}-{np.percentile(a('len'),90):4.0f}) | {np.median(a('D')):5.0f} | {np.median(a('dt')):.2f} | {sp:5.0f} | {np.median(a('roll')):5.1f} | {np.median(a('pitch')):5.1f} | {np.median(a('spread')):5.1f} | {np.median(a('kd')):5.0f} | {100*sum(R['held'])*0.02/R['T']:4.0f}")
