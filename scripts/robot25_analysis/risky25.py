import sys,csv,math,numpy as np
sys.path.insert(0,sys.argv[1]); sys.path.insert(0,'scripts/robot18_analysis'); sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
from steps25 import feet, s
rows=[]
for f in sys.argv[2:]:
    r=list(csv.reader(open(f)));h=r[0];A=np.array([[float(v) if v not in('True','False') else float(v=='True') for v in x] for x in r[1:] if len(x)==len(h)])
    run=f.split('/')[-1][6:-20]; tab,z,ev=s.table(run); st=z['stance'][-1]; n=len(st)
    c=lambda k:A[:,h.index(k)]; u=c('phase')/2/np.pi; G=A[:,[h.index(k) for k in('gx','gy','gz')]]; t=c('t')
    Q=A[:,[h.index('q_'+j) for j in s.J]]; roll=np.degrees(np.arctan2(-G[:,1],-G[:,2])); pitch=np.degrees(np.arctan2(G[:,0],-G[:,2]))
    wx=np.degrees(c('wx')); wz=np.degrees(c('wz')); gate=c('gate_held')
    k0=np.where(np.abs(np.diff(u))>1e-6)[0][0]; r0=roll[k0-25:k0].mean(); p0=pitch[k0-25:k0].mean()
    bad=np.where((np.abs(roll-r0)>20)&(np.arange(len(u))>k0))[0]; b=bad[0] if len(bad) else len(u)-1
    # lift-offs and touchdowns from the table
    lo=[];td=[]
    for i in range(k0,b-1):
        p_0=int((u[i]%1)*n)%n; p_1=int((u[i+1]%1)*n)%n
        for sd in (0,1):
            if st[p_0][sd]>=.5 and st[p_1][sd]<.5: lo.append((i+1,sd))
            if st[p_0][sd]<.5 and st[p_1][sd]>=.5: td.append((i+1,sd))
    for (il,sd) in lo:
        nt=[i for (i,s2) in td if s2==sd and i>il]
        if not nt: continue
        it=nt[0]; je=min(it+15,b-1)
        L,R=feet(Q[min(it+6,b-1)],G[min(it+6,b-1)]); sw,sdd=(L,R) if sd==0 else (R,L)
        seg=slice(il,je)
        rows.append(dict(f=f[-19:-4],t=t[il]-t[k0],side='LR'[sd],
            roll_lo=roll[il]-r0, wx_lo=wx[il], pitch_lo=pitch[il]-p0, gate=gate[il-10:il+1].max(),
            rmax=(roll[seg]-r0).max(), rmin=(roll[seg]-r0).min(), pmax=(pitch[seg]-p0).max(),
            yaw=np.sum(wz[il:je])*0.02, dx=(sw[0]-sdd[0])*1000, dy=abs(L[1]-R[1])*1000,
            fell=bool(len(bad)) and je>=b-30))
import collections
for sd in 'LR':
    R_=[x for x in rows if x['side']==sd]
    sev=np.array([max(abs(x['rmax']),abs(x['rmin'])) for x in R_])
    print(f"\n{sd} swings: {len(R_)}; roll excursion during the swing p50 {np.median(sev):.1f}, p90 {np.percentile(sev,90):.1f}, max {sev.max():.1f}; yaw per step p50 {np.median([abs(x['yaw']) for x in R_]):.1f} deg")
    order=np.argsort(-sev)
    print("   worst: when | at lift-off: roll, roll rate, pitch, gate held | during swing roll range, pitch max, yaw | landing dx/dy")
    for k in order[:8]:
        x=R_[k]; print(f"   {x['f']} t {x['t']:5.1f} | {x['roll_lo']:+5.1f} {x['wx_lo']:+6.1f}/s {x['pitch_lo']:+5.1f} {int(x['gate']):2d} | {x['rmin']:+5.1f}..{x['rmax']:+5.1f}  {x['pmax']:+5.1f}  {x['yaw']:+5.1f} | {x['dx']:+4.0f}/{x['dy']:3.0f}{'  FELL' if x['fell'] else ''}")
    typ=[R_[k] for k in order[len(order)//4:3*len(order)//4]]
    print(f"   typical (middle half): roll at lift-off {np.median([x['roll_lo'] for x in typ]):+.1f}, rate {np.median([x['wx_lo'] for x in typ]):+.1f}/s, pitch {np.median([x['pitch_lo'] for x in typ]):+.1f}, dx/dy {np.median([x['dx'] for x in typ]):+.0f}/{np.median([x['dy'] for x in typ]):.0f}")
    # correlation: excursion vs conditions at lift-off
    for key in ('roll_lo','wx_lo','pitch_lo'):
        v=np.array([x[key] for x in R_]); print(f"   corr(excursion, {key}) {np.corrcoef(sev,v)[0,1]:+.2f}")
np.save(sys.argv[1]+'/risky_rows.npy',rows,allow_pickle=True)
