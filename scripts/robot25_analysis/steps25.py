"""Per step on the robot: at each scheduled touchdown, the feet from FK (encoders + IMU, flat floor assumed):
dx = landing foot ahead of the stance foot (mm), dy = foot spacing (mm), and the body roll/pitch. First 4 steps vs the rest."""
import sys,csv,math,glob,numpy as np,mujoco
sys.path.insert(0,'scripts/robot18_analysis'); sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import swing_clear as s
m,d,tid=s._m,s._d,s._tid
def feet(q,g):
    d.qpos[:]=0; d.qpos[2]=0.3; d.qpos[3:7]=s.gap_fit._quat_from_g(g); d.qpos[7:]=q; mujoco.mj_kinematics(m,d)
    return d.geom_xpos[tid[0]].copy(), d.geom_xpos[tid[1]].copy()
def analyse(f, verbose=False):
    r=list(csv.reader(open(f)));h=r[0];A=np.array([[float(v) if v not in('True','False') else float(v=='True') for v in x] for x in r[1:] if len(x)==len(h)])
    run=f.split('/')[-1][6:-20]; tab,z,ev=s.table(run); st=z['stance'][-1]; n=len(st)
    c=lambda k:A[:,h.index(k)]; u=c('phase')/2/np.pi; G=A[:,[h.index(k) for k in('gx','gy','gz')]]
    Q=A[:,[h.index('q_'+j) for j in s.J]]; roll=np.degrees(np.arctan2(-G[:,1],-G[:,2])); pitch=np.degrees(np.arctan2(G[:,0],-G[:,2]))
    k0=np.where(np.abs(np.diff(u))>1e-6)[0][0]; r0=roll[k0-25:k0].mean()
    bad=np.where((np.abs(roll-r0)>20)&(np.arange(len(u))>k0))[0]; b=bad[0] if len(bad) else len(u)-1
    out=[]
    for i in range(k0,b-8):
        p0=int((u[i]%1)*n)%n; p1=int((u[i+1]%1)*n)%n
        for side in (0,1):
            if st[p0][side]<.5 and st[p1][side]>=.5:
                j=min(i+6,b-1)       # 0.12 s after the scheduled touchdown (servo lag)
                L,R=feet(Q[j],G[j]); sw,sd=(L,R) if side==0 else (R,L)
                out.append(dict(t=(i-k0)*0.02, side='LR'[side], dx=(sw[0]-sd[0])*1000, dy=abs(L[1]-R[1])*1000,
                                roll=np.ptp(roll[max(i-25,k0):j]) , rmax=(roll[max(i-25,k0):j]-r0).max(), rmin=(roll[max(i-25,k0):j]-r0).min()))
    return run,out
if __name__=='__main__':
    for f in sys.argv[1:]:
        run,o=analyse(f)
        print(f"{f.split('stand_')[1][:-4]:30s} first steps (side dx/dy mm): "+'  '.join(f"{x['side']} {x['dx']:+4.0f}/{x['dy']:3.0f}" for x in o[:5])
              +f"  | later median: L dx {np.median([x['dx'] for x in o[5:] if x['side']=='L']):+.0f} R dx {np.median([x['dx'] for x in o[5:] if x['side']=='R']):+.0f}, width {np.median([x['dy'] for x in o[5:]]):.0f}")
