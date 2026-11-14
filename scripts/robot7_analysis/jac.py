import sys; sys.path.insert(0,'.')
import numpy as np, replay, json
S="hardware/leg_p0/software/"
for a in sys.argv[1:]:
    run,f=a.split(":"); pol,O,A,t=replay.build(run,S+f)
    sc=json.load(open(f"runs/{run}/config.json"))["env"]["action_scale"]
    Js=[]
    for o in O[50::10]:
        J=np.zeros((10,3)); e=0.02
        for k in range(3):
            op=o.copy(); om=o.copy(); op[3+k]+=e; om[3+k]-=e
            J[:,k]=(np.clip(pol(op),-1,1)-np.clip(pol(om),-1,1))/(2*e)
        Js.append(J*sc)   # rad cmd per rad/s
    Js=np.array(Js); m=np.sqrt(np.mean(Js**2,0))   # rms gain, (10,3) rad per rad/s == deg per deg/s
    tot=np.sqrt(np.sum(m**2,0))
    print(f"{run:10s} gyro->cmd rms gain (deg per deg/s, all joints): roll {tot[0]:.3f} pitch {tot[1]:.3f} yaw {tot[2]:.3f}   biggest: "+
          ", ".join(f"{replay.J[i]}<-{'rpy'[k]} {m[i,k]:.3f}" for i,k in zip(*np.unravel_index(np.argsort(m.ravel())[::-1][:4],m.shape))))
