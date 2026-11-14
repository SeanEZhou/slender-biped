import sys, json, fitsim, numpy as np
from multiprocessing import Pool
rob=fitsim.robot(fitsim.TARGETS[("pitch",1.5,0.0)])
def job(a):
    dl, rd, g, z = a
    P=dict(fitsim.BASE, knee_off=-0.015, delay=dl, extra={"ring":(("pitch",4.2,z,g,0.02),),"ring_delay":rd}); return a, fitsim.wiggle(P,"pitch",1.5,0.0)
if __name__=="__main__":
    print("robot  gyro", np.round(rob["g"],1), "\n       gph", np.round(rob["gp"]), "jph", np.round(rob["jp"]))
    with Pool(3) as p:
        for a,d in p.map(job, json.loads(sys.argv[1])):
            if d["fell"] is not None: print(a,"FELL",d["fell"]); continue
            print(a, "score", round(fitsim.score(d,rob),2), "gyro", np.round(d["g"],1), "\n       gph", np.round(d["gp"]), "jph", np.round(d["jp"]), "joint", np.round(d["jr"],2))
