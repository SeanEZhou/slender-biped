import sys, itertools, json, numpy as np
from multiprocessing import Pool
import fitsim
def job(a):
    pat, amp, lean, upd = a
    P=dict(fitsim.BASE, knee_off=-0.015); P.update(upd)
    try: return upd, fitsim.wiggle(P, pat, amp, lean)
    except Exception as e: return upd, dict(fell=-1, pitch0=0, err=str(e))
if __name__=="__main__":
    pat=sys.argv[1]; amp,lean={"pitch":(1.5,0),"roll":(1.5,0),"squat":(1.0,2.0),"yaw":(1.5,0)}[pat]
    grid=json.loads(sys.argv[2])   # {"param": [values]}; flex entries given as full tuples
    keys=list(grid); jobs=[(pat,amp,lean,dict(zip(keys,v))) for v in itertools.product(*grid.values())]
    rob=fitsim.robot(fitsim.TARGETS[(pat,amp,lean)]); fitsim.show("ROBOT",rob)
    with Pool(6) as p: res=p.map(job, jobs)
    res=sorted(res, key=lambda r: fitsim.score(r[1],rob))
    for upd,d in res[:int(sys.argv[3]) if len(sys.argv)>3 else 8]:
        print(round(fitsim.score(d,rob),2), json.dumps(upd)); fitsim.show("sim",d)
