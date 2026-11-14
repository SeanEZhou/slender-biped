import sys, os, numpy as np
os.environ["JAX_PLATFORMS"]="cpu"; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
from pathlib import Path
from multiprocessing import Pool
def ep(a):
    run, seed = a
    import eval6
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of
    cfg, rd = load_run(Path("runs")/run); pol=NumpyPolicy(rd/"params.pkl")
    over=dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0, dr_volt=(0.68,0.68), dr_delay_steps=(0,1), prof_acc=(25.0,60.0,60.0), noise_qd=0.25, dr_kp=(0.75,1.25), kp_hip_roll=2.5, drag_back=0.0)
    over.update(eval6.SIMS["r7"])
    env=CpuBiped("hw_ank3_b150_m2_10dof", EvalConfig.from_train(cfg["env"], **over), seed=81000+seed); obs=env.reset((0.015,0,0)); y0=yaw_of(env.d.qpos[3:7])
    up=np.zeros(2,bool); lifts=0; hmax=[]; cur=np.zeros(2)
    for i in range(1500):
        env.cmd[2]=float(np.clip((y0-yaw_of(env.d.qpos[3:7])+np.pi)%(2*np.pi)-np.pi,-.3,.3)); obs[-5:-2]=env.cmd
        obs,_=env.step(pol(obs)); low,_=env.tip_low(); u=low>0.005
        for k in range(2):
            if u[k] and not up[k]: lifts+=1; cur[k]=0
            if u[k]: cur[k]=max(cur[k],low[k])
            if up[k] and not u[k]: hmax.append(cur[k])
        up=u
        if env.fallen(): break
    return run, lifts, (np.median(hmax)*1000 if hmax else 0), i*0.02
if __name__=="__main__":
    runs=sys.argv[1:]
    with Pool(3) as p: res=p.map(ep,[(r,s) for r in runs for s in range(2)])
    for r,l,h,T in res: print(f"{r:14s} lifts>5mm {l:3d} in {T:4.1f} s, median lift {h:4.1f} mm")
