import sys, os, numpy as np, csv
os.environ["JAX_PLATFORMS"]="cpu"; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
from pathlib import Path
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
def bandrms(x, dt=0.02, a=3.5, b=6.5):
    fr=np.fft.rfftfreq(len(x),dt); X=np.fft.rfft(x-x.mean(0),axis=0); X[(fr<a)|(fr>=b)]=0; return np.sqrt(np.mean(np.fft.irfft(X,len(x),axis=0)**2,0))
def report(name, W, C):
    w=bandrms(W); c=bandrms(C); kn=np.sqrt(c[3]**2+c[8]**2); hr=np.sqrt(c[1]**2+c[6]**2+c[4]**2+c[9]**2)
    print(f"{name:34s} gyro 3.5-6.5Hz rms r/p/y {w[0]:5.1f} {w[1]:5.1f} {w[2]:5.1f} deg/s | cmd knees {kn:4.2f} roll-joints {hr:4.2f} deg | pitch/knees {w[1]/kn:5.1f}  roll/roll-joints {w[0]/hr:5.1f}")
if __name__=="__main__":
    S="hardware/leg_p0/software/"
    for f in ["stand_h2_old_20261113_101113.csv","stand_h4_t05_20261113_101029.csv","stand_fb2_lean_20261112_212222.csv","stand_b7_seed2_20261113_004527.csv"]:
        rows=list(csv.reader(open(S+f))); h=rows[0]; A=np.array(rows[1:],float); s=A[:,0]>2.5
        c=lambda ks: A[s][:,[h.index(k) for k in ks]]
        report("robot "+f[6:-20], np.degrees(c(["wx","wy","wz"])), np.degrees(c([f"cmd_{n}" for n in J])))
    import eval6
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of
    for run, model, sim in [("h2_old","hw_ank3_b150_m2_10dof","fitted"),("h4_t05","hw_ank3_b150_m2_10dof","fitted"),("h2_old","hw_ank3_b150_m2_10dof","hard"),("b7_seed2","hw_ank3_b150_legs_pelvis_m2_10dof","fitted")]:
        cfg, rd = load_run(Path("runs")/run); p = rd/"params.pkl" if (rd/"params.pkl").exists() else sorted(rd.glob("params_*M.pkl"))[-1]
        pol=NumpyPolicy(p)
        over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0, dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25, dr_kp=(0.75, 1.25), kp_hip_roll=2.5, drag_back=0.0)
        over.update(eval6.SIMS[sim])
        env=CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=81000); obs=env.reset((0.015,0,0)); y0=yaw_of(env.d.qpos[3:7]); W=[];C=[]
        for i in range(1000):
            env.cmd[2]=float(np.clip((y0-yaw_of(env.d.qpos[3:7])+np.pi)%(2*np.pi)-np.pi,-.3,.3)); obs[-5:-2]=env.cmd
            obs,_=env.step(pol(obs)); W.append(np.degrees(env.d.qvel[3:6])); C.append(np.degrees(env.cmd_buf[0]))
            if env.fallen(): break
        report(f"sim {run} {sim}", np.array(W[25:]), np.array(C[25:]))
