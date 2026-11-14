import sys, numpy as np, replay
S="hardware/leg_p0/software/"
for a in sys.argv[1:]:
    run,f=a.split(":"); pol,O,A,t=replay.build(run,S+f)
    raw=np.array([np.clip(pol(o),-1,1) for o in O])
    # a_t = k a_{t-1} + (1-k) raw_t  ->  (a_t - raw_t) = k (a_{t-1} - raw_t)
    y=(A[1:]-raw[1:]).ravel(); x=(A[:-1]-raw[1:]).ravel(); k=float(x@y/(x@x))
    print(f"{f[6:-4]:30s} action low-pass k = {k:.2f}  (residual {np.abs(y-k*x).mean():.4f})")
