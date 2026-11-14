import sys; sys.path.insert(0,'.')
import numpy as np, replay
S="hardware/leg_p0/software/"
G={"g":slice(0,3),"gyro":slice(3,6),"q":slice(6,16),"qd":slice(16,26),"last_a":slice(26,36)}
def lp(x, fc=2.5, dt=0.02):
    fr=np.fft.rfftfreq(len(x),dt); X=np.fft.rfft(x,axis=0); X[fr>fc]=0; return np.fft.irfft(X,len(x),axis=0)
def band(a, dt=0.02):
    fr=np.fft.rfftfreq(len(a),dt); X=np.fft.rfft(a-a.mean(0),axis=0); X[(fr<3.5)|(fr>=8)]=0; return np.sqrt(np.mean(np.fft.irfft(X,len(a),axis=0)**2))
for run,f in [(a.split(":")[0],S+a.split(":")[1]) for a in sys.argv[1:]]:
    pol,O,A,t=replay.build(run,f); O=O[25:-25]
    P=lambda O: np.array([np.clip(pol(o),-1,1) for o in O])
    base=band(P(O)); out=[]
    for k,s in G.items():
        Ok=O.copy(); Ok[:,s]=lp(O[:,s]); out.append(f"{k} {100*(1-band(P(Ok))/base):+3.0f}%")
    Oall=O.copy()
    for s in G.values(): Oall[:,s]=lp(O[:,s])
    print(f"{run:10s} {f[-19:-4]} act 3.5-8Hz rms {base:.3f} | smoothing one input cuts it by: "+"  ".join(out)+f" | all: {100*(1-band(P(Oall))/base):+3.0f}%")
