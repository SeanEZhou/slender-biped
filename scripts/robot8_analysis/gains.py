"""Static gains of a policy on a robot log's observations (h4's 188 s walk by default): the change of the hip-pitch and
knee commands per degree of body pitch (tilt, through the gravity input) and per deg/s of pitch rate (gyro). Positive
tilt = leaning back. 'stiff' is the sagittal posture answer to a forward lean: d(knee - hip pitch)/d(tilt), deg/deg.

    PYTHONPATH=scripts/robot7_analysis .venv/bin/python scripts/robot8_analysis/gains.py h4_t05 r10_lcp_h4 ...
"""
import sys, json, math, numpy as np
import replay
LOG = "hardware/leg_p0/software/stand_h4_t05_20261113_190622.csv"
for run in sys.argv[1:]:
    run, _, f = run.partition(":")
    pol, O, A, t = replay.build(run, f or LOG)
    sc = json.load(open(f"runs/{run}/config.json"))["env"]["action_scale"]
    act = lambda o: np.clip(pol(o), -1, 1) * sc  # noqa: E731
    e = math.radians(1.0); G, W = [], []
    for o in O[50::10]:
        g = o[0:3]; c, s = math.cos(e), math.sin(e)
        rot = lambda th: np.array([g[0] * math.cos(th) - g[2] * math.sin(th), g[1], g[0] * math.sin(th) + g[2] * math.cos(th)])  # noqa: E731
        op, om = o.copy(), o.copy(); op[0:3] = rot(e); om[0:3] = rot(-e)
        G.append((act(op) - act(om)) / 2)            # rad per deg of tilt about y (sign checked below)
        op, om = o.copy(), o.copy(); op[4] += 0.02; om[4] -= 0.02
        W.append((act(op) - act(om)) / 0.04 * math.radians(1.0))   # rad per deg/s
    G, W = np.degrees(np.mean(G, 0)), np.degrees(np.sqrt(np.mean(np.array(W) ** 2, 0)))
    # which sign of the gravity rotation is "leaning back"? pitch = atan2(-gx, -gz) as in the logs
    o = O[100]; g = o[0:3]; th = e
    gp = np.array([g[0] * math.cos(th) - g[2] * math.sin(th), g[1], g[0] * math.sin(th) + g[2] * math.cos(th)])
    sgn = np.sign(math.atan2(-gp[0], -gp[2]) - math.atan2(-g[0], -g[2]))
    G *= sgn
    hip, knee = G[[2, 7]].mean(), G[[3, 8]].mean()
    print(f"{run:16s} per deg of back tilt: hip pitch {hip:+.3f} knee {knee:+.3f} (knee-hip {knee-hip:+.3f}) deg/deg | "
          f"pitch-rate gain rms hip {W[[2,7]].mean():.3f} knee {W[[3,8]].mean():.3f} deg per deg/s")
