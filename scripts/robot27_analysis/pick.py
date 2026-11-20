"""Round 27: the better run from results/robot27/eval_<run>.txt: mean 'up' over the gait evaluations (all lines) plus
the share of walkers up at the end of the start/stop test (both tests). Prints the best run's name."""
import re, sys
best, score = sys.argv[1], -1.0
for run in sys.argv[1:]:
    try:
        t = open(f"results/robot27/eval_{run}.txt").read()
    except OSError:
        continue
    ups = [int(x) for x in re.findall(r"^\s+up\s+(\d+)%", t, re.M)]
    ends = [int(a) / int(b) * 100 for a, b in re.findall(r"at the end (\d+) of (\d+)", t)]
    if not ups:
        continue
    sc = sum(ups) / len(ups) + (sum(ends) / len(ends) if ends else 0.0)
    print(f"{run}: gait {sum(ups) / len(ups):.0f}, start/stop end {sum(ends) / max(len(ends), 1):.0f} -> {sc:.0f}", file=sys.stderr)
    if sc > score:
        best, score = run, sc
print(best)
