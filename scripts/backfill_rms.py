"""One-off: add rms_tau_frac / rms_over_rated to eval files written before those fields existed."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slender_biped.config import load_actuators  # noqa: E402

act = load_actuators()
for f in glob.glob(str(ROOT / "results/eval/*.json")):
    r = json.load(open(f))
    cfg = json.load(open(ROOT / f"runs/{r['run']}/config.json"))["env"]
    meta = json.load(open(ROOT / f"models/{cfg['model']}.json"))
    specs = [act["catalog"][act["assignments"][meta["cfg"]["actuator_set"]][j.split("_", 1)[1]]] for j in meta["joints"]]
    ts = np.array([x["stall_torque"] for x in specs]) * cfg.get("strength_scale", 1.0)
    rated = np.array([x.get("rated_torque") or np.nan for x in specs])
    for e in r["episodes"]:
        e["rms_tau_frac"] = (np.array(e["rms_tau"]) / ts).tolist()
        e["rms_over_rated"] = (np.array(e["rms_tau"]) / rated).tolist()
    s = r["summary"]
    s["rms_tau_frac_median"] = np.median([e["rms_tau_frac"] for e in r["episodes"]], 0).round(3).tolist()
    s["rms_over_rated_median"] = np.median([e["rms_over_rated"] for e in r["episodes"]], 0).round(3).tolist()
    json.dump(r, open(f, "w"), indent=1)
print("backfilled")
