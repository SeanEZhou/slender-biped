"""Regenerate full (eval) and lean-contact (train) models for every configuration used in experiments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.generate_model import STANDARD, generate  # noqa: E402

EXTRA = [("long_shin", "A", False), ("slim_dynamixel", "A", False)]
for v, f, ank in STANDARD + EXTRA:
    generate(v, f, ank)
    generate(v, f, ank, overrides={"contact_set": "train"})
    print(v, f, ank)

# Control-experiment frame (flat feet); separate from the ball-tip baseline.
FOOTED = [("footed", "F40", True), ("footed", "F60", True), ("footed", "F90", True)]
for v, f, ank in FOOTED:
    generate(v, f, ank)
    generate(v, f, ank, overrides={"contact_set": "train"})
    print(v, f, ank)
# rigid flat foot, no ankle (is it the foot or the ankle?)
for ov in ({}, {"contact_set": "train"}):
    generate("footed", "F60", False, actuator_set="baseline", overrides=ov)
print("footed F60 rigid")

# flat-tipped leg (ball-tip form factor, no ankle): does contact area alone help?
for f in ("P20", "P30"):
    for ov in ({}, {"contact_set": "train"}):
        generate("baseline", f, False, overrides=ov)
print("flat pads")
