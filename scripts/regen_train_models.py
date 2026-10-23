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
