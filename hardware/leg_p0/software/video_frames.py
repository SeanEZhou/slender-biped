"""Contact sheets from a policy_stand run's video (2026-11-14): frames at chosen moments of the log, each labelled with
the log's time, gait clock, roll / pitch and the gate's predicted roll.

    python video_frames.py stand_<run>_<time>.csv --t 1.0 2.5 3.1        # frames at these log times (s)
    python video_frames.py stand_<run>_<time>.csv --liftoffs              # at each clock lift-off (u 0.208 / 0.708)
    python video_frames.py stand_<run>_<time>.csv --span 6.0 8.0 --step 0.1   # every 0.1 s over a window
    options: --crop X0 Y0 X1 Y1 (fractions), --lag 0.0 (s the camera lags the log, from the engage lurch), --cols 4, --width 480, --out sheet.jpg
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def load(log: Path):
    r = list(csv.reader(log.open()))
    h, A = r[0], np.array(r[1:], float)
    fr = np.loadtxt(str(log.with_suffix("")) + "_frames.csv", delimiter=",", skiprows=1)
    return h, A, fr[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--t", type=float, nargs="*", default=[])
    ap.add_argument("--liftoffs", action="store_true")
    ap.add_argument("--span", type=float, nargs=2)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="keep this part of the frame (fractions of width / height), e.g. the feet")
    ap.add_argument("--out")
    a = ap.parse_args()
    log = Path(a.log)
    h, A, ft = load(log)
    t = A[:, 0]
    col = lambda k: A[:, h.index(k)] if k in h else np.zeros(len(t))  # noqa: E731
    g = A[:, [h.index(k) for k in ("gx", "gy", "gz")]]
    roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2]))
    pitch_fwd = -np.degrees(np.arctan2(-g[:, 0], -g[:, 2]))
    u = col("phase") / (2 * math.pi)
    rp = col("roll_pred")
    times = list(a.t)
    if a.liftoffs:
        for i in range(1, len(t)):
            if any(((u[i - 1] - gu) % 1.0) > ((u[i] - gu) % 1.0) for gu in (0.2083, 0.7083)):
                times.append(t[i])
    if a.span:
        times += list(np.arange(a.span[0], a.span[1] + 1e-9, a.step))
    if not times:
        raise SystemExit("no times: give --t, --liftoffs or --span")
    cap = cv2.VideoCapture(str(log.with_suffix(".mkv")))
    tiles = []
    for tt in times:
        k = int(np.argmin(np.abs(ft - (tt + a.lag))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        if not ok:
            continue
        if a.crop:
            H, W = img.shape[:2]
            img = img[int(a.crop[1] * H):int(a.crop[3] * H), int(a.crop[0] * W):int(a.crop[2] * W)]
        hh = int(img.shape[0] * a.width / img.shape[1])
        img = cv2.resize(img, (a.width, hh))
        i = int(np.argmin(np.abs(t - tt)))
        lab = f"t {tt:.2f}  u {u[i]:.2f}  roll {roll[i]:+.1f}  fwd {pitch_fwd[i]:+.1f}  pred {rp[i]:+.1f}"
        cv2.rectangle(img, (0, 0), (a.width, 22), (0, 0, 0), -1)
        cv2.putText(img, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + a.cols]) for r in range(0, len(tiles), a.cols)])
    out = a.out or str(log.with_suffix("")) + "_sheet.jpg"
    cv2.imwrite(out, sheet)
    print(f"{len(times)} frames -> {out}")


if __name__ == "__main__":
    main()
