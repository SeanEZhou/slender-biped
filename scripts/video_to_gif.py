"""Robot video -> README gif (2026-11-15): a time window, a crop (fractions of the frame), resized, 10 fps, adaptive
palette. The crops keep the robot and the floor; nothing above the robot's head goes into a public gif.

    .venv-hw/bin/python scripts/video_to_gif.py in.mp4 out.gif --t 5 11.3 --crop 0 0.25 1 1 --width 240 [--fps 10] [--speed 1]
"""
import argparse
import cv2
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--t", type=float, nargs=2, required=True)
ap.add_argument("--crop", type=float, nargs=4, default=(0, 0, 1, 1), metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--width", type=int, default=240)
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()
c = cv2.VideoCapture(a.src); fps = c.get(cv2.CAP_PROP_FPS)
frames = []
for t in np.arange(a.t[0], a.t[1], a.speed / a.fps):
    c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, im = c.read()
    if not ok:
        break
    H, W = im.shape[:2]
    im = im[int(a.crop[1] * H):int(a.crop[3] * H), int(a.crop[0] * W):int(a.crop[2] * W)]
    h = int(round(im.shape[0] * a.width / im.shape[1]))
    im = cv2.resize(im, (a.width, h), interpolation=cv2.INTER_AREA)
    frames.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
pal = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
q[0].save(a.out, save_all=True, append_images=q[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
import os
print(f"{a.out}: {len(q)} frames {q[0].size}, {os.path.getsize(a.out) // 1024} KB")
