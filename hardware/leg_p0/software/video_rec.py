"""Camera recording alongside a policy_stand run (2026-11-14).

The camera records on the Windows side: a webcam passed into WSL through usbipd loses packets (its isochronous stream
has no retransmit), so every MJPEG frame arrived corrupt. Windows ffmpeg (winget install Gyan.FFmpeg) grabs it over
DirectShow instead and copies the camera's MJPEG frames into a .mkv untouched (no encoding, ~no CPU), each frame
stamped with the Windows wall clock on arrival. The WSL and Windows clocks agree to ~1 ms (measured 2026-11-14), so
stop(t_ref) maps those stamps onto the log's clock and writes <stem>.mkv + <stem>_frames.csv (frame, t: seconds
after t_ref, the log's t = 0; negative before engaging). Frames arrive some tens of ms after the moment they show
(exposure + USB); check against the engage lurch.

    python video_rec.py devices                 # the cameras Windows sees
    python video_rec.py test [--seconds 5]      # a test clip + the frame rate it reached
    python video_rec.py modes --device W1 W2    # the sizes / frame rates each camera offers
    python video_rec.py test --device W1 W2 --fps 60    # both cameras at once
"""
from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import time

import numpy as np
from pathlib import Path, PurePosixPath


def _ffmpeg(tool="ffmpeg"):
    """Windows ffmpeg/ffprobe: on the PATH (a WSL started after the install) or in winget's package folder."""
    p = shutil.which(f"{tool}.exe")
    if p:
        return p
    hits = glob.glob(f"/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/*/bin/{tool}.exe")
    if not hits:
        raise RuntimeError("Windows ffmpeg not found (in PowerShell: winget install Gyan.FFmpeg)")
    return hits[0]


def _win_home() -> Path:
    """The Windows user folder as a WSL path (Windows programs need a Windows working directory)."""
    out = subprocess.run(["cmd.exe", "/c", "echo %USERPROFILE%"], capture_output=True, text=True, cwd="/mnt/c").stdout
    win = out.strip().splitlines()[-1]
    return Path(subprocess.run(["wslpath", "-u", win], capture_output=True, text=True).stdout.strip())


def _winpath(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True).stdout.strip()


class VideoRecorder:
    def __init__(self, path_stem: str | Path, device: str = "W1", width: int = 1280, height: int = 720, fps: int = 30):
        self.stem = Path(path_stem).resolve()
        self.size, self.fps, self.stamps = (width, height), fps, []
        home = _win_home()
        self.tmp_dir = home / "robotvid"            # ffmpeg can't write into \\wsl.localhost; moved afterwards
        self.tmp_dir.mkdir(exist_ok=True)
        self.tmp = self.tmp_dir / (self.stem.name + ".mkv")
        cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "dshow", "-rtbufsize", "200M",
               "-use_wallclock_as_timestamps", "1", "-vcodec", "mjpeg", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", f"video={device}", "-c:v", "copy", "-copyts", _winpath(self.tmp)]
        self.off0 = time.time() - time.perf_counter()  # wall clock minus the log's clock
        # 2026-11-14: ffmpeg's messages go to a file. A pipe nobody read filled up with dshow's buffer warnings and
        # blocked ffmpeg: two videos stopped early (one 1.3 s after engaging, one 17 s before)
        self.errlog = self.tmp_dir / (self.stem.name + "_ffmpeg.log")
        self._err = self.errlog.open("wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err,
                                     cwd=str(home))
        time.sleep(0.3)
        if self.proc.poll() is not None:
            self._err.close()
            raise RuntimeError(f"ffmpeg stopped: {self.errlog.read_text(errors='ignore').strip()[-300:]}")

    def stop(self, t_ref: float, tail: float = 0.0) -> Path:
        """t_ref: the log's t = 0 (time.perf_counter()). Returns the frames .csv."""
        if tail > 0:
            time.sleep(tail)
        off = 0.5 * (self.off0 + time.time() - time.perf_counter())
        early = self.proc.poll() is not None              # ffmpeg ended by itself: the video stops short
        try:
            self.proc.communicate(b"q", timeout=8)     # ffmpeg's own stop key: closes the file cleanly
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.communicate()
        self._err.close()
        msg = self.errlog.read_text(errors="ignore").strip()
        if early or msg:
            print(f"  video: ffmpeg {'stopped early' if early else 'reported'}: " + " | ".join(msg.splitlines()[-3:]))
        self.errlog.unlink(missing_ok=True)
        mkv = self.stem.with_suffix(".mkv")
        if not self.tmp.exists():
            raise RuntimeError(f"no video file: the camera did not record ({msg.splitlines()[-1] if msg else 'no message'})")
        try:
            shutil.move(str(self.tmp), mkv)
        except OSError as e:                          # still locked by a camera that hung
            raise RuntimeError(f"could not move {self.tmp.name}: {e}")
        pts = subprocess.run([_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "v", "-show_entries",
                              "packet=pts_time", "-of", "csv=p=0", _winpath(mkv)],
                             capture_output=True, text=True, cwd=str(self.tmp_dir)).stdout.split()
        wall_ref = t_ref + off
        self.stamps = [float(x) for x in pts if x.strip()]
        if self.stamps and self.stamps[-1] < time.time() - tail - 1.0:
            print(f"  video: the last frame is {time.time() - tail - self.stamps[-1]:.1f} s before the stop: the recording stopped short")
        out = Path(str(self.stem) + "_frames.csv")
        with out.open("w") as f:
            f.write("frame,t\n")
            for i, t in enumerate(self.stamps):
                f.write(f"{i},{t - wall_ref:.4f}\n")
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["test", "devices", "modes"])
    ap.add_argument("--device", nargs="+", default=["W1"], help="one or more cameras (test: all record at once)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    a = ap.parse_args()
    if a.cmd == "devices":
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                           capture_output=True, text=True, cwd=str(_win_home()))
        print("\n".join(l for l in r.stderr.splitlines() if "(video)" in l))
        return
    if a.cmd == "modes":                          # 2026-11-21: the sizes and frame rates each camera offers (MJPEG)
        for dev in a.device:
            r = subprocess.run([_ffmpeg(), "-hide_banner", "-list_options", "true", "-f", "dshow", "-i", f"video={dev}"],
                               capture_output=True, text=True, cwd=str(_win_home()))
            print(f"{dev}:")
            print("\n".join(l for l in r.stderr.splitlines() if "vcodec=mjpeg" in l) or r.stderr[-500:])
        return
    w, h = (int(x) for x in a.size.split("x"))
    t0 = time.perf_counter()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    recs = []
    for k, c in enumerate(a.device):                # DEVICE[@WxH[@fps]]: a mode per camera
        dev, *mode = c.split("@")
        w_, h_ = (int(x) for x in mode[0].split("x")) if mode else (w, h)
        recs.append(VideoRecorder(f"video_test_{stamp}" + (f"_{k}" if len(a.device) > 1 else ""), dev, w_, h_,
                                  int(mode[1]) if len(mode) > 1 else a.fps))
    a.device = [c.split("@")[0] for c in a.device]
    time.sleep(a.seconds)
    for dev, rec in zip(a.device, recs):
        try:
            out = rec.stop(t0)
        except RuntimeError as e:
            print(f"{dev}: FAILED: {e}")
            continue
        s = rec.stamps
        rate = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 else 0.0
        gaps = max(np.diff(s)) * 1000 if len(s) > 2 else float("nan")
        print(f"{dev}: {len(s)} frames {rec.size[0]}x{rec.size[1]}, {rate:.1f} fps (longest gap {gaps:.0f} ms); first frame "
              f"at t = {s[0] - (t0 + rec.off0):.2f} s after start -> {rec.stem.with_suffix('.mkv')}, {out}")


if __name__ == "__main__":
    main()
