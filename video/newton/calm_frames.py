"""Prepare a Newton sim for rendering: cut the yarn at the finger eyelet (the ball side is off-camera) and
low-pass the free yarn far from the hook so small solver jitter does not read as shaking. Points near the
hook are left exactly as simulated.

Usage: python calm_frames.py <in.json> <out.json> <first> <last> "<G x,y,z>" [tau_s=0.15] [near_mm=6]
"""
import json, sys
import numpy as np

src, dst, f0, f1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
G = np.array([float(v) for v in sys.argv[5].split(",")])
tau = float(sys.argv[6]) if len(sys.argv) > 6 else 0.15
near = float(sys.argv[7]) if len(sys.argv) > 7 else 6.0
d = json.load(open(src))
fps = d["fps"]
frames = d["frames"][f0:f1 + 1]
alpha = 1 - np.exp(-1 / (fps * tau))
sm = None
for fr in frames:
    P = np.array(fr["pts"])
    h = fr["hook"]; tip = np.array(h["tip"]); a = np.array(h["a"])
    rel = P - tip; ax = np.clip(rel @ (-a), 0, 60)
    dist = np.linalg.norm(rel + np.outer(ax, a), axis=1)
    w = np.clip((dist - near) / near, 0, 1)[:, None]  # 0 at the hook, 1 well away from it
    sm = P.copy() if sm is None else sm + alpha * (P - sm)
    Q = P * (1 - w) + sm * w
    k = int(np.argmin(np.linalg.norm(P - G, axis=1)))  # the yarn passes the finger here; beyond is the ball side
    fr["pts"] = np.round(Q[:k + 1], 3).tolist(); fr["s"] = fr["s"][:k + 1]
d["frames"] = frames
json.dump(d, open(dst, "w"))
print("frames", len(frames))
