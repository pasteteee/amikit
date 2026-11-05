"""Frame-by-frame checks for a yarn animation JSON ({fps, r, frames:[{pts, s, hook}]}).

  jump   : the same bit of yarn (same material coordinate s) moves sideways more per frame than the hook could
           push it (hook speed + 0.5 mm near the hook, 0.5 mm elsewhere), or slides along itself faster than
           the hook can pull it (2x hook speed + 1 mm): yarn that teleports;
  far    : yarn more than 6 mm from the hook whose sideways motion differs from that of the yarn 3 mm either
           side of it by more than 0.35 mm/frame: a free span may swing as a whole when its ends move (the
           hook pulls, the finger moves), but a bit of it must not move on its own ("lives its own life");
  self   : two bits of yarn more than 4 mm apart along the strand closer than SELF yarn diameters
           (interpenetration; yarn is soft, --self sets the allowed squash, default 0.9);
  hook   : yarn centreline inside the hook body (the renderer would show yarn through the hook).
Usage: python check_anim.py <anim.json> [--every N]
"""
import json, math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from chain_anim import Pose, surf, R  # same hook geometry as the animation and the renderer


def at_s(P, S, s):
    return np.stack([np.interp(s, S, P[:, k]) for k in range(3)], axis=1)


def hook_dist(H, P):
    L = H.local(P)
    x, y, z = L[:, 0], L[:, 1], L[:, 2]
    rr = np.hypot(y, z)
    need = np.array([surf(xi, math.atan2(zi, yi)) if -60 < xi <= 0 else 0.0 for xi, yi, zi in L])
    pen = np.where((x <= 0) & (x > -60), need + R * 0.9 - rr, 0.0)
    radial = np.where(x <= 0, np.maximum(0, rr - need), np.linalg.norm(L, axis=1))
    return pen, radial


SELF = float(sys.argv[sys.argv.index("--self") + 1]) if "--self" in sys.argv else 0.9


def main():
    d = json.load(open(sys.argv[1])); fr = d["frames"]; r = d["r"]
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    worst = dict(jump=(0, None), far=(0, None), self=(9, None), hook=(0, None))
    counts = dict(jump=0, far=0, self=0, hook=0)
    prev = None
    for i in range(0, len(fr), every):
        f = fr[i]; P = np.array(f["pts"]); S = np.array(f["s"]); h = f["hook"]
        H = Pose(h["tip"], h["a"], h["d"])
        pen, radial = hook_dist(H, P)
        if pen.max() > 0.05:
            counts["hook"] += 1
            if pen.max() > worst["hook"][0]: worst["hook"] = (pen.max(), i)
        # self contact: sample every ~0.6 mm
        idx = np.arange(0, len(P), 2); Q = P[idx]; Sq = S[idx]
        DD = np.linalg.norm(Q[:, None] - Q[None], axis=2)
        far_along = np.abs(Sq[:, None] - Sq[None]) > 4.0
        dm = np.where(far_along, DD, 9).min()
        if dm < 2 * r * SELF:
            counts["self"] += 1
            if dm < worst["self"][0]:
                a_, b_ = np.unravel_index(np.argmin(np.where(far_along, DD, 9)), DD.shape)
                worst["self"] = (dm, (i, round(Sq[a_], 1), round(Sq[b_], 1), Q[a_].round(1).tolist()))
        if prev is not None:
            P0, S0, H0 = prev
            s_common = S[(S <= S0[-1])]
            A = at_s(P0, S0, s_common); B = P[: len(s_common)]
            hv = np.linalg.norm(H.tip - H0.tip) + 2.0 * np.linalg.norm(H.a - H0.a) * 10
            _, rad0 = hook_dist(H0, A)
            near = rad0 < 3.0
            tg = np.gradient(B, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
            dv = B - A; along = np.abs((dv * tg).sum(1)); side = np.linalg.norm(dv - tg * (dv * tg).sum(1, keepdims=True), axis=1)
            lim = np.where(near, hv + 0.5, 0.5)
            mv = np.maximum(side - lim, along - (2 * hv + 1.0)) + lim
            bad = mv > lim
            if bad.any():
                counts["jump"] += 1
                k = int(np.argmax(mv - lim))
                if mv[k] - lim[k] > worst["jump"][0]: worst["jump"] = (mv[k] - lim[k], (i, round(s_common[k], 1), round(mv[k], 2)))
            # a free span may swing as a whole: only motion that differs from its neighbours' counts
            sv = dv - tg * (dv * tg).sum(1, keepdims=True)
            ds = np.median(np.diff(s_common)) if len(s_common) > 2 else 0.36
            k = max(1, int(round(3.0 / max(ds, 1e-3))))
            lo = np.clip(np.arange(len(sv)) - k, 0, len(sv) - 1); hi = np.clip(np.arange(len(sv)) + k, 0, len(sv) - 1)
            mv = np.linalg.norm(sv - 0.5 * (sv[lo] + sv[hi]), axis=1)
            far = (rad0 > 6.0) & (mv > 0.35)
            if far.any():
                counts["far"] += 1
                k = int(np.argmax(np.where(rad0 > 6.0, mv, 0)))
                if mv[k] > worst["far"][0]: worst["far"] = (mv[k], (i, round(s_common[k], 1), round(mv[k], 2)))
        prev = (P, S, H)
    n = len(range(0, len(fr), every))
    for k in counts:
        print(f"{k:5s}: {counts[k]:4d}/{n} frames   worst {worst[k]}")
    ok = all(v == 0 for v in counts.values())
    print("ALL CLEAN" if ok else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
