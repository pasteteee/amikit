"""Crochet chain, hand-animated hook + quasi-static yarn (no dynamics, so nothing shakes or swings by itself).

The hook motion is keyframed by hand. The yarn is one strand from the tail (held by the other hand) to the
finger eyelet G, solved every sub-step as a taut string: it is pulled tight (tension), it can not go through the
hook or through itself, and it slides along itself (fed from the finger) to get the length it needs. The only
hand-authored help is the throat: while a stitch is being pulled through, the yarn that lies in the throat is held
in it (the job the crocheter's tension and the hook's lip do).

Because it is solved from the previous sub-step and moves only by small steps, the yarn can not jump, appear or pass
through anything; check_anim.py verifies the output frame by frame.

Usage: python qs_chain.py <out.json> [n_stitches=4] [skip_first=1]
"""
import json, math, sys
import numpy as np
from scipy.spatial import cKDTree

R = 0.95                      # yarn radius
RS, RNECK, XTH = 1.06, 0.51, -2.05
H_SP = 0.36                   # point spacing along the yarn
FPS = 30
SUB = 4                       # sub-steps per frame
ITERS = 14                    # solver iterations per sub-step
TAU = 0.22                    # tension (curve shortening) per iteration
NEIGH = int(4.2 / H_SP)       # points closer than this along the yarn never collide


def norm(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / (n or 1.0)


def ease(u):
    u = min(1.0, max(0.0, u)); return u * u * u * (u * (u * 6 - 15) + 10)


def rot(v, axis, ang):
    axis = norm(axis); v = np.asarray(v, float)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))


class Pose:
    def __init__(self, tip, a, d):
        self.tip = np.asarray(tip, float); self.a = norm(a)
        self.d = norm(np.asarray(d, float) - self.a * np.dot(d, self.a)); self.s = np.cross(self.a, self.d)

    def pt(self, x, y=0.0, z=0.0):
        return self.tip + self.a * x + self.d * y + self.s * z

    def local(self, p):
        q = np.asarray(p) - self.tip
        return np.stack([q @ self.a, q @ self.d, q @ self.s], axis=-1)

    def world(self, L):
        return self.tip + np.outer(L[:, 0], self.a) + np.outer(L[:, 1], self.d) + np.outer(L[:, 2], self.s)

    def lerp(self, o, w):
        return Pose(self.tip + (o.tip - self.tip) * w, slerp(self.a, o.a, w), slerp(self.d, o.d, w))

    def js(self):
        return dict(tip=self.tip.round(4).tolist(), a=self.a.round(5).tolist(), d=self.d.round(5).tolist())


def slerp(u, v, w):
    u, v = norm(u), norm(v); c = float(np.clip(np.dot(u, v), -1, 1)); th = math.acos(c)
    if th < 1e-6: return u
    return (u * math.sin((1 - w) * th) + v * math.sin(w * th)) / math.sin(th)


# ---------------------------------------------------------------- hook body (same as engine/hook.mjs)
def base_radius(x):
    x = np.asarray(x, float)
    u = (x + 1.15) / 1.15
    nose = RS * 0.92 * np.sqrt(np.clip(1 - u * u, 0, None)) + 0.02
    waist = 0.07 * np.exp(-(((x + 1.4) / 0.5) ** 2))
    body = np.where(x > -1.6, RS * 0.92, RS) - waist
    return np.where(x > 0, 0.0, np.where(x > -1.15, nose, body))


def groove(x):
    x = np.asarray(x, float)
    x0, xm, x1 = -4.4, -2.0, -1.45; Dg = RS - RNECK
    t1 = np.clip((x - x0) / (xm - x0), 0, 1); t2 = np.clip((x - xm) / (x1 - xm), 0, 1)
    g = np.where(x < xm, Dg * t1 * t1 * (3 - 2 * t1), Dg * (1 - t2 ** 3))
    return np.where((x <= x0) | (x >= x1), 0.0, g)


def surf(x, phi):
    c = np.clip(np.cos(phi), 0, None)
    return np.maximum(0.05, base_radius(x) - groove(x) * np.minimum(1.0, c ** 0.7 * 1.15))


def hook_push(H, P, ry=R):
    L = H.local(P); x, y, z = L[:, 0], L[:, 1], L[:, 2]
    rr = np.hypot(y, z); phi = np.arctan2(z, y)
    need = surf(x, phi) + ry * 0.98
    on_body = (x <= 0) & (x > -80) & (rr < need)
    sc = np.where(rr > 1e-6, need / np.maximum(rr, 1e-6), 1.0)
    L2 = L.copy()
    L2[on_body, 1] = np.where(rr[on_body] > 1e-6, y[on_body] * sc[on_body], -need[on_body])
    L2[on_body, 2] = np.where(rr[on_body] > 1e-6, z[on_body] * sc[on_body], 0.0)
    rt = np.linalg.norm(L, axis=1); cap = (x > 0) & (rt < ry * 0.98)
    L2[cap] = L[cap] / np.maximum(rt[cap], 1e-6)[:, None] * ry * 0.98
    moved = on_body | cap
    if moved.any(): P[moved] = H.world(L2[moved])
    return P


def throat_hold(H, P, w=0.9, sel=None):
    """Yarn caught by the hook stays at the throat: anything wrapped round the head/neck is kept in the throat's
    length (the lip stops it sliding off the tip), and on the throat side it sits down in the groove."""
    L = H.local(P); x, y, z = L[:, 0], L[:, 1], L[:, 2]
    rr = np.hypot(y, z); phi = np.arctan2(z, y)
    m = (x > -4.3) & (x < 1.2) & (rr < RS + R + 1.0)
    if sel is not None: m &= sel
    if not m.any(): return m
    L2 = L.copy()
    L2[m, 0] = x[m] + w * (np.clip(x[m], XTH - 0.5, XTH + 0.5) - x[m])
    side = m & (np.cos(phi) > 0.3)
    want = RNECK + R
    f = np.where(rr[side] > want, (rr[side] + w * (want - rr[side])) / np.maximum(rr[side], 1e-6), 1.0)
    L2[side, 1] = y[side] * f; L2[side, 2] = z[side] * f
    P[m] = H.world(L2[m])
    return m


# ---------------------------------------------------------------- yarn solver
def resample(P, h=H_SP, keep_last=True):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / h)) + 1)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], axis=1)


def self_collide(P, w=None, iters=1):
    d0 = 2 * R
    for _ in range(iters):
        pairs = cKDTree(P).query_pairs(d0, output_type='ndarray')
        if len(pairs) == 0: return P
        pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > NEIGH]
        if len(pairs) == 0: return P
        i, j = pairs[:, 0], pairs[:, 1]
        v = P[i] - P[j]; d = np.linalg.norm(v, axis=1)
        nrm = v / np.maximum(d, 1e-6)[:, None]
        corr = (d0 - d)[:, None] * nrm
        wi = np.ones(len(i)) * 0.5 if w is None else w[i] / np.maximum(w[i] + w[j], 1e-9)
        wj = 1 - wi if w is not None else wi
        acc = np.zeros_like(P); cnt = np.zeros(len(P))
        np.add.at(acc, i, corr * wi[:, None]); np.add.at(acc, j, -corr * wj[:, None])
        np.add.at(cnt, i, 1); np.add.at(cnt, j, 1)
        k = cnt > 0
        P[k] += acc[k] / np.maximum(1, cnt[k] ** 0.5)[:, None] * 1.1
    return P


def tension(P, tau, step_max=0.15):
    """Multi-scale curve shortening: free spans straighten in a few iterations, not N^2."""
    n = len(P); disp = np.zeros_like(P)
    for k in (1, 2, 4, 8, 16):
        if 2 * k >= n: break
        i = np.arange(1, n - 1)
        lo = np.maximum(i - k, 0); hi = np.minimum(i + k, n - 1)
        wl = (i - lo) / k; wh = (hi - i) / k  # shorter arm near the ends: weight the midpoint properly
        mid = (P[lo] * wh[:, None] + P[hi] * wl[:, None]) / (wl + wh)[:, None]
        disp[1:-1] += tau * (mid - P[1:-1]) / (1 + 0.5 * math.log2(k))
    m = np.linalg.norm(disp, axis=1); sc = np.minimum(1, step_max / np.maximum(m, 1e-9))
    return P + disp * sc[:, None]


def caught_curve(n, sign):
    """The caught yarn hugging the throat (hook-local): round the neck on the throat side."""
    ph = np.radians(np.linspace(-95, 95, n)) * sign
    x = np.full(n, XTH)
    r = surf(x, ph) + R + 0.03
    return np.stack([x, r * np.cos(ph), r * np.sin(ph)], axis=1)


def seated_curve(n, sign, w):
    """Caught yarn -> loop round the shaft (w: 0..1), opening toward the legs (away from the throat)."""
    span = 95 + (172 - 95) * w
    ph = np.radians(np.linspace(-span, span, n)) * sign
    x = np.full(n, XTH + (-6.0 - XTH) * w)
    r = np.maximum(surf(XTH, ph) + R + 0.03 + (RS - RNECK) * w * 0, surf(x, ph) + R + 0.05)
    return np.stack([x, r * np.cos(ph), r * np.sin(ph)], axis=1)


def curve_len(C):
    return float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))


def resample_n(C, n):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, C[:, k]) for k in range(3)], axis=1)


class Yarn:
    """att: the caught yarn, carried by the hook as a piece of the strand (indices i0..i1) whose hook-local shape
    is authored; the rest of the strand is solved."""
    def __init__(self, P, anchor, G):
        self.P = resample(np.asarray(P, float)); self.anchor = np.asarray(anchor, float); self.G = np.asarray(G, float)
        self.att = None; self.H = None
        self.frozen = np.zeros((0, 3))  # finished work held by the other hand: rigid, from the tail end up to anchor

    def freeze(self, f):
        """Everything up to sim index f becomes finished work (held by the hand)."""
        self.frozen = np.vstack([self.frozen, self.P[:f]])
        self.P = self.P[f:].copy(); self.anchor = self.P[0].copy()
        if self.att is not None: self.att['i0'] -= f; self.att['i1'] -= f

    def move_work(self, delta, rot_axis=None, ang=0.0):
        F = self.frozen + delta; a = self.anchor + delta
        if ang:
            k = norm(rot_axis); V = F - a; c, s_ = math.cos(ang), math.sin(ang)
            F = a + V * c + np.cross(k, V) * s_ + np.outer(V @ k, k) * (1 - c)
        self.frozen = F; self.anchor = a; self.P[0] = self.anchor

    def target(self, mode, u):
        a = self.att
        if mode == 'catch':
            n = a['n0']; U = caught_curve(n, a['sign'])
            return a['L0'] + (U - a['L0']) * ease(u)
        if mode == 'hold':
            return caught_curve(a['n0'], a['sign'])
        C = seated_curve(64, a['sign'], 1.0 if mode == 'ring' else ease(u))  # 'seat' / 'ring'
        n = max(4, int(round(curve_len(C) / H_SP)) + 1)
        return resample_n(C, n)

    def step(self, H, mode=None, u=0.0, iters=ITERS, tau=TAU, stick=False, slip=False, hand=None):
        P = self.P
        held_loop = None
        if slip and self.H is not None:
            # the old loop is held by the other hand: the hook slides through it, it does not travel with the hook
            Lh = self.H.local(P); rrh = np.hypot(Lh[:, 1], Lh[:, 2])
            held_loop = (Lh[:, 0] < 0.8) & (Lh[:, 0] > -12) & (rrh < RS + R + 2.8)
            if self.att is not None: held_loop[self.att['i0']:self.att['i1'] + 1] = False
            X0 = P[held_loop].copy(); ax = self.H.a.copy()
        if stick and self.H is not None:
            # a loop on the shaft does not slide along it by itself (friction; the finger keeps it there)
            L0 = self.H.local(P); rr0 = np.hypot(L0[:, 1], L0[:, 2])
            on = (L0[:, 0] < -3.6) & (rr0 < RS + R + 0.35)
            x0 = L0[on, 0]
        else:
            on = None
        self.H = H
        if mode is None or (mode == 'catch' and self.att is not None and self.att.get('kind') == 'ring'):
            self.att = None  # the loop on the hook is let go: it is just yarn round the shaft now
        if mode == 'ring' and self.att is not None:
            self.att['kind'] = 'ring'
        if mode is not None and self.att is None:  # catch: the bit of yarn lying across the throat
            tb = H.pt(XTH, RNECK + R)
            K = int(round(2.7 / H_SP))
            dd = np.linalg.norm(P - tb, axis=1)
            lo = K + 2
            if getattr(self, 'catch_hint', None) is not None: lo = max(lo, self.catch_hint - 25)
            c = lo + int(np.argmin(dd[lo:len(P) - K - 2]))
            i0, i1 = c - K, c + K
            Lc = H.local(P[i0:i1 + 1])
            sign = 1.0 if np.arctan2(Lc[0, 2], Lc[0, 1]) < np.arctan2(Lc[-1, 2], Lc[-1, 1]) else -1.0
            self.att = dict(i0=i0, i1=i1, n0=i1 - i0 + 1, L0=Lc, sign=sign)
        for it in range(iters):
            P[0] = self.anchor; P[-1] = self.G
            P = tension(P, tau)
            P = hook_push(H, P)
            w = None
            if self.att is not None:
                w = np.ones(len(P)); w[self.att['i0']:self.att['i1'] + 1] = 1e-6
            nf = min(len(self.frozen), 90)
            if nf:
                Q = np.vstack([self.frozen[len(self.frozen) - nf:], P])
                wq = np.concatenate([np.full(nf, 1e-6), np.ones(len(P)) if w is None else w])
                P = self_collide(Q, wq, iters=4)[nf:]
            else:
                P = self_collide(P, w, iters=4)
            P = hook_push(H, P)
            if hand is not None and getattr(self, 'hand_rng', None) is not None:
                j0, j1 = self.hand_rng
                if 0 < j0 < j1 < len(P):
                    dh = 0.15 * (hand - P[j0:j1 + 1].mean(axis=0)); P[j0:j1 + 1] += dh * min(1.0, 0.3 / max(np.linalg.norm(dh), 1e-9))
            if held_loop is not None and held_loop.any():
                dX = P[held_loop] - X0
                P[held_loop] = X0 + dX - np.outer(dX @ ax, ax)
            if on is not None and on.any() and len(P) == len(on):
                L = H.local(P[on]); L[:, 0] = x0; P[on] = H.world(L)
            if self.att is not None:
                T = H.world(self.target(mode, u))
                i0, i1 = self.att['i0'], self.att['i1']
                P = np.vstack([P[:i0], T, P[i1 + 1:]])
                self.att['i1'] = i0 + len(T) - 1
                on = None  # point count changed
                if held_loop is not None and len(held_loop) != len(P):
                    held_loop = np.concatenate([held_loop[:i0], np.zeros(len(T), bool), held_loop[i1 + 1:]]) if len(held_loop[:i0]) + len(T) + len(held_loop[i1 + 1:]) == len(P) else None
            P[0] = self.anchor; P[-1] = self.G
        if held_loop is not None and self.att is not None and len(held_loop) == len(P):
            idx = np.where(held_loop[:max(0, self.att['i0'] - 8)])[0]
            if len(idx) > 5: self.hand_rng = (int(idx.min()), int(idx.max()))
        if self.att is None:
            self.P = hook_push(H, resample(P))
        else:
            i0, i1 = self.att['i0'], self.att['i1']
            A = resample(P[:i0 + 1]); B = resample(P[i1:])
            self.att['i0'] = len(A) - 1; self.att['i1'] = len(A) - 1 + (i1 - i0)
            self.P = np.vstack([A[:-1], P[i0:i1 + 1], B[1:]])
            k = np.ones(len(self.P), bool); k[self.att['i0']:self.att['i1'] + 1] = False
            Q = hook_push(H, self.P.copy()); self.P[k] = Q[k]
        return self.P


# ---------------------------------------------------------------- scene
HOME = Pose([0, 0, 0], [-1, 0, 0], [0, 1, 0])
XL = 6.0
G0 = np.array([9.0, 13.0, -9.0])          # finger eyelet at rest: up-behind the hook (the loop stays put)
# the finger turns the working yarn so it runs back-left behind the hook head (where the throat can take it)
GK = dict(low=np.array([7.0, -12.0, -12.0]), rest=G0, mid=np.array([-6.0, 12.0, -22.0]), left=np.array([-30.0, 3.0, -20.0]))
PINCH0 = np.array([XL + 0.4, -12.0, 0.6])  # the other hand pinches the tail under the work
TAIL = 22.0
RLINK = 3.6  # a finished link stays this open (centreline radius), friction stops it closing further


def initial_yarn_tail():
    return resample(np.array([PINCH0 + [0, -TAIL, 0.2], PINCH0]))[:-1]


def initial_yarn():
    """Tail up from the pinch, one full wrap round the shaft at XL, then off to the finger."""
    c = HOME.pt(-XL)
    pts = [PINCH0, PINCH0 + [0, 4, 0.3]]
    rho = RS + R + 0.6
    for k in range(55):
        th = math.radians(370 - 485 * k / 54)  # up the back, over, down the front, under, and up the back again
        pts.append(c + np.array([1.4 - 2.8 * k / 54, -rho * math.cos(th), rho * math.sin(th)]))
    last = pts[-1]
    pts.append(last + norm(G0 - last) * 3)
    pts.append(G0)
    Y = Yarn(np.array(pts), PINCH0, G0)
    Y.frozen = initial_yarn_tail()
    return Y


def key(t_list):
    return t_list


# hook poses for one chain stitch (world mm)
TURN = Pose([1.0, -0.2, -0.6], [-1, 0.0, 0.0], [0, -0.6, -0.8])   # throat turned down-back, back at the loop
PULL = Pose([10.5, -0.6, -0.6], [-1, 0.0, 0.0], [0, -0.6, -0.8])   # pulled back through the loop
UP = Pose([10.5, 1.8, -0.5], [-1, 0.0, 0.0], [0, 1.0, 0.0])         # throat up: the new loop is on the hook


def catch_pose(Y, back=0.0):
    """Throat facing back, its groove on the working yarn where it passes behind the head (back>0: still in front)."""
    xw = HOME.pt(XTH)[0]
    P = Y.P
    # the working yarn: after the last point that touches the loop on the hook
    L = HOME.local(P); on = np.where((np.hypot(L[:, 1], L[:, 2]) < RS + R + 0.6) & (L[:, 0] < -3))[0]
    start = on.max() + 1 if len(on) else len(P) // 2
    k0 = start + int(np.argmin(np.abs(P[start:, 0] - xw) + 50 * (P[start:, 2] > -1.5)))
    Y.catch_hint = k0
    target = P[k0] + np.array([0, 0, back])
    a = np.array([-1.0, 0, 0]); d = np.array([0, 0.25, -1.0]); d = norm(d)
    return Pose(target - a * XTH - d * (RNECK + R), a, d)


def path(keys, u):
    if len(keys) == 1: return GK[keys[0]].copy()
    x = u * (len(keys) - 1); i = min(int(x), len(keys) - 2); w = x - i
    pts = [GK[k] for k in keys]
    p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, len(pts) - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * w + (2 * p0 - 5 * p1 + 4 * p2 - p3) * w * w + (-p0 + 3 * p1 - 3 * p2 + p3) * w ** 3)


def run(n_st, skip, out_path):
    Y = initial_yarn()
    for _ in range(60): Y.step(HOME, iters=20)
    frames = []
    drop_per = 3.6
    t = 0.0
    for st in range(n_st):
        pinch_start = Y.anchor.copy()
        poses = dict(home=HOME, turn=TURN, pull=PULL, up=UP)
        # (hook from, hook to, finger path, seconds, throat holds, loop sticks to the shaft)
        rm = 'ring' if Y.att is not None else None
        link_frozen = False
        seq = [('home', 'home', ['rest'], 0.4, rm, True),
               ('home', 'home', ['rest', 'mid', 'left'], 1.0, rm, True),      # the yarn is brought behind the head
               ('home', 'near', ['left'], 0.6, rm, True),                     # throat turns back, toward it
               ('near', 'catch', ['left'], 0.4, 'catch', True),               # and takes it (yarn over)
               ('catch', 'turn', ['left'], 0.6, 'hold', True),                # throat turns down with the yarn
               ('turn', 'pull', ['left'], 1.3, 'hold', False),                # pulled through the loop on the hook
               ('pull', 'up', ['left', 'mid'], 0.8, 'hold', False),           # throat up: it is the new loop
               ('up', 'home', ['mid', 'rest'], 1.1, 'seat', False),           # the hook goes into the new loop
               ('home', 'home', ['rest'], 0.1, 'ring', True)]
        for (p0, p1, gk, dur, hold, stick) in seq:
            if p0 == 'pull':  # the other hand draws the work down: the old loop closes under the hook
                moved = 0.0; work_shift = np.array([0.0, -4.0, 0.0])
            if p0 == 'up':  # the work follows the new loop as the hook goes home
                work_shift = np.array([-3.0, -1.0, 0.0]); moved = 0.0
            if p1 == 'near':
                Y.G = path(gk, 1.0)
                poses['catch'] = catch_pose(Y); poses['near'] = catch_pose(Y, back=2.2)
            A, B = poses[p0], poses[p1]
            nfr = int(round(dur * FPS))
            for k in range(nfr):
                for sbs in range(SUB):
                    u = (k + (sbs + 1) / SUB) / nfr
                    H = A.lerp(B, ease(u))
                    Y.G = path(gk, ease(u))
                    if p0 in ('pull', 'up'):
                        Y.move_work(work_shift * (ease(u) - moved)); moved = ease(u)
                    if p0 in ('pull',): hx = XTH
                    elif p0 == 'up': hx = XTH + (-6.0 - XTH) * ease(u)
                    else: hx = -6.0
                    hand = H.pt(hx) + np.array([0.0, -4.4, 0.0]) if p0 not in ('turn',) else None
                    Y.step(H, mode=hold, u=u, stick=stick, slip=(p0 == 'turn'), hand=hand)
                    if p0 in ('pull', 'up') and not link_frozen and getattr(Y, 'hand_rng', None):
                        j0, j1 = Y.hand_rng
                        Lk = Y.P[j0:j1 + 1]; rad = np.linalg.norm(Lk - Lk.mean(axis=0), axis=1).max()
                        if rad < RLINK and j1 + 2 < Y.att['i0']:
                            # the new link has closed round the legs; friction keeps it like that: it is finished work
                            Y.freeze(j1 + 1); Y.hand_rng = None; link_frozen = True
                t += 1 / FPS
                if st >= skip:
                    frames.append(frame_json(Y, H, t, st))
        print("stitch", st, "points", len(Y.P), flush=True)
    cam = dict(pos=[18, 4, 46], look=[5, -3, -2], fov=36)
    json.dump(dict(fps=FPS, r=R, frames=frames, cam=cam, G=G0.tolist()), open(out_path, "w"))
    print("frames", len(frames))


def frame_json(Y, H, t, st):
    beyond = Y.G + norm(Y.G - Y.P[-8]) * np.array([[5.0], [12.0]])
    P = np.vstack([Y.frozen, Y.P, beyond])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return dict(t=round(t, 3), stitch=st, pts=P.round(3).tolist(), s=s.round(3).tolist(), hook=H.js())


if __name__ == "__main__":
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4; sk = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run(n, sk, out)
