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
import json, math, os, sys
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


def self_collide(P, w=None, iters=1, skip=None):
    """skip: (ga, gb) index ranges [lo, hi) whose points do not push each other."""
    d0 = 2 * R * 1.1  # a little margin: the tension always wins back a bit
    for _ in range(iters):
        pairs = cKDTree(P).query_pairs(d0, output_type='ndarray')
        if len(pairs) == 0: return P
        pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > NEIGH]
        if skip is not None and len(pairs):
            (a0, a1), (b0, b1) = skip; i, j = pairs[:, 0], pairs[:, 1]
            ia = (i >= a0) & (i < a1); ja = (j >= a0) & (j < a1); ib = (i >= b0) & (i < b1); jb = (j >= b0) & (j < b1)
            pairs = pairs[~((ia & jb) | (ib & ja))]
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


def seated_curve(n, sign, w, c0=0.0, c1=0.0, th=0.0):
    """Caught yarn -> loop round the shaft (w: 0..1), in world-fixed angles: the U in the throat (centred on the
    throat's world angle c0) becomes a ring hanging over the shaft (centred on c1, its legs down to the work).
    th: the hook's present roll; the loop does not turn with the hook once it is out of the throat."""
    U = np.linspace(-95, 95, n) * sign + c0
    F = np.linspace(-160, 160, n) * sign + c1
    ph = np.radians(U + (F - U) * w - th)
    x = np.full(n, XTH + (-6.0 - XTH) * w)
    r = np.maximum(surf(XTH, ph) + R + 0.03, surf(x, ph) + R + 0.05)
    return np.stack([x, r * np.cos(ph), r * np.sin(ph)], axis=1)


def roll(H):
    """World angle of the throat round the shaft (degrees, 0 = up, 90 = away from the viewer)."""
    return math.degrees(math.atan2(-H.d[2], H.d[1]))


def curve_len(C):
    return float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))


def resample_n(C, n):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, C[:, k]) for k in range(3)], axis=1)


def resample_free(P, fixed):
    """Resample the free spans between the fixed (authored) pieces; the fixed pieces keep their points."""
    out = []; idx = []; a = 0
    for (i0, i1) in fixed:
        F = resample(P[a:i0 + 1]) if i0 > a else P[a:a + 1]
        if out: F = F[1:]
        out.append(F); n0 = sum(len(o) for o in out) - 1
        out.append(P[i0 + 1:i1 + 1]); idx.append((n0, n0 + i1 - i0)); a = i1
    F = resample(P[a:]); out.append(F[1:] if out else F)
    return np.vstack(out), idx


def link_ring(n, sign, rho=None):
    """A finished link: an upright oval facing the viewer, open at the bottom where its two legs go down into the
    link below (like the Vs of a real chain). Length (n points) sets the size, the shape is fixed."""
    span = math.radians(320)
    ps = math.radians(90) + sign * (math.radians(20) + np.linspace(0, span, n))
    ell = np.stack([LINK_RX * np.cos(ps), LINK_RV * np.sin(ps)], axis=1)
    L = np.sum(np.linalg.norm(np.diff(ell, axis=0), axis=1)); k = (n - 1) * H_SP / L
    return np.outer(ell[:, 0] * k, [1.0, 0, 0]) + np.outer(ell[:, 1] * k, LINK_B)


def link_pt(c, psi):
    return c + LINK_RX * math.cos(psi) * np.array([1.0, 0, 0]) + LINK_RV * math.sin(psi) * LINK_B


def link_n():
    ell = [(LINK_RX * math.cos(p), LINK_RV * math.sin(p)) for p in np.radians(np.linspace(110, 430, 200))]
    return int(round(np.sum(np.linalg.norm(np.diff(np.array(ell), axis=0), axis=1)) / H_SP)) + 1


LINK_RX, LINK_RV = 2.3, 3.6
LINK_TILT = math.radians(72)   # links face the viewer this much (0: flat rings round a vertical axis)
LINK_B = np.array([0.0, -math.sin(LINK_TILT), math.cos(LINK_TILT)])
LINK_N = np.array([0.0, math.cos(LINK_TILT), math.sin(LINK_TILT)])


def keep_thread(P, i0, i1, L, frac=0.55):
    """A leg that goes through a link stays through it: where it crosses the link's plane, it is kept inside the
    link (the link's yarn is in the way; the solver's point contacts alone let a fast-moving link slip past it)."""
    c = L.mean(axis=0); _, _, Vt = np.linalg.svd(L - c); e1, e2, nn = Vt
    ru = np.abs((L - c) @ e1).max(); rv = np.abs((L - c) @ e2).max()
    X = P[i0:i1 + 1] - c; n = X @ nn
    k = np.where(n[:-1] * n[1:] < 0)[0]
    if not len(k): return
    t = n[k] / (n[k] - n[k + 1]); Q = X[k] + (X[k + 1] - X[k]) * t[:, None]
    u = Q @ e1 / ru; v = Q @ e2 / rv; r = np.hypot(u, v)
    j = int(np.argmin(r))
    if r[j] > 1.0 or r[j] <= frac: return   # not through it, or well inside
    sc = frac / r[j]
    d = (u[j] * (sc - 1)) * ru * e1 + (v[j] * (sc - 1)) * rv * e2
    P[i0 + k[j]] += d; P[i0 + k[j] + 1] += d


class Yarn:
    """The strand from the finished work (anchor) to the finger eyelet G. Two pieces of it can be authored:
    att (the loop / caught yarn carried by the hook, hook-local shape) and link (the old loop closing into a chain
    link, held by the other hand). Everything else is solved as a taut string."""
    def __init__(self, P, anchor, G):
        self.P = resample(np.asarray(P, float)); self.anchor = np.asarray(anchor, float); self.G = np.asarray(G, float)
        self.att = None; self.H = None; self.link = None; self.old = None
        self.frozen = np.zeros((0, 3))

    def freeze(self, f):
        self.frozen = np.vstack([self.frozen, self.P[:f]])
        self.P = self.P[f:].copy(); self.anchor = self.P[0].copy()
        for a in (self.att, self.link):
            if a is not None: a['i0'] -= f; a['i1'] -= f

    def move_work(self, delta):
        self.frozen = self.frozen + delta; self.anchor = self.anchor + delta; self.P[0] = self.anchor

    def target(self, mode, u):
        a = self.att
        if mode == 'catch':
            n = a['n0']; U = caught_curve(n, a['sign'])
            return a['L0'] + (U - a['L0']) * ease(u)
        if mode == 'hold':
            a['c0'] = roll(self.H)
            return caught_curve(a['n0'], a['sign'])
        # the loop turns with the hook (it was turned throat-up while still in the throat)
        C = seated_curve(64, a['sign'], 1.0 if mode == 'ring' else ease(u))  # 'seat' / 'ring'
        n = max(4, int(round(curve_len(C) / H_SP)) + 1)
        return resample_n(C, n)

    def link_target(self):
        k = self.link
        if k.get('mode') == 'fixed':
            return k['T']
        if k.get('mode') == 'hold':
            # the old loop, held open by the fingers round where the hook goes through it
            C = k['C0'].copy(); rr = np.hypot(C[:, 1], C[:, 2])
            fe = 1 + (k['f'] - 1) * np.clip((3.4 - rr) / 1.2, 0, 1)
            C[:, 1] *= fe; C[:, 2] *= fe
            return k['c0'] + C
        e = ease(k['u'])
        c = k['c0'] + (k['c1'] - k['c0']) * e
        return c + k['C0'] + (k['A'] - k['C0']) * e

    def pieces(self):
        return sorted([(p['i0'], p['i1']) for p in (self.att, self.link) if p is not None])

    def step(self, H, mode=None, u=0.0, iters=ITERS, tau=TAU, stick=False, slip=False, flow=False):
        P = self.P
        if flow and self.att is not None and mode in ('ring', 'seat'):
            # the finger keeps the working yarn snug: slack in the leg from the work slides on through the loop
            # on the hook toward the finger (one point = 0.36 mm at a time)
            a = self.att; st = self.link['i1'] if (self.link is not None and self.link['i1'] < a['i0']) else 0
            Lw = float(np.sum(np.linalg.norm(np.diff(P[st:a['i0'] + 1], axis=0), axis=1)))
            if Lw > LW_TARGET and a['i0'] - st > 6:
                a['i0'] -= 1; a['i1'] -= 1
        held_loop = None
        if slip and self.H is not None:
            # the old loop is held by the other hand: the hook slides through it, it does not travel with the hook
            Lh = self.H.local(P); rrh = np.hypot(Lh[:, 1], Lh[:, 2])
            held_loop = (Lh[:, 0] < 0.8) & (Lh[:, 0] > -12) & (rrh < RS + R + 2.8)
            if self.att is not None: held_loop[self.att['i0']:self.att['i1'] + 1] = False
            X0 = P[held_loop].copy(); ax = self.H.a.copy()
        on = None
        if stick and self.H is not None:
            L0 = self.H.local(P); rr0 = np.hypot(L0[:, 1], L0[:, 2])
            on = (L0[:, 0] < -3.6) & (rr0 < RS + R + 0.35)
            # only the loop that was on the hook sticks (the working yarn slides over the shaft freely)
            keep = np.zeros(len(P), bool)
            if self.old is not None: keep[max(0, self.old['i0'] - 3):self.old['i1'] + 4] = True
            on &= keep
            x0 = L0[on, 0]
        self.H = H
        if mode is None or (mode == 'catch' and self.att is not None and self.att.get('kind') == 'ring'):
            if self.att is not None:
                self.old = dict(i0=self.att['i0'], i1=self.att['i1'], sign=self.att['sign'])
            self.att = None  # the loop on the hook is let go: it is just yarn round the shaft now
        if mode == 'ring' and self.att is not None:
            self.att['kind'] = 'ring'
        if mode is not None and self.att is None:  # catch: the bit of yarn lying across the throat
            tb = H.pt(XTH, RNECK + R)
            K = int(round(2.7 / H_SP))
            dd = np.linalg.norm(P - tb, axis=1)
            lo = K + 2
            hi = len(P) - K - 2
            if getattr(self, 'catch_hint', None) is not None:
                lo = max(lo, self.catch_hint - 8); hi = min(hi, self.catch_hint + 9)
            c = lo + int(np.argmin(dd[lo:hi]))
            i0, i1 = c - K, c + K
            Lc = H.local(P[i0:i1 + 1])
            sign = 1.0 if np.arctan2(Lc[0, 2], Lc[0, 1]) < np.arctan2(Lc[-1, 2], Lc[-1, 1]) else -1.0
            self.att = dict(i0=i0, i1=i1, n0=i1 - i0 + 1, L0=Lc, sign=sign)
            print("catch sign", sign, "at", c, flush=True)
            if self.old is not None and self.link is None:
                hold_old(self)
        for it in range(iters):
            P[0] = self.anchor; P[-1] = self.G
            P = tension(P, tau)
            P = hook_push(H, P)
            w = np.ones(len(P))
            for (i0, i1) in self.pieces(): w[i0:i1 + 1] = 1e-6
            nf = min(len(self.frozen), 120)
            skip = None
            if CONN_FREE and self.link is not None and 'u' in self.link:
                # the new link is drawn down by the other hand over the short bit joining it to the last link:
                # that bit slides inside it instead of being swept along (and crumpled) by it
                k_ = self.link; skip = ((nf, nf + k_['i0']), (nf + k_['i0'], nf + k_['i1'] + 1))
            if nf:
                Q = np.vstack([self.frozen[len(self.frozen) - nf:], P])
                wq = np.concatenate([np.full(nf, 1e-6), w])
                P = self_collide(Q, wq, iters=4, skip=skip)[nf:]
            else:
                P = self_collide(P, w, iters=4, skip=skip)
            P = hook_push(H, P)
            if held_loop is not None and held_loop.any() and len(held_loop) == len(P):
                dX = P[held_loop] - X0
                P[held_loop] = X0 + dX - np.outer(dX @ ax, ax)
            if on is not None and on.any() and len(P) == len(on):
                L = H.local(P[on]); L[:, 0] = x0; P[on] = H.world(L)
            if self.link is not None:
                k = self.link; T = self.link_target(); P[k['i0']:k['i1'] + 1] = T
                if THREAD and self.att is not None and 'u' in k and k['i1'] < self.att['i0']:
                    keep_thread(P, k['i1'] + 1, self.att['i0'] - 1, T)
                    keep_thread(P, self.att['i1'] + 1, len(P) - 2, T)
            if self.att is not None:
                T = H.world(self.target(mode, u))
                i0, i1 = self.att['i0'], self.att['i1']
                if len(T) != i1 - i0 + 1:
                    P = np.vstack([P[:i0], T, P[i1 + 1:]]); self.att['i1'] = i0 + len(T) - 1
                    on = None; held_loop = None
                else:
                    P[i0:i1 + 1] = T
            P[0] = self.anchor; P[-1] = self.G
        fx = self.pieces()
        P2, idx = resample_free(P, fx)
        for (i0, i1), (j0, j1) in zip(fx, idx):
            for a in (self.att, self.link):
                if a is not None and a['i0'] == i0: a['i0'], a['i1'] = j0, j1
        if self.old is not None and fx:
            # the old loop's indices move with the free span it is in
            sh = 0
            for (i0, i1), (j0, j1) in zip(fx, idx):
                if i1 < self.old['i0']: sh = j1 - i1
            self.old['i0'] += sh; self.old['i1'] += sh
        k = np.ones(len(P2), bool)
        for (j0, j1) in idx: k[j0:j1 + 1] = False
        Q = hook_push(H, P2.copy()); P2[k] = Q[k]
        self.P = P2
        return self.P


# ---------------------------------------------------------------- scene
HOME = Pose([0, 0, 0], [-1, 0, 0], [0, 1, 0])
XL = 6.0
G0 = np.array([9.0, 13.0, -9.0])          # finger eyelet at rest: up-behind the hook (the loop stays put)
GK = dict(low=np.array([7.0, -12.0, -12.0]), rest=G0, mid=np.array([2.0, 15.0, -10.0]), left=np.array([-4.0, 14.0, -9.0]))
LDROP = 7.0      # the newest link sits this far under the shaft (held there by the other hand)
PITCH = 4.0      # chain links are this far apart
RHO = 2.9        # link ring (centreline) radius: holds the two legs of the next loop
TAIL = 22.0
C_LINK = HOME.pt(-XL) + np.array([0.0, -LDROP, 0.0])


def initial_yarn():
    """Start: a slip knot and one chain already made (so every stitch starts from the same kind of state): tail up
    to link -1, from it through itself up to link 0, then (free) down through link -1, round behind it and up
    through link 0 to the loop on the hook, round the shaft, down through link 0 and up behind to the finger."""
    c0 = C_LINK; cm = C_LINK - np.array([0.0, PITCH, 0.0]); n = link_n()
    ringm = cm + link_ring(n, 1.0); ring0 = c0 + link_ring(n, 1.0)
    tail = resample(np.array([ringm[0] + [0, -TAIL, 0], ringm[0] + [0, -2.0, 0]]))
    em = ringm[-1]
    up0 = [em, em - LINK_N * 1.4, cm - LINK_N * 2.0 + [0.6, 0, 0], cm + LINK_N * 2.0 + [0.6, 0, 0],
           0.5 * (cm + c0) + LINK_N * 1.6, ring0[0] + LINK_N * 0.8]
    loop = HOME.world(resample_n(seated_curve(64, 1.0, 1.0), 40))
    e = ring0[-1]
    pts = list(ring0) + [e + LINK_N * 1.2, cm + LINK_N * 1.8 - [0.6, 0, 0], cm - LINK_N * 2.2 - [0.6, 0, 0],
                         c0 - LINK_N * 2.2 + [0.6, 0, 0], c0 + LINK_N * 2.0 + [0.6, 0, 0], loop[0] + [0, -1.2, 0.2]]
    pts += list(loop)
    pts += [loop[-1] + [0, -1.2, -0.2], c0 + LINK_N * 2.0 - [0.6, 0, 0], c0 - LINK_N * 2.4 - [0.6, 0, 0],
            c0 - LINK_N * 3.5 + [0, 4.0, 0], G0]
    Pf = np.vstack([resample(np.array(up0)), resample(np.array(pts))[1:]])
    Y = Yarn(Pf, em, G0)
    Y.frozen = np.vstack([tail, resample(np.vstack([tail[-1], ringm[0]]))[1:-1], ringm[:-1]])
    Y.H = HOME
    # link 0 and the loop on the hook are authored pieces while everything settles
    k0 = int(np.argmin(np.linalg.norm(Y.P - ring0[0], axis=1)))
    Y.link = dict(i0=k0, i1=k0 + n - 1, mode='fixed', T=ring0.copy())
    i0 = int(np.argmin(np.linalg.norm(Y.P - loop[0], axis=1))); i1 = int(np.argmin(np.linalg.norm(Y.P - loop[-1], axis=1)))
    Y.att = dict(i0=i0, i1=i1, n0=i1 - i0 + 1, L0=None, sign=1.0, kind='ring')
    return Y


def initial_slip():
    """Start from a slip knot only: tail up to ring 0, from its end through it up to the loop on the hook, round
    the shaft, down through ring 0 and up behind to the finger. SSIGN=-1 (default) winds the loop on the hook the
    way every chain stitch leaves it (the leg from the work in front, the working yarn behind), so the first stitch
    starts from the same kind of state as all the others."""
    S = float(os.environ.get('SSIGN', '-1')); Rg = float(os.environ.get('RSIGN', os.environ.get('LSIGN', S)))
    c0 = C_LINK; n = link_n(); X = np.array([0.6 * Rg, 0, 0])
    ring0 = c0 + link_ring(n, Rg)
    tail = resample(np.array([ring0[0] + [0, -TAIL, 0], ring0[0] + [0, -2.0, 0]]))
    frozen = np.vstack([tail, resample(np.vstack([tail[-1], ring0[0]]))[1:-1], ring0])
    loop = HOME.world(resample_n(seated_curve(64, S, 1.0), 40))
    e = ring0[-1]
    pts = [e, e - LINK_N * 1.4, c0 - LINK_N * 2.0 + X, c0 + LINK_N * 2.0 + X, loop[0] + [0, -1.2, 0.2 * S]]
    pts += list(loop)
    pts += [loop[-1] + [0, -1.2, -0.2 * S], c0 + LINK_N * 2.0 - X, c0 - LINK_N * 2.4 - X,
            c0 - LINK_N * 3.5 + [0, 4.0, 0], G0]
    Y = Yarn(resample(np.array(pts)), e, G0)
    Y.frozen = frozen; Y.H = HOME
    i0 = int(np.argmin(np.linalg.norm(Y.P - loop[0], axis=1))); i1 = int(np.argmin(np.linalg.norm(Y.P - loop[-1], axis=1)))
    Y.att = dict(i0=i0, i1=i1, n0=i1 - i0 + 1, L0=None, sign=S, kind='ring')
    return Y


def initial_from(path):
    """Start from where a one-stitch run (from the slip knot) ended: its newest link sits exactly where ring 0
    sits, so the bare slip-knot tail + ring 0 replace the finished chain under it."""
    d = json.load(open(path)); f = d['frames'][-1]; P = np.array(f['pts']); nf = f['nfrozen']
    Y0 = initial_slip()
    Y = Yarn(P[nf:-2], P[nf], G0); Y.P = P[nf:-2].copy(); Y.anchor = P[nf].copy()
    Y.frozen = Y0.frozen; Y.H = HOME
    i0, i1 = f['att'][0] - nf, f['att'][1] - nf
    Y.att = dict(i0=i0, i1=i1, n0=i1 - i0 + 1, L0=None, sign=Y0.att['sign'], kind='ring')
    return Y


def settle_initial(Y, steps=80):
    if Y.link is None:
        for _ in range(60): Y.step(HOME, mode='ring', iters=20)
        return
    for _ in range(steps): Y.step(HOME, mode='ring', iters=20)
    Y.freeze(Y.link['i1'] + 1); Y.link = None


# hook poses for one chain stitch (world mm)
TURN = Pose([1.0, -0.2, -0.6], [-1, 0.0, 0.0], [0, -0.85, 0.5])   # throat turned down-back, back at the loop
PULL = Pose([15.0, -0.6, -0.6], [-1, 0.0, 0.0], [0, -0.85, 0.5])   # pulled back through the loop
UP = Pose([15.0, 1.8, -0.5], [-1, 0.0, 0.0], [0, 1.0, 0.0])         # throat up: the new loop is on the hook


def catch_pose(Y, back=0.0):
    """Throat facing back, its groove on the working yarn where it passes behind the head (back>0: still in front)."""
    xw = HOME.pt(XTH)[0]
    P = Y.P
    start = Y.att['i1'] + 1 if Y.att is not None else len(P) // 2
    start += int(np.argmin(P[start:, 1]))  # past the turn under the last link: the working yarn
    W = P[start:]
    k0 = start + int(np.argmin(np.abs(W[:, 1] - 0.3) + 0.3 * np.abs(W[:, 0] - xw) + 50 * (W[:, 2] > -1.0)))
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


def aim_finger(Y):
    """Where the finger takes the working yarn for the yarn-over: so that it runs from where it leaves the work
    behind the hook's head, at the throat."""
    P = Y.P; a1 = Y.att['i1'] if Y.att is not None else 0
    rest = np.vstack([Y.frozen, P[:a1 + 1]])
    from scipy.spatial import cKDTree as KD
    d, _ = KD(rest).query(P[a1 + 1:])
    near = np.where(d < 2.6)[0]
    ex = P[a1 + 1 + (near.max() if len(near) else 0)]
    T = HOME.pt(XTH) + np.array([0.0, 0.3, -2.6])
    GK['left'] = ex + norm(T - ex) * 24.0
    GK['mid'] = 0.5 * (GK['rest'] + GK['left']) + np.array([0.0, 3.0, 0.0])


def hold_old(Y):
    """The loop that was on the hook is held (by the fingers) as it is; the hook goes through it."""
    o = Y.old
    n = link_n()
    mid = (o['i0'] + o['i1']) // 2
    i0 = max(1, mid - n // 2); i1 = i0 + n - 1
    if i1 > Y.att['i0'] - 3:
        i1 = Y.att['i0'] - 3; i0 = max(1, i1 - n + 1)
    c = Y.P[o['i0']:o['i1'] + 1].mean(axis=0)
    Y.link = dict(i0=i0, i1=i1, C0=Y.P[i0:i1 + 1] - c, c0=c, f=1.0, mode='hold')


def start_link(Y):
    """The old loop (now round the new loop's legs) becomes a link: from its present shape it is drawn into an oval
    under the hook by the other hand."""
    k = Y.link; C = Y.link_target(); c0 = C.mean(axis=0); n = len(C)
    Y.link = dict(i0=k['i0'], i1=k['i1'], C0=C - c0, A=link_ring(n, float(os.environ.get('LSIGN', Y.old['sign']))), c0=c0, c1=C_LINK.copy(), u=0.0)


CATCH = {}
THREAD = os.environ.get('THREAD', '1') == '1'
LW_TARGET = float(os.environ.get('LW', '14.5'))
FLOW = os.environ.get('FLOW', '0') == '1'
CONN_FREE = os.environ.get('CONN_FREE', '1') == '1'


def run(n_st, skip, out_path):
    import os
    st_ = os.environ.get('START', 'slip')
    if st_.endswith('.json'):
        Y = initial_from(st_)
        c = json.load(open(st_)).get('catch')
        if c: CATCH['pose'] = Pose(**c['pose']); CATCH['near'] = Pose(**c['near'])
    else:
        Y = initial_slip() if st_ == 'slip' else initial_yarn()
        settle_initial(Y)
    frames = []
    t = 0.0
    for st in range(n_st):
        poses = dict(home=HOME, turn=TURN, pull=PULL, up=UP)
        # (hook from, hook to, finger path, seconds, throat mode, loop sticks to the shaft)
        seq = [('home', 'home', ['rest'], 0.4, 'ring', True),
               ('home', 'home', ['rest', 'mid', 'left'], 1.0, 'ring', True),  # the yarn is brought behind the head
               ('home', 'near', ['left'], 0.6, 'ring', True),                 # throat turns back, toward it
               ('near', 'catch', ['left'], 0.4, 'catch', True),               # and takes it (yarn over)
               ('catch', 'turn', ['left'], 0.6, 'hold', True),                # throat turns down with the yarn
               ('turn', 'pull', ['left'], 1.3, 'hold', False),                # pulled through the loop on the hook
               ('pull', 'up', ['left', 'mid'], 1.2, 'hold', False),           # throat up: it is the new loop
               ('up', 'home', ['mid', 'rest'], 1.1, 'seat', False),           # the hook goes into the new loop
               ('home', 'home', ['rest'], 0.3, 'ring', True)]
        T_LINK = 1.2 + 1.1; tl = 0.0
        for (p0, p1, gk, dur, mode, stick) in seq:
            if p1 == 'near':
                Y.G = path(gk, 1.0)
                if CATCH.get('pose') is None:
                    CATCH['pose'] = catch_pose(Y); CATCH['near'] = catch_pose(Y, back=2.2)
                else:  # every stitch takes the yarn at the same place: hint = the working yarn nearest the throat
                    tb = CATCH['pose'].pt(XTH, RNECK + R); a1 = Y.att['i1'] + 4
                    Y.catch_hint = a1 + int(np.argmin(np.linalg.norm(Y.P[a1:] - tb, axis=1)))
                poses['catch'] = CATCH['pose']; poses['near'] = CATCH['near']
            if p0 == 'pull' and Y.link is not None:
                start_link(Y)
            if p0 == 'home' and p1 == 'home' and Y.link is not None:
                Y.freeze(Y.link['i1'] + 1); Y.link = None; Y.old = None
            moved = 0.0
            A, B = poses[p0], poses[p1]
            nfr = int(round(dur * FPS))
            for k in range(nfr):
                for sbs in range(SUB):
                    u = (k + (sbs + 1) / SUB) / nfr
                    H = A.lerp(B, ease(u))
                    Y.G = path(gk, ease(u))
                    if Y.link is not None and Y.link.get('mode') == 'hold':
                        if p0 == 'near': Y.link['f'] = 1.0 + 0.15 * ease(u)
                        if p0 == 'turn': Y.link['f'] = 1.15 + (4.0 / 2.06 - 1.15) * ease(min(1.0, u / 0.3))
                    if p0 == 'catch':  # the other hand lowers the work by one link before the hook pulls through
                        Y.move_work(np.array([0.0, -PITCH, 0.0]) * (ease(u) - moved)); moved = ease(u)
                    if Y.link is not None and p0 in ('pull', 'up'):
                        tl += dur / nfr / SUB; Y.link['u'] = min(1.0, tl / T_LINK)
                    Y.step(H, mode=mode, u=u, stick=stick, slip=(p0 == 'turn'), flow=FLOW and sbs % 2 == 1)
                t += 1 / FPS
                if st >= skip:
                    frames.append(frame_json(Y, H, t, st))
        print("stitch", st, "points", len(Y.P), "frozen", len(Y.frozen), flush=True)
    cam = dict(pos=[18, 4, 46], look=[5, -3, -2], fov=36)
    catch = dict(pose=CATCH['pose'].js(), near=CATCH['near'].js()) if CATCH.get('pose') is not None else None
    json.dump(dict(fps=FPS, r=R, frames=frames, cam=cam, G=G0.tolist(), catch=catch), open(out_path, "w"))
    print("frames", len(frames))


def frame_json(Y, H, t, st):
    beyond = Y.G + norm(Y.G - Y.P[-8]) * np.array([[5.0], [12.0]])
    P = np.vstack([Y.frozen, Y.P, beyond])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    nf = len(Y.frozen)
    att = None if Y.att is None else [Y.att['i0'] + nf, Y.att['i1'] + nf]
    link = None if Y.link is None else [Y.link['i0'] + nf, Y.link['i1'] + nf]
    return dict(t=round(t, 3), stitch=st, pts=P.round(3).tolist(), s=s.round(3).tolist(), hook=H.js(), att=att,
                link=link, nfrozen=nf)


if __name__ == "__main__":
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4; sk = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run(n, sk, out)
