"""Hand-authored crochet chain animation (no solver): one continuous yarn, every frame built from explicit geometry.

Rules the construction follows (and check_anim.py verifies on the output):
  * the yarn is ONE strand from the tail end to the ball; its material coordinate s is arc length from the tail end,
    so yarn can only slide along itself (fed from the finger), never appear or vanish;
  * every shape is a continuous function of time: the hook pose is keyframed, everything the hook carries is
    attached to the hook frame, the finished chain only moves as one piece (the other hand lowers the work);
  * topology changes only the way it does in real crochet: the caught yarn enters the old loop only by the throat
    carrying it through the loop's interior, and the old loop only leaves the hook by sliding off the tip.

World (mm): y up, camera in front (+z). Hook at rest: tip at the origin, axis a = -x (shaft at x > 0), throat up.
Usage: python chain_anim.py <out.json> [n_stitches=3]
"""
import json, math, sys
import numpy as np

R = 0.95  # yarn radius
RS, RNECK, XTH = 1.06, 0.51, -2.05  # hook shaft radius, neck radius under the throat, throat position (hook-local x)
RHO = RS + R + 0.12  # loop round the shaft (centreline radius)
XL = 6.0  # the loop on the hook sits 6 mm behind the tip
CX = 6.0  # chain axis x (under the loop)
D = 3.4  # chain link pitch
YB = -RHO - 3.0  # level of the cinch ring of the last finished link (base of the loop on the hook)
RX, RZ = 2.6, 3.0  # cinch ring half axes (z before the tilt)
TILT = math.radians(50)  # front edge of a finished loop hangs down: the chain shows V-loops to the camera
XB = 5.2  # the working yarn rises behind on the tip side of the loop plane (x < XL)
FPS = 30


def norm(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / (n or 1.0)


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (u * 6 - 15) + 10)


def rot(v, axis, ang):
    axis = norm(axis); v = np.asarray(v, float)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))


def resample(P, n):
    """Polyline -> n points evenly spaced by arc length."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9: return np.repeat(P[:1], n, axis=0)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], axis=1)


def catmull(C, per=6):
    """Centripetal Catmull-Rom through control points (keeps the curve through every control)."""
    C = np.asarray(C, float)
    if len(C) < 3: return C
    P = np.vstack([2 * C[0] - C[1], C, 2 * C[-1] - C[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-4) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-4) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-4) ** 0.5
        for t in np.linspace(t1, t2, per, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(C[-1])
    return np.array(out)


# ---------------------------------------------------------------- hook
class Pose:
    def __init__(self, tip, a, d):
        self.tip = np.asarray(tip, float); self.a = norm(a)
        self.d = norm(np.asarray(d, float) - self.a * np.dot(d, self.a)); self.s = np.cross(self.a, self.d)

    def pt(self, x, y=0.0, z=0.0):
        return self.tip + self.a * x + self.d * y + self.s * z

    def local(self, p):
        q = np.asarray(p) - self.tip
        return np.stack([q @ self.a, q @ self.d, q @ self.s], axis=-1)

    def lerp(self, o, w):
        return Pose(self.tip + (o.tip - self.tip) * w, self.a + (o.a - self.a) * w, self.d + (o.d - self.d) * w)

    def js(self):
        return dict(tip=self.tip.round(4).tolist(), a=self.a.round(5).tolist(), d=self.d.round(5).tolist())


def base_radius(x):
    if x > 0: return 0.0
    if x > -1.15:
        u = (x + 1.15) / 1.15; return RS * 0.92 * math.sqrt(max(0, 1 - u * u)) + 0.02
    waist = 0.07 * math.exp(-(((x + 1.4) / 0.5) ** 2))
    return RS * (0.92 if x > -1.6 else 1.0) - waist


def groove(x):
    x0, xm, x1 = -4.4, -2.0, -1.45
    if x <= x0 or x >= x1: return 0.0
    Dg = RS - RNECK
    if x < xm:
        t = (x - x0) / (xm - x0); return Dg * t * t * (3 - 2 * t)
    t = (x - xm) / (x1 - xm); return Dg * (1 - t ** 3)


def surf(x, phi):
    c = max(0.0, math.cos(phi))
    return max(0.05, base_radius(x) - groove(x) * min(1.0, c ** 0.7 * 1.15))


def push_out(H, P, ry=R):
    """Yarn centreline points are pushed out of the solid hook (same geometry as the rendered hook)."""
    L = H.local(P)
    for i, (x, y, z) in enumerate(L):
        if x > 0.9 or x < -60: continue
        if x > 0:
            r = math.sqrt(x * x + y * y + z * z)
            if r < ry * 0.98: P[i] = H.tip + (P[i] - H.tip) / max(r, 1e-6) * ry * 0.98
            continue
        rr = math.hypot(y, z); need = surf(x, math.atan2(z, y)) + ry * 0.98
        if rr < need:
            dy, dz = (y / rr, z / rr) if rr > 1e-6 else (-1.0, 0.0)
            P[i] = H.pt(x, dy * need, dz * need)
    return P


HOME = Pose([0, 0, 0], [-1, 0, 0], [0, 1, 0])


# ---------------------------------------------------------------- yarn pieces
def shaft_ring(H, c=None, rad=RHO, th0=25.0, th1=335.0, n=26, ax0=0.7, ax1=-0.7, shift=None):
    """Loop round the hook (or, once off the hook, round what it holds): theta 0 = bottom, 90 = front, 180 = top."""
    a = H.a
    up = norm(np.array([0, 1.0, 0]) - a * a[1]); fr = np.cross(up, a)
    c = H.pt(-XL) if c is None else c
    if shift is not None: c = c + shift
    out = []
    for k in range(n):
        u = k / (n - 1); th = math.radians(th0 + (th1 - th0) * u)
        out.append(c + rad * (-math.cos(th) * up + math.sin(th) * fr) + a * (ax0 + (ax1 - ax0) * u))
    return np.array(out)


def cinch_ring(y, n=26):
    """Finished link's top: horizontal ring (slight helix) round the next link's legs."""
    out = []
    for k in range(n):
        u = k / (n - 1); ps = math.radians(160 - 335 * u)
        z = RZ * math.sin(ps); h = -0.45 + 0.9 * u
        out.append([CX + RX * math.cos(ps), y + h * math.cos(TILT) - z * math.sin(TILT), 0.2 + z * math.cos(TILT) + h * math.sin(TILT)])
    return np.array(out)


def rise_pts(yr):
    """Strand rising behind from the previous U-turn (below the ring at yr - D) into the ring at yr - D... (link with top ring at yr)."""
    return [[XB, yr - 2 * D - 1.0, -3.0], [XB, yr - 2 * D, -4.0], [XB + 0.2, yr - D - 1.2, -2.2]]


def link_pts(yr):
    """One finished chain link whose top ring is at height yr (its legs pass through the ring at yr - D)."""
    A = rise_pts(yr) + [[CX - 0.7, yr - D, 0.3], [CX - 1.6, yr - 1.0, 1.0]]
    B = [[CX + 0.6, yr - 1.0, 0.0], [CX + 0.7, yr - D, 0.1], [CX + 0.5, yr - D - 1.1, -1.0], [CX + 0.1, yr - D - 1.4, -2.4]]
    return A, B


def tail_pts(y_lowest_ring):
    y = y_lowest_ring - D
    return [[CX + 0.3, y - 30, 0.4], [CX + 0.2, y - 15, 0.3], [CX - 0.7, y - 1.6, 0.3], [CX - 0.7, y, 0.3]]


def chain_polyline(nfin, drop):
    """Tail + nfin finished links; the top finished ring sits at YB - drop + D*(...)"""
    P = []
    y_top = YB - drop
    rings = [y_top - (nfin - 1 - j) * D for j in range(nfin)]
    lowest = rings[0] if nfin else y_top
    P += tail_pts(lowest)
    for j, yr in enumerate(rings):
        A, B = link_pts(yr)
        if j == 0: A = A[3:]  # the first link starts straight from the tail
        P += A + list(cinch_ring(yr)) + B
    return P


# ---------------------------------------------------------------- the working yarn
G = None  # finger eyelet: set once from the catch pose so the yarn really lies across the throat at the catch


def wc_point(H, toward):
    """Where the working yarn rests on the back of the loop on the hook (offset ring, facing the finger)."""
    a = H.a; c = H.pt(-XL)
    v = np.asarray(toward) - c; v = v - a * np.dot(v, a); v = norm(v)
    back = norm(np.array([0, 0, -1.0]) - a * (-a[2]))
    v = norm(v + 0.8 * back)
    return c + v * (RHO + 2 * R + 0.05) + a * 0.9  # tip side of the loop's own yarn


def throat_pts(H, n=5, span=1.3):
    """Caught yarn bent round the neck in the throat (hook-local, running along local z)."""
    return [H.pt(XTH, math.sqrt(max(0.0, (RNECK + R) ** 2 - z * z * 0.15)), z) for z in np.linspace(-span, span, n)]


def ball_pts(g):
    dirOut = norm([-0.6, 0.6, -0.5])
    return [g + dirOut * k for k in (6.0, 14.0)]


# ---------------------------------------------------------------- one stitch, as segments with fixed point counts
NS = dict(A=10, L=26, B=12, W1=22, WT=9, W2=40, BALL=4)


def seg(P, n):
    return resample(catmull(P, 8), n)


def rest_shape(H, base_drop=0.0):
    """Loop on the hook H, working yarn from the last U-turn behind, over the back of the loop, to the finger."""
    yb = YB - base_drop
    A, B = link_pts(yb + D)  # the loop on the hook is the top of a link whose base ring is at yb
    ring = shaft_ring(H)
    sA = seg(A + [ring[0]], NS["A"]); sL = resample(ring, NS["L"]); sB = seg([ring[-1]] + B, NS["B"])
    wc = wc_point(H, G)
    w_start = B[-1]
    sW1 = seg([w_start, [XB, yb, -4.1], [XB, yb + 2.0, -4.4], wc], NS["W1"])
    sWT = resample([wc, G], NS["WT"] + 2)[1:-1]  # straight: nothing holds it here yet
    sW2 = seg([sWT[-1], G], NS["W2"])
    return dict(A=sA, L=sL, B=sB, W1=sW1, WT=sWT, W2=sW2)


def caught_shape(H):
    """Same, the throat now holds the yarn: wc -> throat -> finger."""
    S = rest_shape(H)
    wc = wc_point(H, G)
    T = throat_pts(H, NS["WT"])
    # leg from the loop's back to the throat, the throat, then on to the finger (throat-side order picked so the
    # yarn runs through the throat without crossing itself)
    if np.linalg.norm(T[0] - wc) > np.linalg.norm(T[-1] - wc): T = T[::-1]
    S["WT"] = np.array(T)
    S["W2"] = seg([T[-1], G], NS["W2"])
    S["W1"] = seg([S["W1"][0], [XB, YB, -4.1], [XB, YB + 2.0, -4.4], wc, T[0]], NS["W1"])
    return S


def pull_shape(H, ring_pose, t_pass, wcf):
    """Hook pulls back through the loop (loop stays where the hand holds it, opening over the head and the
    caught yarn); legs of the caught yarn cross the loop plane only inside it once the throat has passed."""
    c0 = ring_pose.pt(-XL)
    a = ring_pose.a
    T = np.array(throat_pts(H, NS["WT"]))
    if np.linalg.norm(T[0] - wcf) > np.linalg.norm(T[-1] - wcf): T = T[::-1]
    # how far the throat is past the loop plane (mm, along -a: the hook retracts)
    past = float(np.dot(np.mean(T, axis=0) - c0, -a))
    head = float(np.dot(H.tip - c0, -a))
    bump = ease((head + 3.5) / 2.5) * (1 - ease((past - 1.5) / 3.0))  # head + caught yarn inside the loop
    rad = RHO + 1.5 * bump
    shift = H.d * (0.9 * bump) if past < 0.5 else norm(H.d - a * np.dot(H.d, a)) * 0.0
    yb = YB
    A, B = link_pts(yb + D)
    ring = shaft_ring(ring_pose, c=c0, rad=rad, shift=shift)
    # once through, both legs pass inside the loop: crossing points clamped into the loop interior
    def crossing(p_out, p_in):
        p_out, p_in = np.asarray(p_out), np.asarray(p_in)
        dpo, dpi = np.dot(p_out - c0, -a), np.dot(p_in - c0, -a)
        if dpi <= 0 or dpo >= 0: return None
        q = p_out + (p_in - p_out) * (dpo / (dpo - dpi))
        v = q - c0; v = v - a * np.dot(v, a)
        lim = max(0.0, rad - 2 * R - 0.05)
        if np.linalg.norm(v) > lim: q = c0 + norm(v) * lim
        return q
    behind = np.array([XB, yb + 2.0, -4.4])
    pa = crossing(behind, T[0]); pb = crossing(G, T[-1])
    W1 = [B[-1], [XB, yb, -4.1], behind] + ([pa] if pa is not None else []) + [T[0]]
    W2 = ([T[-1]] + ([pb] if pb is not None else []) + [G])
    return dict(A=seg(A + [ring[0]], NS["A"]), L=resample(ring, NS["L"]), B=seg([ring[-1]] + B, NS["B"]),
                W1=seg(W1, NS["W1"]), WT=T, W2=seg(W2, NS["W2"])), dict(pa=pa, pb=pb, c0=c0, rad=rad)


def next_rest_split(H, drop):
    """The finished state after the stitch, split to match the pieces of this stitch's yarn."""
    yb = YB - drop  # the new cinch ring sits at YB at the end (drop = D): this is the old loop
    A, B = link_pts(yb + D)  # old link: base ring at yb - ... (shifted with the chain)
    ring_old = cinch_ring(yb + D)
    oA = seg(A + [ring_old[0]], NS["A"]); oL = resample(ring_old, NS["L"]); oB = seg([ring_old[-1]] + B, NS["B"])
    # new loop on the hook (top of the next link, base ring = the old loop now cinched at yb + D)
    A2, B2 = link_pts(yb + 2 * D)
    ring = shaft_ring(H)
    wc = wc_point(H, G)
    nW1 = seg(A2 + [ring[0]], NS["W1"])  # rise behind -> through the old loop -> up to the hook
    nWT = resample(ring, NS["WT"])
    nW2 = seg([ring[-1]] + B2 + [[XB, yb + D, -4.1], [XB, yb + D + 2.0, -4.4], wc, G], NS["W2"])
    return dict(A=oA, L=oL, B=oB, W1=nW1, WT=nWT, W2=nW2)


def blend(S0, S1, w):
    return {k: S0[k] * (1 - w) + S1[k] * w for k in S0}


# ---------------------------------------------------------------- choreography
def catch_pose():
    """Yarn over: pivot about the loop so the throat comes up under the working yarn, throat toward it."""
    c0 = HOME.pt(-XL)
    a = rot(rot(HOME.a, [0, 0, 1], math.radians(-28)), [0, 1, 0], math.radians(-22))  # tip up, then back
    tip = c0 + a * XL
    d = norm(np.array([0, 0.55, -1.0]))
    return Pose(tip, a, d)


def setup_finger():
    """The finger holds the yarn up-left-behind the hook. The yarn-over pose puts the throat exactly on the
    working yarn's straight run (loop back -> finger), a few mm out from the loop: the hook goes to the yarn."""
    global G
    G = np.array(G_POS)
    a = rot(rot(HOME.a, [0, 0, 1], math.radians(-CATCH_PITCH)), [0, 1, 0], math.radians(-CATCH_YAW))
    H = HOME
    for _ in range(30):
        wc = wc_point(H, G); u = norm(G - wc)
        target = wc + u * CATCH_ALONG  # the bit of yarn the throat takes
        m = H.tip + a * XTH
        d = norm((target - m) - a * np.dot(target - m, a))
        tip = target - a * XTH - d * (RNECK + R)
        H = Pose(tip, a, d)
    wc = wc_point(H, G); u = norm(G - wc); T = H.pt(XTH, RNECK + R)
    print("catch residual (mm)", round(float(np.linalg.norm((T - wc) - u * np.dot(T - wc, u))), 4))
    return H


CATCH_PITCH, CATCH_YAW, CATCH_ALONG = 8.0, 6.0, 2.5
G_POS = [-18.0, 6.0, -9.0]


TURN = Pose([0.6, -0.4, 0.0], [-1, 0, 0], [0, -0.45, -0.9])  # throat turned down, back at the loop's height
PULLED = Pose([10.5, -0.4, 0.0], [-1, 0, 0], [0, -0.45, -0.9])
UPPED = Pose([10.5, -0.2, 0.0], [-1, 0, 0], [0, 1.0, 0.0])


def stitch_frames(n_done, CATCH, t0):
    """Frames of one chain stitch. Returns list of (t, pose, yarn pieces dict, chain_drop)."""
    out = []
    seq = [(0.45, 'rest'), (0.9, 'yo'), (0.6, 'turn'), (1.4, 'pull'), (0.5, 'up'), (1.1, 'home')]
    T0 = t0
    ring_pose = None; wcf = None; pull_end = None
    for dur, ph in seq:
        nfr = int(round(dur * FPS))
        for k in range(nfr):
            u = (k + 1) / nfr; w = ease(u)
            drop = 0.0
            if ph == 'rest':
                H = HOME; S = rest_shape(H)
            elif ph == 'yo':
                H = HOME.lerp(CATCH, w); S = rest_shape(H)
            elif ph == 'turn':
                H = CATCH.lerp(TURN, w); S = caught_shape(H)
                ring_pose = H; wcf = wc_point(H, G)
            elif ph == 'pull':
                H = TURN.lerp(PULLED, w); S, info = pull_shape(H, TURN, 0, wc_point(TURN, G))
                if k == 0:  # the yarn leaves the loop's back as the pull starts: ease from the caught shape
                    pass
                wpre = ease(u / 0.25)
                S = blend(caught_shape(TURN) if False else caught_shape(TURN), S, wpre) if u < 0.25 else S
                pull_end = (H, S)
            elif ph == 'up':
                H = PULLED.lerp(UPPED, w); S, info = pull_shape(H, TURN, 0, wc_point(TURN, G))
            else:  # home: the caught yarn becomes the new loop on the hook, the old loop cinches, the work drops
                H = UPPED.lerp(HOME, w); drop = D * w
                Sp, info = pull_shape(H, TURN, 0, wc_point(TURN, G))
                S = blend(Sp, next_rest_split(H, drop), w)
            out.append((T0 + (k + 1) / FPS, H, S, drop))
        T0 += nfr / FPS
    return out, T0


def assemble(n_fin, S, drop):
    chain = chain_polyline(n_fin, drop)
    Cdense = catmull(chain, 6) if len(chain) > 2 else np.array(chain)
    pieces = [Cdense, S["A"], S["L"], S["B"], S["W1"], S["WT"], S["W2"]]
    P = np.vstack(pieces)
    return P


def main():
    out_path = sys.argv[1]
    n_st = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    CATCH = setup_finger()
    frames = []
    t = 0.0
    n_fin = 2
    for k in range(n_st):
        fr, t = stitch_frames(n_fin, CATCH, t)
        for (tt, H, S, drop) in fr:
            P = assemble(n_fin, S, drop)
            P = np.vstack([P, ball_pts(G)])
            P = resample(P, int(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) / 0.3))
            # light smoothing (keeps the strand round), then the hook is solid
            for _ in range(2):
                P[1:-1] = P[1:-1] * 0.6 + (P[:-2] + P[2:]) * 0.2
            P = push_out(H, P)
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            frames.append(dict(t=round(tt, 3), stitch=k, pts=P.round(3).tolist(), s=s.round(3).tolist(), hook=H.js()))
        n_fin += 1
    cam = dict(pos=[16, 6, 46], look=[4, -3, -2], fov=34)
    json.dump(dict(fps=FPS, r=R, frames=frames, cam=cam, G=G.round(3).tolist()), open(out_path, "w"))
    print("frames", len(frames), "G", G.round(2))


if __name__ == "__main__":
    main()
