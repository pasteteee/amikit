"""Stuffed profile of one crocheted piece worked in the round.

The piece is a surface of revolution (or of a stadium for chain-oval starts). Boundary i (the top edge of round i)
has a rest circumference n_i * w; consecutive boundaries are a round height h apart along the meridian. Stuffing is
a pressure that maximises enclosed volume against the yarn's resistance to stretch, which stiffens sharply past
~15-20% strain (yarn lock-up). A flat crochet disc (+6 per round) therefore bulges into a dome when stuffed, a run of
equal rounds into a barrel, which is how real amigurumi look.

Units: millimetres. Energies are weighted by the amount of fabric they act on, so the same pressure stretches a big
head more than a thin arm (Laplace: strain ~ pressure x radius), as in real toys.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# stuffing -> pressure (dimensionless, relative to fabric stiffness)
PRESSURE = {"firm": 0.050, "medium": 0.032, "light": 0.016, "none": 0.0015}
K_LOCK = 40.0          # quartic stiffening of yarn past the elastic range
K_BEND = 0.4          # meridian bending (thick fabric rounds off corners)


def _strain_energy(s):
    # stretch is resisted strongly, compression (fabric gathering) more weakly
    return np.where(s > 0, s * s + K_LOCK * s ** 4, 2.0 * s * s)


def solve(counts, w, h, closed=True, stuffing="firm", oval_len=0.0, loops=None, start="mr"):
    """Return boundary radii r[0..M] and heights z[0..M] (M = len(counts), +1 if closed).

    counts: stitches per round. Boundary 0 is the start (magic ring centre or the chain line), boundary i the top of
    round i; for a closed piece a final boundary at r=0 is the gathered closing hole.
    oval_len: straight length of a chain-oval start (stadium cross-section).
    loops: per round 'both'|'blo'|'flo' (a back-loop round is a fold line: almost no bending stiffness there).
    """
    counts = np.asarray(counts, float)
    N = len(counts)
    L = float(oval_len)
    rest_c = counts * w
    r_rest = np.maximum((rest_c - 2 * L) / (2 * np.pi), 0.0)
    r_rest = np.concatenate([[0.0], r_rest])
    seg_len = np.full(N, h)
    if start == "mr":
        seg_len[0] = max(r_rest[1], 0.6 * h)          # the ring's stitches run radially
    if start == "ring":
        r_rest[0] = r_rest[1]                         # a chain ring (collar, snood): the start is a circle, not a point
    if loops:
        for i, lp in enumerate(loops):
            if lp == "rib":
                seg_len[i] = 2.0 * h                  # (fpdc, bpdc) ribbing: double-crochet rounds
    if closed:
        r_rest = np.concatenate([r_rest, [0.0]])
        seg_len = np.concatenate([seg_len, [max(r_rest[-2], 0.3 * h) * 0.9]])
    M = len(r_rest) - 1
    bend_w = np.ones(M + 1)
    if loops:
        for i, lp in enumerate(loops):
            if lp in ("blo", "flo"):
                bend_w[i] = 0.03                     # the fold sits at the bottom edge of the BLO round
    # initial guess: rest radii, meridian steps from the arclength rule
    dr = np.diff(r_rest)
    dz = np.sqrt(np.maximum(seg_len ** 2 - dr ** 2, (0.35 * seg_len) ** 2))
    z0 = np.concatenate([[0.0], np.cumsum(dz)])
    p = PRESSURE.get(stuffing, stuffing if isinstance(stuffing, (int, float)) else 0.03)

    free_r = np.ones(M + 1, bool)
    free_r[0] = False                                # start centre on the axis (or the chain line)
    if closed:
        free_r[M] = False
    c_rest_b = 2 * np.pi * r_rest + 2 * L            # rest perimeter at each boundary
    fab_c = np.maximum(c_rest_b, w)                  # fabric weight per boundary

    def unpack(x):
        r = r_rest.copy()
        r[free_r] = x[: free_r.sum()]
        z = x[free_r.sum():]
        z = np.concatenate([[0.0], z])
        return r, z

    def energy(x):
        r, z = unpack(x)
        r = np.maximum(r, 0.0)
        drr, dzz = np.diff(r), np.diff(z)
        seg = np.sqrt(drr ** 2 + dzz ** 2 + 1e-12)
        s_m = (seg - seg_len) / seg_len
        band_c = 0.5 * (fab_c[1:] + fab_c[:-1])
        e = np.sum(band_c * seg_len * _strain_energy(s_m)) * 0.3
        per = 2 * np.pi * r + 2 * L
        s_c = np.where(c_rest_b > 1e-9, (per - c_rest_b) / np.maximum(c_rest_b, 1e-9), 0.0)
        e += np.sum(fab_c * h * _strain_energy(s_c))
        # bending: turning angle between consecutive meridian segments
        tx, ty = drr / seg, dzz / seg
        cross = tx[:-1] * ty[1:] - ty[:-1] * tx[1:]
        dot = tx[:-1] * tx[1:] + ty[:-1] * ty[1:]
        ang = np.arctan2(cross, dot)
        e += K_BEND * np.sum(fab_c[1:-1] * bend_w[1:-1] * ang ** 2) * h / h
        # pressure: volume of stadium-revolution solid
        area = np.pi * r ** 2 + 2 * L * r
        vol = np.sum(dzz * (area[1:] + area[:-1] + np.sqrt(area[1:] * area[:-1] + 1e-12)) / 3.0)
        e -= p * vol / w
        return e

    x0 = np.concatenate([r_rest[free_r], z0[1:]])
    bounds = [(0.0, None)] * int(free_r.sum()) + [(None, None)] * M
    res = minimize(energy, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "maxfun": 200000})
    r, z = unpack(res.x)
    return np.maximum(r, 0.0), z, {"converged": bool(res.success), "energy": float(res.fun), "r_rest": r_rest}


if __name__ == "__main__":
    import sys, time
    from rounds import counts as cnt
    t = time.time()
    c = cnt(["mr 6", "inc 6 to 36", "even 6", "dec 6 to 6"])
    r, z, info = solve(c, 4.2, 3.8, closed=True)
    print(time.time() - t, info["converged"])
    print("height", z[-1] - z[0], "width", 2 * r.max(), "h/w", (z[-1] - z[0]) / (2 * r.max()))
    print(np.round(r, 1)); print(np.round(z, 1))
