"""Quasi-static yarn solver.

Model
-----
The yarn is a chain of nodes ``h`` mm apart with radius ``r``.  There are no
velocities, no gravity and no stored elastic energy.  One substep is a
mass-weighted projection of the previous positions onto the constraints

* inextensible segments,
* non-penetration against the hook (signed distance field) with Coulomb friction
  against the moving steel,
* non-penetration yarn against yarn (capsule/capsule) with Coulomb friction,
* a minimum bend radius, and a viscous resistance to *changes* of curvature
  (yarn has almost no springiness: it keeps the shape it was put in),
* the throat of the hook holding the strand it caught (the yarn slides through it),
* the feeder: yarn enters the scene only there, against a drag.

This is implicit overdamped dynamics: the multipliers of the projection are the
real contact and tension forces of a rope dragged against friction, so the
capstan effect (a strand wrapped round another strand locks) comes out of the
solve.  If every constraint already holds, the projection is the identity:
a node that nothing pulls does not move at all, bit for bit.

Node 0 is the tail end; the last node sits at the feeder.
"""
import math
import numpy as np
from numba import njit

from .hook import hook_sdf, hook_sdf_grad

# indices into prm
LINE_T = 9.0
CONV = 2e-4
S_NIT = 7
P_ITERS, P_MU_YY, P_MU_HOOK, P_BEND_ALPHA, P_DMIN, P_RC_YARN, P_RC_HOOK, P_CAP_EPS, P_NSKIP, P_MARGIN, P_TOL, \
    P_MAXCORR = range(12)
# indices into stats
S_MAXDISP, S_STRETCH, S_NPAIRS, S_HOOKPEN, S_SELFPEN, S_CAPDIST, S_FED = range(7)


def default_params(h, r):
    rho_min = 1.25 * r
    th = h / rho_min
    p = np.zeros(16)
    p[P_ITERS] = 60
    p[P_MU_YY] = 0.45
    p[P_MU_HOOK] = 0.18
    p[P_BEND_ALPHA] = 6.0
    p[P_DMIN] = 2 * h * math.cos(th / 2)
    p[P_RC_YARN] = 0.93 * r
    p[P_RC_HOOK] = 0.93 * r
    p[P_CAP_EPS] = 0.15
    p[P_NSKIP] = max(4, int(math.ceil(2.6 * r / h)) + 1)
    p[P_MARGIN] = 0.5
    p[P_TOL] = 1e-7
    p[P_MAXCORR] = 0.3 * r
    p[12] = 1.0      # yarn held in the throat slides along itself in the chain solve
    p[13] = 0.0     # friction yarn/steel inside the chain solve
    p[14] = 0.0     # friction yarn/yarn inside the chain solve
    p[15] = 0.1      # how near a surface must be to count as a wall, mm
    return p


@njit(cache=True)
def _closest_seg_seg(p1, q1, p2, q2):
    d1x = q1[0] - p1[0]
    d1y = q1[1] - p1[1]
    d1z = q1[2] - p1[2]
    d2x = q2[0] - p2[0]
    d2y = q2[1] - p2[1]
    d2z = q2[2] - p2[2]
    rx = p1[0] - p2[0]
    ry = p1[1] - p2[1]
    rz = p1[2] - p2[2]
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    if a <= 1e-12 and e <= 1e-12:
        return 0.0, 0.0
    if a <= 1e-12:
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= 1e-12:
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            den = a * e - b * b
            if den > 1e-12:
                s = min(max((b * f - c * e) / den, 0.0), 1.0)
            else:
                s = 0.5
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)
    return s, t


@njit(cache=True)
def build_pairs(X, W, n, h, reach, nskip, pairs, head, nxt, lo_active, hi_active):
    """Candidate segment pairs (i<j, j-i>=nskip) whose midpoints are within reach.
    Pairs of two fully pinned segments are skipped."""
    M = head.shape[0]
    cs = reach
    inv = 1.0 / cs
    for k in range(M):
        head[k] = -1
    nseg = n - 1
    for i in range(nseg):
        mx = 0.5 * (X[i, 0] + X[i + 1, 0])
        my = 0.5 * (X[i, 1] + X[i + 1, 1])
        mz = 0.5 * (X[i, 2] + X[i + 1, 2])
        cx = int(math.floor(mx * inv))
        cy = int(math.floor(my * inv))
        cz = int(math.floor(mz * inv))
        key = ((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) % M
        nxt[i] = head[key]
        head[key] = i
    np_ = 0
    maxp = pairs.shape[0]
    r2 = reach * reach
    for i in range(nseg):
        pin_i = W[i] == 0.0 and W[i + 1] == 0.0
        mx = 0.5 * (X[i, 0] + X[i + 1, 0])
        my = 0.5 * (X[i, 1] + X[i + 1, 1])
        mz = 0.5 * (X[i, 2] + X[i + 1, 2])
        cx = int(math.floor(mx * inv))
        cy = int(math.floor(my * inv))
        cz = int(math.floor(mz * inv))
        for ax in range(-1, 2):
            for ay in range(-1, 2):
                for az in range(-1, 2):
                    key = (((cx + ax) * 73856093) ^ ((cy + ay) * 19349663) ^ ((cz + az) * 83492791)) % M
                    j = head[key]
                    while j >= 0:
                        if j >= i + nskip:
                            if not (pin_i and W[j] == 0.0 and W[j + 1] == 0.0):
                                dx = 0.5 * (X[j, 0] + X[j + 1, 0]) - mx
                                dy = 0.5 * (X[j, 1] + X[j + 1, 1]) - my
                                dz = 0.5 * (X[j, 2] + X[j + 1, 2]) - mz
                                if dx * dx + dy * dy + dz * dz < r2:
                                    if np_ < maxp:
                                        pairs[np_, 0] = i
                                        pairs[np_, 1] = j
                                        np_ += 1
                        j = nxt[j]
    return np_


@njit(cache=True)
def _pdot(Pm, i, ax, ay, az, bx, by, bz):
    """a^T P_i b with P stored as xx,xy,xz,yy,yz,zz."""
    return (ax * (Pm[i, 0] * bx + Pm[i, 1] * by + Pm[i, 2] * bz)
            + ay * (Pm[i, 1] * bx + Pm[i, 3] * by + Pm[i, 4] * bz)
            + az * (Pm[i, 2] * bx + Pm[i, 4] * by + Pm[i, 5] * bz))


@njit(cache=True)
def _tri_build(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, U, Pm, pressed, use_proj):
    """Tridiagonal system J W J^T lam = -C of the chain's length constraints (+ feeder row,
    feeder locked).  With use_proj a pressed node i has the mobility W[i] * P_i (P_i projects
    on the directions it may still move in: along a surface, along a line, or not at all)."""
    m = n - 1
    mm = m + 1 if feeder else m
    worst = 0.0
    for i in range(mm):
        if i < m:
            ux = X[i + 1, 0] - X[i, 0]
            uy = X[i + 1, 1] - X[i, 1]
            uz = X[i + 1, 2] - X[i, 2]
            dl = math.sqrt(ux * ux + uy * uy + uz * uz) + 1e-30
            C = dl - h
            wa = W[i]
            wb = W[i + 1]
        else:
            ux, uy, uz, dl = fx, fy, fz, fd
            C = fd - feed[3]
            wa = W[n - 1]
            wb = 0.0
        ux /= dl
        uy /= dl
        uz /= dl
        U[i, 0] = ux
        U[i, 1] = uy
        U[i, 2] = uz
        ka = wa
        if use_proj and pressed[i] > 0:
            ka = wa * _pdot(Pm, i, ux, uy, uz, ux, uy, uz)
        kb = wb
        if i < m and use_proj and pressed[i + 1] > 0:
            kb = wb * _pdot(Pm, i + 1, ux, uy, uz, ux, uy, uz)
        if ka + kb < 1e-9:
            # nothing on this segment can move: leave it alone
            b[i] = 1.0
            d[i] = 0.0
            ka = 0.0
        else:
            b[i] = ka + kb + 1e-6
            d[i] = -C
        if abs(C) > worst:
            worst = abs(C)
        if i > 0:
            if use_proj and pressed[i] > 0:
                dd = _pdot(Pm, i, ux, uy, uz, U[i - 1, 0], U[i - 1, 1], U[i - 1, 2])
            else:
                dd = ux * U[i - 1, 0] + uy * U[i - 1, 1] + uz * U[i - 1, 2]
            a[i] = -wa * dd
            c[i - 1] = a[i]
    return worst


@njit(cache=True)
def _thomas(a, b, c, d, lam, k):
    for i in range(1, k):
        w_ = a[i] / b[i - 1]
        b[i] -= w_ * c[i - 1]
        d[i] -= w_ * d[i - 1]
    lam[k - 1] = d[k - 1] / b[k - 1]
    for i in range(k - 2, -1, -1):
        lam[i] = (d[i] - c[i] * lam[i + 1]) / b[i]


@njit(cache=True)
def _chain_solve(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, lam, U, Pm, pressed, use_proj):
    """Solve with the feeder as a friction clutch: it holds the yarn up to the tension the hand
    still has to give in this substep (feed[4] - feed[6]) and lets it slip beyond that."""
    m = n - 1
    worst = _tri_build(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, U, Pm, pressed, use_proj)
    if not feeder:
        _thomas(a, b, c, d, lam, m)
        return worst, False
    _thomas(a, b, c, d, lam, m + 1)
    room = feed[4] - feed[6]
    if room < 0.0:
        room = 0.0
    if lam[m] > 0.0 or -lam[m] <= room:
        if lam[m] > 0.0:
            # would push: a rope cannot; treat as slack feeder
            _tri_build(X, W, n, h, feed, False, fx, fy, fz, fd, a, b, c, d, U, Pm, pressed, use_proj)
            _thomas(a, b, c, d, lam, m)
            lam[m] = 0.0
            return worst, True
        return worst, False
    # slip: the clutch pulls with what is left of its tension and pays out the rest
    _tri_build(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, U, Pm, pressed, use_proj)
    d[m - 1] -= c[m - 1] * (-room)
    _thomas(a, b, c, d, lam, m)
    lam[m] = -room
    return worst, True


@njit(cache=True)
def _node_force(i, n, feeder, U, lam, out):
    """Constraint force on node i (per unit mobility): -u_i lam_i + u_{i-1} lam_{i-1}."""
    m = n - 1
    gx = 0.0
    gy = 0.0
    gz = 0.0
    if i < m:
        gx -= U[i, 0] * lam[i]
        gy -= U[i, 1] * lam[i]
        gz -= U[i, 2] * lam[i]
    elif feeder:
        gx -= U[m, 0] * lam[m]
        gy -= U[m, 1] * lam[m]
        gz -= U[m, 2] * lam[m]
    if i > 0:
        gx += U[i - 1, 0] * lam[i - 1]
        gy += U[i - 1, 1] * lam[i - 1]
        gz += U[i - 1, 2] * lam[i - 1]
    out[0] = gx
    out[1] = gy
    out[2] = gz


@njit(cache=True)
def _stretch_direct(X, W, n, h, feed, a, b, c, d, lam, tol, U, Pm, pressed, cn, cg, cmu, ccnt, capt, Ps, fmu):
    """One Newton step of the mass-weighted projection onto all segment-length constraints at
    once (tridiagonal system, Thomas algorithm), with the feeder as the last constraint.

    Contacts enter the solve: a node that the tension would pull into the hook or into another
    piece of yarn is put on that surface and may only slide along it, and it slides only if the
    pull along the surface beats Coulomb friction mu * (the force pressing it on).  That is the
    capstan: yarn wrapped round yarn holds until the pull exceeds exp(mu * angle).
    Returns the length paid out."""
    m = n - 1
    feeder = False
    feed[7] = 0.0
    fx = 0.0
    fy = 0.0
    fz = 0.0
    fd = 1.0
    if feed[5] > 0.5:
        e = n - 1
        fx = feed[0] - X[e, 0]
        fy = feed[1] - X[e, 1]
        fz = feed[2] - X[e, 2]
        fd = math.sqrt(fx * fx + fy * fy + fz * fz)
        if fd > feed[3] + tol:
            feeder = True
    g = np.zeros(3)
    worst, slip = _chain_solve(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, lam, U, Pm, pressed, False)
    if worst < tol:
        return 0.0
    # ---- which walls are active?  Build each pressed node's sliding projector Ps.
    anyp = False
    anyfric = False
    for i in range(n):
        pressed[i] = 0
        if W[i] == 0.0:
            continue
        if capt[i] == 1:
            # held in the throat: slides along itself only (direction in cn[i,0])
            tx = cn[i, 0, 0]
            ty = cn[i, 0, 1]
            tz = cn[i, 0, 2]
            Ps[i, 0] = tx * tx
            Ps[i, 1] = tx * ty
            Ps[i, 2] = tx * tz
            Ps[i, 3] = ty * ty
            Ps[i, 4] = ty * tz
            Ps[i, 5] = tz * tz
            fmu[i] = 0.0
            pressed[i] = 2
            anyp = True
            continue
        if ccnt[i] == 0:
            continue
        _node_force(i, n, feeder, U, lam, g)
        na = 0
        k1 = -1
        k2 = -1
        for k in range(ccnt[i]):
            gn = (g[0] * cn[i, k, 0] + g[1] * cn[i, k, 1] + g[2] * cn[i, k, 2]) * W[i]
            if gn < -cg[i, k]:
                if na == 0:
                    k1 = k
                else:
                    k2 = k
                na += 1
        if na == 0:
            continue
        # put the node on the wall(s)
        X[i, 0] -= cg[i, k1] * cn[i, k1, 0]
        X[i, 1] -= cg[i, k1] * cn[i, k1, 1]
        X[i, 2] -= cg[i, k1] * cn[i, k1, 2]
        cg[i, k1] = 0.0
        n1x = cn[i, k1, 0]
        n1y = cn[i, k1, 1]
        n1z = cn[i, k1, 2]
        mu_ = cmu[i, k1]
        line = False
        if na == 2:
            X[i, 0] -= cg[i, k2] * cn[i, k2, 0]
            X[i, 1] -= cg[i, k2] * cn[i, k2, 1]
            X[i, 2] -= cg[i, k2] * cn[i, k2, 2]
            cg[i, k2] = 0.0
            tx = n1y * cn[i, k2, 2] - n1z * cn[i, k2, 1]
            ty = n1z * cn[i, k2, 0] - n1x * cn[i, k2, 2]
            tz = n1x * cn[i, k2, 1] - n1y * cn[i, k2, 0]
            tl_ = math.sqrt(tx * tx + ty * ty + tz * tz)
            if cmu[i, k2] > mu_:
                mu_ = cmu[i, k2]
            if tl_ > LINE_T:
                tx /= tl_
                ty /= tl_
                tz /= tl_
                Ps[i, 0] = tx * tx
                Ps[i, 1] = tx * ty
                Ps[i, 2] = tx * tz
                Ps[i, 3] = ty * ty
                Ps[i, 4] = ty * tz
                Ps[i, 5] = tz * tz
                line = True
        if not line:
            Ps[i, 0] = 1.0 - n1x * n1x
            Ps[i, 1] = -n1x * n1y
            Ps[i, 2] = -n1x * n1z
            Ps[i, 3] = 1.0 - n1y * n1y
            Ps[i, 4] = -n1y * n1z
            Ps[i, 5] = 1.0 - n1z * n1z
        fmu[i] = mu_
        pressed[i] = 1
        anyp = True
        if mu_ > 0.0:
            anyfric = True
        # remember the wall normals for the friction test (store the summed normal in cn[i,0])
        if k1 != 0:
            cn[i, 0, 0] = n1x
            cn[i, 0, 1] = n1y
            cn[i, 0, 2] = n1z
        if na == 2:
            if k2 != 1 and k1 != 1:
                cn[i, 1, 0] = cn[i, k2, 0]
                cn[i, 1, 1] = cn[i, k2, 1]
                cn[i, 1, 2] = cn[i, k2, 2]
            ccnt[i] = 2
        else:
            ccnt[i] = 1
    if anyp:
        if anyfric:
            # ---- stick pass: frictional contacts hold; see which ones the pull tears loose
            for i in range(n):
                if pressed[i] == 1 and fmu[i] > 0.0:
                    for q in range(6):
                        Pm[i, q] = 0.0
                elif pressed[i] > 0:
                    for q in range(6):
                        Pm[i, q] = Ps[i, q]
            worst, slip = _chain_solve(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, lam, U, Pm, pressed, True)
            anyslide = False
            for i in range(n):
                if pressed[i] == 1 and fmu[i] > 0.0:
                    _node_force(i, n, feeder, U, lam, g)
                    fn = 0.0
                    for k in range(ccnt[i]):
                        gn = g[0] * cn[i, k, 0] + g[1] * cn[i, k, 1] + g[2] * cn[i, k, 2]
                        if gn < 0.0:
                            fn -= gn
                    tx = Ps[i, 0] * g[0] + Ps[i, 1] * g[1] + Ps[i, 2] * g[2]
                    ty = Ps[i, 1] * g[0] + Ps[i, 3] * g[1] + Ps[i, 4] * g[2]
                    tz = Ps[i, 2] * g[0] + Ps[i, 4] * g[1] + Ps[i, 5] * g[2]
                    ft = math.sqrt(tx * tx + ty * ty + tz * tz)
                    if ft > fmu[i] * fn + 1e-12:
                        kap = 1.0 - fmu[i] * fn / ft
                        for q in range(6):
                            Pm[i, q] = kap * Ps[i, q]
                        anyslide = True
            if anyslide:
                worst, slip = _chain_solve(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, lam, U, Pm, pressed,
                                           True)
        else:
            for i in range(n):
                if pressed[i] > 0:
                    for q in range(6):
                        Pm[i, q] = Ps[i, q]
            worst, slip = _chain_solve(X, W, n, h, feed, feeder, fx, fy, fz, fd, a, b, c, d, lam, U, Pm, pressed, True)
    lf = 0.0
    if feeder:
        lf = lam[m]
        feed[6] -= lf
        feed[7] = -lf
    for i in range(n):
        if W[i] == 0.0:
            continue
        _node_force(i, n, feeder, U, lam, g)
        if anyp and pressed[i] > 0:
            gx = Pm[i, 0] * g[0] + Pm[i, 1] * g[1] + Pm[i, 2] * g[2]
            gy = Pm[i, 1] * g[0] + Pm[i, 3] * g[1] + Pm[i, 4] * g[2]
            gz = Pm[i, 2] * g[0] + Pm[i, 4] * g[1] + Pm[i, 5] * g[2]
        else:
            gx = g[0]
            gy = g[1]
            gz = g[2]
        X[i, 0] += W[i] * gx
        X[i, 1] += W[i] * gy
        X[i, 2] += W[i] * gz
    fed = 0.0
    if feeder and slip:
        e = n - 1
        ex = feed[0] - X[e, 0]
        ey = feed[1] - X[e, 1]
        ez = feed[2] - X[e, 2]
        nd = math.sqrt(ex * ex + ey * ey + ez * ez)
        if nd > feed[3]:
            fed = nd - feed[3]
            feed[3] = nd
    return fed


@njit(cache=True)
def _add_wall(i, nx, ny, nz, gap, mu, cn, cg, cmu, ccnt):
    """Remember that node i has a wall with outward normal n at distance gap (at most two,
    nearly parallel ones are merged, the nearest are kept)."""
    for k in range(ccnt[i]):
        if cn[i, k, 0] * nx + cn[i, k, 1] * ny + cn[i, k, 2] * nz > 0.85:
            if gap < cg[i, k]:
                cn[i, k, 0] = nx
                cn[i, k, 1] = ny
                cn[i, k, 2] = nz
                cg[i, k] = gap
            if mu > cmu[i, k]:
                cmu[i, k] = mu
            return
    if ccnt[i] < 2:
        k = ccnt[i]
        ccnt[i] += 1
    else:
        k = 0 if cg[i, 0] > cg[i, 1] else 1
        if gap >= cg[i, k]:
            return
    cn[i, k, 0] = nx
    cn[i, k, 1] = ny
    cn[i, k, 2] = nz
    cg[i, k] = gap
    cmu[i, k] = mu


@njit(cache=True)
def substep(X, Xp, W, n, h, r, Rp, tp, Rn, tn, hp, prm, cap, feed, grip,
            lamh, lamht, pairs, plam, plamt, lamb, flags, Dv, head, nxt, stats, ta, tb, tc, td, tl, U, Pm, pressed, cn, cg, cmu, ccnt, capt, Ps, fmu, hp2):
    iters = int(prm[P_ITERS])
    mu = prm[P_MU_YY]
    muh = prm[P_MU_HOOK]
    muh_s = prm[13]
    mu_s = prm[14]
    wall_reach = prm[15]
    alpha = prm[P_BEND_ALPHA]
    dmin = prm[P_DMIN]
    rcy = prm[P_RC_YARN]
    rch = prm[P_RC_HOOK]
    cap_eps = prm[P_CAP_EPS]
    nskip = int(prm[P_NSKIP])
    margin = prm[P_MARGIN]
    tol = prm[P_TOL]
    maxcorr = prm[P_MAXCORR]

    for i in range(n):
        Xp[i, 0] = X[i, 0]
        Xp[i, 1] = X[i, 1]
        Xp[i, 2] = X[i, 2]
        flags[i] = 0
        lamb[i, 0] = 0.0
        lamb[i, 1] = 0.0
        lamb[i, 2] = 0.0
    for i in range(2 * n):
        lamh[i] = 0.0
        lamht[i, 0] = 0.0
        lamht[i, 1] = 0.0
        lamht[i, 2] = 0.0

    # kinematic grips (nodes with W == 0 that a tool carries): grip[k] = [node, dx, dy, dz]
    for k in range(grip.shape[0]):
        gi = int(grip[k, 0])
        if gi >= 0:
            X[gi, 0] += grip[k, 1]
            X[gi, 1] += grip[k, 2]
            X[gi, 2] += grip[k, 3]

    feed[6] = 0.0
    reach = 2.0 * rcy + h + margin
    npairs = build_pairs(X, W, n, h, reach, nskip, pairs, head, nxt, 0, n)
    for k in range(npairs):
        plam[k] = 0.0
        plamt[k, 0] = 0.0
        plamt[k, 1] = 0.0
        plamt[k, 2] = 0.0

    L = hp[4]
    Rs = hp[0]
    rad_reach = Rs + rch + 0.6
    fed = 0.0
    capdist = 0.0
    hookpen = 0.0
    selfpen = 0.0
    pl = np.zeros(3)
    g = np.zeros(3)
    xm = np.zeros(3)
    xmp = np.zeros(3)

    Xi = np.empty((n, 3))
    conv = prm[P_TOL] * 0.0 + CONV
    nit = 0
    for it in range(iters):
        last = True
        nit += 1
        hookpen = 0.0
        selfpen = 0.0
        for i in range(n):
            Xi[i, 0] = X[i, 0]
            Xi[i, 1] = X[i, 1]
            Xi[i, 2] = X[i, 2]
        # ---------------------------------------------------------- capture
        if cap[0] > 0.5:
            qx = Rn[0, 0] * hp[11] + Rn[0, 1] * hp[12] + tn[0]
            qy = Rn[1, 0] * hp[11] + Rn[1, 1] * hp[12] + tn[1]
            qz = Rn[2, 0] * hp[11] + Rn[2, 1] * hp[12] + tn[2]
            k0 = int(cap[1])
            best = 1e30
            bk = k0
            bs = 0.0
            for k in range(max(0, k0 - 3), min(n - 1, k0 + 4)):
                ex = X[k + 1, 0] - X[k, 0]
                ey = X[k + 1, 1] - X[k, 1]
                ez = X[k + 1, 2] - X[k, 2]
                ee = ex * ex + ey * ey + ez * ez
                s = ((qx - X[k, 0]) * ex + (qy - X[k, 1]) * ey + (qz - X[k, 2]) * ez) / (ee + 1e-12)
                s = min(max(s, 0.0), 1.0)
                dx = X[k, 0] + s * ex - qx
                dy = X[k, 1] + s * ey - qy
                dz = X[k, 2] + s * ez - qz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best:
                    best = d2
                    bk = k
                    bs = s
            cap[1] = bk + bs
            dist = math.sqrt(best)
            capdist = dist
            if dist > cap_eps:
                k = bk
                s = bs
                px = X[k, 0] + s * (X[k + 1, 0] - X[k, 0])
                py = X[k, 1] + s * (X[k + 1, 1] - X[k, 1])
                pz = X[k, 2] + s * (X[k + 1, 2] - X[k, 2])
                cc = (dist - cap_eps) / dist
                wa = W[k] * (1 - s)
                wb = W[k + 1] * s
                den = wa * (1 - s) + wb * s + 1e-12
                X[k, 0] += wa * cc * (qx - px) / den
                X[k, 1] += wa * cc * (qy - py) / den
                X[k, 2] += wa * cc * (qz - pz) / den
                X[k + 1, 0] += wb * cc * (qx - px) / den
                X[k + 1, 1] += wb * cc * (qy - py) / den
                X[k + 1, 2] += wb * cc * (qz - pz) / den
                flags[k] |= 4
                flags[k + 1] |= 4

        # ----------------------------------------------------- hook contact
        for i in range(n):
            ccnt[i] = 0
            capt[i] = 0
        for pass_ in range(2):
            cnt = n if pass_ == 0 else n - 1
            for i in range(cnt):
                if pass_ == 0:
                    if W[i] == 0.0:
                        continue
                    xm[0] = X[i, 0]
                    xm[1] = X[i, 1]
                    xm[2] = X[i, 2]
                else:
                    if W[i] == 0.0 and W[i + 1] == 0.0:
                        continue
                    xm[0] = 0.5 * (X[i, 0] + X[i + 1, 0])
                    xm[1] = 0.5 * (X[i, 1] + X[i + 1, 1])
                    xm[2] = 0.5 * (X[i, 2] + X[i + 1, 2])
                rx = xm[0] - tn[0]
                ry = xm[1] - tn[1]
                rz = xm[2] - tn[2]
                pl[0] = Rn[0, 0] * rx + Rn[1, 0] * ry + Rn[2, 0] * rz
                if pl[0] < -rch - 0.6 or pl[0] > L:
                    continue
                pl[1] = Rn[0, 1] * rx + Rn[1, 1] * ry + Rn[2, 1] * rz
                pl[2] = Rn[0, 2] * rx + Rn[1, 2] * ry + Rn[2, 2] * rz
                if pl[1] * pl[1] + pl[2] * pl[2] > rad_reach * rad_reach:
                    continue
                # a throat that holds yarn is full: other yarn slides over it as over the shaft
                hq = hp
                if cap[0] > 0.5 and abs(i - cap[1]) > 7.0:
                    hq = hp2
                d = hook_sdf(pl[0], pl[1], pl[2], hq)
                if d >= rch + wall_reach:
                    continue
                d, g0, g1, g2 = hook_sdf_grad(pl[0], pl[1], pl[2], hq)
                nx = Rn[0, 0] * g0 + Rn[0, 1] * g1 + Rn[0, 2] * g2
                ny = Rn[1, 0] * g0 + Rn[1, 1] * g1 + Rn[1, 2] * g2
                nz = Rn[2, 0] * g0 + Rn[2, 1] * g1 + Rn[2, 2] * g2
                gap_ = d - rch
                if gap_ < 0.0:
                    gap_ = 0.0
                if pass_ == 0:
                    _add_wall(i, nx, ny, nz, gap_, muh_s, cn, cg, cmu, ccnt)
                else:
                    _add_wall(i, nx, ny, nz, gap_, muh_s, cn, cg, cmu, ccnt)
                    _add_wall(i + 1, nx, ny, nz, gap_, muh_s, cn, cg, cmu, ccnt)
                if d >= rch:
                    continue
                pen = rch - d
                if pen > maxcorr:
                    corr = maxcorr
                else:
                    corr = pen
                idx = i if pass_ == 0 else n + i
                lamh[idx] += corr
                if last and pen > hookpen:
                    hookpen = pen
                # where the steel under this point went during the substep
                if pass_ == 0:
                    xmp[0] = Xp[i, 0]
                    xmp[1] = Xp[i, 1]
                    xmp[2] = Xp[i, 2]
                else:
                    xmp[0] = 0.5 * (Xp[i, 0] + Xp[i + 1, 0])
                    xmp[1] = 0.5 * (Xp[i, 1] + Xp[i + 1, 1])
                    xmp[2] = 0.5 * (Xp[i, 2] + Xp[i + 1, 2])
                rx = xmp[0] - tp[0]
                ry = xmp[1] - tp[1]
                rz = xmp[2] - tp[2]
                l0 = Rp[0, 0] * rx + Rp[1, 0] * ry + Rp[2, 0] * rz
                l1 = Rp[0, 1] * rx + Rp[1, 1] * ry + Rp[2, 1] * rz
                l2 = Rp[0, 2] * rx + Rp[1, 2] * ry + Rp[2, 2] * rz
                ux = Rn[0, 0] * l0 + Rn[0, 1] * l1 + Rn[0, 2] * l2 + tn[0] - xmp[0]
                uy = Rn[1, 0] * l0 + Rn[1, 1] * l1 + Rn[1, 2] * l2 + tn[1] - xmp[1]
                uz = Rn[2, 0] * l0 + Rn[2, 1] * l1 + Rn[2, 2] * l2 + tn[2] - xmp[2]
                # relative slip of the yarn on the steel (after the normal push)
                sx = xm[0] + corr * nx - xmp[0] - ux
                sy = xm[1] + corr * ny - xmp[1] - uy
                sz = xm[2] + corr * nz - xmp[2] - uz
                sn = sx * nx + sy * ny + sz * nz
                sx -= sn * nx
                sy -= sn * ny
                sz -= sn * nz
                tx = lamht[idx, 0] - sx
                ty = lamht[idx, 1] - sy
                tz = lamht[idx, 2] - sz
                tn_ = math.sqrt(tx * tx + ty * ty + tz * tz)
                lim = muh * lamh[idx]
                if tn_ > lim:
                    sc = lim / (tn_ + 1e-30)
                    tx *= sc
                    ty *= sc
                    tz *= sc
                fx = corr * nx + tx - lamht[idx, 0]
                fy = corr * ny + ty - lamht[idx, 1]
                fz = corr * nz + tz - lamht[idx, 2]
                lamht[idx, 0] = tx
                lamht[idx, 1] = ty
                lamht[idx, 2] = tz
                if pass_ == 0:
                    X[i, 0] += fx
                    X[i, 1] += fy
                    X[i, 2] += fz
                    flags[i] |= 1
                else:
                    wa = W[i]
                    wb = W[i + 1]
                    den = 0.25 * (wa + wb)
                    X[i, 0] += 0.5 * wa * fx / den
                    X[i, 1] += 0.5 * wa * fy / den
                    X[i, 2] += 0.5 * wa * fz / den
                    X[i + 1, 0] += 0.5 * wb * fx / den
                    X[i + 1, 1] += 0.5 * wb * fy / den
                    X[i + 1, 2] += 0.5 * wb * fz / den
                    flags[i] |= 1
                    flags[i + 1] |= 1

        # ------------------------------------------------------ self contact
        for k in range(npairs):
            i = pairs[k, 0]
            j = pairs[k, 1]
            s, t = _closest_seg_seg(X[i], X[i + 1], X[j], X[j + 1])
            ax = X[i, 0] + s * (X[i + 1, 0] - X[i, 0])
            ay = X[i, 1] + s * (X[i + 1, 1] - X[i, 1])
            az = X[i, 2] + s * (X[i + 1, 2] - X[i, 2])
            bx = X[j, 0] + t * (X[j + 1, 0] - X[j, 0])
            by = X[j, 1] + t * (X[j + 1, 1] - X[j, 1])
            bz = X[j, 2] + t * (X[j + 1, 2] - X[j, 2])
            dx = ax - bx
            dy = ay - by
            dz = az - bz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            C = d - 2.0 * rcy
            if d < 1e-9:
                continue
            nx = dx / d
            ny = dy / d
            nz = dz / d
            if C < wall_reach:
                # each side sees the other as a wall half the gap away
                gh = 0.5 * C if C > 0.0 else 0.0
                if s < 0.8:
                    _add_wall(i, nx, ny, nz, gh, mu_s, cn, cg, cmu, ccnt)
                if s > 0.2:
                    _add_wall(i + 1, nx, ny, nz, gh, mu_s, cn, cg, cmu, ccnt)
                if t < 0.8:
                    _add_wall(j, -nx, -ny, -nz, gh, mu_s, cn, cg, cmu, ccnt)
                if t > 0.2:
                    _add_wall(j + 1, -nx, -ny, -nz, gh, mu_s, cn, cg, cmu, ccnt)
            if C >= 0.0 and plam[k] == 0.0:
                continue
            w0 = W[i] * (1 - s)
            w1 = W[i + 1] * s
            w2 = W[j] * (1 - t)
            w3 = W[j + 1] * t
            den = w0 * (1 - s) + w1 * s + w2 * (1 - t) + w3 * t
            if den < 1e-12:
                continue
            dl = 0.0
            if C < 0.0:
                dl = -C / den
                if dl * den > maxcorr:
                    dl = maxcorr / den
                plam[k] += dl
                if last and -C > selfpen:
                    selfpen = -C
            # friction: cancel the relative tangential slip since the substep began
            rx = ((1 - s) * (X[i, 0] - Xp[i, 0]) + s * (X[i + 1, 0] - Xp[i + 1, 0])
                  - (1 - t) * (X[j, 0] - Xp[j, 0]) - t * (X[j + 1, 0] - Xp[j + 1, 0]))
            ry = ((1 - s) * (X[i, 1] - Xp[i, 1]) + s * (X[i + 1, 1] - Xp[i + 1, 1])
                  - (1 - t) * (X[j, 1] - Xp[j, 1]) - t * (X[j + 1, 1] - Xp[j + 1, 1]))
            rz = ((1 - s) * (X[i, 2] - Xp[i, 2]) + s * (X[i + 1, 2] - Xp[i + 1, 2])
                  - (1 - t) * (X[j, 2] - Xp[j, 2]) - t * (X[j + 1, 2] - Xp[j + 1, 2]))
            rn = rx * nx + ry * ny + rz * nz
            rx -= rn * nx
            ry -= rn * ny
            rz -= rn * nz
            tx = plamt[k, 0] - rx / den
            ty = plamt[k, 1] - ry / den
            tz = plamt[k, 2] - rz / den
            tn_ = math.sqrt(tx * tx + ty * ty + tz * tz)
            lim = mu * plam[k]
            if tn_ > lim:
                sc = lim / (tn_ + 1e-30)
                tx *= sc
                ty *= sc
                tz *= sc
            fx = dl * nx + tx - plamt[k, 0]
            fy = dl * ny + ty - plamt[k, 1]
            fz = dl * nz + tz - plamt[k, 2]
            plamt[k, 0] = tx
            plamt[k, 1] = ty
            plamt[k, 2] = tz
            X[i, 0] += w0 * fx
            X[i, 1] += w0 * fy
            X[i, 2] += w0 * fz
            X[i + 1, 0] += w1 * fx
            X[i + 1, 1] += w1 * fy
            X[i + 1, 2] += w1 * fz
            X[j, 0] -= w2 * fx
            X[j, 1] -= w2 * fy
            X[j, 2] -= w2 * fz
            X[j + 1, 0] -= w3 * fx
            X[j + 1, 1] -= w3 * fy
            X[j + 1, 2] -= w3 * fz
            if plam[k] > 0.0:
                flags[i] |= 2
                flags[i + 1] |= 2
                flags[j] |= 2
                flags[j + 1] |= 2

        # -------------------------------------------------------------- bend
        for i in range(1, n - 1):
            wa = W[i - 1]
            wb = W[i]
            wc = W[i + 1]
            ws = wa + 4.0 * wb + wc
            if ws == 0.0:
                continue
            cx = (X[i - 1, 0] - 2 * X[i, 0] + X[i + 1, 0]) - (Xp[i - 1, 0] - 2 * Xp[i, 0] + Xp[i + 1, 0])
            cy = (X[i - 1, 1] - 2 * X[i, 1] + X[i + 1, 1]) - (Xp[i - 1, 1] - 2 * Xp[i, 1] + Xp[i + 1, 1])
            cz = (X[i - 1, 2] - 2 * X[i, 2] + X[i + 1, 2]) - (Xp[i - 1, 2] - 2 * Xp[i, 2] + Xp[i + 1, 2])
            if abs(cx) + abs(cy) + abs(cz) > tol or lamb[i, 0] != 0.0 or lamb[i, 1] != 0.0 or lamb[i, 2] != 0.0:
                ix = (-cx - alpha * lamb[i, 0]) / (ws + alpha)
                iy = (-cy - alpha * lamb[i, 1]) / (ws + alpha)
                iz = (-cz - alpha * lamb[i, 2]) / (ws + alpha)
                lamb[i, 0] += ix
                lamb[i, 1] += iy
                lamb[i, 2] += iz
                X[i - 1, 0] += wa * ix
                X[i - 1, 1] += wa * iy
                X[i - 1, 2] += wa * iz
                X[i, 0] -= 2 * wb * ix
                X[i, 1] -= 2 * wb * iy
                X[i, 2] -= 2 * wb * iz
                X[i + 1, 0] += wc * ix
                X[i + 1, 1] += wc * iy
                X[i + 1, 2] += wc * iz
            # minimum bend radius
            dx = X[i + 1, 0] - X[i - 1, 0]
            dy = X[i + 1, 1] - X[i - 1, 1]
            dz = X[i + 1, 2] - X[i - 1, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < dmin - tol and wa + wc > 0.0 and d > 1e-9:
                s = (dmin - d) / (d * (wa + wc))
                X[i - 1, 0] -= wa * s * dx
                X[i - 1, 1] -= wa * s * dy
                X[i - 1, 2] -= wa * s * dz
                X[i + 1, 0] += wc * s * dx
                X[i + 1, 1] += wc * s * dy
                X[i + 1, 2] += wc * s * dz

        if cap[0] > 0.5 and prm[12] > 0.5:
            # yarn held in the throat can only slide through it, along itself
            k0 = int(cap[1])
            for q in range(k0, min(n, k0 + 2)):
                qa = max(q - 1, 0)
                qb = min(q + 1, n - 1)
                tx = X[qb, 0] - X[qa, 0]
                ty = X[qb, 1] - X[qa, 1]
                tz = X[qb, 2] - X[qa, 2]
                tl_ = math.sqrt(tx * tx + ty * ty + tz * tz) + 1e-30
                capt[q] = 1
                ccnt[q] = 1
                cn[q, 0, 0] = tx / tl_
                cn[q, 0, 1] = ty / tl_
                cn[q, 0, 2] = tz / tl_
        # ------------------------------- stretch: direct solve of the whole chain
        for _rep in range(2):
            f1 = _stretch_direct(X, W, n, h, feed, ta, tb, tc, td, tl, tol, U, Pm, pressed, cn, cg, cmu, ccnt,
                                 capt, Ps, fmu)
            fed += f1
        if it >= 2:
            dm = 0.0
            for i in range(n):
                dd = abs(X[i, 0] - Xi[i, 0]) + abs(X[i, 1] - Xi[i, 1]) + abs(X[i, 2] - Xi[i, 2])
                if dd > dm:
                    dm = dd
            if dm < conv:
                break

    # ------------------------------------------------------------- wrap up
    maxd = 0.0
    maxs = 0.0
    for i in range(n):
        dx = X[i, 0] - Xp[i, 0]
        dy = X[i, 1] - Xp[i, 1]
        dz = X[i, 2] - Xp[i, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > maxd:
            maxd = d
        if i < n - 1:
            ex = X[i + 1, 0] - X[i, 0]
            ey = X[i + 1, 1] - X[i, 1]
            ez = X[i + 1, 2] - X[i, 2]
            e = abs(math.sqrt(ex * ex + ey * ey + ez * ez) / h - 1.0)
            if e > maxs:
                maxs = e
    # material frame: carry each segment's director with the minimal rotation
    for i in range(n - 1):
        ox = Xp[i + 1, 0] - Xp[i, 0]
        oy = Xp[i + 1, 1] - Xp[i, 1]
        oz = Xp[i + 1, 2] - Xp[i, 2]
        nx = X[i + 1, 0] - X[i, 0]
        ny = X[i + 1, 1] - X[i, 1]
        nz = X[i + 1, 2] - X[i, 2]
        if ox == nx and oy == ny and oz == nz:
            continue
        ol = math.sqrt(ox * ox + oy * oy + oz * oz) + 1e-30
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-30
        ox /= ol
        oy /= ol
        oz /= ol
        nx /= nl
        ny /= nl
        nz /= nl
        # rotate D by the rotation taking o to n
        cx = oy * nz - oz * ny
        cy = oz * nx - ox * nz
        cz = ox * ny - oy * nx
        c = ox * nx + oy * ny + oz * nz
        dx = Dv[i, 0]
        dy = Dv[i, 1]
        dz = Dv[i, 2]
        if c > -0.999999:
            k = 1.0 / (1.0 + c)
            dc = cx * dx + cy * dy + cz * dz
            rx = dx * c + (cy * dz - cz * dy) + cx * dc * k
            ry = dy * c + (cz * dx - cx * dz) + cy * dc * k
            rz = dz * c + (cx * dy - cy * dx) + cz * dc * k
        else:
            rx, ry, rz = -dx, -dy, -dz
        dn = rx * nx + ry * ny + rz * nz
        rx -= dn * nx
        ry -= dn * ny
        rz -= dn * nz
        rl = math.sqrt(rx * rx + ry * ry + rz * rz) + 1e-30
        Dv[i, 0] = rx / rl
        Dv[i, 1] = ry / rl
        Dv[i, 2] = rz / rl

    stats[S_MAXDISP] = maxd
    stats[S_STRETCH] = maxs
    stats[S_NPAIRS] = npairs
    stats[S_HOOKPEN] = hookpen
    stats[S_SELFPEN] = selfpen
    stats[S_CAPDIST] = capdist
    stats[S_FED] = fed
    stats[S_NIT] = nit
    return npairs


@njit(cache=True)
def min_self_distance(X, n, nskip):
    """Smallest centreline distance between non-neighbouring segments (O(n^2), for tests)."""
    best = 1e30
    for i in range(n - 1):
        for j in range(i + nskip, n - 1):
            dx = X[i, 0] - X[j, 0]
            if abs(dx) > 6.0:
                continue
            dy = X[i, 1] - X[j, 1]
            if abs(dy) > 6.0:
                continue
            dz = X[i, 2] - X[j, 2]
            if abs(dz) > 6.0:
                continue
            s, t = _closest_seg_seg(X[i], X[i + 1], X[j], X[j + 1])
            ax = X[i, 0] + s * (X[i + 1, 0] - X[i, 0]) - X[j, 0] - t * (X[j + 1, 0] - X[j, 0])
            ay = X[i, 1] + s * (X[i + 1, 1] - X[i, 1]) - X[j, 1] - t * (X[j + 1, 1] - X[j, 1])
            az = X[i, 2] + s * (X[i + 1, 2] - X[i, 2]) - X[j, 2] - t * (X[j + 1, 2] - X[j, 2])
            d = math.sqrt(ax * ax + ay * ay + az * az)
            if d < best:
                best = d
    return best
