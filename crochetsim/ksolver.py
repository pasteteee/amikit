"""Quasi-static yarn solver, global version.

The yarn is a chain of nodes h apart.  It has no velocity, no gravity and no springs.  In every
small step of the tools the new shape of the whole yarn is the one that

    * moves the least from the shape it had           (drag: 1/2 m |x - x_prev|^2)
    * changes its curvature the least                 (yarn does not kink or straighten by itself)
    * is pulled by the one real force in the scene:   the constant tension T0 of the yarn hand
    * and satisfies every hard condition exactly:     segment lengths, steel of the hook,
      yarn against yarn, smallest bend radius, the throat holding the caught strand.

That is one convex-like problem for all nodes at once.  It is solved by Newton steps on the
KKT system (sparse LU) with an active set for the one-sided conditions, so a pull at one end
of a knot is felt through the whole knot inside a single step.  If no tool moves and the hand's
tension is balanced, the solution is "nobody moves".
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

from .hook import hook_sdf, hook_sdf_grad
from .solver import _closest_seg_seg, build_pairs


@njit(cache=True)
def _row3(r, d, gx, gy, gz, rows, cols, vals, nnz):
    if d >= 0:
        rows[nnz] = r
        cols[nnz] = 3 * d
        vals[nnz] = gx
        rows[nnz + 1] = r
        cols[nnz + 1] = 3 * d + 1
        vals[nnz + 1] = gy
        rows[nnz + 2] = r
        cols[nnz + 2] = 3 * d + 2
        vals[nnz + 2] = gz
        return nnz + 3
    return nnz


@njit(cache=True)
def _decide(sl, C, ctol, act, lam, flip):
    """Is this one-sided condition active?  One that was active stays while it pushes; one
    that was not becomes active when violated.  A condition that keeps flipping inside one step
    is held active for the rest of the step (it then pulls back by a hair, which is harmless)."""
    if act[sl] >= 1:
        if lam[sl] > 0.0:
            return True
        if flip[sl] >= 2:
            return True
        flip[sl] += 1
        return False
    return C < -ctol


@njit(cache=True)
def assemble(X, dof, n, N, h, Rn, tn, hp, hp2, cap, par, pairs, npairs, act, lam, first,
             rows, cols, vals, Cv, slot, out, flip):
    """Evaluate all conditions at X, pick the active ones and write their Jacobian rows.
    Slots: [0,n-1) stretch, [n-1,2n-1) hook/node, [2n-1,3n-2) hook/mid, [3n-2,4n-2) bend,
    4n-2 throat, 4n-1+k yarn pair k.   out = [m, nnz, max violation, hookpen, selfpen, capdist]"""
    rch = par[0]
    rcy = par[1]
    dmin = par[2]
    cap_eps = par[3]
    touch = par[4]
    ctol = par[5]
    m = 0
    nnz = 0
    viol = 0.0
    hookpen = 0.0
    selfpen = 0.0
    capdist = 0.0
    # ---------------------------------------------------------------- stretch
    for i in range(n - 1):
        da = dof[i]
        db = dof[i + 1]
        if da < 0 and db < 0:
            continue
        ux = X[i + 1, 0] - X[i, 0]
        uy = X[i + 1, 1] - X[i, 1]
        uz = X[i + 1, 2] - X[i, 2]
        dl = math.sqrt(ux * ux + uy * uy + uz * uz) + 1e-30
        C = dl - h
        ux /= dl
        uy /= dl
        uz /= dl
        if abs(C) > viol:
            viol = abs(C)
        nnz = _row3(m, da, -ux, -uy, -uz, rows, cols, vals, nnz)
        nnz = _row3(m, db, ux, uy, uz, rows, cols, vals, nnz)
        Cv[m] = C
        slot[m] = i
        m += 1
    # ------------------------------------------------------------------- hook
    L = hp[4]
    rad_reach = hp[0] + rch + 1.0
    capon = cap[0] > 0.5
    for pass_ in range(2):
        cnt = n if pass_ == 0 else n - 1
        for i in range(cnt):
            if pass_ == 0:
                if dof[i] < 0:
                    continue
                px = X[i, 0]
                py = X[i, 1]
                pz = X[i, 2]
                sl = N + i
            else:
                if dof[i] < 0 and dof[i + 1] < 0:
                    continue
                px = 0.5 * (X[i, 0] + X[i + 1, 0])
                py = 0.5 * (X[i, 1] + X[i + 1, 1])
                pz = 0.5 * (X[i, 2] + X[i + 1, 2])
                sl = 2 * N + i
            rx = px - tn[0]
            ry = py - tn[1]
            rz = pz - tn[2]
            l0 = Rn[0, 0] * rx + Rn[1, 0] * ry + Rn[2, 0] * rz
            ok = True
            if l0 < -rch - 1.0 or l0 > L:
                ok = False
            else:
                l1 = Rn[0, 1] * rx + Rn[1, 1] * ry + Rn[2, 1] * rz
                l2 = Rn[0, 2] * rx + Rn[1, 2] * ry + Rn[2, 2] * rz
                if l1 * l1 + l2 * l2 > rad_reach * rad_reach:
                    ok = False
            if not ok:
                act[sl] = 0
                lam[sl] = 0.0
                continue
            # a throat that holds yarn is full: other yarn slides over it as over the shaft
            hq = hp
            if capon and abs(i - cap[1]) > 7.0:
                hq = hp2
            d = hook_sdf(l0, l1, l2, hq)
            if pass_ == 0:
                C = d - rch
            else:
                # the middle of a segment may cut the corner of a round surface a little; it
                # only matters where no node touches (thin edges)
                C = d - (rch - 0.08)
                if act[N + i] == 2 and act[N + i + 1] == 2:
                    act[sl] = 0
                    lam[sl] = 0.0
                    continue
            if C < -viol:
                viol = -C
            if -C > hookpen:
                hookpen = -C
            on = _decide(sl, C, ctol, act, lam, flip)
            if not on:
                act[sl] = 0
                lam[sl] = 0.0
                continue
            d, g0, g1, g2 = hook_sdf_grad(l0, l1, l2, hq)
            nx = Rn[0, 0] * g0 + Rn[0, 1] * g1 + Rn[0, 2] * g2
            ny = Rn[1, 0] * g0 + Rn[1, 1] * g1 + Rn[1, 2] * g2
            nz = Rn[2, 0] * g0 + Rn[2, 1] * g1 + Rn[2, 2] * g2
            if pass_ == 0:
                act[sl] = 2      # (2 = active in this very assembly)
                nnz = _row3(m, dof[i], nx, ny, nz, rows, cols, vals, nnz)
            else:
                nnz = _row3(m, dof[i], 0.5 * nx, 0.5 * ny, 0.5 * nz, rows, cols, vals, nnz)
                nnz = _row3(m, dof[i + 1], 0.5 * nx, 0.5 * ny, 0.5 * nz, rows, cols, vals, nnz)
            Cv[m] = C
            slot[m] = sl
            m += 1
    # ------------------------------------------------------ smallest bend radius
    for i in range(1, n - 1):
        sl = 3 * N + i
        da = dof[i - 1]
        db = dof[i + 1]
        if da < 0 or db < 0 or dof[i] < 0:
            continue          # where fingers hold the yarn it may leave them at any angle
        ux = X[i + 1, 0] - X[i - 1, 0]
        uy = X[i + 1, 1] - X[i - 1, 1]
        uz = X[i + 1, 2] - X[i - 1, 2]
        dl = math.sqrt(ux * ux + uy * uy + uz * uz) + 1e-30
        C = dl - dmin
        if C < -viol:
            viol = -C
        on = _decide(sl, C, ctol, act, lam, flip)
        if not on:
            act[sl] = 0
            lam[sl] = 0.0
            continue
        ux /= dl
        uy /= dl
        uz /= dl
        nnz = _row3(m, da, -ux, -uy, -uz, rows, cols, vals, nnz)
        nnz = _row3(m, db, ux, uy, uz, rows, cols, vals, nnz)
        Cv[m] = C
        slot[m] = sl
        m += 1
    # ----------------------------------------------------------------- throat
    sl = 4 * N
    if capon:
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
        C = cap_eps - dist
        if C < -viol:
            viol = -C
        on = _decide(sl, C, ctol, act, lam, flip)
        if on and dist > 1e-9 and (dof[bk] >= 0 or dof[bk + 1] >= 0):
            px = X[bk, 0] + bs * (X[bk + 1, 0] - X[bk, 0])
            py = X[bk, 1] + bs * (X[bk + 1, 1] - X[bk, 1])
            pz = X[bk, 2] + bs * (X[bk + 1, 2] - X[bk, 2])
            gx = -(px - qx) / dist
            gy = -(py - qy) / dist
            gz = -(pz - qz) / dist
            nnz = _row3(m, dof[bk], (1 - bs) * gx, (1 - bs) * gy, (1 - bs) * gz, rows, cols, vals, nnz)
            nnz = _row3(m, dof[bk + 1], bs * gx, bs * gy, bs * gz, rows, cols, vals, nnz)
            Cv[m] = C
            slot[m] = sl
            m += 1
        else:
            act[sl] = 0
            lam[sl] = 0.0
    else:
        act[sl] = 0
        lam[sl] = 0.0
    # ------------------------------------------------------------ yarn on yarn
    for k in range(npairs):
        sl = 4 * N + 1 + k
        i = pairs[k, 0]
        j = pairs[k, 1]
        s, t = _closest_seg_seg(X[i], X[i + 1], X[j], X[j + 1])
        dx = X[i, 0] + s * (X[i + 1, 0] - X[i, 0]) - X[j, 0] - t * (X[j + 1, 0] - X[j, 0])
        dy = X[i, 1] + s * (X[i + 1, 1] - X[i, 1]) - X[j, 1] - t * (X[j + 1, 1] - X[j, 1])
        dz = X[i, 2] + s * (X[i + 1, 2] - X[i, 2]) - X[j, 2] - t * (X[j + 1, 2] - X[j, 2])
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        C = d - 2.0 * rcy
        # a touch at the far end of a segment belongs to the next segment (count it once)
        if (s >= 1.0 and i + 1 < n - 1) or (t >= 1.0 and j + 1 < n - 1):
            act[sl] = 0
            lam[sl] = 0.0
            continue
        if C < -viol:
            viol = -C
        if -C > selfpen:
            selfpen = -C
        on = _decide(sl, C, ctol, act, lam, flip) or (first and C < touch)
        if not on or d < 1e-9:
            act[sl] = 0
            lam[sl] = 0.0
            continue
        nx = dx / d
        ny = dy / d
        nz = dz / d
        nnz = _row3(m, dof[i], (1 - s) * nx, (1 - s) * ny, (1 - s) * nz, rows, cols, vals, nnz)
        nnz = _row3(m, dof[i + 1], s * nx, s * ny, s * nz, rows, cols, vals, nnz)
        nnz = _row3(m, dof[j], -(1 - t) * nx, -(1 - t) * ny, -(1 - t) * nz, rows, cols, vals, nnz)
        nnz = _row3(m, dof[j + 1], -t * nx, -t * ny, -t * nz, rows, cols, vals, nnz)
        Cv[m] = C
        slot[m] = sl
        m += 1
    out[0] = m
    out[1] = nnz
    out[2] = viol
    out[3] = hookpen
    out[4] = selfpen
    out[5] = capdist


@njit(cache=True)
def contact_curvature(P, lamc, Rn, tn, hq_flag, hp, hp2, out):
    """For yarn pressed on the hook with multipliers lamc at world points P: the stiffness
    -lam * Hessian(sdf) (world frame) that the curved steel gives the contact.  out[k] 3x3."""
    e = 1e-2
    g = np.zeros((6, 3))
    Hl = np.zeros((3, 3))
    for k in range(P.shape[0]):
        rx = P[k, 0] - tn[0]
        ry = P[k, 1] - tn[1]
        rz = P[k, 2] - tn[2]
        l = np.zeros(3)
        l[0] = Rn[0, 0] * rx + Rn[1, 0] * ry + Rn[2, 0] * rz
        l[1] = Rn[0, 1] * rx + Rn[1, 1] * ry + Rn[2, 1] * rz
        l[2] = Rn[0, 2] * rx + Rn[1, 2] * ry + Rn[2, 2] * rz
        hq = hp2 if hq_flag[k] == 1 else hp
        for a in range(3):
            for sgn in range(2):
                q0 = l[0]
                q1 = l[1]
                q2 = l[2]
                dlt = e if sgn == 0 else -e
                if a == 0:
                    q0 += dlt
                elif a == 1:
                    q1 += dlt
                else:
                    q2 += dlt
                d, g0, g1, g2 = hook_sdf_grad(q0, q1, q2, hq)
                g[2 * a + sgn, 0] = g0
                g[2 * a + sgn, 1] = g1
                g[2 * a + sgn, 2] = g2
        for a in range(3):
            for b in range(3):
                Hl[b, a] = (g[2 * a, b] - g[2 * a + 1, b]) / (2 * e)
        for a in range(3):
            for b in range(3):
                acc = 0.0
                for c in range(3):
                    for d_ in range(3):
                        acc += Rn[a, c] * 0.5 * (Hl[c, d_] + Hl[d_, c]) * Rn[b, d_]
                out[k, a, b] = -lamc[k] * acc


@njit(cache=True)
def _closest_many(X, pi, pj, out):
    for k in range(pi.shape[0]):
        s, t = _closest_seg_seg(X[pi[k]], X[pi[k] + 1], X[pj[k]], X[pj[k] + 1])
        out[k, 0] = s
        out[k, 1] = t


@njit(cache=True)
def transport_directors(Xp, X, Dv, n):
    """Material frame: carry each segment's director with the minimal rotation."""
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


class KSolver:
    def __init__(self, nmax, h, r):
        self.h = h
        self.r = r
        rho_min = 1.25 * r
        self.par = np.array([0.93 * r, 0.93 * r, 2 * h * math.cos(h / rho_min / 2), 0.15, 1e-6, 2e-6])
        self.nskip = max(4, int(math.ceil(2.6 * r / h)) + 1)
        self.beta = 0.5            # resistance to curvature change (viscous, relative to drag)
        self.m_last = 60.0         # the node in the yarn hand is heavy: the hand draws slowly
        self.eps = 1e-7
        self.maxit = 24
        self.tol_d = 5e-5
        self.tol_c = 1e-5
        self.maxstep = 0.3 * r
        self.pairs = np.zeros((nmax * 40, 2), np.int32)
        self.head = np.zeros(1 << 15, np.int32)
        self.nxt = np.zeros(nmax, np.int32)
        M = 4 * nmax + 1 + nmax * 40
        self.act = np.zeros(M, np.uint8)
        self.flip = np.zeros(M, np.uint8)
        self.lam = np.zeros(M)
        self.rows = np.zeros(M * 12, np.int32)
        self.cols = np.zeros(M * 12, np.int32)
        self.vals = np.zeros(M * 12)
        self.Cv = np.zeros(M)
        self.slot = np.zeros(M, np.int32)
        self.out = np.zeros(6)
        self._Hkey = None
        self.N = nmax
        self.wneg = -50.0
        self.neg_pairs = False
        self._n_prev = -1
        self.tau = np.zeros(nmax)      # tension in each segment (kept from step to step)
        self.debug = False
        self._H = None
        self.stats = {}

    def _hessian(self, n, dof, nf, W):
        key = (n, nf, hash(dof[:n].tobytes()))
        if key == self._Hkey:
            return self._H
        m = np.ones(n)
        m[n - 1] = self.m_last
        ii = np.arange(1, n - 1)
        B = sp.coo_matrix((np.concatenate([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)]),
                           (np.concatenate([ii - 1, ii - 1, ii - 1]), np.concatenate([ii - 1, ii, ii + 1]))),
                          shape=(n - 2, n)).tocsr()
        Hn = (sp.diags(m) + self.beta * (B.T @ B)).tocsr()
        free = np.where(dof[:n] >= 0)[0]
        Hf = Hn[free][:, free]
        H = sp.kron(Hf, sp.identity(3), format="csc")
        self._Hkey = key
        self._H = H
        return H

    def _geometric(self, X, n, dof, nf, force_node, force, Rn, tn, hp, hp2, cap):
        """Geometric stiffness: a strand under tension resists being moved sideways.
        (Second-order term of the length conditions; without it Newton zigzags on taut yarn.)"""
        h = self.h
        tau = self.tau[:n - 1]
        idx = np.where(tau > 1e-12)[0]
        ii = []
        jj = []
        vv = []
        if len(idx):
            u = X[idx + 1] - X[idx]
            l = np.linalg.norm(u, axis=1) + 1e-30
            u /= l[:, None]
            P = (tau[idx] / l)[:, None, None] * (np.eye(3)[None] - u[:, :, None] * u[:, None, :])
            da = dof[idx]
            db = dof[idx + 1]
            for (p, q, sg) in ((da, da, 1.0), (db, db, 1.0), (da, db, -1.0), (db, da, -1.0)):
                ok = (p >= 0) & (q >= 0)
                if not ok.any():
                    continue
                r3 = (3 * p[ok])[:, None, None] + np.arange(3)[None, :, None] + np.zeros((1, 1, 3), np.int64)
                c3 = (3 * q[ok])[:, None, None] + np.arange(3)[None, None, :] + np.zeros((1, 3, 1), np.int64)
                ii.append(r3.reshape(-1))
                jj.append(c3.reshape(-1))
                vv.append(sg * P[ok].reshape(-1))
        if force_node >= 0 and dof[force_node] >= 0:
            T = np.linalg.norm(force)
            if T > 0:
                u = force / T
                # the hand is far: treat its pull as a string of length >= 5 mm
                P = T / 5.0 * (np.eye(3) - np.outer(u, u))
                d0 = 3 * dof[force_node]
                r3 = d0 + np.arange(3)[:, None] + np.zeros((1, 3), np.int64)
                c3 = d0 + np.arange(3)[None, :] + np.zeros((3, 1), np.int64)
                ii.append(r3.reshape(-1))
                jj.append(c3.reshape(-1))
                vv.append(P.reshape(-1))
        # yarn pressed on curved steel (hollow of the throat): the contact itself is stiff sideways
        N = self.N
        act = self.act
        lam = self.lam
        for kind in (0, 1):
            cnt = n if kind == 0 else n - 1
            a0 = N if kind == 0 else 2 * N
            idx = np.where((act[a0:a0 + cnt] >= 1) & (lam[a0:a0 + cnt] > 0))[0]
            if not len(idx):
                continue
            P = X[idx] if kind == 0 else 0.5 * (X[idx] + X[idx + 1])
            flag = ((cap[0] > 0.5) & (np.abs(idx - cap[1]) > 7.0)).astype(np.int64)
            Kc = np.zeros((len(idx), 3, 3))
            contact_curvature(P, lam[a0 + idx], Rn, tn, flag, hp, hp2, Kc)
            w, V = np.linalg.eigh(Kc)
            w = np.clip(w, self.wneg, 50.0)
            Kc = (V * w[:, None, :]) @ np.swapaxes(V, 1, 2)
            if kind == 0:
                combos = ((dof[idx], dof[idx], 1.0),)
            else:
                da = dof[idx]
                db = dof[idx + 1]
                combos = ((da, da, 0.25), (db, db, 0.25), (da, db, 0.25), (db, da, 0.25))
            for (p, q, sg) in combos:
                ok = (p >= 0) & (q >= 0)
                if not ok.any():
                    continue
                r3 = (3 * p[ok])[:, None, None] + np.arange(3)[None, :, None] + np.zeros((1, 1, 3), np.int64)
                c3 = (3 * q[ok])[:, None, None] + np.arange(3)[None, None, :] + np.zeros((1, 3, 1), np.int64)
                ii.append(r3.reshape(-1))
                jj.append(c3.reshape(-1))
                vv.append(sg * Kc[ok].reshape(-1))
        if self.neg_pairs:
            # two round yarns pressed together / a bend at its tightest: these contacts make
            # sideways motion easier (negative curvature)
            def add_rel(nodes, wts, lamv, dvec):
                dist = np.linalg.norm(dvec, axis=1) + 1e-30
                nn = dvec / dist[:, None]
                P3 = (-lamv / dist)[:, None, None] * (np.eye(3)[None] - nn[:, :, None] * nn[:, None, :])
                for a in range(len(nodes)):
                    for b in range(len(nodes)):
                        p = dof[nodes[a]]
                        q = dof[nodes[b]]
                        ok = (p >= 0) & (q >= 0)
                        if not ok.any():
                            continue
                        r3 = (3 * p[ok])[:, None, None] + np.arange(3)[None, :, None] + np.zeros((1, 1, 3), np.int64)
                        c3 = (3 * q[ok])[:, None, None] + np.arange(3)[None, None, :] + np.zeros((1, 3, 1), np.int64)
                        ii.append(r3.reshape(-1))
                        jj.append(c3.reshape(-1))
                        vv.append(((wts[a] * wts[b])[ok][:, None, None] * P3[ok]).reshape(-1))
            npairs = self._npairs
            kk = np.where((act[4 * N + 1:4 * N + 1 + npairs] >= 1) & (lam[4 * N + 1:4 * N + 1 + npairs] > 0))[0]
            if len(kk):
                pi = self.pairs[kk, 0].astype(np.int64)
                pj = self.pairs[kk, 1].astype(np.int64)
                st_ = np.zeros((len(kk), 2))
                _closest_many(X, pi, pj, st_)
                s_ = st_[:, 0]
                t_ = st_[:, 1]
                dvec = X[pi] + s_[:, None] * (X[pi + 1] - X[pi]) - X[pj] - t_[:, None] * (X[pj + 1] - X[pj])
                add_rel([pi, pi + 1, pj, pj + 1], [1 - s_, s_, -(1 - t_), -t_], lam[4 * N + 1 + kk], dvec)
            bi = np.where((act[3 * N + 1:3 * N + n - 1] >= 1) & (lam[3 * N + 1:3 * N + n - 1] > 0))[0] + 1
            if len(bi):
                one = np.ones(len(bi))
                add_rel([bi - 1, bi + 1], [-one, one], lam[3 * N + bi], X[bi + 1] - X[bi - 1])
        # the throat holding the strand
        if cap[0] > 0.5 and act[4 * N] >= 1 and lam[4 * N] > 0:
            q = Rn @ np.array([hp[11], hp[12], 0.0]) + tn
            k = int(cap[1])
            sgm = cap[1] - k
            if k + 1 < n:
                pt = X[k] + sgm * (X[k + 1] - X[k])
                dv = pt - q
                dist = np.linalg.norm(dv)
                if dist > 1e-6:
                    nn = dv / dist
                    P3 = lam[4 * N] / dist * (np.eye(3) - np.outer(nn, nn))
                    for (a, b, wgt) in ((k, k, (1 - sgm) ** 2), (k + 1, k + 1, sgm ** 2), (k, k + 1, sgm * (1 - sgm)),
                                        (k + 1, k, sgm * (1 - sgm))):
                        if dof[a] >= 0 and dof[b] >= 0 and wgt > 0:
                            r3 = 3 * dof[a] + np.arange(3)[:, None] + np.zeros((1, 3), np.int64)
                            c3 = 3 * dof[b] + np.arange(3)[None, :] + np.zeros((3, 1), np.int64)
                            ii.append(r3.reshape(-1))
                            jj.append(c3.reshape(-1))
                            vv.append(wgt * P3.reshape(-1))
        if not ii:
            return sp.csc_matrix((3 * nf, 3 * nf))
        return sp.csc_matrix((np.concatenate(vv), (np.concatenate(ii), np.concatenate(jj))), shape=(3 * nf, 3 * nf))

    def forget(self):
        """Drop the remembered active set (after a step was undone)."""
        self._n_prev = -1

    def step(self, X, Xp, W, n, Rn, tn, hp, hp2, cap, force_node, force):
        """Advance X[:n] (in place) for the hook pose (Rn, tn).  Xp is the shape before the
        step (pinned nodes may already have been moved in X by their grippers)."""
        h = self.h
        dof = -np.ones(n, np.int64)
        free = np.where(W[:n] > 0)[0]
        nf = len(free)
        dof[free] = np.arange(nf)
        H = self._hessian(n, dof, nf, W)
        reach = 2.0 * self.par[1] + h + 0.5
        npairs = build_pairs(X, W, n, h, reach, self.nskip, self.pairs, self.head, self.nxt, 0, n)
        self._npairs = npairs
        N = self.N
        act = self.act
        lam = self.lam
        if n != self._n_prev:
            act[:4 * N + 1] = 0
            lam[:4 * N + 1] = 0.0
            self._n_prev = n
        act[4 * N + 1:4 * N + 1 + npairs] = 0
        self.flip[:4 * N + 1 + npairs] = 0
        lam[4 * N + 1:4 * N + 1 + npairs] = 0.0
        f = np.zeros(3 * nf)
        if force_node >= 0 and dof[force_node] >= 0:
            f[3 * dof[force_node]:3 * dof[force_node] + 3] = force
        Xf_prev = Xp[free].reshape(-1)
        out = self.out
        nit = 0
        relax = 1.0
        dn_prev = 1e30
        lmax = 0.0
        dmax = 0.0
        conv = False
        for it in range(self.maxit):
            assemble(X, dof, n, self.N, h, Rn, tn, hp, hp2, cap, self.par, self.pairs, npairs, act, lam, it == 0,
                     self.rows, self.cols, self.vals, self.Cv, self.slot, out, self.flip)
            m = int(out[0])
            nnz = int(out[1])
            if it > 0 and dmax < self.tol_d and out[2] < max(self.tol_c, 3.0 * self.eps * lmax):
                conv = True
                break
            nit += 1
            J = sp.csc_matrix((self.vals[:nnz], (self.rows[:nnz], self.cols[:nnz])), shape=(m, 3 * nf))
            g = f - H @ (X[free].reshape(-1) - Xf_prev)
            Hg = H + self._geometric(X, n, dof, nf, force_node, force, Rn, tn, hp, hp2, cap)
            K = sp.bmat([[Hg, -J.T], [-J, -self.eps * sp.identity(m)]], format="csc")
            sol = spla.splu(K).solve(np.concatenate([g, self.Cv[:m]]))
            d = sol[:3 * nf]
            lm = sol[3 * nf:]
            lmax = np.abs(lm).max() if m else 0.0
            dn = np.abs(d).max() if nf else 0.0
            sc = 1.0
            if dn > self.maxstep:
                sc = self.maxstep / dn
            # Newton hopping to and fro over a sharp edge of the hook: take shorter steps
            if it >= 2 and dn > (1.0 - 0.4 * relax) * dn_prev:
                relax = max(0.5 * relax, 0.125)
            elif dn < (1.0 - 0.75 * relax) * dn_prev:
                relax = min(1.0, 2.0 * relax)
            dn_prev = dn
            sc *= relax
            X[free] += sc * d.reshape(-1, 3)
            dmax = dn if sc == 1.0 else max(dn, 1.0)
            sl = self.slot[:m]
            if self.debug:
                print('   it %d m %d viol %.2e dn %.2e sc %.2f neg %d minlam %.2e maxlam %.2e hpen %.3f spen %.3f' % (
                    it, m, out[2], dn, sc, int((lm[sl >= self.N] < 0).sum()), lm.min() if m else 0, lm.max() if m else 0, out[3], out[4]))
            lam[sl] = lm
            act[sl] = 1
            # equalities (stretch) keep any sign: mark them with a positive multiplier
            eq = sl < self.N
            self.tau[:n] = 0.0
            self.tau[sl[eq]] = np.maximum(-lm[eq], 0.0)
            lam[sl[eq]] = 1.0
        assemble(X, dof, n, self.N, h, Rn, tn, hp, hp2, cap, self.par, self.pairs, npairs, act, lam, False,
                 self.rows, self.cols, self.vals, self.Cv, self.slot, out, self.flip)
        m = int(out[0])
        sl = self.slot[:m]
        Cm = self.Cv[:m]
        fl = np.zeros(n + 1, np.int32)
        N = self.N
        a = sl[(sl >= N) & (sl < 2 * N)] - N
        fl[a] |= 1
        a = sl[(sl >= 2 * N) & (sl < 3 * N)] - 2 * N
        fl[a] |= 1
        fl[a + 1] |= 1
        kk = sl[(sl >= 4 * N + 1) & (Cm < 1e-3)] - (4 * N + 1)
        for c in (0, 1):
            fl[self.pairs[kk, c]] |= 2
            fl[self.pairs[kk, c] + 1] |= 2
        self.flags = fl
        disp = np.abs(X[:n] - Xp[:n]).max()
        seg = np.linalg.norm(X[1:n] - X[:n - 1], axis=1)
        self.stats = dict(nit=nit, conv=conv, viol=float(out[2]), hookpen=float(out[3]), selfpen=float(out[4]),
                          capdist=float(out[5]), disp=float(disp), stretch=float(np.abs(seg / h - 1).max()),
                          npairs=npairs, m=int(out[0]))
        return self.stats
