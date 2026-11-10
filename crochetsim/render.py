"""Software renderer (numba): the yarn as a swept sphere with a plied-twist surface that is
attached to the material (so sliding yarn is visible), the hook sphere-traced from the same
signed distance field the physics collides with, a shadow map, ambient occlusion."""
import math
import numpy as np
from numba import njit, prange

from .hook import hook_sdf


def look_at(pos, target, up=(0, 0, 1)):
    pos = np.asarray(pos, float)
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])        # rows


class Camera:
    def __init__(self, pos, target, up=(0, 0, 1), fov=28.0):
        self.pos = np.asarray(pos, float)
        self.target = np.asarray(target, float)
        self.up = np.asarray(up, float)
        self.fov = fov

    def basis(self):
        return look_at(self.pos, self.target, self.up)


@njit(cache=True)
def _smooth_samples(X, D, n, sub, r, h, P, T, DD, SS, COL, colnode):
    """Catmull-Rom samples along the chain. Returns count."""
    m = 0
    for i in range(n - 1):
        i0 = i - 1 if i > 0 else i
        i3 = i + 2 if i + 2 < n else i + 1
        for k in range(sub):
            u = k / sub
            u2 = u * u
            u3 = u2 * u
            for c in range(3):
                p0 = X[i0, c]
                p1 = X[i, c]
                p2 = X[i + 1, c]
                p3 = X[i3, c]
                P[m, c] = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                                 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
                T[m, c] = 0.5 * ((-p0 + p2) + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * u
                                 + 3 * (-p0 + 3 * p1 - 3 * p2 + p3) * u2)
            tl = math.sqrt(T[m, 0] ** 2 + T[m, 1] ** 2 + T[m, 2] ** 2) + 1e-30
            T[m, 0] /= tl
            T[m, 1] /= tl
            T[m, 2] /= tl
            # director: blend towards the next segment's, re-orthogonalise
            j = i + 1 if i + 1 < n - 1 else i
            dx = D[i, 0] * (1 - u) + D[j, 0] * u
            dy = D[i, 1] * (1 - u) + D[j, 1] * u
            dz = D[i, 2] * (1 - u) + D[j, 2] * u
            dn = dx * T[m, 0] + dy * T[m, 1] + dz * T[m, 2]
            dx -= dn * T[m, 0]
            dy -= dn * T[m, 1]
            dz -= dn * T[m, 2]
            dl = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dl < 1e-6:
                dx, dy, dz = D[i, 0], D[i, 1], D[i, 2]
                dl = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-30
            DD[m, 0] = dx / dl
            DD[m, 1] = dy / dl
            DD[m, 2] = dz / dl
            SS[m] = (i + u) * h
            COL[m] = colnode[i]
            m += 1
    return m


@njit(cache=True, parallel=True)
def _splat(P, T, DD, SS, COL, m, r, cam_pos, B, fpx, W, H, zb, nb, sb, pb, idb, cb, nband):
    """Perspective sphere splat with exact ray/sphere depth."""
    cx = 0.5 * W
    cy = 0.5 * H
    for band in prange(nband):
        y0b = band * H // nband
        y1b = (band + 1) * H // nband
        for k in range(m):
            vx = P[k, 0] - cam_pos[0]
            vy = P[k, 1] - cam_pos[1]
            vz = P[k, 2] - cam_pos[2]
            xr = B[0, 0] * vx + B[0, 1] * vy + B[0, 2] * vz
            yu = B[1, 0] * vx + B[1, 1] * vy + B[1, 2] * vz
            zf = B[2, 0] * vx + B[2, 1] * vy + B[2, 2] * vz
            if zf < 1.0:
                continue
            sx = cx + fpx * xr / zf
            sy = cy - fpx * yu / zf
            rp = fpx * r / zf * 1.03 + 1.5
            ya = int(max(y0b, math.floor(sy - rp)))
            yb = int(min(y1b - 1, math.ceil(sy + rp)))
            if ya > yb:
                continue
            xa = int(max(0, math.floor(sx - rp)))
            xb = int(min(W - 1, math.ceil(sx + rp)))
            cc = vx * vx + vy * vy + vz * vz - r * r
            for py in range(ya, yb + 1):
                for px in range(xa, xb + 1):
                    # ray through pixel centre (view space dir = (x, y, 1) normalised)
                    dxv = (px + 0.5 - cx) / fpx
                    dyv = -(py + 0.5 - cy) / fpx
                    il = 1.0 / math.sqrt(dxv * dxv + dyv * dyv + 1.0)
                    rx = (B[0, 0] * dxv + B[1, 0] * dyv + B[2, 0]) * il
                    ry = (B[0, 1] * dxv + B[1, 1] * dyv + B[2, 1]) * il
                    rz = (B[0, 2] * dxv + B[1, 2] * dyv + B[2, 2]) * il
                    b = rx * vx + ry * vy + rz * vz
                    disc = b * b - cc
                    if disc <= 0.0:
                        continue
                    t = b - math.sqrt(disc)
                    if t < zb[py, px]:
                        zb[py, px] = t
                        nx = (rx * t - vx) / r
                        ny = (ry * t - vy) / r
                        nz = (rz * t - vz) / r
                        # remove the along-tangent part: the envelope of the spheres is a tube
                        tn = nx * T[k, 0] + ny * T[k, 1] + nz * T[k, 2]
                        ax = nx - tn * T[k, 0]
                        ay = ny - tn * T[k, 1]
                        az = nz - tn * T[k, 2]
                        al = math.sqrt(ax * ax + ay * ay + az * az)
                        if al > 0.2:
                            ax /= al
                            ay /= al
                            az /= al
                        else:
                            ax, ay, az = nx, ny, nz
                        nb[py, px, 0] = ax
                        nb[py, px, 1] = ay
                        nb[py, px, 2] = az
                        # angle round the yarn in its material frame
                        ex = T[k, 1] * DD[k, 2] - T[k, 2] * DD[k, 1]
                        ey = T[k, 2] * DD[k, 0] - T[k, 0] * DD[k, 2]
                        ez = T[k, 0] * DD[k, 1] - T[k, 1] * DD[k, 0]
                        pb[py, px] = math.atan2(ax * ex + ay * ey + az * ez,
                                                ax * DD[k, 0] + ay * DD[k, 1] + az * DD[k, 2])
                        sb[py, px] = SS[k] + tn * r
                        idb[py, px] = 1
                        cb[py, px] = COL[k]


@njit(cache=True, parallel=True)
def _trace_hook(Rh, th, hp, cam_pos, B, fpx, W, H, zb, nb, idb, x0, x1, y0, y1, vis_len):
    cx = 0.5 * W
    cy = 0.5 * H
    Rb = hp[0] + 0.05
    for py in prange(y0, y1):
        for px in range(x0, x1):
            dxv = (px + 0.5 - cx) / fpx
            dyv = -(py + 0.5 - cy) / fpx
            il = 1.0 / math.sqrt(dxv * dxv + dyv * dyv + 1.0)
            rx = (B[0, 0] * dxv + B[1, 0] * dyv + B[2, 0]) * il
            ry = (B[0, 1] * dxv + B[1, 1] * dyv + B[2, 1]) * il
            rz = (B[0, 2] * dxv + B[1, 2] * dyv + B[2, 2]) * il
            # to hook local
            ox = cam_pos[0] - th[0]
            oy = cam_pos[1] - th[1]
            oz = cam_pos[2] - th[2]
            lo0 = Rh[0, 0] * ox + Rh[1, 0] * oy + Rh[2, 0] * oz
            lo1 = Rh[0, 1] * ox + Rh[1, 1] * oy + Rh[2, 1] * oz
            lo2 = Rh[0, 2] * ox + Rh[1, 2] * oy + Rh[2, 2] * oz
            ld0 = Rh[0, 0] * rx + Rh[1, 0] * ry + Rh[2, 0] * rz
            ld1 = Rh[0, 1] * rx + Rh[1, 1] * ry + Rh[2, 1] * rz
            ld2 = Rh[0, 2] * rx + Rh[1, 2] * ry + Rh[2, 2] * rz
            # infinite cylinder round the X axis bounds the hook
            a = ld1 * ld1 + ld2 * ld2
            bq = lo1 * ld1 + lo2 * ld2
            cq = lo1 * lo1 + lo2 * lo2 - Rb * Rb
            if a < 1e-12:
                continue
            disc = bq * bq - a * cq
            if disc <= 0.0:
                continue
            sq = math.sqrt(disc)
            t0 = (-bq - sq) / a
            t1 = (-bq + sq) / a
            if t1 < 0.0:
                continue
            t = max(t0, 0.0)
            zlim = min(t1, zb[py, px])
            hit = False
            for _ in range(64):
                if t > zlim:
                    break
                qx = lo0 + ld0 * t
                qy = lo1 + ld1 * t
                qz = lo2 + ld2 * t
                if qx > vis_len + 2.0 and ld0 > 0:
                    break
                d = hook_sdf(qx, qy, qz, hp)
                if d < 0.004:
                    hit = True
                    break
                t += d * 0.9
            if hit and t < zb[py, px]:
                qx = lo0 + ld0 * t
                qy = lo1 + ld1 * t
                qz = lo2 + ld2 * t
                e = 0.01
                gx = hook_sdf(qx + e, qy, qz, hp) - hook_sdf(qx - e, qy, qz, hp)
                gy = hook_sdf(qx, qy + e, qz, hp) - hook_sdf(qx, qy - e, qz, hp)
                gz = hook_sdf(qx, qy, qz + e, hp) - hook_sdf(qx, qy, qz - e, hp)
                gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
                gx /= gl
                gy /= gl
                gz /= gl
                zb[py, px] = t
                nb[py, px, 0] = Rh[0, 0] * gx + Rh[0, 1] * gy + Rh[0, 2] * gz
                nb[py, px, 1] = Rh[1, 0] * gx + Rh[1, 1] * gy + Rh[1, 2] * gz
                nb[py, px, 2] = Rh[2, 0] * gx + Rh[2, 1] * gy + Rh[2, 2] * gz
                idb[py, px] = 2


@njit(cache=True)
def _shadow_splat(P, m, r, LB, lo, scale, SM, sm):
    """Orthographic depth map from the light. LB rows: right, up, towards-light."""
    for k in range(m):
        x = (LB[0, 0] * P[k, 0] + LB[0, 1] * P[k, 1] + LB[0, 2] * P[k, 2] - lo[0]) * scale
        y = (LB[1, 0] * P[k, 0] + LB[1, 1] * P[k, 1] + LB[1, 2] * P[k, 2] - lo[1]) * scale
        z = LB[2, 0] * P[k, 0] + LB[2, 1] * P[k, 1] + LB[2, 2] * P[k, 2]
        rp = r * scale
        xa = int(max(0, math.floor(x - rp)))
        xb = int(min(sm - 1, math.ceil(x + rp)))
        ya = int(max(0, math.floor(y - rp)))
        yb = int(min(sm - 1, math.ceil(y + rp)))
        for py in range(ya, yb + 1):
            for px in range(xa, xb + 1):
                ddx = (px + 0.5 - x) / scale
                ddy = (py + 0.5 - y) / scale
                q = r * r - ddx * ddx - ddy * ddy
                if q > 0.0:
                    zz = z + math.sqrt(q)
                    if zz > SM[py, px]:
                        SM[py, px] = zz


@njit(cache=True)
def _hash(a, b):
    v = math.sin(a * 127.1 + b * 311.7) * 43758.5453
    return v - math.floor(v)


@njit(cache=True, parallel=True)
def _shade(zb, nb, sb, pb, idb, cb, cam_pos, B, fpx, W, H, LB, lo, scale, SM, sm, ldir, palette,
           bg_top, bg_bot, plane_p, plane_n, hook_col, r, pitch, nply, out, ao_r_px):
    cx = 0.5 * W
    cy = 0.5 * H
    for py in prange(H):
        for px in range(W):
            dxv = (px + 0.5 - cx) / fpx
            dyv = -(py + 0.5 - cy) / fpx
            il = 1.0 / math.sqrt(dxv * dxv + dyv * dyv + 1.0)
            rx = (B[0, 0] * dxv + B[1, 0] * dyv + B[2, 0]) * il
            ry = (B[0, 1] * dxv + B[1, 1] * dyv + B[2, 1]) * il
            rz = (B[0, 2] * dxv + B[1, 2] * dyv + B[2, 2]) * il
            ident = idb[py, px]
            t = zb[py, px]
            v = py / H
            if ident == 0:
                # backdrop: gradient, with the shadow of the scene on a plane behind the work
                c0 = bg_top[0] * (1 - v) + bg_bot[0] * v
                c1 = bg_top[1] * (1 - v) + bg_bot[1] * v
                c2 = bg_top[2] * (1 - v) + bg_bot[2] * v
                den = rx * plane_n[0] + ry * plane_n[1] + rz * plane_n[2]
                sh = 1.0
                if abs(den) > 1e-6:
                    tp = ((plane_p[0] - cam_pos[0]) * plane_n[0] + (plane_p[1] - cam_pos[1]) * plane_n[1]
                          + (plane_p[2] - cam_pos[2]) * plane_n[2]) / den
                    if tp > 0:
                        wx = cam_pos[0] + rx * tp
                        wy = cam_pos[1] + ry * tp
                        wz = cam_pos[2] + rz * tp
                        lx = (LB[0, 0] * wx + LB[0, 1] * wy + LB[0, 2] * wz - lo[0]) * scale
                        ly = (LB[1, 0] * wx + LB[1, 1] * wy + LB[1, 2] * wz - lo[1]) * scale
                        lz = LB[2, 0] * wx + LB[2, 1] * wy + LB[2, 2] * wz
                        acc = 0.0
                        cnt = 0
                        for oy in range(-6, 7, 2):
                            for ox in range(-6, 7, 2):
                                ix = int(lx) + ox
                                iy = int(ly) + oy
                                if 0 <= ix < sm and 0 <= iy < sm:
                                    if SM[iy, ix] > lz + 0.5:
                                        acc += 1.0
                                cnt += 1
                        sh = 1.0 - 0.33 * acc / cnt
                out[py, px, 0] = c0 * sh
                out[py, px, 1] = c1 * sh
                out[py, px, 2] = c2 * sh
                continue
            wx = cam_pos[0] + rx * t
            wy = cam_pos[1] + ry * t
            wz = cam_pos[2] + rz * t
            nx = nb[py, px, 0]
            ny = nb[py, px, 1]
            nz = nb[py, px, 2]
            # shadow (3x3 PCF)
            lx = (LB[0, 0] * wx + LB[0, 1] * wy + LB[0, 2] * wz - lo[0]) * scale
            ly = (LB[1, 0] * wx + LB[1, 1] * wy + LB[1, 2] * wz - lo[1]) * scale
            lz = LB[2, 0] * wx + LB[2, 1] * wy + LB[2, 2] * wz
            acc = 0.0
            cnt = 0
            for oy in range(-2, 3, 2):
                for ox in range(-2, 3, 2):
                    ix = int(lx) + ox
                    iy = int(ly) + oy
                    if 0 <= ix < sm and 0 <= iy < sm:
                        if SM[iy, ix] > lz + 0.35 * r:
                            acc += 1.0
                    cnt += 1
            lit = 1.0 - acc / cnt
            # ambient occlusion from the depth buffer
            occ = 0.0
            ns = 0
            for k in range(12):
                ang = 2.399963 * k + _hash(px * 0.37, py * 0.61) * 6.283
                rad = ao_r_px * math.sqrt((k + 0.5) / 12.0)
                ix = int(px + math.cos(ang) * rad)
                iy = int(py + math.sin(ang) * rad)
                if 0 <= ix < W and 0 <= iy < H:
                    dz = t - zb[iy, ix]
                    if dz > 0.15 * r and dz < 6.0 * r:
                        occ += min(1.0, dz / (1.2 * r)) * (1.0 - dz / (6.0 * r))
                    ns += 1
            ao = 1.0 - 0.75 * occ / max(ns, 1)
            ndv = -(nx * rx + ny * ry + nz * rz)
            if ndv < 0.0:
                ndv = 0.0
            if ident == 1:
                ci = cb[py, px]
                br = palette[ci, 0]
                bgc = palette[ci, 1]
                bb = palette[ci, 2]
                # plied twist: grooves along a helix that is fixed in the material
                psi = nply * (pb[py, px] - 6.2831853 * sb[py, px] / pitch)
                gro = math.cos(psi)
                # tilt the normal across the groove (helix direction ~ tangent + circumferential)
                tilt = -0.42 * math.sin(psi)
                # circumferential direction = cross(view-independent): approximate with normal x ray
                kx = ny * rz - nz * ry
                ky = nz * rx - nx * rz
                kz = nx * ry - ny * rx
                kl = math.sqrt(kx * kx + ky * ky + kz * kz) + 1e-9
                mx = nx + tilt * kx / kl
                my = ny + tilt * ky / kl
                mz = nz + tilt * kz / kl
                ml = math.sqrt(mx * mx + my * my + mz * mz)
                mx /= ml
                my /= ml
                mz /= ml
                ndl = mx * ldir[0] + my * ldir[1] + mz * ldir[2]
                wrap = max(0.0, (ndl + 0.35) / 1.35)
                fib = _hash(math.floor(sb[py, px] * 9.0 + pb[py, px] * 5.0), math.floor(psi * 2.2)) - 0.5
                alb = 0.86 + 0.14 * gro + 0.07 * fib
                amb = 0.46 + 0.20 * (0.5 + 0.5 * mz)
                fuzz = (1.0 - ndv) ** 3 * 0.22
                li = amb * ao + 0.66 * wrap * lit * (0.55 + 0.45 * ao)
                out[py, px, 0] = min(1.0, br * alb * li + fuzz * (0.6 + 0.4 * br))
                out[py, px, 1] = min(1.0, bgc * alb * li + fuzz * (0.6 + 0.4 * bgc))
                out[py, px, 2] = min(1.0, bb * alb * li + fuzz * (0.6 + 0.4 * bb))
            else:
                # brushed metal
                ndl = max(0.0, nx * ldir[0] + ny * ldir[1] + nz * ldir[2])
                hx = ldir[0] - rx
                hy = ldir[1] - ry
                hz = ldir[2] - rz
                hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
                ndh = max(0.0, (nx * hx + ny * hy + nz * hz) / hl)
                spec = ndh ** 60 * 0.9 * lit
                # fake studio reflection: bright band from above, darker below
                refz = nz * 2.0 * ndv + rz * 0.0 + nz
                env = 0.5 + 0.5 * math.tanh(2.2 * (nz + 0.15))
                band = math.exp(-((nz - 0.55) / 0.22) ** 2) * 0.35
                li = (0.30 + 0.50 * env + band) * (0.55 + 0.45 * ao) * (0.75 + 0.25 * lit) + 0.25 * ndl * lit
                fres = (1.0 - ndv) ** 4 * 0.35
                out[py, px, 0] = min(1.0, hook_col[0] * li + spec + fres)
                out[py, px, 1] = min(1.0, hook_col[1] * li + spec + fres)
                out[py, px, 2] = min(1.0, hook_col[2] * li + spec + fres)


class Renderer:
    def __init__(self, W=540, H=960, ss=2, r=1.0, h=0.8, palette=None, light=(-0.45, -0.55, 0.70),
                 bg_top=(0.955, 0.94, 0.92), bg_bot=(0.90, 0.875, 0.85), hook_col=(0.62, 0.70, 0.74),
                 pitch=7.0, nply=3, backdrop_y=38.0, sm=1024):
        self.W, self.H, self.ss = W, H, ss
        self.r, self.h = r, h
        self.palette = np.array(palette if palette is not None else [[0.93, 0.52, 0.42]], float)
        l = np.asarray(light, float)
        self.ldir = l / np.linalg.norm(l)
        self.bg_top = np.asarray(bg_top, float)
        self.bg_bot = np.asarray(bg_bot, float)
        self.hook_col = np.asarray(hook_col, float)
        self.pitch = pitch
        self.nply = nply
        self.plane_p = np.array([0.0, backdrop_y, 0.0])
        self.plane_n = np.array([0.0, -1.0, 0.0])
        self.sm = sm
        w, hh = W * ss, H * ss
        self.zb = np.zeros((hh, w), np.float32)
        self.nb = np.zeros((hh, w, 3), np.float32)
        self.sb = np.zeros((hh, w), np.float32)
        self.pb = np.zeros((hh, w), np.float32)
        self.idb = np.zeros((hh, w), np.uint8)
        self.cb = np.zeros((hh, w), np.int32)
        self.out = np.zeros((hh, w, 3), np.float32)
        self.SM = np.zeros((sm, sm), np.float32)
        self.sub = max(2, int(math.ceil(h / (0.32 * r))))

    def render(self, X, D, hookR, hookt, hp, cam, colnode=None, extra_spheres=None, hook_visible=True):
        n = len(X)
        X = np.ascontiguousarray(X, np.float64)
        D = np.ascontiguousarray(D, np.float64)
        if colnode is None:
            colnode = np.zeros(n, np.int32)
        sub = self.sub
        m_max = (n - 1) * sub + 1
        P = np.zeros((m_max, 3))
        T = np.zeros((m_max, 3))
        DD = np.zeros((m_max, 3))
        SS = np.zeros(m_max)
        COL = np.zeros(m_max, np.int32)
        m = _smooth_samples(X, D, n, sub, self.r, self.h, P, T, DD, SS, COL, colnode)
        w, hh = self.W * self.ss, self.H * self.ss
        B = cam.basis()
        fpx = 0.5 * w / math.tan(math.radians(cam.fov) / 2)
        self.zb.fill(1e30)
        self.idb.fill(0)
        _splat(P, T, DD, SS, COL, m, self.r, cam.pos, B, fpx, w, hh, self.zb, self.nb, self.sb, self.pb, self.idb,
               self.cb, 16)
        if hook_visible:
            _trace_hook(np.ascontiguousarray(hookR), np.ascontiguousarray(hookt), hp, cam.pos, B, fpx, w, hh,
                        self.zb, self.nb, self.idb, 0, w, 0, hh, 140.0)
        # shadow map
        l = self.ldir
        up = np.array([0.0, 0.0, 1.0]) if abs(l[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        lr = np.cross(up, l)
        lr /= np.linalg.norm(lr)
        lu = np.cross(l, lr)
        LB = np.stack([lr, lu, l])
        ctr = cam.target
        ext = 70.0
        lo = np.array([LB[0] @ ctr - ext, LB[1] @ ctr - ext])
        scale = self.sm / (2 * ext)
        self.SM.fill(-1e30)
        _shadow_splat(P, m, self.r, LB, lo, scale, self.SM, self.sm)
        if hook_visible:
            # the hook casts a shadow as a string of spheres along its axis
            ks = np.arange(0.7, 110.0, 0.5)
            HP = hookt[None, :] + hookR[:, 0][None, :] * ks[:, None]
            rad = np.clip(hp[1] + (hp[0] - hp[1]) * (ks - hp[2]) / (hp[3] - hp[2]), hp[1], hp[0])
            for rr in (hp[1], 0.5 * (hp[0] + hp[1]), hp[0]):
                sel = np.abs(rad - rr) < 0.21 if rr != hp[0] else rad > hp[0] - 0.21
                if sel.any():
                    _shadow_splat(np.ascontiguousarray(HP[sel]), int(sel.sum()), float(rr), LB, lo, scale, self.SM,
                                  self.sm)
        _shade(self.zb, self.nb, self.sb, self.pb, self.idb, self.cb, cam.pos, B, fpx, w, hh, LB, lo, scale, self.SM,
               self.sm, self.ldir, self.palette, self.bg_top, self.bg_bot, self.plane_p, self.plane_n,
               self.hook_col, self.r, self.pitch, float(self.nply), self.out, fpx * 2.2 * self.r / max(
                   1.0, float(np.linalg.norm(cam.target - cam.pos))))
        img = self.out
        if self.ss > 1:
            s = self.ss
            img = img.reshape(self.H, s, self.W, s, 3).mean(axis=(1, 3))
        img = np.clip(img, 0, 1) ** (1 / 1.0)
        return (img * 255 + 0.5).astype(np.uint8)
