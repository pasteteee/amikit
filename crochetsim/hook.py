"""Crochet hook: signed distance field + rigid pose.

Local frame: origin at the tip apex, +X runs tip -> handle, +Y is the side the
throat opens to, +Z = X x Y.  Units are millimetres.

hp = [R_shaft, r_tip, x_tip_c, x_cone_end, L,
      slot_ax, slot_ay, slot_ra, slot_bx, slot_by, slot_rb,
      pocket_x, pocket_y, edge_round, finger_x, finger_R]

finger_x/finger_R describe the finger of the hook hand resting on the shaft in front of the
loops it holds back (a collar; 0 = no finger).  Only the loops being held feel it.
"""
import math
import numpy as np
from numba import njit


def default_hook(R=1.5, yarn_r=1.0):
    r_tip = 0.7
    ra = yarn_r * 1.14
    ax, ay = 3.7, 0.32
    bx, by = ax + 3.0, ay + 2.4
    return np.array([R, r_tip, r_tip, 5.6, 150.0,
                     ax, ay, ra, bx, by, ra * 1.45,
                     ax, ay, 0.5, 0.0, 0.0], dtype=np.float64)


@njit(cache=True)
def _sd_round_cone_x(qx, qy, r1, r2, h):
    # 2D profile distance: axis along +x from (0,0) r1 to (h,0) r2; qy >= 0 is radial
    b = (r1 - r2) / h
    a = math.sqrt(max(1.0 - b * b, 1e-9))
    k = -qy * b + qx * a
    if k < 0.0:
        return math.sqrt(qx * qx + qy * qy) - r1
    if k > a * h:
        return math.sqrt((qx - h) ** 2 + qy * qy) - r2
    return qy * a + qx * b - r1


@njit(cache=True)
def hook_sdf(px, py, pz, hp):
    R, rt, xc, xe, L = hp[0], hp[1], hp[2], hp[3], hp[4]
    rad = math.sqrt(py * py + pz * pz)
    d1 = _sd_round_cone_x(px - xc, rad, rt, R, xe - xc)
    t = min(max(px, xe), L)
    d2 = math.sqrt((px - t) ** 2 + rad * rad) - R
    body = min(d1, d2)
    ax, ay, ra, bx, by, rb = hp[5], hp[6], hp[7], hp[8], hp[9], hp[10]
    dx, dy = bx - ax, by - ay
    hh = math.sqrt(dx * dx + dy * dy)
    ux, uy = dx / hh, dy / hh
    qx = (px - ax) * ux + (py - ay) * uy
    qy = abs(-(px - ax) * uy + (py - ay) * ux)
    slot = _sd_round_cone_x(qx, qy, ra, rb, hh)
    # rounded (polished) edges where the throat meets the body
    k = hp[13]
    a_ = body
    b_ = -slot
    hh_ = max(k - abs(a_ - b_), 0.0) / k
    d = max(a_, b_) + hh_ * hh_ * k * 0.25
    if hp[15] > 0.0:
        cx = abs(px - hp[14]) - 0.2
        cr = rad - (hp[15] - 0.3)
        c = min(max(cx, cr), 0.0) + math.sqrt(max(cx, 0.0) ** 2 + max(cr, 0.0) ** 2) - 0.3
        d = min(d, c)
    return d


@njit(cache=True)
def hook_sdf_grad(px, py, pz, hp):
    e = 2e-3
    d = hook_sdf(px, py, pz, hp)
    gx = hook_sdf(px + e, py, pz, hp) - hook_sdf(px - e, py, pz, hp)
    gy = hook_sdf(px, py + e, pz, hp) - hook_sdf(px, py - e, pz, hp)
    gz = hook_sdf(px, py, pz + e, hp) - hook_sdf(px, py, pz - e, hp)
    nn = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
    return d, gx / nn, gy / nn, gz / nn


# ---------------------------------------------------------------- pose utils
def frame(tip_dir, throat_dir):
    """Rotation matrix (columns ex, ey, ez) from the direction the tip points to
    and the direction the throat should open to (made orthogonal to the axis)."""
    a = np.asarray(tip_dir, float)
    a = a / np.linalg.norm(a)
    ex = -a
    n = np.asarray(throat_dir, float)
    n = n - ex * (n @ ex)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        n = np.cross(ex, [0.0, 0.0, 1.0])
        nn = np.linalg.norm(n)
    ey = n / nn
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez], axis=1)


def quat_from_mat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def mat_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, u):
    d = float(q0 @ q1)
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)


def rot_angle(R0, R1):
    c = (np.trace(R0.T @ R1) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))
