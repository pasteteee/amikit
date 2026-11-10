"""Initial yarn layouts."""
import math
import numpy as np


def resample(pts, h):
    """Polyline -> nodes exactly h apart (chord length), starting at pts[0]."""
    pts = np.asarray(pts, float)
    out = [pts[0]]
    i = 0
    cur = pts[0]
    while True:
        # walk along the polyline until the chord from cur reaches h
        found = False
        while i < len(pts) - 1:
            a, b = pts[i], pts[i + 1]
            # solve |a + t (b-a) - cur| = h for t in (t0, 1]
            d = b - a
            f = a - cur
            A = d @ d
            B = 2 * f @ d
            C = f @ f - h * h
            disc = B * B - 4 * A * C
            if disc >= 0 and A > 1e-18:
                t = (-B + math.sqrt(disc)) / (2 * A)
                if 0.0 <= t <= 1.0 and (a + t * d - cur) @ d >= -1e-12:
                    cur = a + t * d
                    out.append(cur)
                    found = True
                    break
            i += 1
        if not found:
            break
    return np.array(out)


def bezier(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t ** 2 * p2


def magic_ring(h, r, rho=7.5, feeder=None, tail_len=26.0):
    """A loop of yarn in the XZ plane (front faces -Y): the tail hangs down from the crossing at
    the right, the yarn runs once round counter-clockwise (seen from the front) and leaves the
    crossing behind the tail towards the feeder.
    Returns X, marks (node index ranges)."""
    feeder = np.asarray(feeder, float)
    tail = np.array([[rho + 0.2, -r * 1.05, -tail_len], [rho, -r * 1.05, -2.0]])
    th = np.linspace(-0.25, 2 * math.pi, 400)
    ring = np.stack([rho * np.cos(th), -r * 1.05 + 2.1 * r * np.clip(th / (2 * math.pi), 0, 1), rho * np.sin(th)], 1)
    xb = ring[-1]
    up = xb + np.array([0.0, 0.6, 6.0])
    d = feeder - up
    d /= np.linalg.norm(d)
    out1 = up + d * 8.0
    leave = bezier(xb, up, out1)
    pts = np.concatenate([tail, ring, leave[1:], [feeder]])
    X = resample(pts, h)
    # node index ranges
    ang = np.arctan2(X[:, 2], X[:, 0])
    rad = np.hypot(X[:, 0], X[:, 2])
    on_ring = np.abs(rad - rho) < 0.3
    i_ring0 = int(np.argmax(on_ring & (X[:, 2] > -2.0)))            # first ring node (crossing, front)
    n_ring = int(round(2 * math.pi * rho / h))
    i_ring1 = i_ring0 + n_ring                                     # crossing again (back)
    marks = dict(ring0=i_ring0, ring1=i_ring1, n_ring=n_ring)
    return X, marks
