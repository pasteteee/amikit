"""Build an n-stitch chain animation from one clean stitch (qs_chain.py run from the slip knot, 1 stitch, skip 0).

After one stitch, the newest link sits exactly where the slip-knot ring sat at the start, and the yarn on the hook is
in the same kind of state: so stitch k is stitch 1 with the finished chain below the newest link instead of the bare
tail. Between stitches the free yarn (from the newest link to the finger) eases from where stitch 1 leaves it to
where stitch 1 starts, while the hook rests: the other hand settles the new link (a few mm at a few tenths of a mm
per frame).
Usage: python compose_chain.py <one_stitch.json> <out.json> <n_stitches> [blend_frames]
"""
import json, sys
import numpy as np
from qs_chain import self_collide, hook_push, Pose


def arclen(P):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def by_fraction(P, m):
    s = arclen(P); t = np.linspace(0, s[-1], m)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def smooth(u):
    return u * u * (3 - 2 * u)


SKIP = __import__("os").environ.get("BSKIP", "1") == "1"


def main():
    src, out, n_st = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nb_fr = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    d = json.load(open(src)); F = d['frames']
    A = np.array(F[0]['pts']); E = np.array(F[-1]['pts'])
    na = F[0]['nfrozen']; ne = F[-1]['nfrozen']
    conn = E[na:ne]                                # the yarn from the old ring up to and round the new link
    drop = E[0] - A[0]                             # the work moves down by one link per stitch
    prefix = A[:na].copy()                         # everything below the ring the hook works into
    frames = []; t = 0.0; dt = 1.0 / d['fps']

    def emit(pts, f, nfro, st):
        nonlocal t
        t += dt
        g = dict(f); g['pts'] = np.round(pts, 3).tolist(); g['s'] = np.round(arclen(pts), 3).tolist()
        sh = len(pts) - len(f['pts'])
        for key in ('att', 'link'):
            if g.get(key) is not None: g[key] = [g[key][0] + sh, g[key][1] + sh]
        g['nfrozen'] = nfro; g['t'] = round(t, 3); g['stitch'] = st
        frames.append(g)

    for st in range(n_st):
        for f in F:
            P = np.array(f['pts']); off = P[0] - A[0]
            emit(np.vstack([prefix + off, P[na:]]), f, len(prefix) + f['nfrozen'] - na, st)
        prefix = np.vstack([prefix + drop, conn])
        if st == n_st - 1: break
        # ease the free yarn from where this stitch leaves it to where the next one starts
        f0, f1 = F[-1], F[0]
        X0 = np.array(f0['pts'])[ne:]; X1 = np.array(f1['pts'])[na:]
        # match the pieces: work leg, the loop on the hook, the leg to the finger
        a0 = [f0['att'][0] - ne, f0['att'][1] - ne]; a1 = [f1['att'][0] - na, f1['att'][1] - na]
        cuts0 = [0, a0[0], a0[1], len(X0) - 1]; cuts1 = [0, a1[0], a1[1], len(X1) - 1]
        Y0 = []; Y1 = []
        for q in range(3):
            P0 = X0[cuts0[q]:cuts0[q + 1] + 1]; P1 = X1[cuts1[q]:cuts1[q + 1] + 1]
            mq = max(len(P0), len(P1)); A_ = by_fraction(P0, mq); B_ = by_fraction(P1, mq)
            Y0.append(A_ if q == 0 else A_[1:]); Y1.append(B_ if q == 0 else B_[1:])
        Y0 = np.vstack(Y0); Y1 = np.vstack(Y1); m = len(Y0)
        X = Y0.copy(); H = Pose(**{k_: f1['hook'][k_] for k_ in ('tip', 'a', 'd')})
        nfix = min(len(prefix), 160); w = np.concatenate([np.full(nfix, 1e-6), np.ones(m)]); w[nfix] = 1e-6; w[-1] = 1e-6
        for k in range(nb_fr):
            u = smooth((k + 1) / nb_fr)
            T = Y0 + (Y1 - Y0) * u
            for _ in range(12):   # move toward the eased pose in small steps, never through other yarn or the hook
                D = T - X; L = np.linalg.norm(D, axis=1, keepdims=True)
                X = X + D * np.minimum(1.0, 0.12 / np.maximum(L, 1e-9))
                Q = self_collide(np.vstack([prefix[-nfix:], X]), w, iters=3, skip=((nfix, nfix + m), (nfix, nfix + m)) if SKIP else None)
                X = hook_push(H, Q[nfix:])
            g = dict(f1 if u > 0.5 else f0); g['att'] = None; g['link'] = None
            emit(np.vstack([prefix, X]), g, len(prefix), st)
        r_ = np.linalg.norm(X - Y1, axis=1); print("blend residual", round(float(r_.max()), 2), "at", np.where(r_ > 1.0)[0][[0, -1]] if (r_ > 1).any() else "", "of", m)
    d['frames'] = frames
    json.dump(d, open(out, 'w'))
    print('frames', len(frames), 'points', len(frames[-1]['pts']))


if __name__ == '__main__':
    main()
