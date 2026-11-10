"""Scene state + the driver that moves the tools in small substeps and records frames."""
import math
import numpy as np

from . import solver as S
from .hook import default_hook, quat_from_mat, mat_from_quat, slerp, rot_angle


class Sim:
    def __init__(self, X0, h, r, feeder=None, hook_R=None, hook_t=None, fps=30, nmax=None, hook=None):
        n = len(X0)
        nmax = nmax or (n + 6000)
        self.h = float(h)
        self.r = float(r)
        self.n = n
        self.X = np.zeros((nmax, 3))
        self.X[:n] = X0
        self.Xp = self.X.copy()
        self.W = np.ones(nmax)
        self.hp = default_hook(yarn_r=r) if hook is None else hook
        self.prm = S.default_params(h, r)
        self.R = np.eye(3) if hook_R is None else np.array(hook_R, float)
        self.t = np.array([0.0, -200.0, 0.0]) if hook_t is None else np.array(hook_t, float)
        self.cap = np.zeros(2)
        self.feed = np.zeros(8)
        if feeder is not None:
            self.feed[:3] = feeder
            self.feed[3] = h
            self.feed[4] = 150.0
            self.feed[5] = 1.0
        self.grip = -np.ones((4, 4))
        self.grip_pos = None           # for rendering the gripper
        self.lamh = np.zeros(2 * nmax)
        self.lamht = np.zeros((2 * nmax, 3))
        self.pairs = np.zeros((nmax * 40, 2), np.int32)
        self.plam = np.zeros(nmax * 40)
        self.plamt = np.zeros((nmax * 40, 3))
        self.lamb = np.zeros((nmax, 3))
        self.flags = np.zeros(nmax, np.int32)
        self.D = np.zeros((nmax, 3))
        self._init_directors(0, n - 1)
        self.head = np.zeros(1 << 15, np.int32)
        self.nxt = np.zeros(nmax, np.int32)
        self.stats = np.zeros(8)
        self.tri = [np.zeros(nmax + 2) for _ in range(5)]
        self.U = np.zeros((nmax + 2, 3))
        self.Pm = np.zeros((nmax, 6))
        self.Ps = np.zeros((nmax, 6))
        self.pressed = np.zeros(nmax, np.int32)
        self.cn = np.zeros((nmax, 2, 3))
        self.cg = np.zeros((nmax, 2))
        self.cmu = np.zeros((nmax, 2))
        self.ccnt = np.zeros(nmax, np.int32)
        self.capt = np.zeros(nmax, np.int32)
        self.fmu = np.zeros(nmax)
        self.fps = fps
        self.time = 0.0
        self.frames = []               # dicts
        self._next_frame_t = 0.0
        self.dmax = 0.04               # tool travel per substep, mm
        self.caption = ""
        self.marks = {}
        self.log = []
        self.worst = dict(stretch=0.0, hookpen=0.0, selfpen=0.0)
        self.takeup = True
        self.wF = 1.0
        self.take = 0.06
        self.T0 = 1500.0
        self.strand_active = True
        self.nprobe_fail = 0
        self.pen_prev = 0.0
        self.trace_bad = True
        self.T_hold = 6.0
        self.take_cur = self.take
        self.take_min = 0.003
        self.probe_sleep = 0
        self.why = [0, 0, 0, 0]
        self.last_lamF = 0.0
        self.nsub = 0
        self.fed_total = 0.0

    # ------------------------------------------------------------ helpers
    def _init_directors(self, a, b):
        for i in range(a, b):
            tdir = self.X[i + 1] - self.X[i]
            tdir = tdir / (np.linalg.norm(tdir) + 1e-30)
            if i > 0 and np.linalg.norm(self.D[i - 1]) > 0.5:
                d = self.D[i - 1] - tdir * (self.D[i - 1] @ tdir)
            else:
                ref = np.array([0.0, 0.0, 1.0]) if abs(tdir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                d = ref - tdir * (ref @ tdir)
            self.D[i] = d / (np.linalg.norm(d) + 1e-30)

    def pin(self, idx, on=True):
        self.W[idx] = 0.0 if on else 1.0

    def pocket(self):
        return self.R @ np.array([self.hp[11], self.hp[12], 0.0]) + self.t

    def axis(self):
        """Direction the tip points to."""
        return -self.R[:, 0]

    def throat(self):
        return self.R[:, 1]

    def shaft_point(self, sigma):
        """Point on the hook axis ``sigma`` mm behind the tip apex."""
        return self.t + self.R[:, 0] * sigma

    # --------------------------------------------------------------- step
    def _solve(self, Rn, tn):
        S.substep(self.X, self.Xp, self.W, self.n, self.h, self.r, self.R, self.t, Rn, tn, self.hp, self.prm,
                  self.cap, self.feed, self.grip, self.lamh, self.lamht, self.pairs, self.plam, self.plamt,
                  self.lamb, self.flags, self.D, self.head, self.nxt, self.stats, *self.tri, self.U, self.Pm, self.pressed, self.cn, self.cg, self.cmu, self.ccnt, self.capt, self.Ps, self.fmu)

    def _substep(self, Rn, tn):
        """One substep = the tools move (the yarn follows, paying out freely at the feeder),
        then the hand that holds the working yarn tries to draw a little yarn back in."""
        moved_tool = (Rn is not self.R) or (tn is not self.t) or self.grip[0, 0] >= 0
        self.feed[4] = self.T_hold
        self._solve(Rn, tn)
        self.R = Rn
        self.t = tn
        self.nsub += 1
        st = self.stats
        self.fed_total += st[S.S_FED]
        w = self.worst
        if st[S.S_STRETCH] > w["stretch"]:
            w["stretch"] = st[S.S_STRETCH]
            w["stretch_at"] = self.nsub
        if st[S.S_HOOKPEN] > w["hookpen"]:
            w["hookpen"] = st[S.S_HOOKPEN]
        if st[S.S_SELFPEN] > w["selfpen"]:
            w["selfpen"] = st[S.S_SELFPEN]
        self.pen_prev = max(st[S.S_HOOKPEN], st[S.S_SELFPEN])
        if self.trace_bad and (st[S.S_STRETCH] > 0.03 or st[S.S_MAXDISP] > 0.3):
            self.log.append((self.nsub, self.time, float(st[S.S_STRETCH]), float(st[S.S_MAXDISP]), float(st[S.S_HOOKPEN]),
                             float(st[S.S_SELFPEN]), float(self.feed[6]), self.n))
        if self.feed[5] > 0.5:
            self._feeder_bookkeeping()
            if moved_tool or st[S.S_MAXDISP] > 1e-7:
                self.strand_active = True
            if self.takeup and self.strand_active:
                if self.probe_sleep > 0:
                    self.probe_sleep -= 1
                else:
                    self._probe(moved_tool)

    def _probe(self, moved_tool):
        """Tension of the working yarn.  With the tools standing, try to draw ``take_cur`` mm of
        yarn in.  The try is kept only if it is a clean, feasible move that needs no more pull
        than the hand gives (T0); otherwise everything is put back exactly as it was.  When even
        the smallest try fails the strand is taut and the hand rests."""
        n = self.n
        Xsave = self.X[:n].copy()
        Dsave = self.D[:n].copy()
        fsave = self.feed.copy()
        csave = self.cap.copy()
        gsave = self.grip[:, 0].copy()
        self.grip[:, 0] = -1
        self.feed[3] -= self.take_cur
        self.feed[4] = 1e30
        self._solve(self.R, self.t)
        self.grip[:, 0] = gsave
        st = self.stats
        lam = self.feed[6]
        self.last_lamF = lam
        pen = max(st[S.S_HOOKPEN], st[S.S_SELFPEN])
        why = -1
        if lam > self.T0:                              # more pull than the hand gives
            why = 0
        elif self.feed[7] > 0.03 * lam + 1e-6:         # the solve did not settle: already taut
            why = 1
        elif pen > max(self.pen_prev, 0.01) + 0.003:   # it would squeeze yarn into something
            why = 2
        elif st[S.S_STRETCH] > 0.01 or st[S.S_MAXDISP] > 0.3:   # not a clean small move
            why = 3
        if why >= 0:
            self.why[why] += 1
            self.X[:n] = Xsave
            self.D[:n] = Dsave
            self.feed[:] = fsave
            self.cap[:] = csave
            self.nprobe_fail += 1
            at_min = self.take_cur <= self.take_min * 1.001
            if why == 0:
                self.take_cur = max(self.take_min, self.take_cur * min(0.9, max(0.2, 0.7 * self.T0 / lam)))
            else:
                self.take_cur = max(self.take_min, self.take_cur * 0.35)
            if at_min:
                if moved_tool:
                    self.probe_sleep = 3
                else:
                    self.strand_active = False
        else:
            self.take_cur = min(self.take, self.take_cur * 1.6)
            self._feeder_bookkeeping()

    def _feeder_bookkeeping(self):
        h = self.h
        F = self.feed[:3]
        # yarn paid out: new nodes enter at the feeder
        while self.feed[3] > h:
            e = self.n - 1
            d = self.X[e] - F
            dist = np.linalg.norm(d)
            u = d / dist if dist > 1e-9 else np.array([1.0, 0, 0])
            self.X[self.n] = F + u * max(self.feed[3] - h, 0.0)
            self.Xp[self.n] = self.X[self.n]
            self.W[self.n] = 1.0
            self.n += 1
            self.feed[3] -= h
            self._init_directors(self.n - 2, self.n - 1)
        # yarn drawn in: nodes that reach the feeder leave the scene
        while self.feed[3] < 0.15 * h and self.n > 8:
            self.n -= 1
            self.feed[3] += h

    def record(self):
        self.frames.append(dict(X=self.X[:self.n].astype(np.float32).copy(),
                                D=self.D[:self.n].astype(np.float32).copy(),
                                R=self.R.copy(), t=self.t.copy(), time=self.time,
                                cap=self.caption, grip=None if self.grip_pos is None else self.grip_pos.copy(),
                                marks=dict(self.marks)))

    def _tick(self, dt):
        self.time += dt
        while self.time >= self._next_frame_t:
            self.record()
            self._next_frame_t += 1.0 / self.fps

    def move_to(self, t1, R1=None, speed=30.0, until=None, lever=25.0):
        """Move the hook rigidly to tip position t1 / orientation R1.
        ``until(sim)`` may stop the move early (returns True)."""
        t0 = self.t.copy()
        R0 = self.R.copy()
        t1 = np.asarray(t1, float)
        if R1 is None:
            R1 = R0
        ang = rot_angle(R0, R1)
        travel = np.linalg.norm(t1 - t0) + ang * lever
        if travel < 1e-9:
            return True
        ns = max(1, int(math.ceil(travel / self.dmax)))
        q0 = quat_from_mat(R0)
        q1 = quat_from_mat(R1)
        dt = travel / speed / ns
        for k in range(1, ns + 1):
            u = k / ns
            u = u * u * (3 - 2 * u) if ns > 40 else u      # ease in/out on long moves
            Rn = mat_from_quat(slerp(q0, q1, u)) if ang > 1e-9 else R0
            tn = t0 + (t1 - t0) * u
            self._substep(Rn, tn)
            self._tick(dt)
            if until is not None and until(self):
                return False
        return True

    def hold(self, seconds, substeps_per_frame=2):
        """Tools stand still.  (Nothing may move; the solver still runs, to prove it.)"""
        nfr = max(1, int(round(seconds * self.fps)))
        for _ in range(nfr):
            for _ in range(substeps_per_frame if not self.strand_active else 12):
                self._substep(self.R, self.t)
            self._tick(1.0 / self.fps)

    def settle(self, nsteps=10):
        for _ in range(nsteps):
            self._substep(self.R, self.t)

    def pull_grip(self, nodes, delta, speed=20.0, until=None):
        """Carry pinned nodes (a gripper) along ``delta``; the hook stands still."""
        delta = np.asarray(delta, float)
        travel = np.linalg.norm(delta)
        ns = max(1, int(math.ceil(travel / self.dmax)))
        step = delta / ns
        dt = travel / speed / ns
        for k, nd in enumerate(nodes):
            self.grip[k, 0] = nd
            self.grip[k, 1:] = step
        ok = True
        for _ in range(ns):
            self._substep(self.R, self.t)
            if self.grip_pos is not None:
                self.grip_pos = self.grip_pos + step
            self._tick(dt)
            if until is not None and until(self):
                ok = False
                break
        self.grip[:, 0] = -1
        return ok


def _try_capture(self, lo=0, hi=None, tol=1.3):
    """Let the throat take the strand that lies in it. Returns the distance, or None."""
    hi = self.n - 1 if hi is None else hi
    q = self.pocket()
    d = np.linalg.norm(self.X[lo:hi + 1] - q, axis=1)
    k = int(np.argmin(d))
    if d[k] > tol:
        return None
    self.cap[0] = 1.0
    self.cap[1] = float(lo + k)
    return float(d[k])


def _release(self):
    self.cap[0] = 0.0


Sim.try_capture = _try_capture
Sim.release = _release
