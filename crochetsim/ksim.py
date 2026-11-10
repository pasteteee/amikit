"""Scene driver on top of the global solver: tools move in small steps, frames are recorded."""
import math
import numpy as np

from .hook import default_hook, quat_from_mat, mat_from_quat, slerp, rot_angle
from .ksolver import KSolver, transport_directors


class Jam(RuntimeError):
    pass


class KSim:
    def __init__(self, X0, h, r, feeder=None, hook_R=None, hook_t=None, fps=30, nmax=None):
        n = len(X0)
        nmax = nmax or (n + 6000)
        self.h = float(h)
        self.r = float(r)
        self.n = n
        self.X = np.zeros((nmax, 3))
        self.X[:n] = X0
        self.W = np.ones(nmax)
        self.D = np.zeros((nmax, 3))
        self._init_directors(0, n - 1)
        self.hp = default_hook(yarn_r=r)
        self.hp_solid = self.hp.copy()
        self.hp_solid[6] += 1000.0
        self.hp_solid[9] += 1000.0
        self.R = np.eye(3) if hook_R is None else np.array(hook_R, float)
        self.t = np.array([0.0, -200.0, 0.0]) if hook_t is None else np.array(hook_t, float)
        self.cap = np.zeros(2)
        self.F = None if feeder is None else np.array(feeder, float)
        self.T0 = 6.0                  # tension of the yarn hand (the only force in the scene)
        self.grip = {}                 # node -> step per substep (set by pull_grip)
        self.grip_pos = None
        self.sol = KSolver(nmax, self.h, self.r)
        self.fps = fps
        self.time = 0.0
        self.frames = []
        self._next_frame_t = 0.0
        self.dmax = 0.05               # tool travel per substep, mm
        self.caption = ""
        self.marks = {}
        self.log = []
        self.worst = dict(stretch=0.0, hookpen=0.0, selfpen=0.0, viol=0.0, disp=0.0)
        self.nsub = 0
        self.nnewton = 0
        self.asleep = False
        self.rest_eps = 2e-4
        self.trace = False
        self.rest_fail = 0
        self.verbose = 0
        self.max_split = 4
        self.tau_max = 400.0
        self.nsplit = 0
        self.flags = np.zeros(nmax, np.int32)

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
        self.asleep = False

    def pocket(self):
        return self.R @ np.array([self.hp[11], self.hp[12], 0.0]) + self.t

    def axis(self):
        return -self.R[:, 0]

    def throat(self):
        return self.R[:, 1]

    @property
    def strand_active(self):
        return not self.asleep

    # --------------------------------------------------------------- step
    def _substep(self, Rn, tn, Fn=None, depth=0):
        """One small step of the tools; the yarn follows.  A step the solver cannot finish
        cleanly is undone and taken again in two halves."""
        moved = (Rn is not self.R) or (tn is not self.t) or bool(self.grip) or Fn is not None
        if moved:
            self.asleep = False
            self.rest_fail = 0
        elif self.asleep:
            return
        n = self.n
        Xp = self.X[:n].copy()
        F_old = None if self.F is None else self.F.copy()
        cap_old = self.cap.copy()
        if Fn is not None:
            self.F = np.asarray(Fn, float)
        for nd, stp in self.grip.items():
            self.X[nd] += stp
        fn = -1
        force = np.zeros(3)
        if self.F is not None and self.W[n - 1] > 0:
            u = self.F - self.X[n - 1]
            du = np.linalg.norm(u)
            if du > 1e-9:
                fn = n - 1
                force = self.T0 * u / du
        st = self.sol.step(self.X, Xp, self.W, n, Rn, tn, self.hp, self.hp_solid, self.cap, fn, force)
        self.nnewton += st["nit"]
        self.flags = self.sol.flags
        if not moved:
            if st["disp"] < self.rest_eps:
                # tools stand and nothing but solver noise is left: the yarn stays exactly where it is
                self.X[:n] = Xp
                self.asleep = True
                return
            if not st["conv"] and st["viol"] > 1e-4:
                # with the tools standing only a clean, finished solution may move the yarn
                self.X[:n] = Xp
                self.cap[:] = cap_old
                self.sol.forget()
                self.rest_fail += 1
                if self.rest_fail >= 3:
                    self.asleep = True
                return
            self.rest_fail = 0
        if (not st["conv"]) and st["viol"] > 2e-3 and depth < self.max_split and moved:
            self.X[:n] = Xp
            self.F = F_old
            self.cap[:] = cap_old
            self.sol.forget()
            self.nsplit += 1
            Rm = mat_from_quat(slerp(quat_from_mat(self.R), quat_from_mat(Rn), 0.5)) if Rn is not self.R else Rn
            tm = 0.5 * (self.t + tn) if tn is not self.t else tn
            Fm = None if Fn is None else 0.5 * (F_old + Fn)
            g0 = self.grip
            self.grip = {k: 0.5 * v for k, v in g0.items()}
            self._substep(Rm, tm, Fm, depth + 1)
            self._substep(Rn, tn, Fn, depth + 1)
            self.grip = g0
            return
        transport_directors(Xp, self.X, self.D, n)
        self.R = Rn
        self.t = tn
        self.nsub += 1
        w = self.worst
        for k in ("stretch", "hookpen", "selfpen", "viol", "disp"):
            if st[k] > w[k]:
                w[k] = st[k]
        if st["disp"] > 0.3 or st["viol"] > 2e-3:
            self.log.append((self.nsub, self.time, st["nit"], st["conv"], st["viol"], st["disp"], st["hookpen"],
                             st["selfpen"], st["stretch"]))
        if self.trace:
            print("sub %d nit %d conv %s viol %.2e disp %.4f hpen %.4f spen %.4f m %d n %d" % (
                self.nsub, st["nit"], st["conv"], st["viol"], st["disp"], st["hookpen"], st["selfpen"], st["m"], n))
        if self.verbose and self.nsub % self.verbose == 0:
            print("  .. nsub %d t %.2f n %d newton %.1f/sub split %d bad %d viol %.1e tau %.1f" % (
                self.nsub, self.time, self.n, self.nnewton / self.nsub, self.nsplit, len(self.log), st["viol"],
                self.sol.tau[:self.n].max()), flush=True)
        if self.F is not None:
            self._feeder_bookkeeping()
        if self.sol.tau[:self.n].max() > getattr(self, "tau_max", 400.0):
            raise Jam("yarn is jammed: tension %.0f at t=%.2f" % (self.sol.tau[:self.n].max(), self.time))

    def _feeder_bookkeeping(self):
        """Yarn enters and leaves the scene at the yarn hand."""
        h = self.h
        F = self.F
        while True:
            e = self.n - 1
            d = F - self.X[e]
            dist = np.linalg.norm(d)
            if dist > 1.5 * h:
                self.X[self.n] = self.X[e] + d / dist * h
                self.W[self.n] = 1.0
                self.n += 1
                self._init_directors(self.n - 2, self.n - 1)
            elif dist < 0.4 * h and self.n > 8 and self.W[e - 1] > 0:
                self.n -= 1
            else:
                break

    def record(self):
        self.frames.append(dict(X=self.X[:self.n].astype(np.float32).copy(),
                                D=self.D[:self.n].astype(np.float32).copy(),
                                R=self.R.copy(), t=self.t.copy(), time=self.time,
                                F=None if self.F is None else self.F.copy(),
                                cap=self.caption, grip=None if self.grip_pos is None else self.grip_pos.copy(),
                                marks=dict(self.marks)))

    def _tick(self, dt):
        self.time += dt
        while self.time >= self._next_frame_t:
            self.record()
            self._next_frame_t += 1.0 / self.fps

    def move_to(self, t1=None, R1=None, speed=30.0, until=None, lever=25.0, feeder=None):
        """Move the hook rigidly to tip position t1 / orientation R1 (and, with ``feeder``, the
        hand that holds the working yarn to a new place at the same time).
        ``until(sim)`` may stop the move early (returns True)."""
        t0 = self.t.copy()
        R0 = self.R.copy()
        t1 = t0 if t1 is None else np.asarray(t1, float)
        if R1 is None:
            R1 = R0
        ang = rot_angle(R0, R1)
        travel = np.linalg.norm(t1 - t0) + ang * lever
        F0 = None if self.F is None else self.F.copy()
        if feeder is not None:
            feeder = np.asarray(feeder, float)
            travel = max(travel, np.linalg.norm(feeder - F0))
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
            self._substep(Rn, tn, None if feeder is None else F0 + (feeder - F0) * u)
            self._tick(dt)
            if until is not None and until(self):
                return False
        return True

    def hold(self, seconds, substeps_per_frame=8):
        """Tools stand still.  The yarn hand may still be drawing slack in; after that nothing
        moves and the solver is not even run."""
        nfr = max(1, int(round(seconds * self.fps)))
        for _ in range(nfr):
            for _ in range(substeps_per_frame):
                self._substep(self.R, self.t)
            self._tick(1.0 / self.fps)

    def wait_taut(self, max_seconds=4.0, substeps_per_frame=8):
        """The tools stand while the yarn hand takes up the slack of the working strand."""
        nfr = max(1, int(round(max_seconds * self.fps)))
        for _ in range(nfr):
            if self.asleep:
                return True
            for _ in range(substeps_per_frame):
                self._substep(self.R, self.t)
            self._tick(1.0 / self.fps)
        return self.asleep

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
        self.grip = {int(nd): step for nd in nodes}
        ok = True
        for _ in range(ns):
            self._substep(self.R, self.t)
            if self.grip_pos is not None:
                self.grip_pos = self.grip_pos + step
            self._tick(dt)
            if until is not None and until(self):
                ok = False
                break
        self.grip = {}
        return ok

    def try_capture(self, lo=0, hi=None, tol=1.3):
        """Let the throat take the strand that lies in it. Returns the distance, or None."""
        hi = self.n - 1 if hi is None else hi
        q = self.pocket()
        d = np.linalg.norm(self.X[lo:hi + 1] - q, axis=1)
        k = int(np.argmin(d))
        if d[k] > tol:
            return None
        self.cap[0] = 1.0
        self.cap[1] = float(lo + k)
        self.asleep = False
        return float(d[k])

    def release(self):
        self.cap[0] = 0.0
        self.asleep = False
