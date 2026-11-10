"""The crocheter: plans hook moves from the live yarn geometry and executes them.

Every stitch is built from one primitive, ``catch``: push the hook along its own axis so the
working strand rides over the head and comes to lie on the shaft behind the throat, then pull
back: the strand drops into the throat and is drawn out as a loop."""
import math
import numpy as np

from .hook import frame


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-30)


class CatchFailed(RuntimeError):
    pass


class Crocheter:
    def __init__(self, sim, speed=26.0):
        self.s = sim
        self.speed = speed
        self.x_over = 7.4       # how far behind the tip the strand must lie before pulling back
        self.x_start = -2.2

    # ----------------------------------------------------------- sensing
    def strand_anchor(self):
        """Index of the node where the free working strand leaves the work."""
        s = self.s
        c = s.n - 1
        while c > 0 and s.flags[c] == 0 and s.W[c] > 0:
            c -= 1
        return c

    def work_anchor(self):
        """Index of the node where the working strand leaves the work (last yarn-yarn contact)."""
        s = self.s
        c = s.n - 1
        while c > 0 and (s.flags[c] & 2) == 0 and s.W[c] > 0:
            c -= 1
        return c

    def strand_point(self, g, c=None):
        """Node on the free strand about g mm from where it leaves the work."""
        s = self.s
        c = self.strand_anchor() if c is None else c
        k = min(s.n - 2, c + max(1, int(round(g / s.h))))
        return k, s.X[k].copy()

    def strand_dir(self, k):
        s = self.s
        return unit(s.F - s.X[k])

    # ---------------------------------------------------------- planning
    def plan_catch(self, G, sdir, free_dir, through=None, a=None):
        """Hook pose line for catching the strand that passes G with direction sdir.
        Either ``through`` (a point the axis must pass, e.g. the hole in the fabric) or the axis
        direction ``a`` is given.  The throat opens to the side ``free_dir`` (where the strand
        has room to be lifted)."""
        if a is None:
            a = unit(G - through)
        for _ in range(4):
            sp = sdir - (sdir @ a) * a
            if np.linalg.norm(sp) < 0.35:
                raise CatchFailed("strand nearly parallel to the hook (%.2f)" % np.linalg.norm(sp))
            sp = unit(sp)
            n = np.cross(a, sp)
            if n @ free_dir < 0:
                n = -n
            Gp = G - self.s.hp[12] * n
            if through is None:
                break
            a = unit(Gp - through)
        R = frame(a, n)
        return dict(a=a, n=n, R=R, Gp=Gp, start=Gp + a * self.x_start, over=Gp + a * self.x_over)

    # --------------------------------------------------------- execution
    def push_and_catch(self, plan, lo, retract_to, speed=None, stop_at_capture=False):
        """Push along the axis to plan['over'], pull back towards ``retract_to`` (tip position on
        the same axis); the throat takes the strand on the way."""
        s = self.s
        speed = speed or self.speed
        s.move_to(plan["over"], plan["R"], speed=speed)

        def grab(sim):
            if sim.cap[0] < 0.5:
                sim.try_capture(lo=max(0, self.strand_anchor() - 2) if lo is None else lo, tol=0.4)
            return sim.cap[0] > 0.5 and stop_at_capture
        s.move_to(retract_to, plan["R"], speed=speed, until=grab)
        if s.cap[0] < 0.5:
            raise CatchFailed("the throat did not take the strand")
