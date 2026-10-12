"""Unit / invariant tests of the amikit engine.

Run from amigurumi-lab/:
    ../.venv/Scripts/python -m unittest amikit.test_engine            (all, ~1.5-2 min)
    AMIKIT_SKIP_TOYS=1 ../.venv/Scripts/python -m unittest amikit.test_engine   (skip building every toy)
    ../.venv/Scripts/python -m unittest amikit.test_engine.TestMirror           (one group)

Small specs and `toy.build(ku=2, kv=2)` keep it fast; only one smoke test renders (tiny Renderer).
A test whose failure points at an engine bug says so in its message ("ENGINE BUG: ..."), with the minimal
reproduction in the docstring; the tests are not weakened to pass.
"""
from __future__ import annotations
import copy
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .piece import LEAN

from . import rounds as R
from .rounds import expand, check, compress, counts, RoundError
from .profile import solve
from .piece import PieceDef, _stadium
from .toy import Toy, DEFAULT_YARN

HERE = Path(__file__).parent
TOYS = sorted((HERE / "toys").glob("*.json"))
YARN = dict(DEFAULT_YARN)
MIRROR_X = np.diag([-1.0, 1.0, 1.0])


def _spec(parts, assembly, face=(), colours=None, **extra):
    s = {"name": "t", "title": "test", "yarn": dict(YARN),
         "colours": colours or {"main": "#a08060", "pink": "#f2a0a8", "tan": "#e0c8a0", "red": "#d04040"},
         "parts": parts, "assembly": assembly, "face": list(face)}
    s.update(extra)
    return s


# a small left/right-symmetric toy with every kind of mirrored placement: mirrored pieces on the root, a mirrored
# child of a mirrored parent (applique), flatten + bend toward, seam flat + avoid, a chain oval with oval_axis,
# asymmetric colour zones of all three kinds (band, polygon, ellipse), mirrored face items
SYM = _spec(
    parts=[
        {"id": "body", "rounds": ["mr 6", "inc 6 to 24", "even 3", "dec 6 to 12"], "colour": "main", "end": "open",
         "stuff": "firm"},
        {"id": "head", "rounds": ["mr 6", "inc 6 to 30", "even 3", "dec 6 to 6"], "colour": "main"},
        {"id": "muzzle", "rounds": ["ch 4", "inc 4 x1", "even 1"], "colour": "tan", "end": "open", "stuff": "light"},
        {"id": "ear", "rounds": ["mr 6", "inc 6 to 12", "even 2"], "colour": "main", "end": "open", "flatten": 0.6,
         "bend": {"deg": 40, "at": 2, "over": 2, "toward": [0, 0, 1]},
         "colours": [{"rounds": [2, 4], "az": [20, 120], "colour": "pink"},
                     {"ellipse": {"center": [3, 60], "radii": [1, 40]}, "colour": "tan"}]},
        {"id": "inner_ear", "rounds": ["mr 6", "inc 6 to 12"], "colour": "pink", "end": "open",
         "colours": [{"rounds": [2, 2], "az": [0, 90], "colour": "tan"}]},
        {"id": "arm", "rounds": ["mr 6", "12", "even 3", "9"], "colour": "main", "end": "open", "stuff": "light",
         "colours": [{"shape": [[1, 0], [4, 90], [1, 150]], "colour": "red"}]},
        {"id": "leg", "rounds": ["ch 4", "inc 4 x1", "even 2", "dec 4 to 8"], "colour": "main", "oval_axis": "front",
         "stuff": "medium", "bend": {"deg": 50, "at": 2, "over": 1.5}},
    ],
    assembly=[
        {"part": "body", "axis": [0, 1, 0]},
        {"part": "head", "on": "body", "at": {"round": "end"}, "attach": "end", "dir": [0, 1, 0], "sink": 3},
        {"part": "muzzle", "on": "head", "at": {"round": 6, "az": 0}, "dir": [0, 0, 1]},
        {"part": "ear", "on": "head", "at": {"round": 3, "az": 60}, "dir": [0.6, 1, 0.2], "mirror": True},
        {"part": "inner_ear", "on": "ear", "at": {"round": 2.5, "az": 0}, "applique": True, "mirror": True,
         "roll": 15},
        {"part": "arm", "on": "body", "at": {"round": 6, "az": 80}, "dir": [0.3, -1, 0.2], "mirror": True,
         "avoid": True, "seam": "flat"},
        {"part": "leg", "on": "body", "at": {"round": 3, "az": 30}, "dir": [0.2, -0.2, 1], "mirror": True,
         "sink": 1.5, "bend_toward": [0.6, -1, 0.2]},
    ],
    face=[
        {"type": "safety_eye", "on": "head", "at": {"round": 5.5, "az": 30}, "size_mm": 6, "mirror": True},
        {"type": "blush", "on": "head", "at": {"round": 7, "az": 45}, "size_mm": 6, "colour": "pink", "mirror": True},
        {"type": "stitch", "on": "head", "points": [[7, 10], [7.5, 25]], "mirror": True},
    ],
)

_CACHE = {}


def _toy(spec, key=None):
    """Build a Toy once per key (Toy() is the expensive step: one profile solve per part)."""
    if key is None:
        return Toy(copy.deepcopy(spec))
    if key not in _CACHE:
        _CACHE[key] = Toy(copy.deepcopy(spec))
    return _CACHE[key]


def _grid(inst, nt=12, nth=48):
    M = inst.p.M
    ts = np.linspace(0.05, M - 0.05, nt)
    ths = 2 * np.pi * np.arange(nth) / nth          # symmetric under th -> -th
    T, TH = np.meshgrid(ts, ths, indexing="ij")
    return T, TH


def _mirror_dist(a, b):
    """Largest distance from a point of `a` mirrored in X to its nearest point of `b` (and vice versa)."""
    am = a @ MIRROR_X
    d1, _ = cKDTree(b).query(am)
    d2, _ = cKDTree(am).query(b)
    return float(max(d1.max(), d2.max()))


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, float))))


# ======================================================================================================== rounds
class TestRounds(unittest.TestCase):
    def ok(self, steps, want=None):
        rs = expand(steps)
        self.assertEqual(check(rs), [], f"arithmetic of {steps}")
        if want is not None:
            self.assertEqual([r.n for r in rs], want, f"{steps}")
        return rs

    def test_magic_ring(self):
        rs = self.ok(["mr 6"], [6])
        self.assertEqual(rs[0].start, "mr")
        self.ok(["magic ring 5"], [5])

    def test_chain_oval(self):
        rs = self.ok(["ch 6"], [12])
        self.assertEqual((rs[0].start, rs[0].chain), ("ch", 6))
        self.ok(["ch 4", "inc 4 x2", "even 2"], [8, 12, 16, 16, 16])
        self.ok(["ch 7", "inc 6 to 38", "dec 6 to 20"])

    def test_inc_to_and_x(self):
        self.ok(["mr 6", "inc 6 to 36"], [6, 12, 18, 24, 30, 36])
        self.ok(["mr 6", "inc 6 x2"], [6, 12, 18])
        self.ok(["mr 5", "inc 5 to 30"], [5, 10, 15, 20, 25, 30])

    def test_dec_to_and_x(self):
        self.ok(["mr 6", "inc 6 to 24", "dec 6 to 6"], [6, 12, 18, 24, 18, 12, 6])
        self.ok(["mr 6", "inc 6 to 24", "dec 4 x2"], [6, 12, 18, 24, 20, 16])

    def test_every(self):
        self.ok(["mr 6", "inc 6 to 24 every 2"], [6, 12, 12, 18, 18, 24])     # no even round after the last inc
        self.ok(["mr 6", "inc 6 to 24", "dec 6 to 12 every 3"], [6, 12, 18, 24, 18, 18, 18, 12])

    def test_even_and_abs(self):
        self.ok(["mr 6", "even 3"], [6, 6, 6, 6])
        self.ok(["mr 6", "12", 18, "15", "even 1"], [6, 12, 18, 15, 15])

    def test_inc3_more_than_doubling(self):
        rs = self.ok(["mr 6", "15"], [6, 15])
        self.assertIn("inc3", rs[1].ops)
        self.ok(["mr 6", "18"], [6, 18])

    def test_loops(self):
        rs = self.ok(["mr 6", "inc 6 to 18", "blo 18", "flo", "blo 12"], [6, 12, 18, 18, 18, 12])
        self.assertEqual([r.loop for r in rs], ["both", "both", "both", "blo", "flo", "blo"])

    def test_errors(self):
        bad = [
            [],                                   # no rounds
            ["even 3"],                           # no start
            ["6"],                                # no start
            ["mr 6", "mr 6"],                     # mr not first
            ["mr 6", "ch 5"],                     # ch not first
            ["ch 2"],                             # chain too short
            ["mr 6", "inc 6 to 40"],              # not a multiple
            ["mr 6", "inc 6 to 24", "dec 6 to 30"],  # dec upward
            ["mr 6", "inc 6 to 24", "inc 6 to 12"],  # inc downward
            ["mr 6", "19"],                       # more than tripling
            ["mr 6", "inc 6 to 24", "11"],        # more than halving
            ["mr 6", "inc 6 to 24", "dec 6 to 0"],   # hmm: reaches 6 -> 0
            ["mr 6", "wiggle 3"],                 # unreadable
            ["mr 6", "inc six to 12"],
        ]
        for steps in bad:
            with self.subTest(steps=steps):
                with self.assertRaises(RoundError):
                    expand(steps)

    def test_random_sequences_arithmetic(self):
        """Any sequence of valid counts (+<=100% / -<=50% per round) expands with clean arithmetic."""
        rnd = random.Random(1234)
        for _ in range(400):
            start = rnd.choice(["mr", "ch"])
            if start == "mr":
                cur = rnd.randint(3, 8); steps = [f"mr {cur}"]
            else:
                c = rnd.randint(3, 9); cur = 2 * c; steps = [f"ch {c}"]
            for _ in range(rnd.randint(1, 12)):
                lo, hi = max(3, (cur + 1) // 2), min(2 * cur, 90)
                cur = rnd.randint(lo, hi)
                steps.append(str(cur))
            with self.subTest(steps=steps):
                rs = expand(steps)
                self.assertEqual(check(rs), [])
                self.assertEqual([r.n for r in rs][-1], cur)

    def test_check_detects_errors(self):
        rs = expand(["mr 6", "inc 6 to 18"])
        rs[2].ops = rs[2].ops[:-1]
        self.assertTrue(check(rs))

    def test_compress_roundtrip(self):
        steps_list = [["mr 6", "inc 6 to 48", "even 3", "dec 6 to 6"], ["ch 7", "inc 6 to 38", "dec 4 x3"],
                      ["mr 6", "15", "21", "17", "13", "9"], ["mr 5", "inc 5 to 35 every 2", "dec 5 to 10"]]
        for steps in steps_list:
            for r in expand(steps)[1:]:
                groups = compress(r.ops)
                flat = [o for unit, reps in groups for _ in range(reps) for o in unit]
                self.assertEqual(flat, r.ops, f"{steps}: compress({r.ops}) -> {groups}")

    def test_staggered(self):
        """Consecutive increase rounds do not stack their increases in the same column (hexagon)."""
        rs = expand(["mr 6", "inc 6 to 36"])
        pos = [tuple(i for i, o in enumerate(r.ops) if o == "inc") for r in rs[1:]]
        frac = [tuple(round(p / len(r.ops), 2) for p in ps) for ps, r in zip(pos, rs[1:])]
        self.assertGreater(len(set(f[0] for f in frac)), 1)


# ======================================================================================================= profile
class TestProfile(unittest.TestCase):
    BALL = ["mr 6", "inc 6 to 36", "even 6", "dec 6 to 6"]

    def test_ball(self):
        c = counts(self.BALL)
        r, z, info = solve(c, 4.0, 3.6, closed=True)
        self.assertTrue(info["converged"])
        self.assertEqual(len(r), len(c) + 2)
        self.assertAlmostEqual(r[0], 0.0); self.assertAlmostEqual(r[-1], 0.0)
        hw = (z[-1] - z[0]) / (2 * r.max())
        self.assertGreater(hw, 0.7, f"ball h/w {hw:.2f}")
        self.assertLess(hw, 1.4, f"ball h/w {hw:.2f}")
        self.assertTrue(np.all(np.diff(z) > -1e-6), f"z not monotone: {np.round(z, 2)}")
        # stuffing stretches: the widest boundary is at least its rest radius
        self.assertGreaterEqual(r.max(), 0.98 * info["r_rest"].max())

    def test_open_vs_closed(self):
        c = counts(["mr 6", "inc 6 to 24", "even 4"])
        r_o, z_o, i_o = solve(c, 4.0, 3.6, closed=False, stuffing="none")
        r_c, z_c, i_c = solve(c, 4.0, 3.6, closed=True, stuffing="firm")
        self.assertTrue(i_o["converged"] and i_c["converged"])
        self.assertEqual(len(r_o), len(c) + 1)
        self.assertEqual(len(r_c), len(c) + 2)
        self.assertGreater(r_o[-1], 5.0)              # an open rim stays open
        self.assertAlmostEqual(r_c[-1], 0.0)
        self.assertTrue(np.all(np.diff(z_o) > -1e-6))

    def test_tube_and_oval_and_loops(self):
        for kw in ({"counts": counts(["mr 6", "12", "even 8"]), "closed": False, "stuffing": "light"},
                   {"counts": counts(["ch 6", "inc 4 x2", "even 4", "dec 4 to 12"]), "closed": True,
                    "oval_len": 4 * 4.0},
                   {"counts": counts(["mr 6", "inc 6 to 30", "blo 30", "even 4"]), "closed": False,
                    "loops": ["both"] * 5 + ["blo"] + ["both"] * 4, "stuffing": "firm"}):
            with self.subTest(kw=str(kw)[:60]):
                r, z, info = solve(w=4.0, h=3.6, **kw)
                self.assertTrue(info["converged"])
                self.assertTrue(_finite(r) and _finite(z))
                self.assertTrue(np.all(np.diff(z) > -1e-6), np.round(z, 2))

    def test_stuffing_bulges(self):
        c = counts(["mr 6", "inc 6 to 24", "even 6", "dec 6 to 6"])
        widths = [2 * solve(c, 4.0, 3.6, closed=True, stuffing=s)[0].max() for s in ("none", "light", "firm")]
        self.assertLessEqual(widths[0], widths[1] + 1e-6)
        self.assertLessEqual(widths[1], widths[2] + 1e-6)


# ========================================================================================================= piece
class TestPiece(unittest.TestCase):
    def pdef(self, **d):
        base = {"rounds": ["mr 6", "inc 6 to 18", "even 3", "dec 6 to 6"]}
        base.update(d)
        return PieceDef("p", base, YARN)

    @staticmethod
    def _steps(P):
        """Neighbour distances along t (axis 0) and theta (axis 1, wrapping)."""
        dt = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        da = np.linalg.norm(np.diff(np.concatenate([P, P[:, :1]], axis=1), axis=1), axis=-1)
        return dt, da

    def assert_continuous(self, pd, nt=120, nth=240, **kw):
        ts = np.linspace(0, pd.M, nt)
        ths = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        T, TH = np.meshgrid(ts, ths, indexing="ij")
        P = pd.local_points(T, TH, **kw)
        self.assertTrue(_finite(P), "non-finite local points")
        dt, da = self._steps(P)
        # sampled 120 x 240: steps must stay small compared with the piece (no jumps / tears)
        size = pd.max_radius() + float(pd.zb[-1] - pd.zb[0])
        self.assertLess(dt.max(), 0.12 * size, f"jump along t: {dt.max():.2f} mm")
        self.assertLess(da.max(), 0.12 * size, f"jump around: {da.max():.2f} mm")
        return P

    def test_plain_closed_and_open(self):
        for d in ({}, {"end": "open", "rounds": ["mr 6", "inc 6 to 18", "even 3"]}):
            pd = self.pdef(**d)
            self.assert_continuous(pd)
            self.assertTrue(pd.solve_info["converged"])

    def test_stadium_continuity(self):
        for r, L in ((5.0, 12.0), (0.0, 8.0), (3.0, 0.0), (10.0, 3.0)):
            th = np.linspace(0, 2 * np.pi, 2001)
            x, y = _stadium(np.full_like(th, r), L, th)
            self.assertTrue(_finite(x) and _finite(y))
            step = np.hypot(np.diff(x), np.diff(y))
            per = 2 * L + 2 * np.pi * r
            self.assertLess(step.max(), 3 * per / 2000 + 1e-9, f"stadium r={r} L={L} jumps {step.max():.3f}")
            # closes on itself and lies on the stadium
            self.assertLess(np.hypot(x[0] - x[-1], y[0] - y[-1]), 1e-6)
            dy = np.maximum(np.abs(y) - L / 2, 0)
            self.assertTrue(np.allclose(np.hypot(x, dy), r, atol=1e-6), f"off the stadium r={r} L={L}")

    def test_chain_oval_piece(self):
        for axis in ("side", "front"):
            pd = self.pdef(rounds=["ch 6", "inc 4 x2", "even 3", "dec 4 to 8"], oval_axis=axis)
            P = self.assert_continuous(pd)
            ext = P.reshape(-1, 3).max(0) - P.reshape(-1, 3).min(0)
            if axis == "side":
                self.assertGreater(ext[1], ext[0] * 1.05, "chain along local y (side)")
            else:
                self.assertGreater(ext[0], ext[1] * 1.05, "chain along local x (front)")

    def test_flatten_squash_bend(self):
        cases = [dict(flatten=0.85, end="open", rounds=["mr 6", "inc 6 to 18", "even 4"]),
                 dict(squash=0.8, end="open", rounds=["ch 5", "inc 4 x1", "even 2"]),
                 dict(bend={"deg": 70, "at": 4, "over": 2}),
                 dict(bend={"deg": -50, "at": 3, "over": 1}),
                 dict(bend={"deg": 90, "at": 5, "over": 3}, flatten=0.5, squash=0.4, end="open",
                      rounds=["mr 6", "12", "even 6"])]
        for d in cases:
            with self.subTest(d=d):
                pd = self.pdef(**d)
                self.assert_continuous(pd)
                if pd.bend:
                    self.assert_continuous(pd, bend_phi=0.7, bend_free_start=True)
                    self.assert_continuous(pd, squash=0.85, squash_phi=1.2, squash_rounds=2.5)

    def test_flatten_presses_x(self):
        pd0 = self.pdef(end="open", rounds=["mr 6", "inc 6 to 18", "even 4"])
        pd1 = self.pdef(end="open", rounds=["mr 6", "inc 6 to 18", "even 4"], flatten=0.85)
        th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        t = np.full_like(th, pd0.M - 0.5)
        a, b = pd0.local_points(t, th), pd1.local_points(t, th)
        self.assertLess(np.ptp(b[:, 0]), 0.5 * np.ptp(a[:, 0]))
        self.assertGreater(np.ptp(b[:, 1]), np.ptp(a[:, 1]))

    def test_bend_turns_axis(self):
        pd = self.pdef(rounds=["mr 6", "12", "even 8"], end="open", bend={"deg": 90, "at": 4, "over": 2})
        a0 = pd.anchor(1.0); a1 = pd.anchor(pd.M - 0.5); a2 = pd.anchor(pd.M - 1.5)
        tip = a1 - a2
        self.assertTrue(_finite(a0) and _finite(a1))
        self.assertGreater(abs(tip[0]), abs(tip[2]), f"free end should run along local x after a 90 deg bend: {tip}")


# ============================================================================================================ toy
class TestPlacement(unittest.TestCase):
    def setUp(self):
        self.toy = _toy(SYM, "sym")

    def test_builds(self):
        names = set(self.toy.inst)
        for n in ("body", "head", "muzzle", "ear_L", "ear_R", "inner_ear_L", "inner_ear_R", "arm_L", "arm_R",
                  "leg_L", "leg_R"):
            self.assertIn(n, names)
        for inst in self.toy.inst.values():
            T, TH = _grid(inst)
            self.assertTrue(_finite(inst.base(T, TH)), inst.name)
            self.assertTrue(np.all(np.isfinite(inst.origin)) and np.allclose(inst.R.T @ inst.R, np.eye(3)), inst.name)

    def test_az_convention(self):
        """+az is toward +X on a vertical piece; az 0 is the front (+Z); on a forward piece az 0 is up."""
        body = self.toy.inst["body"]
        c = body.to_world(body.p.anchor(3.5))
        p0, _ = body.point(4, 0); p90, _ = body.point(4, 90); pm90, _ = body.point(4, -90); p180, _ = body.point(4, 180)
        self.assertGreater(p90[0] - c[0], 0.8 * abs(p90[2] - c[2]) + 1, f"az 90 not toward +X: {p90 - c}")
        self.assertLess(pm90[0] - c[0], -1)
        self.assertGreater(p0[2] - c[2], 1)
        self.assertLess(p180[2] - c[2], -1)
        mz = self.toy.inst["muzzle"]
        cm = mz.to_world(mz.p.anchor(1.5))
        q0, _ = mz.point(2, 0); q90, _ = mz.point(2, 90)
        self.assertGreater(q0[1] - cm[1], 1, "muzzle az 0 should be up")
        self.assertGreater(q90[0] - cm[0], 1, "muzzle az 90 should be toward +X")
        # a hanging arm (axis down): +90 still toward the toy's left (+X), az 0 front
        arm = self.toy.inst["arm_L"]
        ca = arm.to_world(arm.p.anchor(2.5))
        a0, _ = arm.point(3, 0); a90, _ = arm.point(3, 90)
        self.assertGreater(a0[2] - ca[2], 0.5, "arm az 0 should face front")
        self.assertGreater(a90[0] - ca[0], 0.5, "arm az 90 should face +X")

    def test_round_between_boundaries(self):
        """`round: r` (the middle of round r) lies between boundaries r-1 and r along the axis."""
        body = self.toy.inst["body"]
        pd = body.p
        for r in range(1, pd.N + 1):
            for az in (0, 90, 180):
                P, _ = body.point(r, az)
                lo, hi = sorted((float(pd.z_t(r - 1)), float(pd.z_t(r))))
                y = (P - body.origin) @ body.R[:, 2]
                with self.subTest(round=r, az=az):
                    self.assertTrue(lo - 1e-6 <= y <= hi + 1e-6, f"round {r}: {y:.2f} not in [{lo:.2f}, {hi:.2f}]")
        P1, _ = body.point(2, 0); P2, _ = body.point(2.5, 0); P3, _ = body.point(3, 0)
        self.assertTrue(P1[1] < P2[1] < P3[1])

    def test_open_rims_seated(self):
        """After seat + conform the opening of every open piece sits on its parent (within ~1 mm)."""
        for name in ("ear_L", "ear_R", "arm_L", "arm_R", "muzzle"):
            inst = self.toy.inst[name]
            e = next(x for x in self.toy.entries if x.get("name") == name)
            parent = self.toy.inst[e["on"]]
            ths = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            rim = inst.base(np.full_like(ths, float(inst.p.N)), ths)
            sd, _, _ = parent.signed_distance(rim)
            with self.subTest(piece=name):
                self.assertLess(np.median(np.abs(sd)), 1.2, f"{name} rim median gap {np.median(sd):.2f} mm")
                self.assertLess(np.abs(sd).max(), 2.0, f"{name} rim worst gap {sd[np.argmax(np.abs(sd))]:.2f} mm")

    def test_applique_on_parent(self):
        for name in ("inner_ear_L", "inner_ear_R"):
            inst = self.toy.inst[name]
            parent = inst.app[0]
            T, TH = _grid(inst, 8, 36)
            P = inst.base(T, TH).reshape(-1, 3)
            sd, _, _ = parent.signed_distance(P)
            with self.subTest(piece=name):
                self.assertLess(np.abs(sd).max(), 2.0, f"{name}: up to {np.abs(sd).max():.2f} mm off its parent")

    def test_build_idempotent(self):
        """Toy.build() twice gives the same meshes and the same flags.

        Regression guard: Toy._face appends to self.dimples on every build; unless build() resets the list, a
        second build presses every eye dimple in twice (the server rebuilds after edits)."""
        toy = Toy(copy.deepcopy(SYM))
        a = {m["name"]: m["V"].copy() for m in toy.build(ku=2, kv=2) if m["name"] == "head"}
        n_flags = len(toy.flags)
        b = {m["name"]: m["V"].copy() for m in toy.build(ku=2, kv=2) if m["name"] == "head"}
        diff = float(np.abs(a["head"] - b["head"]).max())
        self.assertLess(diff, 1e-9, f"ENGINE BUG (Toy.build not idempotent: dimples accumulate): head moved "
                                    f"{diff:.3f} mm on the second build")
        self.assertEqual(len(toy.flags), n_flags, "flags grew on a rebuild")

    def test_closed_sink(self):
        """A closed piece pressed in by `sink` overlaps its parent a little but is not swallowed."""
        head = self.toy.inst["head"]; body = self.toy.inst["body"]
        P, _, _ = head.samples(per_round=2, per_stitch=0.7)
        sd, _, _ = body.signed_distance(P)
        inside = (body.penetration(P) < 0).mean()
        self.assertLess(inside, 0.35, f"{inside:.0%} of the head inside the body")


class TestMirror(unittest.TestCase):
    PAIRS = ("ear", "inner_ear", "arm", "leg")

    def setUp(self):
        self.toy = _toy(SYM, "sym")

    def test_base_surfaces_mirrored(self):
        for p in self.PAIRS:
            L, Rr = self.toy.inst[p + "_L"], self.toy.inst[p + "_R"]
            T, TH = _grid(L, 10, 48)
            a = L.base(T, TH).reshape(-1, 3); b = Rr.base(T, TH).reshape(-1, 3)
            with self.subTest(pair=p):
                self.assertLess(_mirror_dist(a, b), 0.3, f"{p}_R is not the mirror image of {p}_L")

    def test_whole_toy_mirror(self):
        """The built scene of a symmetric spec is symmetric (stitch relief spirals aside: compare at relief=0)."""
        parts = self.toy.build(ku=2, kv=2, relief=0.0)
        for p in self.PAIRS:
            mL = next(m for m in parts if m["name"] == p + "_L"); mR = next(m for m in parts if m["name"] == p + "_R")
            with self.subTest(pair=p):
                self.assertLess(_mirror_dist(mL["V"], mR["V"]), 0.6)
        for kind in ("eye@head", "thread@head"):
            ms = [m for m in parts if m["name"] == kind]
            self.assertEqual(len(ms), 2, kind)
            ca, cb = ms[0]["V"].mean(0), ms[1]["V"].mean(0)
            self.assertLess(np.linalg.norm(ca @ MIRROR_X - cb), 0.5, f"{kind}: {ca} vs {cb}")
        root = next(m for m in parts if m["name"] == "body")
        self.assertLess(_mirror_dist(root["V"], root["V"]), 0.6, "body itself not symmetric")

    def _chart_points(self, inst):
        chart = self.toy.chart(inst)
        pts, cols = [], []
        for i, row in enumerate(chart, 1):
            n = len(row)
            th = 2 * np.pi * (np.arange(n) + 0.5 - LEAN * (i - 0.5)) / n     # where the stitch really sits
            P = inst.base(np.full(n, i - 0.5), th)
            pts.append(P); cols += row
        return np.concatenate(pts), np.array(cols, dtype=object)

    def assert_chart_mirrored(self, toy, pair):
        L, Rr = toy.inst[pair + "_L"], toy.inst[pair + "_R"]
        self.toy = toy
        pa, ca = self._chart_points(L)
        pb, cb = self._chart_points(Rr)
        _, idx = cKDTree(pb).query(pa @ MIRROR_X)
        bad = np.nonzero(ca != cb[idx])[0]
        self.assertGreater(len(set(ca)), 1, f"{pair}: test zone painted nothing")
        # both copies are crocheted alike, so their stitch columns drift the same way round the spiral: the mirror
        # image of a stitch falls between two stitches of the other copy, and a zone's edge stitch may go either
        # way. Allow those few edge stitches, never a zone on the wrong side.
        edges = sum(sum(r[k] != r[k - 1] for k in range(len(r))) for r in toy.chart(L))
        self.assertLessEqual(len(bad), edges, f"{pair}: {len(bad)} stitches differ from the mirrored side, e.g. "
                                      f"{[(ca[k], cb[idx[k]]) for k in bad[:4]]}")

    def test_colour_charts_mirrored(self):
        for p in ("ear", "arm", "inner_ear"):
            with self.subTest(pair=p):
                self.assert_chart_mirrored(_toy(SYM, "sym"), p)

    def test_sideways_piece_mirrored(self):
        """A piece pointing straight sideways (dir along +-X) is mirrored with its colour zones.

        Found by this suite (failed on toy.py before 2026-10-06 16:39; fixed there by Instance.pair_sign):

        Repro:
            parts: head (closed ball), ear ["mr 6","inc 6 to 12","even 2"] open,
                   colours [{"rounds":[1,4],"az":[30,150],"colour":"pink"}]
            assembly: head root; ear on head at {round 4, az 90} dir [1,0,0] mirror
        Instance.set_frame: e2 = e3 x e1 is vertical for such a piece, so az_sign fell back to -(e2 . Y). That
        fallback flips sign between the left and the right copy, and the mirror flip then flipped it again, so
        both copies got the same az_sign: the right ear's zone landed on the opposite side (top vs bottom)."""
        spec = _spec(
            parts=[{"id": "head", "rounds": ["mr 6", "inc 6 to 24", "even 3", "dec 6 to 6"], "colour": "main"},
                   {"id": "ear", "rounds": ["mr 6", "inc 6 to 12", "even 2"], "colour": "main", "end": "open",
                    "colours": [{"rounds": [1, 4], "az": [30, 150], "colour": "pink"}]}],
            assembly=[{"part": "head", "axis": [0, 1, 0]},
                      {"part": "ear", "on": "head", "at": {"round": 4, "az": 90}, "dir": [1, 0, 0], "mirror": True}])
        toy = _toy(spec, "sideways")
        L, Rr = toy.inst["ear_L"], toy.inst["ear_R"]
        T, TH = _grid(L, 8, 48)
        self.assertLess(_mirror_dist(L.base(T, TH).reshape(-1, 3), Rr.base(T, TH).reshape(-1, 3)), 0.3)
        try:
            self.assert_chart_mirrored(toy, "ear")
        except AssertionError as ex:
            raise AssertionError(f"ENGINE BUG (toy.Instance.set_frame az_sign fallback not mirror-safe): {ex}")

    def test_mirrored_names_and_sides(self):
        for p in self.PAIRS:
            L, Rr = self.toy.inst[p + "_L"], self.toy.inst[p + "_R"]
            cL = L.base(*_grid(L, 4, 12)).reshape(-1, 3).mean(0)
            cR = Rr.base(*_grid(Rr, 4, 12)).reshape(-1, 3).mean(0)
            with self.subTest(pair=p):
                self.assertGreater(cL[0], 0.5, f"{p}_L should be on the toy's left (+X)")
                self.assertLess(cR[0], -0.5)


class TestAvoid(unittest.TestCase):
    def _worst(self, avoid):
        spec = _spec(
            parts=[{"id": "body", "rounds": ["mr 6", "inc 6 to 30", "even 5", "dec 6 to 6"], "colour": "main"},
                   {"id": "arm", "rounds": ["mr 6", "12", "even 5", "9"], "colour": "main", "end": "open",
                    "stuff": "light"}],
            assembly=[{"part": "body", "axis": [0, 1, 0]},
                      {"part": "arm", "on": "body", "at": {"round": 8, "az": 80}, "dir": [-0.4, -1, 0.05],
                       "mirror": True, "avoid": avoid}])
        toy = Toy(spec)
        arm, body = toy.inst["arm_L"], toy.inst["body"]
        P, _, T = arm.samples(per_round=3, per_stitch=1.0)
        free = P[T < 0.55 * arm.p.M]
        return float(min(body.penetration(free).min(), 0.0)), toy

    def test_avoid_reduces_penetration(self):
        w0, _ = self._worst(False)
        w1, toy = self._worst(True)
        self.assertLess(w0, -1.0, "test setup: the arm should pass through the body without avoid")
        self.assertGreater(w1, w0 + 0.5, f"avoid did not reduce penetration: {w0:.2f} -> {w1:.2f} mm")
        self.assertTrue(any("avoid" in f for f in toy.flags), toy.flags)


class TestDetails(unittest.TestCase):
    SPEC = _spec(
        parts=[
            {"id": "body", "rounds": ["mr 6", "inc 6 to 24", "even 4", "dec 6 to 12"], "colour": "main", "end": "open"},
            {"id": "head", "rounds": ["mr 6", "inc 6 to 30", "even 3", "dec 6 to 6"], "colour": "main"},
            {"id": "ear", "rounds": ["mr 6", "inc 6 to 18"], "colour": "main", "end": "open",
             "colours": [{"rounds": [3, 3], "colour": "pink"}]},
            {"id": "tail", "cord": {"length_mm": 60, "thickness_mm": 2.6, "curl_deg": 30}, "colour": "pink"},
            {"id": "bow", "flat": {"width_st": 8, "rows": 5}, "colour": "red", "pinch": 0.85},
            {"id": "scarf", "flat": {"width_st": 20, "rows": 3}, "colour": "red"},
        ],
        assembly=[
            {"part": "body", "axis": [0, 1, 0]},
            {"part": "head", "on": "body", "at": {"round": "end"}, "attach": "end", "dir": [0, 1, 0], "sink": 3},
            {"part": "ear", "on": "head", "at": {"round": 3, "az": 50}, "dir": [0.5, 1, 0], "attach": "edge",
             "mirror": True, "cup": 1.0},
            {"part": "tail", "on": "body", "at": {"round": 2, "az": 180}, "dir": [0, -1, -0.3], "toward": [0, -1, 0]},
            {"part": "bow", "on": "body", "at": {"round": 9, "az": 0}, "lift": 1.5},
            {"part": "scarf", "on": "body", "at": {"round": 8, "az": 0}, "conform": True},
        ],
        face=[{"type": "fringe", "on": "head", "points": [[2, 0], [5, 0], [8, 0]], "count": 6, "length_mm": 80,
               "colours": ["pink", "red"], "droop": 1.0},
              {"type": "safety_nose", "on": "head", "at": {"round": 7, "az": 0}, "size_mm": [6, 4],
               "shape": "triangle"},
              {"type": "satin", "on": "head", "at": {"round": 8.5, "az": 0}, "width_mm": 6, "height_mm": 4},
              {"type": "felt", "on": "head", "at": {"round": 5, "az": -30}, "size_mm": [6, 4], "rotate": 20}],
    )

    @classmethod
    def setUpClass(cls):
        cls.toy = Toy(copy.deepcopy(cls.SPEC))
        cls.parts = cls.toy.build(ku=2, kv=2)

    def _mesh_ok(self, m):
        self.assertTrue(_finite(m["V"]), m["name"])
        self.assertGreater(len(m["F"]), 0, m["name"])
        self.assertTrue(np.all(m["F"] >= 0) and np.all(m["F"] < len(m["V"])), m["name"])
        for k in ("C", "N", "H"):
            self.assertEqual(len(m[k]), len(m["V"]), f"{m['name']}.{k}")
        self.assertTrue(_finite(m["N"]))

    def test_all_meshes_valid(self):
        names = [m["name"] for m in self.parts]
        for want in ("ear_L", "ear_R", "tail", "bow", "scarf", "nose@head", "satin@head", "felt@head"):
            self.assertIn(want, names)
        self.assertEqual(sum(n.startswith("fringe@") for n in names), 12)       # 6 strands, doubled
        for m in self.parts:
            self._mesh_ok(m)

    def test_edge_disc_two_layers(self):
        ear = self.toy.inst["ear_L"]
        self.assertIsNotNone(getattr(ear, "edge_face", None))
        m = next(m for m in self.parts if m["name"] == "ear_L")
        n = len(m["V"]) // 2
        gap = np.linalg.norm(m["V"][:n] - m["V"][n:], axis=1)
        self.assertTrue(np.allclose(gap, gap[0], atol=1e-6) and gap[0] > 0.5)
        # sewn by the edge: its nearest point touches the head
        sd, _, _ = self.toy.inst["head"].signed_distance(m["V"][:n])
        self.assertLess(sd.min(), 1.0)

    def test_ground(self):
        loose = [m for m in self.parts if m["name"].startswith("fringe@") or m["name"] == "tail"]
        solid = [m for m in self.parts if not any(m is x for x in loose)]
        self.assertAlmostEqual(min(m["V"][:, 1].min() for m in solid), 0.0, places=6)
        for m in loose:
            self.assertGreaterEqual(m["V"][:, 1].min(), -1e-6, f"{m['name']} below the floor")
        self.assertFalse(any(f.startswith("ground:") and "tail" in f for f in self.toy.flags), self.toy.flags)

    def test_cord_length(self):
        tail = next(m for m in self.toy.cord_meshes if m["name"] == "tail")
        span = np.ptp(tail["V"], axis=0)
        self.assertLess(np.linalg.norm(span), 70)
        self.assertGreater(np.linalg.norm(span), 20)

    def test_flat_rows_size(self):
        bow = next(m for m in self.toy.flat_meshes if m["name"] == "bow")
        ext = np.sort(np.ptp(bow["V"], axis=0))
        self.assertGreater(ext[-1], 8 * 4.0 * 0.9)                # width_st x stitch_mm
        scarf = next(m for m in self.toy.flat_meshes if m["name"] == "scarf")
        sd, _, _ = self.toy.inst["body"].signed_distance(scarf["V"])
        self.assertGreater(np.median(sd), -0.5, "a conformed flat piece lies on the parent, not inside")
        self.assertLess(np.median(np.abs(sd)), 5.0)


class TestSpecChecks(unittest.TestCase):
    BASE = _spec(parts=[{"id": "ball", "rounds": ["mr 6", "inc 6 to 18", "even 2", "dec 6 to 6"], "colour": "main"},
                        {"id": "nub", "rounds": ["mr 6", "even 2"], "colour": "main", "end": "open"}],
                 assembly=[{"part": "ball", "axis": [0, 1, 0]},
                           {"part": "nub", "on": "ball", "at": {"round": 3, "az": 0}}])

    def test_unknown_keys_flagged(self):
        s = copy.deepcopy(self.BASE)
        s["colur"] = 1
        s["parts"][0]["flaten"] = 0.5
        s["assembly"][1]["dirr"] = [0, 0, 1]
        s["face"] = [{"type": "blush", "on": "ball", "at": {"round": 3, "az": 0}, "sizemm": 5}]
        flags = " | ".join(Toy(s).flags)
        for k in ("colur", "flaten", "dirr", "sizemm"):
            self.assertIn(k, flags)
        self.assertFalse([f for f in Toy(self.BASE).flags if "unknown key" in f])

    def test_duplicate_instance_names(self):
        s = copy.deepcopy(self.BASE)
        s["assembly"].append({"part": "nub", "on": "ball", "at": {"round": 3, "az": 90}})
        with self.assertRaises(ValueError):
            Toy(s)
        s = copy.deepcopy(self.BASE)
        s["assembly"].append({"part": "nub", "on": "ball", "at": {"round": 3, "az": 90}, "mirror": True,
                              "name": "nub", "name_r": "nub"})
        with self.assertRaises(ValueError):
            Toy(s)

    def test_duplicate_mirrored_pairs(self):
        """Two mirrored entries of one part without names would both make <part>_L / <part>_R."""
        s = copy.deepcopy(self.BASE)
        s["assembly"] += [{"part": "nub", "on": "ball", "at": {"round": 3, "az": 60}, "mirror": True},
                          {"part": "nub", "on": "ball", "at": {"round": 4, "az": 100}, "mirror": True}]
        with self.assertRaises(ValueError, msg="ENGINE BUG: the second mirrored entry silently replaces nub_L "
                                               "(only the _R name is checked for duplicates)"):
            Toy(s)

    def test_bad_rounds_raise(self):
        s = copy.deepcopy(self.BASE)
        s["parts"][0]["rounds"] = ["mr 6", "inc 6 to 40"]
        with self.assertRaises(RoundError):
            Toy(s)

    def test_missing_parent(self):
        s = copy.deepcopy(self.BASE)
        s["assembly"][1]["on"] = "nothing"
        with self.assertRaises(KeyError):
            Toy(s)


# =========================================================================================================== edit
class TestEdit(unittest.TestCase):
    def setUp(self):
        from . import edit
        self.edit = edit
        self.spec = copy.deepcopy(SYM)
        self.spec["parts"].append({"id": "tail", "rounds": ["mr 6", "even 2"], "colour": "main", "end": "open"})
        self.spec["assembly"].append({"part": "tail", "on": "body", "at": {"round": 3, "az": 180},
                                      "dir": [0, 0, -1]})

    def test_set(self):
        new, rep = self.edit.apply(self.spec, [{"op": "set", "path": "assembly.ear.at.az", "value": 70},
                                               {"op": "set", "path": "parts.head.colour", "value": "tan"},
                                               {"op": "set", "path": "yarn.stitch_mm", "value": 4.2}])
        ear = next(e for e in new["assembly"] if e["part"] == "ear")
        self.assertEqual(ear["at"]["az"], 70)
        self.assertEqual(next(p for p in new["parts"] if p["id"] == "head")["colour"], "tan")
        self.assertEqual(new["yarn"]["stitch_mm"], 4.2)
        self.assertEqual(next(e for e in self.spec["assembly"] if e["part"] == "ear")["at"]["az"], 60, "input changed")
        self.assertEqual(len(rep["log"]), 3)

    def test_add_and_face_ids(self):
        new, _ = self.edit.apply(self.spec, [{"op": "add", "path": "face", "value": {
            "type": "safety_eye", "on": "head", "at": {"round": 5, "az": 0}, "size_mm": 5}}])
        ids = [f["id"] for f in new["face"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(new["face"]), len(self.spec["face"]) + 1)

    def test_remove_part_and_dependants(self):
        new, _ = self.edit.apply(self.spec, [{"op": "remove", "path": "parts.tail"}])
        self.assertFalse([p for p in new["parts"] if p["id"] == "tail"])
        self.assertFalse([e for e in new["assembly"] if e["part"] == "tail"])
        new, _ = self.edit.apply(self.spec, [{"op": "remove", "path": "parts.ear"}])
        self.assertFalse([e for e in new["assembly"] if e.get("on") == "ear"], "inner_ear (on ear) should go too")

    def test_remove_subfield_keeps_part(self):
        new, _ = self.edit.apply(self.spec, [{"op": "remove", "path": "parts.ear.flatten"}])
        ear = next(p for p in new["parts"] if p["id"] == "ear")
        self.assertNotIn("flatten", ear)
        self.assertIn("rounds", ear)
        self.assertTrue([e for e in new["assembly"] if e["part"] == "ear"])

    def test_scale_part_renumbers_children(self):
        new, _ = self.edit.apply(self.spec, [{"op": "scale_part", "part": "head", "factor": 1.4}])
        old_n = len(counts(next(p for p in self.spec["parts"] if p["id"] == "head")["rounds"]))
        head = next(p for p in new["parts"] if p["id"] == "head")
        c = counts(head["rounds"])
        self.assertEqual(check(expand(head["rounds"])), [])
        self.assertGreater(max(c), 30)
        k = len(c) / old_n
        ear = next(e for e in new["assembly"] if e["part"] == "ear")
        self.assertAlmostEqual(ear["at"]["round"], round(3 * k * 2) / 2)
        eye = next(f for f in new["face"] if f["type"] == "safety_eye")
        self.assertAlmostEqual(eye["at"]["round"], round(5.5 * k * 2) / 2)
        line = next(f for f in new["face"] if f["type"] == "stitch")
        self.assertAlmostEqual(line["points"][0][0], round(7 * k, 2))
        mz = next(e for e in new["assembly"] if e["part"] == "muzzle")
        self.assertLessEqual(mz["at"]["round"], len(c))
        # the body (parent of the head, at round "end") is not renumbered
        hd = next(e for e in new["assembly"] if e["part"] == "head")
        self.assertEqual(hd["at"]["round"], "end")

    def test_remove_part_with_grandchildren(self):
        """Removing a part removes everything sewn on it, transitively.

        Regression guard: the first edit.py removed only direct dependants, so removing parts.head from SYM
        (head <- ear <- inner_ear applique) kept inner_ear and validation failed ("parent ear is not placed")."""
        try:
            new, _ = self.edit.apply(self.spec, [{"op": "remove", "path": "parts.head"}])
        except self.edit.EditError as ex:
            self.fail(f"ENGINE BUG (edit.apply remove is not transitive): {ex}")
        self.assertFalse([e for e in new["assembly"] if e.get("on") in ("head", "ear", "muzzle")])

    def test_scale_part_moves_own_zones_and_bend(self):
        """scale_part renumbers the children's rounds; the part's own colour zones and bend `at` must follow too.

        Regression guard: the first engine version renumbered only the children (assembly/face `at`), so after
        scale_part ear 2.0 (4 -> 8 rounds) a zone {"rounds": [2, 4]} and bend {"at": 2} stayed on the old rounds."""
        new, _ = self.edit.apply(self.spec, [{"op": "scale_part", "part": "ear", "factor": 2.0}])
        ear = next(p for p in new["parts"] if p["id"] == "ear")
        old = next(p for p in self.spec["parts"] if p["id"] == "ear")
        k = len(counts(ear["rounds"])) / len(counts(old["rounds"]))
        self.assertGreater(k, 1.2)
        want = [round(2 * k), round(4 * k)]
        self.assertTrue(abs(ear["bend"]["at"] - 2 * k) <= 0.51,
                        f"ENGINE BUG (scale_part leaves bend.at on the old round): {ear['bend']} after x{k:.2f}")
        self.assertEqual([round(x) for x in ear["colours"][0]["rounds"]], want,
                         f"ENGINE BUG (edit scale_part leaves the part's own colour zones on the old rounds): "
                         f"{ear['colours'][0]['rounds']} after rounds x{k:.2f}")

    def test_scale_part_bare_counts(self):
        """scale_part re-grows a piece to ~factor x width even when it is written with bare counts.

        Regression guard: the first scale_steps rescaled only `inc/dec K to M` and `even N` steps, so
        arm ["mr 6", "12", "even 3", "9"] x1.5 kept its width of 12."""
        new, _ = self.edit.apply(self.spec, [{"op": "scale_part", "part": "arm", "factor": 1.5}])
        arm = next(p for p in new["parts"] if p["id"] == "arm")
        self.assertGreaterEqual(max(counts(arm["rounds"])), 15,
                                f"ENGINE BUG (edit.scale_steps ignores bare-count steps): {arm['rounds']}")

    def test_recolour(self):
        new, _ = self.edit.apply(self.spec, [{"op": "recolour", "from": "pink", "to": "#00ff00"}])
        self.assertEqual(new["colours"]["pink"], "#00ff00")
        with self.assertRaises(self.edit.EditError):
            self.edit.apply(self.spec, [{"op": "recolour", "from": "nope", "to": "#000000"}])

    def test_errors_atomic(self):
        with self.assertRaises(self.edit.EditError):
            self.edit.apply(self.spec, [{"op": "frobnicate"}])
        with self.assertRaises(self.edit.EditError):
            self.edit.apply(self.spec, [{"op": "set", "path": "parts.nothing.colour", "value": "red"}])
        with self.assertRaises(self.edit.EditError):        # validation: impossible rounds
            self.edit.apply(self.spec, [{"op": "set", "path": "parts.head.rounds", "value": ["mr 6", "inc 6 to 40"]}])
        self.assertEqual(next(p for p in self.spec["parts"] if p["id"] == "head")["rounds"],
                         SYM["parts"][1]["rounds"])


# ======================================================================================================== library
class TestLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from . import library
        cls.L = library
        cls.P = library.parts()

    def test_parts_loaded(self):
        self.assertGreater(len(self.P), 5000)
        need = {"role", "group", "designer", "counts", "kind", "max_rel", "len_rel"}
        self.assertTrue(need <= set(self.P[0]))

    def test_stats(self):
        for role, subj in (("ear", "bunny"), ("arm", "bear"), ("leg", "cat"), ("tail", "fox"), ("muzzle", "dog")):
            st = self.L.stats(role, subj)
            with self.subTest(role=role, subj=subj):
                self.assertIsNotNone(st)
                self.assertGreaterEqual(st["N"], 8)
                for k in self.L.FEATS:
                    lo, med, hi = st[k]
                    self.assertTrue(lo <= med <= hi, f"{k}: {st[k]}")

    ROLES = {"ear": [None, "long", "floppy", "round", "pointed"], "arm": [None, "paw", "tube", "long_tube"],
             "leg": [None, "paw", "tube"], "tail": [None, "long", "short"], "muzzle": [None], "body": [None],
             "wing": [None], "horn": [None]}
    ANIMALS = ["bear", "bunny", "cat", "dog", "mouse", "fox", "panda", "penguin", "cow", "unicorn", "duck",
               "sloth", "turtle", "hippo", "capybara", "teddy", "shark", "potato", "red_panda", None]

    @classmethod
    def _suggestions(cls):
        if not hasattr(cls, "_sugg"):
            cls._sugg = []
            for role, kinds in cls.ROLES.items():
                for a in cls.ANIMALS:
                    for kind in kinds:
                        for H in (36, 48):
                            s = cls.L.suggest(role, a, H, kind)
                            if s is not None:
                                cls._sugg.append((f"suggest({role!r}, {a!r}, {H}, {kind!r})", role, s))
        return cls._sugg

    def test_suggest_valid_steps(self):
        """suggest() returns steps that expand, with clean arithmetic, to exactly its `counts`."""
        problems = []
        sugg = self._suggestions()
        for tag, role, s in sugg:
            try:
                rs = expand(s["steps"])
            except RoundError as ex:
                problems.append(f"{tag}: steps {s['steps']} do not expand: {ex}")
                continue
            if check(rs):
                problems.append(f"{tag}: arithmetic {check(rs)}")
            if [r.n for r in rs] != list(s["counts"]):
                problems.append(f"{tag}: steps {s['steps']} make {[r.n for r in rs]}, counts {s['counts']}")
            if not s["steps"][0].startswith("mr "):
                problems.append(f"{tag}: does not start with a magic ring: {s['steps']}")
        self.assertGreater(len(sugg), 100)
        self.assertEqual(problems, [], "ENGINE BUG(s) in library.suggest:\n" + "\n".join(problems[:15]))

    def test_suggest_copy_guard(self):
        """Licence guard: suggest() must never hand out a list identical to one designer's part (6+ rounds).

        Repro: library.suggest('arm', 'bunny', 48, 'tube') -> counts [6, 9, 9, 9, 9, 9, 9, 9] with
        copied_guard=True: the guard is only *reported*, the copy is still returned; and the guard compares only
        against the per-designer-capped subset `ds`, so e.g. suggest('arm', 'cat', 48, 'tube') ->
        [6, 10, 10, 10, 10, 10, 10, 10] equals a library arm but reports copied_guard=False.
        (Most hits are generic plain tubes, so this is a policy question as much as a bug.)"""
        flagged, missed = [], []
        for tag, role, s in self._suggestions():
            lib_role = self._lib_role(role)
            if s["copied_guard"]:
                flagged.append(f"{tag}: {s['counts']}")
            elif len(s["counts"]) >= 6 and tuple(s["counts"]) in lib_role:
                missed.append(f"{tag}: {s['counts']}")
        msg = (f"ENGINE BUG (library.suggest licence guard): {len(flagged)} suggestions are an exact designer list "
               f"and still returned (copied_guard=True), {len(missed)} are exact library lists the guard misses.\n"
               + "\n".join((flagged + missed)[:12]))
        self.assertEqual(flagged + missed, [], msg)

    @classmethod
    def _lib_role(cls, role):
        if not hasattr(cls, "_lr"):
            cls._lr = {}
        if role not in cls._lr:
            cls._lr[role] = {tuple(d["counts"]) for d in cls.P if d["role"] == role}
        return cls._lr[role]

    def test_audit_toys(self):
        for f in TOYS[:6]:
            spec = json.loads(f.read_text(encoding="utf-8-sig"))
            a = self.L.audit(spec)
            with self.subTest(toy=f.stem):
                self.assertGreater(a["head_max"], 0)
                for r in a["parts"]:
                    self.assertIn(r["role"], ("ear", "arm", "leg", "tail", "muzzle", "beak", "wing", "fin", "horn",
                                              "shell", "body"))


# ========================================================================================================= export
class TestExport(unittest.TestCase):
    def test_glb(self):
        import trimesh
        from .export import to_glb
        toy = _toy(TestDetails.SPEC, "details")
        parts = toy.build(ku=2, kv=2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.glb")
            to_glb(parts, path)
            self.assertGreater(os.path.getsize(path), 1000)
            sc = trimesh.load(path, force="scene")
            nodes = [n for n in sc.graph.nodes_geometry]
            self.assertEqual(len(nodes), len(parts), "one node per mesh")
            self.assertEqual(len(set(nodes)), len(nodes), "node names unique")
            self.assertEqual(len(sc.geometry), len(parts))
            nv = sum(len(g.vertices) for g in sc.geometry.values())
            self.assertEqual(nv, sum(len(p["V"]) for p in parts))
            for want in ("body", "head", "ear_L", "ear_R", "tail", "bow"):
                self.assertIn(want, nodes)

    def test_glb_name_collision(self):
        """A part literally named like the de-duplicated name of another (eye@head#2) must still be unique."""
        import trimesh
        from .export import to_glb
        tri = {"V": np.eye(3), "F": np.array([[0, 1, 2]]), "C": np.ones((3, 3)), "N": np.tile([0, 0, 1.0], (3, 1)),
               "H": np.ones(3), "mat": 1}
        parts = [dict(tri, name="a"), dict(tri, name="a"), dict(tri, name="a#2")]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.glb")
            try:
                to_glb(parts, path)
                sc = trimesh.load(path, force="scene")
                nodes = list(sc.graph.nodes_geometry)
            except Exception as ex:  # noqa: BLE001
                self.fail(f"ENGINE BUG (export.to_glb de-dup only appends #k, can collide): {ex!r}")
            self.assertEqual(len(set(nodes)), 3, f"ENGINE BUG (export.to_glb name collision): nodes {nodes}")


# ========================================================================================================= render
class TestRenderSmoke(unittest.TestCase):
    def test_tiny_render(self):
        try:
            from .render import Renderer
            r = Renderer(size=64, ssaa=1)
        except Exception as ex:  # noqa: BLE001 - no GL context on this machine
            self.skipTest(f"no OpenGL context: {ex}")
        toy = _toy(TestSpecChecks.BASE, "base")
        parts = toy.build(ku=2, kv=2)
        r.set_scene(parts, ao_dirs=4)
        im = np.asarray(r.render(35.0))
        self.assertEqual(im.shape, (64, 64, 3))
        self.assertGreater(im.std(), 3.0, "blank render")
        r.release()

    def test_transparent_preview(self):
        """render(transparent=True): RGBA, the toy opaque, the corners (backdrop) clear; crop_alpha squares
        the canvas around what is visible."""
        try:
            from .render import Renderer
            r = Renderer(size=96, ssaa=1)
        except Exception as ex:  # noqa: BLE001 - no GL context on this machine
            self.skipTest(f"no OpenGL context: {ex}")
        from .preview import crop_alpha
        toy = _toy(TestSpecChecks.BASE, "base")
        r.set_scene(toy.build(ku=2, kv=2), ao_dirs=4)
        im = r.render(35.0, transparent=True)
        r.release()
        a = np.asarray(im)[:, :, 3]
        self.assertEqual(im.mode, "RGBA")
        self.assertEqual(int(a[0, 0]), 0, "backdrop corner must be clear")
        self.assertEqual(int(a[0, -1]), 0, "backdrop corner must be clear")
        self.assertEqual(int(a[48, 48]), 255, "the toy (centre of the frame) must be opaque")
        c = crop_alpha(im)
        self.assertEqual(c.width, c.height)
        self.assertLessEqual(c.width, int(96 * 1.2) + 1)


# =========================================================================================================== toys
@unittest.skipIf(os.environ.get("AMIKIT_SKIP_TOYS"), "AMIKIT_SKIP_TOYS set")
class TestAllToys(unittest.TestCase):
    def test_every_toy_builds_and_pattern_checks(self):
        from .pattern_text import pattern_model, self_check
        self.assertGreater(len(TOYS), 0)
        for f in TOYS:
            with self.subTest(toy=f.stem):
                spec = json.loads(f.read_text(encoding="utf-8-sig"))
                toy = Toy(spec)
                parts = toy.build(ku=2, kv=2)
                self.assertTrue(all(_finite(m["V"]) for m in parts), "non-finite vertices")
                for p in spec["parts"]:
                    if "rounds" in p:
                        self.assertEqual(check(expand(p["rounds"])), [], p["id"])
                model = self_check(pattern_model(toy))
                errs = [f"{p['id']}: {e}" for p in model["parts"] for e in p.get("check", []) + p["self_check"]]
                self.assertEqual(errs, [], f"{f.stem}: pattern arithmetic")
                self.assertFalse([x for x in toy.flags if "unknown key" in x], toy.flags)


if __name__ == "__main__":
    unittest.main()
