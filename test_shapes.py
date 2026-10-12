"""Shaped flat pieces (motifs), chain-ring pieces (collars, snoods, split collars), eye-anchored appliques.

    python -m unittest amikit.test_shapes
"""
import copy
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .motif import MotifDef, KINDS, row_text, parse_row_text, infer, check_rows, HT
from .piece import PieceDef
from .rounds import expand, check
from .toy import Toy
from .pattern_text import write_patterns, ru_decline
from .reverse import to_spec
from .roundtrip import compare

HERE = Path(__file__).parent
TEST_TOY = HERE / "toys" / "_shapes_lab" / "shapes_test.json"
YARN = {"stitch_mm": 4.0, "round_mm": 3.6, "thickness_mm": 1.9}
CASES = [{"kind": "triangle", "width": 9}, {"kind": "triangle", "width": 8, "ends": "one", "sew": "side"},
         {"kind": "triangle", "width": 6, "stitch": "hdc", "every": 2}, {"kind": "leaf", "chain": 9},
         {"kind": "leaf", "chain": 10, "tall": "tr", "tip": "none"}, {"kind": "hexagon", "rounds": 3},
         {"kind": "circle", "rounds": 4, "edging": {"kind": "shell", "k": 5, "every": 4}},
         {"kind": "half_circle", "rows": 4, "edging": {"kind": "shell", "k": 7, "every": 4}},
         {"kind": "half_circle", "rows": 5}, {"kind": "crescent", "chain": 12}, {"kind": "heart"},
         {"kind": "mask", "chain": 8, "rounds": 3}, {"kind": "circle", "rounds": 3, "edging": {"kind": "picot", "every": 3}},
         {"kind": "heart", "rounds": 3}, {"kind": "star"}, {"kind": "star", "points": 6, "length": 6, "tall": "tr"},
         {"kind": "lobed", "lobes": 3}, {"kind": "lobed", "lobes": 5, "tall": "tr", "rounds": 2},
         {"kind": "petal", "chain": 6}, {"kind": "petal", "chain": 5, "tall": "tr"},
         {"kind": "half_circle", "rows": 5, "edging": {"kind": "point", "every": 5, "length": 5, "where": "curve"}},
         {"kind": "circle", "rounds": 4, "edging": {"kind": "feather", "every": 4}},
         {"kind": "circle", "rounds": 3, "edging": {"kind": "point", "every": 3, "length": 4}}]


def _load():
    return json.loads(TEST_TOY.read_text(encoding="utf-8-sig"))


class MotifRowsTests(unittest.TestCase):
    def test_every_kind_is_written_checked_and_read_back(self):
        self.assertEqual({c["kind"] for c in CASES}, set(KINDS))
        for m in CASES:
            p = MotifDef("x", {"motif": m}, YARN)
            self.assertEqual(p.check(), [], m)
            ops, phr = [], []
            for r in p.rows:
                for lang in ("ru", "en"):
                    self.assertTrue(row_text(r, lang))
                ph, o = parse_row_text(row_text(r, "en"))
                self.assertEqual(o, r["ops"], m)
                ops.append(o); phr.append(ph)
            back = infer(m["kind"], ops, phr)
            self.assertIsNotNone(back, m)
            self.assertEqual(MotifDef("x", {"motif": back}, YARN).counts, p.counts, m)
            self.assertEqual([r["ops"] for r in MotifDef("x", {"motif": back}, YARN).rows], ops, m)
            if m.get("edging", {}).get("where"):
                self.assertEqual(back["edging"]["where"], m["edging"]["where"])

    def test_check_catches_wrong_counts(self):
        p = MotifDef("x", {"motif": {"kind": "triangle", "width": 9}}, YARN)
        rows = copy.deepcopy(p.rows)
        rows[2]["ops"] = rows[2]["ops"][:-1]
        self.assertTrue(check_rows(rows))

    def test_geometry_follows_the_rows(self):
        w, h = YARN["stitch_mm"], YARN["round_mm"]
        tri = MotifDef("x", {"motif": {"kind": "triangle", "width": 10}}, YARN)
        up, across = tri.size_mm()
        self.assertAlmostEqual(across, 10 * w, delta=0.5)                  # row 1: 10 stitch widths
        self.assertAlmostEqual(up, len(tri.rows) * h, delta=0.5)           # one sc height per row
        hdc = MotifDef("x", {"motif": {"kind": "triangle", "width": 10, "stitch": "hdc"}}, YARN)
        self.assertAlmostEqual(hdc.size_mm()[0], len(hdc.rows) * 1.5 * h, delta=0.5)
        leaf = MotifDef("x", {"motif": {"kind": "leaf", "chain": 10, "tall": "tr"}}, YARN)
        self.assertAlmostEqual(leaf.size_mm()[1], 2 * HT["tr"] * h, delta=1.5)  # widest: a tr each side
        hexa = MotifDef("x", {"motif": {"kind": "hexagon", "rounds": 4}}, YARN)
        self.assertAlmostEqual(hexa.size_mm()[0], 2 * 24 * w / 6, delta=0.6)    # corner to corner: 2 sides
        half = MotifDef("x", {"motif": {"kind": "half_circle", "rows": 5}}, YARN)
        R = 15 * w / math.pi
        self.assertAlmostEqual(half.size_mm()[1], 2 * R, delta=0.8)
        self.assertAlmostEqual(half.size_mm()[0], R, delta=0.8)
        # a crescent curls: its outer edge (more stitches) is longer than its chain
        cr = MotifDef("x", {"motif": {"kind": "crescent", "chain": 12}}, YARN)
        x, y = cr.flat_points(np.full(5, 0.0), np.linspace(0, 2 * np.pi, 5))
        self.assertGreater(x[2], x[0] + 2.0)       # the chain is an arch: its middle rises above its ends

    def test_orientation_is_consistent(self):
        """Every motif's bands run the same way round as a magic-ring disc (the applique normal points out)."""
        for m in CASES:
            p = MotifDef("x", {"motif": m}, YARN)
            for th in np.linspace(0.4, 5.9, 9):
                t = np.array([p.M - 0.5])
                x0, y0 = p.flat_points(t, [th]); x1, y1 = p.flat_points(t, [th + 0.01])
                x2, y2 = p.flat_points(t + 0.01, [th])
                cr = float(((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))[0])
                self.assertLessEqual(cr, 1e-9, (m, th))

    def test_shell_edging_scallops(self):
        p = MotifDef("x", {"motif": {"kind": "circle", "rounds": 4, "edging": {"kind": "shell", "k": 5, "every": 4}}}, YARN)
        th = np.linspace(0, 2 * np.pi, 361)
        x, y = p.flat_points(np.full_like(th, p.M), th)
        r = np.hypot(x, y)
        # 6 shells: the outline rises and falls 6 times, by about a dc
        peaks = np.sum((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:]))
        self.assertEqual(peaks, 6)
        self.assertGreater(r.max() - r.min(), 1.2 * YARN["round_mm"])

    def test_new_kinds_grow_with_their_size_keys(self):
        w, h = YARN["stitch_mm"], YARN["round_mm"]

        def size(m):
            return MotifDef("x", {"motif": m}, YARN).size_mm()

        def lobes(m, frac=0.8):
            """How many separate stretches of the outer edge stand out (points, lobes)."""
            r = np.linalg.norm(MotifDef("x", {"motif": m}, YARN).outline(720), axis=1)   # from the magic ring
            hi = r > r.min() + frac * (r.max() - r.min())
            return int(np.sum(hi & ~np.roll(hi, 1)))
        # a heart with sc rounds: each round adds about two stitch heights across
        h0, h3 = size({"kind": "heart"}), size({"kind": "heart", "rounds": 3})
        self.assertAlmostEqual(h3[1] - h0[1], 2 * 3 * h, delta=0.25 * 6 * h)
        self.assertGreater(size({"kind": "heart", "rounds": 5})[1], 2.5 * h0[1])
        # a star: 5 points, each a chain long
        self.assertEqual(lobes({"kind": "star"}, 0.6), 5)
        self.assertEqual(lobes({"kind": "star", "points": 6, "length": 6, "tall": "tr"}, 0.6), 6)
        s5, s8 = size({"kind": "star", "length": 5}), size({"kind": "star", "length": 8})
        self.assertGreater(s8[0], s5[0] + 3 * w)
        R1 = 10 * w / (2 * np.pi)
        self.assertAlmostEqual(MotifDef("x", {"motif": {"kind": "star"}}, YARN).R_out, R1 + 4 * w + 0.3 * h, delta=2.5)
        # lobed: the lobes are there; rounds of sc make it bigger
        self.assertEqual(lobes({"kind": "lobed", "lobes": 3}, 0.6), 3)
        self.assertEqual(lobes({"kind": "lobed", "lobes": 5, "tall": "tr"}, 0.6), 5)
        l0, l2 = size({"kind": "lobed", "lobes": 4}), size({"kind": "lobed", "lobes": 4, "rounds": 2})
        self.assertAlmostEqual(l2[0] - l0[0], 2 * 2 * h, delta=0.35 * 4 * h)
        # petal: a chain long plus the fan at the tip, two tall stitches wide
        pe = size({"kind": "petal", "chain": 6})
        self.assertAlmostEqual(pe[0], 5.5 * w + HT["dc"] * h, delta=2.0)
        self.assertAlmostEqual(pe[1], 2 * HT["dc"] * h, delta=1.5)
        self.assertGreater(size({"kind": "petal", "chain": 9})[0], pe[0] + 2.5 * w)
        # wing edgings: one point per `every` stitches on the curve, longer chains reach further
        wing = {"kind": "half_circle", "rows": 5, "edging": {"kind": "point", "every": 5, "length": 5}}
        self.assertEqual(lobes(wing, 0.6), 3)
        long = {"kind": "half_circle", "rows": 5, "edging": {"kind": "point", "every": 5, "length": 8}}
        self.assertGreater(size(long)[0], size(wing)[0] + 2 * w)
        fe = MotifDef("x", {"motif": {"kind": "circle", "rounds": 4, "edging": {"kind": "feather", "every": 4}}}, YARN)
        self.assertGreater(fe.R_out, 24 * w / (2 * np.pi) + 1.8 * h)


class RingRoundsTests(unittest.TestCase):
    def test_round_steps(self):
        r = expand(["ring 36", "inc 6 to 48", "rib 48 x2", "shells 5 every 4"])
        self.assertEqual([x.n for x in r], [36, 42, 48, 48, 48, 72])
        self.assertEqual(r[0].start, "ring")
        self.assertEqual([x.loop for x in r][3:], ["rib", "rib", "shell"])
        self.assertEqual(check(r), [])

    def test_ring_geometry(self):
        p = PieceDef("c", {"rounds": ["ring 36", "even 2"], "end": "open", "stuff": "none"}, YARN)
        self.assertAlmostEqual(float(p.rb[0]), 36 * 4.0 / (2 * np.pi), delta=0.3)     # a circle, not a point
        rib = PieceDef("s", {"rounds": ["ring 36", "rib 36 x2"], "end": "open", "stuff": "none"}, YARN)
        tube = PieceDef("s", {"rounds": ["ring 36", "even 2"], "end": "open", "stuff": "none"}, YARN)
        self.assertGreater(np.ptp(rib.zb), np.ptp(tube.zb) + 2 * 3.6 * 0.8)          # dc rounds: twice as tall

    def test_split_ring_opens_a_notch(self):
        slow = PieceDef("c", {"rounds": ["ring 30", "inc 4 x4"], "split": True, "end": "open"}, YARN)
        full = PieceDef("c", {"rounds": ["ring 30", "inc 6 x4"], "split": True, "end": "open"}, YARN)

        def gap(p):
            th = np.array([0.0, 2 * np.pi])
            a, b = p.local_points(np.full(2, float(p.M)), th)
            return float(np.linalg.norm(a - b))
        self.assertGreater(gap(slow), gap(full) + 8.0)   # fewer increases than a flat circle needs: a wider notch


class ToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _load()
        cls.toy = Toy(cls.spec)
        cls.parts = cls.toy.build(ku=3, kv=2)

    def test_builds_without_flags_but_ground(self):
        self.assertEqual([f for f in self.toy.flags if not f.startswith("ground")], [])

    def test_mirrored_motif_is_a_mirror_image(self):
        L, R = self.toy.inst["wing_L"], self.toy.inst["wing_R"]
        t = np.linspace(0.5, L.p.M - 0.5, 6); th = np.linspace(0.3, 6.0, 6)
        T, TH = np.meshgrid(t, th)
        PL = L.base(T, TH).reshape(-1, 3)
        PR = R.base(T, TH).reshape(-1, 3)
        M = PR.copy(); M[:, 0] *= -1
        from scipy.spatial import cKDTree
        dense = R.base(*np.meshgrid(np.linspace(0, R.p.M, 40), np.linspace(0, 2 * np.pi, 80))).reshape(-1, 3)
        dense[:, 0] *= -1
        d, _ = cKDTree(dense).query(PL)
        self.assertLess(d.max(), 1.5)

    def test_collar_lies_on_the_body(self):
        body = self.toy.inst["body"]; col = self.toy.inst["collar"]
        P, _, _ = col.samples(per_round=2, per_stitch=1.0)
        pen = body.penetration(P)
        self.assertGreater(float(pen.min()), -YARN["thickness_mm"])

    def test_mask_follows_the_eyes(self):
        spec = copy.deepcopy(self.spec)
        spec["face"][0]["at"]["round"] = 7
        t2 = Toy(spec)
        a = self.toy.inst["mask"].origin; b = t2.inst["mask"].origin
        self.assertGreater(b[1] - a[1], 3.0)            # the eyes moved up two rounds: the mask went up with them

    def test_drape_keeps_lengths(self):
        k = self.toy.inst["kerchief"]
        th = np.linspace(0, 2 * np.pi, 200)
        P = k.base(np.zeros_like(th), th)                 # the foundation row, laid around the body
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        self.assertAlmostEqual(L, 22 * 4.0, delta=4.0)

    def test_pattern_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            model = write_patterns(self.toy, d)
            en = (Path(d) / "PATTERN_EN.md").read_text(encoding="utf8")
            ru = (Path(d) / "PATTERN_RU.md").read_text(encoding="utf8")
        for p in model["parts"]:
            self.assertEqual(p["check"] + p.get("self_check", []), [], p["id"])
        self.assertIn("Flat motif — triangle", en)
        self.assertIn("(shells): (skip 1 st, 5 dc in one st, skip 1 st, 1 sl st) ×12", en)
        self.assertIn("ch 36 and join into a ring", en)
        self.assertIn("(ribbing): (fpdc, bpdc) ×18", en)
        self.assertIn("Плоский мотив — треугольник", ru)
        self.assertIn("(РЛСН, РИСН) ×18", ru)
        spec2, log = to_spec(en)
        res, _, _ = compare(self.spec, spec2)
        self.assertTrue(res["counts_ok"], res)
        self.assertTrue(res["colours_ok"], res)
        self.assertLess(res["max_mm"], 3.0, res)
        kinds = {p["id"]: p.get("motif", {}).get("kind") for p in spec2["parts"]}
        self.assertEqual(kinds["wing"], "triangle")
        self.assertEqual(sum(1 for p in spec2["parts"] if p.get("split")), 1)


class RussianTextTests(unittest.TestCase):
    def test_neuter_after_hissing(self):
        self.assertEqual(ru_decline("туловище", "gen"), "туловища")
        self.assertEqual(ru_decline("туловище", "dat"), "туловищу")
        self.assertEqual(ru_decline("сердце", "gen"), "сердца")
        self.assertEqual(ru_decline("ушко", "gen"), "ушка")
        self.assertEqual(ru_decline("тело", "gen"), "тела")

    def test_engine_check_hidden_from_the_crocheter(self):
        # a correct pattern shows no engine section; the summary stays in a comment for the engine
        spec = json.loads((HERE / "toys" / "teddy.json").read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory() as d:
            write_patterns(Toy(spec), d)
            ru = (Path(d) / "PATTERN_RU.md").read_text(encoding="utf8")
        self.assertNotIn("## Проверка", ru)
        self.assertNotIn("Точно:", re.sub(r"<!--.*?-->", "", ru, flags=re.S))
        self.assertIn("<!-- engine check:", ru)
        self.assertIn("round arithmetic OK", ru)


class PresetTests(unittest.TestCase):
    def test_round_crown_is_opt_in_and_domed(self):
        from .presets import expand as ex, head_rounds
        base = {"parts": [{"id": "head", "preset": "cute_head", "max": 66}], "assembly": []}
        rnd = {"parts": [{"id": "head", "preset": "cute_head", "max": 66, "crown": "round"}], "assembly": []}
        self.assertEqual(ex(base)["parts"][0]["rounds"], head_rounds(66))       # unchanged default
        r = ex(rnd)["parts"][0]["rounds"]
        self.assertEqual(check(expand(r)), [])
        a = PieceDef("head", {"rounds": head_rounds(66)}, YARN)
        b = PieceDef("head", {"rounds": r}, YARN)
        H = lambda p: float(np.ptp(p.zb))
        self.assertLess(2 * b.rb.max() / H(b), 2 * a.rb.max() / H(a))          # less of a flat-topped box
        self.assertGreater(2 * b.rb.max() / H(b), 1.05)                         # still wider than tall


if __name__ == "__main__":
    unittest.main()
