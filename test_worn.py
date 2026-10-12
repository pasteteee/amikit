"""Tests of the 2026-10-10 constructions: worn pieces (`wear`: a hair cap / hood / hat put on like a hat, keeping its
own shape over a layer of stuffing), front openings (`opening`: turned rows leaving the face window unworked), split
chain-ring pieces that fall along the parent, flat strips laid along a curve without stretching, and the head of a
one-piece toy in cute.py.

Run from amigurumi-lab/:  ../.venv/Scripts/python -m unittest amikit.test_worn
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .toy import Toy, DEFAULT_YARN, is_soft
from .pattern_text import write_patterns
from .reverse import to_spec
from .roundtrip import compare


def _hood_spec(opening=True, pad=3.0, rounds=("mr 6", "inc 6 to 24", "inc 6 to 42 every 2", "48", "even 8")):
    wig = {"id": "wig", "rounds": list(rounds), "colour": "hair", "end": "open", "stuff": "none"}
    if opening:
        wig["opening"] = {"from": 11, "stitches": 16}
    return {
        "name": "t", "title": "t", "yarn": dict(DEFAULT_YARN),
        "colours": {"fur": "#f0f0f0", "hair": "#8a3420"},
        "parts": [
            {"id": "head", "rounds": ["mr 6", "inc 6 to 36", "even 8", "dec 6 to 6"], "colour": "fur"},
            wig,
        ],
        "assembly": [
            {"part": "head", "axis": [0, 1, 0]},
            {"part": "wig", "on": "head", "at": {"round": "end"}, "wear": True, "dir": [0, -1, 0], "pad": pad},
        ],
        "face": [],
    }


def _mesh(toy, name):
    return next(m for m in toy.build(ku=2, kv=2) if m["name"] == name)


class TestWorn(unittest.TestCase):
    def test_rests_on_the_parent_with_the_padding(self):
        """A worn cap slides down its axis until its inside is `pad` mm (+ half the fabric) off the head; nothing
        of it is inside the head, and it keeps its worked shape (it is not laid onto the head like a soft flap)."""
        for pad in (1.0, 4.0):
            toy = Toy(_hood_spec(opening=False, pad=pad))
            wig, head = toy.inst["wig"], toy.inst["head"]
            P, _, _ = wig.samples()
            sd = head.signed_distance(P)[0]
            clear = pad + 0.5 * wig.p.thick
            self.assertGreater(sd.min(), clear - 0.6, f"pad {pad}: the cap passes into the head ({sd.min():.2f})")
            self.assertLess(sd.min(), clear + 0.6, f"pad {pad}: the cap does not rest on the head ({sd.min():.2f})")
            self.assertIsNone(getattr(wig, "push", None), "a worn piece must keep its own shape (no soft push)")
            self.assertTrue(is_soft(wig.p))                 # (an unstuffed cup: soft as a part, kept by `wear`)
            self.assertFalse([f for f in toy.flags if "never rests" in f], toy.flags)

    def test_bigger_cap_stands_off(self):
        """A cap wider than the head stands off its sides: the gap at the side grows with the cap's count."""
        gaps = []
        for mx in (42, 54):
            toy = Toy(_hood_spec(opening=False, rounds=("mr 6", "inc 6 to 24", f"inc 6 to {mx} every 2", "even 6")))
            wig, head = toy.inst["wig"], toy.inst["head"]
            th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            X = wig.base(np.full(24, float(wig.p.N) - 2), th)
            gaps.append(float(np.median(head.signed_distance(X)[0])))
        self.assertGreater(gaps[1], gaps[0] + 2.0, gaps)


class TestOpening(unittest.TestCase):
    def test_window_has_no_fabric(self):
        toy = Toy(_hood_spec())
        m = _mesh(toy, "wig")
        wig = toy.inst["wig"]
        c = wig.origin
        V = m["V"] - np.array([0.0, toy.ground_shift, 0.0])
        # below the opening's first row, nothing in front (az 0 = +z here) within the window's half angle
        low = V[:, 1] < (c[1] - 45.0)
        az = np.degrees(np.arctan2(V[:, 0] - c[0], V[:, 2] - c[2]))
        half = 0.5 * 16 / 48 * 360
        self.assertFalse(np.any(low & (np.abs(az) < half - 8)), "fabric left in the face opening")
        self.assertTrue(np.any(low & (np.abs(az) > 120)), "the back of the hood is missing")
        # every vertex belongs to a face (no stray points in the hole)
        used = np.zeros(len(m["V"]), bool); used[m["F"].reshape(-1)] = True
        self.assertTrue(used.all())

    def test_errors_are_flagged(self):
        s = _hood_spec()
        s["parts"][1]["opening"] = {"from": 30, "stitches": 16}
        self.assertTrue([f for f in Toy(s).flags if "opening" in f])
        s = _hood_spec(rounds=("mr 6", "inc 6 to 24", "inc 6 to 42 every 2", "48", "even 2", "54", "even 4"))
        self.assertTrue([f for f in Toy(s).flags if "opening" in f and "unworked" in f],
                        "an increase inside the unworked window must be flagged")

    def test_pattern_rows_and_round_trip(self):
        """The written rows have count - g stitches, the materials leave the window out, and the reverse reader
        rebuilds the same hood (opening + wear) from the English pattern."""
        spec = _hood_spec()
        toy = Toy(spec)
        with tempfile.TemporaryDirectory() as d:
            write_patterns(toy, d)
            ru = (Path(d) / "PATTERN_RU.md").read_text(encoding="utf8")
            en = (Path(d) / "PATTERN_EN.md").read_text(encoding="utf8")
        self.assertIn("(поворотные ряды): 32 сбн (32)", ru)
        self.assertIn("как шапочку", ru)
        self.assertIn("like a hat", en)
        spec2, log = to_spec(en)
        wig2 = next(p for p in spec2["parts"] if "opening" in p)
        self.assertEqual(wig2["opening"], {"from": 11, "stitches": 16})
        e2 = next(e for e in spec2["assembly"] if e.get("wear"))
        self.assertAlmostEqual(e2["pad"], 3.0)
        res, _, _ = compare(spec, spec2)
        self.assertTrue(res["counts_ok"], res)
        self.assertLess(res["max_mm"], 1.0, f"{res['max_mm']} mm")


class TestSplitFall(unittest.TestCase):
    def test_split_cape_lies_on_the_body(self):
        """ENGINE BUG (fixed 2026-10-10): a split ring piece with `fall` flew out sideways - its push grid wrapped
        theta 2*pi onto 0, i.e. one edge of the gap onto the other."""
        spec = {"name": "t", "colours": {"a": "#ffffff", "b": "#aa3322"},
                "parts": [{"id": "body", "rounds": ["mr 6", "inc 6 to 36", "even 12", "dec 6 to 6"], "colour": "a"},
                          {"id": "cape", "rounds": ["ring 24", "even 4", "27", "30"], "split": True, "colour": "b",
                           "end": "open", "stuff": "none"}],
                "assembly": [{"part": "body", "axis": [0, 1, 0]},
                             {"part": "cape", "on": "body", "at": {"round": 14}, "around": True, "fall": True}]}
        t = Toy(spec)
        parts = t.build(ku=2, kv=2)
        cape = next(m for m in parts if m["name"] == "cape")["V"]
        body = next(m for m in parts if m["name"] == "body")["V"]
        c = body.mean(0)
        r = np.linalg.norm((cape - c)[:, [0, 2]], axis=1)
        rb = np.linalg.norm((body - c)[:, [0, 2]], axis=1).max()
        self.assertLess(r.max(), rb + 6.0, f"cape sticks out to {r.max():.1f} mm (body {rb:.1f})")
        az = np.degrees(np.arctan2(cape[:, 0] - c[0], cape[:, 2] - c[2]))
        self.assertLess(float(np.mean(np.abs(az) < 30)), 0.02, "the split's gap must stay open at the front")


class TestFlatConform(unittest.TestCase):
    def test_strip_keeps_its_length_on_the_surface(self):
        """A flat strip laid along a curve wraps as far as its length reaches (it used to be the flat rectangle
        projected onto the surface: shorter on the parent than the written strip)."""
        W = 30
        spec = {"name": "t", "colours": {"a": "#ffffff", "b": "#aa3322"},
                "parts": [{"id": "body", "rounds": ["mr 6", "inc 6 to 36", "even 12", "dec 6 to 6"], "colour": "a"},
                          {"id": "band", "flat": {"width_st": W, "rows": 3}, "colour": "b"}],
                "assembly": [{"part": "body", "axis": [0, 1, 0]},
                             {"part": "band", "on": "body", "at": {"round": 12, "az": 0}, "conform": True}]}
        t = Toy(spec)
        band = t.flat_meshes[0]["V"]
        body = t.inst["body"]
        c = body.origin + body.R @ body.p.anchor(12.0, **body.lp)
        az = np.degrees(np.arctan2(band[:, 0] - c[0], band[:, 2] - c[2]))
        r = float(body.p.r_t(12.0))
        want = np.degrees(W * float(DEFAULT_YARN["stitch_mm"]) / 2 / (r + 1.5))
        self.assertGreater(float(np.abs(az).max()), 0.85 * want, (float(np.abs(az).max()), want))


class TestOnePieceCute(unittest.TestCase):
    def test_head_band_of_a_one_piece_toy(self):
        """The head of a one-piece toy is the band from the crown to the chin under the muzzle, not the whole toy."""
        from .cute import measure
        spec = {"name": "one_piece", "colours": {"brown": "#8a5a3c", "tan": "#d2a679"},
                "parts": [{"id": "body", "rounds": ["mr 6", "inc 6 to 48", "even 6", "42", "even 8", "dec 6 to 6"],
                           "colour": "brown"},
                          {"id": "muzzle", "rounds": ["mr 6", "inc 6 to 18", "even 2"], "colour": "tan",
                           "end": "open", "stuff": "light"}],
                "assembly": [{"part": "body", "axis": [0, 1, 0]},
                             {"part": "muzzle", "on": "body", "at": {"round": 18, "az": 0}, "dir": [0, 0, 1]}],
                "face": [{"type": "safety_eye", "on": "body", "at": {"round": 21, "az": 22}, "size_mm": 9,
                          "mirror": True}]}
        toy = Toy(spec)
        f = measure(toy, toy.build(ku=2, kv=2))
        self.assertLess(f["head_h_mm"], 0.7 * f["toy_h_mm"], f)
        self.assertIn("head_body", f)


if __name__ == "__main__":
    unittest.main()
