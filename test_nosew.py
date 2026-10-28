"""Tests of the no-sew construction (amikit/nosew.py): pieces crocheted into the parent's rounds.

    ../.venv/Scripts/python -m unittest amikit.test_nosew      (from amigurumi-lab/)
"""
import json
import unittest
from pathlib import Path

from . import nosew
from .pattern_text import pattern_model, render_markdown
from .toy import Toy

HERE = Path(__file__).resolve().parent
BUNNY = json.loads((HERE / "toys" / "bunny.json").read_text(encoding="utf-8-sig"))
DOG = json.loads((HERE / "toys" / "pets" / "dog.json").read_text(encoding="utf-8-sig"))


def nosew_toy(spec):
    return Toy(dict(spec, construction="nosew"))


class TestConvert(unittest.TestCase):
    def test_limbs_ears_tails_joined_rest_sewn(self):
        s = nosew.convert(BUNNY)
        joined = {e["part"] for e in s["assembly"] if e.get("join") == "crochet"}
        self.assertEqual(joined, {"ear", "arm", "leg"})
        kept = dict((n, why) for n, why in s["_nosew_kept"])
        self.assertIn("closed", kept["tail"])              # a closed piece has no opening to join
        self.assertIn("applique", kept["inner_ear"])

    def test_names(self):
        for pid in ("ear", "front_leg", "hind_foot", "tail", "wing", "arm"):
            self.assertTrue(nosew.wants_join(pid), pid)
        for pid in ("muzzle", "inner_ear", "cheek", "belly_patch"):
            self.assertFalse(nosew.wants_join(pid), pid)


class TestJoins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bunny = nosew_toy(BUNNY)
        cls.dog = nosew_toy(DOG)

    def test_stitch_counts(self):
        for toy in (self.bunny, self.dog):
            errs = [x for x in nosew.check(toy) if not x.startswith("note:")]
            self.assertEqual(errs, [])

    def test_one_stitch_per_pair(self):
        for inst, parent, e in nosew.joins(self.bunny):
            jn = e["_join"]
            self.assertEqual(jn["k"], (inst.p.counts[-1] + 1) // 2)
            self.assertEqual(jn["n"], parent.p.counts[jn["round"] - 1])

    def test_edge_lies_on_its_stitches(self):
        # the pressed edge is laid along the run it is crocheted into: same length, same middle, within the two
        # layers' thickness of the stitches
        for toy in (self.bunny, self.dog):
            for name, m in nosew.measure(toy).items():
                self.assertLess(abs(m["len_err_pct"]), 5.0, name)
                self.assertLess(m["centre_err"], 1.5, name)
                self.assertLess(m["pos_max"], 3.5, name)

    def test_pairs_symmetric(self):
        for toy in (self.bunny, self.dog):
            for a, b in toy.pairs.values():
                ea = next((e for e in toy.entries if e["name"] == a), {})
                eb = next((e for e in toy.entries if e["name"] == b), {})
                if ea.get("_join") and eb.get("_join"):
                    self.assertEqual(ea["_join"]["round"], eb["_join"]["round"])
                    step = 360.0 / ea["_join"]["n"]
                    self.assertLessEqual(abs(ea["_join"]["az"] + eb["_join"]["az"]), 0.51 * step + 1e-6, a)

    def test_axis_square_to_seam(self):
        # a piece crocheted in along a round hinges about it: its axis is square to the round there
        import numpy as np
        for inst, _, e in nosew.joins(self.bunny):
            T = np.asarray(e["_join"]["T"])
            self.assertLess(abs(float(inst.R[:, 2] @ T)), 0.02, inst.name)

    def test_pattern_text_and_round_trip(self):
        md = render_markdown(pattern_model(self.bunny), "en")
        self.assertIn("Join the left ear in this round (no sewing)", md)
        self.assertIn("<!-- join crochet -->", md)
        ear = md.split("### Ear")[1].split("###")[0]
        self.assertNotIn("for sewing", ear)
        self.assertIn("leave it open", ear)
        self.assertLess(md.index("### Ear"), md.index("### Head"))      # made before the round that holds it
        rt = nosew.roundtrip(self.bunny)
        self.assertTrue(all(rt["joins_same"].values()))
        self.assertLess(rt["max_mm"], 0.5)


if __name__ == "__main__":
    unittest.main()
