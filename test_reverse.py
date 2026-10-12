"""Reverse generation: written pattern -> spec -> 3D.

    python -m unittest amikit.test_reverse
"""
import json
import tempfile
import unittest
from pathlib import Path

from .reverse import parse_round, to_spec, round_label, build_toy
from .roundtrip import compare
from .toy import Toy
from .pattern_text import write_patterns

HERE = Path(__file__).parent

ROUNDS = [  # (text after the label, previous count, stitches made)
    ("2 sc in each st around", 6, 12), ("sc in each st around", 12, 12),
    ("(2 sc in next st, sc in next st) six times", 12, 18), ("inc in all 6 st", 6, 12),
    ("sc in next 6 st, (sc in next st, dec) repeat 6 times, sc in next 6 st", 30, 24), ("dec 6 times", 12, 6),
    ("(Sc in next 7 sts, sc2tog) rep 6 times", 54, 48), ("Sc in next 6 sts, sc2tog, sc in next 6 sts", 14, 13),
    ("(invdec, sc in next 2 st) six times", 24, 18), ("sc in all 54 st", 54, 54),
    ("Work 2 sc in each st around", 6, 12), ("inc ×6", 6, 12), ("(3 sc, inc) ×6", 24, 30),
    ("1 sc, (inc, 2 sc) ×5, inc, 1 sc", 18, 24), ("start 6 sc in a magic ring (6)", None, 6),
    ("Start 6 sc in a magic ring [6]", None, 6), ("6 sc in magic ring. (6 st)", None, 6),
    ("sc in each st around. (12 st). Stuff body firmly through hole.", 12, 12),
    ("*sc 2, inc* x6 (24)", 18, 24), ("*sc 2, inc; rep from * 5 more times (24)", 18, 24),
    ("(sc, inc) around (18)", 12, 18), ("BLO sc around (24)", 24, 24), ("6 сбн в КА (6)", None, 6),
    ("(1 сбн, пр) ×6 (18)", 12, 18),
]


class ReverseTests(unittest.TestCase):
    def test_round_grammar(self):
        for text, prev, want in ROUNDS:
            r = parse_round(text, prev)
            self.assertEqual(r["n"], want, text)
            self.assertEqual(r["notes"], [], text)

    def test_colour_runs(self):
        r = parse_round("[white] 7 sc, dec, 7 sc [orange] dec, 7 sc, dec, 7 sc, dec, 1 sc [white] 6 sc, dec, 7 sc, dec", 54)
        self.assertEqual(r["runs"], [["white", 15], ["orange", 18], ["white", 15]])
        r = parse_round("2 sc in orange, 15 sc in white, 20 sc in orange, 14 sc in white, 3 sc in orange", 54)
        self.assertEqual(sum(k for _, k in r["runs"]), 54)

    def test_labels(self):
        self.assertEqual(round_label("Rnd 14 – 15: sc in all 66 st (66)")[:2], (14, 15))
        self.assertEqual(round_label("10-20: Sc in each st around")[:2], (10, 20))
        self.assertEqual(round_label("R7–14: 48 sc (48) — 8 rounds")[:2], (7, 14))
        self.assertIsNone(round_label("1. **Body** is the base"))

    def test_round_trip_exact(self):
        """Our written pattern carries everything that shapes the toy: rebuilt within a yarn's thickness."""
        for name in ("fox", "black_cat"):
            spec = json.loads((HERE / "toys" / f"{name}.json").read_text(encoding="utf-8-sig"))
            toy = Toy(spec)
            with tempfile.TemporaryDirectory() as d:
                write_patterns(toy, d)
                text = (Path(d) / "PATTERN_EN.md").read_text(encoding="utf8")
            spec2, _ = to_spec(text)
            res, _, _ = compare(spec, spec2)
            self.assertTrue(res["counts_ok"], name)
            self.assertLess(res["max_mm"], 2.0, f"{name}: {res['max_mm']} mm")

    def test_designer_patterns_build(self):
        for name in ("smugly_bear", "tiny_whale", "lola_cat"):
            src = HERE / "reverse_tests" / name / "source.txt"
            if not src.exists():
                self.skipTest("designer patterns not downloaded")
            spec, log = to_spec(src.read_text(encoding="utf8"))
            toy = build_toy(spec)
            self.assertGreaterEqual(len(toy.inst), 1, name)
            self.assertTrue(any(f["type"] == "safety_eye" for f in spec["face"]), name)


if __name__ == "__main__":
    unittest.main()
