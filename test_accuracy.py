"""Tests of the written pattern's accuracy for a crocheter (2026-10-08) and the placement fixes behind it.

    ../.venv/Scripts/python -m unittest amikit.test_accuracy      (from amigurumi-lab/)
"""
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .pattern_text import _eye_size_text, pos_text, ru_decline, write_patterns
from .reverse import parse_pos
from .toy import Toy

HERE = Path(__file__).resolve().parent


def _spec(parts, assembly, face=()):
    return {"name": "t", "title": "T", "colours": {"main": "#c08050", "pink": "#f0a0a0"},
            "parts": parts, "assembly": assembly, "face": list(face)}


BALL = {"id": "body", "rounds": ["mr 6", "inc 6 to 36", "even 6", "dec 6 to 6"], "colour": "main"}
HEAD = {"id": "head", "rounds": ["mr 6", "inc 6 to 42", "even 7", "dec 6 to 6"], "colour": "main"}


def _patterns(spec):
    with tempfile.TemporaryDirectory() as d:
        write_patterns(Toy(spec), d)
        return ((Path(d) / "PATTERN_RU.md").read_text(encoding="utf8"),
                (Path(d) / "PATTERN_EN.md").read_text(encoding="utf8"))


def visible(md):
    return re.sub(r"<!--.*?-->", "", md, flags=re.S)


class EyesBeforeTheirRound(unittest.TestCase):
    def test_eyes_go_in_after_the_round_that_holds_them(self):
        # eyes low on a head (round 13 of 15): the stuffing note used to come at ~60 % of the width, before round 13
        spec = _spec([BALL, HEAD], [{"part": "body", "axis": [0, 1, 0]},
                                    {"part": "head", "on": "body", "at": {"round": "end"}, "attach": "end",
                                     "dir": [0, 1, 0], "sink": 4}],
                     [{"type": "safety_eye", "on": "head", "at": {"round": 13.5, "az": 25}, "size_mm": 9,
                       "mirror": True}])
        ru, en = _patterns(spec)
        head = en.split("### Head")[1].split("###")[0]
        before = head.split("Insert the safety eyes")[0]
        last = max(int(x) for x in re.findall(r"^R(\d+)", before, flags=re.M))
        self.assertGreaterEqual(last, 14)


class HiddenExactValues(unittest.TestCase):
    def setUp(self):
        spec = json.loads((HERE / "toys" / "teddy.json").read_text(encoding="utf-8-sig"))
        self.ru, self.en = _patterns(spec)

    def test_no_angles_or_engine_text_for_the_crocheter(self):
        v = visible(self.ru)
        self.assertNotIn("Точно:", v)
        self.assertNotIn("## Проверка", v)
        self.assertNotRegex(v, r"\((ложечка|тарелочка) \d+|сплющен[аоы]? на \d+ ?%")
        self.assertNotRegex(v, r"в \d+,[0-46-9] п\.")     # whole or half stitches only (no "в 0,7 п.")
        self.assertIn("Точно:", self.ru)                  # still there for the exact rebuild

    def test_positions_whole_or_half_stitches_with_the_exact_tenth_hidden(self):
        pos = {"round": 12, "stitches": 6.9, "from": "front", "side": "left"}
        t = pos_text(pos, "en")
        self.assertIn("7 st from the centre front line", t)
        self.assertIn("<!-- st 6.9 left -->", t)
        p = parse_pos(t)
        self.assertAlmostEqual(p["k"], 6.9)
        self.assertEqual(p["side"], "left")
        self.assertNotIn("<!--", pos_text({"round": 12, "stitches": 6.5, "from": "front", "side": "left"}, "en"))


class GroupedCopies(unittest.TestCase):
    def test_many_copies_are_one_step_and_read_back(self):
        bob = {"id": "bobble", "rounds": ["mr 6", "12"], "colour": "pink", "end": "open", "stuff": "none"}
        asm = [{"part": "body", "axis": [0, 1, 0]}]
        for k, (r, az) in enumerate([(4, 20), (5, 60), (6, 100), (7, 140), (8, 180), (9, 220), (10, 260)]):
            asm.append({"part": "bobble", "name": f"bob{k}", "on": "body", "at": {"round": r, "az": az},
                        "applique": True})
        spec = _spec([BALL, bob], asm)
        ru, en = _patterns(spec)
        steps = [ln for ln in visible(ru).splitlines() if re.match(r"^\d+\. ", ln)]
        self.assertLessEqual(len(steps), 3)
        self.assertIn("— 7 шт.", ru)
        # the exact steps are kept for the reverse reader
        self.assertEqual(en.count("<!-- step: "), 7)
        from .reverse import parse_our_assembly
        lines = en.split("## Assembly")[1].splitlines()
        steps_back = parse_our_assembly(lines, {"bobble": "bobble", "body": "body"}, {})
        self.assertEqual(sum(1 for s in steps_back if s.get("part") == "bobble"), 7)


class Wording(unittest.TestCase):
    def test_eye_sizes_a_shop_sells(self):
        self.assertEqual(_eye_size_text(12, "ru"), "12")
        self.assertTrue(_eye_size_text(12.5, "ru").startswith("12–13"))

    def test_adjectives_after_the_noun_agree(self):
        self.assertEqual(ru_decline("сегмент жёлтый", "gen"), "сегмента жёлтого")
        self.assertEqual(ru_decline("крыло большое", "gen"), "крыла большого")
        self.assertEqual(ru_decline("голова кота", "gen"), "головы кота")

    def test_not_stuffed_piece_has_no_top_up(self):
        ball = dict(BALL, stuff="none")
        ru, _ = _patterns(_spec([ball], [{"part": "body", "axis": [0, 1, 0]}]))
        self.assertNotIn("Добейте набивку", ru)


class Placement(unittest.TestCase):
    def test_round_past_the_parent_is_flagged_and_clamped(self):
        tail = {"id": "tail", "rounds": ["mr 6", "even 4"], "colour": "main", "end": "open"}
        spec = _spec([BALL, tail], [{"part": "body", "axis": [0, 1, 0]},
                                    {"part": "tail", "on": "body", "at": {"round": 30, "az": 180}}])
        toy = Toy(spec)
        self.assertTrue(any("tail" in f and "has rounds" in f for f in toy.flags))

    def test_sink_along_the_axis(self):
        # a head sewn on the side of a body and pointing up: sunk along the surface normal it moves sideways into
        # the body; sunk along its own axis it moves down
        base = _spec([BALL, HEAD], [{"part": "body", "axis": [0, 1, 0]},
                                    {"part": "head", "on": "body", "at": {"round": 7, "az": 90},
                                     "attach": "end", "dir": [0, 1, 0], "sink": 10}])
        a = Toy(base)
        b_spec = copy.deepcopy(base)
        b_spec["assembly"][1]["sink_dir"] = "axis"
        b = Toy(b_spec)
        da = a.inst["head"].origin
        db = b.inst["head"].origin
        self.assertGreater(db[0] - da[0], 5)            # not pushed sideways into the body (-X)
        self.assertLess(db[1], da[1] - 5)               # pushed down instead
        ru, en = _patterns(b_spec)
        self.assertIn("pressed toward", en)
        from .reverse import parse_exact
        ex = parse_exact(re.search(r"Exact placement: [^>]*", en.split("**Head")[1]).group(0))
        self.assertIn("sink_dir", ex)
        self.assertGreater(-ex["sink_dir"][1], 0.9)


class NewShapes(unittest.TestCase):
    def test_popcorn_round(self):
        from .rounds import expand, check
        r = expand(["mr 6", "inc 6 to 24", "popcorns every 4", "popcorns every 4 from 2"])
        self.assertEqual(r[-2].loop, "popcorn")
        self.assertEqual(r[-2].n, 24)
        self.assertEqual(r[-2].ops.count("pop"), 6)
        self.assertEqual(check(r), [])
        spec = _spec([dict(BALL, rounds=["mr 6", "inc 6 to 36", "popcorns every 3", "even 2", "dec 6 to 6"])],
                     [{"part": "body", "axis": [0, 1, 0]}])
        ru, en = _patterns(spec)
        self.assertIn("(2 сбн, попкорн) ×12 (36)", ru)
        self.assertIn("| попкорн |", ru)
        from .reverse import _special_round
        step = _special_round("(popcorns): (2 sc, popcorn) x12 (36)", 36)
        self.assertEqual(step[0], "popcorns every 3")
        # the bumps are geometry: the popcorn round reaches further out than a plain one
        toy = Toy(spec)
        m = next(x for x in toy.build(ku=4, kv=3) if x["name"] == "body")
        plain = Toy(_spec([dict(BALL, rounds=["mr 6", "inc 6 to 36", "even 3", "dec 6 to 6"])],
                          [{"part": "body", "axis": [0, 1, 0]}]))
        m2 = next(x for x in plain.build(ku=4, kv=3) if x["name"] == "body")
        r1 = np.hypot(m["V"][:, 0], m["V"][:, 2]).max()
        r2 = np.hypot(m2["V"][:, 0], m2["V"][:, 2]).max()
        self.assertGreater(r1, r2 + 1.5)

    def test_coil_bend_turns_past_a_circle_without_crossing(self):
        tail = {"id": "tail", "rounds": ["mr 6", "inc 3 to 18 every 3", "even 6"], "colour": "main",
                "end": "open", "stuff": "firm", "bend": {"deg": 420, "at": 20, "over": 2, "coil": 3}}
        spec = _spec([BALL, tail], [{"part": "body", "axis": [0, 1, 0]},
                                    {"part": "tail", "on": "body", "at": {"round": 4, "az": 90},
                                     "dir": [0, -1, 0]}])
        toy = Toy(spec)
        inst = toy.inst["tail"]
        t = np.linspace(0.2, inst.p.M - 0.2, 80)
        axis = np.array([inst.base(np.full(24, ti), np.linspace(0, 2 * np.pi, 24, endpoint=False)).mean(0)
                         for ti in t])
        # the centre line winds inward: the tip ends up near the middle of the spiral, not back at the seam
        span = np.linalg.norm(axis[-1] - axis[0])
        self.assertLess(np.linalg.norm(axis[0] - axis[len(axis) // 3]), span * 3)
        ru, en = _patterns(spec)
        self.assertIn("спиралью", ru)
        self.assertIn("<!-- coil 3", en)


if __name__ == "__main__":
    unittest.main()
