"""Tests of the pet customisation ops (amikit/custom.py), the patch / inner / side colour zones and the colour
zones' alignment with the face.

    ../.venv/Scripts/python -m unittest amikit.test_custom      (from amigurumi-lab/)
"""
import json
import unittest
from pathlib import Path

import numpy as np

from . import edit, custom
from .piece import LEAN
from .toy import Toy

HERE = Path(__file__).resolve().parent
DOG = json.loads((HERE / "toys" / "pets" / "dog.json").read_text(encoding="utf-8"))
CAT = json.loads((HERE / "toys" / "pets" / "cat.json").read_text(encoding="utf-8"))


def stitches(toy, name, colour):
    """World centres of the stitches of instance `name` in `colour` (where the mesh draws them)."""
    inst = toy.inst[name]
    out = []
    for i, row in enumerate(toy.chart(inst), 1):
        n = len(row)
        k = np.array([j for j, c in enumerate(row) if c == colour])
        if len(k):
            th = 2 * np.pi * (k + 0.5 - LEAN * (i - 0.5)) / n
            out.append(inst.base(np.full(len(k), i - 0.5), th))
    return np.concatenate(out) if out else np.zeros((0, 3))


class TestZonesFollowTheFace(unittest.TestCase):
    def test_symmetric_zone_is_centred(self):
        # a blaze written symmetric about az 0 must be centred on the face (x ~ 0) in every round, also high up
        # where the spiral drift of the stitch columns is largest
        s, _ = edit.apply(DOG, [{"op": "marking", "kind": "blaze", "colour": "white", "amount": 0.3}],
                          validate=False)
        toy = Toy(s)
        P = stitches(toy, "head", "white")
        self.assertGreater(len(P), 10)
        self.assertLess(abs(float(P[:, 0].mean())), 1.0)

    def test_side_zone_only_on_one_copy(self):
        s = json.loads(json.dumps(DOG))
        ear = next(p for p in s["parts"] if p["id"] == "ear")
        ear["colours"] = [{"rounds": [1, 4], "colour": "black", "side": "R"}]
        toy = Toy(s)
        self.assertFalse(any("black" in r for r in toy.chart(toy.inst["ear_L"])))
        self.assertTrue(any("black" in r for r in toy.chart(toy.inst["ear_R"])))


class TestInnerEar(unittest.TestCase):
    def test_inner_zone_follows_the_ear_and_keeps_a_rim(self):
        toy = Toy(CAT)
        for name in ("ear_L", "ear_R"):
            ch = toy.chart(toy.inst[name])
            rows = [sum(c == "pink" for c in r) for r in ch]
            self.assertEqual(rows[0], 0)                    # the tip stays in the ear colour
            self.assertGreater(max(rows), 2)
            for r, k in zip(ch, rows):
                self.assertLess(k, len(r) / 2)              # never the whole front half: a rim is left
            self.assertGreaterEqual(rows[-1], rows[2])       # wider toward the base, like the ear

    def test_preset_inner_key(self):
        from .presets import expand
        s = {"parts": [{"id": "ear", "preset": "ear", "kind": "pointed", "inner": "pink"}], "assembly": []}
        p = expand(s)["parts"][0]
        self.assertEqual(p["colours"][0], {"inner": True, "colour": "pink"})
        self.assertNotIn("inner", p)


class TestSpots(unittest.TestCase):
    def test_spot_under_left_eye(self):
        s, rep = edit.apply(DOG, [{"op": "spot", "near": "eye_L", "place": "under", "colour": "#3a2a20"}])
        cname = next(k for k, v in s["colours"].items() if v == "#3a2a20")
        toy = Toy(s)
        pts = np.concatenate([stitches(toy, n, cname) for n in toy.inst])
        self.assertGreater(len(pts), 3)
        eye = next(f for f in s["face"] if f["type"] == "safety_eye")
        E, _ = toy.inst["head"].point(eye["at"]["round"], eye["at"]["az"])     # the toy's left eye (+X)
        c = pts.mean(0)
        self.assertGreater(c[0], 0)                       # on the toy's left
        self.assertLess(c[1], E[1])                       # below the eye
        self.assertLess(np.linalg.norm(c - E), 25)

    def test_spot_is_pruned_to_the_pieces_it_reaches(self):
        s, _ = edit.apply(DOG, [{"op": "spot", "near": "eye_R", "place": "around", "colour": "black"}],
                          validate=False)
        holders = [p["id"] for p in s["parts"] if any("patch" in z for z in p.get("colours", []))]
        self.assertIn("head", holders)
        self.assertNotIn("tail", holders)
        self.assertNotIn("_pending_prune", s)

    def test_spot_off_the_toy_fails(self):
        with self.assertRaises(edit.EditError):
            edit.apply(DOG, [{"op": "spot", "near": "eye_L", "colour": "black", "offset_mm": [400, 0]}],
                       validate=False)

    def test_satin_spot(self):
        s, _ = edit.apply(DOG, [{"op": "marking", "kind": "eyebrows", "colour": "#a8642e"}], validate=False)
        sat = [f for f in s["face"] if f["type"] == "satin" and str(f.get("id", "")).startswith("spot")]
        self.assertEqual(len(sat), 2)                    # one embroidered brow over each eye
        self.assertEqual(sorted(np.sign(f["at"]["az"]) for f in sat), [-1.0, 1.0])

    def test_pattern_lists_the_spot(self):
        s, _ = edit.apply(DOG, [{"op": "spot", "on": "body", "where": "back", "colour": "black",
                                 "size_mm": [24, 20]}])
        toy = Toy(s)
        ch = toy.chart(toy.inst["body"])
        self.assertTrue(any("black" in r for r in ch))


class TestOps(unittest.TestCase):
    def test_colour_all_and_coat(self):
        s, _ = edit.apply(DOG, [{"op": "colour", "target": "all", "colour": "#f2efe9"},
                                {"op": "colour", "target": "ears", "colour": "black"}], validate=False)
        cols = {p["id"]: p.get("colour") for p in s["parts"]}
        self.assertEqual(cols["ear"], "black")
        self.assertEqual(cols["head"], cols["body"])
        self.assertIn(cols["head"], s["colours"])
        self.assertIn(cols["head"], s.get("colour_names_ru", {}).keys() | {"white"})

    def test_hex_colour_reused_and_named(self):
        s, _ = edit.apply(DOG, [{"op": "colour", "target": "tail", "colour": "#c9a06b"}], validate=False)
        # nearly the coat's own fawn: the same yarn, no new colour
        self.assertEqual(next(p for p in s["parts"] if p["id"] == "tail")["colour"], "fawn")

    def test_markings_build(self):
        ops = [{"op": "colour", "target": "all", "colour": "#2b2726"},
               {"op": "marking", "kind": "tan_points", "colour": "#a8642e"},
               {"op": "marking", "kind": "socks", "colour": "#a8642e", "amount": 0.3},
               {"op": "marking", "kind": "chest", "colour": "white"},
               {"op": "marking", "kind": "tail_tip", "colour": "white"},
               {"op": "marking", "kind": "collar", "colour": "red"},
               {"op": "eyes", "colour": "amber"}, {"op": "nose", "colour": "pink"}]
        s, rep = edit.apply(DOG, ops)
        self.assertIsInstance(rep["flags"], list)

    def test_ear_kinds(self):
        for kind in ("pointed", "round", "long", "folded"):
            s, _ = edit.apply(DOG, [{"op": "ears", "kind": kind, "inner": "pink"}])
            self.assertEqual(custom.ear_kind(s, "ear"), kind)
            ear = next(p for p in s["parts"] if p["id"] == "ear")
            self.assertTrue(any(z.get("inner") for z in ear.get("colours", [])))
        s, _ = edit.apply(DOG, [{"op": "ears", "kind": "floppy", "size": 1.0}], validate=False)
        self.assertEqual(next(p for p in s["parts"] if p["id"] == "ear")["rounds"],
                         next(p for p in DOG["parts"] if p["id"] == "ear")["rounds"])   # already floppy: kept

    def test_tail_and_shape(self):
        from .rounds import counts
        s, _ = edit.apply(DOG, [{"op": "tail", "kind": "curled"}, {"op": "shape", "target": "legs", "length": 0.7}])
        leg0 = len(counts(next(p for p in DOG["parts"] if p["id"] == "front_leg")["rounds"]))
        leg1 = len(counts(next(p for p in s["parts"] if p["id"] == "front_leg")["rounds"]))
        self.assertLess(leg1, leg0)
        self.assertGreater(abs(next(p for p in s["parts"] if p["id"] == "tail")["bend"]["deg"]), 150)

    def test_width(self):
        from .rounds import counts
        s, _ = edit.apply(DOG, [{"op": "shape", "target": "tail", "width": 1.5}])
        a = counts(next(p for p in DOG["parts"] if p["id"] == "tail")["rounds"])
        b = counts(next(p for p in s["parts"] if p["id"] == "tail")["rounds"])
        self.assertEqual(len(a), len(b))
        self.assertGreater(max(b), max(a))

    def test_unknown_things_fail_cleanly(self):
        for op in ({"op": "marking", "kind": "zebra", "colour": "black"},
                   {"op": "colour", "target": "wings", "colour": "black"},
                   {"op": "colour", "target": "ears", "colour": "blurple"},
                   {"op": "ears", "kind": "antlers"}):
            with self.assertRaises(edit.EditError):
                edit.apply(DOG, [op], validate=False)


class TestPetLLM(unittest.TestCase):
    def test_messages_have_the_photo_and_the_ops_reference(self):
        from . import pet_llm
        photo = HERE.parent.parent / "Examples" / "_pets" / "photos" / "corgi.png"
        if not photo.exists():
            self.skipTest("no example photo")
        system, msgs = pet_llm.build_messages(DOG, photo, "пятно под левым глазом")
        self.assertIn('"op": "spot"', system[0]["text"])
        self.assertEqual(msgs[0]["content"][0]["type"], "image")
        self.assertIn("owner_notes", msgs[0]["content"][1]["text"])

    def test_fake_model_round_trip(self):
        from . import pet_llm

        class Block:
            type = "text"
            def __init__(self, t): self.text = t

        class Resp:
            stop_reason = "end_turn"
            def __init__(self, t): self.content = [Block(t)]

        answers = [json.dumps({"seen": "x", "explanation": "y", "ops_json": json.dumps(
                       [{"op": "ears", "kind": "antlers"}])}),
                   json.dumps({"seen": "corgi", "explanation": "ok", "ops_json": json.dumps(
                       [{"op": "colour", "target": "all", "colour": "#d2782f"},
                        {"op": "ears", "kind": "pointed", "inner": "#f1d9c6"}])})]

        class Msgs:
            def create(self, **kw):
                return Resp(answers.pop(0))

        class Beta:
            messages = Msgs()

        class Client:
            beta = Beta()

        photo = HERE.parent.parent / "Examples" / "_pets" / "photos" / "corgi.png"
        if not photo.exists():
            self.skipTest("no example photo")
        try:
            import anthropic  # noqa: F401
        except ImportError:
            self.skipTest("anthropic not installed")
        res = pet_llm.photo_to_ops(DOG, photo, client=Client())
        self.assertEqual(res["attempts"], 2)
        self.assertEqual(custom.ear_kind(res["spec"], "ear"), "pointed")


if __name__ == "__main__":
    unittest.main()
