"""Ear constructions (spoon / dish), presets, cuteness measure, pale-toy silhouettes.

    python -m unittest amikit.test_cute
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .piece import PieceDef
from .toy import Toy
from . import presets

HERE = Path(__file__).parent
LAB = HERE / "toys" / "_cute_lab"
YARN = {"stitch_mm": 4.0, "round_mm": 3.6, "thickness_mm": 1.9}


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


class EarGeometryTests(unittest.TestCase):
    def test_spoon_has_two_layers_and_a_hollow_front(self):
        sp = PieceDef("ear", {"rounds": ["mr 6", "inc 6 to 24", "even 2"], "end": "open", "spoon": 0.5}, YARN)
        flat = PieceDef("ear", {"rounds": ["mr 6", "inc 6 to 24", "even 2"], "end": "open", "flatten": 0.85}, YARN)
        t = float(sp.N) - 1.0
        W, g, D = sp.spoon_dims(np.array([t]))
        # the two walls stay two sc layers apart (a pressed plate is thinner)
        front = sp.local_points(np.array([t]), np.array([0.0]))[0]
        back = sp.local_points(np.array([t]), np.array([np.pi]))[0]
        self.assertAlmostEqual(front[0] - back[0], float(g[0]), delta=0.05)
        self.assertGreaterEqual(float(g[0]), 2.9 * YARN["thickness_mm"])
        fpl = flat.local_points(np.array([t, t]), np.array([0.0, np.pi]))
        self.assertGreater(front[0] - back[0], fpl[0][0] - fpl[1][0] + 1.0)
        # hollow front: the centre of the front wall lies behind its sides (the rim comes forward)
        ths = np.linspace(-0.35, 0.35, 15) * np.pi * 0.25
        P = sp.local_points(np.full_like(ths, t), ths)
        side = sp.local_points(np.array([t]), np.array([0.2 * np.pi]))[0]
        self.assertLess(front[0], side[0] - 0.5 * float(D[0]) * 0.3)
        self.assertTrue(np.all(np.isfinite(P)))
        # rounded fold at the sides: the widest point is the fold, and it is a curve, not a knife edge
        ring = sp.local_points(np.full(200, t), np.linspace(0, 2 * np.pi, 200, endpoint=False))
        self.assertAlmostEqual(float(np.abs(ring[:, 1]).max()), float(W[0]), delta=0.3)

    def test_dish_rim_is_thick_and_ring_is_the_centre(self):
        d = PieceDef("ear", {"rounds": ["mr 6", "inc 6 to 24", "even 1", "21", "18"], "end": "open", "dish": 0.5},
                     YARN)
        rho0, z0 = d.dish_rz(np.array([0.0]))
        self.assertAlmostEqual(float(rho0[0]), 0.0, delta=0.01)          # the magic ring is the centre front
        self.assertGreater(float(z0[0]), 1.0)                               # sunk in: a hollow front
        ts = np.linspace(0, d.M, 400)
        rho, z = d.dish_rz(ts)
        self.assertAlmostEqual(float(rho.max()), d.dish_R + d.dish_rim, delta=0.2)
        self.assertGreaterEqual(2 * d.dish_rim, 2 * 2.0 * YARN["thickness_mm"])   # rim = two layers + the fold
        self.assertGreater(float(z.max()), 2 * d.dish_rim - 0.3)           # the back layer lies behind the front
        # placed as an edge piece it stands on the rim; no back circle is added (the dish has its own back)
        toy = Toy(_load(LAB / "ears_test.json"))
        parts = {m["name"]: m for m in toy.build(ku=2, kv=2)}
        e = parts["ear_L"]
        X = e["V"] - e["V"].mean(0)
        thick = float(np.ptp(X @ np.linalg.svd(X, full_matrices=False)[2][2]))
        self.assertGreater(thick, 5.5)


class PresetTests(unittest.TestCase):
    def test_no_preset_spec_is_untouched(self):
        spec = _load(HERE / "toys" / "teddy.json")
        self.assertIs(presets.expand(spec), spec)

    def test_expansion(self):
        spec = _load(LAB / "ears_test.json")
        sp = presets.expand(spec)
        parts = {p["id"]: p for p in sp["parts"]}
        self.assertEqual(parts["head"]["rounds"][0], "mr 6")
        self.assertIn("dish", parts["ear"]); self.assertIn("spoon", parts["tuft_ear"])
        self.assertNotIn("preset", parts["ear"])
        ear_e = next(e for e in sp["assembly"] if e["part"] == "ear")
        self.assertEqual(ear_e["attach"], "edge")
        self.assertIn("preset", spec["parts"][1])                         # the input spec is not modified
        from .rounds import expand
        for M in (36, 42, 48, 54, 60, 66):
            expand(presets.head_rounds(M)); expand(presets.head_rounds(M, oval=True))
        for k in ("round", "pointed", "long", "floppy"):
            expand(presets.ear_rounds(k, 18))

    def test_cute_head_is_wider_than_tall(self):
        for M in (48, 60):
            p = PieceDef("head", {"rounds": presets.head_rounds(M)}, YARN)
            w = 2 * p.rb.max(); h = p.zb.max() - p.zb.min()
            self.assertGreater(w / h, 1.15, M)


class PatternTests(unittest.TestCase):
    def test_text_says_how_to_fold_and_reverse_reads_it(self):
        from .pattern_text import write_patterns
        from .reverse import to_spec
        from .roundtrip import compare
        spec = _load(LAB / "ears_test.json")
        toy = Toy(spec)
        with tempfile.TemporaryDirectory() as d:
            write_patterns(toy, d)
            ru = (Path(d) / "PATTERN_RU.md").read_text(encoding="utf8")
            en = (Path(d) / "PATTERN_EN.md").read_text(encoding="utf8")
        self.assertIn("тарелочк", ru); self.assertIn("подверните назад", ru)
        self.assertIn("ложечк", ru); self.assertIn("вдавите переднюю стенку внутрь задней", ru)
        self.assertIn("Stand the dish on its rim", en); self.assertIn("(spoon 35 %)", en)
        self.assertIn("(dish 50 %)", en)
        spec2, _ = to_spec(en)
        parts = {p["id"]: p for p in spec2["parts"]}
        self.assertEqual(parts["ear"].get("dish"), 0.5)
        self.assertTrue(any(p.get("spoon") == 0.35 for p in spec2["parts"]))
        res, _, _ = compare(spec, spec2)
        self.assertTrue(res["counts_ok"])
        self.assertLess(res["max_mm"], 1.0)


class CuteMeasureTests(unittest.TestCase):
    def test_measure_known_toy(self):
        from .cute import measure, score
        f = measure(Toy(_load(HERE / "toys" / "bunny.json")))
        self.assertAlmostEqual(f["head_wh"], 1.385, delta=0.05)
        self.assertAlmostEqual(f["eye_height"], 0.583, delta=0.03)
        self.assertAlmostEqual(f["head_body"], 1.26, delta=0.05)
        self.assertEqual(f["ear_kind"], "long")
        self.assertGreater(score(f)["total"], 60)

    def test_flat_ears_get_a_construction_suggestion_that_builds(self):
        from .cute import measure, suggest
        spec = _load(HERE / "toys" / "teddy.json")
        sug = suggest(spec, measure(Toy(spec)), predict=False)
        ear = [c for c in sug if c["feature"] == "ear_thick"]
        self.assertTrue(ear, [c["why"] for c in sug])
        from . import edit
        new, rep = edit.apply(spec, ear[0]["ops"])
        f2 = measure(Toy(new))
        self.assertGreater(f2["ear_thick"], 5.5)


class PaleMaskTests(unittest.TestCase):
    def test_cream_toy_on_light_backdrop(self):
        from .fitview import toy_mask
        W = 400
        bg = (236, 233, 229)
        im = Image.new("RGB", (W, W), bg)
        ImageDraw.Draw(im).ellipse((150, 330, 330, 360), fill=(205, 202, 199))
        im = im.filter(ImageFilter.GaussianBlur(6))                        # a soft floor shadow, as rendered
        d = ImageDraw.Draw(im)
        d.ellipse((120, 80, 280, 240), fill=(242, 232, 214))               # a cream head
        d.ellipse((140, 200, 260, 345), fill=(240, 230, 212))              # a cream body
        rng = np.random.default_rng(1)
        a = np.asarray(im).astype(float)
        truth = np.zeros((W, W), bool)
        yy, xx = np.mgrid[0:W, 0:W]
        truth |= ((xx - 200) / 80.0) ** 2 + ((yy - 160) / 80.0) ** 2 <= 1
        truth |= ((xx - 200) / 60.0) ** 2 + ((yy - 272.5) / 72.5) ** 2 <= 1
        # yarn texture inside the toy, a little noise everywhere
        a[truth] += rng.normal(0, 9, (truth.sum(), 1))
        a += rng.normal(0, 1.5, a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
        m = toy_mask(im)
        iou = (m & truth).sum() / (m | truth).sum()
        self.assertGreater(iou, 0.93)

    def test_alpha_is_used(self):
        from .fitview import toy_mask
        im = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
        ImageDraw.Draw(im).ellipse((50, 50, 150, 150), fill=(250, 250, 250, 255))   # white on transparent
        m = toy_mask(im)
        self.assertAlmostEqual(m.sum() / (np.pi * 50 * 50), 1.0, delta=0.06)


class OnePieceToyTests(unittest.TestCase):
    def test_suggest_without_a_head_part(self):
        # a one-piece head+body toy (hedgehog, penguin bodies) has no "head" part: no rounds to move a muzzle by;
        # the suggestions must skip that instead of crashing (feat muzzle_height may be measured or None)
        from .cute import measure, suggest, candidates
        spec = {"name": "one_piece", "colours": {"brown": "#8a5a3c", "tan": "#d2a679"},
                "parts": [{"id": "body", "rounds": ["mr 6", "inc 6 to 48", "even 10", "dec 6 to 6"], "colour": "brown"},
                          {"id": "muzzle", "rounds": ["mr 6", "inc 6 to 18", "even 2"], "colour": "tan",
                           "end": "open", "stuff": "light"}],
                "assembly": [{"part": "body", "axis": [0, 1, 0]},
                             {"part": "muzzle", "on": "body", "at": {"round": 14, "az": 0}, "dir": [0, 0, 1]}],
                "face": [{"type": "safety_eye", "on": "body", "at": {"round": 17, "az": 22}, "size_mm": 9,
                          "mirror": True}]}
        toy = Toy(spec)
        feat = measure(toy, toy.build(ku=3, kv=2))
        suggest(spec, feat, predict=False)
        for v in (None, 0.2, 0.95):                       # unmeasured / far too low / far too high
            candidates(spec, {**feat, "muzzle_height": v})


if __name__ == "__main__":
    unittest.main()
