"""Tests for the written pattern: run from amigurumi-lab/ with `python -m unittest amikit.test_pattern_text`."""
from __future__ import annotations
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from .toy import Toy
from .pattern_text import (write_patterns, parse_count, colour_segments, round_body, group_ops, ru_decline,
                           part_name, qualifier, colour_word, yarn_ru, change_ru, _usage_groups,
                           COLOUR_OVERRIDE, PART_OVERRIDE)

HERE = Path(__file__).parent

# a small toy exercising colour zones (symmetric and asymmetric), a chain-oval start, BLO/FLO rounds, a closed
# piece with a bend, squash, flatten, an applique and the face item types
TEST_SPEC = {
    "name": "testy", "title": "Test duck",
    "yarn": {"weight": "DK", "hook_mm": 2.5, "stitch_mm": 4.0, "round_mm": 3.6, "thickness_mm": 1.9},
    "colours": {"yellow": "#f2c94c", "white": "#f4f1ea", "orange": "#e8892b", "pink": "#f2a0a8",
                "dark_brown": "#4a2a18"},
    "parts": [
        {"id": "body", "rounds": ["mr 6", "inc 6 to 30", "blo 30", "even 3", "flo", "dec 6 to 18"],
         "colour": "yellow", "end": "open", "stuff": "firm",
         "colours": [{"rounds": [4, 6], "az": [-40, 40], "colour": "white"}, {"rounds": [8, 8], "colour": "orange"}]},
        {"id": "head", "rounds": ["mr 6", "inc 6 to 36", "even 4", "dec 6 to 6"], "colour": "yellow",
         "colours": [{"rounds": [6, 8], "az": [10, 70], "colour": "white"}]},
        {"id": "bill", "rounds": ["ch 5", "inc 4 x1", "even 2"], "colour": "orange", "end": "open",
         "stuff": "none", "squash": 0.8},
        {"id": "wing", "rounds": ["mr 6", "inc 6 to 12", "even 3"], "colour": "yellow", "end": "open",
         "stuff": "none", "flatten": 0.8, "colours": [{"rounds": [3, 5], "az": [20, 120], "colour": "white"}]},
        {"id": "spot", "rounds": ["mr 6", "inc 6 to 12"], "colour": "white", "end": "open", "stuff": "none"},
        {"id": "foot", "rounds": ["mr 6", "inc 6 to 12", "even 4", "dec 6 to 6"], "colour": "orange",
         "stuff": "light", "bend": {"deg": 60, "at": 3, "over": 2}},
    ],
    "assembly": [
        {"part": "body", "axis": [0, 1, 0]},
        {"part": "head", "on": "body", "at": {"round": "end"}, "attach": "end", "dir": [0, 1, 0], "sink": 3},
        {"part": "bill", "on": "head", "at": {"round": 9, "az": 0}, "dir": [0, 0, 1]},
        {"part": "wing", "on": "body", "at": {"round": 9, "az": 85}, "dir": [0.3, -1, 0], "mirror": True},
        {"part": "spot", "on": "body", "at": {"round": 6, "az": 0}, "applique": True},
        {"part": "foot", "on": "body", "at": {"round": 3, "az": 25}, "dir": [0.1, -0.3, 1], "mirror": True},
    ],
    "face": [
        {"type": "safety_eye", "on": "head", "at": {"round": 7.5, "az": 30}, "size_mm": 6, "mirror": True},
        {"type": "blush", "on": "head", "at": {"round": 9, "az": 45}, "size_mm": 7, "colour": "pink", "mirror": True},
        {"type": "satin", "on": "head", "at": {"round": 11, "az": 0}, "width_st": 3, "height_rounds": 2, "rotate": 20},
        {"type": "felt", "on": "head", "at": {"round": 6, "az": -20}, "size_mm": [8, 5], "colour": "black"},
        {"type": "stitch", "on": "head", "points": [[10, -15], [10.5, 0], [10, 15]], "colour": "dark_brown"},
    ],
}

LINE = {
    "ru": re.compile(r"^Р(\d+)(?:–(\d+))?(?: \((ЗПС|ППС)\))?: (.*) \((\d+)\)(?: — .*)?$"),
    "en": re.compile(r"^R(\d+)(?:–(\d+))?(?: \((BLO|FLO)\))?: (.*) \((\d+)\)(?: — .*)?$"),
}
CHANGE = re.compile(r"^(смените нить на \S+|change to [^:]+): ")


def _sections(md):
    """{part heading -> [lines]} for the pieces section."""
    body = md.split("\n## ", 3)
    pieces = md[md.index("## Детали") if "## Детали" in md else md.index("## Pieces"):]
    pieces = pieces[:pieces.index("\n## ", 3)]
    out = {}
    cur = None
    for ln in pieces.splitlines():
        if ln.startswith("### "):
            cur = ln[4:]; out[cur] = []
        elif cur:
            out[cur].append(ln)
    return out


# Toy.chart mirrors colour zones on the right copy of a pair (fixed 6 Oct); kept as a name for the tests below.
MirrorZoneToy = Toy


class PatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results = {}
        teddy = json.loads((HERE / "toys" / "teddy.json").read_text(encoding="utf8"))
        for name, spec, cls_ in (("teddy", teddy, Toy), ("testy", TEST_SPEC, MirrorZoneToy)):
            toy = cls_(spec)
            out = Path(cls.tmp.name) / name
            model = write_patterns(toy, out)
            cls.results[name] = (toy, model, out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_files_both_languages(self):
        for name, (toy, model, out) in self.results.items():
            for f in ("PATTERN_RU.md", "PATTERN_EN.md", "pattern.json"):
                self.assertTrue((out / f).exists(), f"{name}: {f}")
            ru = (out / "PATTERN_RU.md").read_text(encoding="utf8")
            en = (out / "PATTERN_EN.md").read_text(encoding="utf8")
            # the engine's check is a comment (hidden from the crocheter); an error would show as "## Проверка"
            self.assertIn("<!-- engine check:", ru); self.assertIn("<!-- engine check:", en)
            self.assertNotIn("## Проверка", ru)
            self.assertIn("## Сборка", ru); self.assertIn("## Assembly", en)
            json.loads((out / "pattern.json").read_text(encoding="utf8"))

    def test_round_lines_counts_and_ranges(self):
        for name, (toy, model, out) in self.results.items():
            for lang in ("ru", "en"):
                md = (out / f"PATTERN_{lang.upper()}.md").read_text(encoding="utf8")
                secs = _sections(md)
                self.assertEqual(len(secs), len(model["parts"]), f"{name} {lang}")
                for (head, lines), part in zip(secs.items(), model["parts"]):
                    counts = [r["n"] for r in part["rounds"]]
                    expect_next = 1
                    prev = None
                    variant = False
                    for ln in lines:
                        if ln.startswith(("Для правой", "Для левой", "Для детали", "For the right", "For the left",
                                          "For the piece")):
                            variant = True
                        m = LINE[lang].match(ln)
                        if not m:
                            continue
                        a = int(m.group(1)); b = int(m.group(2) or a)
                        stated = int(m.group(5))
                        if variant:
                            for r in range(a, b + 1):
                                self.assertEqual(stated, counts[r - 1], ln)
                            c, p = parse_count(m.group(4))
                            self.assertEqual((c, p), (counts[a - 2], counts[a - 1]), ln)
                            continue
                        self.assertEqual(a, expect_next, f"{name} {lang} {head}: ranges not contiguous at {ln!r}")
                        expect_next = b + 1
                        for r in range(a, b + 1):
                            self.assertEqual(stated, counts[r - 1], f"{name} {lang} {head}: {ln}")
                        if a == 1:
                            prev = counts[b - 1]
                            continue
                        text = CHANGE.sub("", m.group(4))
                        c, p = parse_count(text)
                        self.assertEqual(p, stated, f"{name} {lang}: {ln}")
                        self.assertEqual(c, prev, f"{name} {lang}: works {c} st but previous round has {prev}: {ln}")
                        prev = stated
                    self.assertEqual(expect_next, len(counts) + 1, f"{name} {lang} {head}: rounds missing")

    def test_colour_segments_match_chart(self):
        for name, (toy, model, out) in self.results.items():
            for part in model["parts"]:
                first = [i for i in toy.inst.values() if i.p.id == part["id"]][0]
                chart = toy.chart(first)
                for r, row in zip(part["rounds"], chart):
                    self.assertEqual(sum(c for _, c in r["colour_runs"]), r["n"])
                    got = Counter()
                    for c, k in r["colour_runs"]:
                        got[c] += k
                    self.assertEqual(got, Counter(row), f"{name} {part['id']} R{r['round']}")
                self.assertEqual(part["check"], [])
                self.assertEqual(part["self_check"], [])

    def test_feature_spec_text(self):
        toy, model, out = self.results["testy"]
        ru = (out / "PATTERN_RU.md").read_text(encoding="utf8")
        en = (out / "PATTERN_EN.md").read_text(encoding="utf8")
        self.assertIn("цепочку из 5 цеп. п.", ru)
        self.assertIn("ch 5.", en)
        # a front zone (az -40..40) on a BLO round; the white sits where the stitches end up after the spiral
        # drift of the columns (half a stitch by round 6), so it is split 3 + 4 around the round start
        self.assertRegex(ru, r"Р6 \(ЗПС\): 3 сбн белым, 23 сбн жёлтым, 4 сбн белым \(30\)")
        self.assertRegex(en, r"R6 \(BLO\): 3 sc in white, 23 sc in yellow, 4 sc in white \(30\)")
        self.assertRegex(ru, r"Р10 \(ППС\): 30 сбн \(30\)")
        self.assertRegex(ru, r"\d+ сбн белым, \d+ сбн жёлтым")
        self.assertIn("Смените нить на оранжевую", ru)
        self.assertIn("Change to orange", en)
        self.assertIn("[", ru)                                    # a round mixing colour and increases/decreases
        self.assertIn("Для правой детали", ru)                    # asymmetric zone on a mirrored pair
        self.assertIn("щель", ru)                                 # squash
        self.assertIn("Изгиб", ru)                                # bend
        self.assertIn("Не набивайте. Сложите деталь плоско", ru)  # flatten
        self.assertIn("плоско, по всему краю", ru)                # applique
        self.assertIn("2 шт.", ru)
        for w in ("Глазки", "Румянец", "гладью", "Фетр", "Вышивка"):
            self.assertIn(w, ru)
        self.assertIn("повернув фигуру на 20°", ru)               # satin rotate
        self.assertIn("turned 20°", en)
        self.assertIn("— сделайте 2 детали (левая и правая", ru)  # piece count wording
        self.assertRegex(ru, r"креплении 6 мм \(чёрный `#[0-9a-f]{6}`\) — 2 шт\.")   # eye colour with its hex
        # the chain start round: 2c stitches
        self.assertEqual(model["parts"][[p["id"] for p in model["parts"]].index("bill")]["rounds"][0]["n"], 10)

    def test_helpers(self):
        self.assertEqual(round_body(["sc", "inc"] * 6, ["a"] * 18, "ru"), "(1 сбн, пр) ×6")
        self.assertEqual(round_body(["sc", "inc", "sc"] * 6, ["a"] * 24, "ru"), "1 сбн, (пр, 2 сбн) ×5, пр, 1 сбн")
        self.assertEqual(round_body(["sc"] * 48, ["a"] * 48, "en"), "48 sc")
        self.assertEqual(round_body(["inc"] * 6, ["a"] * 12, "en"), "inc ×6")
        # an increase split across a colour boundary
        segs = colour_segments(["sc", "inc", "sc"], ["a", "a", "b", "b"])
        self.assertEqual(segs, [["a", ["sc", "sc"]], ["b", ["same", "sc"]]])
        txt = round_body(["sc", "inc", "sc"], ["a", "a", "b", "b"], "ru")
        self.assertEqual(parse_count(txt), (3, 4))
        for ops in (["sc", "sc", "inc", "sc"] * 5, ["dec", "sc", "sc"] * 4, ["inc3", "inc"] * 3):
            n = sum({"sc": 1, "inc": 2, "inc3": 3, "dec": 1}[o] for o in ops)
            c = sum({"sc": 1, "inc": 1, "inc3": 1, "dec": 2}[o] for o in ops)
            for lang in ("ru", "en"):
                self.assertEqual(parse_count(round_body(ops, ["a"] * n, lang)), (c, n))
            flat = [t for u, k in group_ops(ops) for t in u * k]
            self.assertEqual(flat, ops)


REAL_TOYS = ("crow", "capybara", "red_panda", "sage_bunny", "highland_cow", "potato", "teddy", "mouse", "shark")


class RealToyPatternTests(unittest.TestCase):
    """The written pattern of real specs read like a Russian crocheter would (one check per reported problem)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ru, cls.en, cls.model = {}, {}, {}
        for name in REAL_TOYS:
            spec = json.loads((HERE / "toys" / f"{name}.json").read_text(encoding="utf-8-sig"))
            out = Path(cls.tmp.name) / name
            cls.model[name] = write_patterns(Toy(spec), out)
            cls.ru[name] = (out / "PATTERN_RU.md").read_text(encoding="utf8")
            cls.en[name] = (out / "PATTERN_EN.md").read_text(encoding="utf8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def section(self, name, head):
        return _sections(self.ru[name])[next(h for h in _sections(self.ru[name]) if h.startswith(head))]

    def test_1_piece_counts(self):
        ru = self.ru["crow"]
        self.assertIn("### Пёрышко — сделайте 6 деталей", ru)
        self.assertIn("### Пальчик — сделайте 6 деталей", ru)
        self.assertIn("Пальчик (№ 2) → голова-тело** (2 шт.)", ru)
        self.assertIn("### Подушечка лапки — сделайте 6 деталей (по 3 на каждую ножку)", self.ru["red_panda"])
        self.assertIn("Из них: передняя — 2 шт. (левая и правая); задняя — 2 шт. (левая и правая).",
                      self.ru["capybara"])
        self.assertIn("Лопасть хвоста (верхняя) → тело", self.ru["shark"])
        self.assertIn("Боковой плавник (правый) → тело", self.ru["shark"])
        self.assertNotRegex(ru, r"— \d+ детали\b")                          # old wording "— 6 детали"
        for name in REAL_TOYS:                                              # every instance counted once
            for p in self.model[name]["parts"]:
                self.assertEqual(p["make"], len(p["instances"]), f"{name} {p['id']}")
                self.assertEqual(p["make"], sum(2 if u["mirror"] else 1 for u in p["uses"]), f"{name} {p['id']}")

    def test_2_chain_ovals_are_ovals(self):
        inner = "\n".join(self.section("sage_bunny", "Внутренняя часть ушка"))
        self.assertIn("Плоский овал", inner)
        self.assertNotIn("кружок", inner)
        self.assertRegex("\n".join(self.section("capybara", "Тело")), r"овал ≈ [\d,]+ × [\d,]+ см")

    def test_3_mirrored_items_say_count_and_side(self):
        self.assertIn("Внутренняя часть ушка (левая) → левое ушко", self.ru["sage_bunny"])
        self.assertIn("Внутренняя часть ушка (правая) → правое ушко", self.ru["sage_bunny"])
        self.assertIn("(2 шт. — по одной на каждое ушко)", self.ru["teddy"])
        self.assertIn("2 овала 14×12 мм", self.ru["crow"])
        self.assertIn("Фетр: белый (2 овала 14×12 мм)", self.ru["crow"])
        self.assertRegex(self.ru["capybara"], r"\*\*Вышивка\*\* .*\(2 шт\., симметрично")

    def test_4_curls(self):
        ru = self.ru["highland_cow"]
        self.assertIn("**Как завить пряди**", ru)
        self.assertIn("спицу", ru)
        self.assertIn("петельчатыми столбиками", ru)
        self.assertEqual(ru.count("**Как завить пряди**"), 1)

    def test_5_flat_seams(self):
        for name in ("teddy", "red_panda", "shark", "highland_cow"):
            self.assertRegex(self.ru[name], r"Сплющите отверстие \(\d+ п\. на \d+ п\.\) и пришейте прямой линией "
                                            r"(спереди назад|из стороны в сторону|сверху вниз)")
        # shark tail lobes have seam_line [0, 1, 0]: a vertical seam
        self.assertRegex(self.ru["shark"], r"Лопасть хвоста \(верхняя\).*прямой линией сверху вниз")
        self.assertNotIn("Пришейте сплющенный край", self.ru["teddy"])

    def test_6_strip_direction_and_ends(self):
        ru = self.ru["potato"]
        self.assertIn("Положите её по изгибу тела поперёк, из стороны в сторону", ru)
        # (an exact-position comment may follow each end: hidden from the reader)
        self.assertRegex(ru, r"концы — в \d+-м ряду.* к правому боку игрушки(?: <!--[^>]*-->)? и в \d+-м ряду.* "
                             r"к левому боку игрушки")
        self.assertIn("Ряд 2: 20 сбн (20).", ru)                           # no 'Ряды 2–2'

    def test_7_front_axis_oval(self):
        body = "\n".join(self.section("capybara", "Тело"))
        self.assertIn("лежит спереди назад", body)
        self.assertIn("начальная цепочка — внизу", body)
        self.assertIn("Маркер начала ряда — на закруглении", body)
        self.assertIn("Маркер начала ряда — посередине длинной стороны", self.ru["teddy"])   # side-axis muzzle

    def test_8_edge_discs(self):
        wing = "\n".join(self.section("crow", "Крылышко"))
        self.assertIn("свяжите 4 кружка — 2 по схеме ниже и 2 однотонных", wing)
        self.assertIn("обвяжите по краю, провязывая обе стенки сразу: 24 сбн", wing)
        self.assertIn("пришейте только нижним краем", self.ru["crow"])
        self.assertIn("кружка — 2 по схеме ниже (с цветами)", self.ru["mouse"])

    def test_9_bends_are_instructions(self):
        for name in ("red_panda", "highland_cow"):
            ru = self.ru[name]
            self.assertNotIn("Счёт петель изгиба не даёт", ru)
            self.assertRegex(ru, r"внутренняя сторона сгиба на готовой игрушке смотрела (вверх|вниз|вперёд|назад|к )")
            self.assertRegex(ru, r"(стяните вместе \d+-й и \d+-й ряды|от \d+-го до \d+-го ряда)")

    def test_10_no_latin_or_raw_ids(self):
        ok = {"DK", "OK", "max", "«Лицо»", "«Левый/правый»", "«[белым] …»", "«бычий узел»",
              "«В k п. от центральной линии»"}
        for name, ru in self.ru.items():
            t = re.sub(r"<!--.*?-->", "", re.sub(r"`#[0-9a-fA-F]{6}`", "", ru), flags=re.S)
            bad = {m for m in re.findall(r"«[^»]*»|[A-Za-z]{2,}", t)} - ok
            self.assertEqual(bad, set(), name)

    def test_11_materials_list_every_colour(self):
        for name, m in self.model.items():
            listed = {c["name"] for c in m["colours"]}
            used = {c for p in m["parts"] for r in p["rounds"] for c, _ in r["colour_runs"]}
            used |= {s["colour"] for s in m["assembly"] if s["kind"] in ("flat", "cord")}
            used |= {f["colour"] for f in m["face"] if f["type"] == "strap"}
            used |= {c for f in m["face"] if f["type"] == "fringe" for c in (f.get("colours") or [f["colour"]])}
            self.assertLessEqual(used, listed, name)
            for c in m["colours"]:            # metres cover the stitches (2.4 cm/sc) and the extra yarn
                self.assertGreaterEqual(c["metres"], c["stitches"] * m["yarn_per_stitch_m"] + c.get("extra_m", 0),
                                        f"{name} {c['name']}")
        self.assertIn("светло-зелёный (лямки)", self.ru["capybara"].split("## Плотность")[0])
        self.assertIn("**Лямка**", self.ru["capybara"])
        self.assertRegex(self.ru["capybara"], r"провяжите цепочку из \d+ цеп\. п\. \(≈ [\d,]+ см\)")

    def test_helpers_ru(self):
        self.assertEqual(ru_decline("голова-тело", "gen"), "головы-тела")
        self.assertEqual(ru_decline("спинной плавник", "dat"), "спинному плавнику")
        self.assertEqual(ru_decline("внутренняя часть ушка", "acc"), "внутреннюю часть ушка")
        self.assertEqual(part_name("hind_foot", "ru"), "задняя лапка")
        self.assertEqual(part_name("t_leg", "ru", "gen"), "ножки")
        self.assertEqual(qualifier("front_leg_R", "leg", "ru"), "передняя, правая")
        self.assertEqual(qualifier("toe_b", "toe", "ru"), "№ 2")
        self.assertEqual(colour_word("ginger", "ru", "acc_f"), "рыжую")
        self.assertEqual(colour_word("navy", "ru", "acc_f"), "тёмно-синюю")


class RussianDeclensionTests(unittest.TestCase):
    """Colour names and part names in the cases the pattern needs (nouns, adjective + noun, invariant words)."""

    COLOURS = {"ochre_dark": "тёмная охра", "ochre": "охра", "cocoa": "какао", "cocoa_dark": "тёмное какао",
               "sage_light": "светлый шалфей", "latte": "кофе с молоком", "earbrown": "коричневый (ушки)",
               "honey": "медово-бежевый", "cherry": "вишня", "lilac_f": "сиреневая", "emerald": "изумруд"}

    def setUp(self):
        self._col = dict(COLOUR_OVERRIDE); self._part = dict(PART_OVERRIDE)
        COLOUR_OVERRIDE.clear(); COLOUR_OVERRIDE.update(self.COLOURS)
        PART_OVERRIDE.clear()

    def tearDown(self):
        COLOUR_OVERRIDE.clear(); COLOUR_OVERRIDE.update(self._col)
        PART_OVERRIDE.clear(); PART_OVERRIDE.update(self._part)

    def test_yarn_phrase(self):
        self.assertEqual(yarn_ru("honey"), "медово-бежевой нитью")
        self.assertEqual(yarn_ru("ginger"), "рыжей нитью")
        self.assertEqual(yarn_ru("lilac_f"), "сиреневой нитью")
        self.assertEqual(yarn_ru("earbrown"), "коричневой нитью (ушки)")
        self.assertEqual(yarn_ru("ochre_dark"), "нитью цвета тёмной охры")
        self.assertEqual(yarn_ru("ochre"), "нитью цвета охры")
        self.assertEqual(yarn_ru("cocoa"), "нитью цвета какао")
        self.assertEqual(yarn_ru("cocoa_dark"), "нитью цвета тёмного какао")
        self.assertEqual(yarn_ru("sage_light"), "нитью цвета светлого шалфея")
        self.assertEqual(yarn_ru("cherry"), "нитью цвета вишни")
        self.assertEqual(yarn_ru("emerald"), "нитью цвета изумруда")
        self.assertEqual(yarn_ru("latte"), "нитью цвета «кофе с молоком»")       # not declined: quoted
        self.assertEqual(yarn_ru("ochre", cap=True), "Нитью цвета охры")

    def test_instrumental_and_accusative(self):
        cases = {"ochre_dark": ("тёмной охрой", "тёмную охру"), "ochre": ("охрой", "охру"),
                 "cocoa": ("какао", "какао"), "cocoa_dark": ("тёмным какао", "тёмное какао"),
                 "sage_light": ("светлым шалфеем", "светлый шалфей"), "cherry": ("вишней", "вишню"),
                 "earbrown": ("коричневым (ушки)", "коричневую (ушки)"), "honey": ("медово-бежевым", "медово-бежевую"),
                 "latte": ("цветом «кофе с молоком»", "«кофе с молоком»")}
        for name, (ins, acc) in cases.items():
            self.assertEqual(colour_word(name, "ru", "ins"), ins, name)
            self.assertEqual(colour_word(name, "ru", "acc_f"), acc, name)
            self.assertEqual(colour_word(name, "ru"), self.COLOURS[name])
        self.assertEqual(change_ru("ochre_dark"), "Смените нить на тёмную охру")
        self.assertEqual(change_ru("honey"), "Смените нить на медово-бежевую")
        self.assertEqual(change_ru("latte"), "Смените нить: дальше — нитью цвета «кофе с молоком»")
        txt = round_body(["sc"] * 6, ["ochre_dark"] * 3 + ["cocoa"] * 3, "ru")
        self.assertEqual(txt, "3 сбн тёмной охрой, 3 сбн какао")
        self.assertEqual(parse_count(txt), (6, 6))
        txt = round_body(["sc", "inc", "sc"], ["latte", "latte", "latte", "honey"], "ru")
        self.assertIn("[цветом «кофе с молоком»]", txt)
        self.assertEqual(parse_count(txt), (3, 4))

    def test_part_names(self):
        self.assertEqual(ru_decline("глазной бугорок", "gen"), "глазного бугорка")      # fleeting vowel
        self.assertEqual(ru_decline("белок глаза", "prep"), "белке глаза")
        self.assertEqual(ru_decline("огонёк", "dat"), "огоньку")
        self.assertEqual(ru_decline("палец", "gen"), "пальца")
        self.assertEqual(ru_decline("урок", "gen"), "урока")                             # not fleeting
        self.assertEqual(ru_decline("хвостик", "gen"), "хвостика")
        self.assertEqual(ru_decline("голова и тело (одной деталью)", "gen"), "головы и тела (одной деталью)")
        self.assertEqual(qualifier("leg_r", "leg", "ru"), "правая")                      # not '№ 18'
        self.assertEqual(qualifier("leg_l", "leg", "en"), "left")

    def test_usage_groups(self):
        def u(on, mirror):
            return {"name": "x", "mirror": mirror, "on": on, "on_part": on, "on_pair": False}
        uses = [u("body", False), u("body", True), u("body", True), u("head", True), u("head", False)]
        self.assertEqual(_usage_groups({"id": "leaf"}, uses, "ru"),
                         "Из них: на тело — 5 шт. (2 пары, слева и справа, и 1 одиночный); "
                         "на голову — 3 шт. (1 пара, слева и справа, и 1 одиночный).")
        self.assertEqual(_usage_groups({"id": "leaf"}, [u("body", False)] * 3, "ru"), "")   # all alike: no list
        self.assertNotIn("деталь —", _usage_groups({"id": "leg"}, uses, "ru"))

    def test_motif_row_wording(self):
        from .motif import row_text
        r = {"pre": [("chain", (10,)), ("hook", (2,))], "ops": ["sc", "sc"], "post": []}
        self.assertEqual(row_text(r, "ru"), "наберите цепочку из 10 цеп. п.; начиная с 2-й цеп. п. от крючка: 2 сбн")
        self.assertEqual(row_text(r, "en"), "ch 10 (foundation chain); starting in the 2nd ch from hook; 2 sc")
        r = {"pre": [("turn", (1,))], "ops": ["sc"], "post": []}
        self.assertEqual(row_text(r, "ru"), "1 цеп. п. подъёма, поверните работу (в счёт не входит); 1 сбн")


if __name__ == "__main__":
    unittest.main()
