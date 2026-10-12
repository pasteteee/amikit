"""Toy -> a written crochet pattern a maker can follow (PATTERN_RU.md, PATTERN_EN.md, pattern.json).

Everything is derived from the Toy: round operations from the spec's round steps (rounds.expand), colours from
Toy.chart in the order the stitches are actually worked, placements from where the engine seated each piece
(seam footprints located on the parent's rounds and stitches). First a language-neutral model is built
(`pattern_model`), then it is rendered in each language (`render_markdown`); pattern.json is the model.

Conventions written into the pattern:
- spiral rounds, no joins; the round start (marker) sits on the piece's centre reference line (the centre front
  of a body/head, the centre top of a forward-pointing muzzle); multi-colour rounds are listed from the marker in
  the working direction of a right-handed crocheter (right to left on the outside of a tube).
- left / right are the TOY's left / right (the toy's left is on your right when it faces you).
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .piece import LEAN
from .rounds import compress, check

# ---------------------------------------------------------------------------------------------------------------
# small helpers

class _OpTable(dict):
    """Stitches made (or used) by a round operation; edging operations (skip, sl, shell<k><st>) from rounds.op_cp."""

    def __init__(self, d, idx):
        super().__init__(d)
        self.idx = idx

    def __missing__(self, op):
        from .rounds import op_cp
        return op_cp(op)[self.idx]


PRODUCE = _OpTable({"sc": 1, "inc": 2, "inc3": 3, "dec": 1, "same": 1}, 1)
CONSUME = _OpTable({"sc": 1, "inc": 1, "inc3": 1, "dec": 2, "same": 0}, 0)
WORLD = {"left": np.array([1.0, 0, 0]), "right": np.array([-1.0, 0, 0]), "up": np.array([0, 1.0, 0]),
         "down": np.array([0, -1.0, 0]), "front": np.array([0, 0, 1.0]), "back": np.array([0, 0, -1.0])}
OPP = {"left": "right", "right": "left", "up": "down", "down": "up", "front": "back", "back": "front"}

# yarn weight -> typical metres per 100 g (for the materials list; approximate)
YARN_M100 = {"lace": 600, "fingering": 400, "sport": 330, "dk": 250, "worsted": 200, "aran": 180,
             "chunky": 120, "bulky": 110}

RU_COLOUR = {
    "brown": "коричневый", "tan": "бежевый", "beige": "бежевый", "red": "красный", "black": "чёрный",
    "white": "белый", "pink": "розовый", "rose": "розовый", "grey": "серый", "gray": "серый",
    "cream": "кремовый", "ivory": "молочный", "yellow": "жёлтый", "orange": "оранжевый", "green": "зелёный",
    "blue": "синий", "navy": "тёмно-синий", "lightblue": "голубой", "sky": "голубой", "purple": "фиолетовый",
    "lilac": "сиреневый", "mint": "мятный", "peach": "персиковый", "gold": "золотистый", "rust": "рыжий",
    "ginger": "рыжий", "pale": "бледно-розовый", "eartan": "тёмно-рыжий", "darksage": "тёмно-шалфейный", "stripe": "тёмный (полоски)", "caramel": "карамельный", "sage": "шалфейный", "coral": "коралловый",
    "olive": "оливковый", "charcoal": "графитовый", "chocolate": "шоколадный", "sand": "песочный",
    "fawn": "песочный", "mustard": "горчичный", "teal": "бирюзовый", "turquoise": "бирюзовый",
    "silver": "серебристый", "khaki": "хаки", "lavender": "лавандовый", "salmon": "лососевый",
    "main": "основной", "dark": "тёмный", "light": "светлый", "blush": "розовый", "lime": "салатовый",
    "lemon": "лимонный", "lilac": "сиреневый",
}
RU_PREFIX = {"dark": "тёмно-", "light": "светло-", "pale": "бледно-", "deep": "тёмно-", "bright": "ярко-"}
STUFF_RU = {"firm": "плотно", "medium": "умеренно", "light": "слегка", "none": ""}
STUFF_EN = {"firm": "firmly", "medium": "moderately", "light": "lightly", "none": ""}


def _plural_ru(n, one, few, many):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _num(x, lang, d=1):
    s = f"{x:.{d}f}".rstrip("0").rstrip(".") if d else str(int(round(x)))
    return s.replace(".", ",") if lang == "ru" else s


COLOUR_OVERRIDE = {}    # colour name -> Russian name from the spec's "colour_names_ru", set per toy


def colour_ru(name):
    """Russian nominative (masculine, 'цвет') of a colour name; unknown names stay quoted."""
    if name in COLOUR_OVERRIDE:
        return COLOUR_OVERRIDE[name]
    words = name.lower().replace("-", "_").replace(" ", "_").split("_")
    pre = ""
    while len(words) > 1 and words[0] in RU_PREFIX:
        pre += RU_PREFIX[words.pop(0)]
    key = "".join(words)
    if key in RU_COLOUR:
        return pre + RU_COLOUR[key]
    if len(words) == 1 and words[0] in RU_COLOUR:
        return pre + RU_COLOUR[words[0]]
    return None


def _colour_adj(w, case):
    """Inflect one colour adjective (masculine nominative) for 'ins' / 'ins_f' / 'acc_f'."""
    if w.endswith(("ая", "яя", "ое", "ее")) and len(w) > 3:      # 'мятная' -> 'мятный' first
        w = w[:-2] + ("ий" if w.endswith(("яя", "ее")) or w[-3] in "кгхжшчщ" else "ый")
    if not w.endswith(("ый", "ой", "ий")):
        return w
    stem = w[:-2]
    if w.endswith("ий"):
        if stem.endswith(("к", "г", "х")):            # мягкий-type: -им, -ой, -ую
            return stem + {"ins": "им", "ins_f": "ой", "acc_f": "ую"}[case]
        if stem.endswith(("ж", "ш", "ч", "щ")):       # рыжий: рыжим, рыжей, рыжую
            return stem + {"ins": "им", "ins_f": "ей", "acc_f": "ую"}[case]
        return stem + {"ins": "им", "ins_f": "ей", "acc_f": "юю"}[case]   # синий: синим, синей, синюю
    return stem + {"ins": "ым", "ins_f": "ой", "acc_f": "ую"}[case]


def colour_word(name, lang, case="nom"):
    """case: 'nom' (коричневый), 'ins' (коричневым — '[коричневым] ...', '5 сбн коричневым'), 'ins_f'
    (коричневой — '... нитью'), 'acc_f' (коричневую — 'смените нить на ...'). English ignores the case.
    A name that is not just adjectives ('тёмная охра', 'какао', 'светлый шалфей') is declined as a noun phrase
    ('тёмной охрой', 'какао', 'светлым шалфеем'; 'тёмную охру'); one that cannot be declined safely is quoted
    ('цветом «кофе с молоком»'). A parenthesised note ('тёмно-коричневый (нос)') stays as it is."""
    if lang == "en":
        return name.replace("_", " ")
    ru = colour_ru(name)
    if ru is None:
        ru = _derive_colour_ru(name)
    if case == "nom":
        return ru
    head, note = _split_note(ru)
    kind, words = _colour_parse(head)
    if kind == "adj":
        out = " ".join(_colour_adj(w, case) for w in words)
    elif kind == "np":
        out = _colour_np(words, {"ins": "ins", "ins_f": "ins", "acc_f": "acc"}[case])
    else:
        out = (f"цвета «{head}»" if case == "ins_f" else f"цветом «{head}»" if case == "ins" else f"«{head}»")
    return out + note


# colour nouns that do not decline (цвета какао, нитью цвета хаки)
_INVARIANT_RU = {"какао", "кофе", "хаки", "беж", "бордо", "индиго", "мокко", "капучино", "латте", "экрю", "тауп",
                 "нюд", "айвори", "шампань", "марсала", "фуксия-неон", "крем-брюле", "электрик", "маренго"}
_SOFT_F_RU = {"сирень", "лазурь", "медь", "синь", "зелень", "морковь", "сталь"}
_SOFT_M_RU = {"янтарь", "миндаль", "имбирь", "кобальт"}


def _split_note(s):
    head, sep, rest = s.partition(" (")
    return head, (sep + rest if sep else "")


def _colour_adj_like(w):
    a = w.rsplit("-", 1)[-1]
    return len(a) > 3 and a.endswith(("ый", "ий", "ой", "ая", "яя", "ое", "ее"))


def _colour_parse(head):
    """('adj', words) for adjectives only ('тёмно-коричневый', 'травяной зелёный'); ('np', words) for
    adjectives + one declinable noun ('тёмная охра', 'светлый шалфей', 'какао'); ('raw', None) otherwise."""
    words = head.split()
    if not words:
        return "raw", None
    if all(_colour_adj_like(w) for w in words):
        return "adj", words
    if all(_colour_adj_like(w) for w in words[:-1]) and _colour_noun(words[-1], "gen") is not None:
        return "np", words
    return "raw", None


def _colour_noun(w, case):
    """A colour noun in 'gen' / 'ins' / 'acc'; None when its declension is not certain (fallback to quotes)."""
    lw = w.lower()
    if lw in _INVARIANT_RU:
        return w
    if lw in _SOFT_F_RU:
        return w if case == "acc" else w[:-1] + {"gen": "и", "ins": "ью"}[case]
    if lw in _SOFT_M_RU:
        if lw.endswith("ь"):
            return w if case == "acc" else w[:-1] + {"gen": "я", "ins": "ём"}[case]
        return w if case == "acc" else w + {"gen": "а", "ins": "ом"}[case]
    if not lw.isalpha() or len(lw) < 3:
        return None
    if lw.endswith("ия"):
        return w[:-1] + {"gen": "и", "ins": "ей", "acc": "ю"}[case]
    if lw.endswith("я"):
        return w[:-1] + {"gen": "и", "ins": "ей", "acc": "ю"}[case]
    if lw.endswith("а"):
        s = w[:-1]
        if s[-1:] in "жшчщц":                    # -ей / -ой depends on stress: not certain
            return None
        return s + {"gen": "и" if s[-1:] in "гкх" else "ы", "ins": "ой", "acc": "у"}[case]
    if lw.endswith("й") and lw[-2:-1] in "еэа":   # шалфей, чай: шалфея, шалфеем
        return w if case == "acc" else w[:-1] + {"gen": "я", "ins": "ем"}[case]
    if lw[-1] in "бвдзлмнпрстф" and not lw.endswith(("ок", "ек", "ёк", "ец")):   # изумруд, графит, асфальт
        return w if case == "acc" else w + {"gen": "а", "ins": "ом"}[case]
    return None


def _colour_adj_case(w, case):
    """One colour adjective in its own gender (from its ending): 'gen' / 'ins' / 'acc'."""
    pre, _, a = w.rpartition("-")
    pre = pre + "-" if pre else ""
    stem, end = a[:-2], a[-2:]
    g = {"ая": "f", "яя": "f", "ое": "n", "ее": "n"}.get(end, "m")
    velar = stem[-1:] in "кгх"
    hush = stem[-1:] in "жшчщ"
    soft = end in ("ий", "яя", "ее") and not velar and not hush
    if case == "acc":
        if g == "f":
            return pre + stem + ("юю" if soft else "ую")
        return w
    if g == "f":
        return pre + stem + ("ей" if soft or hush else "ой")
    if case == "gen":
        return pre + stem + ("его" if soft or (hush and end != "ой") else "ого")
    return pre + stem + ("им" if soft or velar or hush else "ым")


def _colour_np(words, case):
    """Adjectives + a noun ('тёмная охра') in 'gen' / 'ins' / 'acc' (each word keeps the gender it is given in)."""
    return " ".join([_colour_adj_case(w, case) for w in words[:-1]] + [_colour_noun(words[-1], case)])


def yarn_ru(name, cap=False):
    """'медово-бежевой нитью', 'нитью цвета тёмной охры', 'нитью цвета «кофе с молоком»' (+ the name's note)."""
    ru = colour_ru(name) or _derive_colour_ru(name)
    head, note = _split_note(ru)
    kind, words = _colour_parse(head)
    if kind == "adj":
        s = " ".join(_colour_adj(w, "ins_f") for w in words) + " нитью" + note
    elif kind == "np":
        s = "нитью цвета " + _colour_np(words, "gen") + note
    else:
        s = f"нитью цвета «{head}»" + note
    return s[:1].upper() + s[1:] if cap else s


def change_ru(name):
    """'Смените нить на оранжевую' / 'Смените нить на тёмную охру' / 'Смените нить: дальше — нитью цвета «…»'."""
    ru = colour_ru(name) or _derive_colour_ru(name)
    head, note = _split_note(ru)
    kind, _ = _colour_parse(head)
    if kind == "raw":
        return f"Смените нить: дальше — {yarn_ru(name)}"
    return f"Смените нить на {colour_word(name, 'ru', 'acc_f')}"


def _derive_colour_ru(name):
    """A colour name the dictionaries do not know: 'цвет <name>' would leave Latin in the Russian text, so
    describe it by its parts where possible, else as 'основной/дополнительный' plus the id."""
    words = name.lower().replace("-", "_").split("_")
    known = [RU_COLOUR[w] for w in words if w in RU_COLOUR]
    if known:
        return known[-1]
    hx = COLOUR_HEX.get(name, "")
    if isinstance(hx, str) and len(hx) == 7 and hx.startswith("#"):
        rgb = np.array([int(hx[i:i + 2], 16) for i in (1, 3, 5)], float)
        return min(BASIC_HEX, key=lambda k: float(((BASIC_HEX[k] - rgb) ** 2).sum()))
    return "дополнительный"


COLOUR_HEX = {}          # colour name -> hex from the spec, set per toy
BASIC_HEX = {k: np.array([int(v[i:i + 2], 16) for i in (1, 3, 5)], float) for k, v in {
    "чёрный": "#151313", "белый": "#f4f1ea", "серый": "#9a9a9a", "тёмно-серый": "#505050",
    "коричневый": "#7a4a2a", "тёмно-коричневый": "#3e2620", "бежевый": "#dcb488", "кремовый": "#efe3cc",
    "красный": "#c0302a", "розовый": "#f2a0a8", "оранжевый": "#e8892b", "жёлтый": "#f2c94c",
    "зелёный": "#4a8a3a", "салатовый": "#a8d060", "голубой": "#8cc4e8", "синий": "#2a4a9a",
    "фиолетовый": "#7a4a9a", "сиреневый": "#c4a8e0", "бирюзовый": "#3aa8a0"}.items()}


def _dominant(v):
    v = np.asarray(v, float)
    return max(WORLD, key=lambda k: float(WORLD[k] @ v))


# ---------------------------------------------------------------------------------------------------------------
# round text

TOK = {
    "ru": {"sc": "сбн", "inc": "пр", "dec": "уб", "inc3": "3 сбн в 1 п.", "same": "1 сбн в ту же п.",
           "pop": "попкорн"},
    "en": {"sc": "sc", "inc": "inc", "dec": "dec", "inc3": "3 sc in next st", "same": "1 sc in same st",
           "pop": "popcorn"},
}


def _runs(tokens):
    out = []
    for t in tokens:
        if out and out[-1][0] == t:
            out[-1][1] += 1
        else:
            out.append([t, 1])
    return out


def _fmt_run(tok, k, lang):
    w = TOK[lang][tok]
    if tok == "sc":
        return f"{k} {w}"
    if tok in ("inc", "dec"):
        if k == 1:
            return w
        return f"{k} {w}" if lang == "ru" else f"{w} ×{k}"
    return w if k == 1 else f"({w}) ×{k}"


def _fmt_seq(tokens, lang):
    return ", ".join(_fmt_run(t, k, lang) for t, k in _runs(tokens))


def group_ops(tokens):
    """Designer grouping of a round: [(unit, reps)], staggered rounds rotated to 'a sc, (inc, b sc) xR, inc, c sc'."""
    if not tokens:
        return []
    groups = compress(list(tokens))
    if len(groups) == 1:
        unit, reps = groups[0]
        special = [i for i, t in enumerate(unit) if t != "sc"]
        if reps >= 2 and len(special) == 1 and 0 < special[0] < len(unit) - 1:
            i = special[0]; op = unit[i]
            lead, tail = i, len(unit) - 1 - i
            return [(["sc"] * lead, 1), ([op] + ["sc"] * (lead + tail), reps - 1), ([op] + ["sc"] * tail, 1)]
    return [(list(u), r) for u, r in groups]


def fmt_tokens(tokens, lang):
    parts = []
    for unit, reps in group_ops(tokens):
        if reps == 1:
            parts.append(_fmt_seq(unit, lang))
        elif len(set(unit)) == 1:
            parts.append(_fmt_seq(unit * reps, lang))
        else:
            parts.append(f"({_fmt_seq(unit, lang)}) ×{reps}")
    return ", ".join(p for p in parts if p)


def colour_segments(ops, colours):
    """Split a round's operations by the colour of the stitches they make.

    ops: worked operations in working order; colours: colour of every output stitch in the same order.
    An increase whose two stitches differ in colour is split into 'sc' + 'same' (a 2nd sc in the same stitch)."""
    segs = []
    j = 0

    def put(c, t):
        if segs and segs[-1][0] == c:
            segs[-1][1].append(t)
        else:
            segs.append([c, [t]])

    for op in ops:
        k = PRODUCE[op]
        cs = colours[j:j + k]
        j += k
        if k == 0:                         # a skipped stitch makes nothing: it goes with the current colour
            if segs:
                segs[-1][1].append(op)
            else:
                put(colours[min(j, len(colours) - 1)], op)
            continue
        if len(set(cs)) == 1:
            put(cs[0], op)
        else:
            put(cs[0], "sc")
            for c in cs[1:]:
                put(c, "same")
    if j != len(colours):
        raise ValueError(f"ops make {j} stitches, chart has {len(colours)}")
    return segs


def round_body(ops, colours, lang):
    """The instruction for one round (without the 'R5:' label and the total)."""
    segs = colour_segments(ops, colours)
    if len(segs) == 1:
        return fmt_tokens(ops, lang)
    if all(t == "sc" for _, toks in segs for t in toks):
        if lang == "ru":
            return ", ".join(f"{len(t)} сбн {colour_word(c, 'ru', 'ins')}" for c, t in segs)
        return ", ".join(f"{len(t)} sc in {colour_word(c, 'en')}" for c, t in segs)
    out = []
    for c, toks in segs:
        tag = colour_word(c, lang, "ins") if lang == "ru" else colour_word(c, "en")
        out.append(f"[{tag}] {fmt_tokens(toks, lang)}")
    return " ".join(out)


def special_round_text(r, lang):
    """Ribbing and shell-edging rounds (amikit.rounds 'rib N', 'shells K every E')."""
    if r.loop == "rib":
        return (f"(РЛСН, РИСН) ×{r.n // 2}" if lang == "ru" else f"(fpdc, bpdc) ×{r.n // 2}")
    import re as _re
    from .motif import ops_text
    ops = []
    for o in r.ops:
        m = _re.fullmatch(r"shell(\d+)(sc|hdc|dc|tr)", o)
        ops.append(f"inc:{m.group(2)}:{m.group(1)}" if m else o)
    return ops_text(ops, lang)


def parse_count(text):
    """(consumed, produced) of a round body written by round_body (used by tests and the self-check)."""
    import re
    s = re.sub(r"\[[^\]]*\]", ",", text)
    cons = prod = 0

    def item(it):
        it = it.strip()
        if not it:
            return 0, 0
        m = re.fullmatch(r"\((.*)\)\s*×(\d+)", it)
        if m:
            c, p = seq(m.group(1)); r = int(m.group(2))
            return c * r, p * r
        if re.match(r"1 (сбн в ту же п\.|sc in same st)", it):
            return 0, 1
        if re.match(r"3 (сбн в 1 п\.|sc in next st)", it):
            return 1, 3
        m = re.match(r"(\d+)\s+(сбн|sc)\b", it)
        if m:
            return int(m.group(1)), int(m.group(1))
        m = re.match(r"(?:(\d+)\s+)?(попкорн|popcorn)(?:\s*×(\d+))?$", it)
        if m:
            k = int(m.group(1) or m.group(3) or 1)
            return k, k
        m = re.match(r"(?:(\d+)\s+)?(пр|уб|inc|dec)(?:\s*×(\d+))?", it)
        if m:
            k = int(m.group(1) or m.group(3) or 1)
            if m.group(2) in ("пр", "inc"):
                return k, 2 * k
            return 2 * k, k
        raise ValueError(f"cannot read {it!r}")

    def seq(s):
        c = p = 0
        depth = 0; cur = ""
        for ch in s + ",":
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                a, b = item(cur); c += a; p += b; cur = ""
            else:
                cur += ch
        return c, p

    cons, prod = seq(s)
    return cons, prod


# ---------------------------------------------------------------------------------------------------------------
# geometry helpers (read-only use of Toy / Instance)

class _Locator:
    """Find (round, az) on a placed piece for world points."""

    def __init__(self):
        self.cache = {}

    def locate(self, inst, X):
        if inst.name not in self.cache:
            p = inst.p
            ts = np.linspace(0.02, p.M - 0.02, max(12, int(p.M * 6)))
            ths = np.linspace(0, 2 * np.pi, max(64, max(p.counts) * 4), endpoint=False)
            T, TH = np.meshgrid(ts, ths, indexing="ij")
            P = inst.base(T, TH).reshape(-1, 3)
            self.cache[inst.name] = (cKDTree(P), T.reshape(-1), TH.reshape(-1))
        tree, T, TH = self.cache[inst.name]
        d, i = tree.query(np.atleast_2d(X))
        az = (inst.az_of_theta(TH[i]) + 180) % 360 - 180
        return T[i], az, d


def _footprint(inst, t, az, child_n=None):
    """Rounds and azimuth extent of a set of located points on a piece."""
    N = inst.p.N
    rounds = np.clip(np.floor(t).astype(int) + 1, 1, N)
    a = np.radians(az)
    c = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
    # span = 360 minus the largest empty gap between the located azimuths (robust where the points straddle
    # +-180, e.g. a seam on the centre back)
    s = np.sort(np.mod(np.asarray(az, float), 360.0))
    if len(s) > 1:
        gaps = np.diff(np.concatenate([s, [s[0] + 360.0]]))
        k = int(np.argmax(gaps))
        span = float(360.0 - gaps[k])
        lo = s[(k + 1) % len(s)]
        c = float((lo + span / 2 + 180) % 360 - 180)
    else:
        span = 0.0
    r_lo, r_hi = int(rounds.min()), int(rounds.max())
    rmid = int(round((r_lo + r_hi) / 2))
    n = inst.p.counts[rmid - 1]
    around = span > 300 and (child_n is None or child_n >= 0.7 * n)
    # a small seam close to a pole spans every azimuth without going round the piece
    near_pole = None
    if span > 300 and not around:
        near_pole = "end" if r_hi >= N - 1 else ("start" if r_lo <= 2 else None)
        if near_pole is None:
            around = True
    return {"rounds": [r_lo, r_hi], "az_centre": round(c, 1), "az_span": round(span, 1),
            "around": around, "near_pole": near_pole, "stitches_across": int(round(span / 360 * n)),
            "count_at_mid": n}


def _ref_dirs(inst):
    """World names of a piece's reference line (az 0) and of its +az side."""
    e1 = inst.R[:, 0]; e2 = inst.R[:, 1]
    return _dominant(e1), _dominant(inst.az_sign * e2)


def position(inst, rnd, az):
    """Maker's position on a piece: which round(s) and how many stitches from the centre line, which side."""
    N = inst.p.N
    pos = {"piece": inst.p.id}
    if rnd in ("end", "start"):
        pos["pole"] = rnd
        return pos
    r = float(rnd)
    fr = r - math.floor(r)
    a = int(math.floor(r))
    if 0.3 <= fr <= 0.7 and a >= 1 and a + 1 <= N:
        pos["between"] = [a, a + 1]
        ref_round = a
    else:
        ref_round = int(min(max(round(r), 1), N))
        pos["round"] = ref_round
    n = inst.p.counts[min(ref_round, N) - 1]
    pos["count_round"] = ref_round
    pos["n"] = n
    ref, plus = _ref_dirs(inst)
    az = (float(az) + 180) % 360 - 180
    if abs(az) <= 120:
        k = round(abs(az) / 360 * n, 2)
        pos["from"] = ref
    else:
        k = round((180 - abs(az)) / 360 * n, 2)
        pos["from"] = OPP[ref]
    pos["stitches"] = k
    pos["side"] = None if k < 0.05 else (plus if az > 0 else OPP[plus])
    pos["exact_round"] = r                  # the round as the engine has it (e.g. 23.2), kept in a comment
    return pos


def _axis_word(v):
    """'x' (side to side), 'y' (up-down) or 'z' (front to back): the world axis a direction mostly runs along."""
    v = np.abs(np.asarray(v, float))
    return "xyz"[int(np.argmax(v))]


def _round_of_t(t):
    return float(t) + 0.5


def _seam_info(inst, parent, entry):
    """Line of a flattened opening sewn as a straight seam, as the engine lays it: the press direction is
    `squash_phi` in the piece's frame (set from `seam_line` or the default cross(parent normal, piece direction),
    falling back to side-to-side), the line is perpendicular to it and to the piece's axis."""
    phi = float(inst.lp.get("squash_phi", 0.0))
    e3 = inst.R[:, 2]
    press = inst.R @ np.array([math.cos(phi), math.sin(phi), 0.0])
    sl = np.cross(press, e3)
    if np.linalg.norm(sl) < 1e-6:
        sl = np.array([1.0, 0, 0])
    sl = sl / np.linalg.norm(sl)
    info = {"line": _axis_word(sl), "line_vec": [round(float(x), 3) for x in sl], "along_round": None}
    at = entry.get("at", {"round": "end"})
    r = at.get("round", "end")
    if r not in ("end", "start"):
        r = float(min(max(float(r), 1.0), parent.p.N))
        az = at.get("az", 0.0)
        P0, n = parent.point(max(r - 0.5, 0.6), az)
        P1, _ = parent.point(min(r + 0.5, parent.p.N - 0.1), az)
        m = P1 - P0
        m = m - (m @ n) * n
        if np.linalg.norm(m) > 1e-6:
            m = m / np.linalg.norm(m)
            around = np.cross(n, m)
            info["along_round"] = bool(abs(sl @ around) >= abs(sl @ m))
    return info


def _press_dims(pdef):
    """Measures a maker can check on a spoon- or dish-pressed piece (mm, rounded): depth of the hollow in front,
    thickness of the two layers / of the rim, width; for a dish also the rounds that form the rim."""
    if getattr(pdef, "dish", 0) > 0:
        tr = pdef.dish_rim_t()
        return {"depth": round(float(pdef.dish_depth), 1), "rim": round(float(2 * pdef.dish_rim), 1),
                "width": round(float(2 * (pdef.dish_R + pdef.dish_rim)), 0),
                "front_rounds": int(round(pdef.dish_tw)), "rim_round": min(int(np.ceil(tr)), pdef.N)}
    if getattr(pdef, "spoon", 0) > 0:
        W, g, D = pdef.spoon_dims_from_r(pdef.rb)
        i = int(np.argmax(W))
        return {"depth": round(float(D[i]), 1), "layers": round(float(g[i]), 1), "width": round(float(2 * W[i]), 0)}
    return None


def _bend_info(inst):
    """Where and how much a bent piece must be tacked: the inner side of the bend (on the piece and on the toy)
    and how many rounds the fold takes up (bend angle x the piece's radius there / round height)."""
    p = inst.p
    b = p.bend
    deg = float(b.get("deg", 0.0)); at = float(b.get("at", p.N / 2)); over = float(b.get("over", 2.0))
    phi = float(inst.lp.get("bend_phi", 0.0))
    inner_world = inst.R @ np.array([math.cos(phi), math.sin(phi), 0.0])
    if deg < 0:
        inner_world = -inner_world; phi += math.pi
    az = float((inst.az_of_theta(phi) + 180) % 360 - 180)
    ra = int(min(max(round(at), 1), p.N))
    r_mm = float(p.r_t(max(at - 0.5, 0.01)))
    if p.flatten > 0:
        r_mm *= max(1 - p.flatten, 0.3)
    take = math.radians(abs(deg)) * r_mm / p.h
    k = int(round(take))
    a0 = max(1, int(round(at - k / 2)))
    a1 = min(p.N, a0 + max(k, 1))
    n_at = p.counts[ra - 1]
    coil = b.get("coil")
    if coil:
        coil = 4.0 if isinstance(coil, bool) else float(coil)
    drape = None
    if b.get("drape"):
        drape = {k: b[k] for k in ("relax_mm", "min_radius_mm", "clear_mm") if k in b}
        tw = inst.lp.get("drape_twist")
        if tw:
            drape["twist"] = float(tw[1])          # as built (its default follows the inner colour): always stated
        drape["_twist"] = float(tw[1]) if tw else None
    return {"deg": deg, "at": at, "over": over, "rounds": [a0, a1], "take_rounds": round(take, 1),
            "coil": coil, "drape": drape, "pitch_mm": b.get("pitch_mm"),
            "inner": _dominant(inner_world), "inner_pos": position(inst, ra, az),
            "width_st": max(2, int(round(n_at / 4)))}


def _oval_dims(pdef):
    """(length, width) in mm of a chain-oval piece at its widest (3D pieces), or of a flat applique."""
    w = 2 * float(pdef.rb.max())
    return w + float(pdef.oval), w


def _flat_extent(pdef):
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    x, y = pdef.flat_points(np.full_like(th, float(pdef.M)), th)
    return float(np.ptp(x)), float(np.ptp(y))


def _chain_lie(inst):
    """World axis the starting chain of a chain-oval piece lies along on the finished toy, and where the start
    (round 1) of the piece is."""
    p = inst.p
    e = inst.R[:, 0] if getattr(p, "oval_axis", "side") == "front" else inst.R[:, 1]
    return _axis_word(e)


def _surface_walk(parent, loc, P, d, dist, step=1.0):
    """Follow the parent's surface from P along d for `dist` mm (a strip laid along the curve)."""
    P = np.array(P, float); d = np.array(d, float)
    n_steps = max(1, int(dist / step))
    for _ in range(n_steps):
        X = P + d * (dist / n_steps)
        t, az, _ = loc.locate(parent, X)
        Q, n = parent.point(_round_of_t(t[0]), float(az[0]))
        nd = Q - P
        if np.linalg.norm(nd) > 1e-6:
            d = nd / np.linalg.norm(nd)
        d = d - (d @ n) * n
        if np.linalg.norm(d) > 1e-6:
            d = d / np.linalg.norm(d)
        P = Q
    return P


def _path_length(inst, pts):
    """Length (mm) of a path through (round, az) points on a piece, the short way round, as the engine draws
    embroidery and straps."""
    tt = np.array([inst.t_of_round(r) for r, a in pts]); aa = np.array([a for r, a in pts], float)
    L = 0.0
    for k in range(len(pts) - 1):
        a0 = aa[k]; a1 = a0 + ((aa[k + 1] - a0 + 180) % 360 - 180)
        s = np.linspace(0, 1, 24)
        P = inst.base(tt[k] + (tt[k + 1] - tt[k]) * s, inst.theta(a0 + (a1 - a0) * s))
        L += float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return L


# ---------------------------------------------------------------------------------------------------------------
# the model

def _pairs(toy):
    """Mirrored entries: entry name (or part id) -> (left, right) instance names (Toy.pairs; older engines: _L/_R)."""
    if hasattr(toy, "pairs"):
        return dict(toy.pairs)
    pr = {}
    for nm in toy.inst:
        if nm.endswith("_L") and nm[:-2] + "_R" in toy.inst:
            pr.setdefault(nm[:-2], (nm, nm[:-2] + "_R"))
    return pr


def side_of(toy, name):
    """'L' / 'R' when the instance is one copy of a mirrored pair, else None."""
    for l_, r_ in _pairs(toy).values():
        if name == l_:
            return "L"
        if name == r_:
            return "R"
    return None


def _inst_on(toy, on, side="L"):
    """The parent instance for an entry's `on` (a pair's left copy unless side='R')."""
    if hasattr(toy, "_side"):
        try:
            return toy.inst[toy._side(on, side)]
        except KeyError:
            pass
    if on in toy.inst:
        return toy.inst[on]
    pr = _pairs(toy)
    if on in pr:
        return toy.inst[pr[on][0 if side != "R" else 1]]
    raise KeyError(on)


def _is_pair_name(toy, on):
    return on in _pairs(toy)


def _on_pair(e, toy):
    """A mirrored entry sewn onto a mirrored parent (one copy on each of the pair: inner ears, toe beans)."""
    on = e.get("on") or ""
    if not on:
        return False
    if _is_pair_name(toy, on):
        return True
    return bool(e.get("mirror")) and side_of(toy, on) == "L"


def _physical(inst, chart):
    """Colours in working order from the marker at az 0: a tube is worked toward decreasing theta (right-handed,
    right to left on the outside); a flat applique is seen from its right side, i.e. toward increasing theta."""
    if inst.app is not None:
        return [list(r) for r in chart]
    return [list(r[::-1]) for r in chart]


def _toy_size(toy):
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for inst in toy.inst.values():
        P, _, _ = inst.samples(per_round=2, per_stitch=1.0)
        lo = np.minimum(lo, P.min(0)); hi = np.maximum(hi, P.max(0))
    t = float(toy.yarn.get("thickness_mm", 1.9))
    ext = hi - lo + t
    return {"height_cm": round(ext[1] / 10, 1), "width_cm": round(ext[0] / 10, 1), "depth_cm": round(ext[2] / 10, 1)}


def _face_items(toy):
    items = []
    for f in toy.spec.get("face", []):
        if f.get("mirror") and _is_pair_name(toy, f["on"]):
            items.append((f, [_inst_on(toy, f["on"], "L"), _inst_on(toy, f["on"], "R")], [f, f]))
        elif f.get("mirror"):
            b = json.loads(json.dumps(f))
            if "at" in b:
                b["at"]["az"] = -f["at"].get("az", 0)
            if "points" in b:
                b["points"] = [[r, -a] for r, a in f["points"]]
            inst = toy.inst[f["on"]]
            items.append((f, [inst, inst], [f, b]))
        else:
            inst = _inst_on(toy, f["on"])
            items.append((f, [inst], [f]))
    return items


def _direction(inst, entry):
    attach = entry.get("attach", "end")
    if attach == "edge":
        v = inst.R[:, 0]          # a piece sewn by its edge stands out along local x (its face looks along -z)
    else:
        v = -inst.R[:, 2] if attach == "end" else inst.R[:, 2]
    side = np.sign(inst.origin[0]) if abs(inst.origin[0]) > 1.0 else 0.0
    comps = []
    for k, u in WORLD.items():
        c = float(v @ u)
        if c > 0.3:
            name = k
            if k in ("left", "right") and side != 0:
                name = "out" if (k == "left") == (side > 0) else "in"
            comps.append([name, round(c, 2)])
    comps.sort(key=lambda x: -x[1])
    return comps


def _norm3(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _fine(inst, e, attach, applique=False):
    """What fixes a piece's placement exactly, beyond the seam position: the axis direction, which way its
    round-start (centre front) line faces, the bend and seam-line directions, depths and paddings. Written as
    angles and millimetres a maker can check with a ruler; the reverse reader rebuilds the same placement."""
    f = {}
    if e.get("attach") == "edge":
        f["face"] = [float(x) for x in -inst.R[:, 2]]
        f["dir"] = [float(x) for x in inst.R[:, 0]]
        for k in ("cup", "sink"):
            if k in e:
                f[k] = float(e[k])
        return f
    if applique:
        f["front"] = [float(x) for x in inst.R[:, 0]]
        for k in ("lift", "pad", "inset"):
            if k in e:
                f[k] = float(e[k])
        return f
    v = -inst.R[:, 2] if attach == "end" else inst.R[:, 2]
    f["dir"] = [float(x) for x in v]
    f["front"] = [float(x) for x in inst.R[:, 0]]
    for k in ("sink", "conform_span", "pinch", "pinch_rounds"):
        if k in e:
            f[k] = float(e[k])
    if getattr(inst, "sink_vec", None) is not None:
        f["sink_dir"] = [float(x) for x in inst.sink_vec]
    if "pinch" in f and "pinch_rounds" not in f:
        from .toy import seam_pinch_rounds
        f["pinch_rounds"] = seam_pinch_rounds(inst.p, e)
    if e.get("conform") is False:
        f["conform"] = False
    if e.get("conform_parent") is False:
        f["conform_parent"] = False
    if "squash_phi" in inst.lp and e.get("seam") == "flat":
        phi = float(inst.lp["squash_phi"])
        press = inst.R @ np.array([math.cos(phi), math.sin(phi), 0.0])
        sl = np.cross(press, inst.R[:, 2])
        f["seam_line"] = [float(x) for x in _norm3(sl)]
    if inst.p.bend and (inst.lp.get("bend_free_start") is not None or "bend_phi" in inst.lp):
        phi = float(inst.lp.get("bend_phi", 0.0))           # (0 = toward the piece's front reference)
        f["bend"] = [float(x) for x in inst.R @ np.array([math.cos(phi), math.sin(phi), 0.0])]
    return f


def pattern_model(toy):
    spec = toy.spec
    yarn = toy.yarn
    loc = _Locator()
    insts = toy.inst
    by_part = {}
    for name, inst in insts.items():
        by_part.setdefault(inst.p.id, []).append(inst)
    entry_of = {e.get("name", e["part"]): e for e in toy.entries}
    order = []
    for e in toy.entries:
        if e["part"] not in order:
            order.append(e["part"])
    for pid in toy.defs:
        if pid not in order:
            order.append(pid)

    # face items per host part (to say when to insert them)
    face = _face_items(toy)
    safety_on = {}
    safety_round = {}          # the last round that must exist before the eyes/nose go in
    for f, hosts, _ in face:
        if f["type"] in ("safety_eye", "eye", "safety_nose"):
            safety_on.setdefault(hosts[0].p.id, []).append(f["type"])
            r = f.get("at", {}).get("round")
            if isinstance(r, (int, float)):
                pid_ = hosts[0].p.id
                safety_round[pid_] = max(safety_round.get(pid_, 0), int(math.ceil(float(r))))

    # parts ------------------------------------------------------------------------------------------------
    parts = []
    stitches_by_colour = {}
    for pid in order:
        if pid not in toy.defs:      # chain cords have no rounds: described in the assembly
            continue
        pdef = toy.defs[pid]
        ins = by_part.get(pid, [])
        if not ins:
            continue
        charts = [_physical(i, toy.chart(i)) for i in ins]
        op = getattr(pdef, "opening", None)
        for inst_, ch in zip(ins, charts):
            for ri, row in enumerate(ch):
                if op is not None and ri >= op["from"]:
                    row = row[op["left"]:len(row) - op["right"]]      # the unworked front window
                for c in row:
                    stitches_by_colour[c] = stitches_by_colour.get(c, 0) + 1
            if (entry_of.get(inst_.name, {}).get("attach") == "edge" and not pdef.dish > 0
                    and getattr(pdef, "motif", None) is None):                               # the plain back circle
                stitches_by_colour[pdef.colour] = stitches_by_colour.get(pdef.colour, 0) + sum(pdef.counts)
        rounds = []
        cur = charts[0][0][0]
        for i, (r, cols) in enumerate(zip(pdef.rounds, charts[0]), 1):
            ops = list(r.ops)
            turned = op is not None and i > op["from"]
            if turned:
                # a turned row of a front opening: only the stitches that exist are worked (the window is plain sc)
                ops = ops[op["left"]:len(ops) - op["right"]]
                cols = cols[op["left"]:len(cols) - op["right"]]
            segs = colour_segments(ops, cols)
            change = None
            if len(segs) == 1 and segs[0][0] != cur and i > 1:
                change = segs[0][0]
            rd = {"round": i, "n": r.n - (op["stitches"] if turned else 0), "loop": r.loop, "ops": ops,
                  **({"turned": True, "n_full": r.n} if turned else {}),
                  "groups": [[u, k] for u, k in group_ops(ops)],
                  "colour_runs": [[c, sum(PRODUCE[t] for t in toks)] for c, toks in segs],
                  "segments": [[c, toks] for c, toks in segs], "change_to": change}
            if i == 1:
                rd["start"] = r.start
                rd["chain"] = r.chain
            if r.loop in ("rib", "shell"):
                rd["text_ru"] = special_round_text(r, "ru"); rd["text_en"] = special_round_text(r, "en")
            else:
                rd["text_ru"] = round_body(ops, cols, "ru")
                rd["text_en"] = round_body(ops, cols, "en")
            rounds.append(rd)
            cur = segs[-1][0]
        # instances whose colour chart differs (asymmetric zones on a mirrored pair)
        variants = []
        for inst, ch in zip(ins[1:], charts[1:]):
            diff = []
            for i, (r, cols) in enumerate(zip(pdef.rounds, ch), 1):
                if cols != charts[0][i - 1]:
                    diff.append({"round": i, "n": r.n, "text_ru": round_body(list(r.ops), cols, "ru"),
                                 "text_en": round_body(list(r.ops), cols, "en"),
                                 "colour_runs": [[c, sum(PRODUCE[t] for t in toks)]
                                                 for c, toks in colour_segments(list(r.ops), cols)]})
            if diff:
                variants.append({"instance": inst.name, "side": side_of(toy, inst.name), "rounds": diff})
        symmetric = all(ch[i] == ch[i][::-1] or len(set(ch[i])) == 1 for ch in charts for i in range(len(ch)))
        counts = pdef.counts
        mx = max(counts)
        imax = counts.index(mx)
        stuff_after = None
        if pdef.closed:
            stuff_after = pdef.N - 1
            for i in range(imax + 1, pdef.N):
                if counts[i] <= 0.6 * mx:
                    stuff_after = i + 1
                    break
            # safety eyes go in before stuffing: never before the round that holds them exists
            stuff_after = max(stuff_after, safety_round.get(pid, 0))
            stuff_after = max(1, min(stuff_after, pdef.N - 1))
        first = ins[0]
        side_names = sorted({i.name for i in ins})
        # how the copies are used: one assembly entry may make 2 (mirrored), a part may be in several entries
        uses = []
        for e in toy.entries:
            if e["part"] == pid and not e.get("_mirror"):
                on = e.get("on")
                on_pair = _on_pair(e, toy)
                on_part = _inst_on(toy, on).p.id if on else None
                uses.append({"name": e["name"], "mirror": bool(e.get("mirror")), "on": on, "on_pair": on_pair,
                             "side": side_of(toy, e["name"]),
                             "on_part": on_part})
        edge_n = sum(1 for e in toy.entries if e["part"] == pid and e.get("attach") == "edge")
        motif = None
        if getattr(pdef, "motif", None) is not None:
            # a shaped flat piece: its written rows come from amikit/motif.py (the same rows make its geometry)
            from .motif import row_text
            for rd, mr in zip(rounds, pdef.rows):
                rd["text_ru"] = row_text(mr, "ru"); rd["text_en"] = row_text(mr, "en")
                rd["motif_label"] = mr["label"]
            a_, b_ = pdef.size_mm()
            motif = {"kind": pdef.kind, "params": dict(pdef.m), "size": [round(a_, 0), round(b_, 0)],
                     "rows_turned": any(k == "turn" for r in pdef.rows for k, _ in r["pre"]),
                     "sew": pdef.sew}
            edge_n = 0
        size = {"diameter": round(2 * pdef.max_radius(), 0),
                "length": round(float(pdef.zb.max() - pdef.zb.min()), 0)}
        if pdef.start == "ch":
            L_, W_ = (_flat_extent(pdef) if first.app is not None else _oval_dims(pdef))
            size["oval"] = [round(max(L_, W_), 0), round(min(L_, W_), 0)]
            size["chain_lies"] = _chain_lie(first)
            size["start_at"] = _dominant(-first.R[:, 2]) if first.app is None else None
        elif first.app is not None:
            size["flat"] = round(max(_flat_extent(pdef)), 0)
        parts.append({
            "id": pid, "make": len(ins), "instances": side_names, "colour": charts[0][0][0],
            "pair": len(ins) == 2 and {side_of(toy, i.name) for i in ins} == {"L", "R"},
            "main_colour": pdef.colour, "uses": uses, "edge": edge_n > 0 and not pdef.dish > 0,
            "spoon": pdef.spoon, "dish": pdef.dish, "press_dims": _press_dims(pdef),
            "oval_axis": getattr(pdef, "oval_axis", "side"),
            "start": pdef.start, "chain": pdef.rounds[0].chain, "closed": pdef.closed, "stuff": pdef.stuff,
            **({"split": True} if getattr(pdef, "split", False) else {}),
            **({"opening": {"from": op["from"], "stitches": op["stitches"], "left": op["left"],
                            "right": op["right"]}} if op is not None else {}),
            "flatten": pdef.flatten, "squash": pdef.squash, "bend": pdef.bend, "applique": first.app is not None,
            "bend_info": _bend_info(first) if pdef.bend else None,
            "stuff_after": stuff_after, "safety_items": safety_on.get(pid, []),
            "multicolour": any(len(r["colour_runs"]) > 1 for r in rounds),
            "symmetric_colours": symmetric, "rounds": rounds, "variants": variants,
            "ref": _ref_dirs(first)[0],
            "size_mm": size,
            "check": check(pdef.rounds) if motif is None else pdef.check(),
            **({"motif": motif} if motif else {}),
        })

    # assembly -----------------------------------------------------------------------------------------------
    steps = []
    extra_m = {}             # colour -> metres for pieces outside the round charts (cords, strips, straps, fringe)
    w_st = float(yarn["stitch_mm"])
    per_st_m = 6.0 * w_st / 1000.0
    n_entries = {}
    for e in toy.entries:
        if not e.get("_mirror"):
            n_entries[e["part"]] = n_entries.get(e["part"], 0) + 1
    seen = {}
    for e in toy.entries:
        if e.get("_mirror"):
            continue
        seen[e["part"]] = seen.get(e["part"], 0) + 1
        common = {"part": e["part"], "name": e.get("name", e["part"]), "mirror": bool(e.get("mirror")),
                  "on_name": e.get("on"), "entry_no": seen[e["part"]], "entries": n_entries[e["part"]],
                  "side": side_of(toy, e.get("name", e["part"])),
                  "on_side": side_of(toy, e["on"]) if e.get("on") else None}
        if e.get("on"):
            common["on_pair"] = _on_pair(e, toy)
        if e["part"] in getattr(toy, "flats", {}):
            fd = toy.flats[e["part"]]
            parent = _inst_on(toy, e["on"])
            at = e.get("at", {"round": "end"})
            W, R_ = int(fd["flat"]["width_st"]), int(fd["flat"]["rows"])
            P, n = parent.point(at.get("round", "end"), at.get("az", 0.0))
            up = np.array([0, 1.0, 0]) - n[1] * n if abs(n[1]) < 0.95 else np.array([0, 0, 1.0]) - n[2] * n
            up = up / np.linalg.norm(up)
            side = np.cross(up, n)
            rot = math.radians(float(e.get("roll", 0.0)))
            ex = math.cos(rot) * side + math.sin(rot) * up
            half = W * w_st / 2
            if e.get("conform"):
                ends = [_surface_walk(parent, loc, P, s_ * ex, half) for s_ in (1, -1)]
            else:
                ends = [P + ex * half, P - ex * half]
            end_pos = []
            for X in ends:
                t, az, _ = loc.locate(parent, X)
                end_pos.append(position(parent, _round_of_t(t[0]), float(az[0])))
            k = 2 if e.get("mirror") else 1
            c = fd.get("colour", "black")
            extra_m[c] = extra_m.get(c, 0) + k * (per_st_m * (W * R_ + W + 1) + 0.35)
            steps.append({**common, "kind": "flat", "on": parent.p.id,
                          "fine": {k_: e[k_] for k_ in ("roll", "lift") if k_ in e},
                          "at": position(parent, at.get("round", "end"), at.get("az", 0.0)),
                          "width": W, "rows": R_, "length_mm": round(W * w_st, 0),
                          "pinch": float(fd.get("pinch", 0)), "conform": bool(e.get("conform")),
                          "runs": _axis_word(ex), "ends": end_pos, "colour": c})
            continue
        if e["part"] in getattr(toy, "cords", {}):
            cd = toy.cords[e["part"]]
            parent = _inst_on(toy, e["on"])
            at = e.get("at", {"round": "end"})
            L_mm = float(cd["cord"].get("length_mm", 40))
            chains = max(3, int(round(L_mm / (0.75 * w_st))))
            slip = bool(cd["cord"].get("slip_back", True))
            c = cd.get("colour", "black")
            k = 2 if e.get("mirror") else 1
            extra_m[c] = extra_m.get(c, 0) + k * (per_st_m * chains * (1.6 if slip else 0.6) + 0.35)
            fine = {"dir": [float(x) for x in _norm3(e.get("dir", parent.point(at.get("round", "end"),
                                                                                at.get("az", 0.0))[1]))]}
            for k_ in ("thickness_mm", "curl_deg", "link_mm"):
                if k_ in cd["cord"]:
                    fine[k_] = cd["cord"][k_]
            tw_ = e.get("toward", cd["cord"].get("toward"))
            if tw_ is not None:
                fine["toward"] = [float(x) for x in tw_]
            steps.append({**common, "kind": "cord", "on": parent.p.id, "fine": fine,
                          "at": position(parent, at.get("round", "end"), at.get("az", 0.0)),
                          "chains": chains, "length_mm": L_mm, "colour": c, "slip_back": slip})
            continue
        inst = insts[e["name"]]
        pdef = inst.p
        st = {**common, "closed": pdef.closed,
              "seam_flat": e.get("seam") == "flat" and e.get("attach", "end") == "end",
              "squash": pdef.squash or (float(e.get("pinch", 0.8)) if e.get("seam") == "flat" else 0.0),
              "flatten": pdef.flatten, "stuff": pdef.stuff}
        if pdef.spoon > 0:
            st["spoon"] = True
        if "on" not in e:
            st["kind"] = "root"
            st["fine"] = {k: e[k] for k in ("axis", "pos") if k in e}
            st["start_at"] = _dominant(-inst.R[:, 2])
            steps.append(st); continue
        parent = _inst_on(toy, e["on"])
        st["on"] = parent.p.id
        at = e.get("at", {"round": "end"})
        attach = e.get("attach", "end")
        ths = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        if attach == "edge":
            st["kind"] = "by_edge"
            t_rim = pdef.dish_rim_t() if pdef.dish > 0 else float(pdef.N)
            r_rim = min(int(np.ceil(t_rim)), pdef.N) if pdef.dish > 0 else pdef.N
            rim = inst.base(np.full_like(ths, t_rim), ths)
            sd, _, _ = parent.signed_distance(rim)
            m = sd <= sd.min() + 0.5 * w_st
            if pdef.dish > 0:            # a dish pressed into the head: the rim stitches touching it are sewn
                m = sd <= max(sd.min() + 0.5 * w_st, 0.6 * w_st)
            st["edge_stitches"] = max(2, int(round(m.mean() * pdef.counts[r_rim - 1])))
            st["edge_round"] = pdef.counts[r_rim - 1]
            st["dish"] = pdef.dish > 0
            st["rim_round"] = r_rim
            if getattr(pdef, "motif", None) is not None:
                st["motif"] = pdef.kind; st["sew"] = pdef.sew
            P, _, T = inst.samples(per_round=3, per_stitch=1.0)
            sd, _, _ = parent.signed_distance(P)
            m = sd <= np.quantile(sd, 0.04)
            t, az, _ = loc.locate(parent, P[m])
            st["footprint"] = _footprint(parent, t, az)
        elif e.get("around"):
            st["kind"] = "around"
            r_ = at.get("round", "end")
            st["ring_round"] = parent.p.N if r_ == "end" else float(r_)
            st["ring_n"] = pdef.counts[0]
            st["split"] = bool(getattr(pdef, "split", False))
            st["at"] = position(parent, at.get("round", "end"), 0.0)
            st["dir"] = _direction(inst, {"attach": "start"})
            st["fine"] = {}
            if "dir" in e:              # (default: straight down along the parent's own axis — nothing to state)
                st["fine"]["dir"] = [float(x) for x in inst.R[:, 2]]
            if "front" in e or e.get("roll"):
                st["fine"]["front"] = [float(x) for x in inst.R[:, 0]]
            if "shift" in e:
                st["fine"]["shift"] = float(e["shift"])
            st["fall"] = bool(e.get("fall"))
            st["ring_end"] = r_ == "end"
            st["start_at"] = _dominant(-inst.R[:, 2])
            steps.append(st)
            continue
        elif e.get("wear"):
            # put on like a hat: which of its rounds rest on the parent (sewn there), the stuffing under it, and
            # how far its lower edge stands off the parent (tacked with loose stitches of that length)
            st["kind"] = "worn"
            pad = float(e.get("pad", 1.0))
            P_, _, T_ = inst.samples(per_round=3, per_stitch=1.0)
            sd, _, _ = parent.signed_distance(P_)
            near = sd <= pad + float(pdef.thick) + 1.0
            if not near.any():
                near = sd <= np.quantile(sd, 0.03)
            rr = np.clip(np.floor(T_[near]).astype(int) + 1, 1, pdef.N)
            st["rest_rounds"] = [int(rr.min()), int(rr.max())]
            t, az, _ = loc.locate(parent, P_[near])
            st["footprint"] = _footprint(parent, t, az)
            ths_ = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            tr_ = np.full_like(ths_, float(pdef.N))
            keep = ~pdef.in_opening(tr_ - 0.5, ths_)
            rim = inst.base(tr_[keep], ths_[keep])
            st["edge_gap_mm"] = max(0, int(round(float(np.median(parent.signed_distance(rim)[0])))))
            top = inst.base(np.array([0.0]), np.array([0.0]))
            st["top_gap_mm"] = max(0, int(round(float(parent.signed_distance(top)[0][0]))))
            st["opening"] = (pdef.opening or {}).get("stitches")
            st["pad"] = pad
            ew = dict(e, attach="start")
            st["at"] = position(parent, at.get("round", "end"), at.get("az", 0.0))
            st["dir"] = _direction(inst, ew)
            st["fine"] = _fine(inst, ew, "start")
            st["fine"]["pad"] = pad
            st["start_at"] = _dominant(-inst.R[:, 2])
            steps.append(st)
            continue
        elif e.get("applique"):
            st["kind"] = "applique"
            if getattr(pdef, "motif", None) is not None:
                st["motif"] = pdef.kind
            if e.get("drape"):
                st["drape"] = True
            if e.get("anchor") in ("eyes", "eye"):
                st["anchor"] = e["anchor"]
            at = e.get("_at", at)
            rim = inst.base(np.full_like(ths, float(pdef.N)), ths)
            t, az, _ = loc.locate(parent, rim)
            st["footprint"] = _footprint(parent, t, az, child_n=pdef.counts[-1])
            if e.get("offset_mm") and "_at" not in e:
                tc, azc, _ = loc.locate(parent, inst.origin)
                at = {"round": _round_of_t(tc[0]), "az": float(azc[0])}
        elif at.get("round") == "end" and not parent.p.closed and attach == "end" and pdef.closed:
            st["kind"] = "onto_opening"
            rim = parent.base(np.full_like(ths, float(parent.p.N)), ths)
            t, az, _ = loc.locate(inst, rim)
            st["child_footprint"] = _footprint(inst, t, az)
            st["parent_round"] = parent.p.N
        elif not pdef.closed and attach == "end":
            st["kind"] = "by_opening"
            rim = inst.base(np.full_like(ths, float(pdef.N)), ths)
            t, az, _ = loc.locate(parent, rim)
            st["footprint"] = _footprint(parent, t, az, child_n=pdef.counts[-1])
            st["opening"] = pdef.counts[-1]
            if e.get("rim_arc"):
                # sewn by part of its rim only: how many stitches, and which way that part faces on the toy
                a0, a1 = (float(x) for x in e["rim_arc"])
                n_ = pdef.counts[-1]
                k_ = np.arange(n_)
                th_ = 2 * np.pi * (k_ + 0.5 - LEAN * (pdef.N - 0.5)) / n_
                az_ = (np.asarray(inst.az_of_theta(th_)) + 180) % 360 - 180
                mid, half = (a0 + a1) / 2, ((a1 - a0) % 360) / 2
                inside = np.abs((az_ - mid + 180) % 360 - 180) <= half
                P_ = inst.base(np.full(int(inside.sum()), float(pdef.N)), th_[inside])
                c_ = P_.mean(0) - inst.base(np.full(n_, float(pdef.N)), th_).mean(0)
                st["rim_arc"] = {"az": [a0, a1], "stitches": int(inside.sum()), "faces": _dominant(c_)}
            if st["seam_flat"] or st["squash"] > 0 or st["flatten"] >= 0.5:
                st["seam"] = _seam_info(inst, parent, e)
        else:
            st["kind"] = "closed_end" if attach == "end" else "by_start"
            P, _, T = inst.samples(per_round=3, per_stitch=1.0)
            sd, _, _ = parent.signed_distance(P)
            m = sd < 0.5
            if m.sum() < 5:
                m = sd <= np.quantile(sd, 0.05)
            t, az, _ = loc.locate(parent, P[m])
            st["footprint"] = _footprint(parent, t, az)
        st["at"] = position(parent, at.get("round", "end"), at.get("az", 0.0))
        st["dir"] = _direction(inst, e)
        st["fine"] = _fine(inst, e, attach, applique=bool(e.get("applique")))
        st["start_at"] = _dominant(-inst.R[:, 2])
        steps.append(st)

    # face ---------------------------------------------------------------------------------------------------
    face_steps = []
    for f, hosts, variants in face:
        host = hosts[0]
        it = {"type": f["type"], "on": host.p.id, "pair": len(hosts) == 2, "colour": f.get("colour", "black"),
              "on_side": side_of(toy, host.name) if len(hosts) == 1 else None,
              "on_pair": len(hosts) == 2 and hosts[0] is not hosts[1]}
        if len(hosts) == 1 and not it["on_side"] and                 sum(1 for o in toy.inst.values() if o.p.id == host.p.id) > 1:
            it["on_inst"] = host.name        # one of several copies (three honeycomb cells): say which one
        if "at" in f:
            it["at"] = position(host, f["at"]["round"], f["at"].get("az", 0.0))
            if len(hosts) == 2:
                n = it["at"]["n"]
                it["apart"] = int(round(2 * abs(float(f["at"].get("az", 0.0))) / 360 * n))
        if "points" in f:
            it["points"] = [position(host, r, a) for r, a in f["points"]]
            azs = [((float(a) + 180) % 360 - 180) for r, a in f["points"]]
            it["region"] = ("front" if all(abs(a) <= 90 for a in azs) else
                            "back" if all(abs(a) >= 90 for a in azs) else "mixed")
        for k in ("size_mm", "shape", "width_st", "height_rounds", "width_mm", "height_mm", "thickness_mm", "rotate",
                  "count", "length_mm", "colours", "curl", "droop", "offset_mm", "lift_mm", "highlight", "role",
                  "aspect", "sink", "iris", "pupil", "strands", "strength", "spread", "seed", "doubled", "link_mm"):
            if k in f:
                it[k] = f[k]
        k_ = len(hosts)
        if f["type"] == "strap":
            L_mm = _path_length(host, f["points"])
            it["length_mm"] = round(L_mm, 0)
            it["chains"] = max(3, int(math.ceil(L_mm / (0.75 * w_st))))
            c = f.get("colour", "black")
            extra_m[c] = extra_m.get(c, 0) + k_ * (per_st_m * it["chains"] * 1.6 + 0.35)
        if f["type"] == "fringe":
            cols = f.get("colours") or [f.get("colour", "black")]
            n_ = int(f.get("count", 12)); L_ = float(f.get("length_mm", 20))
            for i in range(n_):
                c = cols[i % len(cols)]
                extra_m[c] = extra_m.get(c, 0) + k_ * (2 * L_ + 20) / 1000.0
        face_steps.append(it)

    # materials ----------------------------------------------------------------------------------------------
    per_st = 6.0 * float(yarn["stitch_mm"]) / 1000.0          # metres of yarn per single crochet (estimate)
    tails = {}
    for inst in insts.values():
        c = toy.chart(inst)[-1][0]
        tails[c] = tails.get(c, 0) + (0.15 if inst.p.closed else 0.35)    # closing tail / sewing tail
        if (entry_of.get(inst.name, {}).get("attach") == "edge" and not inst.p.dish > 0
                and getattr(inst.p, "motif", None) is None):
            tails[inst.p.colour] = tails.get(inst.p.colour, 0) + 0.15      # tail of the back circle
    colours = []
    hexes = dict(spec.get("colours", {}))
    hexes.setdefault("black", "#151313"); hexes.setdefault("white", "#f4f1ea")
    for c in set(extra_m) | set(stitches_by_colour):
        stitches_by_colour.setdefault(c, 0)
    for c, n in sorted(stitches_by_colour.items(), key=lambda kv: -(kv[1] + 100 * extra_m.get(kv[0], 0))):
        m = (n * per_st + tails.get(c, 0) + extra_m.get(c, 0)) * 1.15
        colours.append({"name": c, "hex": hexes.get(c, ""), "stitches": n,
                        "extra_m": round(extra_m.get(c, 0), 2),
                        "metres": int(math.ceil(m / 5.0) * 5)})
    hardware = []
    for f, hosts, _ in face:
        if f["type"] in ("safety_eye", "eye", "safety_nose", "felt", "blush", "stitch", "embroidery", "line",
                         "satin"):
            hardware.append({"type": f["type"], "count": len(hosts), "size_mm": f.get("size_mm"),
                             "shape": f.get("shape"), "colour": f.get("colour", "black"),
                             "highlight": bool(f.get("highlight"))})

    return {
        "name": spec.get("name", ""), "title": spec.get("title", spec.get("name", "Toy")),
        "title_ru": spec.get("title_ru"),
        "size": _toy_size(toy), "yarn": dict(yarn), "colours": colours, "hardware": hardware,
        "yarn_per_stitch_m": round(per_st, 4), "parts": parts, "assembly": steps, "face": face_steps,
        "hex": hexes,
        "flags": list(toy.flags),
    }


# ---------------------------------------------------------------------------------------------------------------
# rendering

DIR_RU = {"up": "вверх", "down": "вниз", "front": "вперёд", "back": "назад", "left": "к левому боку игрушки",
          "right": "к правому боку игрушки", "out": "наружу", "in": "внутрь"}
DIR_EN = {"up": "up", "down": "down", "front": "forward", "back": "backward", "left": "toward the toy's left",
          "right": "toward the toy's right", "out": "outward", "in": "inward"}
LINE_RU = {"front": "центральной линии спереди", "back": "центральной линии сзади", "up": "центральной линии сверху",
           "down": "центральной линии снизу", "left": "линии на левом боку", "right": "линии на правом боку"}
LINE_EN = {"front": "centre front line", "back": "centre back line", "up": "centre top line",
           "down": "centre bottom line", "left": "line on the toy's left side", "right": "line on the toy's right side"}
SIDE_RU = {"left": "к левому боку игрушки", "right": "к правому боку игрушки", "up": "вверх", "down": "вниз",
           "front": "вперёд", "back": "назад"}
SIDE_EN = {"left": "toward the toy's left", "right": "toward the toy's right", "up": "upward", "down": "downward",
           "front": "toward the front", "back": "toward the back"}
WHERE_RU = {"up": "сверху", "down": "снизу", "front": "спереди", "back": "сзади", "left": "на левом боку",
            "right": "на правом боку"}
WHERE_EN = {"up": "at the top", "down": "at the bottom", "front": "at the front", "back": "at the back",
            "left": "on the toy's left side", "right": "on the toy's right side"}
# Russian part names: nominative, accusative, genitive, dative, prepositional
PART_RU = {
    "head": ("голова", "голову", "головы", "голове", "голове"),
    "body": ("тело", "тело", "тела", "телу", "теле"),
    "muzzle": ("мордочка", "мордочку", "мордочки", "мордочке", "мордочке"),
    "snout": ("мордочка", "мордочку", "мордочки", "мордочке", "мордочке"),
    "ear": ("ушко", "ушко", "ушка", "ушку", "ушке"),
    "outer_ear": ("ушко", "ушко", "ушка", "ушку", "ушке"),
    "inner_ear": ("внутренняя часть ушка", "внутреннюю часть ушка", "внутренней части ушка",
                  "внутренней части ушка", "внутренней части ушка"),
    "arm": ("ручка", "ручку", "ручки", "ручке", "ручке"),
    "hand": ("ладошка", "ладошку", "ладошки", "ладошке", "ладошке"),
    "leg": ("ножка", "ножку", "ножки", "ножке", "ножке"),
    "foot": ("стопа", "стопу", "стопы", "стопе", "стопе"),
    "paw": ("лапка", "лапку", "лапки", "лапке", "лапке"),
    "tail": ("хвостик", "хвостик", "хвостика", "хвостику", "хвостике"),
    "nose": ("носик", "носик", "носика", "носику", "носике"),
    "beak": ("клюв", "клюв", "клюва", "клюву", "клюве"),
    "bill": ("клюв", "клюв", "клюва", "клюву", "клюве"),
    "wing": ("крылышко", "крылышко", "крылышка", "крылышку", "крылышке"),
    "belly": ("животик", "животик", "животика", "животику", "животике"),
    "cheek": ("щёчка", "щёчку", "щёчки", "щёчке", "щёчке"),
    "hat": ("шапочка", "шапочку", "шапочки", "шапочке", "шапочке"),
    "mane": ("грива", "гриву", "гривы", "гриве", "гриве"),
    "horn": ("рожок", "рожок", "рожка", "рожку", "рожке"),
    "fin": ("плавник", "плавник", "плавника", "плавнику", "плавнике"),
    "eye_patch": ("пятно вокруг глаза", "пятно вокруг глаза", "пятна вокруг глаза", "пятну вокруг глаза",
                  "пятне вокруг глаза"),
    "patch": ("пятно", "пятно", "пятна", "пятну", "пятне"),
    "spot": ("пятнышко", "пятнышко", "пятнышка", "пятнышку", "пятнышке"),
    "bow": ("бантик", "бантик", "бантика", "бантику", "бантике"),
    "scarf": ("шарфик", "шарфик", "шарфика", "шарфику", "шарфике"),
    "flipper": ("ласт", "ласт", "ласта", "ласту", "ласте"),
    "thigh": ("бедро", "бедро", "бедра", "бедру", "бедре"),
    "haunch": ("бедро", "бедро", "бедра", "бедру", "бедре"),
    "toe": ("пальчик", "пальчик", "пальчика", "пальчику", "пальчике"),
    "sole": ("подошва", "подошву", "подошвы", "подошве", "подошве"),
    "front_leg": ("передняя лапка", "переднюю лапку", "передней лапки", "передней лапке", "передней лапке"),
    "hind_leg": ("задняя лапка", "заднюю лапку", "задней лапки", "задней лапке", "задней лапке"),
    "back_leg": ("задняя лапка", "заднюю лапку", "задней лапки", "задней лапке", "задней лапке"),
    "eyebrow": ("бровка", "бровку", "бровки", "бровке", "бровке"),
    "hind_foot": ("задняя лапка", "заднюю лапку", "задней лапки", "задней лапке", "задней лапке"),
    "bean": ("подушечка лапки", "подушечку лапки", "подушечки лапки", "подушечке лапки", "подушечке лапки"),
    "toe_bean": ("подушечка лапки", "подушечку лапки", "подушечки лапки", "подушечке лапки", "подушечке лапки"),
    "feather": ("пёрышко", "пёрышко", "пёрышка", "пёрышку", "пёрышке"),
    "shell": ("панцирь", "панцирь", "панциря", "панцирю", "панцире"),
    "dorsal": ("спинной плавник", "спинной плавник", "спинного плавника", "спинному плавнику", "спинном плавнике"),
    "earcup": ("наушник", "наушник", "наушника", "наушнику", "наушнике"),
    "band": ("ободок", "ободок", "ободка", "ободку", "ободке"),
    "headband": ("ободок", "ободок", "ободка", "ободку", "ободке"),
    "strap": ("лямка", "лямку", "лямки", "лямке", "лямке"),
    "collar": ("воротничок", "воротничок", "воротничка", "воротничку", "воротничке"),
    "neck": ("шея", "шею", "шеи", "шее", "шее"),
    "torso": ("тело", "тело", "тела", "телу", "теле"),
    "face": ("личико", "личико", "личика", "личику", "личике"),
    "mouth": ("ротик", "ротик", "ротика", "ротику", "ротике"),
    "pad": ("подушечка", "подушечку", "подушечки", "подушечке", "подушечке"),
    "claw": ("коготок", "коготок", "коготка", "коготку", "коготке"),
    "lobe": ("лопасть", "лопасть", "лопасти", "лопасти", "лопасти"),
    "fluke": ("лопасть хвоста", "лопасть хвоста", "лопасти хвоста", "лопасти хвоста", "лопасти хвоста"),
    "hoof": ("копытце", "копытце", "копытца", "копытцу", "копытце"),
    "antenna": ("усик", "усик", "усика", "усику", "усике"),
    "leaf": ("листик", "листик", "листика", "листику", "листике"),
    "stem": ("стебелёк", "стебелёк", "стебелька", "стебельку", "стебельке"),
    "button": ("пуговка", "пуговку", "пуговки", "пуговке", "пуговке"),
    "heart": ("сердечко", "сердечко", "сердечка", "сердечку", "сердечке"),
    "star": ("звёздочка", "звёздочку", "звёздочки", "звёздочке", "звёздочке"),
    "petal": ("лепесток", "лепесток", "лепестка", "лепестку", "лепестке"),
    "clover": ("клевер", "клевер", "клевера", "клеверу", "клевере"),
    "flower": ("цветочек", "цветочек", "цветочка", "цветочку", "цветочке"),
    "cloud": ("облачко", "облачко", "облачка", "облачку", "облачке"),
    "crest": ("хохолок", "хохолок", "хохолка", "хохолку", "хохолке"),
    "comb": ("гребешок", "гребешок", "гребешка", "гребешку", "гребешке"),
    "hair": ("причёска", "причёску", "причёски", "причёске", "причёске"),
    "fringe": ("чёлка", "чёлку", "чёлки", "чёлке", "чёлке"),
    "blanket": ("одеяльце", "одеяльце", "одеяльца", "одеяльцу", "одеяльце"),
    "bib": ("нагрудник", "нагрудник", "нагрудника", "нагруднику", "нагруднике"),
    "skirt": ("юбочка", "юбочку", "юбочки", "юбочке", "юбочке"),
    "backpack": ("рюкзачок", "рюкзачок", "рюкзачка", "рюкзачку", "рюкзачке"),
    "chest": ("грудка", "грудку", "грудки", "грудке", "грудке"),
    "mask": ("маска", "маску", "маски", "маске", "маске"),
    "stripe": ("полоска", "полоску", "полоски", "полоске", "полоске"),
    "whisker": ("усик", "усик", "усика", "усику", "усике"),
    "eyeball": ("глазик", "глазик", "глазика", "глазику", "глазике"),
}
# qualifiers in part / instance ids: id word -> (masc, fem, neut) nominative adjective
QUAL_RU = {
    "front": ("передний", "передняя", "переднее"), "back": ("задний", "задняя", "заднее"),
    "hind": ("задний", "задняя", "заднее"), "rear": ("задний", "задняя", "заднее"),
    "top": ("верхний", "верхняя", "верхнее"), "upper": ("верхний", "верхняя", "верхнее"),
    "low": ("нижний", "нижняя", "нижнее"), "lower": ("нижний", "нижняя", "нижнее"),
    "bottom": ("нижний", "нижняя", "нижнее"), "inner": ("внутренний", "внутренняя", "внутреннее"),
    "outer": ("внешний", "внешняя", "внешнее"), "side": ("боковой", "боковая", "боковое"),
    "small": ("маленький", "маленькая", "маленькое"), "big": ("большой", "большая", "большое"),
    "mid": ("средний", "средняя", "среднее"), "middle": ("средний", "средняя", "среднее"),
    "left": ("левый", "левая", "левое"), "right": ("правый", "правая", "правое"),
}
PART_OVERRIDE = {}      # part id -> name from the spec ("name_ru" / "name_en"), set per toy in write_patterns
_CASES = {"nom": 0, "acc": 1, "gen": 2, "dat": 3, "prep": 4}
_ADJ_END = ("ый", "ий", "ой", "ая", "яя", "ое", "ее")


def _is_adj(w):
    return w.split("-")[-1].endswith(_ADJ_END) and len(w) > 3


def ru_gender(phrase):
    """'m' / 'f' / 'n' of a Russian noun phrase in the nominative (from its first adjective or head noun)."""
    for w in phrase.split(" "):
        w = w.strip("()").split("-")[-1]
        if not w:
            continue
        if _is_adj(w):
            return {"ая": "f", "яя": "f", "ое": "n", "ее": "n"}.get(w[-2:], "m")
        if w in ("деталь", "лопасть", "часть", "кисть", "бровь", "морковь"):
            return "f"
        if w.endswith(("а", "я")):
            return "f"
        if w.endswith(("о", "е", "ё")):
            return "n"
        return "m"
    return "f"


def _decl_adj(w, g, case):
    if case == "nom" or (g == "m" and case == "acc"):
        return w
    pre, _, a = w.rpartition("-")
    pre = pre + "-" if pre else ""
    stem, end = a[:-2], a[-2:]
    soft = end in ("ий", "яя", "ее") and not stem.endswith(("к", "г", "х"))
    hush = stem.endswith(("ж", "ш", "ч", "щ"))
    if g == "f":
        forms = {"acc": "юю" if end == "яя" else "ую", "gen": "ей" if (end == "яя" or (soft and end != "ая")) else "ой"}
        forms["dat"] = forms["prep"] = forms["gen"]
    else:
        # after ж/ш/ч/щ a stressed ending stays 'о' ('большое' -> 'большого'), an unstressed one is 'е' ('свежее')
        o = "е" if (soft or (hush and end in ("ий", "ее"))) else "о"
        forms = {"acc": a[-2:] if g == "n" else end, "gen": o + "го", "dat": o + "му", "prep": o + "м"}
        if g == "m" and end == "ой":
            forms.update(gen="ого", dat="ому", prep="ом")
    return pre + stem + forms[case]


def _decl_noun(w, case):
    if case == "nom":
        return w
    if w in ("деталь", "лопасть", "часть", "кисть"):
        return w if case == "acc" else w[:-1] + "и"
    if w.endswith("а"):
        s = w[:-1]
        return s + {"acc": "у", "gen": "и" if s.endswith(tuple("гкхжшчщ")) else "ы", "dat": "е", "prep": "е"}[case]
    if w.endswith("ия"):
        return w[:-1] + {"acc": "ю", "gen": "и", "dat": "и", "prep": "и"}[case]
    if w.endswith("я"):
        return w[:-1] + {"acc": "ю", "gen": "и", "dat": "е", "prep": "е"}[case]
    if w.endswith("о"):
        return w[:-1] + {"acc": "о", "gen": "а", "dat": "у", "prep": "е"}[case]
    if w.endswith("е") and w[-2:-1] in "жшщчц":
        # туловище, сердце: after a hissing consonant or ц the neuter endings are -а / -у (туловища, сердца)
        return w[:-1] + {"acc": "е", "gen": "а", "dat": "у", "prep": "е"}[case]
    if w.endswith("е"):
        return w[:-1] + {"acc": "е", "gen": "я", "dat": "ю", "prep": "и" if w.endswith("ие") else "е"}[case]
    if w.endswith(("ь", "й")):
        return w if case == "acc" else w[:-1] + {"gen": "я", "dat": "ю", "prep": "е"}[case]
    if w[-1:] in "ыиуюэ" or not w[-1:].isalpha():
        return w
    if case == "acc":
        return w
    return _fleeting_stem(w) + {"gen": "а", "dat": "у", "prep": "е"}[case]


_NOT_FLEETING = {"урок", "восток", "поток", "игрок", "порок", "чеснок", "пророк", "челнок", "курок", "человек",
                 "намёк", "намек", "упрёк", "упрек"}


def _fleeting_stem(w):
    """The oblique-case stem of a masculine noun with a fleeting vowel: бугорок -> бугорк(а), белок -> белк(а),
    огонёк -> огоньк(а), палец -> пальц(а); other nouns are returned unchanged."""
    lw = w.lower()
    if lw in _NOT_FLEETING or sum(ch in "аеёиоуыэюя" for ch in lw) < 2:
        return w
    if lw.endswith("ок"):
        return w[:-2] + "к"
    if lw.endswith(("ёк", "ек")):
        p = lw[-3]
        return w[:-2] + ("йк" if p in "аеёиоуыэюя" else "к" if p in "жшчщ" else "ьк")
    if lw.endswith("ец"):
        p = lw[-3]
        return w[:-2] + ("йц" if p in "аеёиоуыэюя" else "ьц" if p == "л" else "ц")
    return w


def ru_decline(phrase, case):
    """Decline a Russian noun phrase given in the nominative: leading adjectives and the head noun (a hyphenated
    'голова-тело' declines both halves); whatever follows the head noun (a genitive, a note in brackets) stays."""
    if case == "nom":
        return phrase
    head, note = _split_note(phrase)
    if " и " in head:                     # 'голова и тело (одной деталью)' -> 'головы и тела (одной деталью)'
        return " и ".join(ru_decline(x, case) for x in head.split(" и ")) + note
    g = ru_gender(phrase)
    out = []
    done = False
    post = False              # adjectives right after the head noun agree with it: 'сегмент жёлтый' -> 'сегмента жёлтого'
    for w in phrase.split(" "):
        if done and post and w and not w.startswith("(") and _is_adj(w):
            out.append(_decl_adj(w, g, case)); continue
        post = False
        if done or w.startswith("(") or not w:
            out.append(w); done = done or bool(w); continue
        if _is_adj(w):
            out.append(_decl_adj(w, g, case)); continue
        if "-" in w and not _is_adj(w):
            out.append("-".join(_decl_noun(x, case) for x in w.split("-")))
        else:
            out.append(_decl_noun(w, case))
        done = True
        post = True
    return " ".join(out)


def _derive_ru(pid):
    """A Russian name for an id the dictionary does not know: 'hind_foot' -> 'задняя стопа', 'toe_a' -> toe,
    't_leg' -> leg; None when nothing is recognisable."""
    words = [w for w in pid.lower().split("_") if w]
    while words and (len(words[-1]) == 1 or words[-1].isdigit() or words[-1] in ("l", "r")):
        words.pop()
    words = [w.rstrip("0123456789") for w in words]
    if not words:
        return None
    if "_".join(words) in PART_RU:
        return PART_RU["_".join(words)][0]
    base = None
    for i in range(len(words)):
        k = "_".join(words[i:])
        if k in PART_RU:
            base = PART_RU[k][0]; quals = words[:i]
            break
    else:
        return None
    g = ru_gender(base)
    adj = [QUAL_RU[q][{"m": 0, "f": 1, "n": 2}[g]] for q in quals if q in QUAL_RU]
    return " ".join(adj + [base])


def part_name(pid, lang, case="nom", cap=False):
    ov = PART_OVERRIDE.get((pid, lang))
    if ov:
        w = ru_decline(ov, case) if lang == "ru" else ov
    elif lang == "en":
        w = pid.replace("_", " ")
    elif pid in PART_RU:
        w = PART_RU[pid][_CASES[case]]
    else:
        d = _derive_ru(pid)
        w = ru_decline(d, case) if d else ru_decline("деталь", case) + f" {pid.replace('_', ' ')}"
    return w[:1].upper() + w[1:] if cap else w


def qualifier(name, pid, lang, side_ok=True, side=None):
    """What tells this instance from the other copies of its part, as an adjective agreeing with the part's name
    ('левая', 'передняя', '№ 2'), from the instance name (arm_L, front_leg_R, tail_top, toe_b, bean1) and its side
    in a mirrored pair ('L' / 'R', from Toy.pairs; else read from an _L / _R suffix)."""
    base = name
    if base.endswith(("_L", "_R")):
        if side is None:
            side = base[-1]
        base = base[:-2]
    side = ({"L": "left", "R": "right"}.get(side) if side_ok else None)
    words = [w for w in base.lower().split("_") if w]
    if len(words) > 1 and words[-1] in ("l", "r"):          # 'honeycomb_l': a side, not the 12th letter
        words[-1] = "left" if words[-1] == "l" else "right"
    pw = pid.lower().split("_")
    quals = [w for w in words if w in QUAL_RU and w not in pw and (side_ok or w not in ("left", "right"))]
    num = None
    rest = [w for w in words if w not in pw and w not in QUAL_RU]
    for w in rest:
        if len(w) == 1 and w.isalpha():
            num = ord(w) - ord("a") + 1
        elif w.isdigit():
            num = int(w) + 1
    if words and words[-1].rstrip("0123456789") != words[-1] and words[-1].rstrip("0123456789") in pw:
        num = int(words[-1][len(words[-1].rstrip("0123456789")):]) + 1
    if lang == "en":
        bits = quals[:]
        if side:
            bits.append(side)
        if num:
            bits.append(f"no. {num}")
        return ", ".join(bits)
    g = {"m": 0, "f": 1, "n": 2}[ru_gender(part_name(pid, "ru"))]
    bits = [QUAL_RU[q][g] for q in quals]
    if side:
        bits.append(QUAL_RU[side][g])
    if num:
        bits.append(f"№ {num}")
    return ", ".join(bits)


def _rounds_ru(n):
    return f"{n} {_plural_ru(n, 'ряд', 'ряда', 'рядов')}"


def pos_text(pos, lang):
    """'between rounds 13 and 14, 3 st to the toy's left of the centre front line' (counted on round 13)."""
    if "pole" in pos:
        if lang == "ru":
            return "в центре последнего ряда" if pos["pole"] == "end" else "в центре кольца амигуруми (ряд 1)"
        return "at the centre of the last round" if pos["pole"] == "end" else "at the centre of the magic ring"
    if "between" in pos:
        a, b = pos["between"]
        rr = f"между {a} и {b} рядами" if lang == "ru" else f"between rounds {a} and {b}"
    else:
        rr = f"в {pos['round']}-м ряду" if lang == "ru" else f"in round {pos['round']}"
    k = pos["stitches"]
    line = (LINE_RU if lang == "ru" else LINE_EN)[pos["from"]]
    k1 = round(k, 2)                # the exact offset (hundredths: the reverse reader rebuilds the same seam)
    kh = round(k * 2) / 2           # what a crocheter counts: whole or half stitches
    bits = []
    if abs(k1 - kh) > 0.005 and pos.get("side"):
        bits.append(f"st {_num(k1, 'en', 2)} {pos.get('side')}")
    # the exact round (23.2 is written "in round 23", 13.4 "between rounds 13 and 14")
    shown = (pos["between"][0] + 0.5) if "between" in pos else float(pos.get("round", 0))
    er = pos.get("exact_round")
    if er is not None and abs(float(er) - shown) > 0.005:
        bits.append(f"rnd {_num(float(er), 'en', 3)}")
    hidden = f" <!-- {' '.join(bits)} -->" if bits else ""
    if kh < 0.25:
        st = f"на {line}" if lang == "ru" else f"on the {line}"
    else:
        ks = _num(kh, lang) if kh != int(kh) else str(int(kh))
        side = (SIDE_RU if lang == "ru" else SIDE_EN)[pos["side"]]
        st = (f"в {ks} п. от {line} {side}" if lang == "ru" else f"{ks} st from the {line}, {side}")
    return f"{rr}, {st}{hidden}"


def _dir_text(comps, lang):
    if not comps:
        return ""
    W = DIR_RU if lang == "ru" else DIR_EN
    first = comps[0]
    words = [W[first[0]]]
    for name, c in comps[1:]:
        slight = c < 0.6 * first[1]
        if lang == "ru":
            words.append(("немного " if slight else "") + W[name])
        else:
            words.append(("slightly " if slight else "") + W[name])
    if lang == "ru":                       # 'наружу, вниз и немного вперёд'
        return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " и " + words[-1]
    return " and ".join(words)


def ang_text(v, lang):
    """A direction a maker can set with a protractor: '25° down, turned 40° to the toy's left' (heading measured
    from straight ahead, i.e. from the toy's front)."""
    v = _norm3(v)
    el = math.degrees(math.asin(float(np.clip(v[1], -1, 1))))
    ru = lang == "ru"
    if el >= 89.95:
        return "прямо вверх" if ru else "straight up"
    if el <= -89.95:
        return "прямо вниз" if ru else "straight down"
    hd = math.degrees(math.atan2(float(v[0]), float(v[2])))
    # tenths of a degree: these words only appear in the hidden "Exact placement" comments, which the reverse
    # reader turns back into vectors (whole degrees moved a hanging ear's tip by millimetres)
    e_ = round(abs(el), 1)
    h_ = round(abs(hd), 1)
    es, hs = _num(e_, lang), _num(h_, lang)
    if ru:
        et = "горизонтально" if e_ == 0 else f"{es}° {'вверх' if el > 0 else 'вниз'}"
        ht = ("прямо вперёд" if h_ == 0 else "прямо назад" if h_ == 180 else
              f"с поворотом на {hs}° к {'левому' if hd > 0 else 'правому'} боку игрушки")
        return f"{et}, {ht}"
    et = "level" if e_ == 0 else f"{es}° {'up' if el > 0 else 'down'}"
    ht = ("straight ahead" if h_ == 0 else "straight back" if h_ == 180 else
          f"turned {hs}° to the toy's {'left' if hd > 0 else 'right'}")
    return f"{et}, {ht}"


# safety eyes are sold in these sizes (mm); a size between two is written as both, so the maker can buy it
EYE_SIZES = (5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 20, 24, 30)


def _eye_size_text(mm, lang):
    """'12' / '12–13' (with the exact size kept for the reader: '12,5' is not a size a shop sells)."""
    mm = float(mm)
    if any(abs(mm - x) < 0.05 for x in EYE_SIZES):
        return str(int(round(mm)))
    lo = max([x for x in EYE_SIZES if x < mm], default=EYE_SIZES[0])
    hi = min([x for x in EYE_SIZES if x > mm], default=EYE_SIZES[-1])
    return f"{lo}–{hi}<!-- {_num(mm, 'en')} mm -->"


def _mm(x, lang):
    return f"{_num(round(float(x), 1), lang)} мм" if lang == "ru" else f"{_num(round(float(x), 1), lang)} mm"


def _exact(bits, lang, en_label="Exact placement"):
    """The machine-exact values (angles, mm, %) as an HTML comment: a person reading the pattern does not see them
    (the site and the PDF drop comments), the reverse reader still rebuilds the identical toy from them."""
    if not bits:
        return ""
    return " <!-- " + ("Точно: " if lang == "ru" else f"{en_label}: ") + "; ".join(bits) + ". -->"


def _sink_hint(mm, lang, what="deep"):
    """A depth a crocheter can act on: only from 3 mm, in whole mm (or cm)."""
    mm = float(mm)
    if mm < 3:
        return ""
    v = f"{_num(round(mm / 10, 1), lang)} см" if mm >= 15 and lang == "ru" else (
        f"{_num(round(mm / 10, 1), lang)} cm" if mm >= 15 else f"{int(round(mm))} {'мм' if lang == 'ru' else 'mm'}")
    if what == "pad":
        return (f" Подложите под деталь немного набивки (слой ≈ {v})." if lang == "ru"
                else f" Put a little stuffing under it (≈ {v} thick).")
    return (f" Вдавите деталь в основу примерно на {v}." if lang == "ru"
            else f" Press it into the base by about {v}.")


def fine_text(st, lang):
    """' Exact placement: axis 25° down, ...; round-start line ...; pressed in 8 mm.' (empty if nothing to add)."""
    f = st.get("fine") or {}
    ru = lang == "ru"
    bits = []
    kind = st.get("kind")
    if kind == "root":
        if "axis" in f and [float(x) for x in f["axis"]] != [0.0, 1.0, 0.0]:
            bits.append(("ось (от кольца амигуруми к последнему ряду) " if ru else
                         "axis (from the magic ring to the last round) ") + ang_text(f["axis"], lang))
        if "pos" in f:
            bits.append(("смещение " if ru else "offset ") + ", ".join(_num(round(float(x), 1), lang) for x in f["pos"])
                        + (" мм" if ru else " mm"))
    elif kind == "by_edge":
        bits.append(("лицевая сторона смотрит " if ru else "front face looks ") + ang_text(f["face"], lang))
        bits.append(("деталь отходит " if ru else "it stands out ") + ang_text(f["dir"], lang))
        if "cup" in f:
            bits.append((f"край выгнут вперёд на {_mm(f['cup'], lang)}" if ru else
                         f"rim brought forward {_mm(f['cup'], lang)}"))
        if "sink" in f:
            bits.append((f"утоплена на {_mm(f['sink'], lang)}" if ru else f"pressed in {_mm(f['sink'], lang)}"))
    elif kind == "applique":
        bits.append(("верх детали (линия начала ряда) смотрит " if ru else "its top (round-start line) faces ")
                    + ang_text(f["front"], lang))
        if "pad" in f:
            bits.append((f"подложите набивку слоем ≈ {_mm(f['pad'], lang)}" if ru else
                         f"padded with stuffing {_mm(f['pad'], lang)} thick"))
        if "inset" in f:
            bits.append((f"утоплена в полотно на {_mm(f['inset'], lang)}" if ru else
                         f"sewn in {_mm(f['inset'], lang)} deep"))
        if "lift" in f:
            bits.append((f"приподнята на {_mm(f['lift'], lang)}" if ru else f"raised {_mm(f['lift'], lang)}"))
    elif kind == "flat":
        if "roll" in f:
            bits.append((f"повёрнута на {_num(f['roll'], lang)}°" if ru else f"turned {_num(f['roll'], lang)}°"))
        if "lift" in f:
            bits.append((f"приподнята на {_mm(f['lift'], lang)}" if ru else f"raised {_mm(f['lift'], lang)}"))
    elif kind == "cord":
        bits.append(("цепочка отходит " if ru else "the cord leaves ") + ang_text(f["dir"], lang))
        if "toward" in f:
            bits.append(("загибается " if ru else "curling toward ") + ang_text(f["toward"], lang))
        if "curl_deg" in f:
            bits.append((f"изгиб {_num(f['curl_deg'], lang)}°" if ru else f"curls {_num(f['curl_deg'], lang)}°"))
        if "thickness_mm" in f:
            bits.append((f"толщина {_mm(f['thickness_mm'], lang)}" if ru else
                         f"cord {_mm(f['thickness_mm'], lang)} thick"))
        if "link_mm" in f:
            bits.append((f"звено {_mm(f['link_mm'], lang)}" if ru else f"links {_mm(f['link_mm'], lang)}"))
    else:
        if "dir" in f:
            bits.append(("ось " if ru else "axis ") + ang_text(f["dir"], lang))
        if "front" in f:
            bits.append(("линия начала ряда смотрит " if ru else "round-start line faces ") + ang_text(f["front"], lang))
        if "seam_line" in f:
            bits.append(("линия шва " if ru else "seam line ") + ang_text(f["seam_line"], lang))
        if "pinch" in f or "pinch_rounds" in f:
            pc = int(round(100 * float(f.get("pinch", 0.8))))
            pr = float(f.get("pinch_rounds", 1.3))
            bits.append((f"сплющено на {pc} % на последних {_num(pr, lang)} рядах" if ru else
                         f"pinched {pc} % over the last {_num(pr, lang)} rounds"))
        if "bend" in f:
            bits.append(("внутренняя сторона изгиба смотрит " if ru else "inner side of the bend faces ")
                        + ang_text(f["bend"], lang))
        if "sink" in f:
            bits.append((f"вдавлена на {_mm(f['sink'], lang)}" if ru else f"pressed in {_mm(f['sink'], lang)}"))
        if "sink_dir" in f:
            bits.append(("вдавлена в направлении " if ru else "pressed toward ") + ang_text(f["sink_dir"], lang))
        if "conform_span" in f:
            bits.append((f"шов разглаживается на {_num(f['conform_span'], lang)} ряда" if ru else
                         f"seam blends over {_num(f['conform_span'], lang)} rounds"))
        if f.get("conform") is False:
            bits.append("край не прилегает к основе" if ru else "rim not pressed to the base")
        if f.get("conform_parent") is False:
            bits.append("край основы не подтягивается" if ru else "base rim not pulled in")
        if kind == "worn" and "pad" in f:
            bits.append((f"подложите набивку слоем ≈ {_mm(f['pad'], lang)}" if ru else
                         f"padded with stuffing {_mm(f['pad'], lang)} thick"))
    hint = ""
    if kind in ("by_edge",) or kind not in ("root", "applique", "flat", "cord"):
        if "sink" in f:
            hint = _sink_hint(f["sink"], lang)
    if kind == "applique" and "pad" in f:
        hint = _sink_hint(f["pad"], lang, "pad")
    return hint + _exact(bits, lang)


def _label(lang, a, b=None):
    p = "Р" if lang == "ru" else "R"
    return f"{p}{a}" if b is None or b == a else f"{p}{a}–{b}"


def _round_lines(part, lang, breaks=()):
    """Round lines with identical consecutive rounds merged (never across a round in `breaks`)."""
    key = "text_" + lang
    rows = part["rounds"]
    lines = []
    # round 1
    r1 = rows[0]
    c0 = yarn_ru(part["colour"]) if lang == "ru" else colour_word(part["colour"], "en")
    if part["start"] == "ch":
        c = r1["chain"]
        if lang == "ru":
            lines.append(f"{c0[:1].upper() + c0[1:]} наберите цепочку из {c} цеп. п.")
            a1 = f"{c - 2} сбн, " if c - 2 > 0 else ""
            a2 = f"{c - 3} сбн, " if c - 3 > 0 else ""
            lines.append(f"{_label(lang, 1)}: начиная со 2-й петли от крючка: {a1}3 сбн в последнюю петлю "
                         f"цепочки; не переворачивая работу, продолжайте по другой стороне цепочки: {a2}"
                         f"2 сбн в последнюю петлю (ту, в которую вязали первый сбн) ({r1['n']})")
        else:
            lines.append(f"With {c0}: ch {c}.")
            lines.append(f"{_label(lang, 1)}: starting in the 2nd ch from hook: sc {c - 2}, 3 sc in the last ch; "
                         f"continue along the other side of the chain: sc {c - 3}, 2 sc in the last ch (the one you "
                         f"made the first sc in) ({r1['n']})")
        if len(r1["colour_runs"]) > 1:
            lines.append(("   цвета ряда 1 по порядку: " if lang == "ru" else "   colours of round 1 in order: ")
                         + r1[key])
    elif part["start"] == "ring" and part.get("split"):
        c = r1["n"]
        if lang == "ru":
            lines.append("Вяжется поворотными рядами, ряды не замыкаются: каждый ряд начинайте с 1 цеп. п. подъёма и "
                         "поворачивайте работу (в счёт не входит).")
            lines.append(f"{_label(lang, 1)}: {c0} наберите {c + 1} цеп. п. (не замыкая в кольцо); начиная со "
                         f"2-й петли от крючка: {c} сбн ({c})")
        else:
            lines.append("Worked in turned rows, not joined: start every row with ch 1 and turn (not counted).")
            lines.append(f"{_label(lang, 1)}: with {c0}: ch {c + 1} (do not join); starting in the 2nd ch from hook: "
                         f"{c} sc ({c})")
    elif part["start"] == "ring":
        c = r1["n"]
        if lang == "ru":
            lines.append(f"{_label(lang, 1)}: {c0} наберите {c} цеп. п. и замкните в кольцо сс в первую петлю "
                         f"(не перекручивая); {c} сбн — по одному в каждую петлю цепочки ({c})")
        else:
            lines.append(f"{_label(lang, 1)}: with {c0}: ch {c} and join into a ring with a sl st in the first ch (do "
                         f"not twist); {c} sc, one in each ch of the ring ({c})")
    else:
        if lang == "ru":
            lines.append(f"{_label(lang, 1)}: {c0}: {r1['n']} сбн в КА ({r1['n']})")
        else:
            lines.append(f"{_label(lang, 1)}: with {c0}: {r1['n']} sc in MR ({r1['n']})")
        if len(r1["colour_runs"]) > 1:
            lines.append(("   цвета ряда 1 по порядку: " if lang == "ru" else "   colours of round 1 in order: ")
                         + r1[key])
    k = 1
    while k < len(rows):
        r = rows[k]
        j = k
        while (j + 1 < len(rows) and rows[j + 1][key] == r[key] and rows[j + 1]["loop"] == r["loop"]
               and rows[j + 1]["change_to"] is None and rows[j + 1]["n"] == r["n"]
               and rows[j + 1].get("turned") == r.get("turned") and rows[j]["round"] not in breaks):
            j += 1
        a, b = r["round"], rows[j]["round"]
        op = part.get("opening")
        if op and r.get("turned") and a == op["from"] + 1:
            g, lft, rgt = op["stitches"], op["left"], op["right"]
            lines.append(
                f"Вырез спереди: дальше, с {a}-го ряда, вяжите поворотными рядами. {g} п. по центру спереди "
                f"(по {lft} и {rgt} п. по обе стороны от маркера начала ряда) не провязывайте — это вырез "
                f"(для мордочки). Провяжите сс по {lft} п. до края выреза (в счёт не входят) и вяжите первый ряд; "
                f"в конце каждого ряда — 1 цеп. п. подъёма (в счёт не входит), поверните работу." if lang == "ru" else
                f"Opening: from round {a} on, work in turned rows: leave the {g} st at the centre front ({lft} and "
                f"{rgt} st on either side of the round-start marker) unworked - the opening (for the face). Sl st "
                f"across {lft} st to the edge of the opening (not counted) and work the first row; at the end of "
                f"every row ch 1 (not counted) and turn.")
        lab = _label(lang, a, b)
        if r.get("turned"):
            lab += " (поворотные ряды)" if lang == "ru" else " (turned rows)"
        if r["loop"] in ("blo", "flo"):
            lab += f" ({'ЗПС' if r['loop'] == 'blo' else 'ППС'})" if lang == "ru" else f" ({r['loop'].upper()})"
        elif r["loop"] == "rib":
            lab += " (резинка)" if lang == "ru" else " (ribbing)"
        elif r["loop"] == "shell":
            lab += " (ракушки)" if lang == "ru" else " (shells)"
        elif r["loop"] == "popcorn":
            lab += " (попкорны)" if lang == "ru" else " (popcorns)"
        pre = ""
        if r["change_to"]:
            lines.append(f"{change_ru(r['change_to'])}; последний протяг "
                         f"Р{a - 1} сделайте уже новой нитью." if lang == "ru" else
                         f"Change to {colour_word(r['change_to'], 'en')} (pull it through on the last yarn-over "
                         f"of R{a - 1}).")
        tail = ""
        if b != a:
            tail = f" — {_rounds_ru(b - a + 1)}" if lang == "ru" else f" — {b - a + 1} rounds"
        lines.append(f"{lab}: {pre}{r[key]} ({r['n']}){tail}")
        k = j + 1
    return lines


AXIS_RU = {"x": "поперёк, из стороны в сторону", "y": "вертикально, сверху вниз", "z": "спереди назад"}
AXIS_EN = {"x": "side to side", "y": "vertically, top to bottom", "z": "front to back"}
_G = {"m": 0, "f": 1, "n": 2}


def _g(pid):
    return ru_gender(part_name(pid, "ru"))


def _each_ru(pid):
    """'каждую ножку' / 'каждое ушко' / 'каждый ласт' (accusative)."""
    return ("каждый", "каждую", "каждое")[_G[_g(pid)]] + " " + part_name(pid, "ru", "acc")


def _one_ru(pid):
    """'по одной' / 'по одному' agreeing with the part."""
    return "по одной" if _g(pid) == "f" else "по одному"


def _pieces_ru(n):
    return f"{n} {_plural_ru(n, 'деталь', 'детали', 'деталей')}"


def _usage_text(part, lang):
    """How the copies of a part are used, for the part's heading / first line."""
    ru = lang == "ru"
    uses = part.get("uses", [])
    make = part["make"]
    if make <= 1:
        return "", ""
    if len(uses) == 1:
        u = uses[0]
        if u["mirror"] and u.get("on_pair"):
            return ((f"{_one_ru(part['id'])} на {_each_ru(u['on_part'])}"
                     + ("" if not part["variants"] else "; ряды ниже — для левой")) if ru else
                    f"one on each {part_name(u['on_part'], 'en')}"), ""
        if u["mirror"]:
            return (("левая и правая, одинаковые" if not part["variants"] else "левая и правая; ряды ниже — для левой")
                    if ru else ("left and right, identical" if not part["variants"]
                                else "left and right; rounds below are for the left one")), ""
        return "", ""
    labels = [qualifier(u["name"], part["id"], lang, side_ok=not u["mirror"], side=u.get("side")) for u in uses]
    sides = {"левый", "левая", "левое", "правый", "правая", "правое", "left", "right"}
    if all(not u["mirror"] for u in uses) and set(labels) <= sides and len(set(labels)) == len(labels):
        return (("левая и правая, одинаковые" if ru else "left and right, identical")
                if not part["variants"] else ("левая и правая; ряды ниже — для левой" if ru
                                              else "left and right; rounds below are for the left one")), ""
    plain = all(l == "" or l.startswith(("№", "no.")) for l in labels)
    if all(u["mirror"] for u in uses) and plain:
        k = len(uses)
        if all(u.get("on_pair") for u in uses):
            return ((f"по {k} на {_each_ru(uses[0]['on_part'])}" if ru else
                     f"{k} on each {part_name(uses[0]['on_part'], 'en')}"), "")
        return ((f"по {k} с каждой стороны — слева и справа" if ru else f"{k} on each side, left and right"), "")
    if all(l == "" or l.startswith(("№", "no.")) for l in labels):
        # copies told apart only by a number: say where they go (grouped by the piece they are sewn on)
        return "", _usage_groups(part, uses, lang)
    # copies without a qualifier among named ones: the number the assembly gives them ('№ 2')
    labels = [l or (f"№ {i}" if ru else f"no. {i}") for i, l in enumerate(labels, 1)]
    items = []
    for u, l in zip(uses, labels):
        cnt = 2 if u["mirror"] else 1
        if ru:
            items.append(f"{l or 'деталь'} — {cnt} шт." + (" (левая и правая)" if u["mirror"] and not u.get("on_pair")
                                                           else f" ({_one_ru(part['id'])} на {_each_ru(u['on_part'])})"
                                                           if u.get("on_pair") else ""))
        else:
            items.append(f"{l or 'piece'} — {cnt}" + (" (left and right)" if u["mirror"] else ""))
    s = ("Из них: " if ru else "Of these: ") + "; ".join(items)
    return "", s if s.endswith(".") else s + "."


def _usage_groups(part, uses, lang):
    """'Из них: на тело — 10 шт. (4 пары, слева и справа, и 2 одиночных); на голову — 12 шт. (...)' for copies
    that have no names of their own; '' when they all go to one place and are all alike."""
    ru = lang == "ru"
    groups = {}
    for u in uses:
        g = groups.setdefault((u.get("on_part"), bool(u.get("on_pair"))), {"single": 0, "pair": 0})
        g["pair" if u["mirror"] else "single"] += 1
    if len(groups) == 1 and not all(v["pair"] and v["single"] for v in groups.values()):
        return ""
    gi = _G[_g(part["id"])]
    items = []
    for (on, on_pair), v in groups.items():
        p, s = v["pair"], v["single"]
        total = 2 * p + s
        if ru:
            if on_pair and not s:
                items.append(f"{('по одной' if gi == 1 else 'по одному') if p == 1 else f'по {p}'} на "
                             f"{_each_ru(on)} — {total} шт.")
                continue
            where = f"на {part_name(on, 'ru', 'acc')}" if on else "основа"
            det = []
            if p:
                det.append("слева и справа" if p == 1 and not s else
                           f"{p} {_plural_ru(p, 'пара', 'пары', 'пар')}, слева и справа")
            if p and s:
                det.append(f"{s} " + _plural_ru(s, *(("одиночный", "одиночных", "одиночных"),
                                                     ("одиночная", "одиночные", "одиночных"),
                                                     ("одиночное", "одиночные", "одиночных"))[gi]))
            items.append(f"{where} — {total} шт." + (f" ({', и '.join(det)})" if det else ""))
        else:
            if on_pair and not s:
                items.append(f"{p} on each {part_name(on, 'en')} — {total}")
                continue
            where = f"on the {part_name(on, 'en')}" if on else "base"
            det = []
            if p:
                det.append("left and right" if p == 1 and not s else
                           f"{p} pair{'s' if p > 1 else ''}, left and right")
            if p and s:
                det.append(f"{s} single")
            items.append(f"{where} — {total}" + (f" ({', and '.join(det)})" if det else ""))
    s = ("Из них: " if ru else "Of these: ") + "; ".join(items)
    return s if s.endswith(".") else s + "."


def _oval_marker_text(part, lang):
    """Where the round start of a chain-oval piece is: the engine counts a side-axis oval's rounds from the centre
    front = the middle of a long side (its increases sit on the turns a quarter round away), a front-axis oval's
    from a turn (the centre front is the turn)."""
    ru = lang == "ru"
    c = int(part.get("chain") or 0)
    if part.get("oval_axis") == "front":
        return ("Маркер начала ряда — на закруглении овала, на центральной линии детали; Р1 начинается там же, "
                "сдвигать начало не нужно. Прибавки приходятся на закругления." if ru else
                "The round-start marker is at a turn of the oval, on the piece's centre line; R1 starts there, so no "
                "shift is needed. The increases fall on the turns.")
    k = max(1, int(round((c - 2) / 2)))
    return (f"Маркер начала ряда — посередине длинной стороны (центральная линия детали), а Р1 начинается у "
            f"закругления: после Р1 сдвиньте начало ряда — провяжите ещё {k} сбн (в счёт рядов они не входят) и "
            "только тогда поставьте маркер. Прибавки приходятся на закругления." if ru else
            f"The round-start marker is in the middle of a long side (the piece's centre line), while R1 starts at a "
            f"turn: after R1 shift the round start — sc {k} more (not counted in any round) and place the marker "
            "there. The increases fall on the turns.")


MOTIF_SEW_EN = {("triangle", "base"): "its foundation chain (the longest straight edge)",
                ("triangle", "side"): "its straight side (the row ends that have no decreases)",
                ("leaf", "base"): "its base (the end with 3 sc in one chain)",
                ("half_circle", "base"): "its straight edge (the row ends)",
                ("crescent", "base"): "the middle of its inner (chain) edge",
                ("petal", "base"): "its narrow base (where the round starts and is joined)"}
MOTIF_SEW_RU = {("triangle", "base"): "начальной цепочкой (самым длинным прямым краем)",
                ("triangle", "side"): "прямой стороной (краями рядов без убавок)",
                ("leaf", "base"): "основанием (концом, где 3 сбн в одну петлю)",
                ("half_circle", "base"): "прямым краем (краями рядов)",
                ("crescent", "base"): "серединой внутреннего края (по цепочке)",
                ("petal", "base"): "узким основанием (там, где ряд начинается и замыкается)"}


def _motif_section(part, model, lang, head, usage_line):
    """A shaped flat piece: kind, size, the rows exactly as amikit/motif.py makes them, finishing."""
    from .motif import KIND_EN, KIND_RU
    ru = lang == "ru"
    mo = part["motif"]
    L = [head, ""]
    if usage_line:
        L.append(usage_line)
    a, b = mo["size"]
    nr = len(part["rounds"])
    turned = mo["rows_turned"]
    if ru:
        how = "поворотными рядами" if turned else "по кругу"
        L.append(f"Плоский мотив — {KIND_RU[mo['kind']]} (вяжется {how}, один слой полотна), ≈ "
                 f"{_num(a / 10, lang)} × {_num(b / 10, lang)} см; {nr} {_plural_ru(nr, 'ряд', 'ряда', 'рядов')}.")
    else:
        how = "in turned rows" if turned else "in rounds"
        nm = mo["kind"].replace("_", " ")
        extra = "" if KIND_EN[mo["kind"]] == nm else KIND_EN[mo["kind"]] + ", "
        L.append(f"Flat motif — {nm} ({extra}worked {how}, one layer of fabric), about {_num(a / 10, lang)} × "
                 f"{_num(b / 10, lang)} cm; {nr} row{'' if nr == 1 else 's'}.")
    if mo["params"].get("point_up"):
        L[-1] += (" Пришивается углом вверх." if ru else " It is sewn on with a corner at the top.")
    if part.get("pair") and mo["params"].get("ends") == "one":
        L.append("Правую деталь вяжите зеркально: убавки — у другого края." if ru else
                 "Work the right piece as a mirror image: the decreases along the other edge.")
    L.append("")
    c0 = yarn_ru(part["colour"]) if ru else colour_word(part["colour"], "en")
    for r in part["rounds"]:
        if r["change_to"]:
            L.append(f"{change_ru(r['change_to'])}." if ru else
                     f"Change to {colour_word(r['change_to'], 'en')}.")
        lab = (("Ряд" if r["motif_label"] == "row" else "Круг") if ru else
               ("Row" if r["motif_label"] == "row" else "Round"))
        pre = ((f"{c0} " if ru else f"with {c0}: ") if r["round"] == 1 else "")
        L.append(f"{lab} {r['round']}: {pre}{r['text_' + lang]} ({r['n']})")
    L.append("")
    if ru:
        L.append("Закрепите нить, оставив длинный конец (~30 см) для пришивания. Не набивайте. Спрячьте начальный "
                 "хвостик, приколите мотив булавками к мягкой поверхности по форме и слегка отпарьте.")
    else:
        L.append("Fasten off leaving a long tail (~30 cm) for sewing. Do not stuff. Weave in the starting tail, pin "
                 "the motif out to shape on a soft surface and steam it lightly.")
    L.append("")
    return L


def _part_section(part, model, lang):
    ru = lang == "ru"
    L = []
    name = part_name(part["id"], lang, cap=True)
    make = part["make"]
    paren, usage_line = _usage_text(part, lang)
    if ru:
        head = f"### {name}" + (f" — сделайте {_pieces_ru(make)}" if make > 1 else "")
    else:
        head = f"### {name}" + (f" — make {make}" if make > 1 else "")
    if paren:
        head += f" ({paren})"
    if part.get("motif"):
        return _motif_section(part, model, lang, head, usage_line)
    L.append(head)
    L.append("")
    if usage_line:
        L.append(usage_line)
    sz = part["size_mm"]
    nr = len(part["rounds"])
    if part["applique"]:
        if "oval" in sz:
            a, b = sz["oval"]
            L.append(f"Плоский овал ≈ {_num(a / 10, lang)} × {_num(b / 10, lang)} см; {_rounds_ru(nr)}." if ru else
                     f"A flat oval about {_num(a / 10, lang)} × {_num(b / 10, lang)} cm; {nr} rounds.")
        else:
            d = sz.get("flat", part["rounds"][-1]["n"] * model["yarn"]["stitch_mm"] / math.pi)
            L.append(f"Плоский кружок ≈ {_num(d / 10, lang)} см в диаметре; {_rounds_ru(nr)}." if ru else
                     f"A flat circle about {_num(d / 10, lang)} cm across; {nr} rounds.")
    elif "oval" in sz:
        a, b = sz["oval"]
        L.append(f"Размер детали до сборки: овал ≈ {_num(a / 10, lang)} × {_num(b / 10, lang)} см в самом широком "
                 f"месте, ≈ {_num(sz['length'] / 10, lang)} см в длину; всего {_rounds_ru(nr)}." if ru else
                 f"Size before assembly: an oval about {_num(a / 10, lang)} × {_num(b / 10, lang)} cm at the widest, "
                 f"{_num(sz['length'] / 10, lang)} cm long; {nr} rounds.")
    elif sz["length"] < 6:
        # a piece that lies flat (a chain-ring collar, a disc): a length of a few mm means nothing to a crocheter
        L.append(f"Размер детали до сборки: ≈ {_num(sz['diameter'] / 10, lang)} см в самом широком месте "
                 f"(лежит плоско); всего {_rounds_ru(nr)}." if ru else
                 f"Size before assembly: about {_num(sz['diameter'] / 10, lang)} cm at the widest (lies flat); "
                 f"{nr} rounds.")
    elif ru:
        L.append(f"Размер детали до сборки: ≈ {_num(sz['diameter'] / 10, lang)} см в самом широком месте, "
                 f"≈ {_num(sz['length'] / 10, lang)} см в длину; всего {_rounds_ru(nr)}.")
    else:
        L.append(f"Size before assembly: about {_num(sz['diameter'] / 10, lang)} cm at the widest, "
                 f"{_num(sz['length'] / 10, lang)} cm long; {nr} rounds.")
    if part["start"] == "ch" and sz.get("chain_lies"):
        ax = sz["chain_lies"]
        if part["applique"]:
            L.append(f"Вяжется по кругу вокруг цепочки, поэтому получается овал: пришивайте его так, чтобы длинная "
                     f"сторона (вдоль цепочки) шла {AXIS_RU[ax]}." if ru else
                     f"Worked around a chain, so it is an oval: sew it on with its long side (along the chain) running "
                     f"{AXIS_EN[ax]}.")
            L[-1] += " " + _oval_marker_text(part, lang)
        else:
            start = sz.get("start_at")
            where = (f"; начальная цепочка — {WHERE2_RU[start]} детали (на готовой игрушке)" if ru and start else
                     f"; the starting chain is {WHERE_EN[start]} of the piece (on the finished toy)" if start else "")
            L.append((f"Деталь овальная: ряды идут вокруг цепочки, и на готовой игрушке цепочка (длинная сторона "
                      f"овала) лежит {AXIS_RU[ax]}{where}." if ru else
                      f"The piece is oval: the rounds go around the chain, and on the finished toy the chain (the long "
                      f"side of the oval) runs {AXIS_EN[ax]}{where}.")
                     + " " + _oval_marker_text(part, lang))
    if part.get("edge"):
        mc = part.get("main_colour", part["colour"])
        n2 = 2 * make
        if ru:
            L.append(f"Каждая деталь — из двух кружков (лицевой и изнаночный): свяжите {n2} "
                     f"{_plural_ru(n2, 'кружок', 'кружка', 'кружков')} — {make} по схеме ниже"
                     + (" (с цветами)" if part["multicolour"] or any(r["change_to"] for r in part["rounds"]) else "")
                     + f" и {make} однотонных {yarn_ru(mc)} (те же ряды, без смены цвета).")
        else:
            L.append(f"Each piece is two circles (front and back): make {n2} circles — {make} as below"
                     + (" (with the colours)" if part["multicolour"] or any(r["change_to"] for r in part["rounds"]) else "")
                     + f" and {make} plain in {colour_word(mc, 'en')} (same rounds, no colour changes).")
    if part["multicolour"]:
        if ru:
            s = ("Разноцветные ряды расписаны по петлям от маркера начала ряда; держите начало ряда на "
                 f"{LINE_RU[part['ref']]} детали (в готовой игрушке).")
            if not part["symmetric_colours"]:
                s += (" Порядок дан для вязания правой рукой (справа налево по лицевой стороне); "
                      "левшам — в обратном порядке.")
        else:
            s = ("Multi-colour rounds are listed stitch by stitch from the round-start marker; keep the round start "
                 f"on the piece's {LINE_EN[part['ref']]} (as it sits on the finished toy).")
            if not part["symmetric_colours"]:
                s += (" The order is for right-handed crocheting (right to left on the outside); "
                      "left-handed: reverse it.")
        L.append(s)
    L.append("")
    stuff_after = part["stuff_after"]
    lines = _round_lines(part, lang, breaks=(stuff_after,) if stuff_after else ())
    for ln in lines:
        L.append(ln)
        if stuff_after and _range_ends_at(ln, stuff_after, lang):
            L.append("")
            L.append(_stuff_note(part, lang))
            L.append("")
    for v in part["variants"]:
        L.append("")
        nm = v["instance"]
        vs = v.get("side") or (nm[-1] if nm.endswith(("_L", "_R")) else None)
        if vs == "R":
            who = "правой детали (на правом боку игрушки)" if ru else "the right piece (on the toy's right)"
        elif vs == "L":
            who = "левой детали (на левом боку игрушки)" if ru else "the left piece (on the toy's left)"
        else:
            q = qualifier(nm, part["id"], lang)
            who = (f"детали «{q}»" if q else "второй детали") if ru else f"the piece “{q or nm}”"
        L.append((f"Для {who} цвета в этих рядах другие (остальные ряды — как выше):" if ru
                  else f"For {who} these rounds differ in colour (all other rounds as above):"))
        rows = v["rounds"]
        k = 0
        while k < len(rows):
            j = k
            while (j + 1 < len(rows) and rows[j + 1]["round"] == rows[j]["round"] + 1
                   and rows[j + 1]["text_" + lang] == rows[k]["text_" + lang]):
                j += 1
            a, b = rows[k]["round"], rows[j]["round"]
            tail = "" if a == b else (f" — {_rounds_ru(b - a + 1)}" if ru else f" — {b - a + 1} rounds")
            L.append(f"{_label(lang, a, b)}: {rows[k]['text_' + lang]} ({rows[k]['n']}){tail}")
            k = j + 1
    L.append("")
    L.extend(_finish_notes(part, lang))
    L.append("")
    return L


def _range_ends_at(line, r, lang):
    p = "Р" if lang == "ru" else "R"
    import re
    m = re.match(rf"{p}(\d+)(?:–(\d+))?[ :]", line)
    return bool(m and int(m.group(2) or m.group(1)) == r)


def _stuff_note(part, lang):
    ru = lang == "ru"
    items = part["safety_items"]
    s = ""
    if items:
        eyes = any(t in ("safety_eye", "eye") for t in items)
        nose = "safety_nose" in items
        what_ru = " и ".join(x for x in (["глазки"] if eyes else []) + (["носик"] if nose else []))
        what_en = " and ".join(x for x in (["the safety eyes"] if eyes else []) + (["the safety nose"] if nose else []))
        s = (f"Вставьте {what_ru} (места — в разделе «Лицо») и закрепите заглушки. " if ru
             else f"Insert {what_en} now (positions: see “Face”) and fix the backs. ")
    if part["stuff"] == "none":
        return "→ " + s + ("Деталь не набивается." if ru else "This piece is not stuffed.")
    w = STUFF_RU[part["stuff"]] if ru else STUFF_EN[part["stuff"]]
    if ru:
        return "→ " + s + f"Начните набивать деталь ({w}) и добавляйте набивку по мере убавок."
    return "→ " + s + f"Start stuffing ({w}) and keep adding as you decrease."


def _coil_text(part, lang):
    """A spiral (a seahorse tail): rolled up from the tip and tacked - the way designers make it."""
    ru = lang == "ru"
    bi = part["bend_info"]
    turns = abs(bi["deg"]) / 360.0
    a0 = int(round(bi["at"]))          # the spiral runs from the tip up to where the bend starts
    hid = f"<!-- coil {_num(bi['coil'], 'en')}" + (f" pitch {_num(bi['pitch_mm'], 'en')}" if bi.get("pitch_mm") else "") + " -->"
    tw = (f"{_num(round(turns, 1), lang)} оборота" if ru else f"{_num(round(turns, 1), lang)} turns")
    if ru:
        return (f"Изгиб ≈ {_num(abs(bi['deg']), lang, 0)}° около {_num(bi['at'], lang)}-го ряда"
                f"<!-- изгиб занимает {_num(bi['over'], lang)} р. -->{hid}. "
                f"Хвост закручивается спиралью (≈ {tw}): после набивки сверните его, начиная с кончика (ряд 1), "
                f"внутрь — к стороне, которая на готовой игрушке смотрит {DIR_RU[bi['inner']]}; витки ложатся "
                f"плотно, каждый следующий шире. От кончика до {a0}-го ряда прихватите соседние витки друг к другу "
                "2–3 стежками нитью цвета детали через каждые 3–4 ряда и спрячьте нить внутри.")
    return (f"Bend ≈ {_num(abs(bi['deg']), lang, 0)}° around round {_num(bi['at'], lang)}"
            f"<!-- the bend spreads over {_num(bi['over'], lang)} rounds -->{hid}. "
            f"The tail rolls up into a spiral (about {tw}): after stuffing, roll it up from the tip (round 1) toward "
            f"the side that faces {DIR_EN[bi['inner']]} on the finished toy; the turns lie close, each wider than "
            f"the last. From the tip to round {a0} tack neighbouring turns together with 2–3 stitches of matching "
            "yarn every 3–4 rounds and hide the yarn inside.")


def _bend_text(part, lang):
    ru = lang == "ru"
    bi = part["bend_info"]
    if bi.get("coil"):
        return _coil_text(part, lang)
    deg = abs(bi["deg"]); a0, a1 = bi["rounds"]; wst = bi["width_st"]
    where = pos_text(bi["inner_pos"], lang)
    mirror = bool(part.get("pair"))
    if bi.get("drape") is not None:
        dr = bi["drape"]
        hid = "<!-- drape" + "".join(f" {k} {_num(dr[k], 'en', 2)}" for k in ("relax_mm", "min_radius_mm", "clear_mm",
                                                                             "twist") if k in dr) + " -->"
        tw = dr.get("_twist")
        turned = tw is not None and tw < 0.75
        # the ear keeps facing where it was sewn (its inside shows: a pinched seam) or lies flat on the head
        end_ru = ("Пусть ушко висит развёрнутым так, как оно пришито (у шва его можно чуть защипнуть): внутренняя "
                  "сторона остаётся видна, к голове плашмя ушко не прижимайте." if turned else
                  "Если ушко торчит, разгладьте его рукой вниз по голове.")
        end_en = ("Let it hang still turned the way it is sewn on (pinch it a little at the seam): its inside stays "
                  "visible, it does not lie against the head." if turned else
                  "If the ear sticks out, smooth it down along the head by hand.")
        if ru:
            return (f"Изгиб ≈ {_num(deg, lang, 0)}° около {_num(bi['at'], lang)}-го ряда"
                    f"<!-- изгиб занимает {_num(bi['over'], lang)} р. -->{hid}. Деталь мягкая, без набивки: "
                    f"складку не подшивайте. Пришитая так, как сказано в сборке, она сама перегибается через "
                    f"голову и свисает вниз; внутренняя сторона изгиба на готовой игрушке смотрит "
                    f"{DIR_RU[bi['inner']]} (на детали это {where}). {end_ru}")
        return (f"Bend ≈ {_num(deg, lang, 0)}° around round {_num(bi['at'], lang)}"
                f"<!-- the bend spreads over {_num(bi['over'], lang)} rounds -->{hid}. The piece is soft and "
                f"unstuffed: do not tack a fold. Sewn on as the assembly says, it bends over the head by itself "
                f"and hangs down; the inner side of the bend faces {DIR_EN[bi['inner']]} on the finished toy "
                f"(on the piece: {where}). {end_en}")
    stuffed = part["stuff"] != "none"
    if ru:
        # one statement of where the bend is: the rounds the crocheter gathers (a0..a1) below
        s = (f"Изгиб ≈ {_num(deg, lang, 0)}° около {_num(bi['at'], lang)}-го ряда"
             f"<!-- изгиб занимает {_num(bi['over'], lang)} р. -->. "
             + ("После набивки с" if stuffed else "С") + f"огните деталь так, чтобы внутренняя сторона сгиба на "
             f"готовой игрушке смотрела {DIR_RU[bi['inner']]} (на детали это {where}). ")
        if bi["take_rounds"] < 0.75:
            s += (f"Изгиб небольшой: его достаточно задать наклоном при пришивании; при желании прихватите "
                  f"внутреннюю сторону 1–2 стежками, стянув ряды {a0} и {a0 + 1}.")
        elif a1 - a0 > 3:
            s += (f"Изгиб плавный: иглой с нитью цвета детали проложите шов по внутренней стороне от {a0}-го до "
                  f"{a1}-го ряда по одной линии петель, захватывая по петле в каждом ряду, затем потяните нить, "
                  f"пока деталь не согнётся на нужный угол (внутренняя сторона сожмётся гармошкой ≈ на "
                  f"{_rounds_ru(a1 - a0)}), и закрепите нить внутри.")
        else:
            s += (f"На внутренней стороне стяните вместе {a0}-й и {a1}-й ряды (в складку уйдёт "
                  f"{_rounds_ru(a1 - a0)}): иглой с нитью цвета детали сделайте 3–4 стежка через петли этих двух "
                  f"рядов на ширину ≈ {wst} п., затяните до нужного угла и спрячьте нить внутри.")
        if mirror:
            s += " Вторую деталь сгибайте зеркально."
        return s
    s = (f"Bend ≈ {_num(deg, lang, 0)}° around round {_num(bi['at'], lang)}"
         f"<!-- the bend spreads over {_num(bi['over'], lang)} rounds -->. "
         + ("After stuffing, b" if stuffed else "B") + f"end the piece so the inner side of the bend faces "
         f"{DIR_EN[bi['inner']]} on the finished toy (on the piece: {where}). ")
    if bi["take_rounds"] < 0.75:
        s += (f"It is a slight bend: tilt the piece as you sew it on; optionally tack the inner side with 1–2 stitches "
              f"drawing rounds {a0} and {a0 + 1} together.")
    elif a1 - a0 > 3:
        s += (f"It is a gradual bend: with the needle and matching yarn run a line of stitches along the inner side "
              f"from round {a0} to round {a1}, one stitch in every round along one column, then pull the yarn until "
              f"the piece bends to the angle (the inner side gathers by about {a1 - a0} rounds) and fasten off inside.")
    else:
        s += (f"On the inner side draw rounds {a0} and {a1} together ({a1 - a0} rounds go into the fold): with the "
              f"needle and matching yarn make 3–4 stitches through the loops of both rounds over about {wst} st, "
              "pull to the angle and hide the end inside.")
    if mirror:
        s += " Bend the second piece as a mirror image."
    return s


def _press_text(part, lang):
    """How to press an unstuffed cup into an ear: a spoon (side to side, partly) or a dish (along its axis)."""
    ru = lang == "ru"
    d = part.get("press_dims") or {}
    n = len(part["rounds"])
    if part.get("dish", 0) > 0:
        pct = int(round(100 * float(part["dish"])))
        k = d.get("front_rounds", 1)
        if ru:
            return (f"Не набивайте. Сформируйте «тарелочку»: ряды 1–{k} — лицевая сторона ушка, кольцо амигуруми — "
                    f"её центр. Вдавите центр внутрь, чтобы лицевая сторона стала вогнутой (≈ {_mm(d.get('depth', 0), lang)} "
                    f"в центре), а ряды {min(k + 1, n)}–{n} подверните назад: они образуют толстый округлый бортик "
                    f"(≈ {_mm(d.get('rim', 0), lang)} толщиной) и ложатся сзади вторым слоем; ширина ушка "
                    f"≈ {_mm(d.get('width', 0), lang)}.<!-- (dish {pct} %) -->")
        return (f"Do not stuff. Shape it into a dish: rounds 1–{k} are the front of the piece, the magic ring its "
                f"centre. Push the centre in so the front is concave (about {_mm(d.get('depth', 0), lang)} at the "
                f"centre) and turn rounds {min(k + 1, n)}–{n} to the back: they make a thick rounded rim (about "
                f"{_mm(d.get('rim', 0), lang)} thick) and lie behind as a second layer; the piece is about "
                f"{_mm(d.get('width', 0), lang)} across.<!-- (dish {pct} %) -->")
    pct = int(round(100 * float(part["spoon"])))
    if part.get("stuff", "none") != "none":
        # a plump spoon: a thin layer of stuffing between the walls, the front still pressed into a hollow
        if ru:
            return (f"Набейте {STUFF_RU[part['stuff']]} — тонким слоем, только чтобы стенки разошлись: ушко ≈ "
                    f"{_mm(d.get('layers', 0), lang)} толщиной, с толстыми круглыми краями. Затем сформируйте «ложечку»: "
                    f"вдавите переднюю стенку к задней — спереди вогнутая ложечка глубиной ≈ "
                    f"{_mm(d.get('depth', 0), lang)}, сзади выпуклая спинка; ширина ≈ {_mm(d.get('width', 0), lang)}. "
                    f"Край отверстия не складывайте в прямую линию: он остаётся дугой, вогнутой вперёд."
                    f"<!-- (spoon {pct} %) -->")
        return (f"Stuff it {STUFF_EN[part['stuff']]} — a thin layer, just enough to hold the walls apart: the ear about "
                f"{_mm(d.get('layers', 0), lang)} thick, with thick rounded edges. Then shape it into a spoon: push the "
                f"front wall toward the back wall — a concave spoon about {_mm(d.get('depth', 0), lang)} deep in front, "
                f"a rounded back; about {_mm(d.get('width', 0), lang)} wide. Keep the opening a curve, hollow side "
                f"forward, not a straight line.<!-- (spoon {pct} %) -->")
    if ru:
        return (f"Не набивайте. Сформируйте «ложечку»: вдавите переднюю стенку внутрь задней (не расплющивая "
                f"до плоской пластинки) — спереди получится вогнутая ложечка глубиной ≈ {_mm(d.get('depth', 0), lang)}, "
                f"сзади выпуклая спинка, а по бокам — округлые сгибы в два слоя (≈ {_mm(d.get('layers', 0), lang)} "
                f"толщиной); ширина ≈ {_mm(d.get('width', 0), lang)}. Край отверстия не складывайте в прямую линию: "
                f"он остаётся дугой, вогнутой вперёд.<!-- (spoon {pct} %) -->")
    return (f"Do not stuff. Shape it into a spoon: push the front wall into the back wall (not down to a thin plate) "
            f"— a concave spoon about {_mm(d.get('depth', 0), lang)} deep in front, a rounded back, and rounded folds "
            f"of two layers at the sides (about {_mm(d.get('layers', 0), lang)} thick); about "
            f"{_mm(d.get('width', 0), lang)} wide. Keep the opening a curve, hollow side forward, not a straight "
            f"line.<!-- (spoon {pct} %) -->")


def _finish_notes(part, lang):
    ru = lang == "ru"
    out = []
    n_last = part["rounds"][-1]["n"]
    pct = int(round(100 * float(part["flatten"] or 0)))
    # a thickness the crocheter can check instead of a percentage (the exact amount stays in a comment)
    dia = float((part.get("size_mm") or {}).get("diameter") or 0)
    thick = max(3.0, dia * (1 - float(part["flatten"] or 0))) if dia else 0
    th_ru = f" до толщины ≈ {_num(round(thick / 10, 1), lang)} см" if thick >= 10 else (
        f" до толщины ≈ {int(round(thick))} мм" if thick else "")
    th_en = f" to about {_num(round(thick / 10, 1), lang)} cm thick" if thick >= 10 else (
        f" to about {int(round(thick))} mm thick" if thick else "")
    if part["closed"] and part["flatten"] > 0:
        out.append(f"Сплющите деталь{th_ru} (в сечении — овал).<!-- press the piece flat by {pct} % -->" if ru else
                   f"Press the piece flat{th_en} (oval cross-section).<!-- press the piece flat by {pct} % -->")
    if part["closed"]:
        if ru:
            out.append(("Добейте набивку, если нужно. " if part["stuff"] != "none" else "")
                       + "Закрепите нить, оставив хвостик ~15 см; иглой пропустите его "
                       f"через передние полупетли всех {n_last} петель последнего ряда, стяните отверстие и спрячьте "
                       "нить внутри детали.")
        else:
            out.append(("Top up the stuffing if needed. " if part["stuff"] != "none" else "")
                       + "Fasten off leaving a 15 cm tail; with the needle weave it "
                       f"through the front loops of all {n_last} stitches of the last round, pull tight and hide the "
                       "end inside.")
        if part.get("dish", 0) > 0 or part.get("spoon", 0) > 0:
            # a closed cup pressed into a dish (a round ear with its back closed): how to press it
            out.append(_press_text(part, lang))
    elif part.get("edge"):
        out.append(f"Закрепите нить. Сложите лицевой и однотонный кружки изнанками друг к другу и обвяжите по краю, "
                   f"провязывая обе стенки сразу: {n_last} сбн. Не набивайте. Оставьте длинный конец (~30 см) для "
                   "пришивания." if ru else
                   f"Fasten off. Hold a front and a plain circle wrong sides together and sc around through both "
                   f"layers: {n_last} sc. Do not stuff. Leave a long tail (~30 cm) for sewing.")
    else:
        if ru:
            out.append("Закрепите нить, оставив длинный конец (~30–40 см) для пришивания.")
        else:
            out.append("Fasten off leaving a long tail (30–40 cm) for sewing.")
        items = part["safety_items"]
        if items:
            out.append("Вставьте носик/глазки сейчас, пока деталь открыта (места — в разделе «Лицо»)." if ru
                       else "Insert the safety nose/eyes now, while the piece is open (positions: see “Face”).")
        if part.get("spoon", 0) > 0 or part.get("dish", 0) > 0:
            out.append(_press_text(part, lang))
        elif part["flatten"] >= 0.5:
            if part["stuff"] == "none":
                out.append("Не набивайте. Сложите деталь плоско (отверстие — в прямую линию), передней и задней "
                           f"стенками друг к другу.<!-- flattened {pct} % -->" if ru else
                           "Do not stuff. Press the piece flat (the opening becomes a straight line), front and back "
                           f"walls together.<!-- flattened {pct} % -->")
            else:
                out.append(f"Набейте совсем {STUFF_RU[part['stuff']]} и сплющите деталь{th_ru}.<!-- flattened {pct} % -->"
                           if ru else
                           f"Stuff {STUFF_EN[part['stuff']]} only and press it flat{th_en}.<!-- flattened {pct} % -->")
        elif part["flatten"] > 0:
            if part["stuff"] == "none" or part["applique"]:
                st_ru, st_en = "Не набивайте. ", "Do not stuff. "
            else:
                st_ru = f"Набейте {STUFF_RU[part['stuff']]}, затем с"
                st_en = f"Stuff {STUFF_EN[part['stuff']]}, then p"
            out.append((st_ru if st_ru.endswith("с") else st_ru + "С")
                       + f"легка сплющите деталь (в сечении — овал; сплющена на {pct} %)." if ru else
                       (st_en if st_en.endswith("p") else st_en + "P")
                       + f"ress the piece slightly flat (oval cross-section; flattened {pct} %).")
        elif part["applique"] or part["stuff"] == "none":
            out.append("Не набивайте." if ru else "Do not stuff.")
        else:
            out.append(f"Набейте {STUFF_RU[part['stuff']]} перед пришиванием; у самого края набивайте чуть меньше "
                       "и добавьте немного, когда шов почти закончен." if ru else
                       f"Stuff {STUFF_EN[part['stuff']]} just before sewing it on; a little less near the edge, "
                       "then add a pinch when the seam is nearly done.")
        if part["squash"] > 0:
            out.append(f"Перед пришиванием сплющите отверстие в щель: сложите край пополам ({n_last // 2} п. на "
                       f"{n_last - n_last // 2} п.) и пришивайте по этому плоскому краю (сжато на "
                       f"{int(round(100 * part['squash']))} %)." if ru else
                       f"Before sewing, flatten the opening into a slit: fold the edge in half "
                       f"({n_last // 2} st against {n_last - n_last // 2} st) and sew along that flat edge "
                       f"(pinched {int(round(100 * part['squash']))} %).")
    if part.get("bend_info"):
        out.append(_bend_text(part, lang))
    return out


def _foot_text(fp, lang, of=""):
    r0, r1 = fp["rounds"]
    if lang == "ru":
        rr = (f"{r0}-му ряду" if r0 == r1 else f"рядам {r0}–{r1}") if of.startswith("@") else (
            f"{r0}-й ряд" if r0 == r1 else f"ряды {r0}–{r1}")
        rr += of.lstrip("@")
        if fp.get("near_pole"):
            return rr + (" (вокруг центра последнего ряда)" if fp["near_pole"] == "end" else " (вокруг кольца амигуруми)")
        return rr + (" (по всей окружности)" if fp["around"] else f", ≈ {max(1, fp['stitches_across'])} п. в ширину")
    rr = (f"round {r0}" if r0 == r1 else f"rounds {r0}–{r1}") + of
    if fp.get("near_pole"):
        return rr + (" (around the centre of the last round)" if fp["near_pole"] == "end" else " (around the magic ring)")
    return rr + (" (all the way round)" if fp["around"] else f", about {max(1, fp['stitches_across'])} st wide")


WHERE2_RU = {"up": "вверху", "down": "внизу", "front": "спереди", "back": "сзади", "left": "на левом боку",
             "right": "на правом боку"}
SEAMLINE_RU = {"x": "из стороны в сторону (поперёк игрушки)", "y": "сверху вниз (вдоль {gen})",
               "z": "спереди назад"}
SEAMLINE_EN = {"x": "side to side (across the toy)", "y": "top to bottom (along the {gen})", "z": "front to back"}


def _ends_ru(st):
    return " и ".join(pos_text(p, "ru") for p in st["ends"])


def _step_title(st, lang, pn):
    """'Ножка (передняя)' / 'Подушечка (№ 2)' / 'Внутренняя часть ушка → левое ушко'."""
    ru = lang == "ru"
    part = st["part"]
    q = qualifier(st["name"], part, lang, side_ok=not st["mirror"], side=st.get("side"))
    if not q and st.get("entries", 1) > 1:
        q = f"№ {st['entry_no']}" if ru else f"no. {st['entry_no']}"
    t = pn(part, cap=True) + (f" ({q})" if q else "")
    return t


def _parent_text(st, lang, pn, case="nom"):
    """The parent's name; for a piece sewn onto one side of a pair ('on': 'ear_L') say which one, and for a part
    used by several entries (front_leg / back_leg of one 'leg' part) which entry ('front leg')."""
    parent = st["on"]
    on = st.get("on_name") or ""
    os_ = st.get("on_side") or (on[-1] if on.endswith(("_L", "_R")) else None)
    base = on[:-2] if on.endswith(("_L", "_R")) else on
    q = qualifier(base, parent, lang, side_ok=False) if base and base != parent else ""
    if os_ and not st["mirror"]:
        side = "left" if os_ == "L" else "right"
        if lang == "ru":
            adj = QUAL_RU[side][_G[_g(parent)]]
            return ru_decline(adj + " " + part_name(parent, "ru"), case) + (f" ({q})" if q else "")
        return f"{side} {q + ' ' if q else ''}{part_name(parent, 'en')}"
    if q:
        return (pn(parent, case) + f" ({q})") if lang == "ru" else f"{q} {pn(parent, case)}"
    return pn(parent, case)


def _pad_words(st, lang):
    """How thick the stuffing under a worn piece is: at its top and where it rests."""
    top, pad = st.get("top_gap_mm", 0), max(1, int(round(st.get("pad", 1))))
    if top > pad + 2:
        return (f"≈ {top} мм на макушке, к месту, где деталь лежит на основе, слой сходит до ≈ {pad} мм" if lang == "ru"
                else f"about {top} mm on top, thinning to about {pad} mm where it rests on the base")
    return f"тонким слоем ≈ {pad} мм" if lang == "ru" else f"a thin layer about {pad} mm thick"


def _assembly(model, lang):
    ru = lang == "ru"
    L = []

    def pn(p, c="nom", cap=False):
        return part_name(p, lang, c, cap)

    for n, st in enumerate(model["assembly"], 1):
        part = st["part"]
        title = _step_title(st, lang, pn)
        if st["kind"] == "root":
            L.append(f"{n}. **{title}** — основа сборки; кольцо амигуруми {pn(part, 'gen')} — "
                     f"{WHERE2_RU[st['start_at']]}." if ru else
                     f"{n}. **{title}** is the base; its magic ring is {WHERE_EN[st['start_at']]}.")
            L[-1] += fine_text(st, lang)
            continue
        parent = st["on"]
        at = pos_text(st["at"], lang)
        two = st["mirror"]
        pair_on = st.get("on_pair")
        _on = st.get("on_name") or ""
        _on = _on[:-2] if _on.endswith(("_L", "_R")) else _on
        _q = qualifier(_on, parent, "ru", side_ok=False) if _on and _on != parent else ""
        cnt_ru = ((f" (2 шт. — {_one_ru(part)} на {_each_ru(parent)}{f' ({_q})' if _q else ''})" if pair_on
                   else " (2 шт.)") if two else "")
        cnt_en = ((f" (2 — one on each {_parent_text({**st, 'on_side': None, 'mirror': True}, lang, pn)})"
                   if pair_on else " (make 2)") if two else "")
        if st["kind"] == "flat":
            col = yarn_ru(st["colour"]) if ru else colour_word(st["colour"], "en")
            W, R = st["width"], st["rows"]
            if ru:
                rows = ("" if R == 1 else f" Ряд 2: {W} сбн ({W})." if R == 2 else
                        f" Ряды 2–{R}: {W} сбн, 1 цеп. п., поверните ({W}).")
                txt = (f"{n}. **{title}**{cnt_ru} (вяжется поворотными рядами): {col} наберите {W + 1} цеп. п.; "
                       f"Ряд 1: начиная со 2-й петли от крючка, {W} сбн"
                       + (", 1 цеп. п. подъёма, поверните работу" if R > 1 else "") + f" ({W}).{rows} Закрепите нить.")
                if st["pinch"]:
                    txt += " Плотно обмотайте середину детали той же нитью несколько раз — получится бантик."
                if st["conform"]:
                    txt += (f" Получится полоска ≈ {_num(st['length_mm'] / 10, lang)} см. Положите её по изгибу "
                            f"{_parent_text(st, lang, pn, 'gen')} {AXIS_RU[st['runs']]}: середина — {at}; концы — "
                            f"{_ends_ru(st)}. Приколите булавками и пришейте по обоим длинным краям.")
                else:
                    txt += f" Пришейте к {_parent_text(st, lang, pn, 'dat')}: {at}."
            else:
                rows = ("" if R == 1 else f" Row 2: {W} sc ({W})." if R == 2 else
                        f" Rows 2–{R}: {W} sc, ch 1, turn ({W}).")
                txt = (f"{n}. **{title}**{cnt_en} (worked flat in turned rows): with {col}, ch {W + 1}; Row 1: "
                       f"starting in the 2nd ch from hook, {W} sc" + (", ch 1, turn" if R > 1 else "")
                       + f" ({W}).{rows} Fasten off.")
                if st["pinch"]:
                    txt += " Wrap the middle tightly several times with the same yarn to make a bow."
                if st["conform"]:
                    txt += (f" The strip is about {_num(st['length_mm'] / 10, lang)} cm long. Lay it along the curve "
                            f"of the {_parent_text(st, lang, pn)}, running {AXIS_EN[st['runs']]}: its middle "
                            f"{at}; its ends " + " and ".join(pos_text(p, 'en') for p in st["ends"])
                            + ". Pin, then sew along both long edges.")
                else:
                    txt += f" Sew it to the {_parent_text(st, lang, pn)}: {at}."
            L.append(txt)
            L[-1] += fine_text(st, lang)
            continue
        if st["kind"] == "cord":
            col = yarn_ru(st["colour"]) if ru else colour_word(st["colour"], "en")
            if ru:
                L.append(f"{n}. **{title}**{cnt_ru}: {col} провяжите цепочку из {st['chains']} цеп. п. "
                         f"(≈ {st['length_mm']:.0f} мм)"
                         + (", затем вернитесь по ней сс, начиная со 2-й петли от крючка" if st["slip_back"] else "")
                         + f". Пришейте начало к {_parent_text(st, lang, pn, 'dat')}: {at}.")
            else:
                L.append(f"{n}. **{title}**{cnt_en}: with {col}, ch {st['chains']} (≈ {st['length_mm']:.0f} mm)"
                         + (", then sl st back along the chain starting in the 2nd ch from hook" if st["slip_back"] else "")
                         + f". Sew its start to the {_parent_text(st, lang, pn)}: {at}.")
            L[-1] += fine_text(st, lang)
            continue
        dirt = _dir_text(st["dir"], lang)
        par_nom = _parent_text(st, lang, pn)
        if st["kind"] == "onto_opening":
            cf = st["child_footprint"]
            if ru:
                L.append(f"{n}. **{title} → {par_nom}.** Плотно набейте {pn(parent, 'acc')}. "
                         f"Приложите {pn(part, 'acc')} точкой стягивания (последним рядом) к центру отверстия "
                         f"{pn(parent, 'gen')} (Р{st['parent_round']}) и пришейте край отверстия к {pn(part, 'dat')} "
                         f"по кругу — шов проходит по {_foot_text(cf, lang, '@ ' + pn(part, 'gen'))}. "
                         "Перед тем как замкнуть шов, добейте шею набивкой, чтобы голова не болталась.")
            else:
                L.append(f"{n}. **{title} onto the {par_nom}.** Stuff the {pn(parent)} firmly. "
                         f"Hold the {pn(part)}'s closing point (last round) at the centre of the {pn(parent)}'s "
                         f"opening (R{st['parent_round']}) and sew the edge of the opening all round to the "
                         f"{pn(part)}: the seam runs along {_foot_text(cf, lang, ' of the ' + pn(part))}. Top up the "
                         "neck with stuffing before closing the seam so the head does not wobble.")
            L[-1] += fine_text(st, lang)
            continue
        if st["kind"] == "around":
            rr = _num(st["ring_round"], lang)
            if ru:
                s = (f"{n}. **{title} → {par_nom}**. Наденьте {pn(part, 'acc')} на {pn(parent, 'acc')}: кольцо из "
                     f"цепочки ({st['ring_n']} п.) ложится " + (f"по краю последнего ряда {pn(parent, 'gen')}" if st.get("ring_end")
                                                  else f"по {rr}-му ряду {pn(parent, 'gen')}") + ", ряды расходятся "
                     f"{dirt or 'вниз'}; пришейте кольцо к {pn(parent, 'dat')} по кругу, остальное полотно свободно "
                     "лежит на детали" + (" (разрез — спереди)" if st.get("split") else "") + "."
                     + (f" Полотно свисает с кольца и ложится на {pn(parent, 'acc')} ниже него." if st.get("fall")
                        else ""))
            else:
                s = (f"{n}. **{title} → {par_nom}**. Put the {pn(part)} around the {pn(parent)}: the chain ring "
                     f"({st['ring_n']} st) lies along " + (f"the edge of the last round of the {pn(parent)}" if st.get("ring_end")
                                                    else f"round {rr} of the {pn(parent)}") + ", the rounds spreading "
                     f"{dirt or 'down'}; sew the ring to the {pn(parent)} all round and let the rest lie free on it"
                     + (" (the opening at the front)" if st.get("split") else "") + "."
                     + (f" The fabric hangs from the ring and lies along the {pn(parent)} below it." if st.get("fall")
                        else ""))
            f = st["fine"]
            bits = []
            if "dir" in f:
                bits.append(("ряды расходятся " if ru else "rounds grow ") + ang_text(f["dir"], lang))
            if "front" in f:
                bits.append(("начало ряда смотрит " if ru else "round-start line faces ") + ang_text(f["front"], lang))
            if "shift" in f:
                bits.append((f"сдвиг вдоль детали {_mm(f['shift'], lang)}" if ru else
                             f"ring moved {_mm(f['shift'], lang)} along the piece"))
            s += _exact(bits, lang)
            L.append(s)
            continue
        fp = st["footprint"]
        of_ru = " " + _parent_text(st, lang, pn, "gen")
        of_en = " of the " + par_nom
        if ru:
            s = f"{n}. **{title} → {par_nom}**{cnt_ru}. Центр — {at}"
            if two and pair_on:
                s += (f" (описано для {pn(parent, 'gen')} на левом боку игрушки; на {pn(parent, 'acc')} справа "
                      "пришейте зеркально)")
            elif two:
                s += " (описана деталь на левом боку игрушки, вторую пришейте зеркально)"
            s += ". "
        else:
            s = f"{n}. **{title} → {par_nom}**{cnt_en}. Centre: {at}"
            if two:
                s += " (the one on the toy's left is described; sew the other as its mirror image)"
            s += ". "
        if st["kind"] == "by_edge" and st.get("dish"):
            k = st.get("edge_stitches", 4)
            rr = st.get("rim_round", "")
            s += (f"Поставьте «тарелочку» на бортик вогнутой стороной вперёд и пришейте нижней дугой бортика — "
                  f"≈ {k} п. ряда {rr}, захватывая и задний слой; шов — {_foot_text(fp, lang, of_ru)}." if ru else
                  f"Stand the dish on its rim, hollow side forward, and sew it on by the lower arc of the rim — about "
                  f"{k} st of round {rr}, catching the back layer too; the seam covers {_foot_text(fp, lang, of_en)}.")
        elif st["kind"] == "by_edge" and st.get("motif"):
            key = (st["motif"], st.get("sew", "base"))
            w_ru = MOTIF_SEW_RU.get(key, "нижним краем"); w_en = MOTIF_SEW_EN.get(key, "its lowest edge")
            s += (f"Поставьте мотив ребром и пришейте его {w_ru}; шов — {_foot_text(fp, lang, of_ru)}." if ru else
                  f"Stand the motif on its edge and sew it on by {w_en}; the seam covers "
                  f"{_foot_text(fp, lang, of_en)}.")
        elif st["kind"] == "by_edge":
            k = st.get("edge_stitches", 4)
            s += (f"Сложенную двойную деталь (два кружка, обвязанные вместе) поставьте ребром и пришейте только "
                  f"нижним краем — ≈ {k} п. края; шов — {_foot_text(fp, lang, of_ru)}." if ru else
                  f"Stand the joined double circle on its edge and sew it on by its bottom edge only — about {k} st "
                  f"of the edge; the seam covers {_foot_text(fp, lang, of_en)}.")
        elif st["kind"] == "applique":
            if st.get("anchor") == "eyes":
                s += ("Центр детали — посередине между глазками (глазки проходят сквозь неё). " if ru else
                      "The piece is centred between the eyes (the eyes go through it). ")
            elif st.get("anchor") == "eye":
                s += ("Деталь кладётся вокруг глазка своей стороны. " if ru else
                      "The piece goes around the eye on its side. ")
            if st.get("drape"):
                s += ("Накиньте мотив на изгиб детали, не растягивая: от центра ряды ложатся вокруг детали, острый "
                      "край/середина — вдоль неё. " if ru else
                      "Drape it over the curve without stretching: from its centre the rows run around the piece "
                      "and its length runs along it. ")
            s += (f"Пришейте {pn(part, 'acc')} плоско, по всему краю, потайным швом; место: "
                  f"{_foot_text(fp, lang, of_ru)}." if ru else
                  f"Sew it flat all round its edge; it covers {_foot_text(fp, lang, of_en)}.")
        elif st["kind"] == "by_opening":
            sm = st.get("seam")
            if st.get("rim_arc"):
                ra = st["rim_arc"]
                hid = f"<!-- rim arc {_num(ra['az'][0], 'en')} {_num(ra['az'][1], 'en')} -->"
                s += (f"Пришейте только часть края отверстия — ≈ {ra['stitches']} п. из {st['opening']}, ту, что "
                      f"на готовой игрушке смотрит {DIR_RU[ra['faces']]}; остальной край оставьте свободным "
                      f"(получится открытый карман). Шов: {_foot_text(fp, lang, of_ru)}.{hid}" if ru else
                      f"Sew on only part of the opening's edge — about {ra['stitches']} of its {st['opening']} st, "
                      f"the part that faces {DIR_EN[ra['faces']]} on the finished toy; leave the rest of the edge "
                      f"free (an open pocket). The seam: {_foot_text(fp, lang, of_en)}.{hid}")
            elif sm:
                n_op = st["opening"]
                gen = _parent_text(st, lang, pn, "gen") if ru else par_nom
                line = (SEAMLINE_RU if ru else SEAMLINE_EN)[sm["line"]].format(gen=gen)
                rel = ""
                if sm["along_round"] is not None:
                    rel = ((", вдоль ряда" if sm["along_round"] else ", поперёк рядов") if ru else
                           (", along the round" if sm["along_round"] else ", across the rounds"))
                if ru:
                    s += (f"Сплющите отверстие ({n_op // 2} п. на {n_op - n_op // 2} п.) и пришейте прямой линией "
                          f"{line}{rel}: {_foot_text(fp, lang, of_ru)}.")
                else:
                    s += (f"Flatten the opening ({n_op // 2} st against {n_op - n_op // 2} st) and sew it on in a "
                          f"straight line {line}{rel}: {_foot_text(fp, lang, of_en)}.")
            else:
                # a round seam is at least as wide as its opening's diameter (a steeply tilted piece made the
                # measured width 1-2 st for a 12-st opening)
                fp = dict(fp, stitches_across=max(fp["stitches_across"], int(round(st["opening"] / math.pi))))
                s += (f"Пришейте по кругу отверстия ({st['opening']} п.); шов занимает "
                      f"{_foot_text(fp, lang, of_ru)}." if ru else
                      f"Sew around the opening ({st['opening']} st); the seam covers {_foot_text(fp, lang, of_en)}.")
            if st.get("spoon"):
                s += (" Держите ложечку вогнутой стороной вперёд; край отверстия ложится дугой по изгибу "
                      f"{_parent_text(st, lang, pn, 'gen')}, оба слоя пришиваются." if ru else
                      " Keep the spoon hollow side forward; the opening lies in a curve following the "
                      f"{par_nom}, both layers sewn down.")
        elif st["kind"] == "worn":
            a_, b_ = st["rest_rounds"]
            rows_ru = f"по {a_}-му ряду" if a_ == b_ else f"по рядам {a_}–{b_}"
            rows_en = f"along round {a_}" if a_ == b_ else f"along rounds {a_}-{b_}"
            gap = st.get("edge_gap_mm", 0)
            if ru:
                s += (f"Наденьте {pn(part, 'acc')} на {pn(parent, 'acc')}, как шапочку: кольцо амигуруми — над этой "
                      f"точкой" + (", вырез (непровязанные петли) — спереди" if st.get("opening") else "")
                      + f". Подложите под деталь набивку — {_pad_words(st, lang)}: деталь шире основы и сама держит "
                      f"форму. Пришейте её {rows_ru} {pn(part, 'gen')} — там, где она лежит на основе; место: "
                      f"{_foot_text(fp, lang, of_ru)}.")
                if gap >= 3:
                    s += (f" Нижний край не притягивайте: изнутри прихватите его к основе в 4–6 местах свободными "
                          f"стежками длиной ≈ {gap} мм, чтобы сохранилась форма.")
            else:
                s += (f"Put the {pn(part)} on the {par_nom} like a hat: its magic ring over this point"
                      + (", the opening (the unworked stitches) at the front" if st.get("opening") else "")
                      + f". Put stuffing under it - {_pad_words(st, lang)}: it is "
                      f"wider than the {par_nom} and keeps its own shape. Sew it on {rows_en} of the {pn(part)}, "
                      f"where it rests on the {par_nom}; it covers {_foot_text(fp, lang, of_en)}.")
                if gap >= 3:
                    s += (f" Do not pull the lower edge in: tack it from the inside in 4-6 places with loose stitches "
                          f"about {gap} mm long so it keeps its shape.")
        elif st["kind"] == "closed_end":
            s += (f"Пришейте деталь закрытым концом (последними рядами); место касания — "
                  f"{_foot_text(fp, lang, of_ru)}." if ru else
                  f"Sew the piece on by its closed end (last rounds); it touches {_foot_text(fp, lang, of_en)}.")
        else:
            s += (f"Пришейте деталь её началом (стороной кольца амигуруми); место касания — "
                  f"{_foot_text(fp, lang, of_ru)}." if ru else
                  f"Sew the piece on by its start (the magic ring side); it touches {_foot_text(fp, lang, of_en)}.")
        if dirt and st["kind"] != "applique" and not st.get("dish"):
            s += (f" Направление: {dirt}." if ru else f" It points {dirt}.")
        s += fine_text(st, lang)
        L.append(s)
    return _group_repeats(L, model, lang)


GROUP_MIN = 5          # this many consecutive steps of one piece on one parent become one written step


def _hide_line(s):
    """A step line kept for the exact rebuild only: one comment line, its own comments escaped."""
    return "<!-- step: " + s.replace("<!--", "<!~~").replace("-->", "~~>") + " -->"


def _group_repeats(L, model, lang):
    """Many copies of one small piece sewn on one parent (bobbles, spikes, spots, toe beans) are one step for the
    crocheter - how many, where - instead of dozens of numbered steps with angles. The original steps stay below it
    as hidden lines for the exact rebuild (reverse.py reads them); the visible steps are numbered again."""
    steps = model["assembly"]
    if len(L) != len(steps):
        return L
    ru = lang == "ru"
    out, i = [], 0
    while i < len(L):
        st = steps[i]
        key = (st["part"], st.get("on"), st["kind"])
        j = i
        while j + 1 < len(L) and st["kind"] != "root" and (
                steps[j + 1]["part"], steps[j + 1].get("on"), steps[j + 1]["kind"]) == key:
            j += 1
        if j - i + 1 < GROUP_MIN:
            out.extend(L[i:j + 1])
            i = j + 1
            continue
        grp = steps[i:j + 1]
        count = sum(2 if s["mirror"] else 1 for s in grp)
        rr = []
        for s in grp:
            at = s.get("at") or {}
            rr += list(at.get("between", [])) or ([at["round"]] if "round" in at else [])
        both = any(s["mirror"] for s in grp) or {(s.get("at") or {}).get("side") for s in grp} >= {"left", "right"}
        part, parent = st["part"], st.get("on")
        name = part_name(part, lang, "nom", True)
        where = (f"ряды {min(rr)}–{max(rr)} {part_name(parent, lang, 'gen')}" if ru else
                 f"rounds {min(rr)}–{max(rr)} of the {part_name(parent, lang)}") if rr else ""
        if ru:
            vis = (f"**{name} → {part_name(parent, lang)}** — {count} шт. Расставьте детали булавками по "
                   f"{part_name(parent, lang, 'dat')}" + (f" ({where})" if where else "")
                   + (", с обеих сторон симметрично" if both else "")
                   + ", сверяясь с картинкой готовой игрушки, затем пришейте по одной.")
        else:
            vis = (f"**{name} → {part_name(parent, lang)}** — {count} pieces. Pin them over the "
                   f"{part_name(parent, lang)}" + (f" ({where})" if where else "")
                   + (", symmetrically on both sides" if both else "")
                   + ", checking against the picture of the finished toy, then sew them on one by one.")
        # how to sew each one: the first copy's own step as the worked example (without its number and title)
        ex = re.sub(r"<!--.*?-->", "", L[i])
        ex = re.sub(r"^\d+\.\s+\*\*.+?\*\*\s*", "", ex).strip().lstrip(".").strip()
        if ex:
            vis += (" Как пришить каждую — на примере первой: " if ru else " How to sew each - the first one as the example: ") + ex
        out.append(vis + " <!-- grouped -->")
        out.extend(_hide_line(x) for x in L[i:j + 1])
        i = j + 1
    # number the visible steps again (hidden lines keep their own numbers; the reader does not use them)
    k = 0
    for idx, s in enumerate(out):
        if s.startswith("<!-- step:"):
            continue
        k += 1
        out[idx] = re.sub(r"^\d+\.\s", "", s) if re.match(r"^\d+\.\s", s) else s
        out[idx] = f"{k}. " + out[idx]
    return out


def _pair_note(it, lang, count=True):
    """' (2 шт., симметрично: описана левая сторона, правую — зеркально)' for mirrored face items."""
    if not it["pair"]:
        return ""
    if it.get("on_pair"):
        if lang == "ru":
            g = _G[_g(it["on"])]
            left = ru_decline(QUAL_RU["left"][g] + " " + part_name(it["on"], "ru"), "gen")
            right = ru_decline(QUAL_RU["right"][g], "prep")
            return ((" (2 шт.: " if count else " (") + f"на {_each_ru(it['on'])}; описано для {left}, на {right} — "
                    "зеркально)")
        return ((" (2 шт.: " if count else " (") + f"на {_each_ru(it['on'])}; описано для левой, на правой — зеркально)"
                if lang == "ru" else
                (" (2: " if count else " (") + f"one on each {part_name(it['on'], 'en')}; the left one is described, "
                "the right is its mirror)")
    return ((" (2 шт., симметрично: " if count else " (симметрично: ")
            + "описана сторона на левом боку игрушки, на правом — зеркально)" if lang == "ru"
            else (" (2, symmetric: " if count else " (symmetric: ")
            + "the toy's left side is described, the right is its mirror image)")


def _curl_ref(it, lang):
    """Short pointer in a curled fringe item to the curling instructions given once at the end of 'Face'."""
    curl = float(it.get("curl", 0) or 0)
    if curl <= 0:
        return ""
    if lang == "ru":
        return (" Пряди завейте (как — в последнем пункте раздела)." if curl >= 90 else
                " Пряди слегка распушите волной (как — в последнем пункте раздела).")
    return (" Curl the strands (see the last item of this section)." if curl >= 90 else
            " Give the strands a soft wave (see the last item of this section).")


def _curl_text(it, lang, cow):
    ru = lang == "ru"
    curl = float(it.get("curl", 0) or 0)
    if curl <= 0:
        return ""
    if curl >= 90:
        s = (" Завитки: до привязывания плотно намотайте отрезки спиралью на спицу или деревянную шпажку Ø 3–4 мм, "
             "закрепите концы скотчем, смочите водой (или подержите над паром утюга, не касаясь пряжи) и дайте "
             "полностью высохнуть на спице; снимите — пряди останутся пружинками. Для пушистых волн вместо этого "
             "раскрутите каждую прядь на отдельные нити и слегка расчешите." if ru else
             " Curls: before knotting, wrap the pieces tightly in a spiral around a knitting needle or a wooden skewer "
             "(3–4 mm), tape the ends, dampen (or steam with an iron held above, not touching) and let them dry "
             "completely on the needle; slide them off and they stay curly. For fluffy waves instead, untwist each "
             "strand into its plies and brush lightly.")
    else:
        s = (" Лёгкая волна: раскрутите каждую прядь на отдельные нити и слегка расчешите." if ru else
             " Soft waves: untwist each strand into its plies and brush lightly.")
    if cow:
        s += (" Примечание: вместо привязанных прядей чёлку можно связать прямо по рядам вытянутыми петлями "
              "(петельчатыми столбиками) — это вариант, в счёт петель схемы он не входит." if ru else
              " Note: instead of knotted strands the fringe can be worked as loop stitches along the rounds — an "
              "alternative that is not part of this pattern's stitch counts.")
    return s


def _hx(c, model_hex):
    h = (model_hex or {}).get(c)
    return f" `{h}`" if h else ""


def _face_extra(it, lang, hexes=None):
    """' Exact: thread 1.1 mm thick; raised 0.9 mm.' — the sizes that decide how a face detail looks."""
    ru = lang == "ru"
    t = it["type"]
    b = []
    if t in ("safety_eye", "eye"):
        if "sink" in it:
            b.append(f"утоплен на {int(round(100 * it['sink']))} %" if ru else f"sunk {int(round(100 * it['sink']))} %")
        if "aspect" in it:
            b.append(f"вытянутость {_num(it['aspect'], lang)}" if ru else f"aspect {_num(it['aspect'], lang)}")
        if "iris" in it:
            b.append(f"радужка {colour_word(it['iris'], lang)}{_hx(it['iris'], hexes)}" if ru else
                     f"iris {it['iris']}{_hx(it['iris'], hexes)}")
        if "pupil" in it:
            b.append(f"зрачок {int(round(100 * it['pupil']))} %" if ru else f"pupil {int(round(100 * it['pupil']))} %")
        if it.get("colour", "black") != "black":
            b.append(f"цвет {colour_word(it['colour'], lang)}{_hx(it['colour'], hexes)}" if ru else
                     f"colour {it['colour']}{_hx(it['colour'], hexes)}")
    elif t == "safety_nose":
        if it.get("colour", "black") != "black":
            b.append(f"цвет {colour_word(it['colour'], lang)}{_hx(it['colour'], hexes)}" if ru else
                     f"colour {it['colour']}{_hx(it['colour'], hexes)}")
    elif t in ("stitch", "embroidery", "line", "satin", "strap"):
        if "thickness_mm" in it:
            b.append(f"нить {_num(it['thickness_mm'], lang)} мм" if ru else f"thread {_num(it['thickness_mm'], lang)} mm thick")
        if "strands" in it:
            b.append(f"{it['strands']} стежков" if ru else f"{it['strands']} strands")
        if "lift_mm" in it:
            b.append(f"приподнято на {_num(it['lift_mm'], lang)} мм" if ru else f"raised {_num(it['lift_mm'], lang)} mm")
    elif t == "blush":
        b.append(f"цвет {colour_word(it.get('colour', 'pink'), lang)}" if ru else f"colour {it.get('colour', 'pink')}")
        if "strength" in it:
            b.append(f"интенсивность {int(round(100 * it['strength']))} %" if ru else
                     f"strength {int(round(100 * it['strength']))} %")
    elif t == "fringe":
        for k, ru_w in (("droop", "провис"), ("spread", "разброс")):
            if k in it:
                b.append(f"{ru_w} {int(round(100 * it[k]))} %" if ru else f"{k} {int(round(100 * it[k]))} %")
        if "curl" in it:
            b.append(f"завиток {_num(it['curl'], lang)}°" if ru else f"curl {_num(it['curl'], lang)}°")
        if "seed" in it:
            b.append(f"вариант {it['seed']}" if ru else f"seed {it['seed']}")
        if "doubled" in it:
            b.append(("пряди сложены вдвое" if it["doubled"] else "одиночные пряди") if ru else
                     ("doubled" if it["doubled"] else "single strands"))
    if not b:
        return ""
    # colours are for the person too; everything else (sink %, mm, strands) only for the exact rebuild
    shown = [x for x in b if x.startswith(("цвет", "радужка", "colour", "iris"))]
    vis = (" " + "; ".join(shown).capitalize() + ".") if shown else ""
    return vis + _exact(b, lang, "Exact")


def _face(model, lang):
    ru = lang == "ru"
    L = []
    cow = "cow" in (model.get("name", "") + " " + model.get("title", "")).lower()
    curled = [it for it in model["face"] if it["type"] == "fringe" and float(it.get("curl", 0) or 0) > 0]
    marks = []
    for it in model["face"]:
        marks.append((len(L), it))
        host_name = part_name(it["on"], lang, "prep") if ru else part_name(it["on"], lang)
        if it.get("on_side"):              # one copy of a pair: say which (two eyeballs, two cheeks)
            sd = "left" if it["on_side"] == "L" else "right"
            host_name = (ru_decline(QUAL_RU[sd][_G[_g(it["on"])]] + " " + part_name(it["on"], "ru"), "prep") if ru
                         else f"{sd} {host_name}")
        if it.get("on_inst") and qualifier(it["on_inst"], it["on"], lang):
            q_ = qualifier(it["on_inst"], it["on"], lang)
            host_name = (ru_decline(q_ + " " + part_name(it["on"], "ru"), "prep") if ru
                         else f"{q_} {part_name(it['on'], lang)}")
        host = ("на " + host_name) if ru else ("on the " + host_name)
        t = it["type"]
        pair = _pair_note(it, lang)
        if t == "fringe":
            cols = it.get("colours") or [it.get("colour", "black")]
            cn = ", ".join(colour_word(c, lang) for c in cols)
            n_ = it.get("count", 12); L_ = it.get("length_mm", 20)
            pts = it.get("points", [])
            rr = [p.get("round", (p.get("between") or [0])[0]) for p in pts]
            region = it.get("region", "mixed")
            if region == "front" and rr and max(rr) - min(rr) <= 3:
                nm = "Чёлка" if ru else "Fringe"
            elif region == "back":
                nm = "Грива" if ru else "Mane"
            else:
                nm = "Пряди (грива)" if ru else "Strands (mane)"
            if ru:
                L.append(f"- **{nm}** ({host}){pair}: нарежьте {n_} {_plural_ru(n_, 'отрезок', 'отрезка', 'отрезков')} "
                         f"пряжи (цвет: {cn}) длиной ≈ {2 * L_ + 20:.0f} мм; сложите каждый пополам и привяжите "
                         "узлом «бычий узел» (крючком протяните петлю под петлёй вязания, концы — в неё) вдоль линии "
                         + " → ".join(pos_text(p, lang) for p in pts) +
                         f", равномерно. Подстригите до ≈ {L_:.0f} мм." + _curl_ref(it, lang))
            else:
                L.append(f"- **{nm}** ({host}){pair}: cut {n_} pieces of yarn ({cn}) about {2 * L_ + 20:.0f} mm long; "
                         "fold each in half and attach with a lark's head knot (pull through a stitch with the hook) along "
                         + " -> ".join(pos_text(p, lang) for p in pts) +
                         f", evenly spaced. Trim to about {L_:.0f} mm." + _curl_ref(it, lang))
            continue
        if t == "strap":
            col = yarn_ru(it["colour"]) if ru else colour_word(it["colour"], "en")
            path = " → ".join("(" + pos_text(p, lang) + ")" for p in it["points"])
            if ru:
                L.append(f"- **Лямка** ({host}){pair}: {col} провяжите цепочку из {it['chains']} цеп. п. "
                         f"(≈ {_num(it['length_mm'] / 10, lang)} см); для более плотной лямки вернитесь по ней сс. "
                         f"Пришейте вдоль линии {path}, прихватывая цепочку через каждые 2–3 петли.")
            else:
                L.append(f"- **Strap** ({host}){pair}: with {col}, ch {it['chains']} (≈ {_num(it['length_mm'] / 10, lang)} "
                         f"cm); for a firmer strap sl st back along it. Sew it along {path}, catching it every 2–3 ch.")
            continue
        if t in ("safety_eye", "eye"):
            size = _eye_size_text(it.get("size_mm", 9), lang)
            if it["pair"] and not it.get("on_pair"):
                a = it["at"]
                rr = (f"между {a['between'][0]} и {a['between'][1]} рядами" if "between" in a
                      else f"в {a['round']}-м ряду") if ru else (
                    f"between rounds {a['between'][0]} and {a['between'][1]}" if "between" in a
                    else f"in round {a['round']}")
                line = (LINE_RU if ru else LINE_EN)[a["from"]]
                # the exact round and offset for the reverse reader (the text says whole rounds / ~stitches)
                ex = f" <!-- eyes exact: rnd {_num(float(a.get('exact_round', 0)), 'en', 3)} st {_num(a['stitches'], 'en', 2)} -->"
                hl_ru = " с бликом" if it.get("highlight") else ""
                hl_en = " with a printed glint" if it.get("highlight") else ""
                if ru:
                    L.append(f"- **Глазки**{hl_ru} {size} мм ({host}): {rr}, на расстоянии {it['apart']} п. друг от друга "
                             f"(считайте по {a['count_round']}-му ряду: по ~{_num(a['stitches'], lang)} п. в каждую "
                             f"сторону от {line}).{ex}")
                else:
                    L.append(f"- **Safety eyes**{hl_en} {size} mm ({host}): {rr}, {it['apart']} st apart (count along "
                             f"round {a['count_round']}: about {_num(a['stitches'], lang)} st each side of the "
                             f"{line}).{ex}")
            else:
                hl_ru = " с бликом" if it.get("highlight") else ""
                hl_en = " with a printed glint" if it.get("highlight") else ""
                L.append((f"- **Глазок**{hl_ru} {size} мм ({host}){pair}: {pos_text(it['at'], lang)}." if ru else
                          f"- **Safety eye**{hl_en} {size} mm ({host}){pair}: {pos_text(it['at'], lang)}."))
        elif t == "safety_nose":
            size = it.get("size_mm", 10)
            sz = f"{_num(size[0], lang)}×{_num(size[1], lang)}" if isinstance(size, list) else _num(size, lang)
            shp = it.get("shape", "oval")
            if ru:
                shp_ru = {"oval": "овальный", "triangle": "треугольный"}.get(shp, "")
                L.append(f"- **Носик** {shp_ru} {sz} мм ({host}){pair}: {pos_text(it['at'], lang)}.")
            else:
                L.append(f"- **Safety nose**, {shp}, {sz} mm ({host}){pair}: {pos_text(it['at'], lang)}.")
        elif t in ("stitch", "embroidery", "line"):
            pts = it["points"]
            col = yarn_ru(it["colour"]) if ru else colour_word(it["colour"], "en")
            path = (" → ".join("(" + pos_text(p, lang) + ")" for p in pts))
            if ru:
                L.append(f"- **Вышивка** ({host}){pair}: {col}, прямыми стежками по точкам: {path}.")
            else:
                L.append(f"- **Embroidery** in {col} thread ({host}){pair}, straight stitches through: {path}.")
        elif t == "satin":
            shp = it.get("shape", "triangle")
            if "width_mm" in it:
                wtxt = (f"шириной ≈ {_num(it['width_mm'], lang)} мм" if ru else f"about {_num(it['width_mm'], lang)} mm wide")
            else:
                w = it.get("width_st", 3)
                wtxt = (f"шириной {w} п." if ru else f"{w} st wide")
            if "height_mm" in it:
                htxt = (f"высотой ≈ {_num(it['height_mm'], lang)} мм" if ru else f"{_num(it['height_mm'], lang)} mm tall")
            else:
                h = it.get("height_rounds", 2)
                htxt = (f"высотой {h} {_plural_ru(h, 'ряд', 'ряда', 'рядов')}" if ru else f"{h} rounds tall")
            rot = float(it.get("rotate", 0) or 0)
            rtxt = ""
            if abs(rot) >= 1:
                rtxt = ((f", повернув фигуру на {_num(abs(rot), lang, 0)}° "
                         + ("по часовой стрелке" if rot > 0 else "против часовой стрелки")
                         + " (если смотреть на эту поверхность)") if ru else
                        (f", turned {_num(abs(rot), lang, 0)}° " + ("clockwise" if rot > 0 else "anticlockwise")
                         + " (looking at that surface)"))
            col = yarn_ru(it["colour"]) if ru else colour_word(it["colour"], "en")
            off = it.get("offset_mm") or [0, 0]
            otxt = ""
            if abs(off[0]) >= 0.3 or abs(off[1]) >= 0.3:
                bits = []
                if abs(off[0]) >= 0.3:
                    bits.append((f"на {_num(abs(off[0]), lang)} мм " + ("правее" if off[0] > 0 else "левее")) if ru else
                                (f"{_num(abs(off[0]), lang)} mm to the " + ("right" if off[0] > 0 else "left")))
                if abs(off[1]) >= 0.3:
                    bits.append((f"на {_num(abs(off[1]), lang)} мм " + ("выше" if off[1] > 0 else "ниже")) if ru else
                                (f"{_num(abs(off[1]), lang)} mm " + ("up" if off[1] > 0 else "down")))
                otxt = ((" и ".join(bits) + " (если смотреть игрушке в лицо)") if ru else
                        (" and ".join(bits) + " (looking at the toy's face)"))
            role = it.get("role")
            if role == "eye":
                ttl_ru, ttl_en = "Глазки вышивкой", "Embroidered eyes"
            elif role == "glint":
                ttl_ru, ttl_en = "Блик на глазке", "Eye glint"
            else:
                ttl_ru, ttl_en = "Вышивка гладью", "Satin stitch"
            if ru:
                shp_ru = {"triangle": "треугольник", "oval": "овал", "heart": "сердечко"}.get(shp, "овал")
                where = (f"центр — {pos_text(it['at'], lang)}" + (f", затем сместите {otxt}" if otxt else ""))
                over = (" поверх уже вышитого глазка" if role == "glint" else
                        " в несколько слоёв, чтобы гладь выступала" if it.get("lift_mm") else "")
                L.append(f"- **{ttl_ru}** ({host}){pair}: {col}{over} — {shp_ru} {wtxt} и {htxt}, "
                         f"горизонтальными стежками{rtxt}; {where}.")
            else:
                where = f"centred {pos_text(it['at'], lang)}" + (f", then moved {otxt}" if otxt else "")
                over = (" over the embroidered eye" if role == "glint" else
                        " in several layers so it stands out" if it.get("lift_mm") else "")
                L.append(f"- **{ttl_en}** ({host}){pair}: in {col} thread{over}, a {shp} {wtxt} and {htxt}, "
                         f"horizontal stitches{rtxt}; {where}.")
        elif t == "blush":
            s = it.get("size_mm", 8)
            if ru:
                pr = ", с обеих сторон симметрично" if it["pair"] else ""
                L.append(f"- **Румянец** ≈ {_num(s, lang)} мм сухой пастелью/румянами ({host}{pr}): "
                         f"{pos_text(it['at'], lang)}.")
            else:
                pr = ", both sides" if it["pair"] else ""
                L.append(f"- **Blush** about {_num(s, lang)} mm, dry pastel or blush ({host}{pr}): "
                         f"{pos_text(it['at'], lang)}.")
        elif t == "felt":
            pair = _pair_note(it, lang, count=False)
            s = it.get("size_mm", [10, 7])
            col = colour_word(it["colour"], lang)
            rot = float(it.get("rotate", 0) or 0)
            rtxt = ""
            if abs(rot) >= 1:
                rtxt = ((f", наклон {_num(abs(rot), lang, 0)}° " + ("по часовой стрелке" if rot > 0 else
                                                                       "против часовой стрелки")) if ru else
                        (f", tilted {_num(abs(rot), lang, 0)}° " + ("clockwise" if rot > 0 else "anticlockwise")))
            if ru:
                what = (f"2 овала {_num(s[0], lang)}×{_num(s[1], lang)} мм" if it["pair"] else
                        f"овал {_num(s[0], lang)}×{_num(s[1], lang)} мм")
                L.append(f"- **Фетр** ({col}), {what}{rtxt} ({host}){pair}, приклейте или пришейте: "
                         f"{pos_text(it['at'], lang)}.")
            else:
                what = (f"2 ovals {_num(s[0], lang)}×{_num(s[1], lang)} mm" if it["pair"] else
                        f"oval {_num(s[0], lang)}×{_num(s[1], lang)} mm")
                L.append(f"- **Felt** ({col}), {what}{rtxt} ({host}){pair}, glue or sew on: "
                         f"{pos_text(it['at'], lang)}.")
    for k in range(len(marks) - 1, -1, -1):
        i0, it = marks[k]
        i1 = marks[k + 1][0] if k + 1 < len(marks) else len(L)
        if i1 > i0:
            L[i1 - 1] = L[i1 - 1] + _face_extra(it, lang, model.get("hex"))
    if curled:
        strong = max(curled, key=lambda x: float(x.get("curl", 0)))
        txt = _curl_text(strong, lang, cow).strip()
        weak = min(curled, key=lambda x: float(x.get("curl", 0)))
        if float(strong.get("curl", 0)) >= 90 and float(weak.get("curl", 0)) < 90:
            txt += " " + _curl_text(weak, lang, False).strip()
        L.append(("- **Как завить пряди**: " if ru else "- **Curling the strands**: ") + txt)
    return L


TITLE_RU = {  # spec "name" (or a word of it) -> the Russian title; the spec has no title_ru key
    "beagle": "Щенок бигль", "black_cat": "Чёрный котик", "bunny": "Розовый зайка", "capybara": "Капибара с черепашкой",
    "corgi": "Щенок корги", "crow": "Воронёнок", "duck": "Утёнок", "fox": "Лисёнок",
    "highland_cow": "Хайлендская коровка", "hippo": "Бегемотик", "mouse": "Мышонок", "panda": "Панда",
    "penguin": "Пингвинёнок", "potato": "Картошка в наушниках", "red_panda": "Красная панда",
    "sage_bunny": "Зайка шалфейного цвета", "shark": "Акула", "sloth": "Ленивец", "tabby": "Полосатый котёнок",
    "teddy": "Мишка", "turtle": "Черепашка", "unicorn": "Единорог",
    "cat": "Котик", "kitten": "Котёнок", "bear": "Мишка", "dog": "Собачка", "puppy": "Щенок", "frog": "Лягушонок",
    "cow": "Коровка", "pig": "Поросёнок", "lion": "Львёнок", "owl": "Совёнок", "chick": "Цыплёнок",
    "elephant": "Слонёнок", "giraffe": "Жирафик", "dino": "Динозаврик", "dinosaur": "Динозаврик",
    "rabbit": "Зайка", "hedgehog": "Ёжик", "raccoon": "Енотик", "koala": "Коала", "sheep": "Овечка",
    "lamb": "Овечка", "monkey": "Обезьянка", "whale": "Кит", "octopus": "Осьминожка", "bee": "Пчёлка",
}


def title_ru(model):
    if model.get("title_ru"):
        return model["title_ru"]
    nm = str(model.get("name", "")).lower()
    if nm in TITLE_RU:
        return TITLE_RU[nm]
    for w in nm.replace("-", "_").split("_")[::-1] + str(model.get("title", "")).lower().split()[::-1]:
        if w in TITLE_RU:
            return TITLE_RU[w]
    return model["title"]


def flag_ru(flag, model):
    """Russian text of an engine flag a maker can use; None for technical flags (left to flags.txt)."""
    import re
    m = re.match(r"(\w+): turned (\d+) deg away from other pieces \(avoid\)", flag)
    if m:
        name, deg = m.group(1), int(m.group(2))
        pid = name
        for p in model["parts"]:
            if name in p["instances"]:
                pid = p["id"]
        q = qualifier(name, pid, "ru")
        return (f"{part_name(pid, 'ru', cap=True)}" + (f" ({q})" if q else "")
                + f" отведена на ≈ {deg}° от соседних деталей, чтобы не упиралась в них; направление в сборке "
                  "это уже учитывает.").replace("отведена", {"m": "отведён", "f": "отведена", "n": "отведено"}[
                      ru_gender(part_name(pid, "ru"))])
    m = re.match(r"(\w+): (?:presses ([\d.]+) mm into other pieces|still intersects after turning away)", flag)
    if m:
        name = m.group(1); pid = name
        for p in model["parts"]:
            if name in p["instances"]:
                pid = p["id"]
        q = qualifier(name, pid, "ru")
        mm = f" на ≈ {_num(float(m.group(2)), 'ru')} мм" if m.group(2) else ""
        return (f"{part_name(pid, 'ru', cap=True)}" + (f" ({q})" if q else "")
                + f" плотно прилегает к соседним деталям (вдавливается{mm}) — это нормально для набитых деталей; "
                  "если мешает, набейте " + ("её" if ru_gender(part_name(pid, "ru")) == "f" else "его")
                + " чуть слабее.")
    m = re.match(r"the toy rests on (\w+); the body is ([\d.]+) mm above the floor", flag)
    if m:
        pid = m.group(1)
        for p in model["parts"]:
            if pid in p["instances"]:
                pid = p["id"]
        return (f"Игрушка опирается на {part_name(pid, 'ru', 'acc')}, а тело приподнято над столом на "
                f"{_num(float(m.group(2)), 'ru')} мм — при сборке проверьте устойчивость.")
    return None


def render_markdown(model, lang):
    ru = lang == "ru"
    y = model["yarn"]
    L = []
    title = model["title"]
    if ru:
        title = title_ru(model)
    L.append(f"# {title}" + (" — схема вязания крючком" if ru else " — crochet pattern"))
    L.append("")
    s = model["size"]
    if ru:
        L.append(f"**Готовый размер:** высота ≈ {_num(s['height_cm'], lang)} см, ширина ≈ {_num(s['width_cm'], lang)} "
                 f"см, глубина ≈ {_num(s['depth_cm'], lang)} см (при указанной плотности).")
    else:
        L.append(f"**Finished size:** about {_num(s['height_cm'], lang)} cm tall, {_num(s['width_cm'], lang)} cm wide, "
                 f"{_num(s['depth_cm'], lang)} cm deep (at the stated gauge).")
    L.append("")
    # materials
    L.append("## Материалы" if ru else "## Materials")
    L.append("")
    wt = str(y.get("weight", "DK"))
    m100 = YARN_M100.get(wt.lower())
    if ru:
        L.append(f"- Пряжа толщины {wt}" + (f" (≈ {m100} м/100 г)" if m100 else "") + ", хлопок или акрил; "
                 f"крючок {_num(y['hook_mm'], lang)} мм (на размер меньше, чем рекомендовано для пряжи, чтобы "
                 "полотно было плотным).")
        L.append("- Цвета (расход — приблизительная оценка: число петель × "
                 f"{_num(model['yarn_per_stitch_m'] * 100, lang)} см на сбн + хвосты для пришивания + 15 %):")
    else:
        L.append(f"- {wt} weight yarn" + (f" (about {m100} m/100 g)" if m100 else "") + ", cotton or acrylic; "
                 f"{_num(y['hook_mm'], lang)} mm hook (smaller than the yarn label suggests, for a tight fabric).")
        L.append("- Colours (amounts are an estimate: stitches × "
                 f"{_num(model['yarn_per_stitch_m'] * 100, lang)} cm per sc + sewing tails + 15 %):")
    for c in model["colours"]:
        nm = colour_word(c["name"], lang)
        bits = []
        if c["stitches"]:
            bits.append(f"{c['stitches']} п." if ru else f"{c['stitches']} st")
        if c.get("extra_m"):
            pre = "+ " if bits else ""
            bits.append(f"{pre}≈ {_num(c['extra_m'], lang)} м на пряди/цепочки/полоски" if ru else
                        f"{pre}about {_num(c['extra_m'], lang)} m for strands/chains/strips")
        L.append(f"  - {nm} `{c['hex']}` — ≈ {c['metres']} м ({' '.join(bits)})" if ru else
                 f"  - {nm} `{c['hex']}` — about {c['metres']} m ({' '.join(bits)})")
    eyes = [h for h in model["hardware"] if h["type"] in ("safety_eye", "eye")]
    noses = [h for h in model["hardware"] if h["type"] == "safety_nose"]
    threads = [h for h in model["hardware"] if h["type"] in ("stitch", "embroidery", "line", "satin")]
    felts = [h for h in model["hardware"] if h["type"] == "felt"]
    blush = [h for h in model["hardware"] if h["type"] == "blush"]
    for h in eyes:
        hl = h.get("highlight")
        L.append(f"- Глазки на безопасном креплении {_eye_size_text(h['size_mm'], lang)} мм"
                 f"{', мультяшные, с белым бликом' if hl else ''}"
                 f" ({colour_word(h['colour'], lang)}{_hx(h['colour'], model.get('hex'))})"
                 f" — {h['count']} шт." if ru else
                 f"- Safety eyes {_eye_size_text(h['size_mm'], lang)} mm"
                 f"{' with a white glint (cartoon eyes)' if hl else ''} "
                 f"({colour_word(h['colour'], lang)}{_hx(h['colour'], model.get('hex'))}) — {h['count']}")
    for h in noses:
        sz = h["size_mm"]
        sz = f"{_num(sz[0], lang)}×{_num(sz[1], lang)}" if isinstance(sz, list) else _num(sz, lang)
        L.append(f"- Носик на безопасном креплении {sz} мм ({'треугольный' if h['shape'] == 'triangle' else 'овальный'})"
                 f" — {h['count']} шт." if ru else f"- Safety nose {sz} mm ({h['shape'] or 'oval'}; "
                 f"{colour_word(h['colour'], lang)}{_hx(h['colour'], model.get('hex'))}) — {h['count']}")
    tcols = sorted({h["colour"] for h in threads})
    if tcols:
        hx = model.get("hex", {})
        L.append(("- Нитки для вышивки (мулине или тонкая пряжа): " if ru else
                  "- Embroidery thread (floss or thin yarn): ")
                 + ", ".join(colour_word(c, lang) + (f" `{hx[c]}`" if hx.get(c) else "") for c in tcols))
    if felts:
        def _fz(h):
            s = h.get("size_mm") or [10, 7]
            return f"{_num(s[0], lang)}×{_num(s[1], lang)}"
        L.append("- Фетр: " + ", ".join(f"{colour_word(h['colour'], lang)} ({h['count']} "
                                        f"{_plural_ru(h['count'], 'овал', 'овала', 'овалов')} {_fz(h)} мм)"
                                        for h in felts) if ru else
                 "- Felt: " + ", ".join(f"{colour_word(h['colour'], lang)} ({h['count']} × {_fz(h)} mm oval)"
                                        for h in felts))
    if blush:
        hx = model.get("hex", {})
        bc = sorted({h["colour"] for h in blush})
        tone = ", ".join(colour_word(c, lang) + (f" `{hx[c]}`" if hx.get(c) else "") for c in bc)
        L.append(f"- Сухая пастель или румяна для щёчек (тон: {tone})" if ru else
                 f"- Dry pastel or blush for the cheeks (shade: {tone})")
    if ru:
        L.append("- Наполнитель (холлофайбер / синтепух), маркер для начала ряда, игла с большим ушком, булавки, "
                 "ножницы.")
        if eyes or noses:
            L.append("- Глазки и носик на креплении не подходят для игрушек детям до 3 лет — замените вышивкой.")
        else:
            L.append("- Мордочка полностью вышита — твёрдых деталей нет, игрушка подходит и малышам.")
    else:
        L.append("- Fibrefill stuffing, a stitch marker, a yarn needle, pins, scissors.")
        if eyes or noses:
            L.append("- Safety eyes and noses are not safe for children under 3 — embroider them instead.")
        else:
            L.append("- The face is fully embroidered — no hard parts, safe for small children.")
    L.append("")
    # gauge
    w, h = float(y["stitch_mm"]), float(y["round_mm"])
    if ru:
        L.append("## Плотность")
        L.append("")
        L.append(f"1 сбн ≈ {_num(w, lang)} мм в ширину, 1 ряд ≈ {_num(h, lang)} мм в высоту "
                 f"(10 см ≈ {_num(100 / w, lang, 0)} сбн × {_num(100 / h, lang, 0)} рядов); толщина нити ≈ "
                 f"{_num(float(y.get('thickness_mm', 1.9)), lang)} мм. От плотности зависит "
                 "размер игрушки; вяжите плотно, чтобы набивка не просвечивала.")
    else:
        L.append("## Gauge")
        L.append("")
        L.append(f"1 sc ≈ {_num(w, lang)} mm wide, 1 round ≈ {_num(h, lang)} mm tall (10 cm ≈ "
                 f"{_num(100 / w, lang, 0)} sc × {_num(100 / h, lang, 0)} rounds); yarn about "
                 f"{_num(float(y.get('thickness_mm', 1.9)), lang)} mm thick. The gauge sets the size of the "
                 "toy; crochet tightly so the stuffing does not show.")
    L.append("")
    # abbreviations
    if ru:
        L += ["## Сокращения", "",
              "| | |", "|---|---|",
              "| КА | кольцо амигуруми |", "| сбн | столбик без накида |",
              "| пр | прибавка: 2 сбн в одну петлю |", "| уб | убавка: 2 петли провязать вместе одним сбн |",
              "| ЗПС | ряд вязать за заднюю полустенку петли |", "| ППС | ряд вязать за переднюю полустенку петли |",
              "| цеп. п. | цепочечная (воздушная) петля |", "| сс | соединительный столбик |",
              "| п. | петля |", "| Р5 | ряд 5; Р7–14 — ряды с 7 по 14 одинаковые |",
              "| (…) ×6 | повторить 6 раз |", "| (24) | петель в ряду после него |"]
        if any(r["loop"] == "popcorn" for p in model["parts"] for r in p["rounds"]):
            L += ["| попкорн | 5 ссн в одну петлю; выньте крючок из петли, введите его в первый из 5 столбиков, "
                  "захватите оставленную петлю и протяните её — столбики собираются в шишечку снаружи |"]
        if any(r["loop"] == "rib" for p in model["parts"] for r in p["rounds"]):
            L += ["| РЛСН | рельефный лицевой столбик с накидом (крючок вводится под столбик спереди) |",
                  "| РИСН | рельефный изнаночный столбик с накидом (крючок вводится под столбик сзади) |"]
        if any(p.get("motif") or any(r["loop"] == "shell" for r in p["rounds"]) for p in model["parts"]):
            L += ["| пссн | полустолбик с накидом |", "| ссн | столбик с одним накидом |",
                  "| сс2н | столбик с двумя накидами |", "| пико | 3 цеп. п., сс в 3-ю цеп. п. от крючка |"]
        L += [""]
        L += ["Вяжите по спирали, не соединяя ряды и без подъёмных петель; отмечайте начало каждого ряда маркером. "
              "«Левый/правый» везде — левый/правый бок самой игрушки (левый бок игрушки — справа от вас, когда "
              "она смотрит на вас)."]
        if any(p["multicolour"] or any(r["change_to"] for r in p["rounds"]) for p in model["parts"]):
            L.append("")
            L.append("Смена цвета: последний протяг предыдущей петли делайте уже новой нитью. Нить неработающего "
                     "цвета ведите по изнанке (внутри детали), не затягивая. Запись «[белым] …» означает: всё до "
                     "следующей скобки вязать этим цветом.")
    else:
        L += ["## Abbreviations", "",
              "| | |", "|---|---|",
              "| MR | magic ring |", "| sc | single crochet |", "| inc | 2 sc in the same stitch |",
              "| dec | single crochet two stitches together (invisible decrease recommended) |",
              "| BLO | work the round in back loops only |", "| FLO | work the round in front loops only |",
              "| ch | chain |", "| sl st | slip stitch |", "| st | stitch |",
              "| R5 | round 5; R7–14 = rounds 7 to 14 are the same |", "| (…) ×6 | repeat 6 times |",
              "| (24) | stitches in the round when it is done |"]
        if any(r["loop"] == "popcorn" for p in model["parts"] for r in p["rounds"]):
            L += ["| popcorn | 5 dc in one stitch; take the hook out of the loop, insert it in the first of the 5 dc, "
                  "catch the dropped loop and pull it through - the dc gather into a bobble on the right side |"]
        if any(r["loop"] == "rib" for p in model["parts"] for r in p["rounds"]):
            L += ["| fpdc | front post dc: yarn over, hook around the post of the st from the front |",
                  "| bpdc | back post dc: yarn over, hook around the post of the st from the back |"]
        if any(p.get("motif") or any(r["loop"] == "shell" for r in p["rounds"]) for p in model["parts"]):
            L += ["| hdc | half double crochet |", "| dc | double crochet |", "| tr | treble crochet |",
                  "| picot | ch 3, sl st in the 3rd ch from hook |", "| hdc2tog | 2 hdc worked together |"]
        L += [""]
        L += ["Work in a continuous spiral, no joins and no turning chains; mark the first stitch of every round. "
              "Left/right always means the toy's own left/right (the toy's left is on your right when it faces "
              "you)."]
        if any(p["multicolour"] or any(r["change_to"] for r in p["rounds"]) for p in model["parts"]):
            L.append("")
            L.append("Colour changes: pull the new colour through on the last yarn-over of the previous stitch. "
                     "Carry the unused colour loosely on the inside. “[white] …” means: everything up to the next "
                     "bracket in that colour.")
    L.append("")
    # parts
    L.append("## Детали" if ru else "## Pieces")
    L.append("")
    for part in model["parts"]:
        L.extend(_part_section(part, model, lang))
    # assembly
    L.append("## Сборка" if ru else "## Assembly")
    L.append("")
    if ru:
        L.append("Перед пришиванием отметьте маркером центр нужного ряда спереди (центральную линию: середина "
                 "мордочки/животика). «В k п. от центральной линии» — считайте k петель по указанному ряду. "
                 "Сначала приколите все детали булавками и сравните с картинкой, потом шейте. Пришивайте концом, "
                 "оставленным на детали, захватывая по одной петле детали и основы.")
    else:
        L.append("Before sewing, mark the centre front of the round you need (the middle of the face / belly). "
                 "“k st from the centre line” means count k stitches along that round. Pin everything first and "
                 "compare with the picture, then sew with the tail left on each piece, one stitch of the piece and "
                 "one of the base at a time.")
    L.append("")
    L.extend(_assembly(model, lang))
    L.append("")
    if model["face"]:
        L.append("### Лицо" if ru else "### Face")
        L.append("")
        hard = any(h["type"] in ("safety_eye", "eye", "safety_nose") for h in model["hardware"])
        if hard:
            L.append("Глазки и носик на креплении вставляются до набивки/закрытия детали (см. выше в деталях); вышивку "
                     "удобнее делать до пришивания детали." if ru else
                     "Safety eyes and noses go in before the piece is stuffed and closed (see the pieces above); "
                     "embroidery is easier before the piece is sewn on.")
        else:
            L.append("Вся мордочка вышивается. Порядок: сначала глазки (чёрная гладь), потом блики белой нитью "
                     "поверх них, затем носик и ротик, румянец — в самом конце. Перед вышивкой наметьте центры "
                     "булавками и сравните с картинкой. Концы ниток выводите в местах, которые потом закроет "
                     "пришитая деталь, или прячьте внутрь набивки." if ru else
                     "The whole face is embroidered. Order: the eyes (black satin), then the white glints over them, "
                     "then the nose and mouth, blush last. Mark the centres with pins first and compare with the "
                     "picture. Bring thread ends out where a sewn-on piece will cover them, or bury them in the "
                     "stuffing.")
        L.append("")
        L.extend(_face(model, lang))
        L.append("")
    # check: for the engine and the designer, not for the crocheter. A real arithmetic error is shown (a pattern
    # with an error must not look finished); the summary and the engine flags go into a comment (the site and the
    # PDF drop it)
    bad = [(part_name(part["id"], lang, cap=True), part["check"] + part.get("self_check", []))
           for part in model["parts"] if part["check"] + part.get("self_check", [])]
    if bad:
        L.append("## Проверка" if ru else "## Check")
        L.append("")
        for nm, errs in bad:
            L.append(f"- {nm}: " + ("ОШИБКИ: " if ru else "ERRORS: ") + "; ".join(errs))
        L.append("")
    C = ["<!-- engine check:"]
    for part in model["parts"]:
        ok = not (part["check"] + part.get("self_check", []))
        C.append(f"  {part['id']}: {len(part['rounds'])} rounds, max {max(r['n'] for r in part['rounds'])}, "
                 + ("round arithmetic OK" if ok else "ERRORS"))
    for fl in model["flags"]:
        C.append("  flag: " + str(fl).replace("--", "-"))
    C.append("-->")
    L.extend(C)
    L.append("")
    return "\n".join(L)


def self_check(model):
    """Re-read every written round: stated count == computed, colour runs sum to the count."""
    for part in model["parts"]:
        errs = []
        prev = None
        for r in part["rounds"]:
            if part.get("motif"):              # motif rows: checked by amikit.motif.check_rows
                break
            if sum(c for _, c in r["colour_runs"]) != r["n"]:
                errs.append(f"R{r['round']}: colour runs sum to {sum(c for _, c in r['colour_runs'])}, not {r['n']}")
            if r["round"] > 1 and r.get("loop") not in ("rib", "shell"):
                for lang in ("ru", "en"):
                    c, p = parse_count(r["text_" + lang])
                    if p != r["n"] or c != prev:
                        errs.append(f"R{r['round']} ({lang}): text makes {p} from {c}, expected {r['n']} from {prev}")
            prev = r["n"]
        part["self_check"] = errs
    return model


def write_patterns(toy, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    PART_OVERRIDE.clear()
    COLOUR_OVERRIDE.clear()
    COLOUR_OVERRIDE.update(toy.spec.get("colour_names_ru", {}))
    COLOUR_HEX.clear()
    COLOUR_HEX.update(toy.spec.get("colours", {}))
    for p in toy.spec.get("parts", []):
        for lang in ("ru", "en"):
            nm = p.get(f"name_{lang}")
            if nm:
                # a name given capitalised ('Тело') is still a common noun mid-sentence; keep acronyms ('DK')
                if lang == "ru" and nm[:1].isupper() and nm[1:2].islower():
                    nm = nm[:1].lower() + nm[1:]
                PART_OVERRIDE[(p["id"], lang)] = nm
    model = self_check(pattern_model(toy))
    (out / "PATTERN_RU.md").write_text(render_markdown(model, "ru"), encoding="utf8")
    (out / "PATTERN_EN.md").write_text(render_markdown(model, "en"), encoding="utf8")
    (out / "pattern.json").write_text(json.dumps(model, ensure_ascii=False, indent=1, default=_json_default),
                                      encoding="utf8")
    return model


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
