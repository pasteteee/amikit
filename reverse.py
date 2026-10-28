"""Reverse generation: a written crochet pattern -> amikit spec -> 3D.

    python -m amikit.reverse <pattern.md|.txt> <out_dir> [--ref photo.jpg] [--fast] [--spec-only]

Reads our own PATTERN_EN.md exactly (every placement, angle and size the writer states) and real designers'
patterns in the usual English wording ("Rnd 5-9: sc around (24)", "(sc 2, inc) x6", "*sc, inc* repeat 6 times",
"sc in each st around", BLO/FLO, colour changes, chain-oval starts, "Head (make 1)" headings). Russian round
tokens (сбн, пр, уб, КА, Р1) are read too.

The pattern decides the model: stitch counts come from the rounds (the stated total wins; a mismatch between a
round's stitches and its total is reported), colours from the colour runs of each round, closing / stuffing /
flattening from the finishing notes, placements from the assembly text. What a designer's text leaves open
(where exactly the ears go, which way an arm points) is filled from the designers' usual construction for that
part and reported in `reverse_report.txt` as "assumed", so the gaps in a pattern are visible.

Round trip (`amikit.roundtrip`): spec -> PATTERN_EN.md -> this reader -> spec' -> 3D, compared piece by piece;
any difference is information the written pattern did not carry.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# text normalisation

def _clean(s: str) -> str:
    s = s.replace(" ", " ").replace("×", "x").replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'").replace("≈", "~")
    return s


def _strip_md(s: str) -> str:
    return re.sub(r"(\*\*|__|`)", "", s).strip()


WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "twice": 2, "once": 1, "thrice": 3}


def _num_words(s):
    return re.sub(r"\b(" + "|".join(WORDNUM) + r")\b", lambda m: str(WORDNUM[m.group(1).lower()]), s, flags=re.I)


# ---------------------------------------------------------------------------------------------------------------
# round labels

_LABEL = re.compile(
    r"""^\s*(?:[-*•]\s+)?(?:\*\*|__)?\s*
    (?:
      (?P<w>rnds?|rounds?|rows?|r|р|ряды?|круги?)\s*\.?\s*(?P<a>\d+)
         (?:\s*(?:-|to|through|&|and|до|по)\s*(?:rnds?|rounds?|rows?|r|р)?\s*\.?\s*(?P<b>\d+))?
     |(?P<a2>\d+)(?:\s*-\s*(?P<b2>\d+))?\s*(?:-?[йяе]\s*)?(?P<w2>ряды?|ряд|круги?|rnds?|rounds?|rows?)
    )
    \s*(?:\*\*|__)?\s*(?:[:.)]|-\s|\s(?=\(|\d|sc|inc|dec|ch|mr|сбн|пр|уб))\s*(?:\*\*|__)?""",
    re.I | re.X)


_BARE = re.compile(r"^\s*(?:[-*•]\s+)?(?P<a>\d+)(?:\s*-\s*(?P<b>\d+))?\s*[:.)]\s*(?P<rest>.*)$")
_CROCHET = re.compile(r"\b(sc|inc|dec|sc2tog|invdec|magic ring|mr|ch|st|sts|сбн|пр|уб|ка)\b", re.I)


def round_label(line):
    """(first, last, is_row, rest) for a round line, else None."""
    m = _LABEL.match(_clean(line))
    if not m:
        mb = _BARE.match(_clean(line))
        if mb and _CROCHET.search(mb.group("rest")) and not re.match(r"\s*\d+\.\s+\*\*", line):
            a = int(mb.group("a")); b = int(mb.group("b") or a)
            if a <= b <= a + 80:
                return a, b, False, mb.group("rest").strip()
        return None
    a = int(m.group("a") or m.group("a2"))
    b = m.group("b") or m.group("b2")
    b = int(b) if b else a
    w = (m.group("w") or m.group("w2") or "").lower()
    if b < a or b - a > 80:
        return None
    return a, b, w.startswith("row"), _clean(line)[m.end():].strip()


# ---------------------------------------------------------------------------------------------------------------
# one round's instruction -> operations

class ParseError(ValueError):
    pass


_SC = r"(?:sc|сбн|single crochets?|hdc|dc|tr|sl\s*sts?|slst|ss|сс|пссн|ссн|сн)"
_INC = r"(?:inc(?:rease)?s?|пр(?:ибавк[аи])?|прибавок)"
_DEC = r"(?:(?:inv(?:isible)?\s*)?dec(?:rease)?s?|sc\s*2\s*tog|sc2tog|invdec|уб(?:авк[аи])?|убавок)"
_AROUND = r"(?:around|in each (?:st(?:itch)?|stitch) around|in each st(?:itch)?|in every st(?:itch)?|all around|to end|" \
          r"in all sts?|по кругу|в каждую петлю|до конца ряда)"


def _norm_ops(low):
    """Designers' stitch wording -> tokens sc / inc / inc3 / dec / same."""
    low = " " + low + " "
    low = re.sub(r"\b(?:then|and|also|work|make|start|begin|now)\b", " ", low)
    low = re.sub(r"single crochets?|сбн", "sc", low)
    low = re.sub(r"\b1 sc in (?:the )?same st(?:itch)?\b|1 сбн в ту же п\.?", " same ", low)
    low = re.sub(r"\b(?:inv(?:isible)?\s*)?dec(?:rease)?s?\b|\bsc\s*2\s*tog\b|\binvdec\b|\bуб(?:авк[аи])?\b", " dec ", low)
    low = re.sub(r"\binc(?:rease)?s?\b|\bпр(?:ибавк[аи])?\b", " inc ", low)
    low = re.sub(r"\b3 sc (?:in|into) (?:the )?(?:next|same|one|1|following) st(?:itch)?\b|3 сбн в 1 п\.?", " inc3 ", low)
    low = re.sub(r"\b2 sc (?:in|into) (?:the )?(?:next|same|one|1|following) st(?:itch)?\b", " inc ", low)
    low = re.sub(r"\b2 sc (?:in|into) (?:each|every|all)(?: (?:the )?(?:remaining )?(?:sts?|stitch(?:es)?))?\b", " inc in each st ", low)
    low = re.sub(r"\b3 sc (?:in|into) (?:each|every|all)(?: (?:sts?|stitch(?:es)?))?\b", " inc3 in each st ", low)
    low = re.sub(r"\b(?:sl\s*sts?|slst|hdc|dc|tr|ss|сс|пссн|ссн|сн)\b", " sc ", low)
    return re.sub(r"\s+", " ", low).strip()


_TOK = r"(sc|inc3|inc|dec|same)"
_STS = r"(?:sts?|stitch(?:es)?|п\.?)"


def _atom(it):
    """One item of a round -> (ops list, around?) ; ops tokens: sc, inc, inc3, dec, same."""
    s = it.strip().strip(".;:").strip()
    if not s:
        return [], False
    low = s.lower()
    if re.fullmatch(r"(ch\s*\d*|turn|ch \d+,? turn|sl\s*st (to|in) (first|1st|the first) (st|sc).*|join.*|"
                    r"(?:fasten|finish) off.*|place marker.*|pm|stuff.*|work in (a )?spiral.*|в\.?п\.?.*|"
                    r"(?:do not|don't) (?:join|turn).*|mark .*|(?:with|in) \w+ yarn)", low):
        return [], False
    low = _norm_ops(low)
    around = False
    m = re.search(r"\s*\b(?:around|all around|to (?:the )?end|по кругу|until end|to end of (?:the )?(?:round|rnd))$", low)
    if m:
        around = True
        low = low[:m.start()].strip()
    m = re.fullmatch(_TOK + r" (?:in|into) (?:each|every|all|all the)(?: (?:remaining )?" + _STS + r")?(?: around)?", low)
    if m:
        return [m.group(1)], True
    m = re.fullmatch(_TOK + r" (?:in|into) (?:each|every)(?: st| stitch)? (?:of (?:the )?)?(?:next|following) (\d+)(?: " + _STS + ")?", low)         or re.fullmatch(_TOK + r" (?:in|into) (?:the )?(?:next|following|all|all the|each of (?:the )?next) (\d+)(?: " + _STS + r")?", low)
    if m:
        k = int(m.group(2))
        if m.group(1) == "dec" and "all" in low:
            k //= 2
        return [m.group(1)] * k, around
    m = re.fullmatch(_TOK + r" (?:in|into) (?:the )?(?:next|following|same|1|one)(?: " + _STS + r")?", low)
    if m:
        return [m.group(1)], around
    m = re.fullmatch(r"(?:(\d+)\s*(?:x\s*)?)?" + _TOK + r"(?:\s*(?:x\s*)?(\d+))?(?: times)?(?: " + _STS + r")?", low)
    if m:
        k = int(m.group(1) or m.group(3) or 1)
        return [m.group(2)] * k, around
    m = re.fullmatch(_TOK + r" (?:in|into) (?:each|every) " + _STS, low)
    if m:
        return [m.group(1)], True
    raise ParseError(f"cannot read {it.strip()!r}")


def _split_top(s, seps=","):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch in seps and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x for x in out if x.strip()]


_REP = re.compile(r"^\s*(?:x\s*(\d+)|(\d+)\s*x(?!\w)|\*\s*(\d+)|(?:rep(?:eat)?\s*)?(\d+)\s*times|rep(?:eat)?\s*(?:x\s*)?(\d+)(?:\s*times)?"
                  r"|(around|to end|all around|по кругу)|(?:repeat|rep)\s+around)\s*(?:times)?\s*", re.I)


def _seq(s, prev):
    """A comma list of items -> (ops, around-item index or None). Nested '(...) xN' / '(...) around' groups."""
    items = _split_top(s)
    ops_list = []           # list of (ops, around_unit)
    for it in items:
        it = it.strip()
        m = re.match(r"^\s*(?:(\d+)\s*x?\s*)?[\(\[](.*)[\)\]]\s*(.*)$", it)
        if m and _balanced(m.group(2)):
            inner, rest = m.group(2), m.group(3)
            pre = m.group(1)
            unit, _ = _seq(inner, None)
            reps, around = None, False
            r = _REP.match(rest) if rest else None
            if r:
                g = [x for x in r.groups()[:5] if x]
                if g:
                    reps = int(g[0])
                else:
                    around = True
                rest = rest[r.end():]
            elif pre:
                reps = int(pre)
            elif re.search(_AROUND, rest.lower()):
                around = True; rest = ""
            if rest.strip(" .;"):
                raise ParseError(f"cannot read {rest.strip()!r} after a group")
            if around:
                ops_list.append((unit, True))
            else:
                ops_list.append((unit * (reps or 1), False))
            continue
        ops, around = _atom(it)
        ops_list.append((ops, around))
    arounds = [i for i, (o, a) in enumerate(ops_list) if a]
    if len(arounds) > 1:
        raise ParseError("two 'around' items in one round")
    if arounds:
        i = arounds[0]
        if prev is None:
            raise ParseError("'around' with an unknown previous round")
        fixed = sum(_cons(o) for j, (o, a) in enumerate(ops_list) if j != i)
        unit = ops_list[i][0]
        cu = _cons(unit)
        left = prev - fixed
        if cu <= 0 or left < 0 or left % cu:
            raise ParseError(f"'around' does not fit: {left} stitches left for a unit of {cu}")
        ops_list[i] = (unit * (left // cu), False)
    out = []
    for o, _ in ops_list:
        out += o
    return out, None


def _balanced(s):
    d = 0
    for ch in s:
        d += ch in "(["
        d -= ch in ")]"
        if d < 0:
            return False
    return d == 0


PRODUCE = {"sc": 1, "inc": 2, "inc3": 3, "dec": 1, "same": 1}
CONSUME = {"sc": 1, "inc": 1, "inc3": 1, "dec": 2, "same": 0}


def _cons(ops):
    return sum(CONSUME[o] for o in ops)


def _prod(ops):
    return sum(PRODUCE[o] for o in ops)


def _stars(s):
    """'*sc 2, inc* x6' / '*sc 2, inc; rep from * 5 more times' / '*sc, inc* around' -> '(...) x6' groups."""
    m = re.search(r"\*([^*]+?)[;,.]?\s*(?:rep(?:eat)?\.?\s*from\s*\*\s*(?:(\d+)\s*more\s*times|(\d+)\s*times|"
                  r"(around|to end|across)|\s*)|$)", s, re.I)
    if "*" in s and m and re.search(r"rep(?:eat)?\.?\s*from\s*\*", s, re.I):
        more, times, around = m.group(2), m.group(3), m.group(4)
        unit = m.group(1).strip().rstrip(",;")
        if more:
            rep = f"x{int(more) + 1}"
        elif times:
            rep = f"x{times}"
        else:
            rep = "around"
        return s[:m.start()] + f"({unit}) {rep}" + s[m.end():]
    m = re.search(r"\*([^*]+)\*\s*(?:,\s*)?(?:(?:rep(?:eat)?\s*)?(?:x\s*(\d+)|(\d+)\s*times|(\d+)\s*x)|(around|"
                  r"(?:rep(?:eat)?\s*)?around|rep(?:eat)?\s*(?:to end|around)))?", s, re.I)
    if m:
        n = m.group(2) or m.group(3) or m.group(4)
        rep = f"x{n}" if n else "around"
        return s[:m.start()] + f"({m.group(1).strip().rstrip(',;')}) {rep}" + s[m.end():]
    return s


COLOUR_WORDS = ("white", "black", "pink", "brown", "tan", "beige", "cream", "grey", "gray", "red", "orange",
                "yellow", "green", "blue", "purple", "lilac", "mint", "peach", "ivory", "coral", "rose", "sage",
                "navy", "teal", "gold", "mustard", "lavender", "light", "dark", "pale", "main", "contrast")


def parse_round(body, prev, colours=()):
    """One round's text (after the label) -> dict(ops, stated, n, runs, loop, start, chain, colour, change_to).

    runs: [[colour, stitches made], ...] in working order when the round changes colour, else None."""
    s = _clean(body)
    s = re.sub(r"\s*-\s*\d+\s*(rounds|rows|рядов|ряда)\s*$", "", s)        # our "— 7 rounds" / "— 7 rows" (turned rows)
    s = re.sub(r"\s*\((?:\d+\s*(?:rounds|rnds|рядов|ряда))\)\s*$", "", s, flags=re.I)
    s = _strip_md(s)
    res = {"ops": None, "stated": None, "loop": "both", "start": None, "chain": 0, "colour": None,
           "change_to": None, "runs": None, "notes": []}
    # a note after the total: "(12 st). Stuff body firmly through hole."
    m = re.search(r"([\(\[]\s*\d+\s*(?:sts?|stitches|sc|п\.?|петель|петли)?\s*[\)\]])\s*[.;]\s+([A-ZА-Я].*)$", s)
    if m:
        res["notes_after"] = m.group(2)
        s = s[:m.start()] + m.group(1)
    # stated total at the end: (24) [24] (24 sts) - 24 sts / = 24
    m = re.search(r"[\(\[]\s*(\d+)\s*(?:sts?|stitches|sc|п\.?|петель|петли)?\s*[\)\]]\s*[.;]?\s*$", s, re.I) or \
        re.search(r"(?:[-=:]|total of|total)\s*(\d+)\s*(?:sts?|stitches|п\.?)?\s*[.;]?\s*$", s, re.I)
    if m:
        res["stated"] = int(m.group(1))
        s = s[:m.start()].strip()
    # leading "with pink:" (our format) / "with white yarn," / "in white:"
    m = re.match(r"^(?:with|in|using)\s+(?:colou?r\s+)?([a-z][\w ]*?)(?:\s+yarn)?\s*[:,]\s*", s, re.I)
    if m and not re.match(r"(?:the |a )?(mr|magic|blo|flo|back|front)", m.group(1), re.I):
        res["colour"] = m.group(1).strip().lower().replace(" ", "_")
        s = s[m.end():]
    # trailing colour change "change to white" / "change colour to X" / "fasten off X, join Y"
    m = re.search(r"[,.;]?\s*(?:then\s+)?(?:change|switch)\s+(?:colou?rs?\s+)?(?:to|for)\s+([a-z][\w ]*?)(?:\s+yarn)?"
                  r"(?:\s+in the last (?:st|stitch)[^,.;]*)?\s*[.;]?\s*$", s, re.I)
    if m:
        res["change_to"] = m.group(1).strip().lower().replace(" ", "_")
        s = s[:m.start()]
    low = s.lower()
    if re.search(r"\bblo\b|back loops? only|in back loops?|за задн", low):
        res["loop"] = "blo"
    elif re.search(r"\bflo\b|front loops? only|in front loops?|за передн", low):
        res["loop"] = "flo"
    s = re.sub(r"\(?\b(?:in\s+)?(?:blo|flo|back loops? only|front loops? only)\b\)?[:,]?", "", s, flags=re.I)
    s = re.sub(r"\b(?:working|work)\s+(?:in|into)\s+(?:the\s+)?(?:back|front)\s+loops?(?:\s+only)?[:,]?", "", s,
               flags=re.I)
    s = s.strip(" ,:;")
    # magic ring / chain-oval starts
    s = re.sub(r"^(?:start|begin|make|work|crochet)\s+(?:with\s+)?", "", s, flags=re.I)
    m = re.match(r"^(?:make\s+(?:a\s+)?)?(?:(\d+)\s*(?:sc|сбн)\s*(?:in(?:to)?\s*|в\s+)?(?:an?\s+|the\s+)?"
                 r"(?:mr|magic\s*(?:ring|circle|loop)|adjustable\s*ring|ка|кольц\w*(?:\s*амигуруми)?)"
                 r"|(?:mr|magic\s*(?:ring|circle)|ка)\s*(?:with\s*)?(?:of\s*)?(\d+)\s*(?:sc|сбн)?"
                 r"|(?:sc|сбн)\s*(\d+)\s*(?:in(?:to)?\s*)?(?:an?\s+|the\s+)?(?:mr|magic\s*(?:ring|circle)|ка))\b\.?\s*",
                 s, re.I)
    if m:
        n = int(m.group(1) or m.group(2) or m.group(3))
        res.update(start="mr", n=n, ops=["sc"] * n)
        if res["stated"] and res["stated"] != n:
            res["notes"].append(f"magic ring with {n} sc but the total says {res['stated']}")
        return res
    if prev is None:
        m = re.match(r"^(?:with\s+\w+[:,]\s*)?ch(?:ain)?\s*(\d+)\b", s, re.I)
        if m or re.search(r"(2nd|second) (ch|chain) from (the )?hook|other side of the chain|from hook", s, re.I):
            # chain-oval round 1: the stated count decides (sc along both sides of the chain)
            n = res["stated"]
            if n is None:
                try:
                    n = _chain_round_count(s)
                except ParseError:
                    n = None
            if n is None:
                raise ParseError("chain-oval round 1 without a total")
            res.update(start="ch", n=n, ops=["sc"] * n, chain=(n + 1) // 2)
            if n % 2:
                res["notes"].append(f"chain-oval round 1 with an odd total {n}: read as {n + 1}")
                res["n"] = n + 1
            return res
    # colour segments: "[white] 7 sc, dec [orange] ..." and "2 sc in orange, 15 sc in white"
    segs = None
    if re.search(r"\[[a-z][\w ]*\]", s, re.I):
        parts = re.split(r"\[([a-z][\w ]*)\]", s, flags=re.I)
        segs = []
        for k in range(1, len(parts), 2):
            segs.append((parts[k].strip().lower().replace(" ", "_"), parts[k + 1].strip(" ,")))
    elif re.search(r"\d+\s*(?:sc|сбн)\s+in\s+[a-z]", s, re.I) and not re.search(
            r"in\s+(?:the\s+)?(?:next|same|each|every|mr|magic|blo|flo|back|front|one|1st|first|2nd)", s, re.I):
        segs = []
        for it in _split_top(s):
            mm = re.fullmatch(r"\s*(\d+)\s*(?:sc|сбн)\s+in\s+([a-z][\w ]*)\s*", it, re.I)
            if not mm:
                segs = None
                break
            segs.append((mm.group(2).strip().lower().replace(" ", "_"), f"{mm.group(1)} sc"))
    if segs:
        ops, runs = [], []
        for c, txt in segs:
            o, _ = _seq(_stars(txt), None)
            ops += o
            if runs and runs[-1][0] == c:
                runs[-1][1] += _prod(o)
            else:
                runs.append([c, _prod(o)])
        res["ops"] = ops
        res["runs"] = runs
    else:
        s2 = _stars(_num_words(s))
        if not s2.strip():
            res["ops"] = None
        else:
            res["ops"], _ = _seq(s2, prev)
    if res["ops"] is not None:
        res["n"] = _prod(res["ops"])
        if prev is not None and _cons(res["ops"]) != prev:
            res["notes"].append(f"works {_cons(res['ops'])} stitches but the previous round has {prev}")
        if res["stated"] is not None and res["stated"] != res["n"]:
            res["notes"].append(f"makes {res['n']} stitches but the total says {res['stated']}")
    if res["stated"] is not None:
        res["n"] = res["stated"]
    elif res["ops"] is None:
        if prev is None:
            raise ParseError("first round without a magic ring / chain")
        res["n"] = prev
        res["notes"].append("no stitches and no total: read as an even round")
    return res


def _chain_round_count(s):
    """Stitches of a chain-oval round 1 from its wording (when no total is given)."""
    s = re.sub(r"(?:starting\s+)?in\s+(?:the\s+)?(?:2nd|second)\s+ch(?:ain)?\s+from\s+(?:the\s+)?hook[:,]?", "", s,
               flags=re.I)
    s = re.sub(r"(?:continue\s+)?(?:along|on|in)\s+the\s+other\s+side\s+of\s+the\s+chain[:,]?|working in back "
               r"(?:bumps|loops) of (?:the )?ch[:,]?|^ch(?:ain)?\s*\d+[.,:]?", "", s, flags=re.I)
    s = re.sub(r"\(the one you made the first sc in\)|in (?:the )?last (?:ch|chain)|in (?:the )?(?:same|first) (?:ch|st)",
               "", s, flags=re.I)
    ops, _ = _seq(s, None)
    return _prod(ops)


# ---------------------------------------------------------------------------------------------------------------
# document -> pieces

PART_WORDS = [
    ("paw_pad", r"paw ?pads?|toe ?beans?|pads?"),
    ("inner_ear", r"inner ears?|ear lining|inside of (?:the )?ears?"), ("head_body", r"head\s*(?:and|&|\+)\s*body"),
    ("muzzle", r"muzzle|snout"), ("ear", r"ears?"), ("head", r"head"), ("body", r"body|torso"),
    ("arm", r"arms?|hands?"), ("leg", r"legs?|feet|foot|back legs?|hind legs?|paws?"),
    ("tail", r"tail"), ("nose", r"nose"), ("beak", r"beak|bill"), ("wing", r"wings?"), ("horn", r"horns?"),
    ("eye_patch", r"eye ?patch(?:es)?"), ("cheek", r"cheeks?"), ("belly", r"belly|tummy"), ("fin", r"fins?|flippers?"),
    ("mane", r"mane"), ("hat", r"hat|cap"), ("spot", r"spots?|patch(?:es)?"), ("bow", r"bow"),
    ("antenna", r"antennae?"), ("leaf", r"leaf|leaves"), ("stem", r"stem"),
]


_QUALS = ("small", "big", "large", "upper", "lower", "inner", "outer", "top", "bottom", "middle", "long", "short",
          "tiny", "front", "back", "hind")


def part_id_of(name):
    s = _strip_md(_clean(name)).lower()
    # a qualifier in brackets tells two pieces apart: "Crest feather (big)" / "Crest feather (small)"
    pq = [w for w in re.findall(r"\(\s*([a-z]+)\s*\)", s) if w in _QUALS]
    s = re.sub(r"\(.*?\)|-\s*make.*$|make \d+.*$|x\s*\d+|\d+\s*(?:pcs|pieces)", "", s)
    if pq and not re.search(r"\b" + pq[0] + r"\b", s):
        s = f"{s} {pq[0]}"
    s = re.sub(r"[^a-zа-я0-9 ]+", " ", s).strip()
    s = re.sub(r"\b(front|hind|back)(leg|legs|paw|paws|foot|feet)\b", r"\1 \2", s)
    for pid, rx in PART_WORDS:
        if re.search(r"\b(?:" + rx + r")\b", s):
            # keep a qualifier word ("front leg" -> front_leg) so two different leg pieces stay two parts
            q = re.search(r"\b(front|back|hind|upper|lower|outer|inner|small|big|large|left|right)\b", s)
            if q and pid in ("leg", "ear", "arm", "spot", "wing", "fin") and q.group(1) not in ("left", "right"):
                return f"{q.group(1)}_{pid}"
            return pid
    s = re.sub(r"\s+", "_", s)
    return s or "piece"


def _make_count(head):
    h = _num_words(_clean(head)).lower()
    for rx in (r"make\s*(\d+)", r"\bx\s*(\d+)\b", r"(\d+)\s*x\b", r"(\d+)\s*(?:pcs|pieces)", r"\((\d+)\)",
               r"сделайте\s*(\d+)", r"(\d+)\s*шт"):
        m = re.search(rx, h)
        if m:
            return int(m.group(1))
    if re.search(r"\b(ears|arms|legs|feet|paws|wings|horns|cheeks|eyes|patches|spots|fins|flippers)\b", h):
        return 2
    return 1


def _is_heading(line):
    s = line.strip()
    if not s:
        return False
    if re.match(r"^#{1,4}\s", s):
        return True
    t = _strip_md(s).strip(" :")
    if len(t) > 60 or round_label(s):
        return False
    if re.fullmatch(r"\*\*[^*]+\*\*:?", s):
        return True
    if t.isupper() and len(t) >= 3:
        return True
    words = t.split()
    if s.endswith(":") and len(words) <= 6:
        return True
    if len(words) <= 5 and any(re.search(r"\b(?:" + rx + r")\b", t.lower()) for _, rx in PART_WORDS):
        # (a colour change naming a part-like colour - "Change to wing." - is an instruction, not a heading)
        return not re.search(r"\b(sew|attach|insert|stuff|embroider|place|pin|fasten|leave|cut|with|change|switch|"
                             r"join|continue)\b", t.lower())
    return False


def split_document(text):
    """-> {'title', 'lines', 'pieces': [{'name', 'head', 'make', 'rounds': [(a, b, rest, lineno)], 'notes': [...],
    'pre': [...]}], 'assembly': [...], 'face': [...], 'materials': [...]}"""
    lines = _clean(text).splitlines()
    doc = {"title": "", "pieces": [], "assembly": [], "face": [], "materials": [], "other": [],
           "lines": [ln.strip() for ln in lines if ln.strip()]}
    for ln in lines:
        if ln.startswith("# "):
            doc["title"] = _strip_md(ln[2:]).split(" - ")[0].strip()
            break
    if not doc["title"]:
        first = next((ln.strip() for ln in lines if ln.strip()), "")
        doc["title"] = _strip_md(first).split(" - ")[0].strip()[:60]
    section = None
    last_head = None
    head_since_rounds = False
    cur = None
    last_label = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if re.match(r"^##\s+(assembly|assembling|finishing|сборка)\b", s, re.I):
            section = "assembly"; cur = None; continue
        if (section == "assembly" or s.startswith("## ")) and re.match(r"^#{2,3}\s+(face|лицо)\b", s, re.I):
            section = "face"; cur = None; continue
        if re.match(r"^##\s+(check|проверка|notes|abbreviations|gauge|materials|материалы|плотность)\b", s, re.I):
            section = s.lstrip("# ").split()[0].lower(); cur = None
            continue
        if re.match(r"^##\s+(pieces|детали|pattern|instructions)\b", s, re.I):
            section = "pieces"; continue
        lab = round_label(s)
        if lab and section not in ("assembly", "face", "check", "abbreviations"):
            a, b, is_row, rest = lab
            if cur is None or a == 1 or (head_since_rounds and a <= last_label):
                if cur is not None and a != 1 and not head_since_rounds:
                    pass
                elif cur is None or a == 1 or head_since_rounds:
                    name = last_head or f"piece {len(doc['pieces']) + 1}"
                    cur = {"name": name, "head": last_head or "", "make": _make_count(last_head or ""),
                           "rounds": [], "notes": [], "pre": list(doc.get("_pre", [])), "rows": is_row,
                           "variants": []}
                    doc["pieces"].append(cur)
                    doc["_pre"] = []
            if cur.get("_variant") is not None:
                cur["variants"][-1]["rounds"].append((a, b, rest, i))
            else:
                cur["rounds"].append((a, b, rest, i))
                cur["after_last"] = []
            last_label = b
            head_since_rounds = False
            section = section if section in ("pieces", None) else section
            continue
        if section == "assembly":
            if s:
                doc["assembly"].append(s)
            continue
        if section == "face":
            if s:
                doc["face"].append(s)
            continue
        if section in ("materials", "gauge", "материалы", "плотность"):
            doc["materials"].append(s)
            continue
        if not s:
            continue
        if cur is not None and re.match(r"^for (the right piece|the left piece|the piece)", s, re.I):
            side = "R" if "right" in s.lower() else "L" if "left" in s.lower() else None
            cur["variants"].append({"side": side, "rounds": [], "text": s})
            cur["_variant"] = side
            continue
        if _is_heading(s):
            last_head = _strip_md(s.lstrip("#").strip())
            head_since_rounds = True
            if cur is not None:
                cur.pop("_variant", None)
            doc["_pre"] = []
            # free text after the heading and before round 1 belongs to the next piece
            continue
        if cur is not None and not head_since_rounds:
            cur["notes"].append(s)
            cur.setdefault("events", []).append((i, s))
            cur.setdefault("after_last", []).append(s)
        else:
            doc.setdefault("_pre", []).append(s)
            doc["other"].append(s)
    doc.pop("_pre", None)
    for p in doc["pieces"]:
        p.pop("_variant", None)
    # free text after the last piece that is not in an assembly section: designers write the assembly there
    if not doc["assembly"] and doc["pieces"]:
        last = doc["pieces"][-1]
        tail = last.get("after_last", [])
        k = next((j for j, t in enumerate(tail) if re.search(r"\b(assembl|sew|attach|finishing)", t, re.I)), None)
        if k is not None:
            doc["assembly"] = tail[k:]
            last["notes"] = last["notes"][:len(last["notes"]) - len(tail) + k]
    return doc


# ---------------------------------------------------------------------------------------------------------------
# materials, colours

NAMED_HEX = {
    "white": "#f4f1ea", "black": "#2d2b2a", "pink": "#efb3bf", "light_pink": "#f6cdd5", "brown": "#8a5a3c",
    "light_brown": "#b98a62", "dark_brown": "#5a3a28", "tan": "#d2a679", "beige": "#e3cba8", "cream": "#f5ecdc",
    "ivory": "#f3eddf", "grey": "#9a9a9a", "gray": "#9a9a9a", "light_grey": "#c9c9c9", "dark_grey": "#5b5b5b",
    "red": "#c8352e", "orange": "#e8833a", "yellow": "#f2cf4a", "green": "#6aa35a", "light_green": "#a8d08d",
    "blue": "#4a78b5", "light_blue": "#a9cbe8", "navy": "#2c3e66", "purple": "#8a63a8", "lilac": "#c3a8d8",
    "lavender": "#c9b8e3", "mint": "#a9dcc5", "peach": "#f6c4a3", "coral": "#ef8a73", "rose": "#e88aa0",
    "sage": "#a9bb98", "teal": "#3f8f8c", "mustard": "#d4a72c", "gold": "#d4a72c", "salmon": "#f19a86",
}


def parse_materials(doc):
    text = "\n".join(doc["materials"] + doc["other"][:40])
    colours = {}
    for m in re.finditer(r"^\s*-\s*([^:`]+?)\s+`(#[0-9a-fA-F]{6})`", text, re.M):
        colours[m.group(1).strip().lower().replace(" ", "_")] = m.group(2).lower()
    yarn = {}
    m = re.search(r"1 sc\s*~?\s*([\d.,]+)\s*mm wide,\s*1 round\s*~?\s*([\d.,]+)\s*mm tall", text)
    if m:
        yarn["stitch_mm"] = float(m.group(1).replace(",", "."))
        yarn["round_mm"] = float(m.group(2).replace(",", "."))
    m = re.search(r"yarn about\s*([\d.,]+)\s*mm thick", text)
    if m:
        yarn["thickness_mm"] = float(m.group(1).replace(",", "."))
    m = re.search(r"([\d.,]+)\s*mm\s*(?:crochet\s*)?hook|hook[^\n]{0,25}?([\d.]+)\s*mm|крючок\s*([\d.,]+)", text, re.I)
    if m:
        yarn["hook_mm"] = float((m.group(1) or m.group(2) or m.group(3)).replace(",", "."))
    m = re.search(r"\b(lace|fingering|sport|dk|worsted|aran|chunky|bulky)\b", text, re.I)
    if m:
        yarn["weight"] = m.group(1).upper() if m.group(1).lower() == "dk" else m.group(1).lower()
    thread_hex = {}
    m = re.search(r"Embroidery thread[^:]*:\s*(.+)$", text, re.M)
    if m:
        for it in m.group(1).split(","):
            mm = re.match(r"\s*([a-z][\w ]*?)\s*(?:`(#[0-9a-fA-F]{6})`)?\s*$", it)
            if mm:
                nm = mm.group(1).strip().lower().replace(" ", "_")
                if mm.group(2):
                    thread_hex[nm] = mm.group(2).lower()
    m = re.search(r"Dry pastel or blush for the cheeks \(shade: (.+)\)", text)
    if m:
        for it in m.group(1).split(","):
            mm = re.match(r"\s*([a-z][\w ]*?)\s*`(#[0-9a-fA-F]{6})`", it)
            if mm:
                thread_hex[mm.group(1).strip().lower().replace(" ", "_")] = mm.group(2).lower()
    for m in re.finditer(r"(?:iris |colour |shade: |\(|; )([a-z][\w]*)\s+`(#[0-9a-fA-F]{6})`",
                         "\n".join(doc["face"]) + text):
        thread_hex.setdefault(m.group(1).lower(), m.group(2).lower())
    colours.update({k: v for k, v in thread_hex.items() if k not in colours})
    return colours, yarn


def gauge_from_hook(yarn):
    """Designers' patterns state a hook, rarely a gauge: typical tight amigurumi fabric for that hook."""
    h = float(yarn.get("hook_mm", 2.5))
    w = round(1.25 + 1.1 * h, 2)           # 2.5 mm hook -> ~4.0 mm per sc; 3.5 mm -> ~5.1 mm
    return {"stitch_mm": w, "round_mm": round(0.9 * w, 2), "thickness_mm": round(0.47 * w, 2)}


# ---------------------------------------------------------------------------------------------------------------
# a piece's text -> part definition

def piece_part(p, colours_known, log, main=None):
    """Pieces' rounds and notes -> spec part (rounds as absolute counts, colour runs, end, stuffing, ...)."""
    pid = p["id"]
    steps, prev, colour, cur = [], None, None, None
    runs_z = []
    stated_mismatch = []
    expanded = []      # per round: (n, loop, runs or None, colour)
    events = list(p.get("events", []))
    cur = _start_colour(p.get("pre", []) + [p["head"]])
    opening = None
    for nt in p.get("notes", []):
        mo_ = re.search(r"Opening: from round (\d+) on, work in turned rows: leave the (\d+) st at the centre front", nt)
        if mo_:
            opening = {"from": int(mo_.group(1)) - 1, "stitches": int(mo_.group(2))}
    for a, b, rest, lineno in p["rounds"]:
        turned = False
        if opening and re.match(r"\(turned rows\):\s*", rest):
            # a turned row of a front opening: the written count leaves out the unworked window
            rest = re.sub(r"^\(turned rows\):\s*", "", rest)
            turned = True
        while events and events[0][0] < lineno:
            c2 = _change_colour(events.pop(0)[1])
            if c2:
                cur = c2
        special = _special_round(rest, prev)
        if special is not None:
            step, n_, col_, split_ = special
            if col_:
                cur = col_
            if split_:
                p["_split"] = True
            if cur is None:
                cur = "main"
            for k in range(a, b + 1):
                steps.append(step)
                expanded.append({"round": k, "n": n_, "runs": None, "colour": cur})
            prev = n_
            continue
        try:
            if re.search(r"left unworked|remaining st(?:itche)?s? (?:are )?(?:left )?unworked|do not work the rest", rest,
                         re.I):
                log.append(f"{p['name']} R{a}: a partial round (the rest is left unworked) — not modelled")
                continue
            r = parse_round(rest, prev - opening["stitches"] if turned and prev else prev)
            if turned:
                r = dict(r, n=r["n"] + opening["stitches"])
                if r.get("runs"):
                    r["runs"] = None             # (a multi-colour turned row: colours of the window unknown)
        except ParseError as ex:
            log.append(f"{p['name']} R{a}: {ex} — line: {rest!r}")
            if prev is None:
                raise
            r = {"n": prev, "ops": None, "loop": "both", "start": None, "runs": None, "colour": None,
                 "change_to": None, "notes": ["unreadable: read as an even round"], "stated": None, "chain": 0}
        for nt in r["notes"]:
            log.append(f"{p['name']} R{a}{'-' + str(b) if b != a else ''}: {nt}")
        if r.get("colour"):
            cur = r["colour"]
        if cur is None:
            cur = r.get("colour") or "main"
        for k in range(a, b + 1):
            if r["start"] == "mr" and k == a:
                steps.append(f"mr {r['n']}")
            elif r["start"] == "ch" and k == a:
                steps.append(f"ch {r['n'] // 2}")
            else:
                steps.append(f"{r['loop']} {r['n']}" if r["loop"] != "both" else str(r["n"]))
            expanded.append({"round": k, "n": r["n"], "runs": r["runs"], "colour": cur})
            if r["runs"]:
                cur = r["runs"][-1][0]
        prev = r["n"]
        if r.get("change_to"):
            cur = r["change_to"]
        # a "Change to X" line between rounds (our format) is in the notes of the round it follows
    part = {"id": pid, "rounds": steps}
    if p.get("_split"):
        part["split"] = True
    if opening:
        part["opening"] = opening
    cnt = {}
    for e in expanded:
        if e["runs"]:
            for c, k in e["runs"]:
                cnt[c] = cnt.get(c, 0) + k
        else:
            cnt[e["colour"]] = cnt.get(e["colour"], 0) + e["n"]
    main = main or (expanded[0]["colour"] if expanded else "main")
    part["colour"] = main
    zones = []
    for e in expanded:
        if e["runs"]:
            zones.append({"round": e["round"], "runs": [list(x) for x in e["runs"]]})
        elif e["colour"] != main:
            zones.append({"rounds": [e["round"], e["round"]], "colour": e["colour"]})
    # "colours of round 1 in order: 5 sc in cream, 2 sc in black, 5 sc in cream" (a multi-colour first round)
    for nt in p["notes"]:
        m1 = re.match(r"\s*colours of round 1 in order:\s*(.+)$", nt)
        if m1 and expanded:
            r1 = parse_round(m1.group(1), expanded[0]["n"])
            if r1.get("runs") and sum(k for _, k in r1["runs"]) == expanded[0]["n"]:
                zones = [z for z in zones if z.get("round") != 1 and z.get("rounds") != [1, 1]]
                zones.insert(0, {"round": 1, "runs": [list(x) for x in r1["runs"]]})
    # merge consecutive single-colour bands
    merged = []
    for z in zones:
        if (merged and "colour" in z and "colour" in merged[-1] and merged[-1]["colour"] == z["colour"]
                and merged[-1]["rounds"][1] + 1 == z["rounds"][0]):
            merged[-1]["rounds"][1] = z["rounds"][1]
        else:
            merged.append(z)
    for v in p.get("variants", []):
        for a, b, rest, _ in v["rounds"]:
            n = next((e["n"] for e in expanded if e["round"] == a), None)
            try:
                r = parse_round(rest, None if a == 1 else n)
            except ParseError as ex:
                log.append(f"{p['name']} variant R{a}: {ex}")
                continue
            for k in range(a, b + 1):
                runs = r["runs"] or [[r.get("colour") or main, r["n"]]]
                merged.append({"round": k, "runs": [list(x) for x in runs], "side": v["side"] or "R"})
    if merged:
        part["colours"] = merged
    return part


_CNAME = r"([a-z][a-z0-9_ ]{0,24}?)"          # (a colour id may carry a digit: bamboo2)


def _expand_simple(text):
    """'1 sc, (popcorn, 3 sc) x5, popcorn, 2 sc' -> ['sc', 'pop', 'sc', ...] (sc and popcorns only)."""
    out = []
    depth, cur, items = 0, "", []
    for ch in text + ",":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur.strip()); cur = ""
        else:
            cur += ch
    for it in items:
        if not it:
            continue
        mm = re.fullmatch(r"\((.*)\)\s*[x×](\d+)", it)
        if mm:
            out += _expand_simple(mm.group(1)) * int(mm.group(2))
            continue
        mm = re.fullmatch(r"(\d+) sc", it)
        if mm:
            out += ["sc"] * int(mm.group(1))
            continue
        mm = re.fullmatch(r"(?:(\d+) )?popcorns?(?:\s*[x×](\d+))?", it)
        if mm:
            out += ["pop"] * int(mm.group(1) or mm.group(2) or 1)
            continue
        return []
    return out


def _special_round(rest, prev):
    """Our chain-ring starts, ribbing and shell-edging rounds -> (step, count, colour, split) or None."""
    r = rest.strip()
    m = re.match(r"(?:with ([a-z][\w ]*?): )?ch (\d+) and join into a ring with a sl st in the first ch \(do not "
                 r"twist\); (\d+) sc, one in each ch of the ring \((\d+)\)", r)
    if m:
        col = m.group(1).strip().replace(" ", "_") if m.group(1) else None
        return f"ring {m.group(2)}", int(m.group(4)), col, False
    m = re.match(r"(?:with ([a-z][\w ]*?): )?ch (\d+) \(do not join\); starting in the 2nd ch from hook: (\d+) sc "
                 r"\((\d+)\)", r)
    if m:
        col = m.group(1).strip().replace(" ", "_") if m.group(1) else None
        return f"ring {m.group(3)}", int(m.group(4)), col, True
    m = re.match(r"\(ribbing\):\s*\(fpdc, bpdc\) x(\d+) \((\d+)\)", r)
    if m:
        return f"rib {m.group(2)}", int(m.group(2)), None, False
    m = re.match(r"\(popcorns\):\s*(.+?) \((\d+)\)$", r)
    if m:
        ops = _expand_simple(m.group(1))
        pops = [i for i, o in enumerate(ops) if o == "pop"]
        if len(ops) == int(m.group(2)) and len(pops) >= 2:
            k = pops[1] - pops[0]
            off = (pops[0] - (k - 1)) % k
            return f"popcorns every {k}" + (f" from {off}" if off else ""), int(m.group(2)), None, False
    m = re.match(r"\(shells\):\s*(.+?) \((\d+)\)$", r)
    if m:
        from .motif import parse_row_text
        _, ops = parse_row_text(m.group(1))
        i = next(k for k, o in enumerate(ops) if o.startswith("inc:"))
        _, st, k = ops[i].split(":")
        every = 2 * i + 2
        return f"shells {k} {st} every {every}", int(m.group(2)), None, False
    return None


def motif_part(p, kind, log):
    """A shaped flat piece (our 'Flat motif — <kind>' section): its rows read back into operations, then the motif
    parameters whose rows are exactly these (amikit.motif.infer); whole-row colour changes."""
    from .motif import parse_row_text, infer, MotifError
    rows_ops, phr = [], []
    colour, cur = None, None
    events = list(p.get("events", []))
    row_col = []
    for a, b, rest, lineno in p["rounds"]:
        while events and events[0][0] < lineno:
            c2 = _change_colour(events.pop(0)[1])
            if c2:
                cur = c2
        body = re.sub(r"\s*\(\d+\)\s*$", "", rest.strip())
        mc = re.match(r"with ([a-z][\w ]*?):\s*", body)
        if mc:
            cur = mc.group(1).strip().replace(" ", "_")
            body = body[mc.end():]
        try:
            ph, ops = parse_row_text(body)
        except MotifError as ex:
            log.append(f"{p['name']} row {a}: {ex} — motif not read")
            return None
        for _ in range(a, b + 1):
            rows_ops.append(ops); phr.append(ph); row_col.append(cur)
    params = infer(kind, rows_ops, phr)
    if params is None:
        log.append(f"{p['name']}: the rows are not a standard {kind} motif — not read")
        return None
    if kind == "hexagon":
        params.pop("point_up", None)
        if "with a corner at the top" in " ".join(p.get("pre", []) + p.get("notes", [])):
            params["point_up"] = True
    main = row_col[0] or "main"
    part = {"id": p["id"], "motif": params, "colour": main}
    zones = []
    for i, c in enumerate(row_col, 1):
        if c and c != main:
            if zones and zones[-1]["colour"] == c and zones[-1]["rounds"][1] == i - 1:
                zones[-1]["rounds"][1] = i
            else:
                zones.append({"rounds": [i, i], "colour": c})
    ed = params.get("edging")
    if ed and zones and zones[-1]["rounds"] == [len(row_col), len(row_col)]:
        ed["colour"] = zones.pop()["colour"]
    if zones:
        part["colours"] = zones
    return part


def _start_colour(lines):
    for ln in lines:
        low = _strip_md(ln).lower()
        m = (re.match(r"^with " + _CNAME + r"\s*:", low) or
             re.search(r"(?:^|\s)(?:start with|starting with|use with|use|using|with|work in|in)\s+(?:the\s+)?"
                       + _CNAME + r"\s+(?:yarn|colou?r|thread)(?:\W|$)", low) or
             re.search(r"(?:^|\s)colou?r\s*:\s*" + _CNAME + r"\s*$", low))
        if m:
            return m.group(1).strip().replace(" ", "_")
    return None


def _change_colour(line):
    low = _strip_md(line).lower()
    m = re.search(r"(?:^|\s)(?:change|switch)\s+(?:colou?r\s+|yarn\s+)?(?:to|for)\s+(?:the\s+)?" + _CNAME +
                  r"(?:\s+(?:yarn|colou?r))?\s*(?:[.(,;]|$)", low) or re.search(
                  r"(?:^|\s)(?:join|continue with|now with)\s+(?:the\s+)?" + _CNAME + r"\s+(?:yarn|colou?r)", low)
    return m.group(1).strip().replace(" ", "_") if m else None


def _notes_text(p):
    return " ".join(p["notes"] + p.get("pre", []) + [p["head"]])


def finish_part(part, p, log):
    """End, stuffing, flattening, bend from the piece's notes."""
    t = _notes_text(p).lower()
    counts = [int(re.sub(r"\D", "", s.split()[-1])) for s in part["rounds"]]
    closed_words = re.search(r"pull (it )?tight|close (the )?(hole|opening|gap)|weave .*?(front loops|through)|"
                             r"стяните|cinch", t)
    open_words = re.search(r"(long|leave a) (tail|end)[^.]*sew|for sewing|sew (it )?(on|to)|leave (it )?open|"
                           r"для пришивания|оставьте .*?конец", t)
    if closed_words and not (open_words and "for sewing" in t and not closed_words):
        part["end"] = "closed"
    elif open_words:
        part["end"] = "open"
    else:
        part["end"] = "closed" if counts[-1] <= 8 and counts[-1] < max(counts) else "open"
        log.append(f"{p['name']}: closing not stated; assumed {part['end']}")
    if re.search(r"not stuffed|do not stuff|don't stuff|no stuffing|не набива|(?:don't|do not|doesn't|does not) need "
                 r"(?:to be )?stuff|no need to stuff|without stuffing|need no stuffing", t):
        part["stuff"] = "none"
    elif re.search(r"stuff\w*\s*\(?(?:very\s+)?(firm|firmly)|firmly stuff|stuff (it )?(well|tightly)|(firm|firmly)\)?", t) \
            and re.search(r"stuff", t):
        part["stuff"] = "firm"
    elif re.search(r"(moderately|medium)", t) and re.search(r"stuff", t):
        part["stuff"] = "medium"
    elif re.search(r"(lightly|a little|slightly|only a bit)", t) and re.search(r"stuff", t):
        part["stuff"] = "light"
    elif re.search(r"\bstuff", t):
        part["stuff"] = "firm"
    else:
        part["stuff"] = "firm" if part["end"] == "closed" else "none"
    # an ear pressed into a spoon (side to side, partly) or a dish (along its axis): our writer states the amount
    m = re.search(r"\((spoon|dish) ([\d.,]+)\s*%\)", t)
    if m:
        part[m.group(1)] = round(float(m.group(2).replace(",", ".")) / 100, 3)
    # flattening: our writer states the amount; designers say "flatten" / "press flat"
    m = re.search(r"flattened ([\d.,]+)\s*%", t) or re.search(r"press the piece flat by ([\d.,]+)\s*%", t)
    if m:
        part["flatten"] = round(float(m.group(1).replace(",", ".")) / 100, 3)
    elif re.search(r"flatten (?:the )?(?:open )?(?:end|opening)(?! into a slit)", t):
        part["squash"] = 1.0
        log.append(f"{p['name']}: the open end is flattened and closed across (pinched opening)")
    elif not (part.get("spoon") or part.get("dish")) and re.search(
            r"press (the piece|it) flat|\bflatten(?:ed)?\b(?! the opening)|press (it )?flat|fold (it )?flat", t):
        # (designers' words; a spoon / dish our writer stated exactly is not flattened on top)
        part["flatten"] = 0.85 if part.get("stuff", "none") == "none" else 0.4
    elif re.search(r"slightly flat", t):
        part["flatten"] = 0.3
    m = re.search(r"opening into a slit.*?\(pinched ([\d.,]+)\s*%", t)
    if m:
        part["squash"] = round(float(m.group(1).replace(",", ".")) / 100, 3)
    elif re.search(r"flatten the opening into a slit", t):
        part["squash"] = 0.8
    m = re.search(r"bend ~?\s*([\d.,]+)\s*(?:°|deg)\s*around round ([\d.,]+)", t)
    if m:
        part["bend"] = {"deg": float(m.group(1).replace(",", ".")), "at": float(m.group(2).replace(",", "."))}
        mo = re.search(r"bend spreads over ([\d.,]+) rounds", t)
        if mo:
            part["bend"]["over"] = float(mo.group(1).replace(",", "."))
        mc = re.search(r"<!-- coil ([\d.]+)(?: pitch ([\d.]+))? -->", t)
        if mc:
            part["bend"]["coil"] = float(mc.group(1))
            if mc.group(2):
                part["bend"]["pitch_mm"] = float(mc.group(2))
        md = re.search(r"<!-- drape((?: [a-z_]+ [\d.]+)*) -->", t)
        if md:
            part["bend"]["drape"] = True
            kv = md.group(1).split()
            for k, v in zip(kv[::2], kv[1::2]):
                part["bend"][k] = float(v)
        mi = re.search(r"inner side of the bend faces ([a-z' ]+?) on the finished toy", t)
        if mi:
            part["_bend_inner"] = mi.group(1).strip()
        mt = re.search(r"inner side of the bend faces[^(]*\(([-\d., ]+)\)", t)
        if mt:
            part["_bend_vec"] = [float(x) for x in mt.group(1).split(",")]
    if re.search(r"(the piece is oval|worked around a chain)", t):
        m = re.search(r"(?:runs|running) (?:vertically, )?(side to side|front to back|top to bottom|up and down)", t)
        part["_chain_lies"] = {"side to side": "x", "front to back": "z", "top to bottom": "y",
                               "up and down": "y"}.get(m.group(1) if m else "", None)
    if re.search(r"two circles \(front and back\)", t):
        part["_edge"] = True
    if re.search(r"a flat (circle|oval)", t):
        part["_applique"] = True
    if part.get("stuff") == "none" and part["end"] == "open":
        part.pop("stuff")
    return part


# ---------------------------------------------------------------------------------------------------------------
# placements (our writer's wording)

LINES = {"centre front line": "front", "centre back line": "back", "centre top line": "up",
         "centre bottom line": "down", "line on the toy's left side": "left", "line on the toy's right side": "right"}
SIDES = {"toward the toy's left": "left", "toward the toy's right": "right", "upward": "up", "downward": "down",
         "toward the front": "front", "toward the back": "back"}
OPP = {"left": "right", "right": "left", "up": "down", "down": "up", "front": "back", "back": "front"}
WORLD = {"left": (1.0, 0, 0), "right": (-1.0, 0, 0), "up": (0, 1.0, 0), "down": (0, -1.0, 0), "front": (0, 0, 1.0),
         "back": (0, 0, -1.0)}

_POS = re.compile(
    r"(?P<where>between rounds (?P<a>\d+) and (?P<b>\d+)|in round (?P<r>[\d.]+)|at the centre of the last round|"
    r"at the centre of the magic ring)"
    r"(?:,\s*(?:on the (?P<on>centre front line|centre back line|centre top line|centre bottom line|line on the toy's "
    r"left side|line on the toy's right side)|(?P<k>[\d.]+) st from the (?P<from>centre front line|centre back line|"
    r"centre top line|centre bottom line|line on the toy's left side|line on the toy's right side),\s*"
    r"(?P<side>toward the toy's left|toward the toy's right|upward|downward|toward the front|toward the back)))?")


_POS_EXACT = re.compile(r"\s*<!-- (?:st ([\d.]+) (left|right|up|down|front|back))? ?(?:rnd ([\d.]+))? -->")


def parse_pos(s):
    m = _POS.search(s)
    if not m:
        return None
    if m.group("where").startswith("at the centre of the last"):
        return {"pole": "end"}
    if m.group("where").startswith("at the centre of the magic"):
        return {"pole": "start"}
    if m.group("a"):
        rnd = (int(m.group("a")) + int(m.group("b"))) / 2.0
        count_round = int(m.group("a"))
    else:
        rnd = float(m.group("r"))
        count_round = int(round(rnd))
    out = {"round": rnd, "count_round": count_round, "span": m.span()}
    if m.group("on"):
        out.update(line=LINES[m.group("on")], k=0.0, side=None)
    elif m.group("k"):
        out.update(line=LINES[m.group("from")], k=float(m.group("k")), side=SIDES[m.group("side")])
    else:
        out.update(line=None, k=0.0, side=None)
    # the writer shows whole / half stitches and keeps the exact tenth in a comment right after the position
    h = _POS_EXACT.match(s, m.end())
    if h and h.group(1) and out.get("line") is not None:
        out.update(k=float(h.group(1)), side=h.group(2))
    if h and h.group(3):
        out["round"] = float(h.group(3))          # the exact round (23.2 is written "in round 23")
    return out


def az_of(pos, ref, plus, n):
    """Inverse of pattern_text.position: (line, k st, side) on a piece whose az 0 is `ref` and +az is `plus`."""
    if pos.get("line") is None:
        return 0.0
    k = pos["k"] / n * 360.0
    if pos["line"] == ref:
        base, sgn_k = 0.0, 1
    elif pos["line"] == OPP[ref]:
        base, sgn_k = 180.0, -1
    elif pos["line"] == plus:
        base, sgn_k = 90.0, None
    else:
        base, sgn_k = -90.0, None
    if pos["side"] is None or k == 0:
        return base
    s = 1.0 if pos["side"] == plus else -1.0
    if sgn_k is None:     # a side line: offsets toward ref/opp
        s2 = 1.0 if pos["side"] == ref else -1.0
        return base - math.copysign(1, base) * s2 * k
    return s * (base + sgn_k * k) if base == 0 else s * (180.0 - k)


ANG = (r"(straight up|straight down|(?:level|(\d+(?:\.\d+)?)° (up|down)), (?:straight ahead|straight back|"
       r"turned (\d+(?:\.\d+)?)° to the toy's (left|right)))")


def ang_vec(m):
    """A match of ANG -> unit vector (inverse of pattern_text.ang_text)."""
    whole = m.group(1)
    if whole == "straight up":
        return [0.0, 1.0, 0.0]
    if whole == "straight down":
        return [0.0, -1.0, 0.0]
    el = 0.0 if whole.startswith("level") else float(m.group(2)) * (1 if m.group(3) == "up" else -1)
    if "straight ahead" in whole:
        hd = 0.0
    elif "straight back" in whole:
        hd = 180.0
    else:
        hd = float(m.group(4)) * (1 if m.group(5) == "left" else -1)
    e, h = math.radians(el), math.radians(hd)
    return [round(math.cos(e) * math.sin(h), 5), round(math.sin(e), 5), round(math.cos(e) * math.cos(h), 5)]


_EXACT_ANG = {"axis (from the magic ring to the last round)": "root_axis", "axis": "dir",
              "round-start line faces": "front", "seam line": "seam_line", "inner side of the bend faces": "bend",
              "its top (round-start line) faces": "front", "front face looks": "face", "it stands out": "dir",
              "the cord leaves": "dir", "curling toward": "toward", "rounds grow": "dir", "pressed toward": "sink_dir"}
_EXACT_NUM = [(r"pressed in ([-\d.]+) mm", "sink"), (r"seam blends over ([\d.]+) rounds", "conform_span"),
              (r"padded with stuffing ([\d.]+) mm thick", "pad"), (r"sewn in ([\d.]+) mm deep", "inset"),
              (r"raised ([-\d.]+) mm", "lift"), (r"rim brought forward ([\d.]+) mm", "cup"),
              (r"curls ([-\d.]+)°", "curl_deg"), (r"cord ([\d.]+) mm thick", "thickness_mm"),
              (r"links ([\d.]+) mm", "link_mm"), (r"^turned ([-\d.]+)°$", "roll"),
              (r"ring moved ([-\d.]+) mm along the piece", "shift")]


def parse_exact(rest):
    """The writer's 'Exact placement: a; b; c.' clause -> dict of engine values."""
    m = re.search(r"Exact placement: (.+?)\.(?:\s|$)", rest)
    out = {}
    if not m:
        return out
    for bit in m.group(1).split("; "):
        bit = bit.strip()
        done = False
        for lead, key in sorted(_EXACT_ANG.items(), key=lambda kv: -len(kv[0])):
            if bit.startswith(lead + " "):
                ma = re.match(ANG, bit[len(lead) + 1:])
                if ma:
                    out[key] = ang_vec(ma)
                    done = True
                break
        if done:
            continue
        mp = re.match(r"pinched (\d+) % over the last ([\d.]+) rounds", bit)
        if mp:
            out["pinch"] = int(mp.group(1)) / 100; out["pinch_rounds"] = float(mp.group(2))
            continue
        mo = re.match(r"offset ([-\d.]+), ([-\d.]+), ([-\d.]+) mm", bit)
        if mo:
            out["pos"] = [float(mo.group(i)) for i in (1, 2, 3)]
            continue
        if bit == "rim not pressed to the base":
            out["conform"] = False; continue
        if bit == "base rim not pulled in":
            out["conform_parent"] = False; continue
        for rx, key in _EXACT_NUM:
            mn = re.search(rx, bit)
            if mn:
                out[key] = float(mn.group(1))
                break
    return out


def parse_dir(s):
    """'It points forward and outward and slightly down (tilt: 25° below level, heading 40° toward the toy's left)'."""
    m = re.search(r"\(direction ([-\d.]+), ([-\d.]+), ([-\d.]+)\)", s)
    if m:
        return [float(m.group(i)) for i in (1, 2, 3)], True
    m = re.search(r"It points ([a-z' ,]+?)(?:\.|\(|$)", s)
    if not m:
        return None, False
    words = m.group(1)
    v = np.zeros(3)
    first = True
    for part in re.split(r"\s+and\s+", words):
        part = part.strip()
        w = 0.45 if part.startswith("slightly") else (1.0 if first else 0.8)
        part = part.replace("slightly ", "")
        key = {"up": "up", "down": "down", "forward": "front", "backward": "back", "toward the toy's left": "left",
               "toward the toy's right": "right", "outward": "out", "inward": "in"}.get(part)
        if key is None:
            continue
        if key in ("out", "in"):
            v += w * np.array([1.0 if key == "out" else -1.0, 0, 0])    # the left copy is described
        else:
            v += w * np.array(WORLD[key])
        first = False
    if np.linalg.norm(v) < 1e-6:
        return None, False
    return list(v / np.linalg.norm(v)), False


# ---------------------------------------------------------------------------------------------------------------
# incremental placement (needs the parent's frame to turn "k st from the centre front line" into an azimuth)

_PD_CACHE = {}


def _cached_piecedef():
    from . import toy as toymod
    from .piece import PieceDef

    class CachedPieceDef(PieceDef):
        def __new__(cls, pid, d, yarn):
            key = json.dumps([pid, d, yarn], sort_keys=True, default=str)
            if key in _PD_CACHE:
                return _PD_CACHE[key]
            obj = PieceDef.__new__(PieceDef)
            PieceDef.__init__(obj, pid, d, yarn)
            _PD_CACHE[key] = obj
            return obj
    return toymod, CachedPieceDef


def build_toy(spec):
    """Toy(spec) with stuffing solves cached across the incremental placements."""
    toymod, C = _cached_piecedef()
    orig = toymod.PieceDef
    toymod.PieceDef = C
    try:
        return toymod.Toy(spec)
    finally:
        toymod.PieceDef = orig


def _frame_words(inst):
    from .pattern_text import _ref_dirs
    return _ref_dirs(inst)


# ---------------------------------------------------------------------------------------------------------------
# our assembly / face wording

def _title_ids(title, by_name):
    """'Inner ear → ear' / 'Head onto the body.' / 'Leg (front)' -> (child part id, qualifier, parent name)."""
    t = _strip_md(title).strip(" .")
    m = re.match(r"(.+?)\s*(?:->|→)\s*(.+)$", t) or re.match(r"(.+?)\s+onto the\s+(.+)$", t)
    child, parent = (m.group(1), m.group(2)) if m else (re.sub(r"\s+is the base.*$", "", t), None)
    q = None
    mq = re.match(r"(.+?)\s*\((.+?)\)\s*$", child)
    if mq:
        child, q = mq.group(1), mq.group(2)
    return child.strip(), q, (parent.strip() if parent else None)


def _lookup(name, by_name):
    k = name.lower().strip()
    k = re.sub(r"^(left|right)\s+", "", k)
    if k in by_name:
        return by_name[k]
    k2 = k.rstrip("s")
    for nm, pid in by_name.items():
        if nm.rstrip("s") == k2:
            return pid
    return part_id_of(name)


def parse_our_assembly(lines, by_name, parts):
    """Numbered steps -> assembly entries with placement still in words (resolved later on the built parents)."""
    steps = []
    text = []
    # many copies of one piece are one written step for the crocheter, with the exact steps kept as hidden lines
    # below it: read the hidden steps, skip the summary
    unfolded = []
    for ln in lines:
        mh = re.match(r"^<!-- step: (.*) -->\s*$", ln)
        if mh:
            unfolded.append(mh.group(1).replace("<!~~", "<!--").replace("~~>", "-->"))
        elif "<!-- grouped -->" not in ln:
            unfolded.append(ln)
    lines = unfolded
    for ln in lines:
        if re.match(r"^\d+\.\s", ln):
            text.append(ln)
        elif text and not ln.startswith(("#", "-")):
            text[-1] += " " + ln
    for ln in text:
        m = re.match(r"^\d+\.\s+\*\*(.+?)\*\*(.*)$", ln)
        if not m:
            continue
        title, rest = m.group(1), m.group(2)
        child_nm, q, parent_nm = _title_ids(title, by_name)
        if q and f"{child_nm} ({q})".lower() in by_name:
            # the bracket is part of the piece's own name ("Crest feather (small)"), not an instance qualifier
            pid, q = by_name[f"{child_nm} ({q})".lower()], None
        else:
            pid = _lookup(child_nm, by_name)
        st = {"part": pid, "q": q, "text": rest, "title": title}
        if parent_nm:
            pm = re.match(r"(left|right)\s+(.+)$", parent_nm.lower())
            st["on_side"] = ("L" if pm.group(1) == "left" else "R") if pm else None
            st["on_part"] = _lookup(pm.group(2) if pm else parent_nm, by_name)
        if "is the base" in rest or "is the base" in title:
            st["kind"] = "root"
            m2 = re.search(r"magic ring is (at the bottom|at the top|at the front|at the back)", rest)
            st["start_at"] = {"at the bottom": "down", "at the top": "up", "at the front": "front",
                              "at the back": "back"}.get(m2.group(1) if m2 else "at the bottom", "down")
        elif "onto the" in title:
            st["kind"] = "onto_opening"
        elif " like a hat: " in rest:
            st["kind"] = "worn"
        elif re.search(r"the chain ring \(\d+ st\) lies along (?:round|the edge of the last round)", rest):
            st["kind"] = "around"
            mr = re.search(r"lies along round ([\d.]+) of the", rest)
            st["ring_round"] = float(mr.group(1)) if mr else None
            st["fall"] = "hangs from the ring" in rest
            mo_ = re.search(r"Put the [\w ]+? around the ((?:left |right )?[\w ]+?):", rest)
            if mo_ and not st.get("on_part"):
                st["on_part"] = _lookup(mo_.group(1), by_name)
        elif "worked flat in turned rows" in rest:
            st["kind"] = "flat"
        elif re.search(r"with [\w ]+, ch \d+ \(", rest) and "Sew its start" in rest:
            st["kind"] = "cord"
        elif ("Stand the joined double circle on its edge" in rest or "Stand the dish on its rim" in rest
              or "Stand the motif on its edge" in rest):
            st["kind"] = "by_edge"
        elif "Sew it flat all round its edge" in rest:
            st["kind"] = "applique"
        elif "by its closed end" in rest:
            st["kind"] = "closed_end"
        elif "by its start" in rest:
            st["kind"] = "by_start"
        else:
            st["kind"] = "by_opening"
        st["mirror"] = bool(re.search(r"\((?:make 2|2 - one on each [^)]*)\)", rest))
        st["on_pair"] = bool(re.search(r"\(2 - one on each", rest))
        mo_ = re.search(r"\(2 - one on each ([\w ]+?)\)", rest)
        if mo_ and not st.get("on_part"):
            st["on_part"] = _lookup(mo_.group(1), by_name)
        cm = re.search(r"Centre: (.+?)(?:\.\s|\s\(the one|$)", rest)
        st["pos"] = parse_pos(cm.group(1)) if cm else parse_pos(rest)
        st["dir"], st["dir_exact"] = parse_dir(rest)
        st["exact"] = parse_exact(rest)
        mra = re.search(r"<!-- rim arc ([-\d.]+) ([-\d.]+) -->", rest)
        if mra:
            st["exact"]["rim_arc"] = [float(mra.group(1)), float(mra.group(2))]
        if "dir" in st["exact"]:
            st["dir"] = st["exact"]["dir"]
        if st["exact"] and "seam_line" not in st["exact"]:
            st.pop("seam", None)   # a flattened piece: its opening is a line anyway, no extra pinch
        mf = re.search(r"in a straight line (side to side|front to back|top to bottom)", rest)
        if mf or "Flatten the opening" in rest:
            st["seam"] = "flat"
            st["seam_axis"] = {"side to side": "x", "front to back": "z", "top to bottom": "y"}.get(
                mf.group(1) if mf else "", None)
            ms = re.search(r"seam line \(([-\d., ]+)\)", rest)
            if ms:
                st["seam_vec"] = [float(x) for x in ms.group(1).split(",")]
            mp = re.search(r"pinched ([\d.]+) ?% over ([\d.]+) rounds", rest)
            if mp:
                st["pinch"] = float(mp.group(1)) / 100; st["pinch_rounds"] = float(mp.group(2))
        for key, rx in (("sink", r"pressed in ([-\d.]+) mm"), ("roll", r"turned ([-\d.]+)° about its own axis"),
                        ("lift", r"raised ([-\d.]+) mm"), ("pad", r"stuffed ([\d.]+) mm thick"),
                        ("inset", r"sewn in ([\d.]+) mm"), ("cup", r"rim brought forward ([\d.]+) mm"),
                        ("conform_span", r"seam blends over ([\d.]+) rounds")):
            mk = re.search(rx, rest)
            if mk:
                st[key] = float(mk.group(1))
        mo = re.search(r"centre moved ([-\d.]+), ([-\d.]+) mm", rest)
        if mo:
            st["offset_mm"] = [float(mo.group(1)), float(mo.group(2))]
        if re.search(r"long side along the (?:ear|piece|parent)", rest):
            st["along"] = "parent"
        if "Drape it over the curve" in rest:
            st["drape"] = True
        if "<!-- join crochet -->" in rest:
            st["join"] = "crochet"            # no-sew: crocheted into the parent's round (amikit/nosew.py)
        if "sew it on by its straight side (the row ends that have no decreases)" in rest:
            st["sew"] = "side"
        mfv = re.search(r"facing \(([-\d., ]+)\)", rest)
        if mfv:
            st["face_vec"] = [float(x) for x in mfv.group(1).split(",")]
        mfr = re.search(r"front reference \(([-\d., ]+)\)", rest)
        if mfr:
            st["front_vec"] = [float(x) for x in mfr.group(1).split(",")]
        if st["kind"] in ("flat", "cord") and not st.get("on_part"):
            mp_ = re.search(r"(?:Sew its start to|Sew it to|along the curve of) the ((?:left |right )?[\w ]+?"
                            r"(?: \([^)]*\))?)[:,]", rest)
            if mp_:
                pm = re.match(r"(left|right)\s+(.+)$", mp_.group(1).lower())
                st["on_side"] = ("L" if pm.group(1) == "left" else "R") if pm else None
                st["on_part"] = _lookup(pm.group(2) if pm else mp_.group(1), by_name)
        if st["kind"] == "flat":
            mw = re.search(r"ch (\d+); Row 1: starting in the 2nd ch from hook, (\d+) sc", rest)
            mr_ = re.search(r"Rows 2-(\d+):", rest) or re.search(r"Row (2):", rest)
            st["flat"] = {"width_st": int(mw.group(2)) if mw else 6, "rows": int(mr_.group(1)) if mr_ else 1}
            st["pinch"] = 0.85 if "Wrap the middle" in rest else 0
            st["conform"] = "Lay it along the curve" in rest
            mc = re.search(r"with ([\w ]+?), ch", rest)
            st["colour"] = mc.group(1).strip().replace(" ", "_") if mc else "black"
        if st["kind"] == "cord":
            mc = re.search(r"with ([\w ]+?), ch (\d+) \(~?\s*([\d.]+) mm\)", rest)
            st["colour"] = mc.group(1).strip().replace(" ", "_") if mc else "black"
            st["cord"] = {"length_mm": float(mc.group(3)) if mc else 40.0,
                          "slip_back": "sl st back" in rest}
            for key, rx in (("thickness_mm", r"cord ([\d.]+) mm thick"), ("curl_deg", r"curls ([-\d.]+)°"),
                            ("link_mm", r"links ([\d.]+) mm")):
                mk = re.search(rx, rest)
                if mk:
                    st["cord"][key] = float(mk.group(1))
            mt = re.search(r"curls [-\d.]+° toward \(([-\d., ]+)\)", rest)
            if mt:
                st["toward"] = [float(x) for x in mt.group(1).split(",")]
        steps.append(st)
    return steps


def _pt_list(s):
    return [parse_pos(x) for x in re.findall(r"\(((?:between|in round|at the centre)[^()]*(?:\([^()]*\))?[^()]*)\)", s)]


def parse_face_exact(rest):
    """The writer's ' Exact: sunk 45 %; thread 1.1 mm thick; ...' clause of a face item."""
    m = re.search(r" Exact: (.+?)\.\s*(?:-->)?\s*$", rest)
    out = {}
    if not m:
        return out
    for bit in m.group(1).split("; "):
        for rx, key, f in ((r"^sunk (\d+) %$", "sink", 0.01), (r"^aspect ([\d.]+)$", "aspect", 1),
                           (r"^pupil (\d+) %$", "pupil", 0.01), (r"^thread ([\d.]+) mm thick$", "thickness_mm", 1),
                           (r"^(\d+) strands$", "strands", None), (r"^raised ([\d.]+) mm$", "lift_mm", 1),
                           (r"^strength (\d+) %$", "strength", 0.01), (r"^droop (\d+) %$", "droop", 0.01),
                           (r"^spread (\d+) %$", "spread", 0.01), (r"^curl ([\d.]+)°$", "curl", 1),
                           (r"^seed (\d+)$", "seed", None)):
            mm = re.match(rx, bit)
            if mm:
                out[key] = int(mm.group(1)) if f is None else round(float(mm.group(1)) * f, 4)
                break
        else:
            mm = re.match(r"^(iris|colour) ([\w ]+?)(?: `#[0-9a-fA-F]{6}`)?$", bit)
            if mm:
                out[mm.group(1)] = mm.group(2).strip().replace(" ", "_")
            elif bit in ("doubled", "single strands"):
                out["doubled"] = bit == "doubled"
    return out


def parse_our_face(lines):
    items = []
    for ln in lines:
        m = re.match(r"^-\s+\*\*(.+?)\*\*(.*)$", ln)
        if not m:
            continue
        title, rest = m.group(1), m.group(2)
        host = (re.search(r"\(on the ([\w ]+? \([\w ]+\))\)", rest) or re.search(r"\(on the ([\w ]+?)\)", rest)
                or re.search(r"\(on (?:the )?([\w ]+?)[,)]", rest))
        it = {"title": title, "text": rest, "host": host.group(1) if host else None}
        it["pair"] = bool(re.search(r"\(2, symmetric|both sides|\(2: one on each|2 ovals|symmetric:", rest))
        it["on_pair"] = "one on each" in rest
        tl = title.lower()
        if tl.startswith("safety eye"):
            it["type"] = "safety_eye"
            # a size between two shop sizes is written '12–13' with the exact size in a comment
            ms = re.search(r"<!-- ([\d.]+) mm -->", rest) or re.search(r"([\d.]+) mm", rest)
            it["size_mm"] = float(ms.group(1)) if ms else 9
            it["highlight"] = "glint" in title.lower() or "glint" in rest[:30]
            ma = re.search(r"(between rounds (\d+) and (\d+)|in round ([\d.]+)), (\d+) st apart", rest)
            if ma:
                it["pair"] = True
                it["pos"] = parse_pos(ma.group(1) + ", on the centre front line")
                mk = re.search(r"about ([\d.]+) st each side of the ([a-z' ]+line)", rest)
                it["pos"]["k"] = float(mk.group(1)) if mk else int(ma.group(5)) / 2
                it["pos"]["line"] = LINES.get(mk.group(2), "front") if mk else "front"
                it["pos"]["side"] = "left"
                it["pos"]["eye_pair"] = True
                mx = re.search(r"<!-- eyes exact: rnd ([\d.]+) st ([\d.]+) -->", rest)
                if mx:
                    it["pos"]["round"] = float(mx.group(1)); it["pos"]["k"] = float(mx.group(2))
            else:
                it["pos"] = parse_pos(rest)
        elif tl.startswith("safety nose"):
            it["type"] = "safety_nose"
            ms = re.search(r"([\d.]+)x([\d.]+) mm", rest) or re.search(r"([\d.]+) mm", rest)
            it["size_mm"] = [float(ms.group(1)), float(ms.group(2))] if ms and ms.lastindex == 2 else (
                float(ms.group(1)) if ms else 10)
            it["shape"] = "triangle" if "triangle" in rest else "oval"
            it["pos"] = parse_pos(rest)
        elif tl.startswith("embroidery"):
            it["type"] = "stitch"
            mc = re.search(r"in ([\w ]+?) thread", title + rest)
            it["colour"] = mc.group(1).strip().replace(" ", "_") if mc else "black"
            it["points"] = _pt_list(rest)
        elif tl.startswith(("satin", "embroidered eyes", "eye glint")):
            it["type"] = "satin"
            it["role"] = "eye" if tl.startswith("embroidered eyes") else "glint" if tl.startswith("eye glint") else None
            mc = re.search(r"in ([\w ]+?) thread", rest)
            it["colour"] = mc.group(1).strip().replace(" ", "_") if mc else "black"
            msh = re.search(r"\ba (triangle|oval|heart)\b", rest)
            it["shape"] = msh.group(1) if msh else "triangle"
            mw = re.search(r"([\d.]+) mm wide", rest); mh = re.search(r"([\d.]+) mm tall", rest)
            if mw:
                it["width_mm"] = float(mw.group(1))
            if mh:
                it["height_mm"] = float(mh.group(1))
            mw = re.search(r"(\d+(?:\.\d+)?) st wide", rest); mh = re.search(r"(\d+(?:\.\d+)?) rounds tall", rest)
            if mw and "width_mm" not in it:
                it["width_st"] = float(mw.group(1))
            if mh and "height_mm" not in it:
                it["height_rounds"] = float(mh.group(1))
            mr_ = re.search(r"turned ([\d.]+)° (clockwise|anticlockwise)", rest)
            if mr_:
                it["rotate"] = float(mr_.group(1)) * (1 if mr_.group(2) == "clockwise" else -1)
            mo = re.search(r"then moved (.+?) \(looking", rest)
            if mo:
                dx = re.search(r"([\d.]+) mm to the (right|left)", mo.group(1))
                dy = re.search(r"([\d.]+) mm (up|down)", mo.group(1))
                it["offset_mm"] = [(float(dx.group(1)) * (1 if dx.group(2) == "right" else -1)) if dx else 0.0,
                                   (float(dy.group(1)) * (1 if dy.group(2) == "up" else -1)) if dy else 0.0]
            for key, rx in (("thickness_mm", r"thread ([\d.]+) mm thick"), ("strands", r"(\d+) strands"),
                            ("lift_mm", r"raised ([\d.]+) mm")):
                mk = re.search(rx, rest)
                if mk:
                    it[key] = float(mk.group(1)) if key != "strands" else int(mk.group(1))
            mcen = re.search(r"centred (.+?)(?:, then moved|;|$)", rest)
            it["pos"] = parse_pos(mcen.group(1) if mcen else rest)
        elif tl.startswith("blush"):
            it["type"] = "blush"
            ms = re.search(r"about ([\d.]+) mm", title + rest)
            it["size_mm"] = float(ms.group(1)) if ms else 8
            it["pair"] = "both sides" in rest
            mst = re.search(r"strength ([\d.]+)", rest)
            if mst:
                it["strength"] = float(mst.group(1))
            mc = re.search(r"colour ([\w]+)", rest)
            if mc:
                it["colour"] = mc.group(1)
            it["pos"] = parse_pos(rest)
        elif tl.startswith("felt"):
            it["type"] = "felt"
            mc = re.match(r"\s*\(([\w ]+)\)", rest)
            it["colour"] = mc.group(1).strip().replace(" ", "_") if mc else "black"
            ms = re.search(r"([\d.]+)x([\d.]+) mm", rest)
            it["size_mm"] = [float(ms.group(1)), float(ms.group(2))] if ms else [10, 7]
            mr_ = re.search(r"tilted ([\d.]+)° (clockwise|anticlockwise)", rest)
            if mr_:
                it["rotate"] = float(mr_.group(1)) * (1 if mr_.group(2) == "clockwise" else -1)
            it["pos"] = parse_pos(rest.split("glue or sew on:")[-1])
        elif tl.startswith("strap"):
            it["type"] = "strap"
            mc = re.search(r"with ([\w ]+?), ch", rest)
            it["colour"] = mc.group(1).strip().replace(" ", "_") if mc else "black"
            mt = re.search(r"strap ([\d.]+) mm thick", rest)
            if mt:
                it["thickness_mm"] = float(mt.group(1))
            it["points"] = _pt_list(rest.split("Sew it along")[-1])
        elif "pieces of yarn" in rest:
            it["type"] = "fringe"
            mn = re.search(r"cut (\d+) pieces of yarn \(([^)]*)\)", rest)
            it["count"] = int(mn.group(1)) if mn else 12
            it["colours"] = [c.strip().replace(" ", "_") for c in mn.group(2).split(",")] if mn else ["black"]
            ml = re.search(r"Trim to about ([\d.]+) mm", rest)
            it["length_mm"] = float(ml.group(1)) if ml else 20
            along = rest.split(" along ", 1)[-1].split(", evenly spaced")[0]
            it["points"] = [parse_pos(x) for x in re.split(r"\s*->\s*", along)]
            for key, rx in (("droop", r"droop ([\d.]+)"), ("curl", r"curl ([\d.]+)"), ("spread", r"spread ([\d.]+)")):
                mk = re.search(rx, rest)
                if mk:
                    it[key] = float(mk.group(1))
            if "curl" not in it and "Curl the strands" in rest:
                it["curl"] = 120.0
            elif "curl" not in it and "soft wave" in rest:
                it["curl"] = 45.0
            if title.lower() != "fringe":
                it["title_name"] = title
        else:
            continue
        ex = parse_face_exact(rest)
        if it["type"] == "fringe":
            it.pop("curl", None)
        it.update(ex)
        items.append(it)
    return items


# ---------------------------------------------------------------------------------------------------------------
# document -> spec

def to_spec(text, log=None, name=None):
    log = [] if log is None else log
    doc = split_document(text)
    colours, yarn = parse_materials(doc)
    ours = bool(re.search(r"^## Assembly", text, re.M)) and bool(re.search(r"^### Face", text, re.M) or
                                                                   "is the base" in text)
    if "stitch_mm" not in yarn:
        g = gauge_from_hook(yarn)
        yarn = {**g, **yarn}
        log.append(f"gauge not stated: assumed {g['stitch_mm']} mm per sc, {g['round_mm']} mm per round "
                   f"(typical for a {yarn.get('hook_mm', 2.5)} mm hook)")
    yarn.setdefault("weight", "DK"); yarn.setdefault("hook_mm", 2.5)
    yarn.setdefault("thickness_mm", round(0.47 * yarn["stitch_mm"], 2))
    parts, by_name = [], {}
    used_ids = {}
    for p in doc["pieces"]:
        hd = re.sub(r"\s*-\s*make \d+.*$", "", p["name"]).strip()
        if re.fullmatch(r"piece \d+", hd) and len(doc["pieces"]) == 1:
            hd = "body"
            log.append("one piece with no name: read as the body (worked in one piece)")
        hd = re.sub(r"\s*\((?:left and right|identical|mirror)[^)]*\)\s*$", "", hd)
        pid = part_id_of(hd)
        if pid in used_ids:
            used_ids[pid] += 1
            pid = f"{pid}{used_ids[pid]}"
        else:
            used_ids[pid] = 1
        p["id"] = pid
        by_name[hd.lower()] = pid
        mm_ = re.search(r"Flat motif - ([a-z ]+?) \(", " ".join(p.get("pre", [])))
        if mm_:
            part = motif_part(p, mm_.group(1).strip().replace(" ", "_"), log)
            if part is not None:
                part["name_en"] = hd
                part["_make"] = p["make"]
                parts.append(part)
                continue
        try:
            mplain = re.search(r"\d+ plain in ([a-z][\w ]*?) \(same rounds", " ".join(p.get("pre", [])))
            part = piece_part(p, colours, log, main=mplain.group(1).replace(" ", "_") if mplain else None)
        except ParseError as ex:
            log.append(f"{p['name']}: skipped ({ex})")
            continue
        finish_part(part, p, log)
        if p["rows"]:
            log.append(f"{p['name']}: worked in rows — read as rounds (flat pieces are approximated)")
        part["name_en"] = hd
        part["_make"] = p["make"]
        parts.append(part)
    # colours used but not listed
    for part in parts:
        for c in [part["colour"]] + [z.get("colour") for z in part.get("colours", [])] + \
                 [c for z in part.get("colours", []) for c, _ in z.get("runs", [])]:
            if c and c not in colours:
                colours[c] = NAMED_HEX.get(c, NAMED_HEX.get(c.split("_")[-1], "#c8b8a8"))
                if c not in NAMED_HEX:
                    log.append(f"colour {c!r}: no hex in the pattern, assumed {colours[c]}")
    spec = {"name": name or re.sub(r"\W+", "_", (doc["title"] or "toy").lower()).strip("_")[:40],
            "title": doc["title"] or "Toy", "yarn": yarn, "colours": colours, "parts": parts, "assembly": [],
            "face": []}
    if ours:
        _our_assembly(spec, doc, by_name, log)
    else:
        from .reverse_free import free_assembly
        free_assembly(spec, doc, by_name, log)
    for part in spec["parts"]:
        for k in [k for k in part if k.startswith("_")]:
            part.pop(k)
    return spec, log


def _vec_of(word, side_sign=1.0):
    v = np.array(WORLD[word], float)
    return v


def _our_assembly(spec, doc, by_name, log):
    steps = parse_our_assembly(doc["assembly"], by_name, spec["parts"])
    parts = {p["id"]: p for p in spec["parts"]}
    for st in steps:
        if st["kind"] == "flat" and st["part"] not in parts:
            spec["parts"].append({"id": st["part"], "flat": st["flat"], "colour": st["colour"],
                                  **({"pinch": st["pinch"]} if st["pinch"] else {})})
            parts[st["part"]] = spec["parts"][-1]
        if st["kind"] == "cord" and st["part"] not in parts:
            spec["parts"].append({"id": st["part"], "cord": st["cord"], "colour": st["colour"]})
            parts[st["part"]] = spec["parts"][-1]
    entries = []
    names = {}
    spec["assembly"] = entries
    toy = None
    for st in steps:
        pid = st["part"]
        if pid not in parts:
            log.append(f"assembly: unknown piece {pid!r} ({st['title']}) — skipped")
            continue
        part = parts[pid]
        e = {"part": pid}
        qq = re.sub(r'[^a-z0-9]+', '_', (st.get("q") or "").lower()).strip('_')
        nm = pid if not qq else (f"{qq}_{pid}" if re.fullmatch(r"[a-z]+", qq) else f"{pid}_{qq}")
        if nm != pid or pid in names:
            if nm in names:
                nm = f"{nm}{len([x for x in names if x.startswith(nm)]) + 1}"
            e["name"] = nm
        names[nm] = st
        ex = st.get("exact", {})
        if st["kind"] == "root":
            ax = {"down": [0, 1, 0], "up": [0, -1, 0], "front": [0, 0, -1], "back": [0, 0, 1]}[st["start_at"]]
            e["axis"] = ex.get("root_axis", ax)
            if "pos" in ex:
                e["pos"] = ex["pos"]
            entries.append(e)
            toy = None
            continue
        on_part = st.get("on_part")
        on_name = _entry_name_for(on_part, entries, st.get("on_side"))
        if on_name is None:
            log.append(f"assembly: parent {on_part!r} of {pid} not placed — skipped")
            continue
        e["on"] = on_name
        if st["mirror"]:
            e["mirror"] = True
        # the parent's frame from the toy built so far
        if toy is None or not _has_inst(toy, on_name, st.get("on_side")):
            toy = build_toy(_clean_spec(spec))
        parent = _parent_inst(toy, on_name, st.get("on_side"))
        if st.get("on_side") and not st["mirror"] and on_name in toy.pairs:
            e["on"] = parent.name                   # "Inner ear (right) -> right ear": that one copy
        if st["kind"] == "around":
            rr = st.get("ring_round")
            e["around"] = True
            e["at"] = {"round": "end" if rr is None else (int(rr) if rr == int(rr) else rr)}
            _apply_exact(e, ex, ("dir", "front", "shift"))
            if st.get("fall"):
                e["fall"] = True
            entries.append(e)
            toy = None
            continue
        if st["kind"] == "onto_opening":
            e["at"] = {"round": "end"}
            e["attach"] = "end"
            e["dir"] = st["dir"] or [0, 1, 0]
            if "sink" in st:
                e["sink"] = st["sink"]
            _apply_exact(e, ex, ("front", "sink", "sink_dir", "conform_span", "conform", "conform_parent"))
        else:
            pos = st["pos"]
            if pos is None:
                log.append(f"assembly: no position for {pid} — skipped")
                continue
            if "pole" in pos:
                e["at"] = {"round": pos["pole"]}
            else:
                ref, plus = _frame_words(parent)
                n = parent.p.counts[min(max(pos["count_round"], 1), parent.p.N) - 1]
                e["at"] = {"round": pos["round"], "az": round(az_of(pos, ref, plus, n), 2)}
            if st["kind"] == "worn":
                e["wear"] = True
                if st.get("dir"):
                    e["dir"] = st["dir"]
                _apply_exact(e, ex, ("dir", "front", "pad"))
                entries.append(e)
                toy = None
                continue
            if st["kind"] == "applique":
                e["applique"] = True
                _apply_exact(e, ex, ("front", "pad", "inset", "lift"))
                if st.get("drape"):
                    e["drape"] = True
                for k in ("lift", "pad", "inset", "roll", "offset_mm", "along", "front_vec"):
                    if k in st:
                        e[{"front_vec": "front"}.get(k, k)] = st[k]
                if "offset_mm" in st:
                    pass
            elif st["kind"] == "by_edge":
                e["attach"] = "edge"
                if st.get("dir"):
                    e["dir"] = st["dir"]
                if st.get("face_vec"):
                    e["face"] = st["face_vec"]
                if "cup" in st:
                    e["cup"] = st["cup"]
                if st.get("sew") and "motif" in part:
                    part["motif"]["sew"] = st["sew"]
                _apply_exact(e, ex, ("face", "dir", "cup", "sink"))
            elif st["kind"] in ("flat", "cord"):
                for k in ("roll", "lift", "conform", "toward"):
                    if k in st and st[k] not in (None, False):
                        e[k] = st[k]
                if st.get("dir"):
                    e["dir"] = st["dir"]
                _apply_exact(e, ex, ("roll", "lift", "dir", "toward"))
                if st["kind"] == "cord":
                    for k in ("curl_deg", "thickness_mm", "link_mm"):
                        if k in ex:
                            parts[pid]["cord"][k] = ex[k]
            else:
                if st["kind"] == "by_start":
                    e["attach"] = "start"
                if st.get("dir"):
                    e["dir"] = st["dir"]
                for k in ("sink", "roll", "conform_span"):
                    if k in st:
                        e[k] = st[k]
                if st.get("front_vec"):
                    e["front"] = st["front_vec"]
                if st.get("seam") == "flat" and (not ex or "seam_line" in ex):
                    # (with an exact clause, only a stated seam line means a pinched flat seam; a flattened
                    # piece's opening is a line anyway)
                    e["seam"] = "flat"
                    if st.get("seam_vec"):
                        e["seam_line"] = st["seam_vec"]
                    elif st.get("seam_axis"):
                        e["seam_line"] = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[st["seam_axis"]]
                    for k in ("pinch", "pinch_rounds"):
                        if k in st:
                            e[k] = st[k]
                _apply_exact(e, ex, ("front", "sink", "sink_dir", "conform_span", "conform", "conform_parent", "pinch",
                                     "pinch_rounds", "rim_arc"))
                if "seam_line" in ex and e.get("seam") == "flat":
                    e["seam_line"] = ex["seam_line"]
                if st.get("join"):
                    e["join"] = "crochet"
                    e["join_fit"] = False         # the written position is where it was crocheted in
                if "bend" in ex:
                    part["_bend_vec"] = None
                    e["bend_toward"] = ex["bend"]
                bt = part.get("_bend_vec")
                if "bend" in ex:
                    pass
                elif bt:
                    e["bend_toward"] = bt
                elif part.get("_bend_inner"):
                    w = part["_bend_inner"]
                    key = {"forward": "front", "backward": "back", "up": "up", "down": "down",
                           "toward the toy's left": "left", "toward the toy's right": "right"}.get(w)
                    if key:
                        e["bend_toward"] = list(WORLD[key])
        entries.append(e)
        toy = None
        # chain-oval orientation: the text says which way the chain lies on the finished toy
        if part.get("_chain_lies") and "oval_axis" not in part and part["rounds"][0].startswith("ch"):
            from .pattern_text import _chain_lie
            t2 = build_toy(_clean_spec(spec))
            inst = t2.inst[_first_inst(t2, e)]
            if _chain_lie(inst) != part["_chain_lies"]:
                part["oval_axis"] = "front"
                t3 = build_toy(_clean_spec(spec))
                if _chain_lie(t3.inst[_first_inst(t3, e)]) != part["_chain_lies"]:
                    part.pop("oval_axis")
    # root chain oval
    for e in entries:
        if "on" not in e:
            part = parts[e["part"]]
            if part.get("_chain_lies") and part["rounds"][0].startswith("ch"):
                from .pattern_text import _chain_lie
                part.pop("oval_axis", None)
                if _chain_lie(build_toy(_clean_spec(spec)).inst[e.get("name", e["part"])]) != part["_chain_lies"]:
                    part["oval_axis"] = "front"
    for p in spec["parts"]:
        if p.get("_edge"):
            pass
    spec["face"] = _our_face(spec, doc, by_name, log)


def _apply_exact(e, ex, keys):
    for k in keys:
        if k in ex:
            e[k] = ex[k]


def _entry_name_for(on_part, entries, on_side):
    for e in reversed(entries):
        if e["part"] == on_part or e.get("name") == on_part:
            return e.get("name", e["part"])
    return None


def _has_inst(toy, on, side):
    try:
        _parent_inst(toy, on, side)
        return True
    except KeyError:
        return False


def _parent_inst(toy, on, side):
    try:
        return toy.inst[toy._side(on, "R" if side == "R" else "L")]
    except KeyError:
        if on in toy.inst:
            return toy.inst[on]
        raise


def _first_inst(toy, e):
    nm = e.get("name", e["part"])
    if e.get("mirror"):
        return toy.pairs[nm][0]
    return nm


def _clean_spec(spec):
    s = copy.deepcopy(spec)
    for p in s["parts"]:
        for k in [k for k in p if k.startswith("_")]:
            p.pop(k)
    return s


def _our_face(spec, doc, by_name, log):
    items = parse_our_face(doc["face"])
    toy = build_toy(_clean_spec(spec))
    out = []
    for it in items:
        host_pid = _lookup(it["host"], by_name) if it.get("host") else "head"
        hs = re.match(r"(left|right) ", it.get("host") or "")
        host_side = ("L" if hs.group(1) == "left" else "R") if hs else None
        host_name = _entry_name_for(host_pid, spec["assembly"], None)
        # one of several copies named in the text ("on the top honeycomb cell" -> entry top_honeycomb_cell)
        slug = re.sub(r"[^a-z0-9]+", "_", (it.get("host") or "").lower()).strip("_")
        if slug and any(e.get("name") == slug for e in spec["assembly"]):
            host_pid, host_side, host_name = slug, None, slug
        if host_name is None:
            log.append(f"face: host {it.get('host')!r} not found — skipped {it['title']}")
            continue
        inst = _parent_inst(toy, host_name, host_side)
        if host_side and host_name in toy.pairs:
            host_name = inst.name
        ref, plus = _frame_words(inst)
        f = {"type": it["type"], "on": host_name}
        if it.get("pair"):
            f["mirror"] = True

        def at_of(pos):
            if "pole" in pos:
                return {"round": pos["pole"]}
            n = inst.p.counts[min(max(pos["count_round"], 1), inst.p.N) - 1]
            return {"round": pos["round"], "az": round(az_of(pos, ref, plus, n), 2)}

        def pt_of(pos):
            a = at_of(pos)
            return [a["round"] if a["round"] not in ("end", "start") else (inst.p.N if a["round"] == "end" else 0.5),
                    a.get("az", 0.0)]
        if it.get("pos"):
            f["at"] = at_of(it["pos"])
            if it["pos"].get("eye_pair"):
                f["at"]["az"] = abs(f["at"]["az"])
        if it.get("points"):
            pts = [p for p in it["points"] if p]
            f["points"] = [pt_of(p) for p in pts]
        for k in ("size_mm", "shape", "colour", "highlight", "aspect", "sink", "pupil", "iris", "width_mm", "height_mm",
                  "width_st", "height_rounds", "rotate", "offset_mm", "thickness_mm", "strands", "lift_mm", "role",
                  "strength", "count", "colours", "length_mm", "droop", "curl", "spread", "seed", "doubled"):
            if k in it and it[k] is not None:
                f[k] = it[k]
        if f["type"] in ("safety_eye", "safety_nose") and f.get("colour") == "black":
            f.pop("colour")
        out.append(f)
    return out


# ---------------------------------------------------------------------------------------------------------------
# CLI

def reverse_file(path, out=None, ref=None, fast=True, build_3d=True, name=None):
    text = Path(path).read_text(encoding="utf-8-sig")
    log = []
    spec, log = to_spec(text, log, name=name)
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        (out / "spec_reversed.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf8")
        (out / "reverse_report.txt").write_text("\n".join(log) or "(nothing assumed, nothing unreadable)",
                                                encoding="utf8")
        if build_3d:
            from .build import build
            build(out / "spec_reversed.json", out, ref=ref, fast=fast)
    return spec, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern"); ap.add_argument("out")
    ap.add_argument("--ref"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--spec-only", action="store_true")
    ap.add_argument("--name")
    a = ap.parse_args()
    spec, log = reverse_file(a.pattern, a.out, ref=a.ref, fast=a.fast, build_3d=not a.spec_only, name=a.name)
    print("\n".join(log[:40]))
