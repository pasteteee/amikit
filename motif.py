"""Shaped flat pieces (motifs): one layer of crochet fabric worked flat, in turned rows or in rounds, whose outline
comes from its written rows. Opt-in: a part with a "motif" key instead of "rounds".

    {"id": "kerchief", "motif": {"kind": "triangle", "width": 24}, "colour": "sage"}
    {"id": "wing", "motif": {"kind": "triangle", "width": 9, "ends": "one", "sew": "side"}}
    {"id": "plate", "motif": {"kind": "leaf", "chain": 9, "tall": "tr"}}
    {"id": "cell", "motif": {"kind": "hexagon", "rounds": 3}}
    {"id": "bag", "motif": {"kind": "half_circle", "rows": 5}}
    {"id": "moon", "motif": {"kind": "crescent", "chain": 12}}
    {"id": "heart", "motif": {"kind": "heart"}}
    {"id": "disc", "motif": {"kind": "circle", "rounds": 4, "edging": {"kind": "shell", "k": 5, "every": 4}}}
    {"id": "mask", "motif": {"kind": "mask", "chain": 8, "rounds": 3}}

Kinds (what a crocheter does):
  triangle     foundation chain, turned rows of sc (hdc, dc), a decrease at one or both ends of every row (`every`
               rows) down to one stitch;
  leaf         a chain; one round along both sides of it with graded stitches (sc, hdc, `tall`..., hdc, sc), 3 sc in
               the last chain, a ch-3 picot at the tip (pointed leaf, petal, dino plate, pangolin scale);
  hexagon      magic ring, rounds of sc with the six increases stacked on the corners (a flat hexagon);
  circle       magic ring, flat rounds of sc (+6 per round, staggered);
  half_circle  magic ring, turned rows of sc, +3 per row (a flat D-shape: a watermelon slice, a bat wing);
  crescent     a chain; one row along one side with graded stitches and increases in the middle: the longer outer
               edge curls the piece into a crescent;
  heart        into a magic ring: ch 3, 3 tr, 3 dc, ch 1, tr, ch 1, 3 dc, 3 tr, ch 3, sl st (two lobes, a point);
  mask         a chain-oval worked in rounds, then a last round of tall stitches (with increases) over the two
               turns and slip stitches in the middle of the top: two lobes joined (a face mask around the eyes).
  star         magic ring, 2p sc; then (ch k, back along the chain: sc, hdc, dc..., skip 1, sl st) x p: p points;
  lobed        into a magic ring: (ch 3, k tall stitches, ch 3, sl st) x lobes (clover, flower, cloud), then optional
               sc rounds around it (sc in the chain spaces, the sl sts skipped);
  petal        a chain; graded stitches rising up one side, a fan of tall stitches in the last chain (a rounded tip),
               back down the other side (a broad petal, an oval wing).
heart `rounds`: sc rounds around the heart (into the stitches only; 2 incs per lobe, 3 sc in the point).
`edging` (circle, hexagon, half_circle, mask): a last round/row of shells ("shell": skip, k dc in one st, skip,
sl st per `every` stitches), picots ("picot": `every` sc then a ch-3 picot), chain points ("point": ch k worked back
with graded stitches, skip 1, sl sts: bat-wing spikes) or feather tips ("feather": sc, hdc... rising to 2 tall in one
st, sl st).

The geometry is derived from the rows: a row of n stitches is n stitch widths long and one stitch height tall
(sc 1, hdc 1.5, dc 2, tr 2.6 x the round height); stitches worked around or along a chain stand perpendicular to
it; increases lengthen the outer edge. A MotifDef quacks like a PieceDef (bands of stitches, flat_points for
appliques, local_points for pieces sewn by an edge), so placement, colour charts and meshing are shared.
"""
from __future__ import annotations
import math
import re

import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator

from .rounds import Round, expand

HT = {"sl": 0.3, "sc": 1.0, "hdc": 1.5, "dc": 2.0, "tr": 2.6}       # stitch height / sc height (one round)
ST_EN = {"sc": "sc", "hdc": "hdc", "dc": "dc", "tr": "tr", "sl": "sl st"}
ST_RU = {"sc": "сбн", "hdc": "пссн", "dc": "ссн", "tr": "сс2н", "sl": "сс"}
TURN_CH = {"sc": 1, "hdc": 2, "dc": 3, "tr": 4}
KIND_EN = {"triangle": "triangle", "leaf": "pointed leaf", "hexagon": "hexagon", "circle": "circle",
           "half_circle": "half circle", "crescent": "crescent", "heart": "heart", "mask": "two-lobed mask",
           "star": "star", "lobed": "lobes round a magic ring", "petal": "rounded petal"}
KIND_RU = {"triangle": "треугольник", "leaf": "заострённый листик", "hexagon": "шестиугольник", "circle": "круг",
           "half_circle": "полукруг", "crescent": "полумесяц", "heart": "сердечко", "mask": "маска из двух долей",
           "star": "звёздочка", "lobed": "мотив из долей", "petal": "округлый лепесток"}
KINDS = tuple(KIND_EN)
EDGING_KINDS = ("circle", "hexagon", "half_circle", "mask")
MOTIF_KEYS = {"kind", "width", "ends", "every", "stitch", "chain", "tall", "tip", "rounds", "rows", "edging", "sew",
              "point_up", "points", "length", "lobes", "stitches"}
LADDER = ["sc", "hdc", "dc", "tr"]
FAN = {"sc": 3, "hdc": 4, "dc": 6, "tr": 7}          # tall stitches in the last chain: a rounded petal tip


class MotifError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# operations: "sc" "hdc" "dc" "tr" "sl" | "inc:<st>:<k>" (k st in one st) | "tog:<st>:<k>" (k st worked together)
# | "skip" | "picot" (ch 3, sl st in the 3rd ch from hook) | "ch:<k>" (a chain space, not counted)

def op_cp(op):
    """(consumed, produced) stitches of one operation."""
    if op in HT:
        return 1, 1
    if op == "skip":
        return 1, 0
    if op == "picot" or op.startswith("ch:"):
        return 0, 0
    if op.startswith("pt:"):                       # a point: ch k, stitches worked back along the chain
        return 0, len(op[3:].split("."))
    if op.startswith("sp:"):                       # k sc into a chain space (a ch-3 post)
        return 0, int(op[3:])
    kind, st, k = op.split(":")
    return (1, int(k)) if kind == "inc" else (int(k), 1)


def op_height(op):
    if op in HT:
        return HT[op]
    if op.startswith(("inc:", "tog:")):
        return HT[op.split(":")[1]]
    if op.startswith("sp:"):
        return HT["sc"]
    return 0.0


def _op_text(op, k, lang):
    """A run of k equal operations."""
    ru = lang == "ru"
    if op in HT:
        return f"{k} {(ST_RU if ru else ST_EN)[op]}"
    if op == "skip":
        return f"пропустите {k} п." if ru else f"skip {k} st"
    if op == "picot":
        one = "пико (3 цеп. п., сс в 3-ю цеп. п. от крючка)" if ru else "picot (ch 3, sl st in the 3rd ch from hook)"
        return one if k == 1 else f"({one}) ×{k}"
    if op.startswith("ch:"):
        n = int(op.split(":")[1]) * k
        return f"{n} цеп. п." if ru else f"ch {n}"
    if op.startswith("pt:"):
        seq = op[3:].split(".")
        inner = ops_text(seq, lang)
        one = (f"зубец ({len(seq) + 1} цеп. п., затем обратно по цепочке, начиная со 2-й цеп. п. от крючка: {inner})"
               if ru else f"point (ch {len(seq) + 1}, then back along the chain from the 2nd ch from hook: {inner})")
        return one if k == 1 else f"({one}) ×{k}"
    if op.startswith("sp:"):
        one = f"{op[3:]} сбн в арку из цепочки" if ru else f"{op[3:]} sc in the chain space"
        return one if k == 1 else f"({one}) ×{k}"
    kind, st, n = op.split(":")
    if kind == "inc":
        if st == "sc" and n == "2":
            one = "пр" if ru else "inc"
            return one if k == 1 else (f"{k} пр" if ru else f"inc ×{k}")
        one = f"{n} {ST_RU[st]} в одну петлю" if ru else f"{n} {st} in one st"
    else:
        if st == "sc" and n == "2":
            one = "уб" if ru else "dec"
            return one if k == 1 else (f"{k} уб" if ru else f"dec ×{k}")
        one = f"{n} {ST_RU[st]} вместе" if ru else f"{st}{n}tog"
    return one if k == 1 else f"({one}) ×{k}"


def _runs(ops):
    out = []
    for o in ops:
        if out and out[-1][0] == o:
            out[-1][1] += 1
        else:
            out.append([o, 1])
    return out


def ops_text(ops, lang):
    """Operations -> text: runs merged, a whole-row repeat written as '(...) ×m'."""
    n = len(ops)
    for size in range(2, n // 2 + 1):
        unit = ops[:size]
        if len(set(unit)) < 2:
            continue
        reps = 0
        while ops[reps * size:(reps + 1) * size] == unit:
            reps += 1
        tail = ops[reps * size:]
        if reps >= 2 and len(tail) < size:
            txt = "(" + ", ".join(_op_text(o, k, lang) for o, k in _runs(unit)) + f") ×{reps}"
            if tail:
                txt += ", " + ", ".join(_op_text(o, k, lang) for o, k in _runs(tail))
            return txt
    return ", ".join(_op_text(o, k, lang) for o, k in _runs(ops))


_ST = r"(sc|hdc|dc|tr|sl st)"


def _parse_item(it):
    """One comma-separated item of our rows (EN) -> operations."""
    it = it.strip()
    m = re.fullmatch(r"\((.+)\)\s*[x×]\s*(\d+)", it)
    if m and not re.fullmatch(r"picot \(ch 3, sl st in the 3rd ch from hook\)", m.group(1).strip()):
        inner = []
        for x in _split(m.group(1)):
            inner += _parse_item(x)
        return inner * int(m.group(2))
    m = re.fullmatch(r"(\d+) " + _ST, it)
    if m:
        st = "sl" if m.group(2) == "sl st" else m.group(2)
        return [st] * int(m.group(1))
    m = re.fullmatch(r"(inc|dec)(?:\s*[x×]\s*(\d+))?", it)
    if m:
        return [("inc:sc:2" if m.group(1) == "inc" else "tog:sc:2")] * int(m.group(2) or 1)
    m = re.fullmatch(r"(\d+) (sc|hdc|dc|tr) in one st", it)
    if m:
        return [f"inc:{m.group(2)}:{m.group(1)}"]
    m = re.fullmatch(r"(sc|hdc|dc|tr)(\d)tog", it)
    if m:
        return [f"tog:{m.group(1)}:{m.group(2)}"]
    m = re.fullmatch(r"skip (\d+) st", it)
    if m:
        return ["skip"] * int(m.group(1))
    if re.fullmatch(r"picot \(ch 3, sl st in the 3rd ch from hook\)", it):
        return ["picot"]
    m = re.fullmatch(r"point \(ch (\d+), then back along the chain from the 2nd ch from hook: (.+)\)", it)
    if m:
        seq = []
        for x in _split(m.group(2)):
            seq += _parse_item(x)
        if len(seq) + 1 != int(m.group(1)) or any(s not in ("sc", "hdc", "dc", "tr") for s in seq):
            raise MotifError(f"cannot read {it!r}")
        return ["pt:" + ".".join(seq)]
    m = re.fullmatch(r"(\d+) sc in the chain space", it)
    if m:
        return [f"sp:{m.group(1)}"]
    m = re.fullmatch(r"ch (\d+)", it)
    if m:
        return [f"ch:{m.group(1)}"]
    raise MotifError(f"cannot read {it!r}")


def _split(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


# phrases around the operations (EN text, RU text, parse regex)
PH = {
    "chain": ("ch {0} (foundation chain)", "наберите цепочку из {0} цеп. п.", r"ch (\d+) \(foundation chain\)"),
    "hook": ("starting in the {0} ch from hook", "начиная с {1}-й цеп. п. от крючка",
             r"starting in the (\d+)(?:st|nd|rd|th) ch from hook"),
    "mr": ("into a magic ring", "в кольцо амигуруми", r"into a magic ring"),
    "turn": ("ch {0} and turn (not counted as a st)", "{0} цеп. п. подъёма, поверните работу (в счёт не входят)",
             r"ch (\d+) and turn \(not counted as a st\)"),
    "other": ("along the other side of the chain", "по другой стороне цепочки", r"along the other side of the chain"),
    "join": ("sl st in the first st to join", "сс в первую петлю, чтобы замкнуть", r"sl st in the first st to join"),
    "ring": ("ch 3 and sl st into the ring, then pull the ring tight", "3 цеп. п. и сс в кольцо, затем затяните кольцо",
             r"ch 3 and sl st into the ring, then pull the ring tight"),
    "ch3": ("ch 3 (the first post, not counted)", "3 цеп. п. (вместо первого столбика, в счёт не входят)",
            r"ch 3 \(the first post, not counted\)"),
    "shift": ("sc {0} more to move the round start to the middle of a long side (not counted)",
              "провяжите ещё {0} сбн, чтобы перенести начало ряда на середину длинной стороны (в счёт не входят)",
              r"sc (\d+) more to move the round start to the middle of a long side \(not counted\)"),
    "round": ("work around the whole edge", "провяжите по всему краю", r"work around the whole edge"),
    "spiral": ("in a spiral", "по спирали", r"in a spiral"),
    "tight": ("pull the ring tight", "затяните кольцо", r"pull the ring tight"),
    "posts": ("starting with the first tr, work into the stitches only (not into the ch-3 posts or the ch-1 spaces)",
              "начиная с первого сс2н, вяжите только в столбики (не в цепочки по краям и не в воздушные петли)",
              r"starting with the first tr, work into the stitches only \(not into the ch-3 posts or the ch-1 "
              r"spaces\)"),
    "curve": ("along the curved edge only (not along the straight edge)",
              "только по закруглённому краю (не по прямому краю)",
              r"along the curved edge only \(not along the straight edge\)"),
}


def _ordinal(k):
    return f"{k}{'st' if k % 10 == 1 and k != 11 else 'nd' if k % 10 == 2 and k != 12 else 'rd' if k % 10 == 3 and k != 13 else 'th'}"


def _phrase(key, args, lang):
    en, ru, _ = PH[key]
    if lang == "ru":
        s = ru.format(*args, *args)
        if args and isinstance(args[0], int) and args[0] % 10 == 1 and args[0] % 100 != 11:
            s = s.replace("не входят", "не входит")       # '1 цеп. п. ... (в счёт не входит)'
        return s
    a = list(args)
    if key == "hook":
        a = [_ordinal(int(args[0]))]
    return en.format(*a)


def row(label, ops, pre=(), post=(), **kw):
    c = sum(op_cp(o)[0] for o in ops)
    n = sum(op_cp(o)[1] for o in ops)
    r = {"label": label, "ops": list(ops), "n": n, "consumed": c, "pre": list(pre), "post": list(post)}
    r.update(kw)
    return r


def row_text(r, lang):
    """The row instruction without its label and total."""
    keys = [k for k, _ in r["pre"]]
    bits = [_phrase(k, a, lang) for k, a in r["pre"]]
    body = []
    for seg in r.get("segments") or [r["ops"]]:
        body.append(ops_text(seg, lang))
    # a segmented row (along both sides of a chain): segments separated by their phrases
    if r.get("segments"):
        mid = r.get("mid", [])
        parts = []
        for i, b in enumerate(body):
            parts.append(b); keys.append(None)
            if i < len(mid):
                parts.append(_phrase(mid[i][0], mid[i][1], lang)); keys.append(mid[i][0])
        bits += parts
    else:
        bits.append(body[0]); keys.append(None)
    bits += [_phrase(k, a, lang) for k, a in r["post"]]
    keys += [k for k, _ in r["post"]]
    if lang != "ru":
        return "; ".join(b for b in bits if b)
    # Russian: a phrase that introduces the stitches ('начиная с 2-й цеп. п. от крючка', 'по другой стороне
    # цепочки', 'в кольцо амигуруми') is followed by a colon, not a semicolon
    out = ""
    prev = None
    for k, b in zip(keys, bits):
        if not b:
            continue
        if out:
            out += ": " if prev in ("hook", "other") and k is None or prev == "mr" else "; "
        out += b
        prev = k
    return out


def parse_row_text(text):
    """Our EN row text -> (phrases [(key, args)], ops). Raises MotifError on anything unknown."""
    phrases, ops = [], []
    text = text.replace("×", "x")
    for seg in [s.strip() for s in text.split(";") if s.strip()]:
        for key, (_, _, rx) in PH.items():
            m = re.fullmatch(rx, seg)
            if m:
                phrases.append((key, tuple(int(g) for g in m.groups())))
                break
        else:
            if seg == "pull the ring tight":
                continue
            for it in _split(seg):
                ops += _parse_item(it)
    return phrases, ops


# ---------------------------------------------------------------------------------------------------------------
# kinds -> rows

def _graded(L, tall):
    """Graded stitches along L chains: short at the ends, `tall` in the middle (a leaf's side)."""
    ladder = ["sc", "hdc", "dc", "tr"]
    top = ladder.index(tall)
    out = []
    for i in range(L):
        d = min(i, L - 1 - i)                  # distance from the nearer end
        out.append(ladder[min(top, d)])
    return out


def _edging_rows(prev_n, ed, label):
    kind = ed.get("kind", "shell")
    if kind == "shell":
        k = int(ed.get("k", 5)); every = int(ed.get("every", 4)); st = ed.get("st", "dc")
        if every < 2 or every % 2:
            raise MotifError("shell edging: 'every' must be an even number >= 2")
        g = prev_n // every
        sk = (every - 2) // 2
        unit = ["skip"] * sk + [f"inc:{st}:{k}"] + ["skip"] * sk + ["sl"]
        ops = unit * g + ["sl"] * (prev_n - g * every)
        return row(label, ops, edging=kind)
    if kind == "picot":
        every = int(ed.get("every", 3))
        g = prev_n // every
        ops = (["sc"] * every + ["picot"]) * g + ["sc"] * (prev_n - g * every)
        return row(label, ops, edging=kind)
    if kind == "point":
        # bat-wing points: (ch k, back along the chain with graded stitches, skip 1, sl st in the rest) per `every`
        every = int(ed.get("every", 4)); k = int(ed.get("length", 5)); st = ed.get("st", "dc")
        if every < 2 or k < 3:
            raise MotifError("point edging: 'every' >= 2, 'length' >= 3")
        g = prev_n // every
        a = (every - 1) // 2                                    # the point in the middle of its stretch
        unit = ["sl"] * a + [_point_op(k, st), "skip"] + ["sl"] * (every - 1 - a)
        return row(label, unit * g + ["sl"] * (prev_n - g * every), edging=kind)
    if kind == "feather":
        # feather tips: graded stitches rising to two tall ones in one st, then a sl st (a saw edge of feathers)
        every = int(ed.get("every", 5)); st = ed.get("st", "tr")
        if every < 3:
            raise MotifError("feather edging: 'every' >= 3")
        top = LADDER.index(st)
        unit = [LADDER[min(top, i)] for i in range(every - 2)] + [f"inc:{st}:2", "sl"]
        g = prev_n // every
        return row(label, unit * g + ["sl"] * (prev_n - g * every), edging=kind)
    raise MotifError(f"unknown edging kind {kind!r}")


def _point_op(k, tall):
    """ch k, then back along the chain from the 2nd ch from hook: sc at the tip, taller toward the base."""
    top = LADDER.index(tall)
    return "pt:" + ".".join(LADDER[min(top, i)] for i in range(k - 1))


def _heart_round(n):
    """A sc round around the heart: 2 incs on each lobe, 3 sc in the point (the middle stitch); +6."""
    ops = ["sc"] * n
    for f in (0.12, 0.28):
        j = int(round(f * (n - 1)))
        ops[j] = ops[n - 1 - j] = "inc:sc:2"
    ops[(n - 1) // 2] = "inc:sc:3"
    return ops


def _lobed_round(prev_n, lobes, first, tall):
    """A sc round around the lobes. The first: per lobe, sc in the chain space (the post), sc / inc alternately in
    the tall stitches, sc in the closing chain space, skip the sl st in the ring (the dent stays). Later rounds: a
    decrease just after each dent, 3 incs over the lobe."""
    m = prev_n // lobes
    if first:
        k = m - 1
        sp = "sp:3" if tall == "tr" else "sp:2"
        body = ["inc:sc:2" if (i % 2 == (0 if k % 2 else 1)) else "sc" for i in range(k)]
        return ([sp] + body + [sp, "skip"]) * lobes
    body = ["sc"] * (m - 2)
    for f in (0.25, 0.5, 0.75):
        body[min(m - 3, int(round(f * (m - 3))))] = "inc:sc:2"
    return (["tog:sc:2"] + body) * lobes


def gen_rows(m):
    """Motif parameters -> written rows (list of row dicts) + layout info."""
    kind = m.get("kind")
    if kind not in KINDS:
        raise MotifError(f"unknown motif kind {kind!r} (one of {', '.join(KINDS)})")
    rows = []
    if kind == "triangle":
        W = int(m.get("width", 12)); st = m.get("stitch", "sc"); ends = m.get("ends", "both")
        every = max(1, int(m.get("every", 1)))
        if W < 2:
            raise MotifError("triangle: width >= 2")
        tc = TURN_CH[st]
        rows.append(row("row", [st] * W, pre=[("chain", (W + tc,)), ("hook", (tc + 1,))]))
        n = W; k = 1
        while n > 1:
            k += 1
            turn = [("turn", (tc,))]
            if (k - 2) % every:                         # a straight row between decrease rows
                rows.append(row("row", [st] * n, pre=turn)); continue
            if ends == "both":
                if n <= 3:
                    ops = [f"tog:{st}:{n}"]
                else:
                    ops = [f"tog:{st}:2"] + [st] * (n - 4) + [f"tog:{st}:2"]
            else:
                # the decreases stay on one edge: the edge where row 1 ended (start of even rows, end of odd rows)
                ops = [f"tog:{st}:2"] + [st] * (n - 2) if k % 2 == 0 else [st] * (n - 2) + [f"tog:{st}:2"]
            rows.append(row("row", ops, pre=turn))
            n = rows[-1]["n"]
        return rows
    if kind == "leaf":
        N = int(m.get("chain", 9)); tall = m.get("tall", "dc"); tip = m.get("tip", "picot")
        if N < 4:
            raise MotifError("leaf: chain >= 4")
        side = _graded(N - 2, tall)
        a = side; b = side[::-1]
        ops = a + ["inc:sc:3"] + b + (["picot"] if tip == "picot" else [])
        r = row("round", ops, pre=[("chain", (N,)), ("hook", (2,))], post=[("join", ())],
                segments=[a + ["inc:sc:3"], b + (["picot"] if tip == "picot" else [])], mid=[("other", ())],
                around_chain=N)
        rows.append(r)
        return rows
    if kind in ("circle", "hexagon"):
        R = int(m.get("rounds", 3))
        if kind == "circle":
            exp = expand(["mr 6"] + ([f"inc 6 to {6 * R}"] if R > 1 else []))
            for i, rr in enumerate(exp):
                ops = ["sc" if o == "sc" else "inc:sc:2" for o in rr.ops]
                rows.append(row("round", ops, pre=[("mr", ())] if i == 0 else []))
        else:
            rows.append(row("round", ["sc"] * 6, pre=[("mr", ())]))
            for r_ in range(2, R + 1):
                rows.append(row("round", (["sc"] * (r_ - 2) + ["inc:sc:2"]) * 6))
        if m.get("edging"):
            rows.append(_edging_rows(rows[-1]["n"], m["edging"], "round"))
            rows[-1]["post"] = [("join", ())]
        return rows
    if kind == "half_circle":
        R = int(m.get("rows", 4))
        exp = expand(["mr 3"] + ([f"inc 3 to {3 * R}"] if R > 1 else []))
        for i, rr in enumerate(exp):
            ops = ["sc" if o == "sc" else "inc:sc:2" for o in rr.ops]
            rows.append(row("row", ops, pre=[("mr", ())] if i == 0 else [("turn", (1,))]))
        if m.get("edging"):
            rows.append(_edging_rows(rows[-1]["n"], m["edging"], "row"))
            rows[-1]["pre"] = [("turn", (1,))] + ([("curve", ())] if m["edging"].get("where") == "curve" else [])
        return rows
    if kind == "crescent":
        N = int(m.get("chain", 12)); tall = m.get("tall", "tr")
        if N < 6:
            raise MotifError("crescent: chain >= 6")
        L = N - 1
        ladder = ["sc", "hdc", "dc", "tr"]
        top = ladder.index(tall)
        ops = []
        for i in range(L):
            d = min(i, L - 1 - i)
            lvl = min(top, d)
            st = ladder[lvl]
            # the middle third gets two stitches in one chain: the outer edge grows, the piece curls
            ops.append(f"inc:{st}:2" if (lvl == top and abs(i - (L - 1) / 2) <= L / 6) else st)
        rows.append(row("row", ops, pre=[("chain", (N,)), ("hook", (2,))], along_chain=N))
        return rows
    if kind == "heart":
        ops = ["tr"] * 3 + ["dc"] * 3 + ["ch:1", "tr", "ch:1"] + ["dc"] * 3 + ["tr"] * 3
        rows.append(row("round", ops, pre=[("mr", ()), ("ch3", ())], post=[("ring", ())], ring_start=True))
        R = int(m.get("rounds", 0))
        if not 0 <= R <= 8:
            raise MotifError("heart: rounds 0..8")
        for i in range(R):
            # sc rounds around the heart, into the stitches only: the notch between the posts stays
            rows.append(row("round", _heart_round(rows[-1]["n"]), pre=[("posts", ())] if i == 0 else [],
                            post=[("join", ())], ring_round=True))
        return rows
    if kind == "star":
        p = int(m.get("points", 5)); k = int(m.get("length", 5)); tall = m.get("tall", "dc")
        if not 3 <= p <= 8 or k < 3 or tall not in ("hdc", "dc", "tr"):
            raise MotifError("star: points 3..8, length >= 3, tall hdc/dc/tr")
        rows.append(row("round", ["sc"] * (2 * p), pre=[("mr", ())], post=[("join", ())]))
        rows.append(row("round", [_point_op(k, tall), "skip", "sl"] * p, edging="point"))
        return rows
    if kind == "lobed":
        L = int(m.get("lobes", 4)); tall = m.get("tall", "dc"); k = int(m.get("stitches", 5))
        R = int(m.get("rounds", 0))
        if not 3 <= L <= 6 or tall not in ("hdc", "dc", "tr") or not 3 <= k <= 9 or not 0 <= R <= 6:
            raise MotifError("lobed: lobes 3..6, tall hdc/dc/tr, stitches 3..9, rounds 0..6")
        post = f"ch:{TURN_CH[tall]}"
        rows.append(row("round", ([post] + [tall] * k + [post, "sl"]) * L, pre=[("mr", ())], post=[("tight", ())],
                        ring_start=True))
        for i in range(R):
            rows.append(row("round", _lobed_round(rows[-1]["n"], L, i == 0, tall), post=[("join", ())],
                            ring_round=True))
        return rows
    if kind == "petal":
        N = int(m.get("chain", 7)); tall = m.get("tall", "dc")
        if N < 4 or tall not in FAN:
            raise MotifError("petal: chain >= 4, tall sc/hdc/dc/tr")
        top = LADDER.index(tall)
        a = [LADDER[min(top, i)] for i in range(N - 2)]          # rising from the narrow base toward the tip
        fan = [f"inc:{tall}:{FAN[tall]}"]                         # the rounded tip: a fan in the last chain
        rows.append(row("round", a + fan + a[::-1], pre=[("chain", (N,)), ("hook", (2,))], post=[("join", ())],
                        segments=[a + fan, a[::-1]], mid=[("other", ())], around_chain=N))
        return rows
    if kind == "mask":
        N = int(m.get("chain", 8)); R = int(m.get("rounds", 3)); lobe = m.get("tall", "dc")
        if N < 4 or R < 1:
            raise MotifError("mask: chain >= 4, rounds >= 1")
        exp = expand([f"ch {N}"] + ([f"inc 6 x{R - 1}"] if R > 1 else []), oval_turns=0.25)
        for i, rr in enumerate(exp):
            ops = ["sc" if o == "sc" else "inc:sc:2" for o in rr.ops]
            if i == 0:
                # round 1 along both sides of the chain (3 sc in the last ch, 2 sc in the first), then the marker
                # moves to the middle of a long side, where the engine counts the rounds from
                a = ["sc"] * (N - 2) + ["inc:sc:3"]; b = ["sc"] * (N - 3) + ["inc:sc:2"]
                rows.append(row("round", a + b, pre=[("chain", (N,)), ("hook", (2,))],
                                post=[("shift", (max(1, int(round((N - 2) / 2))),))],
                                segments=[a, b], mid=[("other", ())], around_chain=N, oval_round=True))
                continue
            rows.append(row("round", ops, oval_round=True))
        # the lobe round: tall stitches with increases over the two turns, slip stitches in the middle of the top
        n = rows[-1]["n"]
        L = (N - 2)
        pos = _stadium_class(n, L, rows)
        ops = []
        k_turn = 0
        for cls in pos:
            if cls == "turn":
                ops.append(f"inc:{lobe}:2" if k_turn % 2 == 0 else lobe)
                k_turn += 1
            elif cls == "dip":
                ops.append("sl")
            elif cls == "top":
                ops.append("hdc" if lobe in ("dc", "tr") else "sc")
            else:
                ops.append(lobe)                 # the bottom: tall stitches too, the mask reaches down the cheeks
        rows.append(row("round", ops, post=[("join", ())], lobes=True))
        if m.get("edging"):
            rows.append(_edging_rows(rows[-1]["n"], m["edging"], "round"))
        return rows
    raise MotifError(kind)


def _stadium_class(n, L, rows):
    """Class of each stitch of an oval round counted from the middle of the top side (the engine's marker):
    'dip' (middle of the top), 'top', 'turn', 'bottom'."""
    out = []
    r = max(n - 2 * L, 1) / (2 * np.pi)                 # in stitch widths
    per = 2 * L + 2 * np.pi * r
    for j in range(n):
        s = (j + 0.5) / n * per                         # from the middle of the top side, toward +y
        e1 = L / 2; e2 = e1 + np.pi * r; e3 = e2 + L; e4 = e3 + np.pi * r
        if s < e1 or s >= e4:
            d = s if s < e1 else per - s                 # distance from the top middle
            out.append("dip" if d < max(0.6, 0.22 * L) else "top")
        elif s < e2 or (e3 <= s < e4):
            out.append("turn")
        else:
            out.append("bottom")
    return out


def check_rows(rows):
    """Arithmetic: every row works the stitches it should and makes the count it states."""
    errs = []
    prev = None
    for i, r in enumerate(rows, 1):
        c = sum(op_cp(o)[0] for o in r["ops"]); n = sum(op_cp(o)[1] for o in r["ops"])
        if n != r["n"]:
            errs.append(f"R{i}: makes {n} stitches, stated {r['n']}")
        if r.get("around_chain"):
            need = 2 * (r["around_chain"] - 2) + 1 if not r.get("oval_round") else 2 * r["around_chain"] - 3
        elif r.get("along_chain"):
            need = r["along_chain"] - 1
        elif i == 1 and any(k == "chain" for k, _ in r["pre"]):
            need = next(a[0] for k, a in r["pre"] if k == "chain") - next(a[0] for k, a in r["pre"] if k == "hook") + 1
        elif i == 1 or r.get("ring_start"):
            need = None                                   # into the magic ring
        else:
            need = prev
        if need is not None and c != need:
            errs.append(f"R{i}: works {c} stitches, there are {need}")
        prev = r["n"]
    return errs


# ---------------------------------------------------------------------------------------------------------------
# geometry: band curves C_0..C_N in flat coordinates (x = the piece's up, y = across); band b (row b) lies between
# C_{b-1} and C_b, its stitches spread along s in [0, 1]

class _Curve:
    def __init__(self, s, pts, closed, linear=False):
        s = np.asarray(s, float); pts = np.asarray(pts, float)
        order = np.argsort(s, kind="stable")
        s, pts = s[order], pts[order]
        self.closed = closed
        if closed:
            ss = np.concatenate([s - 1, s, s + 1]); pp = np.concatenate([pts, pts, pts])
        else:
            ss, pp = s, pts
        # drop duplicate knots
        keep = np.concatenate([[True], np.diff(ss) > 1e-9])
        ss, pp = ss[keep], pp[keep]
        if len(ss) == 1:
            ss = np.array([0.0, 1.0]); pp = np.vstack([pp, pp])
        self.grid = np.linspace(0, 1, 801)
        if linear:                                   # a polygon (the corners of a hexagon stay corners)
            self.vals = np.stack([np.interp(self.grid, ss, pp[:, 0]), np.interp(self.grid, ss, pp[:, 1])], -1)
        else:
            if len(ss) >= 5:                         # smooth, no forced flat spots at a component's extremum
                self.vals = Akima1DInterpolator(ss, pp, axis=0)(self.grid, extrapolate=True)
            else:
                self.vals = PchipInterpolator(ss, pp, axis=0, extrapolate=True)(self.grid)

    def __call__(self, s):
        s = np.asarray(s, float)
        x = np.interp(s, self.grid, self.vals[:, 0]); y = np.interp(s, self.grid, self.vals[:, 1])
        return x, y


def _poly_s(pts, closed):
    """Arc-length fractions of polyline vertices."""
    p = np.asarray(pts, float)
    q = np.vstack([p, p[:1]]) if closed else p
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    tot = max(d[-1], 1e-9)
    return (d / tot)[:len(p)]


def _stitch_s(n):
    return (np.arange(n) + 0.5) / n


def _bump_knots(base_pts, normals, ops, h, s_of_base, closed=True, w=None):
    """Tops of an edging round over a closed/open base: shells fan into a rounded bump, sl sts stay low, a point
    (a chain with stitches worked back along it) stands out one chain length."""
    S, P = [], []
    j = 0                                  # base stitch index
    sk = 0                                 # skips just before the current op
    for oi, o in enumerate(ops):
        c, p = op_cp(o)
        if o == "skip":
            sk += 1
        if o.startswith("pt:"):
            seq = o[3:].split(".")
            Lch = len(seq) * w                                  # the chain the stitches stand on
            ahead = 0                                           # the skipped stitches the point covers
            while oi + 1 + ahead < len(ops) and ops[oi + 1 + ahead] == "skip":
                ahead += 1
            # the chain grows straight out from the last stitch worked (j - 0.5); the stitches worked back stand
            # on its leading side, across the base up to the sl st after the skipped stitches (a right-angled point)
            j0 = j - 0.5
            span = ahead + 0.4
            b0 = _interp_base(base_pts, normals, s_of_base, j0, closed)[0]
            b1 = _interp_base(base_pts, normals, s_of_base, j0 + span, closed)[0]
            W = max(float(np.linalg.norm(b1 - b0)), 1e-6)
            tm = (b1 - b0) / W
            nm = np.array([tm[1], -tm[0]])                      # outward (the round runs counter-clockwise)
            f = min(1.0, W / (HT[seq[-1]] * h + 0.2 * h))       # the tallest stitch fits the base
            knots = [(0.02, 0.3 * h), (0.4 * h * f, Lch + 0.3 * h)]                # the chain root, the tip
            knots += [(HT[st] * h * f, max(Lch - (q + 0.5) * w, 0.35 * h)) for q, st in enumerate(seq)]
            for v, u in knots:
                ss = _interp_base(base_pts, normals, s_of_base, j0 + span * v / W, closed)[2]
                S.append(ss); P.append(b0 + tm * v + nm * u)
        elif o.startswith("inc:"):
            st, k = o.split(":")[1], int(o.split(":")[2])
            H = HT[st] * h
            spread = sk + 0.5
            # the k stitches spread over the base stitches around j (the skipped ones included)
            for i in range(k):
                f = (i + 0.5) / k
                jj = j - spread + 2.0 * spread * f
                bp, nn, ss = _interp_base(base_pts, normals, s_of_base, jj, closed)
                lift = H * (0.35 + 0.65 * math.sin(math.pi * f) ** 0.7)
                S.append(ss); P.append(bp + nn * lift)
        elif o in HT and p:
            bp, nn, ss = _interp_base(base_pts, normals, s_of_base, j, closed)
            S.append(ss); P.append(bp + nn * HT[o] * h)
        elif o == "picot":
            bp, nn, ss = _interp_base(base_pts, normals, s_of_base, j - 0.5, closed)
            S.append(ss); P.append(bp + nn * (h + 0.9 * h))
        j += c
        if o != "skip":
            sk = 0
    return S, P


def _interp_base(base_pts, normals, s_of_base, jj, closed=True):
    nb = len(base_pts)
    if closed:
        jj = float(jj) % nb
        i0 = int(math.floor(jj)); f = jj - i0; i0 %= nb; i1 = (i0 + 1) % nb     # (-1e-17 % nb == nb)
    else:
        jj = min(max(float(jj), 0.0), nb - 1.0)
        i0 = min(int(math.floor(jj)), nb - 2) if nb > 1 else 0; f = jj - i0; i1 = min(i0 + 1, nb - 1)
    bp = base_pts[i0] * (1 - f) + base_pts[i1] * f
    nn = normals[i0] * (1 - f) + normals[i1] * f
    nn = nn / max(np.linalg.norm(nn), 1e-9)
    s0, s1 = s_of_base[i0], s_of_base[i1]
    if s1 < s0:
        s1 += 1
    s_ = s0 * (1 - f) + s1 * f
    return bp, nn, (s_ % 1.0) if closed else s_


def _outward_normals(pts, closed):
    p = np.asarray(pts, float)
    if closed:
        t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    else:
        t = np.gradient(p, axis=0)
    n = np.stack([t[:, 1], -t[:, 0]], -1)                # CCW curve: outward = tangent rotated -90 deg
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def layout(kind, m, rows, w, h):
    """-> (curves [C_0..C_N], closed per band, sew segment (A, B, inward))."""
    curves, closed = [], []
    if kind == "triangle":
        st = m.get("stitch", "sc"); H = HT[st] * h
        W0 = rows[0]["n"]; ends = m.get("ends", "both")
        widths = [W0] + [r["n"] for r in rows]
        lo0 = -W0 * w / 2
        for k, n in enumerate(widths):
            x = k * H
            if ends == "both":
                a, b = -n * w / 2, n * w / 2
            else:
                a, b = lo0, lo0 + n * w
            curves.append(_Curve([0, 1], [[x, a], [x, b]], False))
            closed.append(False)
        tot = len(rows) * H
        sew = {"base": ((0.0, lo0), (0.0, -lo0), (1.0, 0.0)),
               "side": ((0.0, lo0), (tot, lo0), (0.0, 1.0))}
        return curves, closed[1:], sew, (tot / 2, 0.0 if ends == "both" else lo0 + W0 * w / 4)
    if kind == "leaf":
        N = int(m.get("chain", 9)); r = rows[0]
        Lc = (N - 1) * w
        side = r["segments"][0][:-1]
        L = len(side)
        xs = [Lc - (p - 0.5) * w for p in range(1, L + 1)]          # chain positions 1..N-2 from the tip
        tops, base = [], []
        for x, o in zip(xs, side):
            tops.append([x, HT[o] * h]); base.append([x, 0.0])
        for a in (120, 180, 240):                                  # 3 sc in the last chain: around the base end
            aa = math.radians(a)
            tops.append([0.5 * w + h * math.cos(aa), h * math.sin(aa)]); base.append([0.5 * w, 0.0])
        for x, o in zip(xs[::-1], side[::-1]):
            tops.append([x, -HT[o] * h]); base.append([x, 0.0])
        n = len(tops)
        s = list(_stitch_s(n))
        tip_len = 0.9 * h if m.get("tip", "picot") == "picot" else 0.2 * h
        s += [0.0]; tops.append([Lc + tip_len, 0.0]); base.append([Lc, 0.0])
        curves = [_Curve(s, base, True), _Curve(s, tops, True)]
        sew = {"base": ((0.5 * w - h, -0.6 * h), (0.5 * w - h, 0.6 * h), (1.0, 0.0))}
        return curves, [True], sew, (Lc / 2, 0.0)
    if kind in ("circle", "hexagon", "mask", "half_circle", "star"):
        half = kind == "half_circle"
        main = [r for r in rows if not r.get("edging") and not r.get("lobes")]
        curves.append(_Curve([0, 1], [[0, 0], [0, 0]], not half) if kind != "mask" else None)
        if kind == "mask":
            N = int(m.get("chain", 8)); Lst = (N - 2) * w
            curves[0] = _Curve(*_stadium_pts(0.0, Lst, 64), True)
        for i, r in enumerate(main, 1):
            n = r["n"]
            if kind in ("circle", "star"):
                R = n * w / (2 * np.pi)
                a = 2 * np.pi * np.linspace(0, 1, 97)[:-1]
                curves.append(_Curve(np.linspace(0, 1, 97)[:-1], np.stack([R * np.cos(a), R * np.sin(a)], -1), True))
            elif kind == "hexagon":
                R = n * w / 6
                a0 = 0.0 if not m.get("point_up") else np.pi / 6
                v = [[R * math.cos(a0 + k * np.pi / 3), R * math.sin(a0 + k * np.pi / 3)] for k in range(6)]
                curves.append(_Curve(np.arange(6) / 6, v, True, linear=True))
            elif half:
                R = n * w / np.pi
                s = np.linspace(0, 1, 49)
                a = np.pi / 2 + np.pi * s                      # the arc below, the straight edge on top
                curves.append(_Curve(s, np.stack([R * np.cos(a), R * np.sin(a)], -1), False))
            else:
                Lst = (int(m.get("chain", 8)) - 2) * w
                rr = max(n * w - 2 * Lst, 0) / (2 * np.pi)
                curves.append(_Curve(*_stadium_pts(rr, Lst, 128), True))
        extra = [r for r in rows if r.get("edging") or r.get("lobes")]
        for r in extra:
            prev = curves[-1]
            nb = rows[rows.index(r) - 1]["n"]
            sb = _stitch_s(nb) if not half else (np.arange(nb) + 0.5) / nb
            bx, by = prev(sb)
            bp = np.stack([bx, by], -1)
            nn = _outward_normals(bp, not half)
            if r.get("lobes"):
                S, P = [], []
                j = 0
                for o in r["ops"]:
                    c, p = op_cp(o)
                    for i in range(p):
                        jj = j + (i + 0.5) / p - 0.5
                        b_, n_, s_ = _interp_base(bp, nn, sb, jj)
                        S.append(s_); P.append(b_ + n_ * op_height(o) * h)
                    j += c
            else:
                S, P = _bump_knots(bp, nn, r["ops"], h, sb, closed=not half, w=w)
            if half:                                     # an open row: its ends sit on the ends of the row below
                for e in (0.0, 1.0):
                    ex, ey = prev(np.array([e]))
                    S.append(e); P.append([ex[0], ey[0] + (0.3 * h if e > 0.5 else -0.3 * h)])
                S = [x for x in S]; keep = [i for i, x in enumerate(S) if 0.0 <= x <= 1.0]
                S = [S[i] for i in keep]; P = [P[i] for i in keep]
            # chain points stay sharp (straight edges between their knots)
            curves.append(_Curve(S, P, not half, linear=any(o.startswith("pt:") for o in r["ops"])))
        if kind == "star":                               # turn the star so that its first point is at the top
            ox, oy = curves[-1](np.linspace(0, 1.0 / len(rows[-1]["ops"]) * 3, 120))
            k = int(np.argmax(np.hypot(ox, oy)))
            _rotate(curves, -math.atan2(oy[k], ox[k]))
        closed = [not half] * len(rows)
        if half:
            Rt = max(np.abs(curves[-1](np.linspace(0, 1, 50))[1]).max(), 1e-6)
            sew = {"base": ((0.0, -Rt), (0.0, Rt), (-1.0, 0.0))}
            return curves, closed, sew, (-Rt / 2, 0.0)
        ox, oy = curves[-1](np.linspace(0, 1, 200))
        k = int(np.argmin(ox))
        sew = {"base": ((ox[k], oy[k] - w), (ox[k], oy[k] + w), (1.0, 0.0))}
        return curves, closed, sew, (0.0, 0.0)
    if kind == "crescent":
        N = int(m.get("chain", 12)); r = rows[0]
        L = N - 1
        Lc = L * w; Lo = r["n"] * w
        Hs = [op_height(o) * h for o in r["ops"]]
        Hm = max(Hs)
        rho = Hm * Lc / (Lo - Lc) if Lo > Lc * 1.02 else 1e6
        inner, outer, S = [], [], []
        cum = 0
        for i, o in enumerate(r["ops"]):
            a = (i + 0.5 - L / 2) * w
            phi = a / rho
            k = op_cp(o)[1]
            s = (cum + k / 2) / r["n"]
            cum += k
            inner.append([-rho + rho * math.cos(phi), rho * math.sin(phi)])
            outer.append([-rho + (rho + Hs[i]) * math.cos(phi), (rho + Hs[i]) * math.sin(phi)])
            S.append(s)
        for e, sgn in ((0.0, -1), (1.0, 1)):                    # the horns: the chain ends
            phi = sgn * (Lc / 2) / rho
            P = [-rho + rho * math.cos(phi), rho * math.sin(phi)]
            inner.append(P); outer.append(P); S.append(e)
        curves = [_Curve(S, inner, False), _Curve(S, outer, False)]
        sew = {"base": ((inner[len(inner) // 2 - 1][0], -w), (inner[len(inner) // 2 - 1][0], w), (1.0, 0.0))}
        return curves, [False], sew, (Hm / 2, 0.0)
    if kind in ("heart", "lobed"):
        # stitches worked into a magic ring stand out from it like spokes: each takes an angular slot, its top one
        # stitch height from the ring (a chain post a bit less, a ch-1 space a narrow empty slot)
        r = rows[0]
        r0 = 0.3 * h
        if kind == "heart":
            posts = [("ch3", 1.9)] + [(o, HT.get(o, 0)) for o in r["ops"]] + [("ch3", 1.9)]
        else:
            Hp = 0.9 * HT[m.get("tall", "dc")]
            posts = [(o, Hp if o.startswith("ch:") else HT[o]) for o in r["ops"]]
        slots = []
        for o, H in posts:
            if kind == "heart" and o.startswith("ch:"):
                slots.append((o, 0.55, 0.0))
            elif o == "sl":
                slots.append((o, 0.6, HT["sl"]))
            else:
                slots.append((o, 1.0, H))
        tot = sum(sz for _, sz, _ in slots)
        a0 = 0.0
        if kind == "lobed":                                     # the first lobe's middle at the top
            L = int(m.get("lobes", 4)); per = len(slots) // L
            a0 = 2 * np.pi * (sum(sz for _, sz, _ in slots[:per - 2]) + slots[0][1]) / 2 / tot
        S, P = ([0.0], [[r0 + 0.3 * h, 0.0]]) if kind == "heart" else ([], [])   # the heart's notch (sl st in ring)
        st_s = []                                               # the counted stitches (the next round works in them)
        acc = 0.0
        for o, sz, H in slots:
            mid = (acc + sz / 2) / tot
            acc += sz
            if o.startswith("ch:") and kind == "heart":
                continue
            ang = 2 * np.pi * mid - a0
            rr = r0 + H * h
            if kind == "heart" and o == "tr" and abs(mid - 0.5) < 0.04:   # the point: the tr between the ch-1 spaces
                rr += 0.5 * h
            S.append(mid); P.append([rr * math.cos(ang), rr * math.sin(ang)])
            if o in HT:
                st_s.append(mid)
        curves = [_Curve([0, 1], [[0, 0], [0, 0]], True), _Curve(S, P, True)]
        _ring_rounds(curves, rows[1:], st_s, h, notch=kind == "heart")
        ox, oy = curves[-1](np.linspace(0, 1, 200))
        k = int(np.argmin(ox))
        yk = oy[k] if kind == "lobed" else 0.0
        sew = {"base": ((ox[k], yk - w), (ox[k], yk + w), (1.0, 0.0))}
        return curves, [True] * len(rows), sew, (0.0, 0.0)
    if kind == "petal":
        N = int(m.get("chain", 7)); r = rows[0]
        a, b = r["segments"]
        fan = a[-1]; a = a[:-1]
        F = int(fan.split(":")[2]); Hf = HT[fan.split(":")[1]] * h
        xs = [(i + 1.5) * w for i in range(len(a))]                 # chains 2..N-1 from the hook (the base end)
        xt = (N - 0.5) * w                                           # the last chain: the tip
        tops, base = [], []
        for x, o in zip(xs, a):                                      # up one side ...
            tops.append([x, -HT[o] * h]); base.append([x, 0.0])
        for i in range(F):                                           # ... the fan round the tip ...
            ang = -np.pi / 2 + np.pi * (i + 0.5) / F
            tops.append([xt + Hf * math.cos(ang), Hf * math.sin(ang)]); base.append([xt, 0.0])
        for x, o in zip(xs[::-1], b):                                # ... and down the other side
            tops.append([x, HT[o] * h]); base.append([x, 0.0])
        s = list(_stitch_s(len(tops)))
        s += [0.0]; tops.append([0.1 * w, 0.0]); base.append([0.5 * w, 0.0])     # the join at the base
        curves = [_Curve(s, base, True), _Curve(s, tops, True)]
        sew = {"base": ((0.0, -0.6 * h), (0.0, 0.6 * h), (1.0, 0.0))}
        return curves, [True], sew, (xt / 2, 0.0)
    raise MotifError(kind)


def _rotate(curves, ang):
    c, s = math.cos(ang), math.sin(ang)
    Rm = np.array([[c, -s], [s, c]])
    for cv in curves:
        if cv is not None:
            cv.vals = cv.vals @ Rm.T


def _ring_rounds(curves, rows, base_s, h, notch=False):
    """Rounds worked around a piece made into a ring (heart, lobed): every stitch stands one stitch height out from
    the stitches it is worked into (along the outward normal), sc in a chain space between the stitches around it;
    a skipped stitch keeps its dent, the heart's notch stays (the posts are not worked)."""
    sb = np.asarray(base_s, float)
    for r in rows:
        prev = curves[-1]
        bx, by = prev(sb)
        bp = np.stack([bx, by], -1)
        nn = _outward_normals(bp, True)
        dense = np.stack(prev(np.linspace(0, 1, 480, endpoint=False)), -1)
        S, P, XS, XP = [], [], [], []

        def out(b_, H):
            # the fabric one stitch height outside the round below, along the ray through the stitch it is worked
            # into: the outer edge of the round below grown by H (a dent narrower than 2H fills in, no loops)
            u = b_ / max(np.linalg.norm(b_), 1e-9)
            proj = dense @ u
            perp = np.abs(dense[:, 0] * u[1] - dense[:, 1] * u[0])
            ok = (perp < H) & (proj > 0)
            rr = float(np.max(proj[ok] + np.sqrt(H * H - perp[ok] ** 2))) if ok.any() else np.linalg.norm(b_) + H
            return u * rr
        j = 0
        for o in r["ops"]:
            c, p = op_cp(o)
            if o == "skip":
                b_, n_, s_ = _interp_base(bp, nn, sb, j)
                XS.append(s_); XP.append(out(b_, 0.3 * h))
            for i in range(p):
                jj = (j - 1 + (i + 1) / (p + 1)) if c == 0 else (j + (i + 0.5) / p * c - 0.5)
                b_, n_, s_ = _interp_base(bp, nn, sb, jj)
                S.append(s_); P.append(out(b_, op_height(o) * h))
            j += c
        if notch:
            x0, y0 = prev(np.array([0.0]))
            XS.append(0.0); XP.append(out(np.array([x0[0], y0[0]]), 0.6 * h))
        curves.append(_Curve(S + XS, P + XP, True))
        sb = np.array(S)
    return curves


def _stadium_pts(r, L, k):
    """Stadium (half-width r around a straight chain of length L along y) from the middle of the top side (+x),
    counter-clockwise, by perimeter fraction."""
    from .piece import _stadium
    s = np.linspace(0, 1, k + 1)[:-1]
    x, y = _stadium(np.full(k, max(r, 1e-6)), L, 2 * np.pi * s)
    return s, np.stack([x, y], -1)


# ---------------------------------------------------------------------------------------------------------------
class MotifDef:
    """A flat motif with the attributes the engine reads from a PieceDef."""

    def __init__(self, pid, d, yarn):
        self.id = pid
        self.d = d
        self.m = dict(d["motif"])
        self.kind = self.m.get("kind")
        unknown = set(self.m) - MOTIF_KEYS
        self.flags = [f"part {pid}: unknown motif key(s) {sorted(unknown)} (ignored)"] if unknown else []
        if self.m.get("edging") and self.kind not in EDGING_KINDS:
            raise MotifError(f"{pid}: edging is for {', '.join(EDGING_KINDS)} motifs")
        where = (self.m.get("edging") or {}).get("where")
        if where is not None and (where != "curve" or self.kind != "half_circle"):
            raise MotifError(f"{pid}: edging 'where' is \"curve\" (a half_circle's curved edge) only")
        self.w = float(yarn["stitch_mm"]); self.h = float(yarn["round_mm"])
        self.thick = float(yarn.get("thickness_mm", 0.45 * self.w))
        self.rows = gen_rows(self.m)
        self.counts = [r["n"] for r in self.rows]
        self.N = len(self.rows); self.M = self.N
        self.rounds = [Round(n=r["n"], ops=["sc"] * r["n"]) for r in self.rows]
        self.closed = False; self.stuff = "none"; self.start = "motif"; self.oval = 0.0
        self.flatten = 0.0; self.spoon = 0.0; self.dish = 0.0; self.squash = 0.0; self.bend = None
        self.oval_axis = "side"
        self.colour = d.get("colour", "main")
        zones = list(d.get("colours", []))
        keep = []
        for z in zones:
            if set(z) - {"rounds", "colour"}:
                self.flags.append(f"part {pid}: a motif takes whole-row colour zones only ({{rounds, colour}}); "
                                  "zone ignored")
            else:
                keep.append(z)
        if self.m.get("edging", {}).get("colour"):
            keep.append({"rounds": [self.N, self.N], "colour": self.m["edging"]["colour"]})
        self.zones = keep
        self.solve_info = {"converged": True}
        self.motif = self.kind
        curves, closed, sew, centre = layout(self.kind, self.m, self.rows, self.w, self.h)
        self.curves = curves
        self.band_closed = closed
        self.centre = np.zeros(2)
        raw = self.outline()
        self.centre = (raw.min(0) + raw.max(0)) / 2         # the middle of the motif (appliques are centred on it)
        sew_key = self.m.get("sew", "base")
        if sew_key not in sew:
            raise MotifError(f"{pid}: a {self.kind} is sewn by its {' or '.join(sew)} (sew)")
        A, B, inward = sew[sew_key]
        self.sew = sew_key
        self.sew_mid = (np.array(A, float) + np.array(B, float)) / 2
        v = np.array(inward, float); self.sew_in = v / np.linalg.norm(v)
        # outline radius from the centre (for sizes, padding and distance tests)
        ol = self.outline()
        self.R_out = float(np.linalg.norm(ol - self.centre, axis=1).max())
        ts = np.arange(self.M + 1, dtype=float)
        self.rb = np.array([self._r_of_t(t) for t in ts]); self.zb = np.zeros(self.M + 1)

    # ---- profile-like helpers
    def _curve_pts(self, b, s):
        return np.stack(self.curves[b](s), -1)

    def _r_of_t(self, t):
        s = np.linspace(0, 1, 60)
        P = np.stack(self.flat_points(np.full_like(s, t), 2 * np.pi * s), -1)
        return float(np.linalg.norm(P, axis=1).max())

    def r_t(self, t):
        t = np.clip(np.asarray(t, float), 0, self.M)
        return np.interp(t, np.arange(self.M + 1), self.rb)

    def z_t(self, t):
        return np.zeros_like(np.asarray(t, float))

    def _arc(self, t):
        return self.r_t(t)

    def n_stitches(self, t):
        i = np.clip(np.floor(t).astype(int), 0, self.N - 1)
        return np.asarray(self.counts)[i]

    def max_radius(self):
        return self.R_out

    def outline(self, k=240):
        """The motif's outer edge (flat coordinates), a closed polyline."""
        last = self.curves[-1]
        if self.band_closed[-1]:
            return self._curve_pts(len(self.curves) - 1, np.linspace(0, 1, k, endpoint=False))
        # open rows: the foundation, the right side (row ends), the last row back, the left side
        s = np.linspace(0, 1, k // 4)
        bottom = self._curve_pts(0, s)
        top = self._curve_pts(len(self.curves) - 1, s[::-1])
        right = np.array([self._curve_pts(b, np.array([1.0]))[0] for b in range(len(self.curves))])
        left = np.array([self._curve_pts(b, np.array([0.0]))[0] for b in range(len(self.curves))])[::-1]
        return np.vstack([bottom, right, top, left])

    # ---- surfaces
    def _s_of_th(self, th, closed):
        s = np.asarray(th, float) / (2 * np.pi)
        if closed:
            return np.mod(s, 1.0)
        inside = (s >= -0.05) & (s <= 1.05)
        return np.where(inside, np.clip(s, 0.0, 1.0), np.mod(s, 1.0))

    def flat_points(self, t, th):
        """(x, y) of the motif laid flat: x = the piece's up, y = across; centred on its middle."""
        t = np.clip(np.asarray(t, float), 0, self.M); th = np.asarray(th, float)
        t, th = np.broadcast_arrays(t, th)
        b = np.clip(np.floor(t).astype(int), 0, self.N - 1)
        v = t - b
        X = np.zeros(t.shape); Y = np.zeros(t.shape)
        for k in np.unique(b):
            m = b == k
            s = self._s_of_th(th[m], self.band_closed[k])
            x0, y0 = self.curves[k](s); x1, y1 = self.curves[k + 1](s)
            X[m] = x0 * (1 - v[m]) + x1 * v[m]; Y[m] = y0 * (1 - v[m]) + y1 * v[m]
        return X - self.centre[0], Y - self.centre[1]

    def flat_points_m(self, t, th, mirror=False):
        """flat_points of the piece or of its mirror image (the right-hand copy of a pair: the stitch order is
        reversed so the right side still faces out)."""
        if not mirror:
            return self.flat_points(t, th)
        x, y = self.flat_points(t, 2 * np.pi - np.asarray(th, float))
        return x, -y

    def local_points(self, t, th, mirror_y=False, **kw):
        """The motif standing on its sewn edge: local x = into the piece (away from the seam), local y along the
        seam, z = 0 (the piece's plane); the middle of the sewn edge is the origin."""
        if mirror_y:
            th = 2 * np.pi - np.asarray(th, float)
        x, y = self.flat_points(t, th)
        x = x + self.centre[0] - self.sew_mid[0]; y = y + self.centre[1] - self.sew_mid[1]
        a, b = self.sew_in
        X = x * a + y * b
        Y = -x * b + y * a
        if mirror_y:
            Y = -Y
        return np.stack([X, Y, np.zeros_like(X)], axis=-1)

    def outline_local(self, mirror_y=False):
        ol = self.outline() - self.sew_mid
        a, b = self.sew_in
        X = ol[:, 0] * a + ol[:, 1] * b
        Y = -ol[:, 0] * b + ol[:, 1] * a
        return np.stack([X, -Y if mirror_y else Y, np.zeros_like(X)], -1)

    def anchor(self, t, **kw):
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return self.local_points(np.full_like(th, float(t)), th).mean(axis=0)

    def pad_sn(self, t, th):
        x, y = self.flat_points(t, th)
        return np.clip(np.hypot(x, y) / max(self.R_out, 1e-6), 0, 1)

    # ---- written pattern
    def check(self):
        return check_rows(self.rows)

    def size_mm(self):
        ol = self.outline()
        return float(np.ptp(ol[:, 0])), float(np.ptp(ol[:, 1]))


# ---------------------------------------------------------------------------------------------------------------
# reading our rows back: the parameters whose rows are exactly the written ones

def infer(kind, rows_ops, phrases_by_row, edging_hint=None):
    """Written rows (ops per row) of a known kind -> motif parameters that generate exactly those rows."""
    n_rows = len(rows_ops)
    cands = []
    first_chain = next((a[0] for k, a in phrases_by_row[0] if k == "chain"), None) if phrases_by_row else None
    eds = [None]
    if kind in EDGING_KINDS:
        # (the repeat lengths a writer may use: every from 2 up to 24 stitches; a wider net than the presets so a
        # bat wing's point every 9 stitches still reads back)
        eds = [None] + [{"kind": "shell", "k": k, "every": e, "st": st} for k in range(3, 12) for e in range(2, 25)
                        for st in ("dc", "hdc", "tr")] + [{"kind": "picot", "every": e} for e in range(2, 13)]
        last = rows_ops[-1] if rows_ops else []
        pt = next((o for o in last if o.startswith("pt:")), None)
        if pt:                                         # a point edging: its chain and stitches are written out
            seq = pt[3:].split(".")
            eds += [{"kind": "point", "every": e, "length": len(seq) + 1, "st": st}
                    for e in range(2, 25) for st in (seq[-1], "hdc", "dc", "tr")]
        eds += [{"kind": "feather", "every": e, "st": st} for e in range(3, 25) for st in ("tr", "dc", "hdc")]
    if kind == "triangle":
        W = sum(op_cp(o)[1] for o in rows_ops[0])
        st = rows_ops[0][0]
        for ends in ("both", "one"):
            for every in (1, 2, 3):
                cands.append({"kind": kind, "width": W, "stitch": st, "ends": ends, "every": every})
    elif kind == "leaf":
        for tall in ("hdc", "dc", "tr", "sc"):
            for tip in ("picot", "none"):
                cands.append({"kind": kind, "chain": first_chain, "tall": tall, "tip": tip})
    elif kind in ("circle", "hexagon"):
        for e in eds:
            R = n_rows - (1 if e else 0)
            c = {"kind": kind, "rounds": R}
            if e:
                c["edging"] = e
            cands.append(c)
            if kind == "hexagon":
                cands.append({**c, "point_up": True})
    elif kind == "half_circle":
        for e in eds:
            R = n_rows - (1 if e else 0)
            c = {"kind": kind, "rows": R}
            if e:
                c["edging"] = e
            cands.append(c)
    elif kind == "crescent":
        for tall in ("tr", "dc", "hdc"):
            cands.append({"kind": kind, "chain": first_chain, "tall": tall})
    elif kind == "heart":
        cands.append({"kind": kind} if n_rows == 1 else {"kind": kind, "rounds": n_rows - 1})
    elif kind == "star" and n_rows == 2:
        pt = next((o for o in rows_ops[1] if o.startswith("pt:")), None)
        if pt:
            seq = pt[3:].split(".")
            for tall in (seq[-1], "dc", "hdc", "tr"):
                cands.append({"kind": kind, "points": len(rows_ops[0]) // 2, "length": len(seq) + 1, "tall": tall})
    elif kind == "lobed":
        L = rows_ops[0].count("sl")
        talls = [o for o in rows_ops[0] if o in ("hdc", "dc", "tr")]
        if L and talls:
            cands.append({"kind": kind, "lobes": L, "tall": talls[0], "stitches": len(talls) // L,
                          "rounds": n_rows - 1})
    elif kind == "petal":
        for tall in ("dc", "tr", "hdc", "sc"):
            cands.append({"kind": kind, "chain": first_chain, "tall": tall})
    elif kind == "mask":
        for e in eds:
            R = n_rows - 1 - (1 if e else 0)
            for tall in ("dc", "hdc", "tr"):
                c = {"kind": kind, "chain": first_chain, "rounds": R, "tall": tall}
                if e:
                    c["edging"] = e
                cands.append(c)
    for c in cands:
        try:
            rows = gen_rows(c)
        except (MotifError, TypeError, ValueError):
            continue
        if len(rows) == n_rows and all(r["ops"] == o for r, o in zip(rows, rows_ops)):
            if c.get("edging") and phrases_by_row and any(k == "curve" for k, _ in phrases_by_row[-1]):
                c["edging"]["where"] = "curve"
            return c
    return None
