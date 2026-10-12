"""Round steps -> stitch counts -> per-round stitch operations (designer style).

A piece's rounds are written as a list of short steps (the DSL an LLM writes):

    "mr 6"                 magic ring with 6 sc                     -> [6]
    "ch 7"                 chain oval start on 7 chains              -> [2*(7-1)+... ] (sc both sides, 3 in ends)
    "inc 6 to 42"          +6 per round until 42                     -> 12, 18, ..., 42
    "inc 6 x2"             two rounds of +6
    "inc 3 to 15 every 2"  +3 round, then an even round, repeated
    "even 7"               7 plain rounds
    "dec 6 to 6"           -6 per round until 6
    "dec 6 to 24 every 2"  -6 round, then an even round, repeated
    "18"  / 18             one round with exactly 18 stitches (inc/dec distributed evenly)
    "blo 18"               a round worked in back loops only (crisp edge, e.g. a sole); count optional
    "flo 18"               front loops only (leaves a ridge of free loops)
    "ring 36"              (first step) ch 36 joined into a ring with a sl st; round 1 = 36 sc, one in each chain
                           (a collar / snood worked outward from around a neck)
    "rib 36"               a ribbed round: (fpdc, bpdc) around, two sc rounds tall (a snood, a cuff)
    "shells 5 every 4"     an edging round: (skip 1, 5 dc in one st, skip 1, sl st) around -> scallops;
                           "shells 7 hdc every 6" (skip 2, 7 hdc, skip 2, sl st)

Each round yields a list of output stitches and the operations that made them, staggered the way designers stagger
increases so the piece stays round instead of hexagonal.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Round:
    n: int                      # stitches after this round
    ops: list                   # tokens: 'sc', 'inc', 'dec', 'mr', 'ch-start' ... (one per worked stitch/op)
    loop: str = "both"          # 'both' | 'blo' | 'flo'
    start: str | None = None    # 'mr' | 'ch' for round 1
    chain: int = 0              # foundation chain length for 'ch' starts
    note: str = ""


class RoundError(ValueError):
    pass


def _ops_change(prev: int, n: int, stagger: int) -> list:
    """Operations turning `prev` stitches into `n`, changes spread evenly and staggered."""
    if n == prev:
        return ["sc"] * prev
    if n > prev:
        k = n - prev
        if k > prev:
            # more than doubling: some stitches get 3 sc ('inc3')
            if n > 3 * prev:
                raise RoundError(f"cannot grow {prev} -> {n} in one round")
            k3 = n - 2 * prev
            ops = ["inc3"] * k3 + ["inc"] * (prev - k3)
            return _spread_mixed(ops, prev)
        return _spread(prev, k, "inc", stagger)
    k = prev - n
    if 2 * k > prev:
        raise RoundError(f"cannot shrink {prev} -> {n} in one round (max -{prev // 2})")
    return _spread(prev - k, k, "dec", stagger)  # prev - k worked ops: k decs (2 st each) + (prev-2k) sc


def _spread(n_ops: int, k: int, tok: str, stagger: int) -> list:
    """n_ops worked operations, k of them `tok`, evenly spaced; group offset staggered by round parity."""
    ops = []
    # Bresenham placement of k special ops among n_ops
    acc = 0
    base_pos = []
    for i in range(k):
        base_pos.append((i * n_ops) // k)
    group = n_ops // k
    shift = (group // 2) if (stagger % 2 == 1 and group > 1) else group - 1
    pos = set((p + shift) % n_ops for p in base_pos)
    for i in range(n_ops):
        ops.append(tok if i in pos else "sc")
    return ops


def _oval_incs(prev, k, turns=0.0):
    """Increases of a chain-oval piece go around its two turns (the ends of the chain), as designers write it.

    The round starts at the beginning of one long side: turn 1 is centred at prev/2 - ... we place the two clusters
    around 1/2 and the end of the round (the turns of a stadium whose round starts at a side)."""
    ops = ["sc"] * prev
    k1 = (k + 1) // 2; k2 = k - k1
    side = prev // 2
    for c, kk in ((side, k1), (prev, k2)):
        # centre of the turn = end of each half; spread kk increases over the turn's stitches
        span = max(kk * 2, 1)
        start = c - span // 2 - (span % 2) - 1 if c == prev else c - span // 2
        for j in range(kk):
            idx = (start + 2 * j + (1 if span > kk else 0)) % prev
            ops[idx] = "inc"
    if ops.count("inc") != k:      # collisions on tiny rounds: fall back to even spacing
        return _spread(prev, k, "inc", 0)
    if turns:
        sh = int(round(turns * prev))
        ops = ops[-sh:] + ops[:-sh] if sh else ops
    return ops


def _spread_mixed(ops, n_ops):
    return ops[:n_ops]


def _parse_step(step):
    if isinstance(step, int):
        return ("abs", step, {})
    s = str(step).strip().lower()
    m = re.fullmatch(r"(mr|magic ring)\s*(\d+)", s)
    if m:
        return ("mr", int(m.group(2)), {})
    m = re.fullmatch(r"ch\s*(\d+)", s)
    if m:
        return ("ch", int(m.group(1)), {})
    m = re.fullmatch(r"ring\s*(\d+)", s)
    if m:
        return ("ring", int(m.group(1)), {})
    m = re.fullmatch(r"(blo|flo)\s*(\d+)?", s)
    if m:
        return ("loop", int(m.group(2)) if m.group(2) else None, {"loop": m.group(1)})
    m = re.fullmatch(r"rib\s*(\d+)?(?:\s*x\s*(\d+))?", s)
    if m:
        return ("rib", int(m.group(1)) if m.group(1) else None, {"times": int(m.group(2) or 1)})
    m = re.fullmatch(r"popcorns?\s+every\s+(\d+)(?:\s+from\s+(\d+))?", s)
    if m:
        return ("popcorn", int(m.group(1)), {"from": int(m.group(2) or 0)})
    m = re.fullmatch(r"shells\s+(\d+)(?:\s+(sc|hdc|dc|tr))?\s+every\s+(\d+)", s)
    if m:
        return ("shells", int(m.group(1)), {"st": m.group(2) or "dc", "every": int(m.group(3))})
    m = re.fullmatch(r"(inc|dec)\s+(\d+)\s+(?:to\s+(\d+)|x\s*(\d+))(?:\s+every\s+(\d+))?", s)
    if m:
        kind, k = m.group(1), int(m.group(2))
        return (kind, k, {"to": int(m.group(3)) if m.group(3) else None,
                          "times": int(m.group(4)) if m.group(4) else None,
                          "every": int(m.group(5)) if m.group(5) else 1})
    m = re.fullmatch(r"even\s+(\d+)", s)
    if m:
        return ("even", int(m.group(1)), {})
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return ("abs", int(m.group(1)), {})
    raise RoundError(f"cannot read round step {step!r}")


def expand(steps, oval_turns=0.0) -> list[Round]:
    """Steps -> list of Round.

    oval_turns: for chain-oval pieces, where the two turns of the oval sit, as a fraction of the round from the
    round's first stitch (0 = the first stitch is at a turn; 0.25 = it is in the middle of a long side). The
    engine counts stitches from the centre front, so a side-axis oval has its turns at 0.25 / 0.75."""
    rounds: list[Round] = []
    cur = 0

    def add(n, loop="both"):
        nonlocal cur
        if cur == 0:
            raise RoundError("first step must be 'mr N', 'ch N' or 'ring N'")
        if rounds[0].start == "ch" and n > cur and n - cur <= cur // 2:
            ops = _oval_incs(cur, n - cur, oval_turns)
        else:
            ops = _ops_change(cur, n, len(rounds))
        rounds.append(Round(n=n, ops=ops, loop=loop))
        cur = n

    for step in steps:
        kind, val, opt = _parse_step(step)
        if kind == "mr":
            if rounds:
                raise RoundError("'mr' only as the first step")
            rounds.append(Round(n=val, ops=["sc"] * val, start="mr"))
            cur = val
        elif kind == "ch":
            if rounds:
                raise RoundError("'ch' only as the first step")
            # ch c; sc in 2nd ch from hook, sc in next c-3, 3 sc in last ch; along the other side sc in next c-3,
            # inc in the first ch  ->  1 + (c-3) + 3 + (c-3) + 2 = 2c stitches (ch 6 -> 12)
            c = val
            if c < 3:
                raise RoundError("a chain oval needs at least 3 chains")
            n = 2 * c
            rounds.append(Round(n=n, ops=["sc"] * n, start="ch", chain=c))
            cur = n
        elif kind == "ring":
            if rounds:
                raise RoundError("'ring' only as the first step")
            if val < 3:
                raise RoundError("a chain ring needs at least 3 chains")
            rounds.append(Round(n=val, ops=["sc"] * val, start="ring", chain=val))
            cur = val
        elif kind == "rib":
            n = val if val is not None else cur
            if n != cur or n % 2:
                raise RoundError(f"'{step}': ribbing keeps the count ({cur}) and needs an even number of stitches")
            for _ in range(opt["times"]):
                rounds.append(Round(n=cur, ops=["sc"] * cur, loop="rib"))
        elif kind == "popcorn":
            # a round of sc with a popcorn every k stitches (5 dc in one stitch, closed together: a raised bobble
            # on the right side - fleece, berry seeds); the count stays the same
            if not rounds:
                raise RoundError("first step must be 'mr N', 'ch N' or 'ring N'")
            k = val
            if k < 2:
                raise RoundError(f"'{step}': popcorns need at least one sc between them (every >= 2)")
            off = opt["from"] % k
            ops = ["pop" if (i - off) % k == k - 1 else "sc" for i in range(cur)]
            rounds.append(Round(n=cur, ops=ops, loop="popcorn"))
        elif kind == "shells":
            if not rounds:
                raise RoundError("first step must be 'mr N', 'ch N' or 'ring N'")
            e = opt["every"]
            if e < 2 or e % 2:
                raise RoundError(f"'{step}': 'every' must be an even number >= 2")
            sk = (e - 2) // 2
            g = cur // e
            if g < 1:
                raise RoundError(f"'{step}': the round has only {cur} stitches")
            unit = ["skip"] * sk + [f"shell{val}{opt['st']}"] + ["skip"] * sk + ["sl"]
            ops = unit * g + ["sl"] * (cur - g * e)
            n = sum(op_cp(o)[1] for o in ops)
            rounds.append(Round(n=n, ops=ops, loop="shell"))
            cur = n
        elif kind == "abs":
            add(val)
        elif kind == "loop":
            add(val if val is not None else cur, loop=opt["loop"])
        elif kind == "even":
            for _ in range(val):
                add(cur)
        elif kind in ("inc", "dec"):
            if val <= 0:
                raise RoundError(f"'{step}': the step must be at least 1")
            sign = 1 if kind == "inc" else -1
            to, times, every = opt["to"], opt["times"], max(1, opt["every"])
            if to is not None:
                if (to - cur) * sign < 0 or (to - cur) % val:
                    raise RoundError(f"'{step}': {cur} -> {to} is not a multiple of {val} in that direction")
                times = abs(to - cur) // val
            for t in range(times):
                add(cur + sign * val)
                if t < times - 1:
                    for _ in range(every - 1):
                        add(cur)
        else:
            raise RoundError(step)
    if not rounds:
        raise RoundError("no rounds")
    return rounds


def op_cp(o):
    """(consumed, produced) of a round operation."""
    base = {"sc": (1, 1), "inc": (1, 2), "inc3": (1, 3), "dec": (2, 1), "skip": (1, 0), "sl": (1, 1),
            "pop": (1, 1)}
    if o in base:
        return base[o]
    m = re.fullmatch(r"shell(\d+)(sc|hdc|dc|tr)", o)
    if m:
        return 1, int(m.group(1))
    raise KeyError(o)


def counts(steps) -> list[int]:
    return [r.n for r in expand(steps)]


def check(rounds: list[Round]) -> list[str]:
    """Arithmetic check: each round's ops consume the previous count and produce its own count."""
    errs = []
    prev = None
    for i, r in enumerate(rounds, 1):
        if i == 1:
            prev = r.n
            continue
        consumed = sum(op_cp(o)[0] for o in r.ops)
        produced = sum(op_cp(o)[1] for o in r.ops)
        if consumed != prev:
            errs.append(f"R{i}: works {consumed} stitches but previous round has {prev}")
        if produced != r.n:
            errs.append(f"R{i}: makes {produced} stitches, stated {r.n}")
        prev = r.n
    return errs


def compress(ops: list) -> list:
    """['sc','sc','inc','sc','sc','inc'] -> [(['sc','sc','inc'], 2)] style groups (largest repeating unit)."""
    n = len(ops)
    for size in range(1, n + 1):
        if n % size == 0:
            unit = ops[:size]
            if unit * (n // size) == ops:
                return [(unit, n // size)]
    # try a repeated unit with a leading/trailing remainder (staggered rounds: 'a sc, (inc, 2 sc) x5, inc, b sc')
    best = None
    for lead in range(0, min(6, n)):
        body = ops[lead:]
        for size in range(1, len(body) // 2 + 1):
            unit = body[:size]
            reps = 0
            while body[reps * size:(reps + 1) * size] == unit:
                reps += 1
            tail = body[reps * size:]
            if reps >= 2 and len(tail) <= size:
                cand = ([(ops[:lead], 1)] if lead else []) + [(unit, reps)] + ([(tail, 1)] if tail else [])
                score = lead + len(tail) + size
                if best is None or score < best[0]:
                    best = (score, cand)
    if best:
        return best[1]
    return [(ops, 1)]
