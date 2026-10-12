"""Construction presets: a few keys on a part expand into designer-style rounds and the ear/head constructions that
make a toy cute by construction. Opt-in: a spec without "preset" keys is untouched.

    {"id": "head", "preset": "cute_head", "max": 54}                    # wider-than-tall head, low full cheeks
    {"id": "head", "preset": "cute_head", "max": 54, "oval": true}      # chain-oval start: an even wider head
    {"id": "ear", "preset": "ear", "kind": "round"}                     # bear/quokka ear: a dish, ring in front
    {"id": "ear", "preset": "ear", "kind": "pointed", "max": 16}        # cat/fox/red panda: a cone pressed to a spoon
    {"id": "ear", "preset": "ear", "kind": "long", "length": 10}        # bunny: a long tube pressed to a shallow spoon
    {"id": "ear", "preset": "ear", "kind": "floppy"}                    # dog/calf: a hanging spoon
    {"id": "ear", "preset": "ear", "kind": "lop", "inner": "pink"}      # lop bunny: long soft ear over the head, hanging
    {"id": "ear", "preset": "ear", "kind": "floppy", "hang": true}      # soft dog ear that lies on the head and hangs
    {"id": "ear", "preset": "ear", "kind": "pointed", "inner": "pink"}  # + the inside in pink, shaped like the ear

Expansion happens before anything else (Toy, pattern, library audit), so the written pattern lists the plain rounds
and constructions (`spoon`, `dish`) and the reverse reader rebuilds the same toy without knowing about presets.
Explicit keys on the part win over the preset (e.g. your own "rounds" with "preset": "ear" only adds the
construction). For a `round` ear the assembly entries of that part get the dish placement defaults (stand it on its
rim: `attach: "edge"`, face forward and a little up, its lower rim pressed in ~0.55 x the dish radius) unless they
say otherwise.
"""
from __future__ import annotations
import copy

import numpy as np

PRESET_KEYS = ("preset", "kind", "max", "length", "oval", "crown", "inner", "hang")

# designers' library medians (amikit.library stats ear <kind>): widest round / head's widest round
EAR_MAX_REL = {"round": 0.36, "pointed": 0.31, "long": 0.33, "floppy": 0.38, "lop": 0.38}
# how far each ear construction is pressed (spoon / dish amount)
EAR_PRESS = {"round": ("dish", 0.5), "pointed": ("spoon", 0.35), "long": ("spoon", 0.3), "floppy": ("spoon", 0.45), "lop": ("spoon", 0.5)}
# spoon depth of an ear with a coloured inside: deep enough that the inside sits in a visible hollow
INNER_SPOON = {"pointed": 0.55, "long": 0.45, "floppy": 0.55, "lop": 0.5}
# a soft ear that hangs (bend.drape, toy.py _drape): sewn pointing out from the head's upper side, it lies over the
# head and hangs under its own weight (2026-10-08 bunny: replaced the stiff plank + fold of a bent ear)
HANG_BEND = {"drape": True, "deg": 90, "over": 3, "relax_mm": 6, "toward": [0.3, -1, 0.05]}
HANG_DIR = [1.0, 0.2, -0.1]


def _mult(x, k, lo):
    return max(lo, int(round(x / k)) * k)


def head_rounds(M, oval=False):
    """A cute head of widest round M: fast increases to M-12 (a broad crown), the last two increases every other
    round, a short straight band, then slow decreases (every other round, full low cheeks) and a quick close.
    Stuffed, its width / height is ~1.25-1.35 (measured: amikit.cute, test_engine); the visible part above the
    neck ~1.15-1.3, like the cute concepts."""
    M = _mult(M, 6, 36)
    even = max(1, int(round(M / 18)))
    if oval:
        ch = max(4, int(round(M / 9)))
        n0 = 2 * ch
        # chain-oval rounds keep +6 per round; the widest count must stay on that ladder
        top = n0 + 6 * max(2, (M - 12 - n0) // 6)
        M2 = top + 12
        r = (M2 - 12) % 6
        last = r + 6 * max(1, -(-(8 - r) // 6))           # the smallest count >= 8 on the +6 ladder: pull shut
        return [f"ch {ch}", f"inc 6 to {top}", f"inc 6 to {M2} every 2", f"even {even}",
                f"dec 6 to {M2 - 12} every 2", f"dec 6 to {last}"]
    return ["mr 6", f"inc 6 to {M - 12}", f"inc 6 to {M} every 2", f"even {even}", f"dec 6 to {M - 12} every 2",
            "dec 6 to 6"]


def round_crown_rounds(M, w=4.0, h=3.6):
    """A cute head with a round (domed, not flat-topped) crown, for big heads too: the counts follow a sphere of
    the widest round's size from the magic ring (+6 rounds only near the ring, then the increases thin out as the
    sides turn down), one or two straight rounds (fewer for a big head, so it does not get a box-like band), the
    first decreases every other round (full low cheeks), then +6 decreases to close. Opt-in: preset cute_head with
    "crown": "round"."""
    M = _mult(M, 6, 36)
    R = M * w / (2 * np.pi)
    cnt = [6]
    k = 1
    while cnt[-1] < M:
        k += 1
        want = M * np.sin(min(np.pi / 2, k * h / R))
        n = int(round(want / 6.0)) * 6
        n = min(max(n, cnt[-1]), cnt[-1] + 6, M)
        if k > 3 and n == cnt[-1] == cnt[-2] == cnt[-3] and n < M:
            n += 6                                   # never more than two straight rounds before the widest
        cnt.append(n)
        if k > 60:
            break
    even = 1 if M >= 60 else 2
    steps = ["mr 6"] + [str(n) for n in cnt[1:]] + [f"even {even}", f"dec 6 to {M - 12} every 2", "dec 6 to 6"]
    return steps


def ear_rounds(kind, M, length=None):
    if kind == "round":
        M = _mult(M, 6, 18)
        return ["mr 6", f"inc 6 to {M}", "even 1", str(M * 7 // 8), str(M * 3 // 4)]
    if kind == "pointed":
        M = 4 + 3 * max(3, int(round((M - 4) / 3)))
        return ["mr 4", f"inc 3 to {M}", "even 1"]
    if kind == "long":
        M = _mult(M, 6, 12)
        L = int(length or 9)
        k = max(1, M // 6)
        return ["mr 6", f"inc 6 to {M}", f"even {L}", str(M - k), f"even 2", str(M - 2 * k)]
    if kind == "lop":
        # a long soft ear: a rounded tip (+3/+4 per round, not +6), widest a third down, then narrowing slowly to
        # the seam (an oval leaf, never a plank of equal rounds)
        M = max(16, int(round(M)))
        L = int(length or 26)
        up = [6]
        while up[-1] < M:
            up.append(min(M, up[-1] + (4 if up[-1] < 14 else 3 if up[-1] < M - 3 else 1 if M - up[-1] == 1 else 2)))
        steps = ["mr 6"] + [str(n) for n in up[1:]]
        rest = max(L - len(up), 6)
        flat = max(2, rest // 4)
        steps.append(f"even {flat}")
        n, left = M, rest - flat
        lo = max(10, int(round(M * 0.55)))
        nsteps = max(1, (M - lo) // 2)
        gap = max(1, (left - nsteps) // nsteps)
        while n - 2 >= lo and left > 0:
            n -= 2; steps.append(str(n)); left -= 1
            g = min(gap, left - (1 if n - 2 >= lo else 0))
            if g > 0:
                steps.append(f"even {g}"); left -= g
        return steps
    if kind == "floppy":
        M = _mult(M, 6, 12)
        L = int(length or 3)
        k = max(1, M // 6)
        return ["mr 6", f"inc 6 to {M}", f"even {L}", str(M - k), str(M - 2 * k), "even 1"]
    raise ValueError(f"ear preset: unknown kind {kind!r} (round, pointed, long, floppy)")


def _head_max(spec):
    from .rounds import counts
    for p in spec.get("parts", []):
        if p.get("id") == "head" and "rounds" in p:
            return max(counts(p["rounds"]))
    return 48


def expand(spec):
    """A copy of the spec with every part preset turned into plain keys (rounds, end, stuff, spoon/dish) and the
    dish placement defaults on the assembly entries of round ears. Returns the spec unchanged (same content) when
    it uses no presets."""
    if not any(isinstance(p, dict) and "preset" in p for p in spec.get("parts", [])):
        return spec
    out = copy.deepcopy(spec)
    parts = out["parts"]
    # heads first: ear sizes follow the head
    order = sorted(range(len(parts)), key=lambda i: 0 if parts[i].get("preset") == "cute_head" else 1)
    dish_parts = {}
    hang_parts = set()
    for i in order:
        p = parts[i]
        kind = p.get("preset")
        if not kind:
            continue
        if kind == "cute_head":
            if p.get("crown") == "round":
                y = out.get("yarn", {})
                p.setdefault("rounds", round_crown_rounds(int(p.get("max", 54)), float(y.get("stitch_mm", 4.0)),
                                                          float(y.get("round_mm", 3.6))))
            else:
                p.setdefault("rounds", head_rounds(int(p.get("max", 54)), bool(p.get("oval"))))
            p.setdefault("stuff", "firm")
        elif kind == "ear":
            ek = p.get("kind", "round")
            M = p.get("max") or EAR_MAX_REL.get(ek, 0.33) * _head_max(out)
            p.setdefault("rounds", ear_rounds(ek, float(M), p.get("length")))
            p.setdefault("end", "open")
            p.setdefault("stuff", "none")
            key, amount = EAR_PRESS[ek]
            if "spoon" not in p and "dish" not in p:
                p[key] = amount
            if p.get("inner"):
                # the inside of the ear in its own colour, shaped like the ear (toy.py _inner_mask); a coloured
                # inside reads as a hollow only when the spoon is pressed deeper (blind look 2026-10-07: 0.55)
                p.setdefault("colours", []).insert(0, {"inner": True, "colour": p["inner"]})
                if key == "spoon" and p.get("spoon") == amount:
                    p["spoon"] = INNER_SPOON.get(ek, amount)
            if (ek == "lop" or (ek in ("long", "floppy") and p.get("hang"))) and "bend" not in p:
                from .rounds import expand as _exp
                b = dict(HANG_BEND); b["at"] = len(_exp(p["rounds"]))
                b["toward"] = list(HANG_BEND["toward"])
                p["bend"] = b
                hang_parts.add(p["id"])
            if p.get("dish"):
                from .rounds import counts
                c = counts(p["rounds"])
                R = (c.index(max(c)) + 1) * float(out.get("yarn", {}).get("round_mm", 3.6))
                dish_parts[p["id"]] = round(0.55 * R, 1)     # half the lower rim sunk into the head, as sewn
        else:
            raise ValueError(f"part {p.get('id')}: unknown preset {kind!r} (cute_head, ear)")
        for k in PRESET_KEYS:
            p.pop(k, None)
    for e in out.get("assembly", []):
        if e.get("part") in hang_parts and not e.get("applique"):
            e.setdefault("dir", list(HANG_DIR))
        if e.get("part") in dish_parts and not e.get("applique"):
            e.setdefault("attach", "edge")
            e.setdefault("face", [0.1, 0.25, 1.0])
            e.setdefault("sink", dish_parts[e["part"]])
    return out
