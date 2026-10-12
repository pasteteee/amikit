"""Customising a toy in words a person (or a vision model looking at a pet photo) would use: "a dark spot under
the left eye", "white socks", "ears stand up", "a curled tail", "the coat is grey". Each customisation is an edit
op (amikit/edit.py) that rewrites the spec in the engine's own terms - colour zones, patches, rounds, constructions -
so the model is still exactly what the written pattern makes.

Ops (all go through edit.apply, so they mix with the low-level set/add/remove ops):

  {"op": "colour",  "target": T, "colour": C}
        T = all | coat | head | body | muzzle | ears | inner_ears | tail | legs | front_legs | hind_legs | nose |
            eyes | <part id>. `all` = every animal piece in this colour and its markings removed (start a pet from
            a clean coat); `coat` = every piece and zone in the head's colour (the main colour).
  {"op": "spot",    "colour": C, "near": "eye_L" | "eye_R" | "eyes", "place": around|under|over|outer|inner,
                    "size_mm": [w, h] | d, "offset_mm": [out, up], "wobble": 0.25, "rotate": 0}
  {"op": "spot",    "colour": C, "on": <role or part id>, "where": W, ...}
        `method`: yarn (default: stitch-exact colour change in the pieces' rounds) | satin (embroidered over the
        crochet: smooth oval, best under ~14 mm) | felt | auto (satin when small).
        W = top | forehead | back | front | chest | belly | rump | left | right | cheek_L | cheek_R | chin |
            tip | base | middle, or "at": {"round": r, "az": a}. A spot covers every piece it reaches (a patch
            under the eye spreads onto the muzzle), only on the side it is on; `mirror: true` = on both sides.
  {"op": "marking", "kind": K, "colour": C, "amount": 0..1, "side": "L"|"R"|"both"}
        K = blaze | mask | eye_patch | eyebrows | tan_points | socks | front_socks | hind_socks | tail_tip | stripes |
            ear_tips | chest | belly | saddle | spots | muzzle | inner_ears | half_face | collar
        (`amount` = how big / how far up, default per kind; spots take `count`, `seed`, `on`.)
  {"op": "ears",    "kind": pointed|round|floppy|long|folded, "size": 1.0, "colour": C, "inner": C}
  {"op": "tail",    "kind": curled|straight|up|stub|long, "length": 1.0, "colour": C}
  {"op": "shape",   "target": T, "size": f, "length": f, "width": f}       f = factor (1.2 = 20 % more)
  {"op": "eyes",    "size": f | "size_mm": mm, "colour": C, "highlight": true}
  {"op": "nose",    "colour": C, "size": f}

Sides: L / R are the TOY's (and the dog's own) left / right - the toy's left eye is on the viewer's RIGHT in a
front photo. Colours: a name already in the spec, a common name (brown, cream, ...) or "#rrggbb"; new hex colours get
a name and a Russian name for the pattern.
"""
from __future__ import annotations
import copy
import json
import re

import numpy as np

# ------------------------------------------------------------------------------------------------- colours
PALETTE = {   # name: (hex, Russian name) - the yarn shades a maker can buy, used to name a photo's colours
    "black": ("#2b2726", "чёрный"), "charcoal": ("#45403d", "графитовый"), "grey": ("#8a8784", "серый"),
    "silver": ("#b9b7b3", "светло-серый"), "white": ("#f2efe9", "белый"), "cream": ("#eee3cf", "кремовый"),
    "sand": ("#dcc39b", "песочный"), "fawn": ("#c9a06a", "палевый"), "caramel": ("#c4843e", "карамельный"),
    "ginger": ("#d2782f", "рыжий"), "rust": ("#a9532a", "ржаво-рыжий"), "tan": ("#b8834e", "светло-коричневый"),
    "brown": ("#7c4a2a", "коричневый"), "chocolate": ("#4e3022", "шоколадный"), "liver": ("#6a3b2e", "печёночный"),
    "golden": ("#d9a441", "золотистый"), "apricot": ("#e2a06a", "абрикосовый"), "red": ("#b8322b", "красный"),
    "pink": ("#ec9fa8", "розовый"), "nose_pink": ("#d98a8a", "розовый (нос)"), "blue": ("#3b6fb6", "синий"),
    "navy": ("#26355a", "тёмно-синий"), "green": ("#4f8a4b", "зелёный"), "yellow": ("#f1cf47", "жёлтый"),
    "orange": ("#e9822f", "оранжевый"), "purple": ("#7a4f9a", "фиолетовый"), "amber": ("#b8742a", "янтарный"),
    "hazel": ("#8a6a3a", "орехово-карий"), "eye_blue": ("#6fa7d8", "голубой"),
}


def _hex(c):
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(x * 2 for x in c)
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float)


def _lab(h):
    """Rough CIELAB of a hex colour (sRGB, D65) for 'is it the same yarn' comparisons."""
    rgb = _hex(h) / 255.0
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M @ rgb / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def colour_name(spec, c):
    """The spec's colour name for C (a name or "#hex"): reuses a spec colour of (nearly) the same shade, else adds
    one named after the nearest yarn shade, with its Russian name for the pattern."""
    cols = spec.setdefault("colours", {})
    if not isinstance(c, str):
        raise ValueError(f"colour {c!r}: give a name or #rrggbb")
    if not c.startswith("#"):
        if c in cols or c in ("black", "white"):
            return c
        key = c.lower().replace(" ", "_")
        if key in PALETTE:
            cols.setdefault(key, PALETTE[key][0])
            _ru(spec, key, PALETTE[key][1])
            return key
        raise ValueError(f"unknown colour {c!r}: use a spec colour ({', '.join(cols)}), a common name or #rrggbb")
    lab = _lab(c)
    for k, v in cols.items():
        if isinstance(v, str) and v.startswith("#") and np.linalg.norm(_lab(v) - lab) < 6.0:
            return k
    best = min(PALETTE, key=lambda k: np.linalg.norm(_lab(PALETTE[k][0]) - lab))
    name, j = best, 2
    while name in cols:
        name = f"{best}{j}"; j += 1
    cols[name] = c.lower()
    _ru(spec, name, PALETTE[best][1])
    return name


def _ru(spec, name, ru):
    from .pattern_text import colour_ru
    if name in spec.get("colour_names_ru", {}) or colour_ru(name):
        return
    spec.setdefault("colour_names_ru", {})[name] = ru


# --------------------------------------------------------------------------------------------- roles
ROLE_RULES = [  # (role, substrings) - first match wins
    ("inner_ear", ("inner_ear", "innerear", "ear_inner", "ear_lining")),
    ("ear", ("ear",)),
    ("muzzle", ("muzzle", "snout", "mouth_piece")),
    ("head", ("head",)),
    ("body", ("body", "torso")),
    ("tail", ("tail",)),
    ("front_leg", ("front_leg", "front_paw", "arm", "hand", "foreleg")),
    ("haunch", ("haunch", "thigh")),
    ("hind_leg", ("hind_leg", "hind_foot", "hind_paw", "back_leg", "leg", "foot", "feet", "paw")),
]
TARGETS = {
    "ears": ["ear"], "ear": ["ear"], "inner_ears": ["inner_ear"], "head": ["head"], "body": ["body"],
    "muzzle": ["muzzle"], "snout": ["muzzle"], "tail": ["tail"], "legs": ["front_leg", "hind_leg", "haunch"],
    "paws": ["front_leg", "hind_leg"], "front_legs": ["front_leg"], "arms": ["front_leg"],
    "hind_legs": ["hind_leg", "haunch"],
}


def role_of(pid):
    s = pid.lower()
    for role, keys in ROLE_RULES:
        if any(k in s for k in keys):
            return role
    return None


def roles(spec):
    out = {}
    for p in spec.get("parts", []):
        r = role_of(p["id"])
        if r:
            out.setdefault(r, []).append(p["id"])
    return out


def _parts(spec, target):
    """Part ids for a target word, a role or a part id."""
    ids = {p["id"] for p in spec["parts"]}
    if target in ids:
        return [target]
    rs = roles(spec)
    want = TARGETS.get(target, [target])
    got = [pid for r in want for pid in rs.get(r, [])]
    if not got:
        raise ValueError(f"no piece for {target!r} (pieces: {', '.join(sorted(ids))})")
    return got


def _part(spec, pid):
    return next(p for p in spec["parts"] if p["id"] == pid)


def _entries(spec, pid):
    return [e for e in spec.get("assembly", []) if e["part"] == pid]


def _eyes(spec):
    out = []
    for f in spec.get("face", []):
        if f.get("type") in ("safety_eye", "eye") or (f.get("type") == "satin" and f.get("role") == "eye"):
            out.append(f)
    return out


def _eye_d(spec):
    e = _eyes(spec)
    if not e:
        return 10.0
    f = e[0]
    return float(f.get("size_mm", f.get("width_mm", 10.0)) if not isinstance(f.get("size_mm"), list)
                 else f["size_mm"][0])


# ------------------------------------------------------------------------------------- built toy (cached)
_TOY = {}


def _toy(spec):
    """The placed toy for geometry questions (where is the top of the head, which piece is under this point).
    Colours do not move anything, so the cache ignores them: many spots in a row cost one build."""
    from .presets import expand
    from .toy import Toy
    geo = {"parts": [{k: v for k, v in p.items() if k not in ("colours", "colour")} for p in spec["parts"]],
           "assembly": spec.get("assembly"), "yarn": spec.get("yarn")}
    key = json.dumps(geo, sort_keys=True, ensure_ascii=False)
    if key not in _TOY:
        _TOY.clear()
        clean = {k: v for k, v in spec.items() if not k.startswith("_")}
        _TOY[key] = Toy(expand(clean))
    toy = _TOY[key]
    toy.spec["face"] = copy.deepcopy(spec.get("face", []))      # eyes may have moved
    return toy


DIRS = {"top": (0, 1, 0), "forehead": (0, 0.8, 1), "front": (0, 0, 1), "chest": (0, 0.45, 1),
        "belly": (0, -0.5, 1), "back": (0, 0.35, -1), "rump": (0, -0.4, -1), "bottom": (0, -1, 0),
        "left": (1, 0, 0), "right": (-1, 0, 0), "chin": (0, -0.8, 1), "cheek_L": (0.8, -0.3, 1),
        "cheek_R": (-0.8, -0.3, 1), "nape": (0, 0.5, -1)}


def surface_place(spec, pid, where):
    """(round, az) on piece `pid` for a region word: the stitch whose position looks most along the region's world
    direction (top = highest, back = rearmost, ...); tip / base / middle = along the piece (round 1 is the tip of
    a tail, an ear or a paw)."""
    toy = _toy(spec)
    names = [n for n, i in toy.inst.items() if i.p.id == pid]
    if not names:
        raise ValueError(f"piece {pid!r} is not placed")
    inst = toy.inst[names[0]]                      # the left copy of a pair (the spot side is set by the op)
    N = inst.p.N
    if where in ("tip", "base", "middle", "start", "end"):
        r = {"tip": 1.5, "start": 1.5, "base": N - 0.5, "end": N - 0.5, "middle": N / 2}[where]
        return {"round": round(r, 1), "az": 0}
    if where not in DIRS:
        raise ValueError(f"unknown place {where!r} ({', '.join(list(DIRS) + ['tip', 'base', 'middle'])})")
    d = np.array(DIRS[where], float); d /= np.linalg.norm(d)
    t = np.linspace(0.6, inst.p.M - 0.2, 60)
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    T, TH = np.meshgrid(t, th, indexing="ij")
    P = inst.base(T, TH)
    ctr = P.reshape(-1, 3).mean(0)
    s = (P - ctr) @ d
    k = np.unravel_index(int(np.argmax(s)), s.shape)
    az = float((inst.az_of_theta(TH[k]) + 180) % 360 - 180)
    if inst.mirror:
        az = -az
    return {"round": round(float(T[k]) + 0.5, 1), "az": round(az, 1)}


def _size_of(spec, pid):
    """(width, length) of a placed piece in mm."""
    toy = _toy(spec)
    inst = next(i for i in toy.inst.values() if i.p.id == pid)
    return 2 * float(inst.p.rb.max() + inst.p.oval / 2), float(inst.p.zb.max() - inst.p.zb.min())


# --------------------------------------------------------------------------------------------- patches
def _add_patch(spec, pz, colour, parts=None):
    """Add a world-placed patch zone to every candidate piece; prune() later drops the ones it misses."""
    cname = colour_name(spec, colour)
    n = sum(1 for p in spec["parts"] for z in p.get("colours", []) if str(z.get("id", "")).startswith("spot"))
    zid = f"spot{n + 1}"
    pz = dict(pz, exact=True)
    cands = parts or [p["id"] for p in spec["parts"] if "rounds" in p and not p.get("motif")
                      and role_of(p["id"]) not in ("inner_ear",)]
    for pid in cands:
        _part(spec, pid).setdefault("colours", []).append({"patch": copy.deepcopy(pz), "colour": cname,
                                                           "id": zid})
    spec.setdefault("_pending_prune", []).append(zid)
    return zid


def prune(spec):
    """Drop patch zones (added by spot/marking ops) from the pieces where they take no stitch."""
    ids = set(spec.pop("_pending_prune", []))
    if not ids:
        return spec
    from .presets import expand
    from .toy import Toy
    toy = Toy(expand(spec))
    used = {}
    for name, inst in toy.inst.items():
        p = inst.p
        for z in p.zones:
            if z.get("id") not in ids:
                continue
            hit = False
            for i, n in enumerate(p.counts, 1):
                from .piece import LEAN
                th = 2 * np.pi * (np.arange(n) + 0.5 - LEAN * (i - 0.5)) / n
                if toy._patch_mask(inst, i, n, th, z["patch"]).any():
                    hit = True
                    break
            if hit:
                used.setdefault(z["id"], set()).add(p.id)
    for p in spec["parts"]:
        if "colours" in p:
            p["colours"] = [z for z in p["colours"] if z.get("id") not in ids or p["id"] in used.get(z["id"], ())]
            if not p["colours"]:
                del p["colours"]
    missing = ids - set(used)
    if missing:
        raise ValueError(f"the spot(s) {sorted(missing)} land on no piece: move them onto the toy")
    return spec


def _spot(spec, op):
    d = _eye_d(spec)
    pz = {}
    near = op.get("near")
    if near:
        m = re.fullmatch(r"eye_?([LRlr])?|eyes", near)
        if not m:
            raise ValueError(f"near: eye_L, eye_R or eyes (got {near!r})")
        side = (m.group(1) or "L").upper()
        pz["eye"] = side
        if near == "eyes":
            pz["mirror"] = True
        place = op.get("place", "around")
        size = op.get("size_mm") or ([2.3 * d, 2.1 * d] if place == "around" else [max(10, 1.3 * d), max(9, 1.1 * d)])
        w, h = (size, size) if np.isscalar(size) else size
        r = d / 2
        off = {"around": [0, 0], "under": [0.2 * d, -(r + 0.42 * h)], "over": [0, r + 0.42 * h],
               "outer": [r + 0.42 * w, 0], "inner": [-(r + 0.42 * w), 0]}.get(place)
        if off is None:
            raise ValueError(f"place: around, under, over, outer, inner (got {place!r})")
        o2 = op.get("offset_mm", [0, 0])
        pz["offset_mm"] = [round(off[0] + o2[0], 1), round(off[1] + o2[1], 1)]
    else:
        on = op.get("on", "body")
        pid = _parts(spec, on)[0]
        side = str(op.get("side", "L")).upper()[:1]
        if "at" in op:
            at = dict(op["at"])
        else:
            at = surface_place(spec, pid, op.get("where", "top"))
        if side == "R":
            at["az"] = -at.get("az", 0)
        W, L = _size_of(spec, pid)
        size = op.get("size_mm") or [round(0.45 * W, 1), round(0.4 * W, 1)]
        w, h = (size, size) if np.isscalar(size) else size
        pz.update(on=pid, at=at)
        if side == "R":
            pz["anchor_side"] = "R"
        if op.get("offset_mm"):
            pz["offset_mm"] = list(op["offset_mm"])
        if op.get("mirror"):
            pz["mirror"] = True
    pz["size_mm"] = [round(float(w), 1), round(float(h), 1)]
    if op.get("rotate"):
        pz["rotate"] = float(op["rotate"])
    wob = float(op.get("wobble", 0.2))
    method = op.get("method", "yarn")
    if method == "auto":
        method = "satin" if max(w, h) <= 14 else "yarn"
    if method in ("satin", "felt"):
        return _sewn_spot(spec, pz, op, w, h, method)
    if method != "yarn":
        raise ValueError(f"method: yarn (colour change), satin (embroidered), felt or auto (got {method!r})")
    if wob:
        pz["wobble"] = wob
        pz["seed"] = int(op.get("seed", 1 + len(json.dumps(spec)) % 97))
    return _add_patch(spec, pz, op["colour"], op.get("parts"))


def _sewn_spot(spec, pz, op, w, h, method):
    """A spot embroidered in satin stitch (or a felt piece glued/sewn on) instead of a colour change: smooth
    round edges, good for small spots, brows and dots. Placed on the piece nearest to the spot's centre."""
    toy = _toy(spec)
    inst0 = next(iter(toy.inst.values()))
    flips = (False, True) if pz.get("mirror") else (False,)
    cname = colour_name(spec, op["colour"])
    n = sum(1 for f in spec.get("face", []) if str(f.get("id", "")).startswith("spot"))
    made = []
    for flip in flips[:1]:
        c, nrm, _, _, _ = toy.patch_frame(inst0, dict(pz, exact=True), flip)
        best = None
        for name, inst in toy.inst.items():
            if role_of(inst.p.id) in (None, "inner_ear") or inst.app is not None:
                continue
            t = np.linspace(0.3, inst.p.M - 0.1, 40)
            th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            T, TH = np.meshgrid(t, th, indexing="ij")
            dd = float(np.linalg.norm(inst.base(T, TH) - c, axis=-1).min())
            if best is None or dd < best[0]:
                best = (dd, name, inst)
        if best is None or best[0] > 0.5 * max(w, h) + 3:
            raise ValueError("the spot's centre is off the toy")
        _, name, inst = best
        at = toy._locate_on(inst, c)
        on = name
        if inst.p.id + "_L" == name or inst.p.id + "_R" == name:
            on = name
        elif name.endswith("_L") or name.endswith("_R"):
            on = name
        item = {"type": "satin" if method == "satin" else "felt", "on": on, "at": at, "colour": cname,
                "id": f"spot{n + 1}"}
        if method == "satin":
            item.update(width_mm=round(w, 1), height_mm=round(h, 1), shape="oval", thickness_mm=1.4)
        else:
            item.update(size_mm=[round(w, 1), round(h, 1)])
        if op.get("rotate"):
            item["rotate"] = float(op["rotate"])
        if pz.get("mirror"):
            item["mirror"] = True
        spec.setdefault("face", []).append(item)
        made.append(item["id"])
    return made[0]


# --------------------------------------------------------------------------------------------- rounds edits
def _step_spans(steps):
    from .rounds import counts
    spans, prev = [], 0
    for j in range(len(steps)):
        n = len(counts(steps[:j + 1]))
        spans.append((prev + 1, n))
        prev = n
    return spans


def remap_rounds(spec, pid, fmap):
    """Renumber everything that refers to rounds of piece `pid` (its zones and bend, entries and face items on
    it) through fmap(old_round) -> new_round."""
    p = _part(spec, pid)
    for z in p.get("colours", []):
        if "rounds" in z:
            a, b = z["rounds"]
            z["rounds"] = [max(1, int(round(fmap(a)))), max(1, int(round(fmap(b))))]
        if "round" in z:
            z["round"] = int(round(fmap(z["round"])))
        if "shape" in z:
            z["shape"] = [[round(fmap(r), 2), a] for r, a in z["shape"]]
        if "ellipse" in z:
            rc = z["ellipse"]["center"][0]
            k = fmap(rc + 0.5) - fmap(rc)
            z["ellipse"]["center"][0] = round(fmap(rc), 2)
            z["ellipse"]["radii"][0] = round(z["ellipse"]["radii"][0] * 2 * k, 2)
    if isinstance(p.get("bend"), dict) and "at" in p["bend"]:
        p["bend"]["at"] = round(fmap(p["bend"]["at"]), 1)
    names = {e.get("name", e["part"]) for e in spec.get("assembly", []) if e["part"] == pid} | {pid}
    for it in spec.get("assembly", []) + spec.get("face", []):
        if it.get("on") in names:
            at = it.get("at", {})
            if isinstance(at.get("round"), (int, float)):
                at["round"] = round(fmap(at["round"]) * 2) / 2
            if "points" in it:
                it["points"] = [[round(fmap(r), 2), a] for r, a in it["points"]]
    for q in spec["parts"]:
        for z in q.get("colours", []):
            pz = z.get("patch")
            if pz and pz.get("on") in names and isinstance(pz.get("at", {}).get("round"), (int, float)):
                pz["at"]["round"] = round(fmap(pz["at"]["round"]), 2)


def change_length(spec, pid, factor):
    """Longer / shorter by adding or removing straight rounds where the piece is already straight (its longest
    run of equal rounds, or right after its widest round)."""
    from .rounds import counts
    p = _part(spec, pid)
    steps = list(p["rounds"])
    N = len(counts(steps))
    k = int(round((factor - 1.0) * N))
    if k == 0:
        return
    spans = _step_spans(steps)
    evens = [(int(s.split()[1]), j) for j, s in enumerate(steps) if re.fullmatch(r"even \d+", s)]
    if evens:
        K, j = max(evens)
    else:
        c = counts(steps)
        iw = c.index(max(c)) + 1
        j = next(jj for jj, (a, b) in enumerate(spans) if a <= iw <= b)
        steps.insert(j + 1, "even 0")
        spans = _step_spans(steps[:j + 1]) + [(spans[j][1] + 1, spans[j][1])] + spans[j + 1:]
        K, j = 0, j + 1
    newK = max(0, K + k)
    k = newK - K
    if k == 0:
        return
    a = spans[j][0]
    end = a + K - 1
    steps[j] = f"even {newK}"
    if newK == 0:
        steps.pop(j)
    p["rounds"] = steps
    counts(steps)                                    # valid

    def fmap(r):
        r = float(r)
        if r <= a - 0.5 or K == 0 and r <= end + 0.5:
            return r
        if r > end + 0.5:
            return r + k
        return (a - 0.5) + (r - (a - 0.5)) * (newK / max(K, 1e-9))
    remap_rounds(spec, pid, fmap)


def change_width(spec, pid, factor):
    """Wider / narrower: the same number of rounds and the same profile, a bigger / smaller widest round."""
    from .rounds import expand, counts
    from .library import synthesize, counts_to_steps
    p = _part(spec, pid)
    rounds = expand(p["rounds"])
    c = [r.n for r in rounds]
    if rounds[0].start == "ch" or any(r.loop not in ("both", None, "") for r in rounds):
        from .edit import scale_steps
        raise ValueError(f"{pid}: change the width of a chain-oval or special-stitch piece with shape size")
    start = c[0]
    incs = [b - a for a, b in zip(c, c[1:]) if b > a]
    step = int(np.median(incs)) if incs else start
    M = max(start + step, int(round(max(c) * factor / step)) * step)
    new = synthesize(np.array(c, float) / max(c), len(c), M, start=start, step=max(2, step))
    p["rounds"] = counts_to_steps(new)
    counts(p["rounds"])


def change_size(spec, pid, factor):
    from .edit import apply
    new, _ = apply(spec, [{"op": "scale_part", "part": pid, "factor": factor}], validate=False)
    spec.clear(); spec.update(new)


# --------------------------------------------------------------------------------------------- ops
def op_colour(spec, op):
    t = op["target"]
    c = colour_name(spec, op["colour"])
    if t == "all":
        # a clean start for a pet photo: every animal piece in one colour, earlier markings gone (accessories,
        # inner ears, eyes and nose keep theirs)
        for p in spec["parts"]:
            r = role_of(p["id"])
            if r is None or r == "inner_ear" or "rounds" not in p and "preset" not in p:
                continue
            p["colour"] = c
            p.pop("colours", None)
        spec["face"] = [f for f in spec.get("face", []) if f.get("type") != "satin" or f.get("role") in
                        ("eye", "glint", "nose") or f.get("id") == "nose"]
        return
    if t == "coat":
        # the main colour of the coat: the head's colour (a tricolour beagle's body saddle is a marking)
        main = _parts(spec, "head")[0] if roles(spec).get("head") else _parts(spec, "body")[0]
        old = _part(spec, main).get("colour")
        for p in spec["parts"]:
            if p.get("colour") == old:
                p["colour"] = c
            for z in p.get("colours", []):
                if z.get("colour") == old:
                    z["colour"] = c
                if "runs" in z:
                    z["runs"] = [[c if a == old else a, k] for a, k in z["runs"]]
        return
    if t in ("nose", "eyes"):
        kinds = ("safety_nose",) if t == "nose" else ("safety_eye", "eye")
        hit = False
        for f in spec.get("face", []):
            if f.get("type") in kinds or (f.get("type") == "satin" and f.get("role") == t.rstrip("s")):
                if t == "eyes" and f.get("type") != "satin":
                    f["iris"] = c                     # a coloured iris round a black pupil
                    f.setdefault("pupil", 0.55)
                else:
                    f["colour"] = c
                hit = True
            elif t == "nose" and f.get("type") == "satin" and (f.get("id") == "nose" or f.get("shape") == "triangle"):
                f["colour"] = c; hit = True
        if not hit:
            raise ValueError(f"this toy has no {t}")
        return
    if t == "inner_ears":
        return op_marking(spec, {"kind": "inner_ears", "colour": op["colour"]})
    for pid in _parts(spec, t):
        p = _part(spec, pid)
        old = p.get("colour")
        p["colour"] = c
        for z in p.get("colours", []):          # bands in the old main colour (two-tone leg rounds) follow
            if z.get("colour") == old and "patch" not in z:
                z["colour"] = c


def _eye_rounds(spec):
    e = [f for f in _eyes(spec) if "at" in f]
    if not e:
        raise ValueError("this toy has no eyes to place the marking by")
    return float(np.mean([f["at"]["round"] for f in e])), float(np.mean([abs(f["at"].get("az", 0)) for f in e]))


def op_marking(spec, op):
    k = op["kind"]
    amt = op.get("amount")
    d = _eye_d(spec)
    side = str(op.get("side", "both"))
    sides = ["L", "R"] if side == "both" else [side.upper()[:1]]
    c = op["colour"]
    if k == "blaze":
        head = _parts(spec, "head")[0]
        er, eaz = _eye_rounds(spec)
        a = float(amt or 0.5)
        wt = 6 + 8 * a                                   # half-width at the crown, degrees
        wb = max(wt + 2, eaz * (0.35 + 0.4 * a))         # between the eyes
        cname = colour_name(spec, c)
        _part(spec, head).setdefault("colours", []).append(
            {"shape": [[1, -wt * 2], [1, wt * 2], [er * 0.45, wt], [er - 1, wb * 0.8], [er + 1.5, wb],
                       [er + 1.5, -wb], [er - 1, -wb * 0.8], [er * 0.45, -wt]], "colour": cname, "id": "blaze"})
        if op.get("muzzle", True) and roles(spec).get("muzzle"):
            for pid in roles(spec)["muzzle"]:
                _part(spec, pid)["colour"] = cname
        return
    if k == "mask":
        for pid in roles(spec).get("muzzle", []):
            _part(spec, pid)["colour"] = colour_name(spec, c)
        a = float(amt or 0.5)
        _spot(spec, {"near": "eyes", "place": "around", "colour": c, "size_mm": [d * (2.2 + a), d * (2.0 + a)],
                     "offset_mm": [0, -0.15 * d], "wobble": 0.12})
        return
    if k in ("eye_patch", "patch"):
        a = float(amt or 0.5)
        for s in sides:
            _spot(spec, {"near": f"eye_{s}", "place": "around", "colour": c,
                         "size_mm": [d * (2.0 + 1.6 * a), d * (1.9 + 1.4 * a)], "wobble": 0.25,
                         "seed": 7 if s == "L" else 11})
        return
    if k == "eyebrows":
        for s in sides:
            _spot(spec, {"near": f"eye_{s}", "place": "over", "colour": c, "size_mm": [0.75 * d, 0.65 * d],
                         "offset_mm": [0.15 * d, 0.2 * d], "wobble": 0, "method": op.get("method", "satin")})
        return
    if k == "tan_points":                               # rottweiler / dobermann: brows + cheeks + chest (+ socks)
        op_marking(spec, dict(op, kind="eyebrows"))
        for s in sides:
            _spot(spec, {"near": f"eye_{s}", "place": "under", "colour": c, "size_mm": [1.1 * d, 0.9 * d],
                         "offset_mm": [0.45 * d, -0.4 * d], "wobble": 0.1})
        return
    if k in ("socks", "front_socks", "hind_socks"):
        tgt = {"socks": "paws", "front_socks": "front_legs", "hind_socks": "hind_legs"}[k]
        a = float(amt or 0.35)
        cname = colour_name(spec, c)
        from .presets import expand as ex
        for pid in _parts(spec, tgt):
            p = ex({"parts": [_part(spec, pid)]})["parts"][0]
            if "rounds" not in p:
                continue
            from .rounds import counts
            N = len(counts(p["rounds"]))
            last = max(1, int(round(a * N)))
            zs = _part(spec, pid).setdefault("colours", [])
            zs.append({"rounds": [1, last], "colour": cname, "id": k})
        return
    if k in ("tail_tip", "ear_tips"):
        a = float(amt or (0.3 if k == "tail_tip" else 0.35))
        cname = colour_name(spec, c)
        from .rounds import counts
        for pid in _parts(spec, "tail" if k == "tail_tip" else "ears"):
            p = _part(spec, pid)
            if "rounds" not in p:
                continue
            N = len(counts(p["rounds"]))
            p.setdefault("colours", []).append({"rounds": [1, max(1, int(round(a * N)))], "colour": cname, "id": k})
        return
    if k in ("chest", "belly", "saddle", "back"):
        body = _parts(spec, "body")[0]
        W, L = _size_of(spec, body)
        a = float(amt or 0.5)
        where = {"chest": "chest", "belly": "belly", "saddle": "back", "back": "back"}[k]
        size = {"chest": [W * (0.45 + 0.4 * a), L * (0.4 + 0.5 * a)], "belly": [W * (0.4 + 0.4 * a), L * (0.35 + 0.4 * a)],
                "saddle": [W * (0.6 + 0.5 * a), L * (0.5 + 0.5 * a)], "back": [W * (0.6 + 0.5 * a), L * (0.5 + 0.5 * a)]}[k]
        _spot(spec, {"on": body, "where": where, "colour": c, "size_mm": size, "wobble": 0.12 if k != "saddle" else 0.2,
                     "parts": op.get("parts", [body])})
        return
    if k == "spots":
        rng = np.random.default_rng(int(op.get("seed", 3)))
        on = op.get("on", ["body", "head"])
        on = [on] if isinstance(on, str) else on
        n = int(op.get("count", 8))
        sz = float(op.get("size_mm", 0.9 * d))
        regions = ["left", "right", "back", "top", "chest", "rump", "belly", "nape"]
        made = 0
        tries = 0
        while made < n and tries < 4 * n:
            tries += 1
            pid = _parts(spec, on[int(rng.integers(len(on)))])[0]
            base = surface_place(spec, pid, regions[int(rng.integers(len(regions)))])
            at = {"round": round(max(1.5, base["round"] + float(rng.normal(0, 3))), 1),
                  "az": round(base["az"] + float(rng.normal(0, 35)), 1)}
            s = sz * float(rng.uniform(0.7, 1.3))
            try:
                _spot(spec, {"on": pid, "at": at, "colour": c, "size_mm": [s, s * float(rng.uniform(0.75, 1.0))],
                             "wobble": 0.2, "seed": int(rng.integers(1, 999)), "method": op.get("method", "auto")})
                made += 1
            except ValueError:
                pass
        return
    if k == "stripes":
        # tabby stripes: single rounds in the stripe colour every `every` rounds (default 3) - full rounds on the
        # tail and legs, the back and sides only on the body and head (the chest / face stay clear)
        from .rounds import counts
        cname = colour_name(spec, c)
        every = int(op.get("every", 3))
        on = op.get("on", ["tail", "legs", "body", "head"])
        on = [on] if isinstance(on, str) else on
        for tgt in on:
            try:
                pids = _parts(spec, tgt)
            except ValueError:
                continue
            for pid in pids:
                p = _part(spec, pid)
                if "rounds" not in p:
                    continue
                N = len(counts(p["rounds"]))
                r = role_of(pid)
                zs = p.setdefault("colours", [])
                first = 2 if r in ("tail", "front_leg", "hind_leg") else 3
                for i in range(first, N, every):
                    z = {"rounds": [i, i], "colour": cname, "id": "stripes"}
                    if r in ("body", "head"):
                        z["az"] = [70, 290]
                    zs.append(z)
        return
    if k == "muzzle":
        for pid in _parts(spec, "muzzle"):
            _part(spec, pid)["colour"] = colour_name(spec, c)
        return
    if k == "inner_ears":
        cname = colour_name(spec, c)
        rs = roles(spec)
        from .presets import INNER_SPOON
        for pid in rs.get("ear", []):              # a coloured inside needs a visible hollow
            p = _part(spec, pid)
            if p.get("spoon") and float(p["spoon"]) < 0.55 and ear_kind(spec, pid) in INNER_SPOON:
                p["spoon"] = INNER_SPOON[ear_kind(spec, pid)]
        for pid in rs.get("inner_ear", []):                 # replace sewn-on inner ears by the ear-shaped zone
            from .edit import apply
            new, _ = apply(spec, [{"op": "remove", "path": f"parts.{pid}"}], validate=False)
            spec.clear(); spec.update(new)
        for pid in rs.get("ear", []):
            p = _part(spec, pid)
            zs = [z for z in p.get("colours", []) if not z.get("inner")]
            p["colours"] = [{"inner": True, "colour": cname}] + zs
            if amt:
                p["colours"][0]["width"] = float(amt)
        return
    if k == "half_face":
        head = _parts(spec, "head")[0]
        s = sides[0] if side != "both" else "L"
        rng_ = [0, 180] if s == "L" else [-180, 0]
        er, _ = _eye_rounds(spec)
        _part(spec, head).setdefault("colours", []).append(
            {"rounds": [1, int(round(er + 3))], "az": rng_, "colour": colour_name(spec, c), "id": "half_face"})
        return
    if k == "collar":
        body = _parts(spec, "body")[0]
        from .rounds import counts
        bp = _part(spec, body)
        N = len(counts(bp["rounds"]))
        r = N - 1.2
        spec.setdefault("face", []).append(
            {"type": "strap", "on": body, "points": [[r, a] for a in range(-180, 181, 20)],
             "colour": colour_name(spec, c), "thickness_mm": float(op.get("thickness_mm", 4.0)), "id": "collar"})
        return
    raise ValueError(f"unknown marking {k!r}")


EAR_PLACE = {   # kind -> (round as a share of the head's rounds, az, dir, extra entry keys)
    "pointed": (0.24, 58, [0.5, 1, 0.05], {}),
    "round": (0.26, 62, [0.6, 1, 0.0], {}),
    "long": (0.16, 30, [0.25, 1, -0.05], {}),
    # hound ears (the beagle construction that held up in blind A/B): sewn flat on the side of the head by a
    # front-to-back seam, pressed flat, falling down from the seam
    "floppy": (0.25, 95, [1, 0.2, 0.1], {"front": [0, 1, 0], "seam": "flat", "seam_line": [0, 0, 1]}),
    # button / rose ears (terriers, rottweiler): a pointed ear whose top half folds forward and down
    "folded": (0.24, 62, [0.45, 1, 0.15], {}),
}


def ear_kind(spec, pid):
    """The construction kind of an existing ear: round | pointed | long | floppy | folded (None if unclear)."""
    from .rounds import counts
    p = _part(spec, pid)
    if p.get("preset") == "ear":
        return p.get("kind", "round")
    if p.get("dish"):
        return "round"
    es = _entries(spec, pid)
    az = abs(float(es[0].get("at", {}).get("az", 0))) if es else 0
    bend = p.get("bend") or {}
    if bend and abs(float(bend.get("deg", 0))) >= 60:
        return "floppy" if az >= 75 else "folded"
    if p.get("spoon") and az >= 75:
        return "floppy"
    if "rounds" in p:
        c = counts(p["rounds"])
        return "long" if len(c) > 1.6 * (c.index(max(c)) + 1) + 4 else "pointed"
    return None


def op_ears(spec, op):
    from .presets import ear_rounds, EAR_MAX_REL, EAR_PRESS, _head_max
    from .rounds import counts
    rs = roles(spec)
    if not rs.get("ear"):
        raise ValueError("this toy has no ears")
    head = _parts(spec, "head")[0]
    hN = len(counts(_part(spec, head)["rounds"])) if "rounds" in _part(spec, head) else 20
    for pid in rs["ear"]:
        p = _part(spec, pid)
        kind = op.get("kind")
        if kind and kind == ear_kind(spec, pid):
            kind = None                       # already that kind: keep the ear, only resize / recolour
        if kind:
            kk = "pointed" if kind == "folded" else kind
            if kk not in EAR_PRESS:
                raise ValueError(f"ear kind: pointed, round, floppy, long, folded (got {kind!r})")
            size = float(op.get("size", 1.0)) * (1.2 if kind == "folded" else 1.0)
            M = size * EAR_MAX_REL[kk] * _head_max(spec)
            length = op.get("length", 7 if kind == "floppy" else None)
            p["rounds"] = ear_rounds(kk, M, length)
            if kind == "folded":
                p["rounds"] = p["rounds"][:-1] + ["even 3"]          # a longer flap to fold
            for key in ("flatten", "spoon", "dish", "bend", "squash", "preset", "kind", "max", "length"):
                p.pop(key, None)
            key, amount = EAR_PRESS[kk]
            p[key] = amount
            p["end"] = "open"; p["stuff"] = "none"
            fr, az, dr, extra = EAR_PLACE[kind]
            n = len(counts(p["rounds"]))
            if kind == "floppy":
                p.pop("spoon", None)
                p["flatten"] = 0.65
                p["bend"] = {"deg": 100, "at": n - 1, "over": 3, "toward": [0.3, -1, 0.1]}
            if kind == "folded":
                p["bend"] = {"deg": 150, "at": round(n * 0.55, 1), "over": 2, "toward": [0, -0.6, 1]}
            # sewn-on parts of the old ear (inner ear) become the ear-shaped inner zone
            inner = None
            for q in rs.get("inner_ear", []):
                if any(e.get("on") in (pid,) for e in _entries(spec, q)):
                    inner = _part(spec, q).get("colour")
            zones = [z for z in p.get("colours", []) if z.get("inner")]
            p["colours"] = zones
            if not p["colours"]:
                p.pop("colours")
            if inner:
                for q in list(rs.get("inner_ear", [])):
                    from .edit import apply
                    new, _ = apply(spec, [{"op": "remove", "path": f"parts.{q}"}], validate=False)
                    spec.clear(); spec.update(new)
                p = _part(spec, pid)
                p.setdefault("colours", []).insert(0, {"inner": True, "colour": inner})
            for e in _entries(spec, pid):
                for key in ("seam", "seam_line", "front", "attach", "face", "sink", "roll", "bend_toward",
                            "pinch", "pinch_rounds", "cup"):
                    e.pop(key, None)
                e["at"] = {"round": round(fr * hN * 2) / 2, "az": az}
                e["dir"] = list(dr)
                e.update(copy.deepcopy(extra))
                if kk == "round":
                    c = counts(p["rounds"])
                    R = (c.index(max(c)) + 1) * float(spec.get("yarn", {}).get("round_mm", 3.6))
                    e.update(attach="edge", face=[0.1, 0.25, 1.0], sink=round(0.55 * R, 1))
        elif op.get("size") and abs(float(op["size"]) - 1.0) > 0.02:
            change_size(spec, pid, float(op["size"]))
        if op.get("colour"):
            _part(spec, pid)["colour"] = colour_name(spec, op["colour"])
    if op.get("inner"):
        op_marking(spec, {"kind": "inner_ears", "colour": op["inner"]})


def op_tail(spec, op):
    from .rounds import counts
    pids = roles(spec).get("tail")
    if not pids:
        raise ValueError("this toy has no tail")
    for pid in pids:
        p = _part(spec, pid)
        kind = op.get("kind")
        if kind == "stub":
            p["rounds"] = ["mr 6", "12", "even 2", "10"]
            p.pop("bend", None)
            p.pop("colours", None)
        elif kind == "long":
            change_length(spec, pid, 1.5)
        if op.get("length"):
            change_length(spec, pid, float(op["length"]))
        p = _part(spec, pid)
        N = len(counts(p["rounds"]))
        for e in _entries(spec, pid):
            if kind == "curled":
                # a spitz tail: sewn high on the rump, rising and curling forward over the back
                p["bend"] = {"deg": 200, "at": round(N * 0.6, 1), "over": max(2, round(N * 0.45))}
                on = e.get("on")
                if on and on in {q["id"] for q in spec["parts"]} and "rounds" in _part(spec, on)                         and isinstance(e.get("at", {}).get("round"), (int, float)):
                    bN = len(counts(_part(spec, on)["rounds"]))
                    e["at"] = {"round": round(max(e["at"]["round"], 0.42 * bN) * 2) / 2, "az": 180}
                e["dir"] = [0, 0.75, -1]
                e["bend_toward"] = [0, 1, 0.5]
            elif kind == "up":
                p.pop("bend", None); e.pop("bend_toward", None)
                e["dir"] = [0, 1, -0.7]
            elif kind in ("straight", "long"):
                p.pop("bend", None); e.pop("bend_toward", None)
                e["dir"] = [0, 0.25, -1]
            elif kind == "stub":
                e["dir"] = [0, 0.3, -1]; e.pop("bend_toward", None)
            elif kind:
                raise ValueError(f"tail kind: curled, straight, up, stub, long (got {kind!r})")
        if op.get("colour"):
            p["colour"] = colour_name(spec, op["colour"])


def op_shape(spec, op):
    for pid in _parts(spec, op["target"]):
        if op.get("size"):
            change_size(spec, pid, float(op["size"]))
        if op.get("length"):
            change_length(spec, pid, float(op["length"]))
        if op.get("width"):
            change_width(spec, pid, float(op["width"]))


def op_eyes(spec, op):
    eyes = [f for f in _eyes(spec) if f.get("type") in ("safety_eye", "eye")]
    if not eyes:
        raise ValueError("this toy has no safety eyes")
    for f in eyes:
        if "size_mm" in op:
            f["size_mm"] = float(op["size_mm"])
        elif "size" in op:
            f["size_mm"] = round(float(f.get("size_mm", 9)) * float(op["size"]), 1)
        if "highlight" in op:
            f["highlight"] = bool(op["highlight"])
    if op.get("colour"):
        op_colour(spec, {"target": "eyes", "colour": op["colour"]})


def op_nose(spec, op):
    if op.get("colour"):
        op_colour(spec, {"target": "nose", "colour": op["colour"]})
    if op.get("size"):
        k = float(op["size"])
        for f in spec.get("face", []):
            if f.get("type") == "safety_nose":
                s = f.get("size_mm", 9)
                f["size_mm"] = [round(x * k, 1) for x in s] if isinstance(s, list) else round(s * k, 1)
            elif f.get("type") == "satin" and f.get("id") == "nose":
                f["width_mm"] = round(f.get("width_mm", 8) * k, 1); f["height_mm"] = round(f.get("height_mm", 5) * k, 1)


OPS = {"colour": op_colour, "spot": _spot, "marking": op_marking, "ears": op_ears, "tail": op_tail,
       "shape": op_shape, "eyes": op_eyes, "nose": op_nose}


def apply_op(spec, op):
    """Apply one customisation op to spec in place."""
    OPS[op["op"]](spec, {k: v for k, v in op.items() if k != "op"})


def describe(spec):
    """What a customiser can address on this toy (for an LLM prompt)."""
    rs = roles(spec)
    lines = ["pieces by role: " + "; ".join(f"{r}={','.join(v)}" for r, v in rs.items())]
    lines.append("colours: " + ", ".join(f"{k}={v}" for k, v in spec.get("colours", {}).items()))
    e = _eyes(spec)
    if e:
        lines.append(f"eyes: {len(e)} item(s), {_eye_d(spec):.0f} mm, on {e[0].get('on')}")
    spots = sorted({z["id"] for p in spec["parts"] for z in p.get("colours", []) if "id" in z})
    if spots:
        lines.append("markings already on the toy: " + ", ".join(spots))
    return "\n".join(lines)
