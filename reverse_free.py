"""Placements for designers' patterns: the assembly is free text ("Sew the ears between rounds 7-19", "Insert the
safety eyes between rounds 17-18 with an interspace of 13 stitches", "Sew the arms high on the body").

What the text states is used as stated (which piece goes on which, the rounds, the spacing, closing, stuffing,
flattening). What it leaves open is filled with the usual designers' construction for that part (ears on top of
the head, arms just under the neck hanging down, sitting legs on the front of the body pointing forward, tail at the
back) and every such choice is logged as "assumed: ..." in reverse_report.txt — the gaps of the pattern.

Positions are found on the parent as it is actually built: a wanted surface direction ("up and outward") is turned
into the round/stitch whose surface faces that way, so the defaults hold for any head or body shape.
"""
from __future__ import annotations
import math
import re

import numpy as np

from .reverse import build_toy, _clean_spec, part_id_of, PART_WORDS, NAMED_HEX, _strip_md

# ---------------------------------------------------------------------------------------------------------------
# what designers do when the text does not say (world directions: +x = the toy's left, +y up, +z front)
DEFAULTS = {
    #           surface normal wanted        direction the piece extends    note
    "ear":    ([0.74, 0.66, 0.05], [0.45, 0.88, 0.05]),
    "muzzle": ([0.0, -0.28, 1.0], [0.0, -0.1, 1.0]),
    "nose":   ([0.0, -0.1, 1.0], [0.0, 0.0, 1.0]),
    "beak":   ([0.0, -0.2, 1.0], [0.0, -0.1, 1.0]),
    "arm":    ([0.92, 0.25, 0.3], [0.3, -1.0, 0.35]),
    "leg":    ([0.45, -0.55, 0.7], [0.3, -0.25, 1.0]),
    "tail":   ([0.0, -0.45, -1.0], [0.0, 0.2, -1.0]),
    "wing":   ([1.0, 0.15, 0.0], [0.4, -1.0, 0.0]),
    "horn":   ([0.4, 0.9, 0.2], [0.3, 1.0, 0.1]),
    "fin":    ([1.0, 0.0, 0.0], [1.0, -0.3, 0.0]),
}
PARENT_OF = {"ear": "head", "muzzle": "head", "nose": "head", "beak": "head", "horn": "head", "inner_ear": "ear",
             "eye_patch": "head", "cheek": "head", "arm": "body", "leg": "body", "tail": "body", "wing": "body",
             "fin": "body", "belly": "body", "spot": "body"}
HORIZONTAL = r"\b(whale|fish|shark|seal|dolphin|turtle|snail|caterpillar|worm|crocodile|slug|axolotl|orca)\b"


def _kind(pid):
    """Part id -> its kind (front_leg -> leg, hind_leg -> leg, frontleg -> leg)."""
    s = pid.lower()
    for k in ("inner_ear", "eye_patch", "ear", "muzzle", "nose", "beak", "arm", "leg", "tail", "wing", "horn", "fin",
              "head_body", "head", "body", "belly", "cheek", "spot"):
        if k in s:
            return k
    if re.search(r"paw_?pad|pad", s):
        return "pad"
    if "snout" in s:
        return "muzzle"
    if re.search(r"foot|feet|paw", s):
        return "leg"
    return s


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def _sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if x.strip()]


def _context(doc):
    """[(piece index or None, last round label before it, sentence)] for every note in the pattern."""
    out = []
    for i, p in enumerate(doc["pieces"]):
        rl = [(ln, b) for a, b, _, ln in p["rounds"]]
        for ln, s in p.get("events", []):
            last = max([b for l_, b in rl if l_ < ln], default=None)
            for snt in _sentences(s):
                out.append((i, last, snt))
        for a, b, rest, ln in p["rounds"]:
            m = re.search(r"[\)\]]\s*[.;]\s+([A-Z].*)$", rest)
            if m:
                for snt in _sentences(m.group(1)):
                    out.append((i, b, snt))
    for s in doc.get("assembly", []) + doc.get("other", []):
        for snt in _sentences(s):
            out.append((None, None, snt))
    return out


def _part_regex(pid, name):
    q = re.match(r"(front|hind|back|upper|lower|inner|outer|small|big|large)_(\w+)$", pid)
    if q:
        base = {"leg": r"legs?|paws?|feet|foot", "arm": r"arms?", "ear": r"ears?", "wing": r"wings?",
                "fin": r"fins?"}.get(q.group(2), q.group(2) + "s?")
        return r"\b" + q.group(1) + r"\s?(?:" + base + r")\b"
    words = {pid.replace("_", " "), name.lower()}
    k = _kind(pid)
    for kid, rx in PART_WORDS:
        if kid == k:
            words.add(rx)
    if k == "leg":
        words.add(name.lower().replace("leg", " ?legs?"))
    if k == "muzzle":
        words.add("snout")
    return r"\b(?:" + "|".join(sorted(words, key=len, reverse=True)) + r")\b"


def _rounds_in(s):
    """'between rounds 7-19' / 'between rounds 31 and 32' / 'over round 1' / 'rounds 21-25' -> (a, b)."""
    m = re.search(r"(?:rounds?|rnds?|rows?|r)\s*(\d+)\s*(?:-|to|and|&|through)\s*(?:rounds?|rnds?|r)?\s*(\d+)", s, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(?:round|rnd|row)\s*(\d+)", s, re.I)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None


def _spacing(s):
    m = re.search(r"(?:interspace|space|distance|gap) of (?:about |approx\.? )?(\d+)\s*(?:st|sts|stitch)", s, re.I) or \
        re.search(r"(\d+)\s*(?:st|sts|stitches)\s*(?:apart|between them|in between|between the eyes)", s, re.I) or \
        re.search(r"with (\d+)\s*(?:st|sts|stitches)\s*(?:in )?between", s, re.I)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------------------------------
# geometry helpers on the toy built so far

def _find(inst, normal, rounds=None, near=None):
    """(round, az) on a placed piece whose outward surface normal best matches `normal` (world), optionally
    within a round range; az limited to the toy's left half (the left copy of a pair is described)."""
    N = inst.p.N
    lo, hi = (1.0, float(N)) if rounds is None else (float(max(1, rounds[0])), float(min(N, rounds[1])))
    best = None
    want = _unit(normal)
    for r in np.linspace(lo, hi, max(2, int((hi - lo) * 4) + 1)):
        for az in np.arange(-180, 180, 3.0):
            P, n = inst.point(float(r), float(az))
            sc = float(n @ want)
            if best is None or sc > best[0]:
                best = (sc, float(r), float(az), P, n)
    return best


def _round_at_height(inst, frac_from_top):
    """The round of a vertical piece at a fraction of its height from the top."""
    ys = [inst.point(float(r), 0.0)[0][1] for r in range(1, inst.p.N + 1)]
    top, bot = max(ys), min(ys)
    y = top - frac_from_top * (top - bot)
    return int(np.argmin([abs(v - y) for v in ys])) + 1


def _closing_depth(child_pdef, rim_r):
    """How far (mm) a closed piece must sink into an opening of radius rim_r so the rim meets its surface."""
    t = np.linspace(child_pdef.M * 0.5, child_pdef.M, 200)
    r = np.array([child_pdef.r_t(x) for x in t]); z = np.array([child_pdef.z_t(x) for x in t])
    k = np.where(r <= rim_r)[0]
    if not len(k):
        return 0.0
    i = k[0]
    return float(abs(z[-1] - z[i]))


# ---------------------------------------------------------------------------------------------------------------

def free_assembly(spec, doc, by_name, log):
    parts = {p["id"]: p for p in spec["parts"]}
    pieces = {i: p for i, p in enumerate(doc["pieces"])}
    names = {p["id"]: p.get("name_en", p["id"]) for p in spec["parts"]}
    title = (doc.get("title") or "") + " " + " ".join(doc.get("other", [])[:3])
    ctx = _context(doc)
    _yarn_colours(spec, doc, log)
    kinds = {pid: _kind(pid) for pid in parts}
    # the base: body / head_body / the biggest piece
    root = next((pid for pid in parts if kinds[pid] in ("body", "head_body")), None)
    if root is None:
        root = max(parts, key=lambda q: sum(int(re.sub(r"\D", "", s.split()[-1])) for s in parts[q]["rounds"]))
        if len(parts) > 1:
            log.append(f"assumed: {root} is the base (no body piece)")
    one_piece = len([p for p in parts if kinds[p] in ("head", "body", "head_body")]) == 1
    horizontal = bool(re.search(HORIZONTAL, title, re.I))
    entries = [{"part": root, "axis": [0, 0, -1] if horizontal else [0, 1, 0]}]
    if horizontal:
        log.append("assumed: the toy lies along its rounds (magic ring at the front), as a whale/fish is worked")
    spec["assembly"] = entries
    if root in parts and parts[root].get("end") == "open" and one_piece:
        pass
    placed = {root}
    # head on body
    head = next((pid for pid in parts if kinds[pid] == "head"), None)
    if head and root != head:
        e = {"part": head, "on": root, "at": {"round": "end"}, "attach": "end", "dir": [0, 1, 0]}
        H, B = parts[head], parts[root]
        txt = " ".join(s for _, _, s in ctx if re.search(r"\bhead\b", s, re.I) and re.search(r"\bbody\b", s, re.I))
        if H.get("end") == "open" and B.get("end") == "open":
            log.append("head and body are both open: the two openings are sewn together (head on top)")
        elif B.get("end") == "open":
            # the body's opening sewn onto the closed head: sink so the rim meets the head's surface
            t = build_toy(_clean_spec(spec))
            rim_n = t.inst[root].p.counts[-1]
            rim_r = rim_n * float(spec["yarn"]["stitch_mm"]) / (2 * math.pi)
            from .piece import PieceDef
            hp = PieceDef(head, {k: v for k, v in H.items() if not k.startswith("_")}, {**spec["yarn"]})
            e["sink"] = round(_closing_depth(hp, rim_r), 1)
            rr = _rounds_in(txt)
            log.append(f"head: the body's opening is sewn to the head" + (f" (text: rounds {rr[0]}-{rr[1]} of the head)"
                                                                           if rr else ""))
        else:
            log.append("assumed: the head is sewn onto the top of the body")
        entries.append(e)
        placed.add(head)
    # the other pieces, parents first
    order = sorted([pid for pid in parts if pid not in placed],
                   key=lambda q: ["muzzle", "nose", "beak", "ear", "horn", "arm", "wing", "fin", "leg", "tail",
                                  "inner_ear", "pad"].index(kinds[q]) if kinds[q] in (
                       "muzzle", "nose", "beak", "ear", "horn", "arm", "wing", "fin", "leg", "tail", "inner_ear",
                       "pad") else 99)
    face_host = head or root
    _free_eyes(spec, doc, ctx, face_host, log)
    for pid in order:
        P = parts[pid]
        k = kinds[pid]
        rx = _part_regex(pid, names[pid])
        own = [s for i, _, s in ctx if i is not None and pieces[i].get("id") == pid]
        about = []
        for j, (i, _, s) in enumerate(ctx):
            if re.search(rx, s, re.I) and re.search(r"\b(sew|attach|join|pin|place)", s, re.I):
                about.append(s)
                if j + 1 < len(ctx) and re.match(r"(?:the|they|its|it)\b", ctx[j + 1][2], re.I) and \
                        _rounds_in(ctx[j + 1][2]) and not any(re.search(_part_regex(q, names[q]), ctx[j + 1][2], re.I)
                                                              for q in parts if q != pid and kinds[q] != kinds[pid]):
                    about.append(ctx[j + 1][2])
        sew = [s for s in own if re.search(r"\b(sew|attach)\b", s, re.I) and (
               re.search(r"\b(it|them)\b", s, re.I) or re.search(rx, s, re.I))] + about
        sew = list(dict.fromkeys(sew))
        # parent: named in the sentence, else the designers' usual one
        parent = None
        for s in sew:
            m = re.search(r"\b(?:to|onto|on|over)\s+(?:the\s+)?(?:\w+'s\s+)?([a-z ]+?)(?:\s+of\b|\s+between|\s+below|"
                          r"\s+above|\s+skipping|\s+at|\s+over|[.,(]|$)", s, re.I)
            if m:
                cand = part_id_of(m.group(1))
                cand = next((q for q in placed if q == cand or kinds[q] == _kind(cand)), None)
                if cand and cand != pid:
                    parent = cand
                    break
        if parent is None:
            want = PARENT_OF.get(k, "body")
            parent = next((q for q in placed if kinds[q] == want), None) or (
                face_host if want == "head" else root)
            if k == "pad":
                parent = next((q for q in placed if kinds[q] == "leg" and "hind" in q), None) or parent
            log.append(f"assumed: {pid} goes on the {parent} (the text does not say where)" if not sew else
                       f"{pid}: on the {parent}")
        if parent not in placed:
            log.append(f"{pid}: parent {parent} not placed — skipped")
            continue
        t = build_toy(_clean_spec(spec))
        pinst = _inst(t, parent)
        txt = " ".join(sew)
        rr = None
        for s in sew:                       # a stated range of rounds wins over a single round
            r_ = _rounds_in(s)
            if r_ and (rr is None or (r_[1] > r_[0] and rr[1] == rr[0])):
                rr = r_
        normal, d = DEFAULTS.get(k, ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]))
        normal = list(normal); d = list(d)
        e = {"part": pid, "on": parent}
        mirror = P.get("_make", 1) >= 2 or (re.search(r"\b(ears|arms|legs|wings|horns|pads|eyes)\b", txt, re.I) and
                                             P.get("_make", 1) >= 2)
        if P.get("_make", 1) >= 2:
            e["mirror"] = True
        N = pinst.p.N
        # where along the parent
        rounds = None
        if rr:
            rounds = (min(rr), max(rr))
        m = re.search(r"skipping (?:about |approx\.? )?(\d+) rounds? from the (bottom|top)", txt, re.I)
        if m:
            n_ = int(m.group(1))
            rounds = (n_, n_ + 1) if m.group(2).lower() == "bottom" else (N - n_ - 1, N - n_)
        m = re.search(r"below round (\d+)", txt, re.I)
        if m and not rr:
            rounds = (int(m.group(1)) - 3, int(m.group(1)) - 1)
        if re.search(r"\bhigh on\b", txt, re.I) and not rounds:
            top = _round_at_height(pinst, 0.12)
            rounds = (top, top)
            log.append(f"{pid}: 'high on the {parent}' read as round {top} of the {parent}")
        if re.search(r"\bback of\b", txt, re.I):
            normal = [0.0, -0.3, -1.0]
        # flattened open ears / stuffed-and-flattened limbs
        if k in ("ear",) and P.get("end") == "open" and "flatten" not in P and P.get("stuff", "none") == "none":
            P["flatten"] = 0.85
            log.append(f"assumed: the {pid} is pressed flat (an unstuffed open ear)")
        ear_flat = k == "ear" and P.get("flatten", 0) >= 0.5
        if k == "leg" and P.get("end") == "closed" and rounds and pinst.p.N - rounds[1] <= 6 and \
                re.search(r"front", pid):
            # front legs of a sitting animal: hang down along the chest from just under the head
            normal = [0.25, 0.0, 1.0]; d = [0.06, -1.0, 0.12]
            e["attach"] = "end"
            log.append(f"{pid}: sewn by its closed top end under the head, hanging down the chest (front legs)")
        if k == "leg" and re.search(r"hind|back", pid):
            normal = [0.8, -0.3, 0.45]; d = [0.35, -0.1, 1.0]
        if k == "tail" and P.get("end") == "closed" and len(P["rounds"]) >= 20:
            d = [0.45, -0.45, -1.0]           # a long tail lies on the floor, curving round to one side
            log.append(f"assumed: the long {pid} lies on the floor behind, toward the toy's left")
        if k == "pad":
            e = {"part": pid, "on": parent, "at": {"round": "start"}, "applique": True}
            if P.get("_make", 1) >= 2:
                e["mirror"] = True
            spec["assembly"].append(e)
            placed.add(pid)
            log.append(f"{pid}: sewn flat over round 1 of the {parent} (its sole)")
            continue
        if rounds and rr:
            mid = (rounds[0] + rounds[1]) / 2.0
            best = _find(pinst, normal, (mid, mid))
            if rounds[1] - rounds[0] >= 4:
                log.append(f"{pid}: sewn along rounds {rounds[0]}-{rounds[1]} of the {parent} (centre round {mid:g})")
        else:
            best = _find(pinst, normal, rounds)
        if best is None:
            continue
        sc_, r, az, Pw, nw = best
        if rounds is None:
            log.append(f"assumed: {pid} at round {r:.1f}, {az:.0f} deg from the centre front of the {parent} "
                       f"(designers' usual place)")
        # ears: "about 7 stitches between the lower side of each ear and the eyes"
        m = re.search(r"(?:about |approx\.? )?(\d+) (?:st|sts|stitches) between the (?:lower )?(?:side|edge) of (?:each|the) "
                      r"ear and the eyes", " ".join(own + sew), re.I)
        if m and k == "ear":
            gap = int(m.group(1))
            eye = next((f for f in spec["face"] if f["type"] == "safety_eye"), None)
            if eye:
                n_e = pinst.p.counts[int(eye["at"]["round"]) - 1]
                n_r = pinst.p.counts[int(round(r)) - 1]
                # the ear's seam runs across the rounds, so its lower end is level with the eyes
                az = abs(eye["at"]["az"]) + 360 * ((gap + 0.5) / n_e)
                log.append(f"{pid}: {gap} st beyond the eyes -> {az:.0f} deg")
        e["at"] = {"round": round(r, 2), "az": round(abs(az) if e.get("mirror") else az, 1)}
        if k == "ear":
            # an ear stands out of the surface it is sewn to, leaning up (sides of a big head: mostly outward)
            Pn, nn = pinst.point(e["at"]["round"], e["at"]["az"])
            d = list(_unit(_unit(nn) + np.array([0, 0.9, 0])))
            if P.get("flatten", 0) >= 0.5:
                e["seat"] = "curve"         # the flattened edge is sewn following the head's curve
        if e.get("attach") != "end" or "dir" not in e:
            e["dir"] = [round(x, 3) for x in _unit(d)]
        if k in ("arm", "leg", "tail", "wing") and e.get("attach") != "edge":
            e["avoid"] = True
            if k == "leg" and "front" in pid:
                e.pop("avoid")              # pressed against the chest, as sewn
        if k == "leg" and "front" in pid and e.get("attach") == "end":
            e["dir"] = [0.04, -1.0, 0.22]
        if P.get("end") == "open" and k in ("ear",) and P.get("flatten", 0) < 0.5:
            e["seam"] = "flat"
        spec["assembly"].append(e)
        placed.add(pid)
    _free_face(spec, doc, ctx, face_host, kinds, log)
    _not_modelled(ctx, log)
    _through_both_layers(spec, doc, log)
    _settle(spec, kinds, log)
    for p in spec["parts"]:
        p.pop("_make", None)


def _settle(spec, kinds, log):
    """A sitting toy rests on its body: legs and tails that the defaults point below the body's bottom are lifted
    (their direction tilted up) until the body touches the floor."""
    for it in range(8):
        t = build_toy(_clean_spec(spec))
        parts = t.build(ku=2, kv=2)
        root = parts[0]
        lift = root["V"][:, 1].min()
        if lift <= 2.0:
            return
        low = min(parts[1:], key=lambda m: m["V"][:, 1].min(), default=None)
        if low is None:
            return
        e = next((e for e in spec["assembly"] if low["name"] in (e.get("name"), e["part"], e["part"] + "_L",
                                                                   e["part"] + "_R")), None)
        if e is None or kinds.get(e["part"]) not in ("leg", "tail", "arm") or "dir" not in e:
            return
        d = np.array(e["dir"], float)
        d[1] += 0.15
        e["dir"] = [round(x, 3) for x in _unit(d)]
        if it == 0:
            log.append(f"settled: {e['part']} tilted up so the body sits on the floor (was {lift:.0f} mm above)")


def _through_both_layers(spec, doc, log):
    """A flattened end finished by working across both layers ('3 sc in next st, sc in next st, sl st in next st,
    sc in next st, 3 sc in next st' - a whale's flukes): the edge gets as many stitches as that row makes, i.e. one
    more pinched round with the wider edge."""
    from .reverse import _seq, _prod, _cons, _norm_ops
    groups = [doc.get("lines", [])]
    pairs = [(g, k) for g in groups for k in range(len(g))]
    for g, k in pairs:
        ln = g[k]
        lines = g
        if not re.search(r"both layers", ln, re.I) or not re.search(r"as follows|:\s*$", ln, re.I):
            continue
        row = next((x for x in lines[k:k + 3] if re.search(r"\b(sc|sl st)\b", x, re.I) and "," in x), None)
        part = next((p for p in spec["parts"] if p.get("squash")), None)
        if not row or not part:
            continue
        try:
            ops, _ = _seq(row.rstrip("."), None)
        except Exception:  # noqa: BLE001
            continue
        last = int(re.sub(r"\D", "", part["rounds"][-1].split()[-1]))
        half = last // 2
        edge = half - _cons(ops) + _prod(ops)
        part["rounds"].append(str(2 * edge))
        log.append(f"{part['id']}: the row through both layers ({_cons(ops)} st -> {_prod(ops)} st) widens the flattened "
                   f"edge from {half} to {edge} st: modelled as one more pinched round of {2 * edge}")
        return


def P_width_st(P):
    try:
        return max(int(re.sub(r"\D", "", s.split()[-1])) for s in P["rounds"]) / 2
    except ValueError:
        return 6


def _inst(t, name):
    if name in t.inst:
        return t.inst[name]
    return t.inst[t._side(name, "L")]


def _yarn_colours(spec, doc, log):
    """'main' (pieces without a stated colour) -> the yarn colour named in the materials."""
    if "main" not in spec["colours"]:
        return
    text = " ".join(doc.get("other", []) + doc.get("materials", []))
    m = re.search(r"yarn[^.\n]{0,40}?\b(" + "|".join(sorted(NAMED_HEX, key=len, reverse=True)).replace("_", " ")
                  + r")\b", text, re.I) or re.search(
        r"\b(" + "|".join(sorted(NAMED_HEX, key=len, reverse=True)).replace("_", " ") + r")\b[^.\n]{0,40}?yarn", text, re.I)
    if m:
        c = m.group(1).lower().replace(" ", "_")
        spec["colours"]["main"] = NAMED_HEX[c]
        log.append(f"yarn colour: {c} (from the materials)")


def _free_eyes(spec, doc, ctx, host, log):
    t = build_toy(_clean_spec(spec))
    size = None
    for ln in doc.get("other", []) + doc.get("materials", []) + [s for _, _, s in ctx]:
        if re.search(r"\beyes?\b", ln, re.I):
            m = re.search(r"(\d+(?:\.\d+)?)\s*mm", ln)
            if m:
                size = float(m.group(1))
                break
    if size is None:
        size = 8.0
        log.append("assumed: safety eyes 8 mm (size not stated)")
    done_eyes = False
    for i, last, s in ctx:
        if done_eyes or not re.search(r"\beyes?\b", s, re.I) or not re.search(r"\b(insert|fasten|place|attach|put)", s, re.I):
            continue
        hid = doc["pieces"][i]["id"] if i is not None else host
        if hid not in t.inst and hid + "_L" not in t.inst:
            hid = host
        inst = _inst(t, hid)
        rr = _rounds_in(s)
        sp = _spacing(s)
        m = re.search(r"(\d+)\s*rounds?\s*down", s, re.I)
        if m and last:
            r = float(last - int(m.group(1)))
            log.append(f"eyes: {m.group(1)} rounds down from round {last} -> round {r:.0f}")
        elif rr:
            r = (rr[0] + rr[1]) / 2.0
        else:
            r = None
        if r is None:
            continue
        n = inst.p.counts[int(r) - 1]
        if sp is not None:
            az = 360.0 * (sp + 1) / 2 / n
        else:
            # "the other side": the sides of the piece
            best = _find(inst, [1.0, -0.2, 0.15], (r, r))
            az = abs(best[2])
            log.append(f"assumed: eyes on the sides ({az:.0f} deg) — spacing not stated")
        spec["face"].append({"type": "safety_eye", "on": hid, "at": {"round": r, "az": round(az, 1)},
                             "size_mm": size, "mirror": True})
        done_eyes = True


def _free_face(spec, doc, ctx, host, kinds, log):
    yarn = spec["yarn"]
    t = build_toy(_clean_spec(spec))
    # felt / fabric pieces and embroidery the text asks for
    head = _inst(t, host)
    hw = max(head.p.counts) * float(yarn["stitch_mm"]) / math.pi
    eye = next((f for f in spec["face"] if f["type"] == "safety_eye"), None)
    for i, last, s in ctx:
        if re.search(r"felt", s, re.I) and re.search(r"muzzle|snout|nose", s, re.I) and re.search(r"cut", s, re.I):
            col = next((c for c in NAMED_HEX if re.search(r"\b" + c.replace("_", " ") + r"\b", s, re.I)), "white")
            spec["colours"].setdefault(col, NAMED_HEX[col])
            r = (eye["at"]["round"] + 3.5) if eye else head.p.N * 0.6
            spec["face"].append({"type": "felt", "on": host, "at": {"round": round(r, 1), "az": 0},
                                 "size_mm": [round(0.5 * hw, 1), round(0.37 * hw, 1)], "colour": col})
            spec["face"].append({"type": "satin", "on": host, "at": {"round": round(r - 1.0, 1), "az": 0},
                                 "width_mm": round(0.11 * hw, 1), "height_mm": round(0.06 * hw, 1),
                                 "shape": "triangle", "colour": "black"})
            spec["face"].append({"type": "stitch", "on": host, "points": [[round(r - 0.8, 1), 0], [round(r + 0.9, 1), 0]],
                                 "colour": "black", "thickness_mm": 1.0})
            log.append("felt muzzle: an oval of felt (size not stated: assumed 50 % x 37 % of the head's width) with an "
                       "embroidered nose and mouth line")
        if re.search(r"patch(?:es)? of fabric|fabric", s, re.I) and re.search(r"\bears?\b", s, re.I):
            ear = next((q for q in kinds if kinds[q] == "ear"), None)
            if ear:
                ei = _inst(t, ear)
                spec["colours"].setdefault("fabric", "#efe2cf")
                spec["face"].append({"type": "felt", "on": ear, "at": {"round": round(ei.p.N * 0.62, 1), "az": 0},
                                     "size_mm": [round(0.5 * max(ei.p.counts) * float(yarn["stitch_mm"]) / math.pi, 1),
                                                 round(0.62 * max(ei.p.counts) * float(yarn["stitch_mm"]) / math.pi, 1)],
                                     "colour": "fabric", "mirror": True})
                log.append("ear patches: cotton fabric modelled as a felt oval in a cream colour")
        if re.search(r"embroider", s, re.I) and re.search(r"\bnose\b", s, re.I) and not re.search(r"felt", s, re.I):
            sn = next((q for q in kinds if kinds[q] == "muzzle"), None)
            if sn:
                rr = _rounds_in(s)
                rc = (rr[0] + rr[1]) / 2 if rr else 2
                si = _inst(t, sn)
                w = max(si.p.counts) * float(yarn["stitch_mm"]) / math.pi
                spec["face"].append({"type": "satin", "on": sn, "at": {"round": rc, "az": 0},
                                     "width_mm": round(0.42 * w, 1), "height_mm": round(0.3 * w, 1),
                                     "shape": "triangle", "colour": "black"})
                spec["colours"].setdefault("black", NAMED_HEX["black"])
                log.append(f"nose: satin-stitch triangle on the {sn} around round {rc:g} (size not stated: assumed)")
        if re.search(r"embroider", s, re.I) and re.search(r"paw pads?", s, re.I):
            for q in kinds:
                if kinds[q] == "leg":
                    li = _inst(t, q)
                    spec["colours"].setdefault("beige", NAMED_HEX["beige"])
                    if "front" in q:
                        spec["face"].append({"type": "satin", "on": q, "at": {"round": 2.5, "az": 180},
                                             "width_mm": 6, "height_mm": 4, "shape": "oval", "colour": "beige",
                                             "mirror": True})
                    for daz in (-45, 0, 45):
                        if "front" in q:
                            continue
                        spec["face"].append({"type": "satin", "on": q, "at": {"round": 3.0, "az": daz},
                                             "width_mm": 3.2, "height_mm": 4.2, "shape": "oval", "colour": "beige",
                                             "mirror": True})
            log.append("paw pads: embroidered ovals in beige (sizes not stated: assumed)")
            break


def _not_modelled(ctx, log):
    seen = set()
    for _, _, s in ctx:
        for w in ("ribbon", "button", "bow", "heart"):
            if re.search(r"\b" + w + r"s?\b", s, re.I) and w not in seen:
                seen.add(w)
                log.append(f"not modelled: {w} ({_strip_md(s)[:90]})")
