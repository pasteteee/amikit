"""The harvested designer library (learn-only) as statistics the engine and its authors can use.

~39,000 in-the-round parts with valid stitch arithmetic (pattern-harvest/v3 + v2), each described relative to the
head of its own pattern:
  max_rel   widest round / the head's widest round          (how big the part is)
  len_rel   rounds / (head max / pi)                         (how long, in head diameters, if round height ~ width)
  open_rel  last round / widest round                        (closed pole ~ small, open seam ~ 1)
  start     first round count (mr 4 = pointed tip, mr 6 = round, chain oval = big start)
  plateau   rounds at >= 95% of the widest                   (how much straight "volume" it has)
  maxpos    where the widest round sits (0 start .. 1 end)
  inc       typical increase per increase round
  kind      a construction type from the counts (ears: long / floppy / round / pointed ...; limbs: tube / paw / ...)

Use:
  stats(role, subject)              percentiles of every feature, with how many designers stand behind them
  audit(spec)                       every part of a toy spec checked against designers of that animal (and of all
                                    animals): size, length, volume, opening; with a concrete suggestion
  suggest(role, subject, head_max)  our own designer-style round steps from the consensus of >= 3 designers
                                    (never a copy of one designer's counts)
  python -m amikit.library audit amikit/toys/teddy.json
  python -m amikit.library stats ear bunny
  python -m amikit.library suggest ear bunny 48 [kind]
"""
from __future__ import annotations
import json
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "pattern-harvest/v3/library.jsonl", ROOT / "pattern-harvest/v3/llm_extract/validated_parts.jsonl",
           ROOT / "pattern-harvest/v3/clean/parts.jsonl", ROOT / "pattern-harvest/v2/clean/parts.jsonl"]
CACHE = ROOT / "out/amikit/_library_cache.json"

# subject words -> group used for statistics (fallback: all animals)
GROUPS = {
    "bear": "bear teddy koala", "panda": "panda", "bunny": "bunny rabbit hare", "cat": "cat kitty kitten",
    "dog": "dog puppy beagle corgi poodle pug husky", "mouse": "mouse rat hamster", "fox": "fox wolf raccoon",
    "cow": "cow bull ox calf highland", "pig": "pig piglet", "unicorn": "unicorn pony horse", "duck": "duck chick hen chicken",
    "bird": "bird owl crow penguin parrot", "penguin": "penguin", "hippo": "hippo hippopotamus", "elephant": "elephant",
    "sloth": "sloth", "turtle": "turtle tortoise", "frog": "frog", "dinosaur": "dinosaur dragon", "lion": "lion tiger",
    "sheep": "sheep lamb", "monkey": "monkey", "deer": "deer reindeer giraffe", "whale": "whale shark fish dolphin",
    "food": "food potato fruit vegetable avocado", "capybara": "capybara guinea",
}
_WORDS = {w: g for g, ws in GROUPS.items() for w in ws.split()}
RELATED = {"teddy": "bear", "red_panda": "fox", "capybara": "bear", "black_cat": "cat", "tabby": "cat",
           "highland_cow": "cow", "sage_bunny": "bunny", "beagle": "dog", "corgi": "dog", "crow": "bird",
           "shark": "whale", "potato": "food"}

ROLE_OF_ID = [("inner_ear", None), ("ear", "ear"), ("muzzle", "muzzle"), ("snout", "muzzle"), ("nose", "muzzle"),
              ("bill", "beak"), ("beak", "beak"), ("head", "head"), ("body", "body"), ("arm", "arm"),
              ("hand", "arm"), ("paw", "arm"), ("flipper", "wing"), ("wing", "wing"), ("leg", "leg"),
              ("foot", "leg"), ("feet", "leg"), ("thigh", None), ("tail", "tail"), ("horn", "horn"),
              ("fin", "fin"), ("dorsal", "fin"), ("shell", "shell")]


def group_of(text):
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in _WORDS:
            return _WORDS[w]
        if w.endswith("s") and w[:-1] in _WORDS:
            return _WORDS[w[:-1]]
    return None


def role_of_id(pid):
    """Role of a part from its id words (ear, inner_ear, front_leg ...): whole words, so 'heart' is not an ear."""
    words = re.split(r"[^a-z]+", pid.lower())
    joined = "_".join(w for w in words if w)
    for k, r in ROLE_OF_ID:
        if "_" in k:
            if k in joined:
                return r
        elif k in words or (k + "s") in words:
            return r
    return None


def _good(rec):
    c = rec.get("counts") or []
    if any(not isinstance(x, int) for x in c):
        return False
    if len(c) < 2 or max(c) > 150 or min(c) < 3:
        return False
    ar = rec.get("arith") or []
    if ar:
        okf = sum(a == "ok" for a in ar) / len(ar)
        if okf < 0.8:
            return False
    elif rec.get("confidence") != "high":
        return False
    # valid crochet steps: never more than doubling or halving
    return all(b <= 2 * a + 1 and b * 2 + 1 >= a for a, b in zip(c, c[1:]))


def describe(counts, H):
    c = [int(x) for x in counts]
    Mx = max(c); n = len(c)
    incs = [b - a for a, b in zip(c, c[1:]) if b > a]
    d = dict(n=n, max=Mx, start=c[0], max_rel=Mx / H, len_rel=n / (H / math.pi), open_rel=c[-1] / Mx,
             maxpos=c.index(Mx) / max(1, n - 1), inc=int(np.median(incs)) if incs else 0,
             plateau=sum(1 for x in c if x >= 0.95 * Mx))
    return d


def kind_of(role, d):
    """Construction type from the counts (how designers make that kind of part)."""
    L, Mx, Op, pos, st = d["len_rel"], d["max_rel"], d["open_rel"], d["maxpos"], d["start"]
    if role == "ear":
        if L >= 0.9:
            return "long"                        # bunny ears
        if L >= 0.55 and Op <= 0.8:
            return "floppy"                      # widest in the middle, narrower seam: hangs (dogs)
        if st <= 4 or (Op >= 0.9 and pos >= 0.8 and d["inc"] <= 3):
            return "pointed"                     # grows slowly to the seam: a cone (cats, foxes)
        if L <= 0.5 and Op >= 0.85:
            return "round"                       # a short cup (bears, mice, pandas)
        return "other"
    if role in ("arm", "leg"):
        if pos <= 0.35 and Op <= 0.85:
            return "paw"                         # a wider paw/foot at the start, narrower to the seam
        if L >= 0.9:
            return "long_tube"
        return "tube"
    if role == "tail":
        return "long" if L >= 0.6 else "short"
    return "any"


@lru_cache(maxsize=1)
def parts():
    """All usable parts with features (cached to out/amikit/_library_cache.json)."""
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf8"))
    raw = []
    seen = set()
    for src in SOURCES:
        if not src.exists():
            continue
        for line in open(src, encoding="utf8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _good(r):
                continue
            key = (r.get("pattern_id"), tuple(r["counts"]))
            if key in seen:
                continue
            seen.add(key)
            raw.append(r)
    heads = {}
    for r in raw:
        if r.get("role") in ("head", "head_body"):
            heads[r["pattern_id"]] = max(heads.get(r["pattern_id"], 0), max(r["counts"]))
    out = []
    for r in raw:
        H = heads.get(r["pattern_id"])
        role = r.get("role") or "unknown"
        if not H or H < 18:
            continue
        g = group_of(r.get("subject")) or group_of(r.get("title")) or (r.get("subject") or "other")
        d = describe(r["counts"], H)
        d.update(role=role, group=g, designer=r.get("designer") or r.get("site") or "?", head=H,
                 counts=r["counts"], pid=r["pattern_id"], stuffed=r.get("stuffed"))
        d["kind"] = kind_of(role, d)
        out.append(d)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out), encoding="utf8")
    return out


def _select(role, group=None, kind=None, max_per_designer=2):
    per = defaultdict(int)
    res = []
    for d in parts():
        if d["role"] != role or (group and d["group"] != group) or (kind and d["kind"] != kind):
            continue
        if per[d["designer"]] >= max_per_designer:
            continue
        per[d["designer"]] += 1
        res.append(d)
    return res


FEATS = ("max_rel", "len_rel", "open_rel", "plateau", "start", "maxpos", "n")


def stats(role, subject=None, kind=None):
    g = RELATED.get(subject, group_of(subject) or subject) if subject else None
    ds = _select(role, g, kind)
    scope = g
    if len(ds) < 8:
        ds, scope = _select(role, None, kind), "all animals"
    if not ds:
        return None
    q = lambda k: [round(float(x), 2) for x in np.percentile([d[k] for d in ds], [15, 50, 85])]
    kinds = defaultdict(int)
    for d in ds:
        kinds[d["kind"]] += 1
    return {"role": role, "scope": scope, "N": len(ds), "designers": len({d["designer"] for d in ds}),
            **{k: q(k) for k in FEATS}, "kinds": dict(sorted(kinds.items(), key=lambda x: -x[1]))}


@lru_cache(maxsize=1)
def _all_count_lists():
    return {tuple(d["counts"]) for d in parts()}


def _profile(c, k=25):
    c = np.asarray(c, float)
    x = (np.arange(len(c)) + 0.5) / len(c)
    return np.interp(np.linspace(0, 1, k), x, c / c.max())


def synthesize(profile, n, M, start=6, step=None):
    """Rounds of n stitch counts following a normalised shape profile at widest count M, written the way
    designers write them: growth in +start steps, plateaus kept flat, gentle decreases, every step valid."""
    x = (np.arange(n) + 0.5) / n
    want = np.interp(x, np.linspace(0, 1, len(profile)), profile) * M
    step = step or start
    out = [start]
    for i in range(1, n):
        prev = out[-1]
        w = want[i]
        if w > prev + 0.5 * step:                        # growing: whole designer steps (+6, or +2/+3 for cones)
            nxt = min(prev + step * max(1, int(round((w - prev) / step))), 2 * prev)
        elif abs(w - prev) <= max(1.5, 0.07 * M):        # plateau: keep the count
            nxt = prev
        elif w < prev:                                   # shrinking: designers take off 1-6 per round
            nxt = max(int(round(w)), (prev + 1) // 2, 3)
            nxt = max(nxt, prev - max(start, 6))
        else:
            nxt = prev + max(1, int(round(w - prev)))
        out.append(int(nxt))
    return out


def counts_to_steps(c):
    """Stitch counts -> DSL steps (mr N, then one count per round, merged into runs)."""
    steps = [f"mr {c[0]}"]
    i = 1
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[i]:
            j += 1
        if c[i] == c[i - 1]:
            steps.append(f"even {j - i + 1}")
        else:
            steps.append(str(c[i]))
            if j > i:
                steps.append(f"even {j - i}")
        i = j + 1
    # merge regular +k runs into 'inc k to M'
    merged = [steps[0]]
    prev = c[0]
    k = 1
    out_counts = []
    return _merge_runs(steps)


def _merge_runs(steps):
    res = [steps[0]]
    cur = int(steps[0].split()[1])
    run = None
    for st in steps[1:]:
        if st.isdigit():
            n = int(st)
            d = n - cur
            if run and run[0] == d and d != 0:
                run[2] = n
            else:
                if run:
                    res.append(_run_text(run))
                run = [d, cur, n]
            cur = n
        else:
            if run:
                res.append(_run_text(run)); run = None
            res.append(st)
    if run:
        res.append(_run_text(run))
    return res


def _run_text(run):
    d, a, b = run
    if (b - a) == d:
        return str(b)
    return f"{'inc' if d > 0 else 'dec'} {abs(d)} to {b}"


# how designers' ears of each kind are finished, as engine constructions (SPEC.md "Ears"; amikit/presets.py)
EAR_CONSTRUCTION = {
    "round":   {"part": {"dish": 0.5}, "entry": {"attach": "edge", "face": [0.1, 0.25, 1.0], "sink": 5.0},
                "how": "a cup pressed along its axis: ring = centre of a concave front, thick rolled rim; stood on "
                       "the rim and sewn by its lower arc"},
    "pointed": {"part": {"spoon": 0.35}, "entry": {},
                "how": "a cone pressed partly side to side: two layers, rounded folds, slightly hollow front; sewn "
                       "by the opening in a curve, hollow side forward"},
    "long":    {"part": {"spoon": 0.3}, "entry": {},
                "how": "a long tube pressed to a shallow spoon, sewn by the opening"},
    "floppy":  {"part": {"spoon": 0.45}, "entry": {},
                "how": "a spoon hanging to the side/down, hollow side forward"},
}


def suggest(role, subject, head_max, kind=None, closed=None, n_rounds=None):
    """Designer-consensus rounds for a part of this animal at our head size: our own counts, synthesised from
    the median shape of >= 3 designers' parts of that kind (never one designer's list)."""
    g = RELATED.get(subject, group_of(subject) or subject) if subject else None
    ds = _select(role, g, kind)
    scope = g
    if len(ds) < 8:
        ds, scope = _select(role, None, kind), "all animals"
    if len(ds) < 3:
        return None
    prof = np.median(np.array([_profile(d["counts"]) for d in ds]), axis=0)
    prof = prof / prof.max()
    mx_rel = float(np.median([d["max_rel"] for d in ds]))
    len_rel = float(np.median([d["len_rel"] for d in ds]))
    start = int(np.median([d["start"] for d in ds]))
    start = 4 if start <= 4 else (5 if start == 5 else 6)
    step = max(2, int(np.median([d["inc"] for d in ds if d["inc"] > 0] or [6])))
    M = max(start + step, int(round(mx_rel * head_max / step)) * step)
    n = n_rounds or max(3, int(round(len_rel * head_max / math.pi)))
    # licence guard: never hand out a list identical to any designer's part of 6+ rounds (whole library, not only
    # the consensus sample); vary the length of the straight run until it is our own
    lib = _all_count_lists()
    c = None
    for dn in (0, 1, -1, 2, -2, 3, -3):
        cand = synthesize(prof, max(3, n + dn), M, start, step)
        if len(cand) < 6 or tuple(cand) not in lib:
            c = cand
            break
    copied = c is None
    if copied:
        c = synthesize(prof, n, M, start, step)
    from .rounds import expand
    steps = counts_to_steps(c)
    try:
        expand(steps)
    except Exception:  # noqa: BLE001 - fall back to plain counts
        steps = [f"mr {c[0]}"] + [str(x) for x in c[1:]]
    if copied:
        return None
    res = {"steps": steps, "counts": c, "copied_guard": copied, "kind": kind,
            "from": {"scope": scope, "N": len(ds), "designers": len({d["designer"] for d in ds}),
                     "max_rel": round(mx_rel, 2), "len_rel": round(len_rel, 2)}}
    if role == "ear" and kind in EAR_CONSTRUCTION:
        res["construction"] = EAR_CONSTRUCTION[kind]
        if kind == "round":
            # a dish rolls its rim from the rounds after the widest: one round less at full width, then two
            # decrease rounds make the back layer (the designers' cup + a closing edge)
            M = max(c)
            res["counts"] = c[:c.index(M) + 1] + [M, M * 7 // 8, M * 3 // 4]
            res["steps"] = counts_to_steps(res["counts"])
    return res


def _spec_features(part, H):
    from .rounds import counts
    c = counts(part["rounds"])
    return describe(c, H), c


def audit(spec, subject=None):
    """Check each part of a toy against designers of the same animal; return findings with suggestions."""
    from .rounds import counts
    subject = subject or spec.get("subject") or spec.get("name")
    heads = [p for p in spec["parts"] if "rounds" in p and role_of_id(p["id"]) == "head"]
    one_piece = not heads
    if heads:
        H = max(counts(heads[0]["rounds"]))
    else:  # one-piece toys (head and body together): the body is the head's reference
        bodies = [p for p in spec["parts"] if "rounds" in p and role_of_id(p["id"]) == "body"]
        H = max(counts(bodies[0]["rounds"])) if bodies else 42
    out = []
    for p in spec["parts"]:
        if "rounds" not in p:
            continue
        role = role_of_id(p["id"])
        if role is None or (one_piece and role == "body") or role == "head":
            continue
        d, c = _spec_features(p, H)
        d["kind"] = kind_of(role, d)
        # compare with designers' parts of the same construction kind (a paw arm with paw arms, a pointed ear
        # with pointed ears); fall back to the whole role when that kind is rare
        st = stats(role, subject, d["kind"]) if d["kind"] != "any" else None
        if not st or st["N"] < 15:
            st = stats(role, subject)
        if not st:
            continue
        issues = []
        for k, label, fmt in (("max_rel", "width (widest round / head's widest)", "{:.2f}"),
                              ("len_rel", "length (rounds per head diameter)", "{:.2f}"),
                              ("plateau", "volume (rounds at full width)", "{:.0f}")):
            lo, med, hi = st[k]
            v = d[k]
            if v < lo * 0.92 or v > hi * 1.08:
                issues.append(f"{label}: ours {fmt.format(v)}, designers {fmt.format(lo)}-{fmt.format(hi)} "
                              f"(median {fmt.format(med)})")
        want_kind = d["kind"] if (role == "ear" or d["kind"] in ("paw",)) else (
            max(st["kinds"], key=st["kinds"].get) if st["kinds"] else None)
        sug = suggest(role, subject, H, want_kind)
        out.append({"part": p["id"], "role": role, "kind": d["kind"], "ours": c, "issues": issues,
                    "library": {k: st[k] for k in ("max_rel", "len_rel", "plateau", "open_rel")} | {
                        "N": st["N"], "designers": st["designers"], "scope": st["scope"], "kinds": st["kinds"]},
                    "suggest": (sug["steps"] + [f"(kind {want_kind}, {sug['from']['N']} parts)"]) if sug else None})
    return {"subject": subject, "head_max": H, "one_piece": one_piece, "parts": out}


def _print_audit(a):
    print(f"subject {a['subject']}  head max {a['head_max']:.0f}{'  (one-piece toy)' if a['one_piece'] else ''}")
    for r in a["parts"]:
        lib = r["library"]
        flag = "OK " if not r["issues"] else "!! "
        print(f"{flag}{r['part']:<12} {r['role']:<6} kind={r['kind']:<9} ours {r['ours']}")
        print(f"      library: {lib['N']} parts / {lib['designers']} designers ({lib['scope']}); kinds {lib['kinds']}")
        for i in r["issues"]:
            print("      - " + i)
        if r["issues"]:
            print(f"      consensus rounds: {r['suggest']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "audit":
        sp = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
        _print_audit(audit(sp, sys.argv[3] if len(sys.argv) > 3 else None))
    elif cmd == "stats":
        print(json.dumps(stats(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
                               sys.argv[4] if len(sys.argv) > 4 else None), indent=1))
    elif cmd == "suggest":
        print(json.dumps(suggest(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else None),
                         indent=1))
