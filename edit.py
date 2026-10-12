"""Edits of a toy spec: the one interface shared by an LLM (text request -> ops) and the visual editor.

An edit is a list of operations applied to a copy of the spec; the result is validated by building the toy
(stitch arithmetic, placement). Nothing is applied when any op or the validation fails.

Paths are dotted, and list items are addressed by their id, not their index:
    parts.<part id>.<field>            e.g. parts.head.rounds, parts.ear.flatten, parts.body.colours
    assembly.<entry name>.<field>      entry name = "name" if given, else its part id: assembly.arm.dir
    face.<item id>.<field>             face items get ids ("eye", "nose", "mouth", ...); index works too
    colours.<colour name>              colours.brown = "#9c5428"
    yarn.<field>                       yarn.stitch_mm
    title, name

Operations:
    {"op": "set",    "path": "assembly.ear.at.az", "value": 70}
    {"op": "add",    "path": "parts" | "assembly" | "face", "value": {...}}       append an item
    {"op": "remove", "path": "parts.tail"}            removes the part and every entry/face item using it
    {"op": "scale_part", "part": "head", "factor": 1.15}   re-grows a piece to ~factor x width (keeps its shape:
                                                           increase/decrease runs and even rounds rescaled)
    {"op": "recolour", "from": "brown", "to": "#7a7a7a"}

Customisation ops (a pet's look in words; amikit/custom.py has the full reference):
    {"op": "colour", "target": "all"|"coat"|"ears"|"tail"|"legs"|"muzzle"|"nose"|"eyes"|<part>, "colour": "#6b4a2f"}
    {"op": "spot", "near": "eye_L", "place": "under", "colour": "#3a2a20"}       a spot under the toy's left eye
    {"op": "spot", "on": "body", "where": "back", "colour": "black", "size_mm": [30, 24]}
    {"op": "marking", "kind": "blaze"|"mask"|"eye_patch"|"eyebrows"|"tan_points"|"socks"|"tail_tip"|"ear_tips"|
                      "chest"|"belly"|"saddle"|"spots"|"stripes"|"muzzle"|"inner_ears"|"half_face"|"collar", "colour": C}
    {"op": "ears", "kind": "pointed"|"round"|"floppy"|"long"|"folded", "size": 1.1, "inner": "pink"}
    {"op": "tail", "kind": "curled"|"straight"|"up"|"stub"|"long", "length": 1.2}
    {"op": "shape", "target": "legs", "length": 0.8}          size / length / width factors
    {"op": "eyes", "size": 1.2, "colour": "amber"}   {"op": "nose", "colour": "pink"}
"""
from __future__ import annotations
import copy
import json
import re


class EditError(ValueError):
    pass


# customisation ops in the words of a pet's owner (amikit/custom.py): colour, spot, marking, ears, tail, shape,
# eyes, nose
CUSTOM_OPS = ("colour", "spot", "marking", "ears", "tail", "shape", "eyes", "nose")


def _ensure_ids(spec):
    seen = set()
    for i, f in enumerate(spec.get("face", [])):
        if "id" not in f:
            base = {"safety_eye": "eye", "eye": "eye", "safety_nose": "nose", "satin": "nose",
                    "stitch": "line", "blush": "blush", "felt": "felt"}.get(f.get("type"), f.get("type", "item"))
            k, j = base, 2
            while k in seen:
                k = f"{base}{j}"; j += 1
            f["id"] = k
        seen.add(f["id"])
    return spec


def _find(lst, key, attrs):
    for a in attrs:                      # attribute order is priority: an entry's name before its part id
        for i, it in enumerate(lst):
            if str(it.get(a)) == key:
                return i
    if key.isdigit() and int(key) < len(lst):
        return int(key)
    raise EditError(f"no item {key!r}")


def _resolve(spec, path):
    """Return (container, key) for a dotted path."""
    parts = path.split(".")
    head = parts[0]
    if head in ("parts", "assembly", "face"):
        lst = spec.setdefault(head, [])
        if len(parts) == 1:
            return spec, head
        attrs = {"parts": ["id"], "assembly": ["name", "part"], "face": ["id"]}[head]
        obj = lst[_find(lst, parts[1], attrs)]
        if len(parts) == 2:
            return lst, _find(lst, parts[1], attrs)
        rest = parts[2:]
    else:
        obj, rest = spec, parts
    for k in rest[:-1]:
        if isinstance(obj, list):
            obj = obj[int(k)]
        else:
            obj = obj.setdefault(k, {})
    last = rest[-1]
    if isinstance(obj, list):
        last = int(last)
    return obj, last


def scale_steps(steps, factor):
    """Scale a piece to ~factor x its width and length while keeping its shape (the same profile of increases,
    straight rounds and decreases), written again as designer-style steps."""
    import numpy as np
    from .rounds import expand
    from .library import synthesize, counts_to_steps
    rounds = expand(steps)
    c = [r.n for r in rounds]
    if rounds[0].start == "ch":
        # chain ovals: scale the chain and keep +inc steps around the turns
        ch = max(3, int(round(rounds[0].chain * factor)))
        n0 = 2 * ch
        prof = np.array(c, float) / max(c)
        n_new = max(2, int(round(len(c) * factor)))
        M = max(n0, int(round(max(c) * factor / 6.0)) * 6 + (n0 % 6))
        new = synthesize(prof, n_new, M, start=n0, step=6)
        out = [f"ch {ch}"] + counts_to_steps(new)[1:]
    else:
        start = c[0]
        incs = [b - a for a, b in zip(c, c[1:]) if b > a]
        step = int(np.median(incs)) if incs else start
        prof = np.array(c, float) / max(c)
        n_new = max(2, int(round(len(c) * factor)))
        M = max(start + step, int(round(max(c) * factor / step)) * step)
        new = synthesize(prof, n_new, M, start=start, step=max(2, step))
        out = counts_to_steps(new)
    expand(out)
    return out


def apply(spec, ops, validate=True):
    """Apply ops to a copy of spec. Returns (new_spec, report). Raises EditError on failure."""
    new = _ensure_ids(copy.deepcopy(spec))
    log = []
    for op in ops:
        kind = op.get("op")
        try:
            if kind == "set":
                obj, key = _resolve(new, op["path"])
                obj[key] = op["value"]
            elif kind == "add":
                obj, key = _resolve(new, op["path"])
                obj[key].append(op["value"])
                _ensure_ids(new)
            elif kind == "remove":
                path = op["path"]
                if path.startswith("parts.") and path.count(".") == 1:
                    pid = path.split(".")[1]
                    if pid not in {p["id"] for p in new["parts"]}:
                        raise EditError(f"no part {pid!r}")
                    # remove the part, every entry using it, and (transitively) everything sewn onto those
                    gone_parts = {pid}
                    new["parts"] = [p for p in new["parts"] if p["id"] != pid]
                    names = set()
                    changed = True
                    while changed:
                        changed = False
                        for e in list(new.get("assembly", [])):
                            nm = e.get("name", e["part"])
                            if e["part"] in gone_parts or e.get("on") in names | gone_parts:
                                if nm not in names:
                                    names.add(nm); changed = True
                        new["assembly"] = [e for e in new.get("assembly", [])
                                           if e.get("name", e["part"]) not in names]
                    new["face"] = [f for f in new.get("face", []) if f.get("on") not in names | gone_parts]
                else:
                    obj, key = _resolve(new, path)
                    del obj[key]
            elif kind == "scale_part":
                from .rounds import counts
                obj, key = _resolve(new, f"parts.{op['part']}.rounds")
                n_old = len(counts(obj[key]))
                obj[key] = scale_steps(obj[key], float(op["factor"]))
                k = len(counts(obj[key])) / n_old
                part = obj
                for z in part.get("colours", []):                     # colour zones follow the rounds
                    if "rounds" in z:
                        z["rounds"] = [max(1, int(round(r * k))) for r in z["rounds"]]
                    if "shape" in z:
                        z["shape"] = [[round(r * k, 1), a] for r, a in z["shape"]]
                    if "ellipse" in z:
                        (rc, ac), (rr, ar) = z["ellipse"]["center"], z["ellipse"]["radii"]
                        z["ellipse"] = {"center": [rc * k, ac], "radii": [rr * k, ar]}
                if isinstance(part.get("bend"), dict) and "at" in part["bend"]:
                    part["bend"]["at"] = round(part["bend"]["at"] * k, 1)
                # things sewn to this piece keep their relative place (rounds renumbered)
                names = {e.get("name", e["part"]) for e in new.get("assembly", []) if e["part"] == op["part"]}
                for it in new.get("assembly", []) + new.get("face", []):
                    if it.get("on") in names | {op["part"]}:
                        at = it.get("at", {})
                        if isinstance(at.get("round"), (int, float)):
                            at["round"] = round(at["round"] * k * 2) / 2
                        if "points" in it:
                            it["points"] = [[round(r * k, 2), a] for r, a in it["points"]]
            elif kind in CUSTOM_OPS:
                from . import custom
                try:
                    custom.apply_op(new, op)
                except ValueError as ex:
                    raise EditError(f"{kind}: {ex}") from ex
            elif kind == "recolour":
                src = op["from"]
                if src in new.get("colours", {}):
                    new["colours"][src] = op["to"]
                else:
                    raise EditError(f"no colour {src!r}")
            else:
                raise EditError(f"unknown op {kind!r}")
            log.append(f"ok {json.dumps(op, ensure_ascii=False)}")
        except EditError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as ex:
            raise EditError(f"{kind} {op}: {ex}") from ex
    if new.get("_pending_prune"):
        from . import custom
        try:
            custom.prune(new)
        except ValueError as ex:
            raise EditError(str(ex)) from ex
    report = {"log": log, "flags": []}
    if validate:
        from .toy import Toy
        try:
            toy = Toy(new)
        except Exception as ex:  # noqa: BLE001 - surface any build failure to the caller
            raise EditError(f"the edited toy does not build: {ex}") from ex
        report["flags"] = toy.flags
    return new, report


def summary(spec):
    """Compact human/LLM-readable summary of a spec (what can be addressed by edits)."""
    from .rounds import counts
    from .presets import expand as expand_presets
    spec = expand_presets(spec)
    lines = [f"toy {spec.get('name')}: {spec.get('title', '')}"]
    lines.append("colours: " + ", ".join(f"{k}={v}" for k, v in spec.get("colours", {}).items()))
    for p in spec["parts"]:
        if "rounds" not in p:
            kind = "chain cord" if "cord" in p else ("worked flat in rows" if "flat" in p else "no rounds")
            lines.append(f"part {p['id']}: {kind} {p.get('cord') or p.get('flat') or ''}, colour={p.get('colour')}")
            continue
        c = counts(p["rounds"])
        lines.append(f"part {p['id']}: {len(c)} rounds, max {max(c)} st, end={p.get('end', 'closed')},"
                     f" stuff={p.get('stuff', 'firm' if p.get('end', 'closed') == 'closed' else 'none')},"
                     f" colour={p.get('colour')}" + (f", flatten={p['flatten']}" if p.get("flatten") else ""))
    for e in spec.get("assembly", []):
        lines.append(f"assembly {e.get('name', e['part'])}: on={e.get('on', '-')} at={e.get('at')} dir={e.get('dir')}"
                     + (" mirror" if e.get("mirror") else "") + (" applique" if e.get("applique") else ""))
    for f in _ensure_ids(copy.deepcopy(spec)).get("face", []):
        lines.append(f"face {f['id']}: {f['type']} on={f['on']} at={f.get('at')}"
                     + (f" points={f['points']}" if "points" in f else ""))
    return "\n".join(lines)
