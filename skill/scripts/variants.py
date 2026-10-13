"""The design loop in one command: a base spec + named variants (edit ops) -> parallel builds -> one compare sheet.

    python variants.py <base.json> <work_dir> <variants.json> [--refs DIR] [--full] [--jobs 4] [--no-base]

variants.json maps a variant name to a list of edit ops (amikit/edit.py: set / add / remove / scale_part /
recolour / customisation ops), e.g.

    {"eyes16":   [{"op": "set", "path": "face.eye.size_mm", "value": 16}],
     "eyes_low": [{"op": "set", "path": "face.eye.at.round", "value": 19.5},
                  {"op": "set", "path": "face.mouth.points", "value": [[17.6, 0], [17.1, -6]]}]}

Writes <work>/<name>.json and builds <work>/<name>/ for the base and every variant (in parallel, `--fast` unless
--full), then
  <work>/compare.png       rows: the concept (if --refs), base, variants; columns: front, 3/4, side, back; every toy
                           cut out and scaled to the SAME HEIGHT, so sizes and proportions compare directly
  <work>/compare_face.png  the upper part (face, ears) of front and 3/4, larger
  <work>/summary.json      per row: ok, flags, face problems, cute score, front/side/back silhouette IoU (with refs)
and prints the summary. LOOK at both pictures, pick the variant that is closest to the concept by eye, and only
then read the numbers. Change one thing (one parameter group) per variant; 2-4 variants per round.
`<work>/<name>.json` of the winner is the new base. Decide head shape and final picks on --full builds (fast
renders make heads look wider and flatter).

One extra op, because the face must move as a group (eyes, glints, nose, mouth, cheeks, muzzle, face patches):
    {"op": "shift_face", "on": "head", "rounds": -1.5}          + = toward the piece's end (usually down the head)
moves every face item on that piece, every assembly entry sewn on it at a numeric round (except parts whose id
contains a word in "skip", default ["ear", "hat", "horn", "tail", "hair", "wig"]) and, with "zones": true, the
ellipse / patch / polygon colour zones of that piece. Use it after re-rounding a head and to raise or lower a face.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _engine as E  # noqa: E402

from amikit import edit  # noqa: E402


SKIP = ["ear", "hat", "horn", "tail", "hair", "wig"]


def _shift_at(at, d):
    if isinstance(at, dict) and isinstance(at.get("round"), (int, float)):
        at["round"] = round(at["round"] + d, 2)


def shift_face(spec, op):
    """The face group of one piece moved by `rounds` (see the module doc)."""
    on, d = op["on"], float(op["rounds"])
    skip = op.get("skip", SKIP)
    names = {on}
    for e in spec.get("assembly", []):          # instances of the piece (mirrored copies keep the entry's name)
        if e.get("part") == on:
            names.add(e.get("name", on))
    for f in spec.get("face", []):
        if f.get("on") in names:
            _shift_at(f.get("at"), d)
            if isinstance(f.get("points"), list):
                f["points"] = [[round(p[0] + d, 2), p[1]] if isinstance(p[0], (int, float)) else p
                               for p in f["points"]]
    for e in spec.get("assembly", []):
        if e.get("on") in names and not any(w in e.get("part", "") for w in skip):
            _shift_at(e.get("at"), d)
    if op.get("zones"):
        for p in spec.get("parts", []):
            if p.get("id") != on:
                continue
            for z in p.get("colours", []):
                if "ellipse" in z:
                    z["ellipse"]["center"][0] = round(z["ellipse"]["center"][0] + d, 2)
                elif "shape" in z:
                    z["shape"] = [[round(v[0] + d, 2), v[1]] for v in z["shape"]]
                elif "patch" in z:
                    _shift_at(z["patch"].get("at"), d)
    return spec


def apply_ops(base, ops):
    spec = base
    plain = []
    for op in ops:
        if op.get("op") == "shift_face":
            if plain:
                spec, _ = edit.apply(spec, plain, validate=False)
                plain = []
            spec = shift_face(json.loads(json.dumps(spec)), op)
        else:
            plain.append(op)
    if plain or spec is base:
        spec, _ = edit.apply(spec, plain, validate=False)
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="variants.py")
    ap.add_argument("base")
    ap.add_argument("work")
    ap.add_argument("variants", nargs="?", help="JSON file {name: [ops]}; omit to build the base only")
    ap.add_argument("--refs", help="folder with front.png / left.png / back.png (refs.py)")
    ap.add_argument("--full", action="store_true", help="full-quality builds (slower; use for final decisions)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-base", action="store_true", help="skip rebuilding the base")
    ap.add_argument("--size", type=int, default=300, help="tile size in px")
    a = ap.parse_args(argv)

    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    base = E.read_json(a.base)
    variants = E.read_json(a.variants) if a.variants else {}
    refs = Path(a.refs).resolve() if a.refs else None

    specs, errors = {}, {}
    if not a.no_base:
        specs["base"] = base
    for name, ops in variants.items():
        try:
            specs[name] = apply_ops(base, ops)
        except Exception as ex:  # noqa: BLE001 - report and go on with the others
            errors[name] = f"edit failed: {ex}"
    for name, spec in specs.items():
        E.write_json(work / f"{name}.json", spec)

    ref = (refs / "front.png") if refs and (refs / "front.png").is_file() else None

    def run(name):
        ok, log = E.build(work / f"{name}.json", work / name, ref, fast=not a.full)
        return name, ok, log

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        results = list(ex.map(run, specs))

    summary = {}
    for name, ok, log in results:
        out = work / name
        row = {"ok": ok}
        if not ok:
            row["error"] = log[-1500:]
        else:
            fl = E.flags(out)
            row["flags"] = fl
            row["face_problems"] = [f for f in fl if f.startswith("face:")]
            row["cute"] = E.cute_score(out)
            if refs:
                for view, ours in (("front", "view_front.png"), ("side", "view_side.png"),
                                   ("back", "view_back.png")):
                    rp = refs / E.CONCEPT_FOR[view if view != "side" else "side"]
                    if rp.is_file():
                        row[f"{view}_iou"] = E.iou(rp, out / ours, out / f"fit_{view}.png")
        summary[name] = row
    for name, err in errors.items():
        summary[name] = {"ok": False, "error": err}

    S = a.size
    rows, face_rows = [], []
    if refs:
        ct = []
        for v in E.VIEWS:
            f = E.CONCEPT_FOR[v]
            ct.append(E.tile(E.toy_crop(Image.open(refs / f)), S) if f and (refs / f).is_file() else E.tile(None, S))
        rows.append(("CONCEPT", ct))
        if (refs / "front.png").is_file():
            face_rows.append(("CONCEPT", [E.tile(E.head_crop(Image.open(refs / "front.png")), int(S * 1.4)),
                                          E.tile(None, int(S * 1.4))]))
    for name, ok, _ in results:
        if not ok:
            continue
        out = work / name
        r = summary[name]
        label = name + "\n" + "  ".join(f"{k[0]}{r[k]:.2f}" for k in ("front_iou", "side_iou", "back_iou")
                                        if r.get(k) is not None)
        label += f"\ncute {r.get('cute')}  flags {len(r.get('flags', []))}"
        if r.get("face_problems"):
            label += "\nFACE HIDDEN"
        rows.append((label, [E.tile(E.toy_crop(Image.open(out / f"view_{v}.png")), S) for v in E.VIEWS]))
        face_rows.append((label, [E.tile(E.head_crop(Image.open(out / f"view_{v}.png")), int(S * 1.4))
                                  for v in ("front", "34")]))
    if rows:
        E.grid(rows, S, ["front", "3/4", "side (toy's left)", "back"]).save(work / "compare.png")
        E.grid(face_rows, int(S * 1.4), ["face front", "face 3/4"]).save(work / "compare_face.png")
    E.write_json(work / "summary.json", summary)
    print(json.dumps({"compare": str(work / "compare.png"), "compare_face": str(work / "compare_face.png"),
                      "rows": summary}, ensure_ascii=False, indent=1))
    return 0 if all(r.get("ok") for r in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
