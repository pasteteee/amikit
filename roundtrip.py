"""Round trip: spec -> PATTERN_EN.md -> reverse -> spec' -> 3D, compared with the original.

    python -m amikit.roundtrip [spec.json ...] [--out DIR] [--renders]

Every difference is something the written pattern does not say (or says ambiguously): the test for "the model is
exactly what the written pattern makes". Per toy: stitch counts and colour charts piece by piece, and the
geometry (per-instance vertex distance in mm after grounding, same mesh topology when the counts agree; a
chamfer distance otherwise), plus face items. Writes report.md / report.json (and side-by-side sheets with
--renders).
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .toy import Toy
from .pattern_text import write_patterns, pattern_model, render_markdown
from .reverse import to_spec, build_toy

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "out" / "amikit" / "_roundtrip"


def _mesh(toy):
    parts = toy.build(ku=2, kv=2)
    return parts


def _inst_order(toy):
    return list(toy.inst.keys())


def _surface_dist(X, mesh):
    """Distance (mm) from points X to the triangle surface of a built mesh (dict with V, F)."""
    import trimesh
    tm = trimesh.Trimesh(mesh["V"], mesh["F"], process=False)
    _, d, _ = trimesh.proximity.closest_point(tm, X)
    return np.asarray(d, float)


def compare(spec_a, spec_b):
    """Original vs rebuilt: dict with per-piece findings and the worst geometry deviation (mm)."""
    ta, tb = Toy(spec_a), build_toy(spec_b)
    res = {"pieces": [], "counts_ok": True, "colours_ok": True, "max_mm": 0.0, "mean_mm": 0.0, "face": []}
    ia, ib = _inst_order(ta), _inst_order(tb)
    if len(ia) != len(ib):
        res["pieces"].append({"issue": f"{len(ia)} pieces placed vs {len(ib)} rebuilt", "a": ia, "b": ib})
    ma = {m["name"]: m for m in _mesh(ta)}
    mb = {m["name"]: m for m in _mesh(tb)}
    # compare in the toys' own frame (before standing them on the floor); the floor difference is reported apart
    for t, M in ((ta, ma), (tb, mb)):
        for m in M.values():
            m["V"] = m["V"] - np.array([0.0, t.ground_shift, 0.0])
    res["ground_mm"] = round(float(ta.ground_shift - tb.ground_shift), 2)
    devs = []
    for na, nb in zip(ia, ib):
        A, B = ta.inst[na], tb.inst[nb]
        row = {"a": na, "b": nb}
        if A.p.counts != B.p.counts:
            row["counts"] = [A.p.counts, B.p.counts]
            res["counts_ok"] = False
        ca, cb = ta.chart(A), tb.chart(B)
        if ca != cb:
            bad = [i + 1 for i, (x, y) in enumerate(zip(ca, cb)) if x != y]
            row["colour_rounds"] = bad[:12]
            res["colours_ok"] = False
        for k in ("closed", "stuff", "flatten", "squash"):
            va, vb = getattr(A.p, k), getattr(B.p, k)
            if va != vb:
                row[k] = [va, vb]
        Va, Vb = ma[na]["V"], mb[nb]["V"]
        # the surface: symmetric nearest-point distance (a 180-degree-symmetric oval sewn on turned round is the
        # same surface); the colours: the colour at the nearest point of the other surface
        from scipy.spatial import cKDTree
        _, i1 = cKDTree(Vb).query(Va)
        # distance to the other SURFACE (not its nearest vertex: a piece turned by part of a stitch about its own
        # axis is the same surface, but its vertices sit up to half a vertex spacing apart)
        d1, d2 = _surface_dist(Va, mb[nb]), _surface_dist(Vb, ma[na])
        d = np.concatenate([d1, d2])
        if Va.shape == Vb.shape:
            row["index_max_mm"] = round(float(np.linalg.norm(Va - Vb, axis=1).max()), 2)
        cdiff = np.abs(ma[na]["C"] - mb[nb]["C"][i1]).max(1) > 0.08
        if cdiff.mean() > 0.05:            # (boundary vertices move with sub-mm geometry shifts)
            row["colour_area"] = round(float(cdiff.mean()), 3)
            res["colours_ok"] = False
        row["max_mm"] = round(float(d.max()), 2); row["mean_mm"] = round(float(d.mean()), 2)
        devs.append(d)
        res["pieces"].append(row)
    # face / detail meshes (eyes, embroidery, cords, flat pieces): compare by order and centroid
    fa = [m for m in ma.values() if m["name"] not in ta.inst]
    fb = [m for m in mb.values() if m["name"] not in tb.inst]
    if len(fa) != len(fb):
        res["face"].append({"issue": f"{len(fa)} detail meshes vs {len(fb)}", "a": [m["name"] for m in fa],
                            "b": [m["name"] for m in fb]})
    # pair the details by position (the reader may list them in another order), closest pairs first
    pairs, used_a, used_b = [], set(), set()
    cand = sorted(((float(np.linalg.norm(x["V"].mean(0) - y["V"].mean(0))), i, j) for i, x in enumerate(fa)
                   for j, y in enumerate(fb)))
    for dd, i, j in cand:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); pairs.append((fa[i], fb[j]))
    for x, y in pairs:
        ca_, cb_ = x["V"].mean(0), y["V"].mean(0)
        sa, sb = np.ptp(x["V"], 0), np.ptp(y["V"], 0)
        dc = float(np.linalg.norm(ca_ - cb_)); ds = float(np.abs(sa - sb).max())
        colour = bool(np.abs(x["C"].mean(0) - y["C"].mean(0)).max() > 0.03)
        if dc > 0.8 or ds > 0.8 or colour:
            res["face"].append({"a": x["name"], "b": y["name"], "centre_mm": round(dc, 2), "size_mm": round(ds, 2),
                                "colour_differs": colour})
    if devs:
        allv = np.concatenate(devs)
        res["max_mm"] = round(float(allv.max()), 2)
        res["mean_mm"] = round(float(allv.mean()), 2)
    return res, ta, tb


def run(specs, out=OUT, renders=False):
    out = Path(out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sp in specs:
        sp = Path(sp)
        name = sp.stem
        d = out / name
        d.mkdir(exist_ok=True)
        t0 = time.time()
        row = {"toy": name}
        try:
            spec = json.loads(sp.read_text(encoding="utf-8-sig"))
            toy = Toy(spec)
            write_patterns(toy, d)
            text = (d / "PATTERN_EN.md").read_text(encoding="utf8")
            spec2, log = to_spec(text, name=spec.get("name"))
            (d / "spec_reversed.json").write_text(json.dumps(spec2, indent=1, ensure_ascii=False), encoding="utf8")
            (d / "reverse_report.txt").write_text("\n".join(log), encoding="utf8")
            res, ta, tb = compare(spec, spec2)
            row.update(res)
            row["log"] = log
            if renders:
                _sheet(ta, tb, d / "roundtrip_sheet.png")
        except Exception as ex:  # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {ex}"
            row["trace"] = traceback.format_exc()[-1500:]
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print(_line(row), flush=True)
    (out / "report.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf8")
    (out / "report.md").write_text(_report(rows), encoding="utf8")
    print(f"-> {out}")
    return rows, out


def _line(r):
    if "error" in r:
        return f"{r['toy']:28s} ERROR {r['error'][:150]}"
    bad = [p for p in r["pieces"] if p.get("max_mm", 0) > 2.0 or "counts" in p or "colour_rounds" in p or "issue" in p]
    return (f"{r['toy']:28s} counts {'ok' if r['counts_ok'] else 'DIFF'} colours {'ok' if r['colours_ok'] else 'DIFF'} "
            f"max {r['max_mm']:6.2f} mm mean {r['mean_mm']:5.2f} | worst: "
            + ", ".join(f"{p.get('a')}:{p.get('max_mm', '?')}" for p in sorted(bad, key=lambda p: -p.get('max_mm', 99))[:4])
            + (f" | face diffs {len(r['face'])}" if r["face"] else ""))


def _report(rows):
    L = ["# Round trip: spec -> PATTERN_EN.md -> reverse -> 3D", "",
         "| toy | counts | colours | max mm | mean mm | face diffs |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['toy']} | ERROR | {r['error'][:80]} | | | |")
            continue
        L.append(f"| {r['toy']} | {'ok' if r['counts_ok'] else 'DIFF'} | {'ok' if r['colours_ok'] else 'DIFF'} | "
                 f"{r['max_mm']} | {r['mean_mm']} | {len(r['face'])} |")
    L.append("")
    for r in rows:
        if "error" in r:
            L += [f"## {r['toy']}", "```", r.get("trace", ""), "```"]
            continue
        bad = [p for p in r["pieces"] if p.get("max_mm", 0) > 1.0 or len(p) > 4]
        if bad or r["face"]:
            L.append(f"## {r['toy']}")
            L += [f"- {json.dumps(p, default=str)[:400]}" for p in bad]
            L += [f"- face: {json.dumps(f, default=str)[:300]}" for f in r["face"]]
    return "\n".join(L)


def _sheet(ta, tb, path):
    from .render import Renderer
    from PIL import Image
    ims = []
    for t in (ta, tb):
        r = Renderer(size=420, ssaa=1)
        r.set_scene(t.build(ku=3, kv=3), ao_dirs=24)
        ims.append([r.render(a) for a in (0.0, 35.0, 90.0)])
    W = Image.new("RGB", (420 * 3, 840), "white")
    for j, row in enumerate(ims):
        for i, im in enumerate(row):
            W.paste(im.resize((420, 420)), (420 * i, 420 * j))
    W.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="*")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--renders", action="store_true")
    ap.add_argument("--top50", action="store_true")
    a = ap.parse_args()
    specs = a.specs or sorted(str(p) for p in (HERE / "toys").glob("*.json"))
    if a.top50:
        specs += sorted(str(p) for p in (HERE / "toys" / "top50").glob("*.json"))
    run(specs, a.out, a.renders)
