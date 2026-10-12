"""Regression runner: build every toy in amikit/toys, record what matters, compare with an earlier run.

    # from amigurumi-lab/
    ../.venv/Scripts/python -m amikit.regress                       # all toys -> ../out/amikit/_regress_runs/<ts>/
    ../.venv/Scripts/python -m amikit.regress --fast --jobs 4       # coarser mesh, smaller renders, 4 at a time
    ../.venv/Scripts/python -m amikit.regress --compare ../out/amikit/_regress_runs/<earlier ts>
    ../.venv/Scripts/python -m amikit.regress --toys teddy,fox --compare latest

Per toy (each built in its own subprocess, with a timeout, so one hanging/crashing toy never stops the run):
geometry time (Toy() + build), render time, triangle count, flags, library-audit outliers, pattern arithmetic
(rounds.check + pattern_text.self_check), the 3/4 view (views/<toy>_34.png) and a thumbnail.
Writes report.json, report.md, contact_sheet.png. With --compare: new / gone flags, time regressions, triangle
and audit changes, and the mean absolute pixel difference of the 3/4 view per toy (0..255), plus side-by-side
images of the toys that changed (compare/<toy>.png). Compare runs made with the same --fast / size settings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
RUNS = LAB.parent / "out" / "amikit" / "_regress_runs"
TIME_RATIO = 1.3          # geometry time regression: slower by this factor ...
TIME_ABS = 0.5            # ... and by at least this many seconds
IMG_CHANGED = 1.0         # mean abs pixel difference (0..255) above which a toy counts as visually changed


# ------------------------------------------------------------------------------------------------- one toy
def run_one(spec_path, out_dir, fast=False, size=None):
    """Build one toy and write <out_dir>/<toy>.json (+ views/<toy>_34.png, thumbs/<toy>.png)."""
    import numpy as np  # noqa: F401  (imported early so import time is not counted as geometry)
    from .toy import Toy
    from .rounds import expand, check
    spec_path = Path(spec_path); out = Path(out_dir)
    name = spec_path.stem
    res = {"toy": name, "spec": str(spec_path), "ok": False}
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
        t0 = time.perf_counter()
        toy = Toy(spec)
        t1 = time.perf_counter()
        parts = toy.build(ku=4 if fast else 6, kv=3 if fast else 5)
        t2 = time.perf_counter()
        res.update(t_toy=round(t1 - t0, 3), t_mesh=round(t2 - t1, 3), t_geometry=round(t2 - t0, 3),
                   tris=int(sum(len(p["F"]) for p in parts)), verts=int(sum(len(p["V"]) for p in parts)),
                   meshes=len(parts), flags=list(toy.flags), instances=len(toy.inst))
        # pattern arithmetic
        errs = []
        for p in spec["parts"]:
            if "rounds" in p:
                errs += [f"{p['id']}: {e}" for e in check(expand(p["rounds"]))]
        try:
            from .pattern_text import pattern_model, self_check
            model = self_check(pattern_model(toy))
            errs += [f"{p['id']}: {e}" for p in model["parts"] for e in p.get("check", []) + p.get("self_check", [])]
            res["pattern"] = "ok" if not errs else f"{len(errs)} error(s)"
        except Exception as ex:  # noqa: BLE001
            res["pattern"] = f"pattern model failed: {ex!r}"
        res["pattern_errors"] = errs
        # library audit (advice): parts outside the designers' 15-85% range
        try:
            from .library import audit
            a = audit(spec)
            res["audit_outliers"] = sum(1 for r in a["parts"] if r["issues"])
            res["audit_outlier_parts"] = [r["part"] for r in a["parts"] if r["issues"]]
            res["audit_parts"] = len(a["parts"])
        except Exception as ex:  # noqa: BLE001
            res["audit_outliers"] = None
            res["audit_error"] = repr(ex)
        # the 3/4 view
        t3 = time.perf_counter()
        try:
            from .render import Renderer
            r = Renderer(size=size or (360 if fast else 480), ssaa=2)
            r.set_scene(parts, ao_dirs=32 if fast else 64)
            im = r.render(35.0)
            r.release()
            (out / "views").mkdir(parents=True, exist_ok=True); (out / "thumbs").mkdir(parents=True, exist_ok=True)
            im.save(out / "views" / f"{name}_34.png")
            th = im.copy(); th.thumbnail((160, 160)); th.save(out / "thumbs" / f"{name}.png")
            res["view_34"] = f"views/{name}_34.png"; res["thumb"] = f"thumbs/{name}.png"
        except Exception as ex:  # noqa: BLE001
            res["render_error"] = repr(ex)
        res["t_render"] = round(time.perf_counter() - t3, 3)
        res["ok"] = True
    except Exception as ex:  # noqa: BLE001 - the report says which toy broke and how
        res["error"] = f"{type(ex).__name__}: {ex}"
        res["traceback"] = traceback.format_exc()
    (out / "toys").mkdir(parents=True, exist_ok=True)
    (out / "toys" / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
    return res


def _run_sub(spec, out, fast, size, timeout):
    cmd = [sys.executable, "-m", "amikit.regress", "--worker", str(spec), "--out", str(out)]
    if fast:
        cmd.append("--fast")
    if size:
        cmd += ["--size", str(size)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=LAB, capture_output=True, text=True, timeout=timeout)
        f = Path(out) / "toys" / f"{Path(spec).stem}.json"
        if f.exists():
            res = json.loads(f.read_text(encoding="utf8"))
        else:
            res = {"toy": Path(spec).stem, "ok": False,
                   "error": f"worker exit {p.returncode}: {(p.stderr or '')[-1500:]}"}
    except subprocess.TimeoutExpired:
        res = {"toy": Path(spec).stem, "ok": False, "error": f"timeout after {timeout}s"}
    res["t_wall"] = round(time.perf_counter() - t0, 2)
    return res


# ------------------------------------------------------------------------------------------------- reports
def engine_fingerprint():
    out = {}
    for f in sorted(HERE.glob("*.py")):
        if f.name.startswith("test_") or f.name == "regress.py":
            continue
        out[f.name] = hashlib.md5(f.read_bytes()).hexdigest()[:10]
    return out


def contact_sheet(run_dir, results, cols=6, tile=220):
    from PIL import Image, ImageDraw
    run_dir = Path(run_dir)
    n = len(results)
    rows = max(1, (n + cols - 1) // cols)
    lab = 34
    sheet = Image.new("RGB", (cols * tile, rows * (tile + lab)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(results):
        x, y = (i % cols) * tile, (i // cols) * (tile + lab)
        v = r.get("view_34")
        if v and (run_dir / v).exists():
            im = Image.open(run_dir / v).convert("RGB"); im.thumbnail((tile, tile))
            sheet.paste(im, (x + (tile - im.width) // 2, y))
        else:
            dr.rectangle([x + 4, y + 4, x + tile - 4, y + tile - 4], outline=(200, 40, 40), width=3)
            dr.text((x + 12, y + tile // 2), "FAILED" if not r.get("ok") else "no render", fill=(200, 40, 40))
        status = "" if r.get("ok") else "  FAILED"
        dr.text((x + 6, y + tile + 2), f"{r['toy']}{status}", fill=(0, 0, 0))
        dr.text((x + 6, y + tile + 16), f"{r.get('t_geometry', '-')}s  {len(r.get('flags', []))} flags  "
                                        f"aud {r.get('audit_outliers', '-')}  pat {r.get('pattern', '-')}"[:44],
                fill=(90, 90, 90))
    p = run_dir / "contact_sheet.png"
    sheet.save(p)
    return p


def image_diff(a_path, b_path):
    """Mean absolute pixel difference (0..255) between two images (b resized to a's size if needed)."""
    import numpy as np
    from PIL import Image
    a = Image.open(a_path).convert("RGB"); b = Image.open(b_path).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())


def compare(run_dir, prev_dir, results):
    from PIL import Image
    run_dir, prev_dir = Path(run_dir), Path(prev_dir)
    prev = json.loads((prev_dir / "report.json").read_text(encoding="utf8"))
    pmap = {r["toy"]: r for r in prev["toys"]}
    rows = []
    (run_dir / "compare").mkdir(exist_ok=True)
    for r in results:
        name = r["toy"]
        p = pmap.get(name)
        row = {"toy": name}
        if p is None:
            row["status"] = "new toy"
            rows.append(row); continue
        if not r.get("ok") or not p.get("ok"):
            row["status"] = f"ok {p.get('ok')} -> {r.get('ok')}"
            if not r.get("ok"):
                row["error"] = r.get("error")
            rows.append(row); continue
        fn, fp = set(r.get("flags", [])), set(p.get("flags", []))
        row["new_flags"] = sorted(fn - fp); row["gone_flags"] = sorted(fp - fn)
        tn, tp = r.get("t_geometry", 0), p.get("t_geometry", 0)
        row["t_geometry"] = [tp, tn]
        row["time_regression"] = bool(tp and tn > tp * TIME_RATIO and tn - tp > TIME_ABS)
        row["tris"] = [p.get("tris"), r.get("tris")]
        row["audit_outliers"] = [p.get("audit_outliers"), r.get("audit_outliers")]
        row["pattern"] = [p.get("pattern"), r.get("pattern")]
        a, b = run_dir / r.get("view_34", "-"), prev_dir / p.get("view_34", "-")
        if r.get("view_34") and p.get("view_34") and a.exists() and b.exists():
            d = image_diff(a, b)
            row["img_diff"] = round(d, 2)
            if d > IMG_CHANGED:
                ia = Image.open(b).convert("RGB"); ib = Image.open(a).convert("RGB").resize(ia.size)
                sb = Image.new("RGB", (ia.width * 2 + 8, ia.height), (255, 255, 255))
                sb.paste(ia, (0, 0)); sb.paste(ib, (ia.width + 8, 0))
                sb.save(run_dir / "compare" / f"{name}.png")
                row["side_by_side"] = f"compare/{name}.png"
        else:
            row["img_diff"] = None
        changed = (row["new_flags"] or row["gone_flags"] or row["time_regression"] or row["tris"][0] != row["tris"][1]
                   or row["audit_outliers"][0] != row["audit_outliers"][1] or row["pattern"][0] != row["pattern"][1]
                   or (row["img_diff"] or 0) > IMG_CHANGED)
        row["status"] = "changed" if changed else "same"
        rows.append(row)
    for name in sorted(set(pmap) - {r["toy"] for r in results}):
        rows.append({"toy": name, "status": "missing in this run"})
    return {"previous": str(prev_dir), "previous_engine": prev.get("engine"), "rows": rows,
            "engine_files_changed": sorted(k for k, v in (prev.get("engine") or {}).items()
                                           if engine_fingerprint().get(k) != v)}


def write_md(run_dir, report):
    L = [f"# amikit regression run {report['timestamp']}", "",
         f"mode: {'fast' if report['fast'] else 'full'}, toys: {len(report['toys'])}, "
         f"total wall {report['t_total']:.0f}s, jobs {report['jobs']}", ""]
    if report.get("run_note"):
        L += [f"run note: {report['run_note']}", ""]
    fails = [r for r in report["toys"] if not r.get("ok")]
    L += [f"**{len(fails)} toy(s) failed to build.**" if fails else "All toys built.", "",
          "![contact sheet](contact_sheet.png)", "",
          "| toy | geometry s | render s | tris | flags | audit outliers | pattern |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for r in report["toys"]:
        if not r.get("ok"):
            L.append(f"| {r['toy']} | FAILED | | | | | {str(r.get('error'))[:80]} |")
            continue
        L.append(f"| {r['toy']} | {r['t_geometry']:.2f} | {r.get('t_render', 0):.1f} | {r['tris']} | "
                 f"{len(r['flags'])} | {r.get('audit_outliers')} | {r.get('pattern')} |")
    L += ["", "## Flags", ""]
    for r in report["toys"]:
        if r.get("flags"):
            L.append(f"- **{r['toy']}**: " + "; ".join(r["flags"]))
    perr = [r for r in report["toys"] if r.get("pattern_errors")]
    if perr:
        L += ["", "## Pattern arithmetic errors", ""]
        for r in perr:
            L.append(f"- **{r['toy']}**: " + "; ".join(r["pattern_errors"][:8]))
    if fails:
        L += ["", "## Failures", ""]
        for r in fails:
            L += [f"### {r['toy']}", "```", (r.get("traceback") or r.get("error") or "")[-2500:], "```"]
    cmp_ = report.get("compare")
    if cmp_:
        rows = cmp_["rows"]
        ch = [x for x in rows if x.get("status") != "same"]
        L += ["", f"## Compared with {cmp_['previous']}", "",
              f"engine files changed since then: {', '.join(cmp_['engine_files_changed']) or 'none'}", "",
              f"{len(ch)} of {len(rows)} toys changed (image threshold {IMG_CHANGED} mean abs px diff, time "
              f"regression x{TIME_RATIO} and +{TIME_ABS}s).", "",
              "| toy | status | img diff | geometry s (prev -> now) | tris | audit | pattern | new flags | gone flags |",
              "|---|---|---:|---|---|---|---|---|---|"]
        for x in rows:
            if "t_geometry" not in x:
                L.append(f"| {x['toy']} | {x['status']} | | | | | | | {x.get('error', '')} |")
                continue
            tg = f"{x['t_geometry'][0]:.2f} -> {x['t_geometry'][1]:.2f}" + (" **SLOWER**" if x["time_regression"] else "")
            img = "-" if x["img_diff"] is None else (f"**{x['img_diff']:.2f}**" if x["img_diff"] > IMG_CHANGED
                                                     else f"{x['img_diff']:.2f}")
            tr = "" if x["tris"][0] == x["tris"][1] else f"{x['tris'][0]} -> {x['tris'][1]}"
            au = "" if x["audit_outliers"][0] == x["audit_outliers"][1] else f"{x['audit_outliers'][0]} -> {x['audit_outliers'][1]}"
            pa = "" if x["pattern"][0] == x["pattern"][1] else f"{x['pattern'][0]} -> {x['pattern'][1]}"
            L.append(f"| {x['toy']} | {x['status']} | {img} | {tg} | {tr} | {au} | {pa} | "
                     f"{'; '.join(x['new_flags'])} | {'; '.join(x['gone_flags'])} |")
        sbs = [x for x in rows if x.get("side_by_side")]
        if sbs:
            L += ["", "### Changed 3/4 views (previous | now)", ""]
            L += [f"**{x['toy']}** ({x['img_diff']:.2f})\n\n![{x['toy']}]({x['side_by_side']})\n" for x in sbs]
    (Path(run_dir) / "report.md").write_text("\n".join(L) + "\n", encoding="utf8")


def _latest_run(exclude=None):
    runs = sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "report.json").exists() and p != exclude)
    return runs[-1] if runs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="coarser mesh (ku=4, kv=3), 360 px render, 32 AO dirs")
    ap.add_argument("--toys", help="comma-separated toy names (default: every amikit/toys/*.json)")
    ap.add_argument("--compare", help="earlier run dir to compare with ('latest' = the newest earlier run)")
    ap.add_argument("--out", help="run dir (default ../out/amikit/_regress_runs/<timestamp>)")
    ap.add_argument("--jobs", type=int, default=1, help="toys built in parallel (timings are noisier when > 1)")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per toy")
    ap.add_argument("--size", type=int, help="render size in px (default 360 fast / 480 full)")
    ap.add_argument("--note", help="run-notes/*.md recap this run belongs to (recorded in report.json)")
    ap.add_argument("--time-ratio", type=float, default=TIME_RATIO,
                    help=f"geometry time regression factor (default {TIME_RATIO}; timings are noisy on a busy machine)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    globals()["TIME_RATIO"] = a.time_ratio

    if a.worker:
        run_one(a.worker, a.out, a.fast, a.size)
        return 0

    specs = sorted((HERE / "toys").glob("*.json"))
    if a.toys:
        want = {t.strip() for t in a.toys.split(",")}
        specs = [s for s in specs if s.stem in want]
        missing = want - {s.stem for s in specs}
        if missing:
            print(f"unknown toys: {sorted(missing)}", file=sys.stderr)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S") + ("-fast" if a.fast else "")
    run_dir = Path(a.out) if a.out else RUNS / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    prev = None
    if a.compare:
        prev = _latest_run(exclude=run_dir) if a.compare == "latest" else Path(a.compare)
        if prev is None or not (prev / "report.json").exists():
            print(f"--compare: no report.json in {prev}", file=sys.stderr)
            return 2
    print(f"regression run -> {run_dir}  ({len(specs)} toys, {'fast' if a.fast else 'full'}, jobs {a.jobs})")
    t0 = time.perf_counter()

    def one(sp):
        r = _run_sub(sp, run_dir, a.fast, a.size, a.timeout)
        st = "ok " if r.get("ok") else "FAIL"
        print(f"  {st} {r['toy']:<14} geometry {r.get('t_geometry', '-')}s render {r.get('t_render', '-')}s "
              f"tris {r.get('tris', '-')} flags {len(r.get('flags', []))} audit {r.get('audit_outliers', '-')} "
              f"pattern {r.get('pattern', '-')}" + ("" if r.get("ok") else f"  {str(r.get('error'))[:120]}"),
              flush=True)
        return r

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        results = list(ex.map(one, specs))
    report = {"timestamp": ts, "fast": a.fast, "jobs": a.jobs, "size": a.size, "python": sys.version.split()[0],
              "engine": engine_fingerprint(), "run_note": a.note, "t_total": round(time.perf_counter() - t0, 1),
              "toys": results}
    contact_sheet(run_dir, results)
    if prev is not None:
        report["compare"] = compare(run_dir, prev, results)
        if bool(report["fast"]) != bool(json.loads((prev / "report.json").read_text(encoding="utf8")).get("fast")):
            print("warning: comparing a fast run with a full run: image and time differences are not meaningful")
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf8")
    write_md(run_dir, report)
    n_fail = sum(1 for r in results if not r.get("ok"))
    print(f"done in {report['t_total']:.0f}s: {len(results) - n_fail} ok, {n_fail} failed -> {run_dir / 'report.md'}")
    if prev is not None:
        rows = report["compare"]["rows"]
        for x in rows:
            if x.get("status") != "same":
                bits = [x["status"]]
                if x.get("img_diff") is not None:
                    bits.append(f"img {x['img_diff']:.2f}")
                if x.get("time_regression"):
                    bits.append(f"SLOWER {x['t_geometry'][0]:.2f}->{x['t_geometry'][1]:.2f}s")
                if x.get("new_flags"):
                    bits.append(f"new flags {x['new_flags']}")
                if x.get("gone_flags"):
                    bits.append(f"gone flags {x['gone_flags']}")
                print(f"  {x['toy']:<14} " + ", ".join(bits))
        print(f"compare: {sum(x.get('status') != 'same' for x in rows)} of {len(rows)} toys changed vs {prev}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
