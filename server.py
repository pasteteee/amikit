"""Local editor server: static editor UI + JSON API over amikit (edit ops, fast GLB builds, renders, LLM edits).

    python -m amikit.server [port]          (from amigurumi-lab/, default port 8765, binds 127.0.0.1)
    -> open http://127.0.0.1:8765

API (JSON in/out; errors are {"error": message} with 4xx/5xx):
    GET  /api/toys                    [{id, name, title, source}]   (amikit/toys/*.json + ../out/editor/*.json)
    GET  /api/toy/<id>                {spec, summary, source}       (<id> = "teddy" or "saved/<name>")
    POST /api/build   {spec}          {glb_url, flags, summary, timings, pattern_url?}
    POST /api/edit    {spec, ops, build?}  {spec, report[, build]}   (400 on EditError)
    POST /api/command {spec, text}    {ops, explanation, spec, report}   (503 without ANTHROPIC_API_KEY)
    POST /api/render  {spec}          {png_url, flags, timings}     (four full-quality views, one sheet)
    POST /api/save    {name, spec}    {id, path}                    (../out/editor/<name>.json only)

All GL work (moderngl contexts are per-thread) and toy builds run on one dedicated worker thread.
"""
from __future__ import annotations
import concurrent.futures as cf
import hashlib
import json
import mimetypes
import re
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import edit

HERE = Path(__file__).resolve().parent
TOYS = HERE / "toys"
EDITOR = HERE / "editor"
OUT = HERE.parent.parent / "out" / "editor"
CACHE = OUT / "cache"
CACHE_KEEP = 80

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("text/markdown", ".md")


class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg); self.status = status; self.msg = msg


# ---------------------------------------------------------------------------------------------------------------
# GL / build worker: a single thread owns the Renderer (and builds toys, so builds are serialised)
_gl = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="amikit-gl")
_renderer = None


def _get_renderer():
    global _renderer
    if _renderer is None:
        from .render import Renderer
        _renderer = Renderer(size=700, ssaa=2)
    return _renderer


def _set_scene(r, parts, ao_dirs):
    # release the previous scene's GPU buffers (Renderer re-creates them on every set_scene)
    for k in ("vao", "vbo", "ibo", "vao_d", "vbo_d", "ibo_d"):
        o = getattr(r, k, None)
        if o is not None:
            try:
                o.release()
            except Exception:  # noqa: BLE001
                pass
    r.set_scene(parts, ao_dirs=ao_dirs)


def _spec_hash(spec, salt=""):
    return hashlib.sha1((salt + json.dumps(spec, sort_keys=True)).encode()).hexdigest()[:16]


def _prune_cache():
    files = sorted(CACHE.glob("*"), key=lambda p: p.stat().st_mtime)
    for p in files[:-CACHE_KEEP]:
        try:
            if p.is_dir():
                for q in p.iterdir():
                    q.unlink()
                p.rmdir()
            else:
                p.unlink()
        except OSError:
            pass


def _make_toy(spec):
    from .toy import Toy
    try:
        edit.summary(spec)  # cheap stitch-arithmetic check, also needed for the response
        return Toy(spec)
    except Exception as ex:  # noqa: BLE001
        raise ApiError(400, f"the toy does not build: {type(ex).__name__}: {ex}") from ex


def _pattern(toy, key):
    """Write the written pattern if amikit/pattern_text.py exists (another module; optional)."""
    try:
        from .pattern_text import write_patterns
    except ImportError:
        return None
    d = CACHE / f"{key}_pattern"
    try:
        d.mkdir(parents=True, exist_ok=True)
        write_patterns(toy, d)
    except Exception as ex:  # noqa: BLE001
        (d / "PATTERN_RU.md").write_text(f"Ошибка генерации схемы: {ex}", encoding="utf8")
    for n in ("PATTERN_RU.md", "PATTERN_EN.md"):
        if (d / n).exists():
            return f"/cache/{d.name}/{n}"
    return None


def _do_build(spec):
    # any change to the engine (geometry, render, pattern text) invalidates cached builds
    stamp = "".join(str(f.stat().st_mtime_ns) for f in sorted(HERE.glob("*.py")))
    key = _spec_hash(spec, "fast2" + stamp)
    meta = CACHE / f"{key}.json"
    if meta.exists() and (CACHE / f"{key}.glb").exists():
        res = json.loads(meta.read_text(encoding="utf8")); res["timings"] = {**res["timings"], "cached": True}
        return res
    from .export import to_glb
    t0 = time.time()
    toy = _make_toy(spec)
    try:
        parts = toy.build(ku=4, kv=3)
    except Exception as ex:  # noqa: BLE001
        raise ApiError(400, f"the toy does not build: {type(ex).__name__}: {ex}") from ex
    t1 = time.time()
    r = _get_renderer()
    _set_scene(r, parts, ao_dirs=32)
    t2 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    # export.py names nodes by part name; duplicate names (two "eye@head") make trimesh drop a node, so make
    # them unique here ("eye@head", "eye@head#2")
    seen = {}
    uniq = []
    for p in parts:
        k = seen[p["name"]] = seen.get(p["name"], 0) + 1
        uniq.append(p if k == 1 else {**p, "name": f"{p['name']}#{k}"})
    to_glb(uniq, str(CACHE / f"{key}.glb"), ao=r.AO[:r.n_obj])
    t3 = time.time()
    pattern_url = _pattern(toy, key)
    t4 = time.time()
    res = {"glb_url": f"/cache/{key}.glb", "flags": toy.flags, "summary": edit.summary(spec),
           "meshes": [p["name"] for p in parts],
           "timings": {"geometry": round(t1 - t0, 2), "ao": round(t2 - t1, 2), "glb": round(t3 - t2, 2),
                       "pattern": round(t4 - t3, 2), "total": round(t4 - t0, 2)},
           "tris": int(sum(len(p["F"]) for p in parts))}
    if pattern_url:
        res["pattern_url"] = pattern_url
    meta.write_text(json.dumps(res, ensure_ascii=False), encoding="utf8")
    _prune_cache()
    return res


def _do_render(spec):
    from .render import sheet
    key = _spec_hash(spec, "render1")
    png = CACHE / f"{key}_views.png"
    t0 = time.time()
    toy = _make_toy(spec)
    parts = toy.build(ku=6, kv=5)
    t1 = time.time()
    r = _get_renderer()
    _set_scene(r, parts, ao_dirs=128)
    ims = r.views(which=("front", "34", "side", "back"))
    t2 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    sheet([ims[v] for v in ("front", "34", "side", "back")], size=700).save(png)
    return {"png_url": f"/cache/{png.name}", "flags": toy.flags,
            "timings": {"geometry": round(t1 - t0, 2), "render": round(t2 - t1, 2), "total": round(time.time() - t0, 2)}}


def build(spec):
    return _gl.submit(_do_build, spec).result()


def render(spec):
    return _gl.submit(_do_render, spec).result()


# ---------------------------------------------------------------------------------------------------------------
def _list_toys():
    out = []
    for p in sorted(TOYS.glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf8"))
            out.append({"id": p.stem, "name": s.get("name", p.stem), "title": s.get("title", ""), "source": "toys"})
        except (OSError, ValueError):
            pass
    if OUT.exists():
        for p in sorted(OUT.glob("*.json")):
            try:
                s = json.loads(p.read_text(encoding="utf8"))
                out.append({"id": "saved/" + p.stem, "name": s.get("name", p.stem), "title": s.get("title", ""),
                            "source": "saved"})
            except (OSError, ValueError):
                pass
    return out


def _load_toy(tid):
    if tid.startswith("saved/"):
        name, base, src = tid[6:], OUT, "saved"
    else:
        name, base, src = tid, TOYS, "toys"
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ApiError(400, "bad toy name")
    p = base / f"{name}.json"
    if not p.exists():
        raise ApiError(404, f"no toy {tid!r}")
    spec = json.loads(p.read_text(encoding="utf8"))
    return {"id": tid, "spec": spec, "summary": edit.summary(spec), "source": src}


def _need_spec(body):
    spec = body.get("spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("parts"), list):
        raise ApiError(400, "body needs a 'spec' object with 'parts'")
    return spec


def api_edit(body):
    spec = _need_spec(body)
    ops = body.get("ops")
    if not isinstance(ops, list):
        raise ApiError(400, "body needs 'ops' (a list)")
    want_build = bool(body.get("build"))
    try:
        # with build=true the build below validates (one Toy construction instead of two)
        new, report = edit.apply(spec, ops, validate=not want_build)
    except edit.EditError as ex:
        raise ApiError(400, str(ex)) from ex
    res = {"spec": new, "report": report}
    if want_build:
        b = build(new)  # raises ApiError(400) when the edited toy does not build
        report["flags"] = b["flags"]
        res["build"] = b
    try:
        res["summary"] = edit.summary(new)
    except Exception as ex:  # noqa: BLE001
        raise ApiError(400, f"the edited toy is invalid: {type(ex).__name__}: {ex}") from ex
    return res


def api_command(body):
    spec = _need_spec(body)
    text = str(body.get("text", "")).strip()
    if not text:
        raise ApiError(400, "empty request text")
    from .llm_edit import LLMError, LLMUnavailable, text_to_ops
    try:
        return text_to_ops(spec, text)
    except LLMUnavailable as ex:
        raise ApiError(503, str(ex)) from ex
    except LLMError as ex:
        raise ApiError(502, str(ex)) from ex


def api_save(body):
    spec = _need_spec(body)
    name = str(body.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", name):
        raise ApiError(400, "name: 1-64 latin letters, digits, '_' or '-'")
    OUT.mkdir(parents=True, exist_ok=True)
    p = (OUT / f"{name}.json").resolve()
    if p.parent != OUT.resolve():
        raise ApiError(400, "bad name")
    p.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf8")
    return {"id": "saved/" + name, "path": str(p)}


ROUTES_POST = {
    "/api/build": lambda b: build(_need_spec(b)),
    "/api/edit": api_edit,
    "/api/command": api_command,
    "/api/render": lambda b: render(_need_spec(b)),
    "/api/save": api_save,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "amikit-editor/1"

    def log_message(self, fmt, *args):  # quieter log: method, path, status
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, status, body, ctype="application/json; charset=utf-8", extra=None):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "no-cache")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status, msg):
        self._send(status, {"error": msg})

    def _static(self, root, rel):
        p = (root / rel).resolve()
        if root.resolve() not in p.parents and p != root.resolve():
            return self._err(403, "forbidden")
        if p.is_dir():
            p = p / "index.html"
        if not p.is_file():
            return self._err(404, f"not found: {rel}")
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json",):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/toys":
                return self._send(200, _list_toys())
            if path.startswith("/api/toy/"):
                return self._send(200, _load_toy(path[len("/api/toy/"):]))
            if path.startswith("/api/"):
                return self._err(404, "unknown endpoint")
            if path.startswith("/cache/"):
                return self._static(CACHE, path[len("/cache/"):])
            return self._static(EDITOR, path.lstrip("/") or "index.html")
        except ApiError as ex:
            self._err(ex.status, ex.msg)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc(); self._err(500, f"{type(ex).__name__}: {ex}")

    def do_POST(self):
        path = urlparse(self.path).path
        fn = ROUTES_POST.get(path)
        if fn is None:
            return self._err(404, "unknown endpoint")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20_000_000:
                raise ApiError(413, "request too large")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError as ex:
                raise ApiError(400, f"bad JSON: {ex}") from ex
            if not isinstance(body, dict):
                raise ApiError(400, "body must be a JSON object")
            self._send(200, fn(body))
        except ApiError as ex:
            self._err(ex.status, ex.msg)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc(); self._err(500, f"{type(ex).__name__}: {ex}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv else 8765
    CACHE.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print(f"amikit editor: http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close(); _gl.shutdown(wait=False)


if __name__ == "__main__":
    main()
