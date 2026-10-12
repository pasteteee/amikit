"""Can the face be seen? Visible share of every face feature on the built toy.

A face feature (each safety eye, the nose, the mouth embroidery, the muzzle and other pieces sewn on the front of
the head) is drawn twice from the front camera (and the 3/4 one): once with the whole toy (an id buffer with depth
test) and once with only the piece it is sewn on (an eye, a nose, embroidery) or alone (a muzzle). visible share =
pixels it keeps with the toy / pixels it has in that second picture. An eye under a cheek, an ear or a fringe, a
nose and mouth on a muzzle pressed into the head, a muzzle sunk into the head all show up as a low share, so a
build that "passes" with its face gone is flagged. (A safety eye's own sinking into its host is not a defect and
does not count: the host is in both pictures.)

    python -m amikit.visible <spec.json> [--fast]

Used by build.py (visibility.json + flags) and the skill's quality check.
"""
from __future__ import annotations

import json
import sys

import numpy as np

# minimum visible share from the front camera, per kind of feature
# (calibrated on 177 toys: real overlaps such as a muzzle bridge over the lower rim of small eyes, 0.68-0.78, pass;
# a muzzle buried in the head and the nose / mouth on it score 0)
LIMITS = {"eye": 0.6, "nose": 0.5, "mouth": 0.4, "muzzle": 0.2}
VIEWS = {"front": (0.0, 8.0), "34": (35.0, 8.0)}

_VS = """
#version 330
uniform mat4 mvp;
in vec3 in_pos; in float in_id;
flat out float v_id;
void main(){ v_id = in_id; gl_Position = mvp * vec4(in_pos, 1.0); }
"""
_FS = """
#version 330
flat in float v_id;
out float frag;
void main(){ frag = v_id; }
"""


def _ortho_mvp(center, radius, az_deg, elev_deg):
    from .render import look_at, ortho
    a, e = np.radians(az_deg), np.radians(elev_deg)
    fwd = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    eye = center + fwd * radius * 3
    return ortho(radius * 1.05, radius * 0.5, radius * 6) @ look_at(eye, center)


def features(toy, parts):
    """Face feature groups: name -> (kind, [mesh indices], {host piece names}). Eyes one by one, the nose, the mouth
    (all embroidery lines on the face pieces), and the pieces sewn on the front of the head that carry the face (a
    muzzle, a face patch: not appliques under 6 mm such as cheeks, not ears; their host set is empty: sinking into
    the head is exactly what is measured)."""
    groups = {}
    eye_hosts = set()

    def add(key, kind, i, host):
        g = groups.setdefault(key, (kind, [], set()))
        g[1].append(i)
        if host:
            g[2].add(host)
    for i, m in enumerate(parts):
        fi = m.get("face_item")
        if not fi:
            continue
        t, role, fid = fi["type"], fi.get("role") or "", str(fi.get("id") or "")
        if t in ("safety_eye", "eye") or (t == "satin" and role == "eye"):
            at = fi.get("at") or {}
            side = "L" if float(at.get("az", 0.0)) >= 0 else "R"
            add(f"eye {side} ({fi['on']})", "eye", i, fi["on"])
            eye_hosts.add(fi["on"])
        elif t == "safety_nose" or (t in ("satin", "felt") and "nose" in (fid + role).lower()):
            add(f"nose ({fi['on']})", "nose", i, fi["on"])
    hosts = set(eye_hosts)
    face_pieces = []
    for e in toy.entries:
        name = e.get("name")
        inst = toy.inst.get(name)
        if inst is None or e.get("on") is None:
            continue
        try:
            parent = toy._side(e["on"], None)
        except KeyError:
            continue
        if parent not in eye_hosts or "ear" in name.lower():
            continue
        at = e.get("_at") or e.get("at") or {}
        az = at.get("az")
        if az is None or abs(((float(az) + 180) % 360) - 180) > 30:
            continue
        if e.get("applique") and float(e.get("pad", 0.0)) < 6.0 and "muzzle" not in name.lower()                 and "face" not in name.lower() and "snout" not in name.lower():
            continue
        face_pieces.append(name)
        hosts.add(name)
    for i, m in enumerate(parts):
        if m["name"] in face_pieces:
            add(f"muzzle {m['name']}", "muzzle", i, None)
    for i, m in enumerate(parts):
        fi = m.get("face_item")
        if fi and fi["type"] in ("stitch", "embroidery", "line") and fi["on"] in hosts:
            add("mouth", "mouth", i, fi["on"])
    return groups


def visibility(toy, parts, views=("front", "34"), res=900):
    """{feature: {"kind", "front", "34"}} visible shares (0..1), and the list of flags."""
    import moderngl
    groups = features(toy, parts)
    if not groups:
        return {}, []
    solid = [m for m in parts if len(m["V"]) and len(m["F"])]
    ids = {id(m): k + 1 for k, m in enumerate(solid)}
    by_name = {}
    for m in solid:
        if "face_item" not in m:
            by_name.setdefault(m["name"], m)
    V = np.concatenate([m["V"] for m in solid]).astype("f4")
    I = np.concatenate([np.full(len(m["V"]), ids[id(m)], "f4") for m in solid])
    offs = np.cumsum([0] + [len(m["V"]) for m in solid])
    F = np.concatenate([m["F"] + offs[k] for k, m in enumerate(solid)]).astype("i4")
    lo, hi = V.min(0), V.max(0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)
    ctx = moderngl.create_standalone_context()
    try:
        prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        vbo = ctx.buffer(np.hstack([V, I[:, None]]).astype("f4").tobytes())
        col = ctx.renderbuffer((res, res), components=1, dtype="f4")
        dep = ctx.depth_renderbuffer((res, res))
        fbo = ctx.framebuffer(color_attachments=[col], depth_attachment=dep)
        ctx.enable(moderngl.DEPTH_TEST)

        def draw(faces, mvp):
            ibo = ctx.buffer(faces.astype("i4").tobytes())
            vao = ctx.vertex_array(prog, [(vbo, "3f 1f", "in_pos", "in_id")], ibo)
            fbo.use(); fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            prog["mvp"].write(mvp.T.astype("f4").tobytes())
            vao.render(moderngl.TRIANGLES)
            img = np.frombuffer(fbo.read(components=1, dtype="f4"), dtype="f4").reshape(res, res)
            vao.release(); ibo.release()
            return np.rint(img).astype(int)

        def faces_of(k):
            return solid[k - 1]["F"] + offs[k - 1]

        out = {name: {"kind": kind} for name, (kind, _, _) in groups.items()}
        for v in views:
            mvp = _ortho_mvp(center, radius, *VIEWS[v])
            full = draw(F, mvp)
            counts = np.bincount(full.reshape(-1), minlength=len(solid) + 1)
            for name, (kind, idx, hs) in groups.items():
                mine = sorted({ids[id(parts[i])] for i in idx})
                with_host = mine + [ids[id(by_name[h])] for h in hs if h in by_name]
                ref = draw(np.concatenate([faces_of(k) for k in with_host]), mvp)
                a = int(np.isin(ref, mine).sum())
                kept = int(sum(counts[k] for k in mine))
                out[name][v] = round(min(kept / a, 1.0), 3) if a else None
                out[name][v + "_px"] = a
    finally:
        ctx.release()
    flags = []
    for name, r in out.items():
        lim = LIMITS[r["kind"]]
        f = r.get("front")
        if f is not None and f < lim:
            what = {"eye": "the eye", "nose": "the nose", "mouth": "the mouth embroidery",
                    "muzzle": "the face piece"}[r["kind"]]
            why = ("sunk into the head" if r["kind"] == "muzzle" else
                   "covered by other pieces (or its piece is sunk into another)")
            flags.append(f"face: {what} {name} is only {100 * f:.0f}% visible from the front "
                         f"(needs {100 * lim:.0f}%): {why}")
    return out, flags


def main(argv=None):
    import argparse
    from .toy import Toy
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    toy = Toy(json.loads(open(a.spec, encoding="utf-8-sig").read()))
    parts = toy.build(ku=4 if a.fast else 6, kv=3 if a.fast else 5)
    res, flags = visibility(toy, parts)
    print(json.dumps(res, indent=1))
    print("\n".join(flags) or "(face fully visible)")


if __name__ == "__main__":
    main(sys.argv[1:])
