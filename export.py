"""Scene meshes -> GLB (vertex colours with baked ambient occlusion and stitch-groove shading, per-material
primitives), for the browser viewer/editor."""
from __future__ import annotations
import numpy as np
import trimesh


def to_glb(parts, path, ao=None, decimate_stitch=False):
    scene = trimesh.Scene()
    off = 0
    used = {}
    for p in parts:
        name = p["name"]
        k = 1
        while name in used:                    # unique node names (an eye pair, or a name that looks like "a#2")
            k += 1
            name = f"{p['name']}#{k}"
        used[name] = True
        V, F, C = p["V"], p["F"], p["C"].copy()
        n = len(V)
        shade = np.ones(n)
        if p["mat"] == 0:
            shade *= 0.72 + 0.28 * np.clip(p["H"], 0, 1)        # grooves between stitches
        if ao is not None:
            shade *= 0.35 + 0.65 * ao[off:off + n]
        off += n
        rgb = np.clip(C * shade[:, None], 0, 1)
        rgba = np.hstack([(rgb * 255).astype(np.uint8), np.full((n, 1), 255, np.uint8)])
        m = trimesh.Trimesh(vertices=V, faces=F, vertex_normals=p["N"], process=False)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=rgba)
        m.metadata["name"] = name
        m.metadata["material"] = int(p["mat"])
        scene.add_geometry(m, node_name=name, geom_name=name)
    scene.export(path)
    return path
