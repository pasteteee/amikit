"""Physical yarn for a whole tutorial (NVIDIA Newton, VBD rods).

The stitch engine (engine/sim.mjs, exported with export_targets.mjs) knows the choreography: where the hook goes, what
the finished stitches look like, which yarn the hook holds. Its strand is only a target. The yarn that is rendered is
simulated here: inextensible capsule chain with friction and contact (yarn-yarn, yarn-hook), fed through an eyelet at
the finger with a steady pull from the ball. Only bits that something actually holds are pulled toward the target:
  * yarn the hook carries (target lies on the hook surface),
  * yarn that is part of the finished fabric (the other hand holds the work),
  * the tail end (the other hand pulls it to close the ring).
Every other bit of yarn moves only because it is pushed or pulled by its neighbours, the hook or the finger.

Each body tracks its own place on the target strand (local nearest-point search within a few mm of material), so
re-parameterisations of the engine's strand never yank the yarn: the yarn can slide along its path like real yarn.

Usage: python hybrid.py <targets.jsonl> <out.jsonl> [from_frame] [to_frame]
"""
import json, math, os, sys, time
import numpy as np
import warp as wp
import newton
from newton import BodyFlags, ModelFlags

S = 0.01  # 1 mm -> 0.01 model units
R_COL = float(os.environ.get("R_COL", 0.78))  # collision radius (mm)
SEG = 1.0
FPS = 30
SUB = int(os.environ.get("SUB", 12))
src, out_path = sys.argv[1], sys.argv[2]
f_from = int(sys.argv[3]) if len(sys.argv) > 3 else 0
f_to = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9

wp.config.quiet = True
wp.init()


def norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n or 1.0)


# ---------------- targets
TG = []
with open(src) as fh:
    for k, line in enumerate(fh):
        if k > f_to + 1: break
        d = json.loads(line)
        P = np.array(d["pts"], float); Sx = np.array(d["s"], float)
        TG.append(dict(P=P, U=Sx - Sx[0], FF=np.array(d["ff"], float), HK=np.array(d["hk"], float),
                       hook=d["hook"], G=np.array(d["G"], float), t=d["t"]))
NF = min(len(TG), f_to + 1)
U_MAX = max(T["U"][-1] for T in TG[:NF])
SUPPLY = float(os.environ.get("SUPPLY", 70.0))
NB = int((U_MAX + SUPPLY) / SEG) + 1
print(f"frames {NF}, max material {U_MAX:.0f} mm, bodies {NB}", flush=True)


def interp_path(T, u):
    """Target point(s) at material coordinate(s) u, clamped to the path."""
    U = T["U"]; P = T["P"]
    return np.stack([np.interp(u, U, P[:, k]) for k in range(3)], axis=-1)


# ---------------- initial yarn: along the frame-0 target, the rest beyond the finger toward the ball
T0 = TG[f_from]
G0 = T0["G"]
dirOut = norm(os.environ.get("DIROUT") and [float(v) for v in os.environ["DIROUT"].split(",")] or (G0 - T0["P"][-8]))
u0 = np.arange(NB) * SEG
L0 = T0["U"][-1]
pts = interp_path(T0, np.minimum(u0, L0))
beyond = u0 > L0
pts[beyond] = G0 + dirOut[None, :] * (u0[beyond] - L0)[:, None]
# nudge exact duplicates (zero-length segments break the rod frames)
for i in range(1, NB):
    if np.linalg.norm(pts[i] - pts[i - 1]) < 1e-3: pts[i] = pts[i - 1] + dirOut * 1e-2
track_u = np.minimum(u0, L0).astype(float)

builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -float(os.environ.get("GRAV", 0.3)), 0.0))
builder.rigid_gap = 0.0
KE = float(os.environ.get("KE", 1e5))
cfg = newton.ModelBuilder.ShapeConfig(density=300.0, ke=KE, kd=0.0, mu=float(os.environ.get("MUY", 0.25)))
rod = newton.Rod([wp.vec3(*(p * S)) for p in pts], radius=R_COL * S)
rod.compute_frames()
bodies, joints = builder.add_rod(rod=rod, cfg=cfg, stretch_stiffness=float(os.environ.get("KS", 5e5)),
                                 bend_stiffness=float(os.environ.get("KB", 2e-4)), bend_damping=float(os.environ.get("KBD", 1e-4)),
                                 label="yarn", body_frame_origin="com")
assert len(bodies) == NB - 1 or len(bodies) == NB, (len(bodies), NB)
NY = len(bodies)
track_u = np.minimum((np.arange(NY) + 0.5) * SEG, L0)  # bodies sit between the rod points
for i in range(NY):
    for j in range(i + 2, min(NY, i + 4)):
        for sa in builder.body_shapes.get(bodies[i], []):
            for sb in builder.body_shapes.get(bodies[j], []):
                builder.add_shape_collision_filter_pair(int(sa), int(sb))
builder.rigid_gap = float(os.environ.get("GAP", 0.3)) * S

# hook (slim 2 mm): local x toward the tip (tip at 0), y = throat side
hook_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=0.0, mu=float(os.environ.get("MUH", 0.15)))
hook = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True, label="hook")
qx = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)


def cap(body, x0, x1, r, y=0.0, z=0.0):
    builder.add_shape_capsule(body=body, xform=wp.transform(wp.vec3((x0 + x1) / 2 * S, y * S, z * S), qx),
                              radius=r * S, half_height=(x1 - x0) / 2 * S, cfg=hook_cfg)


R_SH = float(os.environ.get("R_SHAFT", 1.0))
cap(hook, -60.0, -4.6, R_SH)  # shaft
cap(hook, -4.8, -1.3, 0.45, -0.52)  # spine under the throat
cap(hook, -1.35, -0.9, 0.92)  # head
# eyelet at the finger: four capsules round the yarn, a kinematic body that follows the engine's finger G
finger = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True, label="finger")
eye_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=0.0, mu=0.05)
u_ = norm(np.cross(dirOut, [0, 1, 0])); v_ = norm(np.cross(dirOut, u_)); hw = R_COL + 0.45
for (c, ax) in [(u_ * hw, v_), (-u_ * hw, v_), (v_ * hw, u_), (-v_ * hw, u_)]:
    z = np.array([0, 0, 1.0]); ax = norm(ax); axis = np.cross(z, ax); ang = math.acos(np.clip(np.dot(z, ax), -1, 1))
    q = wp.quat_from_axis_angle(wp.vec3(*norm(axis)), ang) if np.linalg.norm(axis) > 1e-6 else wp.quat_identity()
    builder.add_shape_capsule(body=finger, xform=wp.transform(wp.vec3(*((G0 + c) * S)), q), radius=0.4 * S,
                              half_height=(hw + 0.6) * S, cfg=eye_cfg)
for b_ in (hook, finger):
    builder.body_mass[b_] = 0.0; builder.body_inv_mass[b_] = 0.0
    builder.body_inertia[b_] = wp.mat33(0.0); builder.body_inv_inertia[b_] = wp.mat33(0.0)
builder.color()
model = builder.finalize()
pipe = newton.CollisionPipeline(model, contact_matching="sticky", broad_phase="sap")
contacts = pipe.contacts()
solver = newton.solvers.SolverVBD(model, iterations=int(os.environ.get("IT", 6)), rigid_compliant_alm=True,
                                  rigid_body_contact_buffer_size=256, rigid_contact_history=True)
s0, s1, ctrl = model.state(), model.state(), model.control()
dt = 1.0 / FPS / SUB
MASS = float(model.body_mass.numpy()[0])
print(f"yarn bodies {NY}, body mass {MASS:.2e}", flush=True)


def mat_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        w = math.sqrt(1 + t) * 2
        return np.array([(R[2, 1] - R[1, 2]) / w, (R[0, 2] - R[2, 0]) / w, (R[1, 0] - R[0, 1]) / w, 0.25 * w])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]])); j, k = (i + 1) % 3, (i + 2) % 3
    w = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4); q[i] = 0.25 * w; q[j] = (R[j, i] + R[i, j]) / w; q[k] = (R[k, i] + R[i, k]) / w; q[3] = (R[k, j] - R[j, k]) / w
    return q


def hook_xform(h):
    a = norm(h["a"]); d = norm(np.asarray(h["d"]) - a * np.dot(h["d"], a)); z = np.cross(a, d)
    q = mat_to_quat(np.stack([a, d, z], axis=1)); q /= np.linalg.norm(q)
    return np.concatenate([np.asarray(h["tip"], float) * S, q])


def lerp_hook(h0, h1, w):
    return dict(tip=np.asarray(h0["tip"]) + (np.asarray(h1["tip"]) - np.asarray(h0["tip"])) * w,
                a=np.asarray(h0["a"]) + (np.asarray(h1["a"]) - np.asarray(h0["a"])) * w,
                d=np.asarray(h0["d"]) + (np.asarray(h1["d"]) - np.asarray(h0["d"])) * w)


@wp.kernel
def set_body(b: int, xf: wp.transform, v: wp.spatial_vector, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
    body_q[b] = xf
    body_qd[b] = v


@wp.kernel
def yarn_forces(T0: wp.array[wp.vec3], T1: wp.array[wp.vec3], kk: wp.array[float], w: float, c: float,
                g: wp.vec3, dr: wp.vec3, drag: float, last: int, tension: wp.vec3,
                body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    x = wp.transform_get_translation(body_q[i])
    v = wp.spatial_top(body_qd[i])
    f = wp.vec3(0.0, 0.0, 0.0)
    k = kk[i]
    if k > 0.0:
        T = T0[i] * (1.0 - w) + T1[i] * w
        f = k * (T - x) - c * wp.sqrt(k) * v
    # beyond the finger the yarn runs to the ball: viscous drag (no whipping) and a steady pull on the end
    if wp.dot(x - g, dr) > 0.02:
        f = f - drag * v
    if i == last:
        f = f + tension
    body_f[i] = wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))


K_FAB = float(os.environ.get("K_FAB", 4.0))
K_HOOK = float(os.environ.get("K_HOOK", 3.0))
K_TAIL = float(os.environ.get("K_TAIL", 3.0))
K_WEAK = float(os.environ.get("K_WEAK", 0.0))
MISS = float(os.environ.get("MISS", 6.0))
DAMP = float(os.environ.get("DAMP", 2.0 * 0.9 * math.sqrt(MASS)))  # critical-ish: c*sqrt(k) = 2 zeta sqrt(k m)
TENSION = float(os.environ.get("TENSION", 4e-3))
DRAG = float(os.environ.get("DRAG", 0.02))
WIN = float(os.environ.get("WIN", 3.0))  # mm of material a body may slide along its path per frame
NWIN = 25
SOFT = float(os.environ.get("SOFT", 0.35))  # s: holds switch on/off smoothly
k_now = np.zeros(NY)


def track(T, x_mm, u_prev):
    """Each body finds its own place on the target strand: nearest point within +-WIN mm of material."""
    cand = u_prev[:, None] + np.linspace(-WIN, WIN, NWIN)[None, :]
    cand = np.clip(cand, 0.0, T["U"][-1])
    Q = interp_path(T, cand.ravel()).reshape(NY, NWIN, 3)
    dd = np.linalg.norm(Q - x_mm[:, None, :], axis=2)
    j = np.argmin(dd, axis=1)
    u = cand[np.arange(NY), j]
    # material order is preserved: no body overtakes its neighbour by more than a segment
    u = np.maximum.accumulate(np.minimum(u, np.maximum.accumulate(u)))
    return u, dd[np.arange(NY), j]


out = open(out_path, "w")
t_start = time.time()
prev_hx = hook_xform(TG[f_from]["hook"])
prev_g = np.zeros(3)
for f in range(f_from, NF - 1):
    A, B = TG[f], TG[f + 1]
    xq = s0.body_q.numpy()[:NY, :3] / S
    track_u, miss = track(A, xq, track_u)
    beyond_finger = track_u >= A["U"][-1] - 0.5
    ffu = np.interp(track_u, A["U"], A["FF"]); hku = np.interp(track_u, A["U"], A["HK"])
    want = np.where(ffu > 0.5, K_FAB, np.where(hku > 0.5, K_HOOK, K_WEAK))  # K_WEAK: the hands keep the rest of the yarn in place
    want[:3] = K_TAIL  # the other hand holds the tail end
    want[beyond_finger] = 0.0
    want[miss > MISS] = 0.0  # lost its place (target jumped away): leave it to physics rather than yank it
    a_ = 1 - math.exp(-1.0 / FPS / SOFT)
    k_now += (want - k_now) * a_
    k_now[want == 0] *= 0.6  # letting go is quicker than taking hold
    T0w = wp.array((interp_path(A, track_u) * S).astype(np.float32), dtype=wp.vec3)
    T1w = wp.array((interp_path(B, track_u) * S).astype(np.float32), dtype=wp.vec3)
    kw = wp.array(k_now.astype(np.float32), dtype=float)
    for sub in range(SUB):
        w = (sub + 1) / SUB
        hp = lerp_hook(A["hook"], B["hook"], w)
        hx = hook_xform(hp)
        vel = np.zeros(6); vel[:3] = (hx[:3] - prev_hx[:3]) / dt
        g = A["G"] + (B["G"] - A["G"]) * w
        gx = (g - G0) * S; gv = np.zeros(6); gv[:3] = (gx - prev_g) / dt
        for st in (s0, s1):
            wp.launch(set_body, dim=1, inputs=[hook, wp.transform(wp.vec3(*hx[:3]), wp.quat(*hx[3:])), wp.spatial_vector(*vel), st.body_q, st.body_qd])
            wp.launch(set_body, dim=1, inputs=[finger, wp.transform(wp.vec3(*gx), wp.quat_identity()), wp.spatial_vector(*gv), st.body_q, st.body_qd])
        prev_hx = hx; prev_g = gx
        s0.clear_forces()
        wp.launch(yarn_forces, dim=NY, inputs=[T0w, T1w, kw, w, DAMP, wp.vec3(*(g * S)), wp.vec3(*dirOut), DRAG, NY - 1,
                                               wp.vec3(*(dirOut * TENSION)), s0.body_q, s0.body_qd, s0.body_f])
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0
    c = s0.body_q.numpy()[:NY, :3] / S
    if not np.isfinite(c).all():
        print("NaN at frame", f); break
    # render up to a little past the finger (the rest is off-screen supply)
    gF = B["G"]
    nshow = int(np.searchsorted(track_u, B["U"][-1] - 0.5)) + 25
    nshow = min(NY, nshow)
    rr = np.where(np.interp(track_u[:nshow], B["U"], B["FF"]) > 0.5, 0.74, 0.92)
    out.write(json.dumps(dict(f=f + 1, pts=np.round(c[:nshow], 3).tolist(), u=np.round(track_u[:nshow], 2).tolist(),
                              r=np.round(rr, 2).tolist(), held=np.round(k_now[:nshow], 2).tolist())) + "\n")
    if (f - f_from) % 15 == 0:
        el = time.time() - t_start
        nh = int((k_now > 0.5).sum())
        print(f"frame {f + 1}/{NF}  {el / (f - f_from + 1):.2f} s/frame  held {nh}/{NY}  miss>6: {int((miss > 6).sum())}", flush=True)
        out.flush()
out.close()
print("done", f"{(time.time() - t_start) / max(1, NF - 1 - f_from):.2f} s/frame")
