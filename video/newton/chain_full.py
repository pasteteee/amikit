"""Chain-stitch test on NVIDIA Newton (VBD rods): real yarn made of capsules with friction, a rigid hook
with a real throat, a finger eyelet with yarn tension. Nothing is scripted on the yarn: it only moves where
the hook, the eyelet, gravity or neighbouring yarn push it.

Usage: python chain_newton.py <out.json> [n_stitches] [seconds_limit]
Output (mm): {fps, r, frames:[{pts, s, hook:{tip,a,d}}], cam}
"""
import json, math, os, sys, time
import numpy as np
import warp as wp
import newton

S = 0.01  # 1 mm -> 0.01 model units
R_Y = 0.95  # yarn radius, mm
SEG = 1.0  # segment length, mm
FPS = 30

out_path = sys.argv[1]
n_st = int(sys.argv[2]) if len(sys.argv) > 2 else 1
t_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9

wp.config.quiet = True
wp.init()


def norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


# ---------------- initial yarn (mm): anchored tail -> one loop round the shaft -> eyelet -> pulled end
X0 = float(os.environ.get("X0", 10.0))
RL = 1.25 * float(os.environ.get('HS', 1.0)) + R_Y + float(os.environ.get("SLACK", 0.15))
G = np.array([float(v) for v in os.environ.get("G", "-2,14,-12").split(",")])  # finger eyelet
dirOut = norm([-0.3, 0.6, -0.75])  # yarn leaves the eyelet towards the ball
pts = []
for y in np.arange(-26.0, -0.5, SEG):  # tail rises in front of the hook
    pts.append([X0, y, RL])
n_tail = len(pts)
TURNS = 1.25  # front -> over the top -> back -> under -> front -> top again, then up to the finger
turn_len = 2 * math.pi * RL * TURNS
m = int(turn_len / SEG)
for k in range(1, m + 1):
    th = 2 * math.pi * TURNS * k / m
    pts.append([X0 + 3.0 * k / m, RL * math.sin(th), RL * math.cos(th)])
a = np.array(pts[-1])
L = np.linalg.norm(G - a)
for k in range(1, int(L / SEG) + 1):
    pts.append(list(a + (G - a) * k / int(L / SEG)))
for k in range(1, int(os.environ.get("SUPPLY", 160))):
    pts.append(list(G + dirOut * k * SEG))
pts = np.array(pts)
if os.environ.get("SAVEPTS"): np.save(os.environ["SAVEPTS"], pts)

builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -float(os.environ.get('GRAV', 0.6)), 0.0))
builder.rigid_gap = 0.0
KE = float(os.environ.get("KE", 1e5))
cfg = newton.ModelBuilder.ShapeConfig(density=300.0, ke=KE, kd=0.0, mu=float(os.environ.get("MUY", 0.35)))
rod = newton.Rod([wp.vec3(*(p * S)) for p in pts], radius=R_Y * S)
rod.compute_frames()
bodies, joints = builder.add_rod(rod=rod, cfg=cfg, stretch_stiffness=float(os.environ.get("KS", 5e5)), bend_stiffness=float(os.environ.get("KB", 1e-4)), bend_damping=float(os.environ.get("KBD", 1e-4)),
                                 label="yarn", body_frame_origin="com")
n_body_yarn = len(bodies)
# a yarn is thicker than its segments are long: neighbours within ~2 diameters never collide with each other
for i in range(n_body_yarn):
    for j in range(i + 2, min(n_body_yarn, i + 4)):
        for sa in builder.body_shapes.get(bodies[i], []):
            for sb in builder.body_shapes.get(bodies[j], []):
                builder.add_shape_collision_filter_pair(int(sa), int(sb))
builder.rigid_gap = float(os.environ.get("GAP", 0.4)) * S
# anchor the tail (the other hand pinches it)
for b in bodies[: n_tail - 3]:
    builder.body_mass[b] = 0.0; builder.body_inv_mass[b] = 0.0
    builder.body_inertia[b] = wp.mat33(0.0); builder.body_inv_inertia[b] = wp.mat33(0.0)

# ---------------- hook: kinematic body, local x toward the tip (tip at x=0), y = throat side
hook_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=0.0, mu=float(os.environ.get("MUH", 0.25)))
hook = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True, label="hook")
qx = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)  # capsule Z -> local X


HOOK_XT = -2.05 * float(os.environ.get('HS', 1.0))
HOOK_BODY = -1 if os.environ.get('STATICHOOK') else hook
def cap(x0, x1, r, y=0.0):
    if os.environ.get('NOHOOK'): return
    builder.add_shape_capsule(body=HOOK_BODY, xform=wp.transform(wp.vec3((x0 + x1) / 2 * S * (-1 if HOOK_BODY == -1 else 1), y * S, 0.0), qx),
                              radius=r * S, half_height=(x1 - x0) / 2 * S, cfg=hook_cfg)


HS = float(os.environ.get('HS', 1.0))  # hook scale (1.0 = 2.5 mm hook)
cap(-70.0, -5.0 * HS, 1.25 * HS)  # shaft
cap(-5.2 * HS, -1.4 * HS, 0.6 * HS, -0.62 * HS)  # spine under the throat (groove on +y)
cap(-1.55 * HS, -1.15 * HS, 1.15 * HS)  # head
if not os.environ.get('NOBEAK'):  # the beak overhangs the throat so caught yarn cannot slip forward off the tip
    BX0, BY0, BX1, BY1, BR = -0.9 * HS, 0.35 * HS, -2.7 * HS, 1.05 * HS, 0.42 * HS
    L_ = math.hypot(BX1 - BX0, BY1 - BY0); ang = math.atan2(BY1 - BY0, BX1 - BX0)
    qb = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), ang), qx)
    builder.add_shape_capsule(body=HOOK_BODY, xform=wp.transform(wp.vec3((BX0 + BX1) / 2 * S, (BY0 + BY1) / 2 * S, 0.0), qb),
                              radius=BR * S, half_height=L_ / 2 * S, cfg=hook_cfg)
builder.body_mass[hook] = 0.0; builder.body_inv_mass[hook] = 0.0
builder.body_inertia[hook] = wp.mat33(0.0); builder.body_inv_inertia[hook] = wp.mat33(0.0)

# ---------------- finger eyelet: four static capsules round G, perpendicular to the yarn
eye_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=0.0, mu=0.05)
u = norm(np.cross(dirOut, [0, 1, 0])); v = norm(np.cross(dirOut, u)); hw = R_Y + 0.5
for (c, ax) in ([] if os.environ.get('NOEYE') else [(u * hw, v), (-u * hw, v), (v * hw, u), (-v * hw, u)]):
    z = np.array([0, 0, 1.0]); ax = norm(ax)
    axis = np.cross(z, ax); ang = math.acos(np.clip(np.dot(z, ax), -1, 1))
    q = wp.quat_from_axis_angle(wp.vec3(*norm(axis)), ang) if np.linalg.norm(axis) > 1e-6 else wp.quat_identity()
    builder.add_shape_capsule(body=-1, xform=wp.transform(wp.vec3(*((G + c) * S)), q), radius=0.45 * S,
                              half_height=(hw + 0.6) * S, cfg=eye_cfg)

# throat gate: an invisible bar that closes the throat once yarn lies in it (a deep, enclosed throat), so the
# caught yarn can only leave along the groove when the gate opens again
gate = builder.add_body(xform=wp.transform(wp.vec3(0.0, 50.0, 0.0), wp.quat_identity()), is_kinematic=True, label="gate")
GATE_Y = float(os.environ.get("GATE_Y", 2.45)) * HS + 0.95 * (1 - HS) * 0
builder.add_shape_capsule(body=gate, xform=wp.transform(wp.vec3(-3.0 * HS * S, GATE_Y * S, 0.0), qx), radius=0.35 * S,
                          half_height=1.9 * HS * S, cfg=hook_cfg)
for b_ in (gate,):
    builder.body_mass[b_] = 0.0; builder.body_inv_mass[b_] = 0.0
    builder.body_inertia[b_] = wp.mat33(0.0); builder.body_inv_inertia[b_] = wp.mat33(0.0)
builder.color()
model = builder.finalize()
pipe = newton.CollisionPipeline(model, contact_matching="sticky")
contacts = pipe.contacts()
solver = newton.solvers.SolverVBD(model, iterations=int(os.environ.get("IT", 8)), rigid_compliant_alm=True, rigid_body_contact_buffer_size=256,
                                  rigid_contact_history=True)
s0, s1, ctrl = model.state(), model.state(), model.control()
SUB = int(os.environ.get("SUB", 16))
dt = 1.0 / FPS / SUB
DAMP = float(os.environ.get('DAMP', 0.0))  # air drag on the yarn (1/s)
TENSION = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0e-4  # pull on the free end (model units)


# ---------------- hook choreography (poses in mm, world: tip, a = axis toward tip, d = throat side)
def P(tip, a, d):
    a = norm(a); d = norm(np.asarray(d) - a * np.dot(d, a))
    return dict(tip=np.asarray(tip, float), a=a, d=d)


HOME = P([0, 0, 0], [-1, 0, 0], [0, 0.3, -1])
moves = [(0.5, lambda st: HOME, None)]


REACHES = [0]
LOOP = [n_tail, n_tail + m]  # material range of the loop that is on the hook now


def update_loop(state):
    # after a stitch the loop on the hook is the yarn that was pulled through: find it round the shaft
    REACHES[0] += 1
    if REACHES[0] == 1: return
    c = yarn_pts_mm(state); p = cur["to"]
    rel = c - p["tip"]; lx = rel @ p["a"]; ly = rel @ p["d"]; lz = rel @ np.cross(p["a"], p["d"])
    on = np.where((lx > -14) & (lx < 0.5) & (np.hypot(ly, lz) < 1.25 * HS + R_Y + 1.6))[0]
    on = on[on >= LOOP[1] - 2]
    if len(on) >= 4:
        LOOP[0], LOOP[1] = int(on.min()), int(on.max()) + 1
    print(f"loop on hook: bodies {LOOP[0]}..{LOOP[1]}", flush=True)


def yarn_pts_mm(state):
    q = state.body_q.numpy()[:n_body_yarn]
    c = q[:, :3] / S
    return c


def working_point(state, yplane=float(os.environ.get("YW", 6.0))):
    # where the working yarn (loop -> finger) rises through the height y = yplane: there the hook takes it
    c = yarn_pts_mm(state)
    for i in range(LOOP[1] - 2, len(c) - 1):
        A, B = c[i], c[i + 1]
        if (A[1] - yplane) * (B[1] - yplane) <= 0:
            t = (yplane - A[1]) / ((B[1] - A[1]) or 1e-9)
            return A + (B - A) * t
    return c[LOOP[1] + 6]


for k in range(n_st):
    def reach(st):
        # pivot the hook about the loop on it so the shaft swings under the working yarn (yarn over)
        c = yarn_pts_mm(st); update_loop(st); Lc = c[LOOP[0]:LOOP[1]].mean(axis=0); E = c[LOOP[1]]
        A0 = cur["to"]; u = norm(G - E); f = float(os.environ.get("PIV", 0.7))
        a2 = norm((1 - f) * A0["a"] + f * u)
        Ltip = float(np.dot(A0["tip"] - Lc, A0["a"]))
        tip = Lc + a2 * Ltip
        thr = tip + a2 * HOOK_XT
        # the yarn line E + u s: closest point to the throat; the throat opens toward it
        sl = float(np.dot(thr - E, u)); Y = E + u * sl
        d2 = norm((Y - thr) - a2 * np.dot(Y - thr, a2))
        cur["reach_off"] = float(np.linalg.norm((Y - thr) - a2 * np.dot(Y - thr, a2)))
        print(f"reach: throat-to-yarn {cur['reach_off']:.2f} mm", flush=True)
        return P(tip, a2, d2)
    moves.append((0.9, reach, "reach"))
    # 2) move toward the yarn until it lies in the throat
    moves.append((0.4, lambda st: P(cur["to"]["tip"] + cur["to"]["d"] * (cur["reach_off"] - float(os.environ.get("SEAT", 0.95))), cur["to"]["a"], cur["to"]["d"]), "catch"))
    # 3) rotate the throat down (yarn wraps over the hook) and come back toward the loop axis
    moves.append((0.8, lambda st: P([0.0, 0.6, -0.4], [-1, 0, 0], [0, -0.5, -0.85]), "turn"))
    # 4) pull back through the loop on the hook
    def pull(st):
        c = yarn_pts_mm(st); xl = c[LOOP[0]:LOOP[1]][:, 0].max()
        return P([xl + float(os.environ.get("PULL_PAST", 1.0)), 0.4, 0.0], [-1, 0, 0], [0, -0.6, -0.8])
    moves.append((1.4, pull, "pull"))
    # 5) slide forward so the new loop sits on the shaft, throat up again
    # 5) turn the throat back up with the gate still closed: the new loop now hangs round the hook body
    moves.append((0.5, lambda st: P(cur["to"]["tip"], [-1, 0, 0], [0, 0.8, -0.6]), "rotate"))
    # 6) open the gate and slide forward so the new loop sits on the shaft
    moves.append((0.9, lambda st: P([0.0, 0.0, 0.0], [-1, 0, 0], [0, 0.3, -1]), "home"))
moves.append((0.5, lambda st: HOME, None))
T_total = min(sum(m_[0] for m_ in moves), t_limit)


def ease(x):
    return x * x * x * (x * (x * 6 - 15) + 10)


def lerp_pose(A, B, w):
    return P(A["tip"] + (B["tip"] - A["tip"]) * w, A["a"] + (B["a"] - A["a"]) * w, A["d"] + (B["d"] - A["d"]) * w)


cur = dict(i=-1, t0=0.0, frm=HOME, to=HOME)
gate_state = dict(closed=False)


def throat_count(state, p):
    c = yarn_pts_mm(state); rel = c - p["tip"]; lx = rel @ p["a"]; ly = rel @ p["d"]; lz = rel @ np.cross(p["a"], p["d"])
    return int(((lx > -4.5) & (lx < -0.5) & (ly > -0.5) & (np.hypot(ly, lz) < 3.2)).sum())


def pose_at(t, state):
    while cur["i"] + 1 < len(moves) and (cur["i"] < 0 or t >= cur["t0"] + moves[cur["i"]][0]):
        if cur["i"] >= 0:
            cur["t0"] += moves[cur["i"]][0]; cur["frm"] = cur["to"]
        cur["i"] += 1
        cur["to"] = moves[cur["i"]][1](state)
        tag = moves[cur["i"]][2]
        if tag == "turn" and not os.environ.get("NOGATE"):
            gate_state["closed"] = throat_count(state, cur["frm"]) > 0
            print("gate closed" if gate_state["closed"] else "gate: nothing in the throat", flush=True)
        if tag == "home": gate_state["closed"] = False
        print(f"t={t:.2f} move {cur['i']} {moves[cur['i']][2]} -> tip {np.round(cur['to']['tip'], 2)}", flush=True)
    d = moves[cur["i"]][0]
    return lerp_pose(cur["frm"], cur["to"], ease(min(1.0, (t - cur["t0"]) / d)))


def mat_to_quat(R):
    """Rotation matrix -> quaternion (x, y, z, w), Shepperd's method (robust for any angle)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        w = math.sqrt(1 + t) * 2
        return np.array([(R[2, 1] - R[1, 2]) / w, (R[0, 2] - R[2, 0]) / w, (R[1, 0] - R[0, 1]) / w, 0.25 * w])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    w = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4); q[i] = 0.25 * w; q[j] = (R[j, i] + R[i, j]) / w; q[k] = (R[k, i] + R[i, k]) / w; q[3] = (R[k, j] - R[j, k]) / w
    return q


def pose_xform(p):
    a, d = p["a"], p["d"]; z = np.cross(a, d)
    R = np.stack([a, d, z], axis=1)  # columns: local x,y,z in world
    q = mat_to_quat(R)
    return np.concatenate([p["tip"] * S, q / np.linalg.norm(q)])


@wp.kernel
def set_body(b: int, xf: wp.transform, v: wp.spatial_vector, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
    body_q[b] = xf
    body_qd[b] = v


@wp.kernel
def damp_yarn(n: int, k: float, body_qd: wp.array[wp.spatial_vector]):
    i = wp.tid()
    if i < n:
        body_qd[i] = body_qd[i] * k


@wp.kernel
def set_force(b: int, f: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    body_f[b] = wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))


bq = s0.body_q.numpy()
c0 = bq[:n_body_yarn, :3] / S
last = n_body_yarn - 1
frames = []
t = 0.0
prev_x = pose_xform(HOME)
t_start = time.time()
nF = int(T_total * FPS) + 1
for f in range(nF):
    for sub in range(SUB):
        t = (f * SUB + sub) * dt
        p = HOME if os.environ.get('STATIC') else pose_at(t, s0)
        x = pose_xform(p)
        vel = np.zeros(6); vel[:3] = (x[:3] - prev_x[:3]) / dt
        for st in (s0, s1):
            wp.launch(set_body, dim=1, inputs=[hook, wp.transform(wp.vec3(*x[:3]), wp.quat(*x[3:])), wp.spatial_vector(*vel), st.body_q, st.body_qd])
            gx = x[:3] if gate_state["closed"] else np.array([0.0, 50.0, 0.0])
            wp.launch(set_body, dim=1, inputs=[gate, wp.transform(wp.vec3(*gx), wp.quat(*x[3:])), wp.spatial_vector(*(vel if gate_state["closed"] else np.zeros(6))), st.body_q, st.body_qd])
        prev_x = x
        s0.clear_forces()
        wp.launch(set_force, dim=1, inputs=[last, wp.vec3(*(dirOut * TENSION)), s0.body_f])
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0
        if DAMP > 0: wp.launch(damp_yarn, dim=n_body_yarn, inputs=[n_body_yarn, math.exp(-DAMP * dt), s0.body_qd])
    c = yarn_pts_mm(s0)
    if not np.isfinite(c).all():
        print("NaN at frame", f); break
    p = pose_at(t, s0)
    frames.append(dict(pts=np.round(c, 3).tolist(), s=[i * SEG for i in range(len(c))],
                       hook=dict(tip=p["tip"].tolist(), a=p["a"].tolist(), d=p["d"].tolist())))
    # yarn sitting in the throat groove: hook-local x in [-3.6, -1.0], on the throat side, close to the spine
    rel = c - p["tip"]; lx = rel @ p["a"]; ly = rel @ p["d"]; lz = rel @ np.cross(p["a"], p["d"])
    in_thr = int(((lx > -3.6) & (lx < -1.0) & (ly > 0.2) & (np.hypot(ly, lz) < 2.4)).sum())
    if os.environ.get('THR') and f % 3 == 0:
        ti = np.where((lx > -3.6) & (lx < -1.0) & (ly > 0.2) & (np.hypot(ly, lz) < 2.4))[0]
        lp = slice(n_tail, n_tail + m)
        print(f"f{f} move {cur['i']} throat {in_thr} idx {ti.tolist()} loop lx {lx[lp].min():.1f}..{lx[lp].max():.1f} rho {np.hypot(ly, lz)[lp].min():.1f}..{np.hypot(ly, lz)[lp].max():.1f}")
    if f % 5 == 0 and os.environ.get('DBG'):
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1); print('  seglen', seg.min().round(3), seg.max().round(3), 'disp from init', np.abs(c - c0).max().round(2), 'argmax', int(np.argmax(np.linalg.norm(c - c0, axis=1))))
        loop = c[n_tail:n_tail + m]; rho = np.hypot(loop[:, 1], loop[:, 2])
        nc = int(contacts.rigid_contact_count.numpy()[0]); s0_ = contacts.rigid_contact_shape0.numpy()[:nc]; s1_ = contacts.rigid_contact_shape1.numpy()[:nc]
        nys = n_body_yarn
        if f == 0: print('  pairs', sorted(set((int(min(a_, b_)), int(max(a_, b_))) for a_, b_ in zip(s0_, s1_)))[:30], 'n_tail', n_tail, 'm', m)
        print('  hook pairs', int((((s0_ >= nys) & (s0_ < nys + 3)) | ((s1_ >= nys) & (s1_ < nys + 3))).sum()), 'eye pairs', int(((s0_ >= nys + 4) | (s1_ >= nys + 4)).sum()), 'shapes total', model.shape_count, 'yarn shapes', nys, 'max', contacts.rigid_contact_shape0.shape)
        print(f"f{f} contacts {int(contacts.rigid_contact_count.numpy()[0])} loop rho min {rho.min():.2f} max {rho.max():.2f} x {loop[:,0].min():.1f}..{loop[:,0].max():.1f} hook {np.round(s0.body_q.numpy()[hook][:3]/S,2)}")
    if f % 15 == 0:
        el = time.time() - t_start
        print(f"frame {f}/{nF}  {el / (f + 1):.2f} s/frame", flush=True)
json.dump(dict(fps=FPS, r=R_Y, frames=frames, cam=dict(pos=[8, 8, 36], look=[4, 2, 0], fov=42)), open(out_path, "w"))
print("done", len(frames), "frames", f"{(time.time() - t_start) / max(1, len(frames)):.2f} s/frame")
