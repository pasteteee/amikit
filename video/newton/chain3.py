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
RL = 1.25 + R_Y + float(os.environ.get("SLACK", 0.15))
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
for b in bodies[: n_tail - int(os.environ.get("PIN0", 3))]:
    builder.body_mass[b] = 0.0; builder.body_inv_mass[b] = 0.0
    builder.body_inertia[b] = wp.mat33(0.0); builder.body_inv_inertia[b] = wp.mat33(0.0)

# ---------------- hook: kinematic body, local x toward the tip (tip at x=0), y = throat side
hook_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=0.0, mu=float(os.environ.get("MUH", 0.25)))
hook = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True, label="hook")
qx = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)  # capsule Z -> local X


HOOK_XT = -2.05
HOOK_BODY = -1 if os.environ.get('STATICHOOK') else hook
def cap(x0, x1, r, y=0.0):
    if os.environ.get('NOHOOK'): return
    builder.add_shape_capsule(body=HOOK_BODY, xform=wp.transform(wp.vec3((x0 + x1) / 2 * S * (-1 if HOOK_BODY == -1 else 1), y * S, 0.0), qx),
                              radius=r * S, half_height=(x1 - x0) / 2 * S, cfg=hook_cfg)


cap(-70.0, -5.0, 1.25)  # shaft
cap(-5.2, -1.4, 0.6, -0.62)  # spine under the throat (groove on +y)
cap(-1.55, -1.15, 1.15)  # head
if not os.environ.get('NOBEAK'):  # the beak overhangs the throat so caught yarn cannot slip forward off the tip
    BX0, BY0, BX1, BY1, BR = -0.9, 0.35, -2.7, 1.05, 0.42
    L_ = math.hypot(BX1 - BX0, BY1 - BY0); ang = math.atan2(BY1 - BY0, BX1 - BX0)
    qb = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), ang), qx)
    builder.add_shape_capsule(body=HOOK_BODY, xform=wp.transform(wp.vec3((BX0 + BX1) / 2 * S, (BY0 + BY1) / 2 * S, 0.0), qb),
                              radius=BR * S, half_height=L_ / 2 * S, cfg=hook_cfg)
builder.body_mass[hook] = 0.0; builder.body_inv_mass[hook] = 0.0
builder.body_inertia[hook] = wp.mat33(0.0); builder.body_inv_inertia[hook] = wp.mat33(0.0)

# ---------------- finger eyelet: four static capsules round G, perpendicular to the yarn
# the finger (eyelet) is a kinematic body whose frame sits on the hook axis line: it orbits the hook to lay the yarn over
finger = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True, label="finger")
builder.body_mass[finger] = 0.0; builder.body_inv_mass[finger] = 0.0
builder.body_inertia[finger] = wp.mat33(0.0); builder.body_inv_inertia[finger] = wp.mat33(0.0)
eye_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=0.0, mu=0.05)
u = norm(np.cross(dirOut, [0, 1, 0])); v = norm(np.cross(dirOut, u)); hw = R_Y + 0.5
for (c, ax) in ([] if os.environ.get('NOEYE') else [(u * hw, v), (-u * hw, v), (v * hw, u), (-v * hw, u)]):
    z = np.array([0, 0, 1.0]); ax = norm(ax)
    axis = np.cross(z, ax); ang = math.acos(np.clip(np.dot(z, ax), -1, 1))
    q = wp.quat_from_axis_angle(wp.vec3(*norm(axis)), ang) if np.linalg.norm(axis) > 1e-6 else wp.quat_identity()
    builder.add_shape_capsule(body=finger, xform=wp.transform(wp.vec3(*((G + c) * S)), q), radius=0.45 * S,
                              half_height=(hw + 0.6) * S, cfg=eye_cfg)

# throat gate: an invisible bar that closes the throat once yarn lies in it (a deep, enclosed throat), so the
# caught yarn can only leave along the groove when the gate opens again
gate = builder.add_body(xform=wp.transform(wp.vec3(0.0, 50.0, 0.0), wp.quat_identity()), is_kinematic=True, label="gate")
GATE_Y = float(os.environ.get("GATE_Y", 2.45))
builder.add_shape_capsule(body=gate, xform=wp.transform(wp.vec3(float(os.environ.get("GATE_X", -3.0)) * S, GATE_Y * S, 0.0), qx), radius=0.35 * S,
                          half_height=float(os.environ.get("GATE_H", 1.9)) * S, cfg=hook_cfg)
if os.environ.get("GATE_CAP"):
    # the latch closes onto the tip: the eye is closed, the caught yarn cannot slide forward off the head
    CX = float(os.environ.get("GATE_CAP"))
    qy = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2)  # capsule Z -> local Y
    builder.add_shape_capsule(body=gate, xform=wp.transform(wp.vec3(CX * S, GATE_Y / 2 * S, 0.0), qy), radius=0.35 * S,
                              half_height=GATE_Y / 2 * S, cfg=hook_cfg)
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
PIN = dict(upto=n_tail - int(os.environ.get("PIN0", 3)))
from newton import BodyFlags, ModelFlags


def pin_set(ids):
    ids = [i for i in ids if 0 <= i < n_body_yarn - 40]
    if not ids: return
    fl = model.body_flags.numpy(); fl[ids] |= int(BodyFlags.KINEMATIC); model.body_flags.assign(fl)
    for st_ in (s0, s1):
        qd = st_.body_qd.numpy(); qd[ids] = 0; st_.body_qd.assign(qd)
    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
    print(f"pinned (hand pinch) {ids}", flush=True)


def unpin_set(ids):
    fl = model.body_flags.numpy(); fl[ids] &= ~int(BodyFlags.KINEMATIC); model.body_flags.assign(fl)
    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)


MAG = dict(ids=[], loc=[], rot=[])


def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def hook_R(p):
    a, d = p["a"], p["d"]; return np.stack([a, d, np.cross(a, d)], axis=1)


def magnet_catch(state, p):
    # the throat takes the yarn that lies in it: those bits now travel with the hook (kinematic) until released
    c = yarn_pts_mm(state); R = hook_R(p)
    thr = p["tip"] + R @ np.array([HOOK_XT, float(os.environ.get("MAG_Y", 1.55)), 0.0])
    i1 = cur.get("loop", (0, n_tail + m))[1]
    cand = np.arange(i1 + 3, n_body_yarn - 40)
    dd = np.linalg.norm(c[cand] - thr, axis=1); k = int(cand[np.argmin(dd)])
    print(f"magnet: nearest yarn bit {k} at {dd.min():.2f} mm from the throat", flush=True)
    if dd.min() > float(os.environ.get("MAG_R", 3.0)): return False
    w = int(os.environ.get("MAG_W", 1)); ids = list(range(k - w, k + w + 1))
    bq = state.body_q.numpy(); qh = mat_to_quat(R); qh /= np.linalg.norm(qh)
    MAG["ids"] = ids
    MAG["loc"] = [R.T @ (bq[i, :3] / S - p["tip"]) for i in ids]
    MAG["rot"] = [quat_mul(quat_conj(qh), bq[i, 3:7]) for i in ids]
    pin_set(ids)
    return True


def magnet_release():
    if MAG["ids"]:
        unpin_set(MAG["ids"]); print(f"magnet: released {MAG['ids']}", flush=True); MAG["ids"] = []


def pin_upto(idx):
    idx = min(idx, n_body_yarn - 40)
    if idx <= PIN["upto"]: return
    fl = model.body_flags.numpy(); fl[PIN["upto"]:idx] |= int(BodyFlags.KINEMATIC); model.body_flags.assign(fl)
    for st_ in (s0, s1):
        qd = st_.body_qd.numpy(); qd[PIN["upto"]:idx] = 0; st_.body_qd.assign(qd)
    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
    print(f"pinned yarn bodies {PIN['upto']}..{idx}", flush=True)
    PIN["upto"] = idx


SUB = int(os.environ.get("SUB", 16))
dt = 1.0 / FPS / SUB
TENSION = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0e-4  # pull on the free end (model units)


# ---------------- hook choreography (poses in mm, world: tip, a = axis toward tip, d = throat side)
def P(tip, a, d):
    a = norm(a); d = norm(np.asarray(d) - a * np.dot(d, a))
    return dict(tip=np.asarray(tip, float), a=a, d=d, fx=FG["x"], fr=FG["r"], fth=FG["th"])


G_R = float(math.hypot(G[1], G[2])); G_TH = float(math.atan2(G[2], G[1]))
FG = dict(x=float(G[0]), r=G_R, th=G_TH)  # finger eyelet in cylindrical coords round the hook axis line (y = z = 0)
HOME = P([0, 0, 0], [-1, 0, 0], [0, 0.3, -1])
moves = [(0.5, lambda st: HOME, None)]


def yarn_pts_mm(state):
    q = state.body_q.numpy()[:n_body_yarn]
    c = q[:, :3] / S
    return c


def working_point(state, yplane=float(os.environ.get("YW", 6.0))):
    # where the working yarn (loop -> finger) rises through the height y = yplane: there the hook takes it
    c = yarn_pts_mm(state)
    for i in range(n_tail + m - 2, len(c) - 1):
        A, B = c[i], c[i + 1]
        if (A[1] - yplane) * (B[1] - yplane) <= 0:
            t = (yplane - A[1]) / ((B[1] - A[1]) or 1e-9)
            return A + (B - A) * t
    return c[n_tail + m + 6]


BASE = dict(x=0.0)


def loop_on_hook(st, p):
    # the contiguous run of free yarn wrapped round the shaft behind the throat = the loop on the hook
    c = yarn_pts_mm(st); rel = c - p["tip"]; lx = rel @ p["a"]
    rho = np.linalg.norm(rel - np.outer(lx, p["a"]), axis=1)
    on = (rho < 3.8) & (lx < -3.0) & (lx > -25)
    idx = np.where(on)[0]
    idx = idx[idx >= PIN["upto"]]
    if len(idx) == 0:
        print("WARN: no loop on hook", flush=True); return n_tail, n_tail + m
    runs = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    r = max(runs, key=len)
    return int(r[0]), int(r[-1]) + 1


HTH = math.radians(float(os.environ.get("HOME_TH", -73)))
TTH = math.radians(float(os.environ.get("TURN_TH", -120)))


def home_pose():
    return P([BASE["x"], 0, 0], [-1, 0, 0], [0, math.cos(HTH), math.sin(HTH)])


for k in range(n_st):
    def reach(st):
        # pivot the hook about the loop on it so the shaft swings under the working yarn (yarn over)
        c = yarn_pts_mm(st); A0 = cur["to"]
        i0, i1 = loop_on_hook(st, A0); cur["loop"] = (i0, i1)
        Lc = c[i0:i1].mean(axis=0); E = c[min(i1, len(c) - 1)]
        u = norm(G - E); f = float(os.environ.get("PIV", 0.7))
        a2 = norm((1 - f) * A0["a"] + f * u)
        Ltip = float(np.dot(A0["tip"] - Lc, A0["a"]))
        tip = Lc + a2 * Ltip
        thr = tip + a2 * HOOK_XT
        sl = float(np.dot(thr - E, u)); Y = E + u * sl
        d2 = norm((Y - thr) - a2 * np.dot(Y - thr, a2))
        cur["reach_off"] = float(np.linalg.norm((Y - thr) - a2 * np.dot(Y - thr, a2)))
        print(f"reach: loop {i0}..{i1} centre {np.round(Lc,1)}, throat-to-yarn {cur['reach_off']:.2f} mm", flush=True)
        return P(tip, a2, d2)
    if os.environ.get("YO", "finger") == "finger":
        # yarn over by the left hand: the finger carries the yarn once round the hook just behind the throat
        WR = float(os.environ.get("WRAP", -1.0)) * 2 * math.pi
        FR = float(os.environ.get("FR", 8.0))
        def approach(st):
            i0, i1 = loop_on_hook(st, cur["to"]); cur["loop"] = (i0, i1)
            print(f"wrap: loop {i0}..{i1}", flush=True)
            if os.environ.get("PINLEG"):
                k0 = int(os.environ.get("PINLEG")); pin_set(list(range(i1 + k0, i1 + k0 + 2)))
            FG["x"] = BASE["x"] + float(os.environ.get("FX", 2.5)); FG["r"] = FR; return home_pose()
        moves.append((0.6, approach, "approach"))
        NW = 6
        for q in range(NW):
            def wr(st):
                FG["th"] += WR / NW; return home_pose()
            moves.append((float(os.environ.get("WT", 1.5)) / NW, wr, "wrap"))
        def retreat(st):
            FG["r"] = G_R; FG["x"] = float(G[0]) + BASE["x"]; return home_pose()
        moves.append((0.5, retreat, "retreat"))
    else:
        moves.append((0.9, reach, "reach"))
        moves.append((0.4, lambda st: P(cur["to"]["tip"] + cur["to"]["d"] * (cur["reach_off"] - float(os.environ.get("SEAT", 0.95))), cur["to"]["a"], cur["to"]["d"]), "catch"))
    if os.environ.get("SWEEPGATE"):
        def aim(st):
            # turn the throat to face the yarn just laid over the hook
            c = yarn_pts_mm(st); A = cur["to"]; i0, i1 = cur["loop"]
            rel = c - A["tip"]; lx = rel @ A["a"]; perp = rel - np.outer(lx, A["a"]); rho = np.linalg.norm(perp, axis=1)
            sel = (rho < 4.0) & (lx > float(os.environ.get("AIM_LX", -11))) & (lx < 0.5) & (np.arange(len(c)) > i1 + 1)
            if sel.sum() == 0:
                print("aim: no yarn near the throat", flush=True); dvec = A["d"]
            else:
                dvec = norm(perp[sel].mean(axis=0)); print(f"aim: {int(sel.sum())} yarn bits near the throat", flush=True)
                if os.environ.get("MAGNET"):
                    ids_ = np.where(sel)[0]; k_ = ids_[np.argmax(lx[ids_])]
                    dvec = norm(perp[k_]); shift = lx[k_] - HOOK_XT
                    print(f"aim: yarn bit {k_} at lx {lx[k_]:.1f}, shifting the hook {shift:.1f} mm", flush=True)
                    cur["aim_d"] = dvec
                    return P(A["tip"] + A["a"] * shift, [-1, 0, 0], dvec)
            cur["aim_d"] = dvec
            return P([BASE["x"], 0, 0], [-1, 0, 0], dvec)
        moves.append((0.6, aim, "aim"))
        if os.environ.get("MAGNET"):
            moves.append((0.15, lambda st: cur["to"], "mcatch"))
            moves.append((0.7, lambda st: P(cur["to"]["tip"], [-1, 0, 0], [0, math.cos(TTH), math.sin(TTH)]), "turn"))
        elif not os.environ.get("DYNSWEEP"): moves.append((0.45, lambda st: cur["to"], "sweep"))
    else:
        moves.append((0.8, lambda st: P([BASE["x"], 0.6, -0.4], [-1, 0, 0], [0, math.cos(TTH), math.sin(TTH)]), "turn"))
    def pull(st):
        i0, i1 = cur["loop"]; c = yarn_pts_mm(st); xl = c[i0:i1][:, 0].max()
        cur["pull_x"] = xl + float(os.environ.get("PULL_PAST", 1.0))
        return P([cur["pull_x"], 0.0, 0.0], [-1, 0, 0], cur.get("aim_d", [0, -0.6, -0.8]))
    if os.environ.get("PULLSEEK"):
        pass
        def pull0(st):
            cur["pulldone"] = False; return pull(st)
        moves.append((float(os.environ.get("PULL_T", 1.0)), pull0, "pull"))
        def pull_more(st):
            # keep drawing back until the old loop has slid off the tip
            i0, i1 = cur["loop"]; c = yarn_pts_mm(st); A = cur["to"]
            rel = c[i0:i1] - A["tip"]; lx = rel @ A["a"]; rho = np.linalg.norm(rel - np.outer(lx, A["a"]), axis=1)
            on = int(((rho < 3.4) & (lx < 0.3)).sum())
            if on == 0 or cur["pulldone"]:
                if not cur["pulldone"]: print(f"old loop off the tip at x={A['tip'][0]:.1f}", flush=True)
                cur["pulldone"] = True; return A
            cur["pull_x"] = A["tip"][0] + 1.5
            return P([cur["pull_x"], 0.4, 0.0], A["a"], A["d"])
        for _ in range(int(os.environ.get("PULLSEEK"))):
            moves.append((0.12, pull_more, "pull"))
    else:
        moves.append((1.4, pull, "pull"))
    if os.environ.get("ROT_TH") is None:
        moves.append((0.5, lambda st: P(cur["to"]["tip"], [-1, 0, 0], [0, 0.8, -0.6]), "rotate"))
    elif os.environ.get("ROT_TH") != "none":
        def rot(st):
            # turn the throat by ROT_TH degrees about the hook axis
            A = cur["to"]; z_ = np.cross(A["a"], A["d"]); th = math.radians(float(os.environ["ROT_TH"]))
            return P(A["tip"], A["a"], A["d"] * math.cos(th) + z_ * math.sin(th))
        moves.append((0.5, rot, "rotate"))
    def home(st):
        # the left hand pinches the finished chain just below the hook; the hook slides the new loop up the shaft
        i0, i1 = cur["loop"]
        if not os.environ.get("NOPIN"): pin_upto(i1 + int(os.environ.get("PIN_EXTRA", 2)))
        BASE["x"] = cur["pull_x"] - float(os.environ.get("BACK", 10.0))
        return home_pose()
    if os.environ.get("FWD"):
        def fwd1(st):
            # latch still closed: bring the head back over the new knot so the caught yarn drapes over the hook
            i0, i1 = cur["loop"]; c = yarn_pts_mm(st)
            if not os.environ.get("NOPIN"): pin_upto(i1 + int(os.environ.get("PIN_EXTRA", 2)))
            cur["knot_x"] = float(c[i0:i1][:, 0].mean())
            A = cur["to"]
            return P([cur["knot_x"] + float(os.environ.get("FWD1", -1.0)), 0.0, 0.0], A["a"], A["d"])
        moves.append((0.6, fwd1, "fwd1"))
        moves.append((0.25, lambda st: cur["to"], "open"))
        def fwd2(st):
            BASE["x"] = cur["knot_x"] - float(os.environ.get("BACK", 10.0)); return home_pose()
        moves.append((0.9, fwd2, "fwd2"))
    else:
        moves.append((0.9, home, "home"))
moves.append((0.5, lambda st: home_pose(), None))
T_total = min(sum(m_[0] for m_ in moves), t_limit)


def ease(x):
    return x * x * x * (x * (x * 6 - 15) + 10)


def lerp_pose(A, B, w):
    p = P(A["tip"] + (B["tip"] - A["tip"]) * w, A["a"] + (B["a"] - A["a"]) * w, A["d"] + (B["d"] - A["d"]) * w)
    for k_ in ("fx", "fr", "fth"): p[k_] = A[k_] + (B[k_] - A[k_]) * w
    return p


def fpos(p):
    return np.array([p["fx"], p["fr"] * math.cos(p["fth"]), p["fr"] * math.sin(p["fth"])])


cur = dict(i=-1, t0=0.0, frm=HOME, to=HOME)
gate_state = dict(closed=False)
SWEEP = float(os.environ.get('SWEEP', 5.0))


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
        if tag == "turn" and not os.environ.get("NOGATE") and not os.environ.get("DYNGATE"):
            gate_state["closed"] = throat_count(state, cur["frm"]) > 0
            print("gate closed" if gate_state["closed"] else "gate: nothing in the throat", flush=True)
        if tag in ("home", "open"): gate_state["closed"] = False; gate_state.pop("t_close", None); magnet_release()
        if tag == "mcatch": magnet_catch(state, cur["frm"])
        if tag == "sweep":
            gate_state["closed"] = True; gate_state["t_close"] = t; print(f"t={t:.2f} latch sweeps the yarn into the throat", flush=True)
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


def rotx(phi):
    c_, s_ = math.cos(phi), math.sin(phi)
    return np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]])


@wp.kernel
def set_body(b: int, xf: wp.transform, v: wp.spatial_vector, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
    body_q[b] = xf
    body_qd[b] = v


@wp.kernel
def supply_forces(last: int, g: wp.vec3, dr: wp.vec3, c: float, tension: wp.vec3, body_q: wp.array[wp.transform],
                  body_qd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    # yarn beyond the finger runs off to the ball: viscous drag there (no whipping), steady tension on the end
    i = wp.tid()
    x = wp.transform_get_translation(body_q[i])
    f = wp.vec3(0.0, 0.0, 0.0)
    if wp.dot(x - g, dr) > 0.03:
        v = wp.spatial_top(body_qd[i])
        f = -c * v
    if i == last:
        f = f + tension
    body_f[i] = wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def set_force(b: int, f: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    body_f[b] = wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))


bq = s0.body_q.numpy()
c0 = bq[:n_body_yarn, :3] / S
last = n_body_yarn - 1
frames = []
t = 0.0
prev_x = pose_xform(HOME)
prev_f = [np.zeros(3)]
DRAG = float(os.environ.get('DRAG', 0.02))
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
            off = 0.0
            if gate_state["closed"] and "t_close" in gate_state:
                off = SWEEP * (1 - ease(min(1.0, (t - gate_state["t_close"]) / 0.35)))
            gx = x[:3] + p["d"] * off * S if gate_state["closed"] else np.array([0.0, 50.0, 0.0])
            wp.launch(set_body, dim=1, inputs=[gate, wp.transform(wp.vec3(*gx), wp.quat(*x[3:])), wp.spatial_vector(*(vel if gate_state["closed"] else np.zeros(6))), st.body_q, st.body_qd])
        if MAG["ids"]:
            R_ = hook_R(p); qh_ = x[3:]
            for i_, lp_, rq_ in zip(MAG["ids"], MAG["loc"], MAG["rot"]):
                wpos = (p["tip"] + R_ @ lp_) * S; wq = quat_mul(qh_, rq_); wq /= np.linalg.norm(wq)
                for st in (s0, s1):
                    wp.launch(set_body, dim=1, inputs=[i_, wp.transform(wp.vec3(*wpos), wp.quat(*wq)), wp.spatial_vector(*vel), st.body_q, st.body_qd])
        prev_x = x
        fx_ = (fpos(p) - G) * S; fq = np.array([0.0, 0.0, 0.0, 1.0])
        fv = np.zeros(6); fv[:3] = (fx_ - prev_f[0]) / dt
        for st in (s0, s1):
            wp.launch(set_body, dim=1, inputs=[finger, wp.transform(wp.vec3(*fx_), wp.quat(*fq)), wp.spatial_vector(*fv), st.body_q, st.body_qd])
        prev_f[0] = fx_
        s0.clear_forces()
        wp.launch(supply_forces, dim=n_body_yarn, inputs=[last, wp.vec3(*(fpos(p) * S)), wp.vec3(*dirOut), DRAG, wp.vec3(*(dirOut * TENSION)), s0.body_q, s0.body_qd, s0.body_f])
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0
    c = yarn_pts_mm(s0)
    if not np.isfinite(c).all():
        print("NaN at frame", f); break
    p = pose_at(t, s0)
    frames.append(dict(pts=np.round(c, 3).tolist(), s=[i * SEG for i in range(len(c))],
                       hook=dict(tip=p["tip"].tolist(), a=p["a"].tolist(), d=p["d"].tolist()), finger=fpos(p).tolist()))
    # yarn sitting in the throat groove: hook-local x in [-3.6, -1.0], on the throat side, close to the spine
    rel = c - p["tip"]; lx = rel @ p["a"]; ly = rel @ p["d"]; lz = rel @ np.cross(p["a"], p["d"])
    in_thr = int(((lx > -3.6) & (lx < -1.0) & (ly > 0.2) & (np.hypot(ly, lz) < 2.4)).sum())
    if os.environ.get("DYNSWEEP") and not gate_state["closed"] and moves[cur["i"]][2] in ("pull",):
        new = np.arange(len(c)) >= cur.get("loop", (0, n_tail + m))[1] + 2
        rho_ = np.hypot(ly, lz)
        hit = (lx > -4.6) & (lx < -0.8) & (rho_ < 3.6) & (ly > -0.3) & new
        if hit.sum() > 0:
            gate_state["closed"] = True; gate_state["t_close"] = t; print(f"f{f} latch sweeps {int(hit.sum())} yarn bits into the throat", flush=True)
    if os.environ.get("DYNGATE") and not gate_state["closed"] and moves[cur["i"]][2] in ("turn", "pull"):
        new = np.arange(len(c)) >= cur.get("loop", (0, n_tail + m))[1] + 2
        if int(((lx > -3.6) & (lx < -1.0) & (ly > 0.2) & (np.hypot(ly, lz) < float(os.environ.get("THR_R", 2.4))) & new).sum()) > 0:
            gate_state["closed"] = True; print(f"f{f} gate closed on caught yarn", flush=True)
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
