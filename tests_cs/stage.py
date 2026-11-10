"""Staged development driver:  python tests_cs/stage.py A   /  B  ...  each stage loads the
state the previous one saved."""
import sys, math, pickle, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim.ksim import KSim
from crochetsim.scene import magic_ring
from crochetsim.hook import frame
from crochetsim.crocheter import Crocheter, unit
from crochetsim.render import Camera
from dbgview import sheet
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
h, r, rho = 0.8, 1.0, 7.5
EY = np.array([0, 1.0, 0])
def ER(th): return np.array([math.cos(th), 0, math.sin(th)])
def ET(th): return np.array([-math.sin(th), 0, math.cos(th)])
CAMS = [Camera([-14, -70, 28], [2, 2, 5], fov=26), Camera([-60, 12, 45], [2, 4, 6], fov=26)]

def save(sim, name):
    fr = sim.frames; sim.frames = []
    pickle.dump(sim, open(SC + name + '.pkl', 'wb')); sim.frames = fr
    pickle.dump(fr, open(SC + name + '_frames.pkl', 'wb'))
def load(name):
    return pickle.load(open(SC + name + '.pkl', 'rb'))
def report(sim, tag, i0=0, nfr=8, cams=None, size=(400, 400)):
    cams = cams or CAMS
    print(tag, 'n', sim.n, 'nsub', sim.nsub, 'worst', {k: float('%.2g' % v) for k, v in sim.worst.items()}, 'bad', len(sim.log), 'cap', sim.cap, 'nsplit', sim.nsplit, 'newton/sub %.1f' % (sim.nnewton / max(1, sim.nsub)))
    sheet(sim, SC + 'st_' + tag + '.png', cams, nfr=nfr, i0=i0, size=size)

stage = sys.argv[1]
t0 = time.time()
if stage == 'A':
    th = math.radians(38)
    F = np.array([-17.0, 36.0, 38.0])
    X0, mk = magic_ring(h, r, rho=rho, feeder=F)
    sim = KSim(X0[:-1], h, r, feeder=F, hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[3, -30, 2])
    sim.mk = mk; sim.verbose = 500
    ang = np.degrees(np.arctan2(X0[:, 2], X0[:, 0])) % 360
    ringn = np.arange(mk['ring0'], mk['ring1'])
    sim.pin(0)
    sim.pin(np.arange(mk['ring0'] - 3, mk['ring0'] + 3)); sim.pin(np.arange(mk['ring1'] - 3, mk['ring1'] + 4))
    sim.pin(ringn)      # fingers hold the whole ring
    sim.wait_taut(); print('settled', sim.asleep, sim.worst); sim.frames.clear(); sim.time = 0; sim._next_frame_t = 0
    cr = Crocheter(sim)
    er, et = ER(th), ET(th)
    O = er * (rho - 3.3)
    c = cr.strand_anchor()
    k, G = cr.strand_point(6.5, c)
    plan = cr.plan_catch(G, cr.strand_dir(k), free_dir=er, through=O)
    sim.move_to(O - plan['a'] * 6, plan['R'], speed=40)
    cr.push_and_catch(plan, c, O - plan['a'] * 9.0)
    sim.hold(0.2)
    sim.th = th
    report(sim, 'A'); save(sim, 'A')
if stage == 'B':
    sim = load('A'); cr = Crocheter(sim); th = sim.th; sim.verbose = 500
    er, et = ER(th), ET(th)
    i0 = len(sim.frames)
    top = rho + 3.7
    # lift the hook (with its loop) in front of the rim; the yarn hand goes up behind
    F2 = er * 24 + EY * 10 + et * 0.0
    Rk = sim.R.copy()
    sim.move_to(er * top - EY * 7.0, frame(EY, et), speed=26, feeder=F2)
    # the hand that holds the work pinches the yarn where it leaves the work: the loop that is
    # let out of the throat onto the shaft keeps its size
    c = cr.work_anchor(); sim.pin(c); temp = [c]; print('pinch', c, sim.X[c])
    sim.release()
    sim.wait_taut()
    # where does the strand cross the level of the hook, behind the ring?
    Xs = sim.X[c:sim.n]
    lev = Xs @ er
    kk = c + int(np.argmax((lev > top) & (Xs[:, 1] > 1.0)))
    G = sim.X[kk].copy()
    print('anchor', c, 'G node', kk, G, 'sdir', cr.strand_dir(kk))
    plan = cr.plan_catch(G, cr.strand_dir(kk), free_dir=et, a=EY)
    print('plan n', plan['n'], 'start', plan['start'], 'over', plan['over'])
    sim.move_to(plan['start'] - EY * 3, plan['R'], speed=26)
    try: cr.push_and_catch(plan, c, plan['Gp'] - EY * 6, stop_at_capture=True)
    except Exception as e: print('FAILED', e); report(sim, 'B', i0, nfr=8); save(sim, 'Bfail'); sys.exit()
    print('captured at tip', sim.t, sim.cap, 'time', sim.time); i_cap = len(sim.frames)
    # turn the throat down so the loop on the shaft slides off over the back of the head
    try:
        sim.move_to(sim.t, frame(EY, -er), speed=26)
        sim.move_to(er * top - EY * 9.0, speed=26)
    except Exception as e: print('FAILED', e); report(sim, 'B', i0, nfr=8); save(sim, 'Bfail'); sys.exit()
    i_pull = len(sim.frames); print('pulled time', sim.time)
    for c in temp: sim.pin(c, False)
    # the yarn hand comes down behind the work and draws the stitch tight
    sim.move_to(er * top - EY * 10.5, speed=10, feeder=er * (rho + 3.0) + EY * 22)
    sim.wait_taut()
    sim.hold(0.2)
    # the hand that holds the work keeps the finished stitch where it is on the ring
    X = sim.X
    for i in range(sim.mk['ring1'] + 4, sim.n - 1):
        if X[i, 1] * X[i + 1, 1] <= 0 and math.hypot(X[i, 0], X[i, 2]) < rho:
            sim.pin(i); print('hold', i, X[i])
    c = cr.work_anchor(); K = sim.X[c].copy()
    cross = [i for i in range(sim.mk['ring1'] + 4, c) if sim.W[i] == 0]
    j = cross[0] + int(np.argmin(np.linalg.norm(sim.X[cross[0]:cross[-1]] - K, axis=1)))
    sim.pin(j); print('hold head', j, sim.X[j], 'K', c, K)
    sim.pin(c); sim.K = c        # the tightened knot does not give yarn back: the loop on the hook keeps its size
    print('frames', i0, i_cap, len(sim.frames))
    report(sim, 'B', i_cap - 6, nfr=16, cams=[Camera([-30, -45, 50], [9, 0, 9], fov=22)], size=(520, 520)); save(sim, 'B')
if stage == 'C':
    sim = load(sys.argv[3] if len(sys.argv) > 3 else 'B'); cr = Crocheter(sim); sim.verbose = 250
    th0 = sim.th
    th = th0 + math.radians(float(sys.argv[2]) if len(sys.argv) > 2 else 36)
    er, et, er0 = ER(th), ET(th), ER(th0)
    i0 = len(sim.frames)
    unit = lambda v: v / np.linalg.norm(v)
    try:
        kpin = sim.K; K = sim.X[kpin].copy()
        T = er * 4.0                            # the place in the ring where the hook goes through
        a = EY.copy()
        C0 = T + a * ((K - T) @ a)              # the place on the shaft nearest to the knot of the last stitch
        u = unit(C0 - K)
        # the strand is laid across behind the ring, beside the way of the hook (on the side w)
        w = np.cross(a, u)
        if w @ er < 0:
            w = -w
        Gn = T + a * 4.5 + w * 3.3
        sim.move_to(feeder=Gn + unit(Gn - K) * 18.0, speed=40)
        sim.wait_taut()
        for _ in range(5):      # the yarn hand lays the strand across the place where the hook comes out
            kk = kpin + int(np.argmin(np.linalg.norm(sim.X[kpin:sim.n] - Gn, axis=1)))
            e = Gn - sim.X[kk]
            print('aim: strand misses by', np.linalg.norm(e))
            if np.linalg.norm(e) < 0.4:
                break
            sim.move_to(feeder=sim.F + e * np.linalg.norm(sim.F - K) / max(np.linalg.norm(sim.X[kk] - K), 2.0), speed=26)
            sim.wait_taut()
        # the hook lies over the last stitch, throat turned away from it, and points into the ring
        R0 = frame(a, u)
        hp = sim.hp
        slot = np.array([hp[8] - hp[5], hp[9] - hp[6]]) * 1.2
        sim.move_to(C0 + a * 1.7 + u * slot[1], R0, speed=26)
        loop = (int(np.where(sim.W[:kpin] == 0)[0][-1]) + 1, kpin)
        print('K', kpin, K, 'loop', loop, 'a', a, 'u', u, 'cap', sim.cap)
        # the hook slips forward under its loop: the loop leaves the throat along the slot and is on the shaft
        sim.release()
        sim.move_to(C0 + a * (1.7 + slot[0]), speed=26)
        sim.move_to(T + a * 10.0, speed=26)
        xl = sim.hook_local(sim.X[loop[0]:loop[1]])
        near = np.hypot(xl[:, 1], xl[:, 2]) < 4.5
        print('loop on shaft at', xl[near, 0].min() if near.any() else None, xl[near, 0].max() if near.any() else None)
        save(sim, 'Cpre')
        # the throat turns to the strand and the yarn hand lays the strand over the hook
        sim.move_to(None, frame(a, w), speed=26)
        sim.move_to(feeder=sim.F - w * 6.0 * np.linalg.norm(sim.F - K) / max(np.linalg.norm(Gn - K), 2.0), speed=26)
        sim.wait_taut()
        Xs = sim.X[kpin + 2:sim.n]
        xl = sim.hook_local(Xs)
        ks = int(np.argmin(np.hypot(xl[:, 1], xl[:, 2])))
        print('strand on hook at', xl[ks], 'w', w)
        save(sim, 'Cpre')
        def grab(sm):
            if sm.cap[0] < 0.5:
                sm.try_capture(lo=kpin + 2, tol=0.4)
            return sm.cap[0] > 0.5
        sim.move_to(T - a * 5.5, speed=26, until=grab)
        print('cap', sim.cap, 'tip', sim.t)
        if sim.cap[0] < 0.5:
            raise RuntimeError('the throat did not take the strand')
        save(sim, 'Ccap')
        # the yarn hand goes straight behind the hook, so the yarn runs freely through the throat
        sim.move_to(feeder=T + a * 22.0 + w * 3.0, speed=40)
        # the throat turns away from the last stitch; a finger of the hook hand rests on the old loop
        sim.move_to(None, frame(a, u), speed=26)
        lo = loop[0]
        xl = sim.hook_local(sim.X[lo:kpin]); near = np.where(np.hypot(xl[:, 1], xl[:, 2]) < 3.6)[0]
        nd = lo + int(near[len(near) // 2]); print('finger on node', nd, 'at', xl[nd - lo])
        sim.hold_on_hook(nd)
        # the hook comes back out of the ring with the new loop, its handle swinging towards the last stitch
        tilt = math.radians(35)
        a2 = a * math.cos(tilt) + u * math.sin(tilt)
        sim.move_to(T + a2 * 1.6 - EY * 2.0, frame(a2, u), speed=20)
        sim.wait_taut()
        sim.th1 = th
    except Exception as e:
        print('FAILED', e)
    report(sim, 'C', i0, nfr=12, cams=[Camera([-30, -45, 50], [9, 0, 9], fov=22)], size=(480, 480)); save(sim, 'C')
if stage == 'D':
    sim = load(sys.argv[2] if len(sys.argv) > 2 else 'C'); cr = Crocheter(sim); sim.verbose = 250
    th = sim.th1; er, et = ER(th), ET(th)
    i0 = len(sim.frames)
    unit = lambda v: v / np.linalg.norm(v)
    top = rho + 3.7
    a = EY.copy(); w = et.copy()
    hp = sim.hp
    slot = np.array([hp[8] - hp[5], hp[9] - hp[6]]) * 1.2
    try:
        kpin = sim.K
        # the hook rises in front of the rim with its two loops, throat up
        sim.let_go()
        sim.move_to(None, frame(-sim.R[:, 0], er), speed=20)
        sim.move_to(er * (top + slot[1]) + EY * 0.0, frame(a, er), speed=20)
        sim.wait_taut()
        # the hand that holds the work pinches the yarn where the new loop comes out of the ring
        X = sim.X; k0 = int(sim.cap[1]) + 2
        base = [i for i in range(k0, sim.n - 1) if X[i, 1] * X[i + 1, 1] <= 0 and math.hypot(X[i, 0], X[i, 2]) < rho]
        base = base[0] if base else cr.work_anchor()
        sim.pin(base); B = sim.X[base].copy(); print('pinch', base, B, 'cap', sim.cap)
        # the yarn hand lifts the strand behind the rim, beside the way of the hook
        Gn = er * top + EY * 8.0 + w * 3.3
        sim.move_to(feeder=Gn + unit(Gn - B) * 18.0, speed=40)
        sim.wait_taut()
        for _ in range(5):
            kk = base + int(np.argmin(np.linalg.norm(sim.X[base:sim.n] - Gn, axis=1)))
            e = Gn - sim.X[kk]
            print('aim: strand misses by', np.linalg.norm(e))
            if np.linalg.norm(e) < 0.4:
                break
            sim.move_to(feeder=sim.F + e * np.linalg.norm(sim.F - B) / max(np.linalg.norm(sim.X[kk] - B), 2.0), speed=26)
            sim.wait_taut()
        # the hook slips forward under the new loop (it joins the old one on the shaft) and on past the strand
        sim.release()
        sim.move_to(er * top + EY * slot[0], speed=26)
        sim.move_to(er * top + EY * 13.5, speed=26)
        save(sim, 'Dpre')
        sim.move_to(None, frame(a, w), speed=26)
        sim.move_to(feeder=sim.F - w * 6.0 * np.linalg.norm(sim.F - B) / max(np.linalg.norm(Gn - B), 2.0), speed=26)
        sim.wait_taut()
        xl = sim.hook_local(sim.X[base + 2:sim.n])
        ks = int(np.argmin(np.hypot(xl[:, 1], xl[:, 2])))
        print('strand on hook at', xl[ks])
        def grab(sm):
            if sm.cap[0] < 0.5:
                sm.try_capture(lo=base + 2, tol=0.4)
            return sm.cap[0] > 0.5
        sim.move_to(er * top + EY * 4.0, speed=26, until=grab)
        print('cap', sim.cap, 'tip', sim.t)
        if sim.cap[0] < 0.5:
            raise RuntimeError('the throat did not take the strand')
        save(sim, 'Dcap')
        sim.move_to(feeder=er * top + EY * 30.0 + w * 3.0, speed=40)
        # throat down, finger off: the hook draws the strand through both loops
        sim.move_to(None, frame(a, -er), speed=26)
        sim.let_go()
        sim.move_to(er * top - EY * 10.5, speed=26)
        print('pulled through', sim.time)
        save(sim, 'Dpull')
        # the work is let go for a moment and the yarn hand draws the stitch tight
        sim.pin(base, False); sim.pin(kpin, False)
        sim.move_to(feeder=er * (rho + 3.0) + EY * 22, speed=20)
        sim.wait_taut()
        sim.hold(0.2)
        X = sim.X
        c = cr.work_anchor(); K = sim.X[c].copy()
        for i in range(kpin, c):
            if X[i, 1] * X[i + 1, 1] <= 0 and math.hypot(X[i, 0], X[i, 2]) < rho:
                sim.pin(i); print('hold', i, X[i])
        sim.pin(c); sim.K = c; sim.th = th; print('K', c, K)
    except Exception as e:
        print('FAILED', e)
    report(sim, 'D', i0, nfr=16, cams=[Camera([-30, -45, 50], [6, 0, 9], fov=24)], size=(480, 480)); save(sim, 'D')
print('time %.1f' % (time.time() - t0))
