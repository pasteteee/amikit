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
    sim.release()
    save(sim, 'Bplan')
    sim.wait_taut()
    c = cr.strand_anchor()
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
    sim.move_to(sim.t, frame(EY, -er), speed=26)
    sim.move_to(er * top - EY * 9.0, speed=26)
    i_pull = len(sim.frames); print('pulled time', sim.time)
    # the yarn hand comes down behind the work and draws the stitch tight
    sim.move_to(er * top - EY * 7.5, speed=10, feeder=er * (rho + 3.0) + EY * 22)
    sim.wait_taut()
    sim.hold(0.2)
    print('frames', i0, i_cap, len(sim.frames))
    report(sim, 'B', i_cap - 6, nfr=16, cams=[Camera([-30, -45, 50], [9, 0, 9], fov=22)], size=(520, 520)); save(sim, 'B')
if stage == 'C':
    sim = load('B'); cr = Crocheter(sim); sim.verbose = 250
    th = sim.th + math.radians(float(sys.argv[2]) if len(sys.argv) > 2 else 36)
    er, et = ER(th), ET(th)
    er0 = ER(sim.th)
    i0 = len(sim.frames)
    # the hand that holds the work keeps the finished stitch where it is on the ring
    X = sim.X
    for i in range(sim.mk['ring1'] + 4, sim.n - 1):
        if X[i, 1] * X[i + 1, 1] <= 0 and math.hypot(X[i, 0], X[i, 2]) < rho:
            sim.pin(i); print('hold', i, X[i])
    O = er * (rho - 3.3)
    top1 = er0 * (rho + 2.0)
    Gn = O + EY * 4.5
    F1 = Gn + (Gn - top1) * 2.2
    print('F1', F1, 'in stitch frame', F1 @ er, F1 @ et, F1[1])
    # hook goes in front of the next place in the ring, throat up; yarn hand goes down behind
    sim.move_to(O - EY * 7.0, frame(EY, er), speed=26, feeder=F1)
    sim.wait_taut()
    for _ in range(2):      # the yarn hand lays the strand across the place where the hook comes out
        A = sim.X[cr.strand_anchor()].copy()
        F1 = Gn + (Gn - A) / np.linalg.norm(Gn - A) * 16.0
        sim.move_to(feeder=F1, speed=26)
        sim.wait_taut()
    save(sim, 'Cpre')
    c = cr.strand_anchor()
    Xs = sim.X[c:sim.n]
    kk = c + int(np.argmin(np.linalg.norm(Xs - Gn, axis=1)))
    G = sim.X[kk].copy()
    print('anchor', c, sim.X[c], 'G node', kk, G, 'wanted', Gn, 'sdir', cr.strand_dir(kk))
    try:
        plan = cr.plan_catch(G, cr.strand_dir(kk), free_dir=er, through=O)
        print('plan a', plan['a'], 'n', plan['n'])
        sim.move_to(O - plan['a'] * 6, plan['R'], speed=26)
        sim.release()
        cr.push_and_catch(plan, None, O - plan['a'] * 6.5)
    except Exception as e:
        print('FAILED', e)
    sim.wait_taut()
    report(sim, 'C', i0, nfr=12, cams=[Camera([-30, -45, 50], [9, 0, 9], fov=22)], size=(480, 480)); save(sim, 'C')
print('time %.1f' % (time.time() - t0))
