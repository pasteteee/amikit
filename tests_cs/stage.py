"""Staged development driver:  python tests_cs/stage.py A   /  B  ...  each stage loads the
state the previous one saved."""
import sys, math, pickle, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim.sim import Sim
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
def load(name):
    return pickle.load(open(SC + name + '.pkl', 'rb'))
def report(sim, tag, i0=0, nfr=8, cams=None, size=(400, 400)):
    cams = cams or CAMS
    print(tag, 'n', sim.n, 'nsub', sim.nsub, 'worst', {k: round(float(v), 4) for k, v in sim.worst.items()}, 'bad', len(sim.log), 'why', sim.why, 'cap', sim.cap, 'nsplit', sim.nsplit)
    sheet(sim, SC + 'st_' + tag + '.png', cams, nfr=nfr, i0=i0, size=size)

stage = sys.argv[1]
t0 = time.time()
if stage == 'A':
    th = math.radians(38)
    F = np.array([-17.0, 36.0, 38.0])
    X0, mk = magic_ring(h, r, rho=rho, feeder=F)
    sim = Sim(X0, h, r, feeder=F, hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[3, -30, 2])
    sim.feed[3] = np.linalg.norm(X0[-1] - F) + 1e-9
    sim.mk = mk
    ang = np.degrees(np.arctan2(X0[:, 2], X0[:, 0])) % 360
    ringn = np.arange(mk['ring0'], mk['ring1'])
    sim.pin(0)
    sim.pin(np.arange(mk['ring0'] - 3, mk['ring0'] + 3)); sim.pin(np.arange(mk['ring1'] - 3, mk['ring1'] + 4))
    sim.pin(ringn[(ang[ringn] > 110) & (ang[ringn] < 345)])
    sim.settle(50); sim.frames.clear()
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
    sim = load('A'); cr = Crocheter(sim); th = sim.th
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
    print('captured at tip', sim.t, sim.cap); i_cap = len(sim.frames)
    # turn the throat down so the loop on the shaft slides off over the back of the head
    sim.move_to(sim.t, frame(EY, -er), speed=26)
    sim.move_to(er * top - EY * 9.0, speed=26)
    i_pull = len(sim.frames)
    # the yarn hand comes down behind the work and draws the stitch tight
    sim.move_to(er * top - EY * 7.5, speed=10, feeder=er * (rho + 3.0) + EY * 22)
    sim.wait_taut()
    sim.hold(0.2)
    print('frames', i0, i_cap, len(sim.frames))
    report(sim, 'B', i_cap - 6, nfr=16, cams=[Camera([-30, -45, 50], [9, 0, 9], fov=22)], size=(520, 520)); save(sim, 'B')
print('time %.1f' % (time.time() - t0))
