import sys, numpy as np, math
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim.sim import Sim
from crochetsim.scene import magic_ring
from crochetsim.hook import frame
from crochetsim.crocheter import Crocheter, unit
from crochetsim.render import Camera
from dbgview import sheet
h, r = 0.8, 1.0
rho = 7.5
F = np.array([-17.0, 36.0, 38.0])
X0, mk = magic_ring(h, r, rho=rho, feeder=F)
sim = Sim(X0, h, r, feeder=F, hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[3, -30, 2])
sim.feed[3] = np.linalg.norm(X0[-1] - F) + 1e-9
sim.pin(0)
for kv in sys.argv[2:]:
    k_, v_ = kv.split("=")
    if k_.startswith("p"): sim.prm[int(k_[1:])] = float(v_)
    else: setattr(sim, k_, float(v_))
sim.pin(np.arange(mk['ring0'] - 3, mk['ring0'] + 3)); sim.pin(np.arange(mk['ring1'] - 3, mk['ring1'] + 4))
sim.settle(400); sim.frames.clear()
cr = Crocheter(sim)
th = math.radians(38)
er = np.array([math.cos(th), 0, math.sin(th)]); et = np.array([-math.sin(th), 0, math.cos(th)]); w = np.array([0, 1.0, 0])
O = er * (rho - 3.3)
c = cr.strand_anchor(); print('anchor', c, sim.X[c], 'n', sim.n)
k, G = cr.strand_point(6.5, c)
plan = cr.plan_catch(G, cr.strand_dir(k), free_dir=er, through=O)
print('G', G, 'a', plan['a'], 'n', plan['n'])
sim.move_to(O - plan['a'] * 6, plan['R'], speed=40)
i0 = len(sim.frames)
cr.push_and_catch(plan, c, O - plan['a'] * 9.0)
print('captured', sim.cap, 'n', sim.n, sim.worst, 'why', sim.why)

print(len(sim.log), 'bad of', sim.nsub)
sim.hold(0.3)
Xa = sim.X[:sim.n].copy(); sim.hold(0.3); print('still', np.abs(sim.X[:sim.n] - Xa).max())
sheet(sim, sys.argv[1], [Camera([-14, -70, 28], [2, 2, 3], fov=24), Camera([-55, 10, 45], [2, 4, 4], fov=24)], nfr=8, i0=i0)
