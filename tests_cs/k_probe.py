import sys, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim.ksim import KSim
from crochetsim.hook import frame
from crochetsim.render import Camera
from dbgview import sheet
h, r = 0.8, 1.0
xs = np.arange(-40, 59.21, h)
X0 = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
sim = KSim(X0, h, r, feeder=[60.0, 0, 0], hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[0, -12, -0.3])
sim.pin(0)
t0 = time.time()
sim.hold(0.2); print('rest: asleep', sim.asleep, 'nsub', sim.nsub)
sim.move_to([0, 7.2, -0.3], speed=20)
print('pushed', sim.n, sim.worst, 'bad', len(sim.log), 'newton/sub %.1f' % (sim.nnewton / max(1, sim.nsub)), 'time %.1f' % (time.time() - t0))
def grab(s):
    if s.cap[0] < 0.5: s.try_capture(tol=0.4)
    return False
sim.move_to([0, -20, -0.3], speed=20, until=grab)
print('pulled cap', sim.cap, 'n', sim.n, sim.worst, 'bad', len(sim.log), 'capd', sim.sol.stats['capdist'])
sim.move_to([0, -5, -0.3], speed=20)
ok = sim.wait_taut(); print('returned n', sim.n, 'taut', ok, sim.worst, 'bad', len(sim.log), 'nsub', sim.nsub, 'newton/sub %.1f' % (sim.nnewton / sim.nsub), 'time %.1f' % (time.time() - t0))
Xa = sim.X[:sim.n].copy(); sim.hold(0.5); print('still', np.abs(sim.X[:sim.n] - Xa).max())
big=[l for l in sim.log if l[4]>1e-3]; print(len(big),'big');
for l in big[:12]: print(l)
sheet(sim, sys.argv[1], [Camera([-30, -90, 60], [5, -10, 0], fov=30)], nfr=8, size=(360, 480))
