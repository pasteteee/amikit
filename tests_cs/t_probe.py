import sys, numpy as np
sys.path.insert(0, '.')
from crochetsim.sim import Sim
from crochetsim import solver as S
from crochetsim.hook import frame
from crochetsim.render import Renderer, Camera
from PIL import Image
h, r = 0.8, 1.0
xs = np.arange(-40, 59.21, h)
X0 = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
sim = Sim(X0, h, r, feeder=[60.0, 0, 0], hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[0, -12, -0.3])
sim.pin(0)
sim.T0 = float(sys.argv[2]); sim.prm[12] = float(sys.argv[3]); sim.T_hold = float(sys.argv[4])
orig = sim._solve
tr = []
def solve(Rn, tn):
    orig(Rn, tn)
    if sim.stats[S.S_STRETCH] > 0.03 and sim.feed[4] < 1e20: print('  BAD nsub', sim.nsub, 'probe' if sim.feed[4] > 1e20 else 'tool', 'tipy %.2f' % sim.t[1], 'stretch %.3f' % sim.stats[S.S_STRETCH], 'disp %.3f' % sim.stats[S.S_MAXDISP], 'pen %.3f' % sim.stats[S.S_HOOKPEN], 'lam %.1f' % sim.feed[6], 'n', sim.n, 'cap', sim.cap)
    tr.append((sim.nsub, sim.feed[6], sim.feed[7], sim.stats[S.S_HOOKPEN], sim.stats[S.S_MAXDISP], sim.feed[4], sim.n))
sim._solve = solve
sim.move_to([0, 7.2, -0.3], speed=20)
sim.hold(0.2)
ok = sim.move_to([0, -20, -0.3], speed=20, until=lambda s: s.cap[0] < 0.5 and s.try_capture(tol=0.3) is not None and False)
print('cap', sim.cap, 'n', sim.n, 'capdist', sim.stats[S.S_CAPDIST])
sim.hold(0.3)
print('why', sim.why); print('pulled: n', sim.n, 'fails', sim.nprobe_fail, 'nsub', sim.nsub)
k0 = len(tr)
sim.move_to([0, -5, -0.3], speed=20)
sim.hold(0.5)
print('why', sim.why); print('returned: n', sim.n, 'fails', sim.nprobe_fail, 'nsub', sim.nsub, sim.worst)
for k in range(k0, len(tr), 25):
    print('%5d lamF %8.3f last %8.4f pen %.4f disp %.4f wF %.1f n %d' % tr[k])
Xa = sim.X[:sim.n].copy(); sim.hold(0.5); print('still', np.abs(sim.X[:sim.n]-Xa).max())
cam = Camera([-30, -90, 60], [5, -10, 0], fov=30)
rn = Renderer(360, 480, ss=2)
idx = np.linspace(0, len(sim.frames) - 1, 8).astype(int)
ims = [rn.render(sim.frames[i]['X'], sim.frames[i]['D'], sim.frames[i]['R'], sim.frames[i]['t'], sim.hp, cam) for i in idx]
Image.fromarray(np.concatenate([np.concatenate(ims[:4], 1), np.concatenate(ims[4:], 1)], 0)).save(sys.argv[1])
