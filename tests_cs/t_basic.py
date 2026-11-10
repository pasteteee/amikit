import sys, time, numpy as np
sys.path.insert(0, '.')
from crochetsim.sim import Sim
from crochetsim.hook import frame
from crochetsim.render import Renderer, Camera
from PIL import Image
out = sys.argv[1]
h, r = 0.8, 1.0
xs = np.arange(-40, 59.21, h)
X0 = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
# a wavy free piece far from the hook that must never move
ys = np.arange(0, 60, h)
F = np.array([60.0, 0, 0])
sim = Sim(X0, h, r, feeder=F, hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[0, -12, -0.3])
sim.pin(0)
X_init = sim.X[:sim.n].copy()
t0 = time.time()
sim.hold(0.5)
print('stillness (hook away):', np.abs(sim.X[:len(X_init)] - X_init).max(), 'compile+run', time.time() - t0)
t0 = time.time()
# push under the strand
sim.move_to([0, 4.0, -0.3], speed=20)
d = sim.try_capture()
print('capture dist', d, 'n', sim.n)
for k in range(6):
    Xa = sim.X[:sim.n].copy(); na = sim.n
    sim.hold(0.1)
    print('  standing, moved in 0.1s:', np.abs(sim.X[:min(na,sim.n)] - Xa[:min(na,sim.n)]).max(), sim.n, 'lamF', sim.last_lamF, 'fails', sim.nprobe_fail, 'active', sim.strand_active)
sim.move_to([0, -30, -0.3], speed=20)
sim.hold(0.3)
print('after pull n', sim.n, 'fed', sim.fed_total, 'worst', sim.worst, 'nsub', sim.nsub, 'time', time.time() - t0)
cam = Camera([-30, -90, 60], [5, -10, 0], fov=30)
rn = Renderer(360, 480, ss=2)
t0 = time.time()
ims = []
idx = np.linspace(0, len(sim.frames) - 1, 8).astype(int)
for i in idx:
    f = sim.frames[i]
    ims.append(rn.render(f['X'], f['D'], f['R'], f['t'], sim.hp, cam))
print('render', (time.time() - t0) / len(idx), 's/frame')
Image.fromarray(np.concatenate([np.concatenate(ims[:4], 1), np.concatenate(ims[4:], 1)], 0)).save(out)
