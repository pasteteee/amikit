import sys, numpy as np
sys.path.insert(0, '.')
from crochetsim.sim import Sim
from crochetsim.scene import magic_ring
from crochetsim.hook import frame
from crochetsim.render import Renderer, Camera
from PIL import Image
h, r = 0.8, 1.0
F = np.array([-17.0, 36.0, 38.0])
X0, mk = magic_ring(h, r, feeder=F)
print(len(X0), mk, np.abs(np.linalg.norm(np.diff(X0, axis=0), axis=1) - h).max(), X0[-1], np.linalg.norm(X0[-1]-F))
sim = Sim(X0, h, r, feeder=F, hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[3, -30, 2])
sim.feed[3] = np.linalg.norm(X0[-1] - F) + 1e-9
sim.pin(0)
sim.pin(np.arange(mk['ring0'] - 3, mk['ring0'] + 3)); sim.pin(np.arange(mk['ring1'] - 3, mk['ring1'] + 4))
Xa = sim.X[:sim.n].copy(); sim.hold(0.3)
print('still', np.abs(sim.X[:len(Xa)] - Xa).max(), sim.n, sim.worst, sim.strand_active)
rn = Renderer(480, 640, ss=2)
f = sim.frames[-1]
a = rn.render(f['X'], f['D'], f['R'], f['t'], sim.hp, Camera([-14, -75, 30], [0, 2, 2], fov=30))
b = rn.render(f['X'], f['D'], f['R'], f['t'], sim.hp, Camera([-60, 12, 50], [0, 4, 4], fov=30))
Image.fromarray(np.concatenate([a, b], 1)).save(sys.argv[1])
