"""retr.py key=value ... : from the saved moment of capture, retract with different parameters."""
import sys, math, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
_av = sys.argv; sys.argv = ['x', 'none']
exec(open('tests_cs/stage.py').read().split("stage = sys.argv[1]")[0])
from crochetsim.ksim import Jam
opt = dict(kv.split('=') for kv in _av[1:])
sim = load('Ccap'); sim.frames = []
sol = sim.sol
if 'k' in opt: sim.hp[13] = sim.hp_solid[13] = sim.hp_finger[13] = float(opt['k'])
if 'eb' in opt: sol.eps_b = float(opt['eb'])
if 'es' in opt: sol.eps_s = float(opt['es'])
if 'rho' in opt: sol.par[2] = 2 * h * math.cos(h / (float(opt['rho']) * r) / 2)
if 'unpin' in opt: sim.pin(sim.K, False)
if 'dmax' in opt: sim.dmax = float(opt['dmax'])
th = sim.th + math.radians(36); er = ER(th); T = er * 4.0; a = EY
K = sim.X[sim.K].copy(); u = (T - K); u[1] = 0; u /= np.linalg.norm(u); w = np.cross(a, u); w = w if w @ er > 0 else -w
tmax = 0
if 'hold' in opt:
    pins = np.where(sim.W[:sim.K] == 0)[0]; lo = int(pins[-1]) + 1
    xl = sim.hook_local(sim.X[lo:sim.K]); near = np.where(np.hypot(xl[:, 1], xl[:, 2]) < 3.6)[0]
    nd = lo + int(near[len(near) // 2]); print('finger on node', nd, 'at', xl[nd - lo], 'near', lo + near[0], lo + near[-1])
    sim.hold_on_hook(nd)
if 'fx' in opt:
    sim.hp_finger[14] = float(opt['fx'])
    pins = np.where(sim.W[:sim.K] == 0)[0]
    sim.finger(int(pins[-1]) + 1, sim.K)
try:
    if 'F' in opt: sim.move_to(feeder=T + a * 22.0 + w * 3.0, speed=40)
    if 'roll' in opt:
        ang = math.radians(float(opt['roll'])); sim.move_to(None, frame(a, w * math.cos(ang) + u * math.sin(ang)), speed=26)
    tilt = math.radians(float(opt.get('tilt', 45)))
    a2 = a * math.cos(tilt) + u * math.sin(tilt)
    back = float(opt.get('back', 1.2))
    sim.move_to(T + a2 * back - EY * float(opt.get('sh', 0)), frame(a2, u), speed=20)
    sim.wait_taut()
    print('tau %.0f worst viol %.3f stretch %.3f bad %d cap' % (sim.sol.tau[:sim.n].max(), sim.worst['viol'], sim.worst['stretch'], len(sim.log)), sim.cap)
    pins = np.where(sim.W[:sim.K] == 0)[0]
    xl = sim.hook_local(sim.X[pins[-1] + 1:sim.K]); near = np.hypot(xl[:, 1], xl[:, 2]) < 4.5
    print('L on shaft', xl[near, 0].min() if near.any() else None, xl[near, 0].max() if near.any() else None)
except Jam as e:
    print('JAM', e)
save(sim, 'retr')
