import sys, numpy as np
sys.path.insert(0, '.')
from crochetsim.sim import Sim
from crochetsim import solver as S
from crochetsim.hook import frame, hook_sdf
h, r = 0.8, 1.0
xs = np.arange(-40, 59.21, h)
X0 = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
sim = Sim(X0, h, r, feeder=[60.0, 0, 0], hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[0, -12, -0.3])
sim.pin(0)
orig=sim._probe
tr=[]
def pr(mt):
    f0=sim.nprobe_fail; w0=list(sim.why); tk=sim.take_cur; orig(mt)
    tr.append((sim.nsub, tk, sim.last_lamF, sim.feed[7] if False else lastv[0], [a-b for a,b in zip(sim.why,w0)], sim.stats[S.S_MAXDISP], sim.stats[S.S_HOOKPEN]))
lastv=[0]
os=sim._solve
def sv(R,t):
    os(R,t); lastv[0]=sim.feed[7]
sim._solve=sv
sim._probe=pr
def rep(tag):
    P = (sim.X[:sim.n] - sim.t) @ sim.R
    k = np.argmin(np.abs(P[:, 2]))
    print(tag, 'tip y %.2f' % sim.t[1], 'node', k, 'local', np.round(P[k], 2), 'sdf', round(hook_sdf(*P[k], sim.hp), 3), 'why', sim.why, 'act', sim.strand_active, 'take', round(sim.take_cur, 4), 'n', sim.n, 'str %.4f' % sim.worst['stretch'])
for y in np.arange(-1, 7.21, 0.75):
    sim.move_to([0, y, -0.3], speed=20); rep('mv')
sim.hold(0.2); rep('hold')
for y in np.arange(7, 0, -0.5):
    sim.move_to([0, y, -0.3], speed=20); rep('back')

for t in tr[-40:]: print('%d take %.4f lam %.2f last %.3f why %s disp %.4f pen %.4f' % t)
