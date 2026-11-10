import sys, numpy as np
sys.path.insert(0, '.')
from crochetsim.sim import Sim
from crochetsim import solver as S
from crochetsim.hook import frame
h, r = 0.8, 1.0
xs = np.arange(-40, 59.21, h)
X0 = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
sim = Sim(X0, h, r, feeder=[60.0, 0, 0], hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[0, -12, -0.3])
sim.pin(0)
k=0
while sim.t[1] < 7.2:
    sim._substep(sim.R, sim.t + np.array([0, 0.04, 0])); k+=1
    if k % 20 == 0 or sim.stats[S.S_STRETCH] > 0.01:
        print(k, 'tipy %.2f' % sim.t[1], 'n', sim.n, 'le %.3f' % sim.feed[3], 'lamacc %.1f' % sim.feed[6], 'last %.2f' % sim.feed[7], 'disp %.4f' % sim.stats[S.S_MAXDISP], 'stretch %.4f' % sim.stats[S.S_STRETCH], 'pen %.3f' % sim.stats[S.S_HOOKPEN])
    if sim.stats[S.S_STRETCH] > 0.3: break
