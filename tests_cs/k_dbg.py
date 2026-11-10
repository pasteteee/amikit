import sys, time, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim.ksim import KSim
from crochetsim.hook import frame
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
h, r = 0.8, 1.0
xs = np.arange(-40, 59.21, h)
X0 = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
sim = KSim(X0, h, r, feeder=[60.0, 0, 0], hook_R=frame([0, 1, 0], [0, 0, 1]), hook_t=[0, -12, -0.3])
sim.pin(0)
sim.move_to([0, 7.2, -0.3], speed=20)
K = int(sys.argv[1])
def grab(s):
    if s.cap[0] < 0.5: s.try_capture(tol=0.4)
    return s.nsub >= K
sim.move_to([0, -20, -0.3], speed=20, until=grab)
print('at nsub', sim.nsub, 'tip', sim.t, 'cap', sim.cap, 'n', sim.n, 'log', len(sim.log))
sim.sol.debug = True
for k in range(int(sys.argv[2])):
    sim._substep(sim.R, sim.t + np.array([0, -0.05, 0])); st = sim.sol.stats; print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()})
    fl = sim.flags[:sim.n]
    print('   contact nodes', np.where(fl > 0)[0], 'tau max', sim.sol.tau[:sim.n].max())
