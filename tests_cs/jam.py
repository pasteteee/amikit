import sys, math, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
sys.argv = ['x', 'none']
exec(open('tests_cs/stage.py').read().split("stage = sys.argv[1]")[0])
from crochetsim.ksim import Jam
sim = load('B'); rho_ = rho
X = sim.X
for i in range(sim.mk['ring1'] + 4, sim.n - 1):
    if X[i, 1] * X[i + 1, 1] <= 0 and math.hypot(X[i, 0], X[i, 2]) < rho: sim.pin(i)
th = sim.th + math.radians(36); er = ER(th); O = er * (rho - 3.3)
try:
    sim.move_to(O - EY * 7.0, frame(EY, er), speed=26, feeder=[-12.8, 14.4, 0.05])
except Jam as e:
    print(e)
tau = sim.sol.tau[:sim.n]; k = int(np.argmax(tau))
np.set_printoptions(precision=1, suppress=True)
print('argmax', k, 'cap', sim.cap, 'n', sim.n, 'pins', np.where(sim.W[:sim.n] == 0)[0][-8:])
print(tau[115:sim.n])
print('pocket', sim.pocket(), 'X[k]', sim.X[k], 't', sim.t)
