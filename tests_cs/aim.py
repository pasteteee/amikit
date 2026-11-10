import sys, math, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
_av = sys.argv; sys.argv = ['x', 'none']
exec(open('tests_cs/stage.py').read().split("stage = sys.argv[1]")[0])
sim = load('B'); cr = Crocheter(sim)
th = sim.th + math.radians(36); er = ER(th); O = er * (rho - 3.3); Gn = O + EY * 4.5
np.set_printoptions(precision=1, suppress=True, linewidth=220)
c = cr.work_anchor(); print('B end: anchor', c, sim.X[c], 'F', sim.F, 'n', sim.n, 'cap', sim.cap, 'pocket', sim.pocket())
print(sim.X[c-4:sim.n:3].T); print('flags', sim.flags[c-8:sim.n])
A = sim.X[c].copy()
sim.move_to(O - EY * 7.0, frame(EY, er), speed=26, feeder=Gn + (Gn - A) / np.linalg.norm(Gn - A) * 16.0)
sim.wait_taut()
c = cr.work_anchor(); print('after: anchor', c, sim.X[c], 'F', sim.F, 'n', sim.n, 'Gn', Gn)
print(sim.X[c-4:sim.n:2].T); print('flags', sim.flags[c-8:sim.n])
save(sim, 'aim')
