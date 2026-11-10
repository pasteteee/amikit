import sys, math, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim import solver as S
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
sim = pickle.load(open(SC + sys.argv[1] + '.pkl', 'rb'))
for kv in sys.argv[2:]:
    k_, v_ = kv.split("=")
    if k_.startswith("p"): sim.prm[int(k_[1:])] = float(v_)
    else: setattr(sim, k_, float(v_))
sim.feed[4] = sim.T_hold
for rep in range(6):
    Xa = sim.X[:sim.n].copy()
    sim._solve(sim.R, sim.t)
    d = np.linalg.norm(sim.X[:sim.n] - Xa, axis=1)
    top = np.argsort(-d)[:6]
    st = sim.stats
    print('tool nit %d disp %.5f str %.2e hpen %.4f spen %.4f fed %.4f lam %.2f | movers' % (st[S.S_NIT], st[S.S_MAXDISP], st[S.S_STRETCH], st[S.S_HOOKPEN], st[S.S_SELFPEN], st[S.S_FED], sim.feed[6]), [(int(i), round(float(d[i]), 4), int(sim.flags[i])) for i in top])

print('strand_active', sim.strand_active, 'asleep', sim.asleep, 'take_cur', sim.take_cur, 'probe_sleep', sim.probe_sleep)
sim.strand_active = True; sim.asleep = False
for rep in range(40):
    w0 = list(sim.why); n0 = sim.n; le = sim.feed[3]
    sim._substep(sim.R, sim.t)
    st = sim.stats
    print('sub: n %d le %.3f->%.3f take %.4f why %s lamF %.2f last %.3f nit %d disp %.4f str %.2e pen %.4f/%.4f act %s' % (sim.n, le, sim.feed[3], sim.take_cur, [a - b for a, b in zip(sim.why, w0)], sim.last_lamF, sim.feed[7], st[S.S_NIT], st[S.S_MAXDISP], st[S.S_STRETCH], st[S.S_HOOKPEN], st[S.S_SELFPEN], sim.strand_active))
