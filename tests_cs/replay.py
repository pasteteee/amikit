import sys, pickle, numpy as np
sys.path.insert(0, '.')
from crochetsim import solver as S
from crochetsim.sim import Sim
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
def run(over, verbose=True):
    p = pickle.load(open(SC + 'trap.pkl', 'rb'))
    sim = Sim(p['X'], 0.8, 1.0, feeder=p['feed'][:3], hook_R=p['R'], hook_t=p['t'])
    sim.W[:p['n']] = p['W']; sim.D[:p['n']] = p['D']; sim.feed[:] = p['feed']; sim.cap[:] = p['cap']; sim.prm[:] = p['prm']
    for k, v in over.items():
        if k.startswith('p'): sim.prm[int(k[1:])] = v
    Xa = sim.X[:sim.n].copy()
    sim._solve(p['Rn'], p['tn'])
    st = sim.stats
    d = np.linalg.norm(sim.X[:sim.n] - Xa, axis=1)
    seg = np.abs(np.linalg.norm(sim.X[1:sim.n] - sim.X[:sim.n - 1], axis=1) / 0.8 - 1)
    if verbose:
        print(over, 'nit %d stretch %.4f disp %.3f hpen %.3f spen %.3f fed %.3f lam %.2f npairs %d' % (st[S.S_NIT], st[S.S_STRETCH], st[S.S_MAXDISP], st[S.S_HOOKPEN], st[S.S_SELFPEN], st[S.S_FED], sim.feed[6], st[S.S_NPAIRS]))
        print('   movers', [(int(i), round(float(d[i]), 3)) for i in np.argsort(-d)[:8]], 'stretched', [(int(i), round(float(seg[i]), 3)) for i in np.argsort(-seg)[:6]])
    return sim, p
if __name__ == '__main__':
    run({})
    for a in sys.argv[1:]:
        run({kv.split('=')[0]: float(kv.split('=')[1]) for kv in a.split(',')})
