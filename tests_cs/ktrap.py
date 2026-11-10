"""ktrap.py STAGE K : run a stage, save the solver input of the K-th step that did not converge.
   ktrap.py replay   : run that step again with the Newton trace."""
import sys, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim.ksolver import KSolver
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
if sys.argv[1] == 'replay':
    a = pickle.load(open(SC + 'ktrap.pkl', 'rb'))
    sol = a['sol']; sol.debug = True
    for kv in sys.argv[2:]:
        k, v = kv.split('='); setattr(sol, k, float(v))
    X = a['X'].copy()
    st = sol.step(X, a['Xp'], a['W'], a['n'], a['Rn'], a['tn'], a['hp'], a['hp2'], a['cap'].copy(), a['fn'], a['force'])
    print(st)
    N = sol.N; m = st['m']; sl = sol.slot[:m]
    kinds = np.digitize(sl, [N, 2 * N, 3 * N, 4 * N, 4 * N + 1])
    print('active kinds [stretch, hooknode, hookmid, bend, cap, pair]:', np.bincount(kinds, minlength=6))
    print('hook nodes', sl[kinds == 1] - N, 'mids', sl[kinds == 2] - 2 * N, 'bend', sl[kinds == 3] - 3 * N)
    pk = sl[kinds == 5] - 4 * N - 1
    print('pairs', [(int(sol.pairs[k, 0]), int(sol.pairs[k, 1]), round(float(sol.lam[4 * N + 1 + k]), 3)) for k in pk])
    sys.exit()
K = int(sys.argv[2]); cnt = [0]
orig = KSolver.step
def step(self, X, Xp, W, n, Rn, tn, hp, hp2, cap, fn, force):
    import copy
    pre = dict(sol=copy.deepcopy(self), X=X.copy(), Xp=Xp.copy(), W=W.copy(), n=n, Rn=np.array(Rn), tn=np.array(tn), hp=hp, hp2=hp2, cap=cap.copy(), fn=fn, force=force.copy())
    st = orig(self, X, Xp, W, n, Rn, tn, hp, hp2, cap, fn, force)
    if not st['conv'] and st['viol'] > 1e-3:
        cnt[0] += 1
        if cnt[0] == K:
            pickle.dump(pre, open(SC + 'ktrap.pkl', 'wb')); print('TRAPPED', st); sys.exit()
    return st
KSolver.step = step
sys.argv = ['stage.py', sys.argv[1]]
exec(open('tests_cs/stage.py').read())
