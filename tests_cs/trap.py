"""Run a stage with a trap: save the solver input of the first bad tool solve after nsub>=K."""
import sys, pickle, copy, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim import solver as S
from crochetsim.sim import Sim
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
K = int(sys.argv[2]); cnt = [0]
orig = Sim._solve
def solve(self, Rn, tn):
    pre = dict(X=self.X[:self.n].copy(), D=self.D[:self.n].copy(), W=self.W[:self.n].copy(), R=self.R.copy(), t=self.t.copy(), Rn=np.array(Rn), tn=np.array(tn), feed=self.feed.copy(), cap=self.cap.copy(), n=self.n, prm=self.prm.copy(), hp=self.hp.copy())
    orig(self, Rn, tn)
    st = self.stats
    if self.nsub >= K and self.feed[4] < 1e20 and (st[S.S_STRETCH] > 0.03 or st[S.S_MAXDISP] > 0.3):
        cnt[0] += 1
        if cnt[0] == int(sys.argv[3]) if len(sys.argv) > 3 else 1:
            pickle.dump(pre, open(SC + 'trap.pkl', 'wb'))
            print('TRAPPED nsub', self.nsub, 'stretch', st[S.S_STRETCH], 'disp', st[S.S_MAXDISP], 'pens', st[S.S_HOOKPEN], st[S.S_SELFPEN], 'nit', st[S.S_NIT])
            sys.exit()
Sim._solve = solve
sys.argv = ['stage.py', sys.argv[1]]
exec(open('tests_cs/stage.py').read())
