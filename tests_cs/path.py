import sys, math, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
sim = pickle.load(open(SC + sys.argv[1] + '.pkl', 'rb'))
th = sim.th
er = np.array([math.cos(th), 0, math.sin(th)]); et = np.array([-math.sin(th), 0, math.cos(th)])
a = int(sys.argv[2]); st = int(sys.argv[3]) if len(sys.argv) > 3 else 2
print('hook tip (er,et,y)', sim.t @ er, sim.t @ et, sim.t[1], 'axis', sim.axis(), 'throat', sim.throat(), 'F', sim.feed[:3] @ er, sim.feed[:3] @ et, sim.feed[1], 'cap', sim.cap)
for i in range(a, sim.n, st):
    x = sim.X[i]; l = np.linalg.inv(sim.R) @ (x - sim.t)
    print(i, 'er %6.2f et %6.2f y %6.2f | hook local x %6.2f rad %5.2f | flags %d W %d' % (x @ er, x @ et, x[1], l[0], math.hypot(l[1], l[2]), sim.flags[i], sim.W[i]))
