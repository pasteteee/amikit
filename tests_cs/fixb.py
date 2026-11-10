import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
sys.argv = ['x', 'none']
exec(open('tests_cs/stage.py').read().split("stage = sys.argv[1]")[0])
sim = load('B'); cr = Crocheter(sim)
c = cr.work_anchor(); sim.pin(c); print('hold head', c, sim.X[c]); save(sim, 'B')
