"""view.py NAME t0 t1 nfr [cam: px,py,pz,tx,ty,tz,fov] : contact sheet of saved frames."""
import sys, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'tests_cs')
from crochetsim.render import Renderer, Camera
from crochetsim.hook import default_hook
from PIL import Image
SC = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
fr = pickle.load(open(SC + sys.argv[1] + '_frames.pkl', 'rb'))
t0, t1, nfr = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
c = [float(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else [-30, -45, 50, 9, 0, 9, 22]
cam = Camera(c[:3], c[3:6], fov=c[6])
ts = np.array([f['time'] for f in fr])
idx = [int(np.argmin(np.abs(ts - t))) for t in np.linspace(t0, t1, nfr)]
rn = Renderer(480, 480, ss=1)
hp = default_hook(yarn_r=1.0)
ims = [rn.render(fr[i]['X'], fr[i]['D'], fr[i]['R'], fr[i]['t'], hp, cam) for i in idx]
cols = 4
rows = [np.concatenate(ims[a:a + cols], 1) for a in range(0, len(ims), cols)]
Image.fromarray(np.concatenate(rows, 0)).save(SC + 'view.png')
print('frames', idx, 'times', [round(float(ts[i]), 2) for i in idx])
