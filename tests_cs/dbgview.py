import numpy as np
from crochetsim.render import Renderer, Camera
from PIL import Image
_rn = {}
def sheet(sim, path, cams, nfr=8, i0=0, i1=None, size=(400, 400), cols=4):
    key = size
    if key not in _rn: _rn[key] = Renderer(size[0], size[1], ss=1)
    rn = _rn[key]
    i1 = len(sim.frames) - 1 if i1 is None else i1
    idx = np.linspace(i0, i1, nfr).astype(int)
    rows = []
    for cam in cams:
        ims = [rn.render(sim.frames[i]['X'], sim.frames[i]['D'], sim.frames[i]['R'], sim.frames[i]['t'], sim.hp, cam) for i in idx]
        for a in range(0, len(ims), cols):
            rows.append(np.concatenate(ims[a:a + cols], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)
