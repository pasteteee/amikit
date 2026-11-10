"""mkvideo.py NAME out.mp4 [W H] : render saved frames of a stage to a video."""
import pickle, sys
sys.path.insert(0, '.')
from crochetsim.video import write_video
from crochetsim.render import Camera
from crochetsim.hook import default_hook
S = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
caps = {'A': 'Кольцо: крючок вытягивает первую петлю', 'B': 'Накид и протягивание: первый столбик',
        'C': 'Второй столбик: ввод в кольцо, подхват, две петли'}
fr = []
for st in sys.argv[1].split(','):
    part = pickle.load(open(S + st + '_frames.pkl', 'rb'))
    for f in part:
        f['cap'] = caps.get(st, '')
    fr += part
    fr += [part[-1]] * 12
W = int(sys.argv[3]) if len(sys.argv) > 3 else 720
H = int(sys.argv[4]) if len(sys.argv) > 4 else 720
write_video(fr, default_hook(yarn_r=1.0), sys.argv[2], Camera([-34, -52, 52], [5, 1, 7], fov=30), size=(W, H), ss=2)
print('frames', len(fr), 'seconds', len(fr) / 30)
