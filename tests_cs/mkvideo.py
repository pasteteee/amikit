"""mkvideo.py NAME out.mp4 [W H] : render saved frames of a stage to a video."""
import pickle, sys
sys.path.insert(0, '.')
from crochetsim.video import write_video
from crochetsim.render import Camera
from crochetsim.hook import default_hook
S = '/tmp/claude-0/-home-user/9927774d-73d8-54c0-b17a-f988f34635c2/scratchpad/'
fr = pickle.load(open(S + sys.argv[1] + '_frames.pkl', 'rb'))
na = len(pickle.load(open(S + 'A_frames.pkl', 'rb'))); nb = len(pickle.load(open(S + 'B_frames.pkl', 'rb')))
for i, f in enumerate(fr):
    f['cap'] = ('Кольцо: крючок вытягивает первую петлю' if i < na else
                'Накид и протягивание: первый столбик' if i < nb else
                'Второй столбик: ввод в кольцо, подхват, две петли')
W = int(sys.argv[3]) if len(sys.argv) > 3 else 720
H = int(sys.argv[4]) if len(sys.argv) > 4 else 720
write_video(fr, default_hook(yarn_r=1.0), sys.argv[2], Camera([-34, -52, 52], [5, 1, 7], fov=30), size=(W, H), ss=2)
print('frames', len(fr), 'seconds', len(fr) / 30)
