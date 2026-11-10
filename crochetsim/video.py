"""Frames -> mp4.  The working yarn is drawn on beyond the yarn hand to a ball off screen."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render import Renderer


def extend_to_ball(X, D, F, ball, h):
    """Continue the yarn from its last node through the yarn hand F to the ball."""
    if F is None:
        return X, D
    pts = [X[-1]]
    for target in (np.asarray(F, float), np.asarray(ball, float)):
        p = pts[-1]
        v = target - p
        L = np.linalg.norm(v)
        k = int(L / h)
        if k >= 1:
            pts.extend(list(p + v[None, :] / L * (h * np.arange(1, k + 1))[:, None]))
    E = np.array(pts[1:], np.float32)
    if len(E) == 0:
        return X, D
    Xe = np.concatenate([X, E]).astype(np.float32)
    De = np.zeros_like(Xe)
    De[:len(D)] = D
    d = D[len(X) - 2].astype(np.float64)
    for i in range(len(X) - 1, len(Xe) - 1):
        t = Xe[i + 1] - Xe[i]
        t = t / (np.linalg.norm(t) + 1e-30)
        d = d - t * (d @ t)
        d = d / (np.linalg.norm(d) + 1e-30)
        De[i] = d
    return Xe, De


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def write_video(frames, hp, path, cam, size=(540, 960), fps=30, ss=2, ball=(-60.0, 30.0, 140.0), h=0.8, r=1.0,
                every=1, progress=None):
    W, H = size
    rn = Renderer(W, H, ss=ss, r=r, h=h)
    font = _font(max(14, H // 34))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(0, len(frames), every):
        f = frames[k]
        X, D = extend_to_ball(f["X"], f["D"], f.get("F"), ball, h)
        c = cam(f) if callable(cam) else cam
        im = rn.render(X, D, f["R"], f["t"], hp, c)
        if f.get("cap"):
            img = Image.fromarray(im)
            dr = ImageDraw.Draw(img)
            tw = dr.textlength(f["cap"], font=font)
            x0 = (W - tw) / 2
            y0 = H - int(H * 0.085)
            dr.rounded_rectangle([x0 - 14, y0 - 8, x0 + tw + 14, y0 + font.size + 12], 12, fill=(40, 36, 34))
            dr.text((x0, y0), f["cap"], font=font, fill=(250, 246, 240))
            im = np.asarray(img)
        p.stdin.write(np.ascontiguousarray(im).tobytes())
        if progress and k % (30 * every) == 0:
            progress(k, len(frames))
    p.stdin.close()
    p.wait()
