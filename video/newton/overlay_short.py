"""Draw the Shorts overlay (progress bar, step chip, outlined captions, loop counter) over rendered frames.

Usage: python overlay_short.py <frames_dir> <sim_log> <out_dir> <fps> <n_stitches> <cyr.ttf> <lat.ttf>
The sim log holds lines "t=1.23 move 4 pull -> ..." written by chain_newton.py; captions follow those moves.
"""
import os, re, sys
from PIL import Image, ImageDraw, ImageFont

src, log, dst, fps, nst, f_cyr, f_lat = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
os.makedirs(dst, exist_ok=True)
PINK, INK, WHITE = (247, 126, 160), (58, 34, 72), (255, 255, 255)

moves = []  # (t, tag)
for l in open(log, encoding="utf8"):
    m = re.match(r"t=([\d.]+) move \d+ (\w+)", l)
    if m: moves.append((float(m.group(1)), m.group(2)))

CAP = {
    "None": "Петля на крючке, нить на пальце",
    "reach": "Накид: крючок идёт к нитке",
    "catch": "Нитка ложится в зев крючка",
    "turn": "Поверните зев вниз",
    "pull": "Протяните через петлю на крючке",
    "rotate": "Старая петля соскользнула",
    "home": "Новая петля на крючке",
}


def font(path, size, weight=800):
    f = ImageFont.truetype(path, size)
    try: f.set_variation_by_axes([weight])
    except Exception: pass
    return f


def runs(text):
    out, cur, cyr = [], "", None
    for ch in text:
        c = "Ѐ" <= ch <= "ӿ"
        if cyr is None or c == cyr or ch == " ": cur += ch; cyr = c if ch != " " and cyr is None else cyr
        else: out.append((cur, cyr)); cur, cyr = ch, c
    if cur: out.append((cur, cyr))
    return out


def text_w(d, text, size, weight=800):
    return sum(d.textlength(t, font=font(f_cyr if c else f_lat, size, weight)) for t, c in runs(text))


def draw_text(d, xy, text, size, fill, stroke=0, stroke_fill=None, weight=800):
    x, y = xy
    for t, c in runs(text):
        f = font(f_cyr if c else f_lat, size, weight)
        d.text((x, y), t, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        x += d.textlength(t, font=f)


def wrap(d, text, size, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if text_w(d, t, size) <= maxw: cur = t
        else: lines.append(cur); cur = w
    lines.append(cur)
    return lines


files = sorted(f for f in os.listdir(src) if f.endswith(".jpg"))
N = len(files)
for i, fn in enumerate(files):
    t = i / fps
    im = Image.open(os.path.join(src, fn)).convert("RGB")
    W, H = im.size
    k = W / 1080
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    # which stitch and which move
    tag, idx = "None", 0
    for j, (tm, tg) in enumerate(moves):
        if t >= tm: tag = tg; idx = j
    stitch = min(nst, max(1, sum(1 for tm, tg in moves if tg == "reach" and tm <= t + 1e-6)))
    done = sum(1 for tm, tg in moves if tg == "home" and tm + 0.9 <= t)
    # progress bar
    d.rounded_rectangle([60 * k, 70 * k, W - 60 * k, 82 * k], radius=6 * k, fill=(255, 255, 255, 150))
    d.rounded_rectangle([60 * k, 70 * k, 60 * k + (W - 120 * k) * (i + 1) / N, 82 * k], radius=6 * k, fill=PINK + (255,))
    # step chip + title
    chip = f"ПЕТЛЯ {stitch}/{nst}"
    cw = text_w(d, chip, 40 * k) + 48 * k
    d.rounded_rectangle([60 * k, 112 * k, 60 * k + cw, 178 * k], radius=33 * k, fill=PINK + (255,))
    draw_text(d, (84 * k, 118 * k), chip, 40 * k, WHITE)
    draw_text(d, (60 * k + cw + 24 * k, 118 * k), "Воздушная петля", 40 * k, INK, stroke=int(5 * k), stroke_fill=WHITE)
    # caption (bottom safe zone)
    cap = CAP.get(tag, "")
    if t > N / fps - 1.2: cap = f"Готово: {nst} воздушные петли" if nst > 1 else "Готово: воздушная петля"
    lines = wrap(d, cap, 66 * k, W - 160 * k)
    y = H * 0.70
    for ln in lines:
        w = text_w(d, ln, 66 * k)
        draw_text(d, ((W - w) / 2, y), ln, 66 * k, WHITE, stroke=int(9 * k), stroke_fill=INK)
        y += 84 * k
    # loops counter
    lab = f"готово {done} из {nst}"
    lw = text_w(d, lab, 36 * k, 700)
    d.rounded_rectangle([W - 60 * k - lw - 40 * k, H * 0.82, W - 60 * k, H * 0.82 + 60 * k], radius=30 * k, fill=(255, 255, 255, 210))
    draw_text(d, (W - 60 * k - lw - 20 * k, H * 0.82 + 8 * k), lab, 36 * k, INK, weight=700)
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    im.save(os.path.join(dst, fn), quality=92)
print("overlay", N, "frames")
