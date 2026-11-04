// Shorts v2 overlay: progress bar, step chip, outlined captions, stitch counter, hook highlights,
// intro/outro cards. Layout keeps text inside the Shorts safe zone (no UI-covered bottom/right areas).
const W = 1080, H = 1920;
let toyImg = null;
const C = {
  ink: '#3a1d3a', accent: '#7b2d6b', hi: '#ffd447', white: '#fffaf5', soft: 'rgba(58,29,58,0.82)',
};

export async function loadShortAssets(opts) {
  const faces = [
    new FontFace('Nunito', 'url(/web/fonts/nunito-cyr.woff2)', { weight: '200 1000', unicodeRange: 'U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116' }),
    new FontFace('Nunito', 'url(/web/fonts/nunito-lat.woff2)', { weight: '200 1000' }),
  ];
  for (const f of faces) { await f.load(); document.fonts.add(f); }
  if (opts.toyImage) {
    toyImg = new Image(); toyImg.src = opts.toyImage;
    await toyImg.decode();
  }
  if (opts.palette) Object.assign(C, opts.palette);
}

const easeOut = (x) => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 3);
const easeBack = (x) => { x = Math.min(1, Math.max(0, x)); const c1 = 1.7, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };

function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
function wrap(ctx, text, maxW) {
  const words = text.split(' '); const lines = []; let cur = '';
  for (const w of words) { const t = cur ? cur + ' ' + w : w; if (ctx.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t; }
  if (cur) lines.push(cur); return lines;
}
function strokeText(ctx, text, x, y, fill, stroke, sw) {
  ctx.lineJoin = 'round'; ctx.miterLimit = 2;
  ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.strokeText(text, x, y);
  ctx.fillStyle = fill; ctx.fillText(text, x, y);
}

export function drawShortOverlay(ctx, scale, ov, opts) {
  ctx.save(); ctx.scale(scale, scale);
  // soft vignette for depth
  const vg = ctx.createRadialGradient(W / 2, H * 0.45, H * 0.28, W / 2, H * 0.5, H * 0.78);
  vg.addColorStop(0, 'rgba(60,30,40,0)'); vg.addColorStop(1, 'rgba(60,30,40,0.22)');
  ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);

  const introOn = ov.intro >= 0 && ov.intro < 1;
  // progress bar
  rr(ctx, 70, 64, W - 140, 12, 6); ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.fill();
  rr(ctx, 70, 64, Math.max(12, (W - 140) * ov.progress), 12, 6); ctx.fillStyle = C.accent; ctx.fill();

  // step chip + label
  if (ov.step && !introOn) {
    const s = ov.step, k = easeBack(s.age / 0.35);
    ctx.save();
    ctx.translate(80, 150); ctx.scale(k, k);
    ctx.font = '900 34px Nunito'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    const chip = `ШАГ ${s.n}/${s.of}`;
    const cw = ctx.measureText(chip).width + 44;
    rr(ctx, 0, -28, cw, 56, 28); ctx.fillStyle = C.accent; ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillText(chip, 22, 2);
    ctx.font = '900 62px Nunito';
    strokeText(ctx, s.label, 0, 88, C.ink, 'rgba(255,250,245,0.95)', 14);
    ctx.restore();
  }
  // stitch counter + speed (left side, below the label)
  if (ov.counter && !introOn && ov.outro < 0) {
    ctx.font = '900 40px Nunito'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    const t = `Р${ov.counter.round} · ${ov.counter.j}/${ov.counter.n}`;
    const tw = ctx.measureText(t).width;
    const ff = ov.speed > 1.5 ? `  ▶▶ ×${Math.round(ov.speed)}` : '';
    ctx.font = '800 34px Nunito'; const fw = ctx.measureText(ff).width; ctx.font = '900 40px Nunito';
    rr(ctx, 80, 290, tw + fw + 48, 64, 32); ctx.fillStyle = C.soft; ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillText(t, 104, 323);
    if (ff) { ctx.font = '800 34px Nunito'; ctx.fillStyle = C.hi; ctx.fillText(ff, 104 + tw, 323); }
  }
  // loops on the hook (detailed stitches only)
  if (ov.detail && ov.onHook > 0) {
    ctx.font = '800 36px Nunito'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    const label = 'на крючке:';
    const lw = ctx.measureText(label).width;
    const w = lw + 48 + ov.onHook * 40;
    rr(ctx, 80, 370, w, 60, 30); ctx.fillStyle = 'rgba(255,250,245,0.9)'; ctx.fill();
    ctx.fillStyle = C.ink; ctx.fillText(label, 102, 401);
    for (let k = 0; k < ov.onHook; k++) {
      ctx.beginPath(); ctx.arc(102 + lw + 26 + k * 40, 401, 13, 0, Math.PI * 2);
      ctx.lineWidth = 7; ctx.strokeStyle = C.accent; ctx.stroke();
    }
  }
  // hook highlights: pulsing ring at the throat while it catches the yarn, arrow while pulling
  if (ov.detail && ov.throat && ov.phase) {
    const [x, y] = ov.throat;
    const catching = (ov.phase === 'yo1' || ov.phase === 'yo2');
    const pulling = (ov.phase === 'pullup' || ov.phase === 'through1' || ov.phase === 'through2');
    if (catching) {
      const a = Math.min(1, ov.phaseU * 4) * (1 - Math.max(0, (ov.phaseU - 0.85) / 0.15));
      const r = 46 + 8 * Math.sin(ov.t * 9);
      ctx.save(); ctx.globalAlpha = 0.9 * a;
      ctx.shadowColor = C.hi; ctx.shadowBlur = 24;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.lineWidth = 8; ctx.strokeStyle = C.hi; ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.font = '900 38px Nunito'; ctx.textAlign = 'center';
      strokeText(ctx, 'накид', x, y - r - 22, C.ink, C.hi, 12);
      ctx.restore();
    }
    if (pulling) {
      const [x2, y2] = ov.pullTo;
      const dx = x2 - x, dy = y2 - y, L = Math.hypot(dx, dy);
      if (L > 30) {
        const ux = dx / L, uy = dy / L, len = Math.min(170, L) * easeOut(ov.phaseU * 2.5);
        const a = 1 - Math.max(0, (ov.phaseU - 0.8) / 0.2);
        const sx = x + ux * 40 - uy * 46, sy = y + uy * 40 + ux * 46; // offset beside the hook
        const ex = sx + ux * len, ey = sy + uy * len;
        ctx.save(); ctx.globalAlpha = a;
        ctx.lineCap = 'round'; ctx.lineWidth = 14; ctx.strokeStyle = C.hi; ctx.shadowColor = 'rgba(0,0,0,0.25)'; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ex + ux * 18, ey + uy * 18); ctx.lineTo(ex - ux * 18 - uy * 24, ey - uy * 18 + ux * 24); ctx.lineTo(ex - ux * 18 + uy * 24, ey - uy * 18 - ux * 24); ctx.closePath();
        ctx.fillStyle = C.hi; ctx.fill();
        ctx.restore();
      }
    }
  }
  // caption (centre-lower safe zone, max 3 lines)
  if (ov.cap && !introOn) {
    const k = easeOut(ov.cap.age / 0.25);
    ctx.save(); ctx.globalAlpha = k;
    ctx.font = '900 56px Nunito'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    const lines = wrap(ctx, ov.cap.text, 860);
    const y0 = 1330 - (lines.length - 1) * 36 + (1 - k) * 24;
    lines.forEach((l, i) => strokeText(ctx, l, W / 2, y0 + i * 72, '#ffffff', C.ink, 16));
    ctx.restore();
  }
  // intro card: result first
  if (introOn || (ov.intro >= 1 && false)) drawCard(ctx, ov.intro, opts.meta, 'intro');
  if (ov.outro >= 0) drawCard(ctx, ov.outro, opts.meta, 'outro');
  ctx.restore();
}

function drawCard(ctx, u, meta, kind) {
  const appear = kind === 'intro' ? 1 - easeOut((u - 0.82) / 0.18) : easeBack((u - 0.25) / 0.3);
  if (appear <= 0.001) return;
  const slide = kind === 'intro' ? (1 - appear) * -300 : 0;
  ctx.save();
  ctx.globalAlpha = Math.min(1, appear * 1.2);
  // dim the scene a little behind the card
  ctx.fillStyle = `rgba(255,248,240,${0.35 * Math.min(1, appear)})`; ctx.fillRect(0, 0, W, H);
  const s = 0.6 + 0.4 * appear;
  const cx = W / 2, cy = (kind === 'intro' ? 640 : 610) + slide;
  ctx.translate(cx, cy); ctx.rotate((kind === 'intro' ? -3 : 3) * Math.PI / 180); ctx.scale(s, s);
  const cw = 600, ch = 600;
  ctx.shadowColor = 'rgba(60,20,50,0.35)'; ctx.shadowBlur = 40; ctx.shadowOffsetY = 16;
  rr(ctx, -cw / 2 - 18, -ch / 2 - 18, cw + 36, ch + 36, 46); ctx.fillStyle = '#fff'; ctx.fill();
  ctx.shadowColor = 'transparent';
  if (toyImg) { ctx.save(); rr(ctx, -cw / 2, -ch / 2, cw, ch, 32); ctx.clip(); ctx.drawImage(toyImg, -cw / 2 - 60, -ch / 2 - 40, cw + 120, ch + 120); ctx.restore(); }
  ctx.restore();
  ctx.save(); ctx.globalAlpha = Math.min(1, appear * 1.3);
  ctx.textAlign = 'center';
  const ty = (kind === 'intro' ? 1060 : 1030) + slide;
  ctx.font = '900 84px Nunito';
  const head = kind === 'intro' ? (meta.title || '') : (meta.outroHead || 'Готово!');
  wrap(ctx, head, 920).forEach((l, i) => strokeText(ctx, l, W / 2, ty + i * 92, C.ink, '#fffaf5', 18));
  ctx.font = '800 46px Nunito';
  const sub = kind === 'intro' ? (meta.sub || '') : (meta.outroSub || '');
  const nHead = wrap(ctx, head, 920).length;
  if (sub) wrap(ctx, sub, 900).forEach((l, i) => strokeText(ctx, l, W / 2, ty + nHead * 92 + 4 + i * 58, C.accent, '#fffaf5', 12));
  ctx.restore();
}
