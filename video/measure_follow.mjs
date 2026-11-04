// Usage: node measure_follow.mjs <scn.json> [follow 0|1] [seconds] -- smoothness of the rendered yarn
// (max per-frame jump of a material point beyond the hook's own motion, radius change per frame).
import fs from 'fs';
import { Sim } from './engine/sim.mjs';
import { YarnFollower } from './engine/follow.mjs';
const scn = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const useF = process.argv[3] !== '0';
scn.meta.follow = useF ? true : undefined;
const sim = new Sim(scn), fol = useF ? new YarnFollower() : null;
const T = Math.min(sim.duration, +(process.argv[4] || 1e9));
let prev = null, worst = [], rJump = 0, t0 = Date.now(), nF = 0;
for (let f = 0; f <= T * sim.fps; f++) {
  let st = sim.step(f / sim.fps); if (fol) st = fol.step(st, 1 / sim.fps); nF++;
  if (prev) {
    // compare by material coordinate
    const m = new Map(); prev.s.forEach((s, i) => m.set(Math.round(s * 100), i));
    let mx = 0, mi = -1;
    st.s.forEach((s, i) => { const j = m.get(Math.round(s * 100)); if (j == null) return; const a = st.pts[i], b = prev.pts[j]; const dd = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); if (dd > mx) { mx = dd; mi = i; } if (st.r && prev.r) rJump = Math.max(rJump, Math.abs(st.r[i] - prev.r[j])); });
    const hm = Math.hypot(...[0, 1, 2].map((k) => st.hook.tip[k] - prev.hook.tip[k]));
    worst.push(mx - hm); if (process.env.V && mx - hm > 3) console.log(f, (mx - hm).toFixed(2), st.act?.type, st.act?.speed, st.phase, 's', st.s[mi].toFixed(1), 'of', st.s[0].toFixed(0), st.s[st.s.length-1].toFixed(0), 'ff', st.ff ? 'y' : '', 'p', st.pts[mi].map(c=>c.toFixed(1)).join(','));
  }
  prev = { s: st.s.slice(), pts: st.pts.map((p) => p.slice()), r: st.r?.slice(), hook: { tip: st.hook.tip.slice() } };
}
worst.sort((a, b) => b - a);
console.log(`follow=${useF} frames ${nF} ${((Date.now() - t0) / nF).toFixed(1)} ms/frame; excess jump p50 ${worst[worst.length >> 1].toFixed(2)} p99 ${worst[Math.floor(worst.length * 0.01)].toFixed(2)} max ${worst[0].toFixed(2)} mm; max radius jump ${rJump.toFixed(3)} mm`);
