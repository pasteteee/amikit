// Usage: node export_targets.mjs <scenario.json> <out.jsonl> [step]
// Dumps the stitch engine's per-frame target strand (the yarn path the choreography implies), the hook pose and the
// finger, plus which bits of yarn are held (by the hook, by the finished fabric) -- the physics layer (Newton) makes
// the actual yarn from this.
import fs from 'fs';
import { Sim } from './engine/sim.mjs';
import { hookFrame, toHookLocal, surfaceRadius } from './engine/hook.mjs';
import { RY } from './engine/fabric.mjs';
const scn = JSON.parse(fs.readFileSync(process.argv[2]));
scn.meta = { ...(scn.meta || {}), follow: true }; // raw taut targets: no lag, no ad-hoc yarn separation
const out = fs.createWriteStream(process.argv[3]);
const sim = new Sim(scn);
const N = Math.ceil(sim.duration * sim.fps);
const r2 = (v) => Math.round(v * 100) / 100;
for (let f = 0; f < N; f++) {
  const st = sim.step(f / sim.fps);
  const F = hookFrame(st.hook);
  // held by the hook: the target lies on the hook surface (wrapped round the shaft or in the throat)
  const hk = st.pts.map((p) => {
    const [x, y, z] = toHookLocal(F, p);
    if (x > 1.2 || x < -40) return 0;
    const rr = Math.hypot(y, z), need = x > 0 ? 0 : surfaceRadius(x, Math.atan2(z, y));
    return rr - need - RY < 0.45 ? 1 : 0;
  });
  out.write(JSON.stringify({ f, t: r2(st.t), pts: st.pts.map((p) => p.map(r2)), s: st.s.map(r2), ff: st.ff.map((v) => Math.round(v)), hk,
    hook: { tip: st.hook.tip, a: st.hook.a, d: st.hook.d }, G: sim.G, phase: st.phase, onHook: st.onHook }) + '\n');
}
out.end();
console.log('frames', N);
