// Usage: node proto_rope_chain.mjs <out.json> [stitches] -- chain stitches with the contact-driven yarn (engine/rope.mjs)
import fs from 'fs';
import { Rope } from './engine/rope.mjs';
import { hookFrame } from './engine/hook.mjs';
import { madd, norm, sub, lerp, smooth, dot, clamp } from './engine/vec.mjs';
const out = process.argv[2], NST = +(process.argv[3] || 1), FPS = 30;
const G0 = (process.env.G0 || "-4,-6,-20").split(",").map(Number), GR = Math.hypot(G0[1], G0[2]), GTH = Math.atan2(G0[2], G0[1]);
const fing = (x, r, th) => [x, r * Math.cos(th), r * Math.sin(th)];
const rope = new Rope([
  { type: 'fix', p: [10, -3.2, 0.3] }, // slip knot, pinched by the other hand
  { type: 'ring', x: -10, w: [10, 0, 0], R: 2.0 },
  { type: 'finger', p: G0.slice() },
], { tailLen: 30, log: (m) => console.log(m) });
rope.debug = !!process.env.DBG; rope.catchAny = !process.env.SIDECATCH;
// key poses: hook {tip, a, d}, finger cylindrical {x, r, th}
const D_UP = [0, 1, 0.15];
let pose = { tip: [0, 0, 0], a: [-1, 0, 0], d: D_UP, fx: G0[0], fr: GR, fth: GTH };
const keys = [{ t: 0, ...pose }];
let T = 0;
const go = (dur, ch, tag) => { T += dur; pose = { ...pose, ...ch }; keys.push({ t: T, ...pose, tag }); };
const ORB = +(process.env.ORB || 0.6) * 2 * Math.PI;
let baseX = 0;
for (let k = 0; k < NST; k++) {
  go(0.6, { fx: baseX + +(process.env.FX || 2.2), fr: +(process.env.FR || 9) }, "approach");
  for (let q = 1; q <= 6; q++) go(0.25, { fth: pose.fth + ORB / 6 }, 'wrap');
  go(0.5, { fr: GR, fx: G0[0] + baseX }, 'retreat');
  go(0.6, { d: [0, -0.55, -0.83] }, 'turn');
  go(1.3, { tip: [baseX + 14, 0, 0] }, 'pull');
  go(0.2, {}, 'release');
  baseX += 3.5;
  go(0.9, { tip: [baseX, 0, 0], d: D_UP }, 'forward');
}
go(0.5, {}, 'end');
const at = (t) => {
  let i = 0; while (i < keys.length - 2 && keys[i + 1].t <= t) i++;
  const A = keys[i], B = keys[i + 1], w = smooth((t - A.t) / (B.t - A.t || 1));
  const L = (u, v) => (Array.isArray(u) ? lerp(u, v, w) : u + (v - u) * w);
  return { tip: L(A.tip, B.tip), a: norm(L(A.a, B.a)), d: norm(L(A.d, B.d)), fx: L(A.fx, B.fx), fr: L(A.fr, B.fr), fth: L(A.fth, B.fth), tag: B.tag };
};
const frames = [];
const knots = [];
let lastTag = null;
const nF = Math.ceil(T * FPS);
for (let f = 0; f <= nF; f++) {
  const t = f / FPS, p = at(t);
  const F = hookFrame({ tip: p.tip, a: p.a, d: p.d });
  // the finger eyelet
  rope.nodes[rope.nodes.length - 1].p = fing(p.fx, p.fr, p.fth);
  rope.t = t;
  // event: after the pull the caught yarn is the new loop on the hook
  if (p.tag === 'forward' && lastTag !== 'forward') {
    for (const n of rope.nodes) if (n.type === 'throat') { n.type = 'ring'; n.w = madd(F.tip, F.a, n.x); n.xMax = -1.0; console.log(`t=${t.toFixed(2)} new loop on the hook`); }
  }
  lastTag = p.tag;
  for (let sub = 0; sub < 4; sub++) rope.update(F, 1 / FPS / 4);
  // the other hand pulls each finished knot down under the loop on the hook (the chain hangs)
  const ks = rope.nodes.filter((n) => n.type === 'knot');
  ks.forEach((n, i) => {
    const age = ks.length - 1 - i;
    const ring = rope.nodes.find((m) => m.type === 'ring');
    const top = ring ? ring.w : n.c;
    const tgt = [top[0], top[1] - 3.4 - 3.2 * age, 0.3];
    n.c = lerp(n.c, tgt, clamp((1 / FPS) / 0.25));
    const nx = rope.nodes[rope.nodes.indexOf(n) + 1];
    const up = nx ? sub(rope.center(nx, F), n.c) : [0, 1, 0];
    n.axis = norm(lerp(n.axis, norm(up), 0.3)); n.R = Math.max(1.75, n.R - (1 / FPS) * 3);
  });
  // the other hand holds the chain just under the work: the slip knot follows the oldest knot down
  if (ks.length) {
    const fx = rope.nodes[0];
    const tg = [ks[0].c[0], ks[0].c[1] - 3.2, 0.3];
    fx.p = lerp(fx.p, tg, clamp((1 / FPS) / 0.4));
  }
  const st = rope.strand(F, madd(rope.nodes[0].p, [0, -1, 0.12], 28));
  frames.push({ pts: st.pts.map((q) => q.map((v) => Math.round(v * 1000) / 1000)), s: st.s, hook: { tip: F.tip, a: F.a, d: F.d }, nodes: st.nodes.join(',') });
  if (f % 6 === 0) console.log(`f${f} ${p.tag} ` + rope.nodes.map((n) => n.type + (n.x != null ? '@' + n.x.toFixed(1) : '')).join(' ') + ` fing ${rope.nodes[rope.nodes.length-1].p.map(v=>v.toFixed(1))}`);
}
fs.writeFileSync(out, JSON.stringify({ fps: FPS, r: 0.95, frames, cam: { pos: [10, 8, 52], look: [6, -2, 0], fov: 36 } }));
