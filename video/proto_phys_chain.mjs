// Usage: node proto_phys_chain.mjs <out.json> -- one chain stitch where only the hook is animated; the yarn is a
// physical rod (engine/physyarn.mjs): gravity, fixed length, bending, friction on the hook, contact with itself.
// The other hand pinches the slip knot (fixed), the yarn hand feeds through a fixed eyelet.
import fs from 'fs';
import { PhysYarn, resample } from './engine/physyarn.mjs';
import { hookFrame, toHookLocal, HOOK } from './engine/hook.mjs';
if (process.env.NECK) HOOK.rNeck = +process.env.NECK;
if (process.env.RSHAFT) HOOK.rShaft = +process.env.RSHAFT;
import { add, sub, mul, madd, norm, lerp, smooth, dot, clamp, len, cross } from './engine/vec.mjs';

const out = process.argv[2] || '/tmp/pc.json', FPS = 30;
const env = (k, d) => (process.env[k] != null ? process.env[k].split(',').map(Number) : d);
const R = +(process.env.LOOPR || HOOK.rShaft + 1.4); // loop radius round the shaft
const X0 = 10; // loop position along the shaft (world x)
const PX = +(process.env.PX || 0);
const Pf = [X0 + PX + 0.5, -5, 0.9], Pb = [X0 + PX - 0.5, -5, -0.9];
const E = env('EYE', [0, 8, -12]);
const loop = [];
for (let k = 0; k <= 40; k++) {
  const f = 0.35 + (k / 40) * (2 * Math.PI - 0.7);
  loop.push([X0 + 0.5 - (k / 40), -R * Math.cos(f), R * Math.sin(f)]);
}
const tailEnd = [X0 + 0.7, -26, 1.2];
const path = [tailEnd, Pf, ...loop, Pb, E];
const pts = resample(path, 0.55);
const sim = new PhysYarn(pts, { eye: { p: E }, mu: +(process.env.MU || 0.35), drag: +(process.env.DRAG || 18) });
if (process.env.RHOOK) sim.rHook = +process.env.RHOOK;
// the pinch: the yarn between y = -6.5 and -4 (both legs of the slip knot) is held by thumb and middle finger
pts.forEach((p, i) => { if (p[1] < -4 && p[1] > -6.5 && Math.abs(p[0] - X0 - PX) < 1.5) sim.pin.set(i, p.slice()); });
console.log(`particles ${pts.length}, pinned ${sim.pin.size}`);

// hook keys: tip, a (toward the tip), d (throat side)
const A = [-1, 0, 0];
const D_UP = [0, 1, 0.15];
// throat faces where the wrapped yarn presses on the hook: away from the two directions the yarn leaves to
const cThroat = [2, 0, 0];
const press = norm(mul(add(norm(sub(lerp(Pf, Pb, 0.5), cThroat)), norm(sub(E, cThroat))), -1));
const D_CATCH = env('DCATCH', press);
console.log('catch throat dir', D_CATCH.map((v) => v.toFixed(2)).join(','));
const keys = [];
let T = 0, pose = { tip: [0, 0, 0], a: A, d: D_UP };
keys.push({ t: 0, ...pose });
const go = (dur, ch, tag) => { T += dur; pose = { ...pose, ...ch }; keys.push({ t: T, ...pose, tag }); };
// the hand pivots the hook about the loop (the loop stays put while the head swings round the working yarn)
const PIV = [X0, 0, 0];
const aim = (tip) => ({ tip, a: norm(sub(tip, PIV)) });
const TX = +(process.env.TX || -2), DN = +(process.env.DN || -5.5), BK = +(process.env.BK || -5.5), UP = +(process.env.UP || 2.5);
go(1.0, {}, 'settle');
// the grab is planned from where the yarn actually is when the hand starts (the crocheter aims at the yarn)
const QX = -2.3; // the throat point on the hook axis
function planGrab() {
  const F = hookFrame(poseAt(sim.t));
  const Q = madd(F.tip, F.a, QX);
  let best = null;
  sim.x.forEach((q, i) => { if (sim.s[i] > sWork + 3) { const d = len(sub(q, Q)); if (!best || d < best.d) best = { d, i }; } });
  const C = sim.x[best.i].slice();
  // slide the hook along its axis until the throat is level with that point of the yarn
  const ax = dot(sub(C, Q), F.a);
  const tip1 = madd(F.tip, F.a, ax);
  let o = sub(madd(tip1, F.a, QX), C); o = madd(o, F.a, -dot(o, F.a));
  const rho = +(process.env.RHO || (HOOK.rShaft + 0.95 + 0.5));
  const e1 = norm(o), e2 = cross(F.a, e1);
  console.log(`grab: yarn point ${best.i} at ${C.map((v) => v.toFixed(1))}, distance ${best.d.toFixed(1)}, slide ${ax.toFixed(1)}, radius ${rho.toFixed(1)}`);
  go(0.7, { tip: madd(C, e1, rho).map((v, k) => v - F.a[k] * QX) }, 'slide');
  const turns = +(process.env.TURNS || 1.0), sgn = +(process.env.SGN || 1), n = 16;
  for (let k = 1; k <= n; k++) {
    const th = sgn * turns * 2 * Math.PI * (k / n);
    const P = madd(madd(C, e1, rho * Math.cos(th)), e2, rho * Math.sin(th));
    go(+(process.env.CIRC || 2.0) / n, { tip: madd(P, F.a, -QX), lin: true }, 'yarnover');
  }
  go(0.4, {}, 'turn');
  go(1.8, { tip: madd(madd(madd(C, e1, rho), F.a, -QX), F.a, -(+(process.env.PULL || 14))), lin: false }, 'pull');
  // the new loop now sits in the throat: push the hook forward so it settles onto the shaft
  const tipEnd = keys[keys.length - 1].tip;
  go(0.3, {}, 'hold');
  go(0.8, { tip: madd(tipEnd, F.a, +(process.env.FWD || 4)) }, 'forward');
  go(0.8, { tip: madd(tipEnd, F.a, +(process.env.FWD || 4) + +(process.env.PUSH || 5)) }, 'push');
  go(0.8, {}, 'rest');
}
const poseAt = (t) => {
  let i = 0; while (i < keys.length - 2 && keys[i + 1].t <= t) i++;
  if (t >= keys[keys.length - 1].t) i = keys.length - 2;
  const a = keys[i], b = keys[i + 1], u = clamp((t - a.t) / (b.t - a.t || 1)), w = b.lin ? u : smooth(u);
  return { tip: lerp(a.tip, b.tip, w), a: norm(lerp(a.a, b.a, w)), d: ctl.d || norm(lerp(a.d, b.d, w)), tag: b.tag };
};
const ctl = { d: null };
const frames = [];
let planned = false;
const t0 = Date.now();
const sWork = Math.max(...[...sim.pin.keys()].map((i) => sim.s[i]));
let turned = false;
for (let f = 0; ; f++) {
  if (!planned && sim.t >= 1.0 - 1e-6) { planned = true; planGrab(); }
  if (planned && sim.t > T + 1e-6) break;
  // the hand feels where the working yarn lies on the head and turns the throat toward it (a limited turning rate);
  // the yarn itself is never touched by this: it only changes the hook's roll
  {
    const pz = poseAt(sim.t), Fh = hookFrame(pz);
    let acc = [0, 0, 0], n = 0, i0 = -1, i1 = -1;
    sim.x.forEach((q, i) => {
      if (sim.s[i] <= sWork + 1) return;
      const [x, y, z] = toHookLocal(Fh, q);
      if (x < -7 || x > 0.3 || Math.hypot(y, z) > 2.4) return;
      acc = add(acc, norm(add(mul(Fh.d, y), mul(Fh.s, z)))); n++;
      if (i0 < 0) i0 = i; i1 = i;
    });
    const cur = Fh.d;
    if (n && ['turn', 'pull', 'hold', 'forward', 'rest'].includes(pz.tag) || (n && pz.tag === 'yarnover' && process.env.ROLLEARLY)) {
      let want = norm(acc);
      if (process.env.LEGS && i0 > 4 && i1 < sim.x.length - 5) {
        // the throat turns away from where the two legs of the caught yarn pull, so they press it into the groove
        const H = madd(Fh.tip, Fh.a, QX);
        const l0 = norm(sub(sim.x[i0 - 4], H)), l1 = norm(sub(sim.x[i1 + 4], H));
        const pl = madd(add(l0, l1), Fh.a, -dot(add(l0, l1), Fh.a));
        if (len(pl) > 0.2) want = mul(norm(pl), -1);
      }
      if (process.env.DOWN && ['turn', 'pull'].includes(pz.tag)) {
        // turn the throat down toward the hand holding the work, so the old loop rides over the back of the head
        const H = madd(Fh.tip, Fh.a, QX), K = lerp(Pf, Pb, 0.5);
        const v = sub(K, H), pv = madd(v, Fh.a, -dot(v, Fh.a));
        const down = norm(pv);
        want = norm(lerp(want, down, +process.env.DOWN));
      }
      const ang = Math.acos(clamp(dot(cur, want), -1, 1)), maxA = (+(process.env.ROLL || 5)) / FPS;
      ctl.d = ang < 1e-4 ? want : norm(lerp(cur, want, Math.min(1, maxA / ang)));
      if (f % 6 === 0) console.log(`  roll: ${n} pts, ${(ang * 57.3).toFixed(0)} deg off`);
    } else ctl.d = cur;
  }
  // the throat holds what it has caught: from the turn on, yarn lying in the throat moves with the hook until the
  // new loop is pushed back onto the shaft
  {
    const pz = poseAt(sim.t);
    if (process.env.GRIP && !sim.grip && ['turn', 'pull'].includes(pz.tag)) {
      const Fh = hookFrame(pz); sim.grip = new Map();
      sim.x.forEach((q, i) => { if (sim.s[i] <= sWork + 1) return; const l = toHookLocal(Fh, q); if (l[0] > +(process.env.GX0 || -3.6) && l[0] < -0.6 && Math.hypot(l[1], l[2]) < HOOK.rShaft + +(process.env.GR || 0.9)) sim.grip.set(i, l); });
      console.log(`grip: ${sim.grip.size} yarn points held in the throat`);
      if (!sim.grip.size) sim.grip = null;
    }
    if (sim.grip && sim.grip.size && pz.tag === (process.env.REL || 'rest')) { sim.grip.clear(); console.log('release'); }
  }
  if (f) sim.step(1 / FPS, poseAt);
  const p = poseAt(sim.t);
  const hk = { tip: p.tip, a: p.a, d: norm(madd(p.d, p.a, -dot(p.d, p.a))) };
  frames.push({ pts: sim.x.map((q) => q.map((v) => Math.round(v * 1000) / 1000)), s: sim.s.slice(), hook: hk, tag: p.tag });
  if (f % +(process.env.EVERY || 15) === 0) {
    let stretch = 0; for (let i = 0; i < sim.x.length - 1; i++) stretch = Math.max(stretch, Math.abs(len(sub(sim.x[i + 1], sim.x[i])) - sim.rest[i]));
    const nc = (sim.contact || []).reduce((a, b) => a + b, 0);
    const Fh = hookFrame(hk); let thr = 0, minX = 0;
    let thrOld = 0; sim.x.forEach((q, i) => { const [x, y, z] = toHookLocal(Fh, q); if (x > -4.2 && x < -1.2 && Math.hypot(y, z) < HOOK.rShaft + 0.75) { if (sim.s[i] > sWork + 1) thr++; else thrOld++; } });
    let wb = null; sim.x.forEach((q, i) => { if (sim.s[i] <= sWork + 1) return; const l = toHookLocal(Fh, q); const r = Math.hypot(l[1], l[2]); const xx = Math.max(0, l[0]); const dd = Math.hypot(xx, r); if (!wb || dd < wb.d) wb = { d: dd, x: l[0], y: l[1], z: l[2] }; });
    const W = wb ? `work@x${wb.x.toFixed(1)} y${wb.y.toFixed(1)} z${wb.z.toFixed(1)}` : '';
    let oldX = []; sim.x.forEach((q, i) => { if (sim.s[i] > 22 && sim.s[i] < sWork - 1) { const l = toHookLocal(Fh, q); if (Math.hypot(l[1], l[2]) < 3.2 && l[0] < 1) oldX.push(l[0]); } });
    const OL = oldX.length ? `old@${Math.min(...oldX).toFixed(1)}..${Math.max(...oldX).toFixed(1)}(${oldX.length})` : 'old:off';
    console.log(`f${f} ${p.tag} ${OL} ${W} throat=${thr}/old${thrOld} n=${sim.x.length} fed=${sim.s[sim.s.length - 1].toFixed(1)} maxStretch=${stretch.toFixed(3)} hookContacts=${nc}`);
  }
}
console.log(`sim ${(Date.now() - t0) / 1000}s`);
fs.writeFileSync(out, JSON.stringify({ fps: FPS, r: 0.95, frames, cam: { pos: [12, 4, 50], look: [8, -3, 0], fov: 40 } }));
