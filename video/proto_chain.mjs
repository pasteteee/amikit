// Prototype of the persistent yarn simulation: a loop on the hook, yarn over, pull through (chain
// stitches). Usage: node proto_chain.mjs <out.json> [nStitches]
import fs from 'fs';
import { YarnSim } from './engine/yarnsim.mjs';
import { HOOK } from './engine/hook.mjs';

const h = 0.6, R_Y = 0.95;
const nSt = +(process.argv[3] || 2);
const KEEP_BACK = +(process.argv[4] || 2);
const PULL_PAST = +(process.argv[5] || 1.5);
const HOLD_GAP = +(process.argv[6] || 0.8);
let ringFrom = 0, ringEnd = 0; const CATCH_K = +(process.env.CK || 0.35), HOLD_K = +(process.env.HK || 0.25); const DBG = process.env.DBG;
const norm = (v) => { const l = Math.hypot(...v) || 1; return v.map((c) => c / l); };
const G = [-9, 24, -16];
const dirBall = norm([-0.35, 0.55, -0.75]);

// ---- initial yarn: held tail (pinned) -> loop around the shaft -> working yarn to the finger
const pts = [];
const X0 = 6.5, RL = HOOK.rShaft + R_Y + 0.05;
for (let y = -34; y < -RL - 0.3; y += h) pts.push([X0, y, 0.6]);
const nTail = pts.length;
const turns = 1.25, steps = Math.ceil((turns * 2 * Math.PI * RL) / h);
for (let k = 1; k <= steps; k++) {
  const th = -Math.PI / 2 + (k / steps) * turns * 2 * Math.PI;
  pts.push([X0 + 2.0 * (k / steps) * turns, RL * Math.sin(th), RL * Math.cos(th)]);
}
{
  const a = pts[pts.length - 1];
  const L = Math.hypot(G[0] - a[0], G[1] - a[1], G[2] - a[2]);
  const m = Math.floor(L / h);
  for (let k = 1; k <= m; k++) pts.push([a[0] + (G[0] - a[0]) * k / m, a[1] + (G[1] - a[1]) * k / m, a[2] + (G[2] - a[2]) * k / m]);
}
const simInit = 0;
const sim = new YarnSim(pts, { h, r: R_Y, finger: { G, dir: dirBall, T: 120000 }, substeps: 30, iters: 6, muHook: 0.4, gravity: [0, -300, 0] });
for (let i = 0; i < nTail; i++) sim.pin(i, pts[i]);
sim.keepFrom = nTail; ringFrom = nTail;

// ---- hook choreography: each move's target is computed from the yarn's actual state when it starts
const P = (tip, a, d) => ({ tip, a: norm(a), d: norm(d) });
const HOME = P([0, 0, 0], [-1, 0, 0], [0, 0.2, -1]);
const add = (u, v) => u.map((c, k) => c + v[k]), mul = (u, f) => u.map((c) => c * f);
const throatToTip = (C, a) => add(C, mul(a, -HOOK.xThroat)); // tip = throat - a*xThroat
function workingYarnPoint(xPlane) {
  // the free working yarn between the loop on the hook and the finger, where it crosses the plane x = xPlane
  const ps = sim.positions();
  for (let i = ps.length - 2; i > nTail; i--) {
    const A = ps[i], B = ps[i + 1];
    if ((A[0] - xPlane) * (B[0] - xPlane) <= 0 && A[1] > 1.2) { const t = (xPlane - A[0]) / ((B[0] - A[0]) || 1e-9); return add(A, mul(add(B, mul(A, -1)), t)); }
  }
  return [xPlane, 9, -5];
}
const moves = [];
const move = (dur, fn, tag) => moves.push({ dur, fn, tag });
move(0.6, () => HOME);
for (let k = 0; k < nSt; k++) {
  const v = norm([0, 0.8, -0.6]); // the throat opens toward the yarn, the hook approaches along it
  const a1 = norm([-1, 0.08, -0.1]);
  move(0.9, () => { const Y = workingYarnPoint(2.0); return P(throatToTip(add(Y, mul(v, -1.6)), a1), a1, v); }, 'reach');
  move(0.25, () => to, 'catch'); // throat touches the yarn: it lies in the groove from now on (slides freely)
  move(0.8, () => P([0.0, 0.8, -0.6], [-1, 0, 0], [0, -0.45, -0.9])); // bring it down, throat turned down
  move(1.6, () => { // pull back until the throat is just past the old loop (measured from the yarn now)
    const ps = sim.positions(); let sx = 0, c = 0;
    for (let i = nTail; i < (sim.catcher?.k ?? nTail + 20) - 8; i++) { if (Math.hypot(ps[i][1], ps[i][2]) < 4) { sx += ps[i][0]; c++; } }
    const xl = c ? sx / c : 6.5;
    return P([xl + PULL_PAST - HOOK.xThroat * 0 , 0.5, 0.0], [-1, 0, 0], [0, -0.6, -0.8]);
  }, 'pull');
  move(0.5, () => P(add(to.tip, [-0.6, 0.6, 0]), [-1, 0, 0], [0, -0.2, -1]), 'tighten'); // the hand lets go: tension closes the old loop under the hook
  move(0.3, () => to, 'release');
  move(0.9, () => P([0.0, 0.0, 0.0], [-1, 0, 0], [0, 0.2, -1])); // slide the new loop back onto the shaft
}
move(0.6, () => HOME);
const ease = (u) => u * u * u * (u * (u * 6 - 15) + 10);
const slerpPose = (A, B, u) => { const L = (p, q) => p.map((c, k) => c + (q[k] - c) * u); return { tip: L(A.tip, B.tip), a: norm(L(A.a, B.a)), d: norm(L(A.d, B.d)) }; };
let T = moves.reduce((t, m) => t + m.dur, 0);
let curMove = -1, mStart = 0, from = HOME, to = HOME;
function poseAt(t) {
  while (curMove + 1 < moves.length && t >= mStart + (curMove >= 0 ? moves[curMove].dur : 0)) {
    if (curMove >= 0) { mStart += moves[curMove].dur; from = to; }
    if (curMove >= 0) {
      const F = { tip: to.tip, a: to.a, d: norm(to.d.map((c, k) => c - to.a[k] * (to.d[0] * to.a[0] + to.d[1] * to.a[1] + to.d[2] * to.a[2]))) };
      let inG = 0, near = 1e9;
      for (const q of sim.positions()) {
        const r = q.map((c, k) => c - F.tip[k]);
        const lx = r[0] * F.a[0] + r[1] * F.a[1] + r[2] * F.a[2];
        const ly = r[0] * F.d[0] + r[1] * F.d[1] + r[2] * F.d[2];
        const rho = Math.hypot(...r.map((c, k) => c - F.a[k] * lx));
        if (lx > -2.8 && lx < -1.3) { near = Math.min(near, rho); if (rho < HOOK.rShaft + R_Y + 0.15 && ly > 0.3) inG++; }
      }
      console.log(`move ${curMove} end t=${t.toFixed(2)}: particles in throat groove ${inG}, nearest at throat ${near.toFixed(2)} mm`);
    }
    curMove++; to = moves[curMove].fn();
    if (moves[curMove].tag === 'catch') { sim.catcher = { xc: HOOK.xThroat, rc: 1.55, from: nTail + 10, stiff: CATCH_K }; }
    if (moves[curMove].tag === 'pull') { let started = false, i0r = ringFrom; const i0 = ringFrom; let i1 = i0; { const k = sim.catcher?.k ?? nTail + 30, P0 = sim.positions();
      for (let i = i0; i < k - 2; i++) { const r = P0[i].map((c, m) => c - from.tip[m]); const lx = r[0] * from.a[0] + r[1] * from.a[1] + r[2] * from.a[2];
        const rho = Math.hypot(...r.map((c, m) => c - from.a[m] * lx)); const inside = rho < 3.6 && lx < HOOK.xThroat - HOLD_GAP; if (DBG) console.log(i, lx.toFixed(1), rho.toFixed(1));
        if (inside) { if (!started) i0r = i; started = true; i1 = i; } else if (started) break; } }
      ringEnd = i1;
      console.log('ring', i0, i1); const ps = sim.positions(); sim.hold = { i0, i1, axis: 0, stiff: HOLD_K, val: ps.slice(i0, i1 + 1).map((p) => p[0]) }; sim.keepFrom = null; console.log('catcher k', sim.catcher?.k, 'nTail', nTail); } // the old loop must slide off the tip; the throat holds the new one
    if (moves[curMove].tag === 'tighten') sim.hold = null;
    if (moves[curMove].tag === 'release') { sim.keepFrom = ringEnd + KEEP_BACK; ringFrom = ringEnd + 1; sim.catcher = null; sim.hold = null; }
  }
  return slerpPose(from, to, ease(Math.min(1, (t - mStart) / moves[curMove].dur)));
}

const fps = 30, frames = [];
const t0 = Date.now();
for (let f = 0; f <= Math.ceil(T * fps); f++) {
  const pose = poseAt(f / fps);
  sim.step(1 / fps, pose);
  const ps = sim.positions();
  frames.push({ pts: ps.map((p) => p.map((c) => +c.toFixed(3))), s: sim.material().map((c) => +c.toFixed(3)), hook: { tip: pose.tip, a: pose.a, d: pose.d } });
}
const ms = (Date.now() - t0) / frames.length;
// smoothness metric: max per-frame displacement of material points vs hook motion
let worst = 0;
for (let f = 1; f < frames.length; f++) {
  const A = frames[f - 1], B = frames[f];
  const n = Math.min(A.pts.length, B.pts.length);
  let mx = 0; for (let i = 0; i < n; i++) mx = Math.max(mx, Math.hypot(...[0, 1, 2].map((k) => A.pts[i][k] - B.pts[i][k])));
  const hm = Math.hypot(...[0, 1, 2].map((k) => A.hook.tip[k] - B.hook.tip[k]));
  worst = Math.max(worst, mx - hm);
}
console.log(`frames ${frames.length}, ${ms.toFixed(1)} ms/frame, particles ${sim.n}, fed ${sim.stats.fed}, retracted ${sim.stats.retracted}, worst yarn-vs-hook excess ${worst.toFixed(2)} mm/frame`);
fs.writeFileSync(process.argv[2], JSON.stringify({ fps, r: R_Y, frames, cam: { pos: [22, 24, 78], look: [4, 3, -2], fov: 30 } }));
