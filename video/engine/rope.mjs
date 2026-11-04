// Contact-driven yarn ("rope through contacts"): the yarn is one strand of fixed material that runs through a
// chain of contact nodes. Nodes come from real contact, not from a script:
//  * a `slide` wrap appears the moment a straight span of yarn would cut into the hook; it sits on the hook surface,
//    slides along the shaft to shorten the yarn and disappears again when the yarn no longer bends round the hook;
//  * a slide wrap that slides into the throat while the throat faces it is caught (`throat`);
//  * a `ring` is a loop round the hook: it keeps its place in the world while the hook moves along its axis and
//    slips off over the tip when the hook is drawn back far enough; it then becomes a `knot` (a ring round the
//    legs of the next loop);
//  * `fix` (the other hand: tail end, slip knot) and `finger` (the eyelet the yarn feeds through).
// Between nodes the yarn is a straight span (taut, pulled by the finger) or, where it is not under tension (the
// tail), a sagging curve of its own fixed length. The material coordinate s is integrated along the path from the
// slip knot, so the ply texture slides exactly where yarn is drawn through a contact and nowhere else.
import { add, sub, mul, madd, dot, cross, norm, len, dist, lerp, perp, clamp, smoothstep } from './vec.mjs';
import { wrapArc, polyLength, closestOnSeg } from './curve.mjs';
import { HOOK, hookFrame, baseRadius } from './hook.mjs';
import { RY } from './fabric.mjs';

const R_LOOP = () => HOOK.rShaft + RY * 0.95;
const R_THR = () => HOOK.rNeck + RY * 0.95;

export class Rope {
  constructor(nodes, opts = {}) {
    this.nodes = nodes; // [{type, ...}]
    this.tailLen = opts.tailLen ?? 30; // material between the tail end and the slip knot
    this.log = opts.log || (() => {});
    this.events = [];
  }

  // ---- geometry of one node given its neighbours' reference points
  center(n, F) {
    if (n.type === 'fix' || n.type === 'finger') return n.p;
    if (n.type === 'knot') return n.c;
    return madd(F.tip, F.a, n.x);
  }

  nodeGeom(n, prev, next, F) {
    if (n.type === 'fix' || n.type === 'finger') return [n.p];
    if (n.type === 'knot') return wrapArc(prev, next, n.c, n.axis, n.R, n.via || null, 20);
    const c = madd(F.tip, F.a, n.x);
    if (n.type === 'throat') return wrapArc(prev, next, c, F.a, n.R, F.d, 15);
    if (n.type === 'ring') return ringLoop(prev, c, F.a, n.R);
    return wrapArc(prev, next, c, F.a, n.R, null, 15);
  }

  layout(F) {
    const N = this.nodes;
    let refs = N.map((n) => this.center(n, F));
    let G = [];
    for (let it = 0; it < 3; it++) {
      G = N.map((n, k) => {
        const prev = k > 0 ? (G[k - 1] ? G[k - 1][G[k - 1].length - 1] : refs[k - 1]) : refs[k];
        const next = k < N.length - 1 ? (G[k + 1] ? G[k + 1][0] : refs[k + 1]) : refs[k];
        return this.nodeGeom(n, prev, next, F);
      });
    }
    return G;
  }

  sweepOf(arc, R) { return polyLength(arc) / Math.max(R, 1e-6); }

  // ---- contact events: insertion of slide wraps where a span cuts the hook, sliding, capture, release, slip-off
  update(F, dt) {
    const N = this.nodes;
    // rings keep their world place along the axis; slide wraps slide to shorten the yarn
    let G = this.layout(F);
    for (let k = 0; k < N.length; k++) {
      const n = N[k];
      if (n.type === 'ring') {
        n.x = dot(sub(n.w, F.tip), F.a);
        if (n.xMax != null) n.x = Math.min(n.x, n.xMax);
        n.w = madd(F.tip, F.a, n.x);
        n.R += (R_LOOP() - n.R) * clamp(dt / 0.25);
      } else if (n.type === 'slide') {
        const arc = G[k], pIn = k > 0 ? G[k - 1][G[k - 1].length - 1] : arc[0], pOut = k < N.length - 1 ? G[k + 1][0] : arc[arc.length - 1];
        // friction holds the wrap in its world place while the shaft slides under it; tension slides it along the
        // shaft toward the shorter path
        if (n.w) n.x = dot(sub(n.w, F.tip), F.a);
        const c = madd(F.tip, F.a, n.x);
        const pull = dot(add(norm(sub(pIn, c)), norm(sub(pOut, c))), F.a);
        n.x += clamp(pull * 40 * dt, -1.0, 1.0);
        n.x = Math.min(n.x, -0.05);
        n.w = madd(F.tip, F.a, n.x);
        n.R = baseRadius(Math.min(n.x, -0.01)) + RY * 0.95;
      }
    }
    G = this.layout(F);
    // capture into the throat, removal of slide wraps that no longer bend, slip-off over the tip
    for (let k = N.length - 1; k >= 0; k--) {
      const n = N[k];
      if (n.type === 'slide') {
        const arc = G[k];
        const mid = arc[Math.floor(arc.length / 2)];
        const side = norm(perp(sub(mid, madd(F.tip, F.a, n.x)), F.a));
        if (!N.some((m) => m.type === 'throat') && n.x > HOOK.xThroat - 1.6 && n.x < HOOK.xThroat + 0.9 && (this.catchAny ? this.sweepOf(arc, n.R) > 1.0 : dot(side, F.d) > Math.cos((65 * Math.PI) / 180))) {
          n.type = 'throat'; n.x = HOOK.xThroat; n.R = R_THR(); n.tCaught = this.t;
          this.events.push({ t: this.t, ev: 'catch', k }); this.log(`catch: yarn slid into the throat (node ${k})`);
        } else if (n.x > -0.15 || this.sweepOf(arc, n.R) < 0.12) {
          N.splice(k, 1); this.events.push({ t: this.t, ev: 'unwrap', k });
        }
      } else if (n.type === 'ring' && n.x > -0.25) {
        // drawn back past the head: the loop is off the hook and closes round the legs of the next loop
        const next = N[k + 1];
        const c = madd(F.tip, F.a, -0.2);
        n.type = 'knot'; n.c = c; n.axis = norm(sub(madd(F.tip, F.a, (next && next.x) || HOOK.xThroat), madd(c, [0, -1, 0], 3)));
        n.R = R_LOOP(); n.via = null; n.tKnot = this.t;
        this.events.push({ t: this.t, ev: 'slipoff', k }); this.log(`slip-off: loop ${k} is off the hook`);
      }
    }
    G = this.layout(F);
    // a straight span that would cut into the hook gets a slide wrap at the contact
    let ins = 0;
    for (let k = 0; k < N.length - 1 && ins < 2; k++) {
      const A = G[k][G[k].length - 1], B = G[k + 1][0];
      let hit = this.hookHit(A, B, F);
      // the head is already wrapped by the yarn in the throat / the loop that just slipped off: no extra contacts there
      const headBusy = N.some((m) => m.type === 'throat') || [N[k].type, N[k + 1].type].includes('knot');
      if (hit && hit.x > -3.4 && headBusy) hit = null;
      if (hit) {
        N.splice(k + 1, 0, { type: 'slide', x: hit.x, R: baseRadius(hit.x) + RY * 0.95 });
        this.events.push({ t: this.t, ev: 'wrap', k: k + 1 }); ins++;
        if (this.debug) this.log(`wrap at x=${hit.x.toFixed(2)} between ${N[k].type} and ${N[k + 2].type}`);
        G = this.layout(F);
        k++;
      }
    }
  }

  hookHit(A, B, F) {
    // closest approach between the span AB and the hook axis (x in [-34, -0.2])
    const P0 = madd(F.tip, F.a, -34), P1 = madd(F.tip, F.a, -0.2);
    const [sc, tc] = segSeg(A, B, P0, P1);
    if (sc <= 0.03 || sc >= 0.97) return null; // touching at an end = the span leaves from the hook itself
    const pa = lerp(A, B, sc), pb = lerp(P0, P1, tc);
    const x = -34 + tc * 33.8;
    const r = baseRadius(x) + RY * 0.95;
    if (dist(pa, pb) < r - 0.08) return { x };
    return null;
  }

  // ---- full strand for rendering: points + material coordinate
  strand(F, tailEnd, sagDir = [0, -1, 0]) {
    const N = this.nodes;
    const G = this.layout(F);
    const pts = [], S = [];
    // tail: hangs from the slip knot to the held end with its own length (sags)
    const K0 = G[0][0];
    const tail = sagCurve(tailEnd, K0, this.tailLen, sagDir, 24);
    let s = -this.tailLen;
    for (let i = 0; i < tail.length; i++) { if (i) s += dist(tail[i - 1], tail[i]); pts.push(tail[i]); S.push(s); }
    s = 0;
    for (let k = 0; k < N.length; k++) {
      const arc = G[k];
      for (let i = 0; i < arc.length; i++) {
        const p = arc[i];
        if (pts.length) s += dist(pts[pts.length - 1], p);
        pts.push(p); S.push(s);
      }
    }
    return { pts, s: S, nodes: N.map((n) => n.type), G };
  }
}

/** A loop round the hook: both legs meet at a crossing point under the shaft (toward the fabric), the yarn goes
 *  up one side, over the top and down the other side; the next span leaves from the crossing point. */
export function ringLoop(prev, c, a, R, gap = 0.75, spread = 0.38) {
  let dn = perp(sub(prev, c), a);
  dn = len(dn) > 1e-6 ? norm(dn) : [0, -1, 0];
  const e2 = cross(a, dn);
  const P = madd(c, dn, R + gap);
  const out = [madd(P, a, -spread)];
  const n = 24;
  for (let i = 0; i <= n; i++) {
    const t = i / n, f = t * Math.PI * 2;
    out.push(madd(madd(madd(c, dn, R * Math.cos(f)), e2, R * Math.sin(f)), a, spread * (2 * t - 1)));
  }
  out.push(madd(P, a, spread));
  return out;
}

/** Curve from A to B of arc length L (>= |AB|), sagging toward dir: a parabola whose depth matches the length. */
export function sagCurve(A, B, L, dir, n = 20) {
  const d = dist(A, B);
  const ex = Math.max(0, L - d);
  const h = Math.sqrt((3 * d * ex) / 8);
  const m = norm(perp(dir, norm(sub(B, A))));
  const out = [];
  for (let i = 0; i <= n; i++) { const t = i / n; out.push(madd(lerp(A, B, t), m, 4 * h * t * (1 - t))); }
  return out;
}

/** Closest points between segments p1q1 and p2q2: returns [s, t] parameters. */
export function segSeg(p1, q1, p2, q2) {
  const d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
  const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r), c = dot(d1, r), b = dot(d1, d2);
  const den = a * e - b * b;
  let s = den > 1e-9 ? clamp((b * f - c * e) / den) : 0;
  let t = (b * s + f) / e;
  if (t < 0) { t = 0; s = clamp(-c / a); } else if (t > 1) { t = 1; s = clamp((b - c) / a); }
  return [s, t];
}
