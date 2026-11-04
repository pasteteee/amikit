// Crochet simulation: hook choreography + inextensible taut-yarn model + PBD free tail.
// Produces, for any time t (evaluated sequentially), the yarn centreline with material coordinate,
// the hook pose, the fabric transform, camera and caption state.
import { add, sub, mul, madd, dot, cross, len, dist, norm, lerp, mix, clamp, smooth, smoothstep, slerpV, perp, quatFromBasis, quatSlerp, quatToBasis } from './vec.mjs';
import { catmull, resampleN, wrapArc, resampleAttr, laplace, closestOnSeg, polyLength, limitCurvature } from './curve.mjs';
import { HOOK, hookFrame, hookPoint, pushOutOfHook } from './hook.mjs';
import { Fabric, RY, RY_FAB, SW, RING_OPEN, RING_CLOSED, canonicalCtrl, ringCtrl, tailRoot } from './fabric.mjs';

const R_SHAFT_LOOP = HOOK.rShaft + RY * 0.92;
const R_THROAT = HOOK.rNeck + RY * 0.95;
const X_W = -8.2; // working loop position on the shaft
const X_L1 = -4.6; // second loop position
const G_HIGH = [-13, 19, -24]; // left index finger (yarn guide), world, relative to work point

// Base phase durations (seconds at speed 1)
const PH_SC = [
  ['insert', 0.8], ['yo1', 0.7], ['pullup', 0.75], ['raise', 0.45], ['yo2', 0.65], ['through1', 0.35], ['through2', 0.6], ['ret', 0.6],
];
const PH_DEC = [
  ['insertFL1', 0.8], ['insertFL2', 0.8], ['yo1', 0.7], ['pullup', 0.75], ['raise', 0.45], ['yo2', 0.65], ['through1', 0.35], ['through2', 0.6], ['ret', 0.6],
];

export function phasesFor(kind) { return kind === 'dec' ? PH_DEC : PH_SC; }

/** Interpolated pose helper */
function poseLerp(p0, p1, u, arc = null) {
  const e = smooth(u);
  let tip = lerp(p0.tip, p1.tip, e);
  if (arc) tip = madd(tip, arc, Math.sin(Math.PI * e));
  const a = slerpV(p0.a, p1.a, e);
  const d = slerpV(norm(perp(p0.d, p0.a)), norm(perp(p1.d, p1.a)), e);
  return { tip, a, d };
}
/** Pose whose shaft passes through pivot point H: tip = H + a*l. */
function pivotPose(H, a0, a1, l0, l1, d0, d1, u) {
  const e = smooth(u);
  const a = slerpV(a0, a1, e);
  const l = mix(l0, l1, e);
  return { tip: madd(H, a, l), a, d: slerpV(d0, d1, e) };
}

/** Per-point tube radius: yarn is slightly compressed inside the fabric and thins at very sharp bends
 *  (a real yarn flattens there) so the tube surface never folds through itself. */
export function tubeRadii(P, fab) {
  const n = P.length;
  const r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let R = 1e9;
    if (i > 0 && i < n - 1) {
      const a = P[i - 1], b = P[i], c = P[i + 1];
      const d1 = sub(b, a), d2 = sub(c, b);
      const l1 = len(d1), l2 = len(d2);
      const cs = clamp(dot(d1, d2) / (l1 * l2 + 1e-12), -1, 1);
      const ang = Math.acos(cs);
      R = ((l1 + l2) / 2) / Math.max(ang, 1e-6);
    }
    const base = mix(RY, RY_FAB, fab[i] ?? 0);
    r[i] = Math.min(base, Math.max(0.3, R * 0.92));
  }
  // spread the thinning a little (min filter then blur)
  const m = Float32Array.from(r);
  for (let i = 0; i < n; i++) for (let k = -2; k <= 2; k++) { const j = i + k; if (j >= 0 && j < n) m[i] = Math.min(m[i], r[j] + Math.abs(k) * 0.12); }
  const o = Float32Array.from(m);
  for (let i = 1; i < n - 1; i++) o[i] = (m[i - 1] + 2 * m[i] + m[i + 1]) / 4;
  return Array.from(o);
}

export class Sim {
  constructor(scn) {
    this.scn = scn;
    this.fab = new Fabric(scn.rounds, scn.theta0 ?? Math.PI / 2);
    this.fps = scn.fps || 30;
    this.buildTimeline();
    this.reset();
  }

  buildTimeline() {
    const acts = [];
    let t = 0;
    for (const a of this.scn.actions) {
      const act = { ...a, t0: t };
      if (a.type === 'stitch') {
        const s = this.fab.stitches[a.i];
        const ph = phasesFor(s.kind);
        act.phases = [];
        let tt = t;
        for (const [name, d] of ph) {
          const dd = d / (a.speed || 1) * (a.slow?.[name] || 1);
          act.phases.push({ name, t0: tt, dur: dd });
          tt += dd;
        }
        act.dur = tt - t;
      }
      t += act.dur;
      act.t1 = t;
      acts.push(act);
    }
    this.actions = acts;
    this.duration = t;
  }

  reset() {
    const fab = this.fab;
    fab.ring = RING_OPEN;
    this.t = -1e9;
    this.ai = -1; this.pi = -1;
    this.G = G_HIGH.slice();
    this.loops = {}; // active loops by role: W, L1, L2
    this.cont = {}; // geometric contact state of the current stitch (catches, holes)
    this.doneUpTo = 0; // number of strand pieces (ring=0, P0=1, T0=2, ...) that are fabric
    this.morphs = {}; // piece index -> {from, t0, dur}
    this.cur = 0; // current stitch index (fabric alignment)
    this.align = this.alignFor(0);
    this.hook = { tip: [40, 18, 30], a: norm([-0.8, -0.25, -0.5]), d: [0, -1, 0] };
    this.tail = null;
    this.tailLen = this.scn.tailLength ?? 38;
    this.ringLen0 = null;
    this.camera = null;
  }

  // ---------- fabric alignment (fabric local -> world)
  alignFor(i) {
    const f = this.fab.frame(this.fab.stitches[i]);
    return { q: quatFromBasis(mul(f.u, -1), f.v, f.w), T: f.T };
  }
  get xf() {
    const B = quatToBasis(this.align.q); // columns u,v,w
    const [u, v, w] = B; // u here is the world +x direction in fabric space (= -stitch u)
    const T = this.align.T;
    return {
      u, v, w, T,
      to: (p) => { const d = sub(p, T); return [dot(u, d), dot(v, d), dot(w, d)]; },
      from: (q) => add(add(add(T, mul(u, q[0])), mul(v, q[1])), mul(w, q[2])),
    };
  }

  // ---------- strand piece helpers
  pieceIndex(role, i) { return role === 'P' ? 1 + 2 * i : 2 + 2 * i; }

  canonicalPieces(upTo) {
    // global Catmull-Rom over ring + pieces [1..upTo-1]
    const fab = this.fab;
    const ctrl = [], owner = [];
    const rc = ringCtrl(fab);
    rc.forEach((p) => { ctrl.push(p); owner.push(0); });
    for (let k = 1; k < upTo; k++) {
      const si = Math.floor((k - 1) / 2), role = (k - 1) % 2 === 0 ? 'P' : 'T';
      const cc = canonicalCtrl(fab, fab.stitches[si], role);
      cc.forEach((p) => { ctrl.push(p); owner.push(k); });
    }
    const per = 5;
    const dense = catmull(ctrl, per);
    const pieces = Array.from({ length: upTo }, () => []);
    pieces[0].push(dense[0]);
    for (let c = 1; c < ctrl.length; c++) {
      const k = owner[c];
      for (let s = 0; s < per; s++) pieces[k].push(dense[(c - 1) * per + 1 + s]);
    }
    return pieces;
  }

  // ---------- timeline stepping
  step(t) {
    const dt = this.t < -1e8 ? 1 / this.fps : Math.max(1e-4, t - this.t);
    this.t = t;
    // advance actions / phases, firing enter events in order
    while (true) {
      const act = this.actions[this.ai];
      if (act && t < act.t1) {
        if (act.type === 'stitch') {
          while (this.pi + 1 < act.phases.length && t >= act.phases[this.pi + 1].t0) { this.pi++; this.enterPhase(act, act.phases[this.pi]); }
        }
        break;
      }
      if (this.ai + 1 >= this.actions.length) break;
      if (act && act.type === 'stitch') this.finishStitch(act);
      this.ai++; this.pi = -1;
      this.enterAction(this.actions[this.ai]);
    }
    const act = this.actions[this.ai];
    if (act.type === 'stitch') {
      const ph = act.phases[Math.max(0, this.pi)];
      const u = clamp((t - ph.t0) / ph.dur);
      this.phName = ph.name; this.phU = u;
      this.updatePhase(act, ph, u);
    } else { this.phName = act.type; this.phU = clamp((t - act.t0) / act.dur); this.updateAction(act, this.phU); }
    this.updateTail(dt);
    return this.snapshot(act);
  }

  enterAction(act) {
    this.actStart = { hook: { ...this.hook }, G: this.G.slice(), ring: this.fab.ring };
    if (act.type === 'stitch') {
      this.st = this.fab.stitches[act.i];
      this.prevAlign = this.align;
    }
  }

  // Key world positions for current stitch (valid once aligned)
  keyPts(s) {
    const fab = this.fab, X = this.xf;
    const P = (a, b, c) => X.to(fab.pt(s, a, b, c));
    return {
      H: P(0, -0.3, 0), E: P(0, 0.05, 0),
      E1: P(-0.45, 0.85, -0.3), Bk1: P(-0.3, 0.35, -1.1), Hb1: P(-0.13, -0.3, -0.8), Hf1: P(-0.13, -0.3, 0.8),
      Hf2: P(0.13, -0.3, 0.8), Hb2: P(0.13, -0.3, -0.8), U1: P(0.18, 0.55, -1.15), U2: P(0.22, 0.98, -0.9),
    };
  }

  enterPhase(act, ph) {
    const s = this.st;
    const K = this.keyPts(s);
    this.K = K;
    const h0 = { tip: this.hook.tip.slice(), a: this.hook.a.slice(), d: this.hook.d.slice() };
    this.ph0 = h0;
    this.G0 = this.G.slice();
    const aIns = norm([-0.36, -0.32, -0.88]);
    const dDown = [0, -1, 0.35];
    switch (ph.name) {
      case 'insert': {
        this.pivotH = K.H;
        this.target = { lA: -7.5, lB: 4.2, a: aIns, d: dDown };
        // approach first to READY (in front of the hole), then push through along the axis
        this.ready = { tip: madd(K.H, aIns, -7.5), a: aIns, d: dDown };
        break;
      }
      case 'yo1': {
        // The hook goes to the yarn (never the other way round): aim the throat a little past the
        // working yarn line behind the work, keeping the shaft pivoting in the hole.
        // The working loop rides on the shaft, so the yarn line moves with the hook. Search the pivot
        // (hook stays in the hole) that brings the throat just inside the yarn line behind the work.
        const wW = this.fabricNormal(K);
        const a0 = norm(this.hook.a);
        const e1 = norm(perp(sub(this.G, K.H), a0)), e2 = cross(a0, e1);
        let best = null;
        for (let th = 0; th <= 75; th += 2.5) for (let ps = -90; ps <= 90; ps += 7.5) for (const l of [4.2, 5.0, 5.8]) {
          const T = th * Math.PI / 180, P = ps * Math.PI / 180;
          const a = norm(add(mul(a0, Math.cos(T)), mul(add(mul(e1, Math.cos(P)), mul(e2, Math.sin(P))), Math.sin(T))));
          const C = madd(K.H, a, l + HOOK.xThroat);
          if (dot(sub(C, K.H), wW) > -0.8) continue;
          const Wc = madd(K.H, a, l + X_W);
          const Wx = madd(Wc, norm(perp(sub(this.G, Wc), a)), R_SHAFT_LOOP);
          const [q] = closestOnSeg(C, Wx, this.G);
          const cost = Math.abs(dist(q, C) - 1.9) * 3 + th * 0.01 + Math.abs(l - 4.6) * 0.1;
          if (!best || cost < best.cost) best = { cost, a, l, C, q };
        }
        const a = best.a, d = norm(perp(sub(best.q, best.C), a));
        const C = best.C;
        this.catch1 = { a, l: dist(C, K.H) - HOOK.xThroat, d };
        this.catchCost = best.cost;
        break;
      }
      case 'pullup': break;
      case 'raise': {
        this.target = { tip: [-2.2, 1.6, 3.6], a: norm([-0.86, -0.18, -0.48]), d: [0, -1, 0.2] };
        break;
      }
      case 'yo2': {
        // reach over the top edge to the yarn rising from the back of the hole to the finger
        const [A, B] = this.workingSegment(K);
        const Y = this.yarnPointAbove(A, B, K.E, 2.0);
        const a2 = norm([-0.55, -0.62, -0.56]);
        const C0 = hookPoint(hookFrame(this.hook), HOOK.xThroat);
        const d2 = norm(perp(sub(Y, C0), a2));
        const throatC = madd(Y, d2, -1.9);
        this.target = { tip: madd(throatC, a2, -HOOK.xThroat), a: a2, d: d2 };
        break;
      }
      case 'through1': {
        this.target = { tip: [-1.6, 1.9, 3.2], a: norm([-0.86, -0.18, -0.48]), d: [0, -0.4, 1] };
        break;
      }
      case 'through2': {
        const a = this.hook.a;
        this.target = { tip: madd(madd(this.hook.tip, a, -10.5), [0, 0.6, 0.6], 1), a, d: this.hook.d };
        // loops stay where the fabric holds them; the hook slides out of them
        for (const r of ['W', 'L1']) {
          const L = this.loops[r];
          if (L) { L.fixed = hookPoint(hookFrame(this.hook), L.x); L.mode = 'fixed'; }
        }
        break;
      }
      case 'ret': {
        // fabric turns to bring the next stitch to the hook; hook goes to the next READY pose
        const next = this.actions[this.ai + 1];
        this.retNext = next && next.type === 'stitch' ? next.i : null;
        if (this.retNext != null) {
          this.alignA = this.align;
          this.alignB = this.alignFor(this.retNext);
          // compute next READY in the destination alignment
          const save = this.align; this.align = this.alignB;
          const Kn = this.keyPts(this.fab.stitches[this.retNext]);
          this.align = save;
          const nextKind = this.fab.stitches[this.retNext].kind;
          const aN = norm([-0.36, -0.32, -0.88]);
          this.target = nextKind === 'dec'
            ? { tip: add(Kn.H, [0.5, -2.8, 6.0]), a: norm([-0.3, 0.35, -0.89]), d: dDown }
            : { tip: madd(Kn.H, aN, -7.5), a: aN, d: dDown };
        } else {
          this.target = { tip: [-3, 6, 9], a: norm([-0.8, -0.2, -0.55]), d: [0, -1, 0.3] };
        }
        break;
      }
      case 'insertFL1': case 'insertFL2': {
        const p = this.fab.stitches[s.parents[ph.name === 'insertFL1' ? 0 : 1]];
        const X = this.xf;
        const FL = X.to(this.fab.pt(p, 0.55, 1.06, 0.85));
        const aF = norm([-0.3, 0.35, -0.89]);
        this.target = { tip: madd(FL, aF, ph.name === 'insertFL1' ? 2.4 : 2.2), a: ph.name === 'insertFL1' ? aF : norm([-0.55, 0.3, -0.78]), d: dDown };
        this.flPoint = FL;
        break;
      }
    }
  }

  /** Point on the line from A to B (taut yarn) that sits `h` mm above edge point E (in world y). */
  yarnPointAbove(A, B, E, h) {
    const ty = E[1] + h;
    const t = clamp((ty - A[1]) / ((B[1] - A[1]) || 1e-6), 0.02, 0.9);
    return lerp(A, B, t);
  }

  updatePhase(act, ph, u) {
    const s = this.st;
    const K = this.K;
    const hasW = !!this.loops.W;
    switch (ph.name) {
      case 'insert': {
        // u<0.45: move to READY; then push along the axis through the hole
        if (u < 0.45) this.hook = poseLerp(this.ph0, this.ready, u / 0.45, [0.6, 1.2, 1.5]);
        else this.hook = pivotPose(K.H, this.ready.a, this.ready.a, -7.5, 4.2, this.ready.d, this.ready.d, (u - 0.45) / 0.55);
        if (hasW) this.loops.W.x = X_W;
        break;
      }
      case 'yo1': {
        const c = this.catch1;
        // pivot (hook in the hole) and turn the throat up to the yarn; the finger holds still
        const e = clamp(u / 0.8);
        this.hook = pivotPose(K.H, norm([-0.36, -0.32, -0.88]), c.a, 4.2, c.l, [0, -1, 0.35], c.d, e);
        this.tryCatch('L1', this.pieceIndex('P', s.i), u >= 0.999);
        break;
      }
      case 'pullup': {
        const c = this.catch1;
        this.tryCatch('L1', this.pieceIndex('P', s.i), true);
        this.hook = pivotPose(K.H, c.a, norm([-0.42, -0.25, -0.87]), c.l, -2.9, c.d, [0, -1, 0.4], u);
        // the throat (with the yarn) comes back through the hole: from now on both legs pass through it
        if (!this.cont.hole) {
          const thr = hookPoint(hookFrame(this.hook), HOOK.xThroat);
          const ahead = s.kind === 'dec' ? u > 0.12 : dot(sub(thr, K.H), this.fabricNormal(K)) > 0;
          if (ahead || u > 0.6) this.cont.hole = { t0: this.t, dur: 0.22 / Math.max(0.6, act.speed || 1) };
        }
        break;
      }
      case 'raise': {
        this.hook = poseLerp(this.ph0, this.target, u, [0, 0.8, 0.4]);
        const L1 = this.loops.L1;
        L1.x = mix(HOOK.xThroat, X_L1, smooth(u)); L1.mode = u < 0.5 ? 'throat' : 'shaft';
        break;
      }
      case 'yo2': {
        this.hook = poseLerp(this.ph0, this.target, clamp(u / 0.85), [0, 2.0, 0.5]);
        this.tryCatch('L2', this.pieceIndex('T', s.i), u >= 0.999);
        break;
      }
      case 'through1': {
        this.tryCatch('L2', this.pieceIndex('T', s.i), true);
        this.hook = poseLerp(this.ph0, this.target, u, [0, 1.2, 0.5]);
        break;
      }
      case 'through2': {
        this.hook = poseLerp(this.ph0, this.target, u);
        const F = hookFrame(this.hook);
        for (const r of ['W', 'L1']) {
          const L = this.loops[r];
          if (!L || L.mode !== 'fixed') continue;
          L.x = dot(sub(L.fixed, F.tip), F.a);
          if (L.x > -0.5 && !L.released) this.releaseLoop(r);
        }
        break;
      }
      case 'ret': {
        // make sure released loops are gone
        for (const r of ['W', 'L1']) if (this.loops[r] && !this.loops[r].released) this.releaseLoop(r);
        this.hook = poseLerp(this.ph0, this.target, u, [1.0, 2.5, 1.0]);
        if (this.retNext != null) {
          const e = smooth(u);
          this.align = { q: quatSlerp(this.alignA.q, this.alignB.q, e), T: lerp(this.alignA.T, this.alignB.T, e) };
        }
        const L2 = this.loops.L2;
        if (L2) { L2.x = mix(HOOK.xThroat, X_W, smooth(u)); L2.mode = u < 0.35 ? 'throat' : 'shaft'; }
        break;
      }
      case 'insertFL1': case 'insertFL2': {
        this.hook = poseLerp(this.ph0, this.target, u, ph.name === 'insertFL1' ? [0.3, -1.5, 1.0] : [0, -1.2, 1.2]);
        if (hasW) this.loops.W.x = X_W;
        break;
      }
    }
  }

  /** Fabric outward normal (w) of the current stitch, in world space. */
  fabricNormal(K) {
    const f = this.fab.frame(this.st);
    const X = this.xf;
    return norm(sub(X.to(add(f.T, f.w)), X.to(f.T)));
  }
  fabricUp() {
    const f = this.fab.frame(this.st);
    const X = this.xf;
    return norm(sub(X.to(add(f.T, f.v)), X.to(f.T)));
  }
  /** The straight run of working yarn the hook has to catch: from the last contact to the finger. */
  workingSegment(K) {
    const F = hookFrame(this.hook);
    const nodes = this.buildActive();
    const last = nodes[nodes.length - 1];
    let A;
    if (!last) A = K.E;
    else if (last.kind === 'pt') A = last.p;
    else A = madd(last.c, norm(perp(sub(this.G, last.c), last.axis)), last.R);
    void F;
    return [A, this.G];
  }
  /** Throat catches the yarn only when it actually reaches it (distance to the yarn line <= contact radius). */
  tryCatch(role, piece, force) {
    if (this.loops[role]) return;
    const F = hookFrame(this.hook);
    const C = hookPoint(F, HOOK.xThroat);
    const [A, B] = this.workingSegment(this.K);
    const [q] = closestOnSeg(C, A, B);
    // the yarn rests against the hook (shaft radius + yarn) and the open throat faces it: it drops into the groove
    const facing = dot(F.d, norm(perp(sub(q, C), F.a))) > 0.55;
    const near = dist(q, C) <= HOOK.rShaft + RY + 0.25;
    if ((near && facing) || force) {
      this.loops[role] = { role, piece, x: HOOK.xThroat, mode: 'throat' };
      this.cont[role] = { t0: this.t, forced: !(near && facing), d: dist(q, C) };
    }
  }

  releaseLoop(role) {
    const L = this.loops[role];
    L.released = true;
    // snapshot current geometry of that piece (fabric-local) to morph into its relaxed shape
    const geo = this._lastPieces?.[L.piece];
    const X = this.xf;
    if (geo && geo.length > 1) this.morphs[L.piece] = { from: geo.map((p) => X.from(p)), t0: this.t, dur: 0.32 / Math.max(1, (this.actions[this.ai].speed || 1) * 0.6) };
    this.doneUpTo = Math.max(this.doneUpTo, L.piece + 1);
    delete this.loops[role];
  }

  finishStitch(act) {
    // L2 becomes the working loop W of the next stitch
    for (const r of ['W', 'L1']) if (this.loops[r]) this.releaseLoop(r);
    const L2 = this.loops.L2;
    if (L2) { L2.role = 'W'; this.loops.W = L2; delete this.loops.L2; }
    this.cont = {};
    this.cur = act.i;
    if (this.retNext != null) this.align = this.alignB;
  }

  updateAction(act, u) {
    switch (act.type) {
      case 'hold':
        if (act.hook) this.hook = poseLerp(this.actStart.hook, act.hook, u, act.arc || null);
        break;
      case 'closeRing': {
        const e = smooth(clamp(u / 0.85));
        this.fab.ring = mix(this.actStart.ring, RING_CLOSED, e);
        // keep the work stitch aligned while the ring shrinks
        this.align = this.alignFor(this.cur);
        this.tailPull = e;
        if (act.hook) this.hook = poseLerp(this.actStart.hook, act.hook, u);
        break;
      }
      case 'outro':
        if (act.hook) this.hook = poseLerp(this.actStart.hook, act.hook, u, [2, 6, 2]);
        break;
    }
  }

  // ---------- free tail (PBD rope, world space)
  updateTail(dt) {
    const X = this.xf;
    const root = X.to(tailRoot(this.fab));
    const ringLen = polyLength(catmull(ringCtrl(this.fab), 4));
    if (this.ringLen0 == null) this.ringLen0 = ringLen;
    const L = this.tailLen + (this.ringLen0 - ringLen);
    const N = 34;
    const rest = L / (N - 1);
    if (!this.tail) {
      this.tail = { p: [], q: [] };
      for (let k = 0; k < N; k++) {
        const p = add(root, [k * rest * 0.35, -k * rest * 0.93, -k * 0.05]);
        this.tail.p.push(p); this.tail.q.push(p.slice());
      }
    }
    const T = this.tail;
    const sub_ = 4;
    const h = Math.min(dt, 1 / 20) / sub_;
    const g = [0, -2500, 0];
    for (let it = 0; it < sub_; it++) {
      for (let k = 1; k < N; k++) {
        const p = T.p[k], q = T.q[k];
        const v = mul(sub(p, q), 0.985);
        T.q[k] = p.slice();
        T.p[k] = add(add(p, v), mul(g, h * h));
      }
      T.p[0] = root.slice();
      if (this.tailPull) {
        // the other hand pulls the tail end downward to close the ring
        const target = add(root, [4, -L * 0.98, 2]);
        T.p[N - 1] = lerp(T.p[N - 1], target, 0.25 * this.tailPull);
      }
      for (let c = 0; c < 14; c++) {
        for (let k = 0; k < N - 1; k++) {
          const a = T.p[k], b = T.p[k + 1];
          const d = sub(b, a), l = len(d) || 1e-9;
          const corr = (l - rest) / l;
          if (k === 0) T.p[k + 1] = madd(b, d, -corr);
          else { T.p[k] = madd(a, d, corr * 0.5); T.p[k + 1] = madd(b, d, -corr * 0.5); }
        }
        // keep the tail behind the fabric disc
        for (let k = 2; k < N; k++) {
          const loc = X.from(T.p[k]);
          const r = Math.hypot(loc[0], loc[1]);
          const [rho] = this.fab.profileAt(Math.min(this.fab.profile.K, (this.cur ? this.fab.stitches[this.cur].r : 0) + 0.3));
          if (r < Math.max(rho, this.fab.ring + 2) + 1.5 && loc[2] > -1.6) {
            loc[2] = -1.6;
            T.p[k] = X.to(loc);
          }
        }
      }
    }
  }

  // ---------- assemble the yarn
  buildActive() {
    // Returns list of nodes: {kind:'pt', p, piece, pres} | {kind:'wrap', ...}
    const s = this.st;
    const K = this.K;
    const F = hookFrame(this.hook);
    const nodes = [];
    const X = this.xf;
    const pres = (p) => p;
    if (!s) return nodes;
    const wrapNode = (L) => {
      if (!L) return null;
      if (L.mode === 'fixed') {
        // loop held by the fabric while the hook slides out: centre stays on the (moving) axis
        const c = hookPoint(F, Math.min(L.x, 0.2));
        const r = L.x > HOOK.xThroat - HOOK.throatHalf && L.x < HOOK.xThroat + HOOK.throatHalf ? R_THROAT + 0.3 : R_SHAFT_LOOP;
        return { kind: 'wrap', c, axis: F.a, R: L.x > -1.2 ? RY * 1.6 : r, piece: L.piece };
      }
      if (L.mode === 'throat') return { kind: 'wrap', c: hookPoint(F, L.x), axis: F.a, R: R_THROAT, via: F.d, piece: L.piece };
      const tR = smoothstep(HOOK.xThroat, X_L1, L.x);
      return { kind: 'wrap', c: hookPoint(F, L.x), axis: F.a, R: mix(R_THROAT, R_SHAFT_LOOP, tR), piece: L.piece };
    };
    // W: in-leg anchored on the fabric (start of its relaxed top path)
    const W = this.loops.W;
    if (W) {
      const prev = this.fab.stitches[s.i - (s.kind === 'inc' && s.sub === 1 ? 1 : 1)];
      const ps = this.fab.stitches[W.piece % 2 === 0 ? (W.piece - 2) / 2 : 0];
      void prev;
      nodes.push({ kind: 'pt', p: X.to(this.fab.pt(ps, 0.18, 0.55, -1.15)), piece: W.piece, pres: 1, anchor: true });
      nodes.push({ kind: 'pt', p: X.to(this.fab.pt(ps, 0.22, 0.98, -0.9)), piece: W.piece, pres: 1, anchor: true });
      nodes.push(wrapNode(W));
    }
    const L1 = this.loops.L1;
    if (L1) {
      const P1 = L1.piece;
      // both legs pass through the hole once the throat has carried the yarn through it; the hole
      // contacts start on the straight yarn line and settle into the hole (no popping)
      const H = this.cont.hole;
      const hp = L1.released ? 1 : H ? smooth((this.t - H.t0) / H.dur) : -1;
      const hole = (p) => (hp >= 0 ? { kind: 'pt', p, piece: P1, pres: hp, hole: true } : null);
      if (s.kind === 'dec') {
        const Pl = this.fab.stitches[s.parents[0]], Pr = this.fab.stitches[s.parents[1]];
        const FL1 = X.to(this.fab.pt(Pl, 0.55, 1.0, 1.5)), FL2 = X.to(this.fab.pt(Pr, 0.55, 1.0, 1.5));
        nodes.push(hole(FL2));
        nodes.push(wrapNode(L1));
        nodes.push(hole(FL1));
      } else {
        nodes.push(hole(K.Hb1), hole(K.Hf1));
        nodes.push(wrapNode(L1));
        nodes.push(hole(K.Hf2), hole(K.Hb2));
      }
    }
    const L2 = this.loops.L2;
    if (L2) nodes.push(wrapNode(L2));
    void pres;
    return nodes.filter(Boolean);
  }

  /** Taut yarn crossing the work below its top edge rests on the edge (stateless, so a contact
   *  appears exactly when the straight yarn touches the edge and never pops). */
  addEdges(nodes, G) {
    if (!this.st || !this.K || !nodes.length) return nodes;
    const K = this.K, wW = this.fabricNormal(K), up = this.fabricUp();
    const h0 = RY_FAB * 1.3;
    const cen = (n) => (n.kind === 'pt' ? n.p : n.c);
    const out = [];
    const all = [...nodes, { kind: 'pt', p: G, piece: 9999, end: true }];
    for (let k = 0; k < all.length - 1; k++) {
      const n0 = all[k], n1 = all[k + 1];
      out.push(n0);
      if (n0.anchor || (n0.hole && n1.hole)) continue;
      const P = cen(n0), Q = cen(n1);
      const cP = dot(sub(P, K.H), wW), cQ = dot(sub(Q, K.H), wW);
      if (!((cP > 0.2 && cQ < -0.2) || (cP < -0.2 && cQ > 0.2))) continue;
      const Xc = lerp(P, Q, cP / (cP - cQ));
      const hE = dot(sub(Xc, K.E), up);
      if (hE >= h0) continue;
      out.push({ kind: 'pt', p: madd(Xc, up, h0 - hE), piece: n1.end ? (n0.piece ?? 9999) : n1.piece, pres: 1, edge: true });
    }
    return out;
  }

  expandNodes(start, nodes, end) {
    // resolve wrap arcs with neighbour positions, then apply presence blending
    const centers = nodes.map((n) => (n.kind === 'pt' ? n.p : n.c));
    const pts = nodes.map((n, k) => {
      if (n.kind === 'pt') return [n.p];
      const prev = k > 0 ? centers[k - 1] : start;
      const next = k < nodes.length - 1 ? centers[k + 1] : end;
      return wrapArc(prev, next, n.c, n.axis, n.R, n.via || null);
    });
    // presence: blend partially-present nodes toward the straight line between present neighbours
    const full = nodes.map((n) => (n.pres ?? 1) >= 0.999);
    for (let k = 0; k < nodes.length; k++) {
      const lam = nodes[k].pres ?? 1;
      if (lam >= 0.999) continue;
      let a = k - 1; while (a >= 0 && !full[a]) a--;
      let b = k + 1; while (b < nodes.length && !full[b]) b++;
      const A = a >= 0 ? pts[a][pts[a].length - 1] : start;
      const Bp = b < nodes.length ? pts[b][0] : end;
      pts[k] = pts[k].map((q) => { const [c] = closestOnSeg(q, A, Bp); return lerp(c, q, lam); });
    }
    return pts;
  }

  snapshot(act) {
    const X = this.xf;
    const fab = this.fab;
    const outP = [], outPiece = [];
    // tail (free end -> root)
    for (let k = this.tail.p.length - 1; k >= 1; k--) { outP.push(this.tail.p[k]); outPiece.push(-1); }
    // canonical fabric pieces
    const upTo = Math.max(1, this.doneUpTo);
    const canon = this.canonicalPieces(Math.max(upTo, ...Object.keys(this.morphs).map((k) => +k + 1), 1));
    const pieces = {};
    for (let k = 0; k < upTo; k++) {
      let pts = canon[k];
      const m = this.morphs[k];
      if (m) {
        const e = smooth((this.t - m.t0) / m.dur);
        if (e >= 1) delete this.morphs[k];
        else {
          const n = Math.max(m.from.length, pts.length, 8);
          const A = resampleN(m.from, n), B = resampleN(pts, n);
          pts = A.map((p, i) => lerp(p, B[i], e));
        }
      }
      const w = pts.map((p) => X.to(p));
      pieces[k] = w;
      for (const p of w) { outP.push(p); outPiece.push(k); }
    }
    // active part
    const start = outP[outP.length - 1];
    const G = this.G;
    const nodes = this.addEdges(this.buildActive(), G);
    const ex = this.expandNodes(start, nodes, G);
    ex.forEach((pp, k) => {
      const pc = nodes[k].piece;
      (pieces[pc] ||= []).push(...pp);
      for (const p of pp) { outP.push(p); outPiece.push(pc); }
    });
    // supply to the finger
    const last = outP[outP.length - 1];
    const nS = Math.max(2, Math.ceil(dist(last, G) / 2));
    for (let k = 1; k <= nS; k++) { outP.push(lerp(last, G, k / nS)); outPiece.push(9999); }
    this._lastPieces = Object.fromEntries(Object.entries(pieces).map(([k, v]) => [k, v]));
    // material coordinate (0 at the start of the first post)
    let idx0 = outPiece.indexOf(1);
    if (idx0 < 0) idx0 = outPiece.findIndex((x) => x > 0);
    if (idx0 < 0) idx0 = outP.length - 1;
    const sArr = new Array(outP.length).fill(0);
    for (let k = idx0 + 1; k < outP.length; k++) sArr[k] = sArr[k - 1] + dist(outP[k - 1], outP[k]);
    for (let k = idx0 - 1; k >= 0; k--) sArr[k] = sArr[k + 1] - dist(outP[k + 1], outP[k]);
    // uniform resample + smoothing + hook contact
    const fabFlag = outPiece.map((pc) => (pc >= 0 && pc < Math.max(1, this.doneUpTo) && !this.morphs[pc] ? 1 : 0));
    let [P, S] = resampleAttr(outP, sArr, 0.32);
    const [, FF] = resampleAttr(outP, fabFlag, 0.32);
    P = laplace(P, 2, 0.35);
    P = limitCurvature(P, 0.32, RY * 0.95, 12);
    const F = hookFrame(this.hook);
    for (let it = 0; it < 2; it++) { pushOutOfHook(F, P, RY); P = limitCurvature(laplace(P, 1, 0.2), 0.32, RY * 0.95, 4); }
    pushOutOfHook(F, P, RY);
    const R = tubeRadii(P, FF);
    const onHook = Object.values(this.loops).filter((L) => !L.released && !(L.mode === 'fixed' && L.x > -1.2)).length;
    return { t: this.t, pts: P, s: S, r: R, hook: { tip: this.hook.tip, a: F.a, d: F.d }, act, ring: fab.ring, phase: this.phName, phaseU: this.phU, onHook };
  }
}
