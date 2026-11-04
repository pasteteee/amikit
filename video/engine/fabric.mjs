// Fabric model: pattern -> stitches on a spiral surface of revolution, stitch frames and the
// canonical (relaxed) yarn path of every completed stitch.
import { add, sub, mul, madd, norm, cross, lerp, mix, smooth, clamp } from './vec.mjs';

export const SW = 4.4; // stitch width, mm (slightly open gauge so the structure reads on a phone)
export const ROW = 4.0; // row height, mm
export const RY = 0.95; // yarn radius, mm (DK ~1.9 mm)
export const RING_OPEN = 7.0;
export const RING_CLOSED = 1.0;
export const RING_TOP = 3.0;
export const RY_FAB = 0.72; // yarn radius once settled in the fabric (compressed) // first-round stitch height above the ring

/**
 * Expand compact pattern rounds ("mr 6", "inc 6 to 18", "even 2", "dec 6 to 6", "18") into
 * per-round stitch sequences: [{n, seq:[{kind:'sc'|'inc'|'dec'}...]}]
 */
export function expandRounds(rounds) {
  const out = [];
  let n = 0;
  const pushTo = (target) => {
    if (target === n) { out.push({ n, seq: Array.from({ length: n }, () => ({ kind: 'sc' })) }); return; }
    if (target > n) {
      const incs = target - n, g = n / incs;
      const seq = [];
      for (let k = 0; k < incs; k++) {
        const scs = Math.round((k + 1) * g) - Math.round(k * g) - 1;
        for (let s = 0; s < scs; s++) seq.push({ kind: 'sc' });
        seq.push({ kind: 'inc' });
      }
      out.push({ n: target, seq });
    } else {
      const decs = n - target, g = n / decs;
      const seq = [];
      for (let k = 0; k < decs; k++) {
        const used = Math.round((k + 1) * g) - Math.round(k * g);
        for (let s = 0; s < used - 2; s++) seq.push({ kind: 'sc' });
        seq.push({ kind: 'dec' });
      }
      out.push({ n: target, seq });
    }
    n = target;
  };
  for (const r of rounds) {
    const t = r.trim().split(/\s+/);
    if (t[0] === 'mr') { n = +t[1]; out.push({ n, seq: Array.from({ length: n }, () => ({ kind: 'sc' })), mr: true }); continue; }
    if (t[0] === 'even') { for (let k = 0; k < +t[1]; k++) pushTo(n); continue; }
    if (t[0] === 'inc' || t[0] === 'dec') {
      const step = +t[1], target = +t[3];
      const every = t[4] === 'every' ? +t[5] : 1;
      while (n !== target) {
        const next = t[0] === 'inc' ? Math.min(target, n + step) : Math.max(target, n - step);
        pushTo(next);
        for (let e = 1; e < every && n !== target; e++) pushTo(n);
      }
      continue;
    }
    if (/^\d+$/.test(t[0])) { pushTo(+t[0]); continue; }
    throw new Error('Unknown round: ' + r);
  }
  return out;
}

/** Build the stitch list. Stitch 0 is the starting chain made on the magic ring. */
export function buildStitches(rounds) {
  const st = [];
  st.push({ i: 0, round: 1, kind: 'ch', parents: [], j: -1 });
  let prev = null;
  rounds.forEach((rd, ki) => {
    const k = ki + 1;
    const cur = [];
    let pi = 0;
    let j = 0;
    for (const s of rd.seq) {
      if (k === 1) { cur.push({ kind: 'sc', parents: [] }); continue; }
      if (s.kind === 'sc') cur.push({ kind: 'sc', parents: [prev[pi++]] });
      else if (s.kind === 'inc') { const p = prev[pi++]; cur.push({ kind: 'inc', sub: 0, parents: [p] }); cur.push({ kind: 'inc', sub: 1, parents: [p] }); }
      else if (s.kind === 'dec') { cur.push({ kind: 'dec', parents: [prev[pi], prev[pi + 1]] }); pi += 2; }
    }
    for (const c of cur) { c.i = st.length; c.round = k; c.j = j++; c.n = rd.n; st.push(c); }
    prev = cur.map((c) => c.i);
  });
  // round coordinate & angle along the spiral
  for (const s of st) {
    if (s.kind === 'ch') { s.r = 0; continue; }
    s.r = s.round - 1 + (s.j + 1) / s.n;
  }
  return st;
}

/** Profile of the surface of revolution: radius & depth per round coordinate r (0 = ring). */
export function buildProfile(rounds) {
  const rho = [RING_CLOSED], z = [0];
  rounds.forEach((rd, ki) => {
    const R = Math.max(RING_CLOSED + 0.5, (rd.n * SW) / (2 * Math.PI));
    const dr = R - rho[ki];
    const dz = -Math.sqrt(Math.max(ROW * ROW - dr * dr, (0.3 * ROW) ** 2));
    rho.push(ki === 0 ? RING_CLOSED + RING_TOP : R);
    z.push(ki === 0 ? 0 : z[ki] + dz);
  });
  return { rho, z, K: rounds.length };
}

export class Fabric {
  constructor(roundsSpec, theta0 = Math.PI / 2) {
    this.rounds = expandRounds(roundsSpec);
    this.stitches = buildStitches(this.rounds);
    this.profile = buildProfile(this.rounds);
    this.theta0 = theta0;
    this.ring = RING_OPEN; // current ring radius
  }
  get closedness() { return clamp((RING_OPEN - this.ring) / (RING_OPEN - RING_CLOSED)); }
  profileAt(r) {
    const { rho, z, K } = this.profile;
    const rr = clamp(r, 0, K);
    const k = Math.min(K - 1, Math.floor(rr));
    const t = rr - k;
    // smooth (cosine) interpolation for a rounder look
    const tt = t;
    return [mix(rho[k], rho[k + 1], tt), mix(z[k], z[k + 1], tt)];
  }
  /** Angle of a stitch (depends on ring state for round 1). */
  theta(s) {
    if (s.round === 1) {
      const c = smooth(this.closedness);
      const dOpen = SW / (this.ring + 1.6);
      const d = mix(SW / (RING_OPEN + 1.6), (2 * Math.PI) / 6, c);
      void dOpen;
      return this.theta0 + (s.kind === 'ch' ? 0 : s.j + 1) * d;
    }
    return this.theta0 + 2 * Math.PI * s.r;
  }
  posAt(r, th) {
    const [rho, z] = this.profileAt(r);
    return [rho * Math.cos(th), rho * Math.sin(th), z];
  }
  /** Top point, base point and frame (u: work direction, v: up the stitch, w: outward) of a stitch. */
  frame(s) {
    if (s._frame && s._frameRing === this.ring) return s._frame;
    const th = this.theta(s);
    const u = [-Math.sin(th), Math.cos(th), 0];
    let T, B, v;
    if (s.round === 1) {
      const rb = this.ring, rt = this.ring + (s.kind === 'ch' ? RING_TOP * 0.75 : RING_TOP);
      B = [rb * Math.cos(th), rb * Math.sin(th), 0];
      T = [rt * Math.cos(th), rt * Math.sin(th), 0];
      v = [Math.cos(th), Math.sin(th), 0];
    } else {
      // tops sit on their round's circle (the spiral only advances the angle)
      T = this.posAt(s.round, th);
      v = norm(sub(T, this.posAt(s.round - 1, th)));
      const ps = s.parents.map((pi) => this.frame(this.stitches[pi]).T);
      B = ps.length === 1 ? ps[0] : lerp(ps[0], ps[1], 0.5);
    }
    const w = norm(cross(v, u));
    const uu = norm(cross(w, v)); // re-orthogonalise
    const f = { T, B, u: uu, v, w, th };
    s._frame = f; s._frameRing = this.ring;
    return f;
  }
  /** Local stitch coordinates -> fabric coordinates. a in stitch widths, b in height fraction, c in yarn radii. */
  pt(s, a, b, c) {
    const f = this.frame(s);
    return add(add(lerp(f.B, f.T, b), mul(f.u, a * SW)), mul(f.w, c * RY_FAB * 1.15));
  }
  ringPoint(th, c = 0) {
    const r = this.ring;
    return [r * Math.cos(th), r * Math.sin(th), c * RY];
  }
}

// Canonical relaxed paths in local stitch coordinates [a, b, c]
// (a: stitch widths along the round, b: fraction of row height above the parent's top, c: yarn radii outward).
// Post (L1): working yarn goes down the back, through the stitch below to the front, up the left arm of
// the "V", arches over the top and comes down the right arm, back through the same hole.
export const POST = [
  [-0.42, 0.7, -1.1], [-0.22, 0.2, -1.15], [-0.13, -0.02, -1.1], [-0.13, -0.32, 0.0], [-0.13, 0.08, 1.1],
  [-0.27, 0.55, 1.12], [-0.34, 0.9, 0.9], [-0.12, 1.04, 0.35], [0.12, 1.04, 0.35], [0.34, 0.9, 0.9],
  [0.27, 0.55, 1.12], [0.13, 0.08, 1.1], [0.13, -0.32, 0.0], [0.13, -0.02, -1.1],
];
// Top (L2): the loop that was on the hook; becomes the stitch's top "chain" (front & back loops).
export const TOP = [
  [0.18, 0.55, -1.15], [0.22, 0.98, -0.9], [0.4, 1.22, -0.35], [0.56, 1.2, 0.5], [0.64, 0.98, -0.55],
];
// Invisible decrease: the post only passes under the two front loops.
export const POST_DEC = POST.map(([a, b, c]) => (b < 0.12 && Math.abs(c) < 1.2 ? [a * 1.6, b < -0.1 ? 0.12 : 0.2, c < -0.5 ? -0.2 : c < 0.5 ? 0.5 : 1.15] : [a, b, c]));

export function canonicalCtrl(fab, s, piece) {
  const tpl = piece === 'P' ? (s.kind === 'dec' ? POST_DEC : POST) : TOP;
  return tpl.map(([a, b, c]) => fab.pt(s, a, b, c));
}

/** Ring control points from the tail root around to the start of the first post. */
export function ringCtrl(fab) {
  const th0 = fab.theta0 - 0.95;
  const out = [];
  const n = 16;
  for (let k = 0; k <= n; k++) {
    const t = k / n;
    const th = th0 + t * (2 * Math.PI - 0.5);
    const c = mix(-0.9, 0.9, smooth(t));
    out.push(fab.ringPoint(th, c));
  }
  return out;
}
export function tailRoot(fab) { return fab.ringPoint(fab.theta0 - 0.95, -0.9); }
