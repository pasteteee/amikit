// Persistent yarn simulation (XPBD-style position dynamics). The yarn is ONE fixed chain of material
// particles that lives for the whole video: constant radius, inextensible, solid against itself and
// against the hook. Choreography only moves the hook, the left index finger (yarn feed) and pins; the
// yarn reacts. New material enters through the finger (feed) and can be pulled back (tension).
import { surfaceRadius, baseRadius } from './hook.mjs';

const TAU = Math.PI * 2;
const HOOK_R_KEEP = 1.25 + 0.95 + 1.0;

export class YarnSim {
  /**
   * pts: initial centreline [[x,y,z]...] (spacing ~h), opts: {h, r, finger:{G, dir}, tension}
   */
  constructor(pts, opts = {}) {
    this.h = opts.h ?? 0.6;
    this.r = opts.r ?? 0.95;
    this.n = pts.length;
    this.cap = this.n + 4000;
    this.x = new Float64Array(this.cap * 3);
    this.p = new Float64Array(this.cap * 3); // previous positions
    this.w = new Float64Array(this.cap); // inverse mass (0 = pinned)
    this.s = new Float64Array(this.cap); // material coordinate
    for (let i = 0; i < this.n; i++) {
      for (let k = 0; k < 3; k++) this.x[i * 3 + k] = this.p[i * 3 + k] = pts[i][k];
      this.w[i] = 1; this.s[i] = i * this.h;
    }
    this.gravity = opts.gravity ?? [0, -9810, 0];
    this.damping = opts.damping ?? 6; // 1/s, air + internal friction
    this.substeps = opts.substeps ?? 24;
    this.iters = opts.iters ?? 4;
    this.bend = opts.bend ?? 0.08; // bending stiffness per substep (yarn is soft)
    this.muYarn = opts.muYarn ?? 0.35;
    this.muHook = opts.muHook ?? 0.15;
    this.finger = opts.finger ?? null; // {G:[..], dir:[..] unit toward the ball, T: tension force (N-ish)}
    this.pins = new Map(); // index -> [x,y,z]
    this.hook = null; this.hookPrev = null;
    this.stats = { fed: 0, retracted: 0 };
    this.keepFrom = null; this.keepMargin = 1.7;
  }

  pin(i, pos) { this.pins.set(i, pos.slice()); this.w[i] = 0; }
  unpin(i) { this.pins.delete(i); this.w[i] = 1; }

  positions() {
    const out = [];
    for (let i = 0; i < this.n; i++) out.push([this.x[i * 3], this.x[i * 3 + 1], this.x[i * 3 + 2]]);
    return out;
  }
  material() { return Array.from(this.s.subarray(0, this.n)); }

  /** Advance by dt with the hook moving from its previous pose to `hook` ({tip,a,d}). */
  step(dt, hook) {
    this.hookPrev = this.hook || hook;
    this.hook = hook;
    const S = this.substeps, h = dt / S;
    for (let k = 1; k <= S; k++) {
      const hk = lerpPose(this.hookPrev, this.hook, k / S);
      const hkPrev = lerpPose(this.hookPrev, this.hook, (k - 1) / S);
      this.substep(h, hk, hkPrev);
    }
  }

  substep(h, hk, hkPrev) {
    const { x, p, w } = this;
    const n = this.n;
    const damp = Math.exp(-this.damping * h);
    const g = this.gravity;
    // predict
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      if (w[i] === 0) { const q = this.pins.get(i); if (q) { p[o] = x[o]; p[o + 1] = x[o + 1]; p[o + 2] = x[o + 2]; x[o] = q[0]; x[o + 1] = q[1]; x[o + 2] = q[2]; } continue; }
      const vx = (x[o] - p[o]) * damp, vy = (x[o + 1] - p[o + 1]) * damp, vz = (x[o + 2] - p[o + 2]) * damp;
      p[o] = x[o]; p[o + 1] = x[o + 1]; p[o + 2] = x[o + 2];
      x[o] += vx + g[0] * h * h; x[o + 1] += vy + g[1] * h * h; x[o + 2] += vz + g[2] * h * h;
    }
    // finger: the end particle slides on the feed ray and is pulled into the hand with tension T
    if (this.finger) this.fingerForce(h);
    for (let it = 0; it < this.iters; it++) {
      this.solveStretch();
      if (it % 2 === 0) this.solveBend();
      this.solveHook(hk, hkPrev);
      if (this.catcher) this.solveCatch(hk);
      if (this.hold) this.solveHold();
      this.solveSelf();
    }
    this.solveStretch();
    this.solveHook(hk, hkPrev);
    this.solveSelf(); this.solveSelf(); // last word goes to yarn-yarn contact: nothing may tunnel through a strand
    if (this.finger) this.fingerFeed();
  }

  solveStretch() {
    const { x, w, h } = this;
    const n = this.n;
    // forward + backward Gauss-Seidel sweep: tension travels both ways in one iteration
    for (let pass = 0; pass < 2; pass++) {
      for (let q = 0; q < n - 1; q++) {
        const i = pass === 0 ? q : n - 2 - q, j = i + 1;
        const wi = w[i], wj = w[j], ws = wi + wj;
        if (ws === 0) continue;
        const oi = i * 3, oj = j * 3;
        const dx = x[oj] - x[oi], dy = x[oj + 1] - x[oi + 1], dz = x[oj + 2] - x[oi + 2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-9;
        const c = (d - h) / (d * ws);
        x[oi] += dx * c * wi; x[oi + 1] += dy * c * wi; x[oi + 2] += dz * c * wi;
        x[oj] -= dx * c * wj; x[oj + 1] -= dy * c * wj; x[oj + 2] -= dz * c * wj;
      }
    }
  }

  solveBend() {
    // soft bending: pull i toward the line through its neighbours (keeps yarn round, no kinks)
    const { x, w } = this;
    const k = this.bend;
    for (let i = 1; i < this.n - 1; i++) {
      if (w[i] === 0) continue;
      const a = (i - 1) * 3, b = i * 3, c = (i + 1) * 3;
      for (let m = 0; m < 3; m++) {
        const mid = (x[a + m] + x[c + m]) * 0.5;
        const d = (mid - x[b + m]) * k;
        x[b + m] += d * 0.6;
        if (w[i - 1]) x[a + m] -= d * 0.2;
        if (w[i + 1]) x[c + m] -= d * 0.2;
      }
    }
  }

  /** Hook as a solid: signed distance to its surface (round shaft, tapered head, throat groove). */
  solveHook(hk, hkPrev) {
    if (!hk) return;
    const { x, p, w } = this;
    const r = this.r;
    const a = hk.a, d = hk.d, s = hk.s, t = hk.tip;
    for (let i = 0; i < this.n; i++) {
      if (w[i] === 0) continue;
      const o = i * 3;
      const qx = x[o] - t[0], qy = x[o + 1] - t[1], qz = x[o + 2] - t[2];
      let lx = qx * a[0] + qy * a[1] + qz * a[2];
      if (lx > r + 0.2 || lx < -60) continue;
      let ly = qx * d[0] + qy * d[1] + qz * d[2];
      let lz = qx * s[0] + qy * s[1] + qz * s[2];
      let rho = Math.hypot(ly, lz);
      if (rho > 1.4 + r + 0.3 && lx < -2) { if (rho > 4.5) continue; }
      let moved = false;
      for (let rep = 0; rep < 3; rep++) {
        let sd, nx, nr; // signed distance minus radius, normal in (axial, radial) components
        if (lx > -0.05) {
          const R = Math.hypot(lx, rho);
          sd = R - r; nx = lx / (R || 1); nr = rho / (R || 1);
        } else {
          const phi = Math.atan2(lz, ly);
          const Rs = surfaceRadius(lx, phi);
          const e = 0.04;
          const dR = (surfaceRadius(lx + e, phi) - surfaceRadius(lx - e, phi)) / (2 * e);
          const cs = 1 / Math.sqrt(1 + dR * dR);
          sd = (rho - Rs) * cs - r; nx = -dR * cs; nr = cs;
        }
        if (sd >= 0) break;
        const push = -sd;
        const ur = rho > 1e-6 ? [ly / rho, lz / rho] : [-1, 0];
        lx += nx * push; ly += ur[0] * nr * push; lz += ur[1] * nr * push;
        rho = Math.hypot(ly, lz);
        moved = true;
      }
      // the hand keeps loops on the hook: yarn wrapped around the shaft (index >= keepFrom) may slide along it
      // but not off the tip
      if (this.keepFrom != null && i >= this.keepFrom && rho < HOOK_R_KEEP && lx > -this.keepMargin && lx < -0.0) { lx = -this.keepMargin; moved = true; }
      if (!moved) continue;
      const nxw = t[0] + a[0] * lx + d[0] * ly + s[0] * lz;
      const nyw = t[1] + a[1] * lx + d[1] * ly + s[1] * lz;
      const nzw = t[2] + a[2] * lx + d[2] * ly + s[2] * lz;
      // friction: the contact point moves with the hook; damp sliding relative to it
      const hp = hkPrev;
      const lpx = lx, lpy = ly, lpz = lz;
      const ox = hp.tip[0] + hp.a[0] * lpx + hp.d[0] * lpy + hp.s[0] * lpz;
      const oy = hp.tip[1] + hp.a[1] * lpx + hp.d[1] * lpy + hp.s[1] * lpz;
      const oz = hp.tip[2] + hp.a[2] * lpx + hp.d[2] * lpy + hp.s[2] * lpz;
      const hvx = nxw - ox, hvy = nyw - oy, hvz = nzw - oz; // hook surface displacement this substep
      const rvx = (nxw - p[o]) - hvx, rvy = (nyw - p[o + 1]) - hvy, rvz = (nzw - p[o + 2]) - hvz;
      const mu = this.muHook;
      x[o] = nxw - rvx * mu; x[o + 1] = nyw - rvy * mu; x[o + 2] = nzw - rvz * mu;
    }
  }

  /** Yarn is solid against itself: particle spheres of non-neighbours may not overlap. */
  solveSelf() {
    const { x, p, w } = this;
    const n = this.n, r2 = 2 * this.r;
    const skip = Math.ceil(r2 / this.h) + 1;
    const cell = r2;
    const grid = new Map();
    const key = (a, b, c) => ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) | 0;
    const ci = new Int32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const a = Math.floor(x[i * 3] / cell), b = Math.floor(x[i * 3 + 1] / cell), c = Math.floor(x[i * 3 + 2] / cell);
      ci[i * 3] = a; ci[i * 3 + 1] = b; ci[i * 3 + 2] = c;
      const k = key(a, b, c);
      let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(i);
    }
    const mu = this.muYarn;
    for (let i = 0; i < n; i++) {
      const A = ci[i * 3], B = ci[i * 3 + 1], Cc = ci[i * 3 + 2];
      for (let da = -1; da <= 1; da++) for (let db = -1; db <= 1; db++) for (let dc = -1; dc <= 1; dc++) {
        const l = grid.get(key(A + da, B + db, Cc + dc)); if (!l) continue;
        for (const j of l) {
          if (j <= i + skip) continue;
          const wi = w[i], wj = w[j], ws = wi + wj; if (ws === 0) continue;
          const oi = i * 3, oj = j * 3;
          const dx = x[oj] - x[oi], dy = x[oj + 1] - x[oi + 1], dz = x[oj + 2] - x[oi + 2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 >= r2 * r2 || d2 < 1e-12) continue;
          const d = Math.sqrt(d2), nx = dx / d, ny = dy / d, nz = dz / d;
          const c = (r2 - d) / ws;
          x[oi] -= nx * c * wi; x[oi + 1] -= ny * c * wi; x[oi + 2] -= nz * c * wi;
          x[oj] += nx * c * wj; x[oj + 1] += ny * c * wj; x[oj + 2] += nz * c * wj;
          // friction: remove part of the relative tangential motion
          const rvx = (x[oj] - p[oj]) - (x[oi] - p[oi]), rvy = (x[oj + 1] - p[oj + 1]) - (x[oi + 1] - p[oi + 1]), rvz = (x[oj + 2] - p[oj + 2]) - (x[oi + 2] - p[oi + 2]);
          const vn = rvx * nx + rvy * ny + rvz * nz;
          const tx = rvx - vn * nx, ty = rvy - vn * ny, tz = rvz - vn * nz;
          const f = mu / ws;
          x[oi] += tx * f * wi; x[oi + 1] += ty * f * wi; x[oi + 2] += tz * f * wi;
          x[oj] -= tx * f * wj; x[oj + 1] -= ty * f * wj; x[oj + 2] -= tz * f * wj;
        }
      }
    }
  }

  /**
   * The throat holds a bight of yarn: the yarn must pass through the groove point (hook-local x = xc,
   * radius rc on the throat side) but may SLIDE through it freely (Eulerian contact). Each iteration the
   * currently closest particle (and its neighbours, less) is pulled onto that point.
   */
  // the other hand pinches the stitch: particles i0..i1 keep their coordinate along one axis (the hook slides through)
  solveHold() {
    const H = this.hold, x = this.x;
    for (let i = H.i0; i <= H.i1 && i < this.n; i++) { if (this.w[i] === 0) continue; const o = i * 3 + H.axis; x[o] += (H.val[i - H.i0] - x[o]) * H.stiff; }
  }

  solveCatch(hk) {
    const C = this.catcher, x = this.x, w = this.w;
    const P = [hk.tip[0] + hk.a[0] * C.xc + hk.d[0] * C.rc, hk.tip[1] + hk.a[1] * C.xc + hk.d[1] * C.rc, hk.tip[2] + hk.a[2] * C.xc + hk.d[2] * C.rc];
    let best = -1, bd = 1e9;
    const lo = Math.max(1, (C.k ?? 0) - 12), hi = Math.min(this.n - 2, C.k == null ? this.n - 2 : C.k + 12);
    for (let i = C.k == null ? C.from : lo; i <= (C.k == null ? this.n - 2 : hi); i++) {
      const o = i * 3; const d2 = (x[o] - P[0]) ** 2 + (x[o + 1] - P[1]) ** 2 + (x[o + 2] - P[2]) ** 2;
      if (d2 < bd) { bd = d2; best = i; }
    }
    if (best < 0) return;
    C.k = best;
    const pull = (i, f) => { if (w[i] === 0) return; const o = i * 3; for (let m = 0; m < 3; m++) x[o + m] += (P[m] - x[o + m]) * f; };
    pull(best, C.stiff ?? 0.7);
  }

  // ----- finger feed: the last particle sits on the feed ray G + dir*u (u in [0, h]); tension pulls it in
  fingerForce(h) {
    const F = this.finger, e = (this.n - 1) * 3, x = this.x;
    // tension: constant pull along the ray into the hand
    const T = F.T ?? 2000; // mm/s^2 equivalent acceleration on the end particle
    x[e] += F.dir[0] * T * h * h; x[e + 1] += F.dir[1] * T * h * h; x[e + 2] += F.dir[2] * T * h * h;
  }
  fingerFeed() {
    const F = this.finger, x = this.x, p = this.p;
    let e = (this.n - 1) * 3;
    // project the end particle onto the ray
    let u = (x[e] - F.G[0]) * F.dir[0] + (x[e + 1] - F.G[1]) * F.dir[1] + (x[e + 2] - F.G[2]) * F.dir[2];
    // pulled out of the hand (u < 0): feed new yarn at the finger
    while (u < 0 && this.n < this.cap - 1) {
      const i = this.n;
      const o = i * 3;
      for (let k = 0; k < 3; k++) { x[o + k] = x[e + k] + F.dir[k] * this.h; p[o + k] = x[o + k]; }
      this.w[i] = 1; this.s[i] = this.s[i - 1] + this.h;
      this.n++; this.stats.fed++;
      e = o; u += this.h;
    }
    // drawn into the hand (u > h): retract the end particle
    while (u > this.h && this.n > 4) {
      this.n--; this.stats.retracted++;
      e = (this.n - 1) * 3;
      u = (x[e] - F.G[0]) * F.dir[0] + (x[e + 1] - F.G[1]) * F.dir[1] + (x[e + 2] - F.G[2]) * F.dir[2];
    }
    // keep the end on the ray
    const uu = Math.max(0, Math.min(this.h, u));
    for (let k = 0; k < 3; k++) x[e + k] = F.G[k] + F.dir[k] * uu;
  }
}

function lerpPose(A, B, t) {
  const tip = [A.tip[0] + (B.tip[0] - A.tip[0]) * t, A.tip[1] + (B.tip[1] - A.tip[1]) * t, A.tip[2] + (B.tip[2] - A.tip[2]) * t];
  const a = nrm([A.a[0] + (B.a[0] - A.a[0]) * t, A.a[1] + (B.a[1] - A.a[1]) * t, A.a[2] + (B.a[2] - A.a[2]) * t]);
  let d = [A.d[0] + (B.d[0] - A.d[0]) * t, A.d[1] + (B.d[1] - A.d[1]) * t, A.d[2] + (B.d[2] - A.d[2]) * t];
  const k = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
  d = nrm([d[0] - a[0] * k, d[1] - a[1] * k, d[2] - a[2] * k]);
  const s = [a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2], a[0] * d[1] - a[1] * d[0]];
  return { tip, a, d, s };
}
function nrm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
export { lerpPose };
void TAU; void baseRadius;
