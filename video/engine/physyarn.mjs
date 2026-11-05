// Physical yarn: an inextensible, slightly stiff rod of particles (XPBD, small substeps) under gravity and air drag.
// Nothing moves the yarn except physics: gravity, its own length and bending stiffness, contact with the hook
// (with friction, the hook is the only animated object), contact with itself, the other hand's pinch (a fixed hold)
// and the feeding finger (an eyelet that pays yarn out when it is pulled and takes up slack gently, like the
// index finger of the yarn hand). Every particle carries its material coordinate s, so the ply texture moves only
// where the yarn itself moves.
import { add, sub, mul, madd, dot, cross, norm, len, dist, lerp, clamp } from './vec.mjs';
import { HOOK, surfaceRadius, toHookLocal, hookFrame } from './hook.mjs';

export class PhysYarn {
  constructor(pts, opts = {}) {
    this.r = opts.r ?? 0.95; // yarn radius (mm)
    this.L0 = opts.L0 ?? 0.55; // particle spacing
    this.g = opts.g ?? [0, -9810, 0]; // mm/s^2
    this.drag = opts.drag ?? 6.0; // air drag 1/s (yarn is light and fluffy)
    this.bend = opts.bend ?? 0.08; // bending stiffness (0..1 per substep)
    this.mu = opts.mu ?? 0.35; // yarn-hook friction (polished aluminium)
    this.muSelf = opts.muSelf ?? 0.5;
    this.feedT = opts.feedT ?? 0.03; // eyelet: stretch held by the yarn hand (its tension)
    this.takeUp = opts.takeUp ?? 0.002; // eyelet: slack taken up per substep at most (mm)
    this.x = pts.map((p) => p.slice());
    this.v = pts.map(() => [0, 0, 0]);
    this.s = [];
    let s = 0;
    for (let i = 0; i < pts.length; i++) { if (i) s += dist(pts[i - 1], pts[i]); this.s.push(s); }
    this.rest = []; for (let i = 0; i < pts.length - 1; i++) this.rest.push(dist(pts[i], pts[i + 1]));
    this.pin = new Map(); // particle index -> fixed world point (the other hand's pinch)
    this.eye = opts.eye || null; // feeding finger eyelet: the last particle is attached here
    this.colliders = opts.colliders || []; // static capsules {a, b, r} (fingers)
    this.hook = null; this.hookPrev = null;
    this.t = 0;
  }

  setHook(pose) { this.hookPrev = this.hook || pose; this.hook = pose; }

  // ---- signed distance to the hook (hook-local radial model, same surface the renderer draws)
  hookSDF(F, p, filled = false) {
    const [x, y, z] = toHookLocal(F, p);
    if (x > 0) return { d: Math.hypot(x, y, z), n: norm(sub(p, F.tip)) };
    if (x < -80) return { d: 1e9 };
    const rr = Math.hypot(y, z), phi = filled ? Math.PI : Math.atan2(z, y);
    const R = surfaceRadius(x, phi);
    // radial normal, tilted by the slope of the surface along the axis (so the throat wall and the head push the
    // yarn along the axis as well as outward)
    const h = 0.05;
    const dR = (surfaceRadius(x + h, phi) - surfaceRadius(x - h, phi)) / (2 * h);
    // (filled: the throat is occupied by yarn the hook holds, so other yarn slides over it as over the round body)
    const radial = rr > 1e-6 ? norm(add(mul(F.d, y / rr), mul(F.s, z / rr))) : F.d;
    const n = norm(add(radial, mul(F.a, -dR)));
    return { d: (rr - R) / Math.sqrt(1 + dR * dR), n };
  }

  step(dt, poseAt) {
    const N = this.x.length;
    const S = this.substeps || 48;
    const h = dt / S;
    for (let k = 0; k < S; k++) {
      const u0 = k / S, u1 = (k + 1) / S;
      const F0 = hookFrame(poseAt(this.t + u0 * dt)), F1 = hookFrame(poseAt(this.t + u1 * dt));
      this.substep(h, F0, F1);
    }
    this.t += dt;
  }

  substep(h, F0, F1) {
    const X = this.x, V = this.v, N = X.length;
    const prev = X.map((p) => p.slice());
    const damp = Math.exp(-this.drag * h);
    for (let i = 0; i < N; i++) {
      V[i] = mul(madd(V[i], this.g, h), damp);
      X[i] = madd(X[i], V[i], h);
    }
    for (let it = 0; it < (this.iters || 6); it++) {
      this.solveStretch();
      this.solveBend();
    }
    this.solveSelf(prev);
    this.solvePins();
    for (let it = 0; it < 2; it++) { this.solveStretch(); this.solveHook(prev, F0, F1); }
    this.solveColliders();
    for (let i = 0; i < N; i++) V[i] = mul(sub(X[i], prev[i]), 1 / h);
    this.feed();
  }

  solvePins() { const k = this.pinK ?? 0.5; for (const [i, p] of this.pin) this.x[i] = lerp(this.x[i], p, k); if (this.eye) this.x[this.x.length - 1] = this.eye.p.slice(); }

  w(i) { return this.pin.has(i) ? 0.25 : (this.eye && i === this.x.length - 1) ? 0 : 1; }

  solveStretch() {
    const X = this.x;
    // alternate direction so tension propagates both ways
    const fwd = (this._dir = !this._dir);
    const n = X.length - 1;
    for (let q = 0; q < n; q++) {
      const i = fwd ? q : n - 1 - q;
      const wa = this.w(i), wb = this.w(i + 1); if (wa + wb === 0) continue;
      const d = sub(X[i + 1], X[i]), l = len(d) || 1e-9;
      const c = (l - this.rest[i]) / (wa + wb);
      const cor = mul(d, c / l);
      if (wa) X[i] = add(X[i], mul(cor, wa));
      if (wb) X[i + 1] = sub(X[i + 1], mul(cor, wb));
    }
  }

  solveBend() {
    // pull each interior particle toward the midpoint of its neighbours (discrete curvature), weakly
    const X = this.x, k = this.bend;
    for (let i = 1; i < X.length - 1; i++) {
      if (!this.w(i)) continue;
      const m = lerp(X[i - 1], X[i + 1], 0.5);
      const d = sub(m, X[i]);
      X[i] = madd(X[i], d, k * 0.66);
      if (this.w(i - 1)) X[i - 1] = madd(X[i - 1], d, -k * 0.17);
      if (this.w(i + 1)) X[i + 1] = madd(X[i + 1], d, -k * 0.17);
    }
  }

  solveHook(prev, F0, F1) {
    const X = this.x, need = this.rHook ?? this.r * 0.98;
    this.contact = this.contact || new Array(X.length).fill(0);
    for (let i = 0; i < X.length; i++) {
      this.contact[i] = 0;
      if (!this.w(i)) continue;
      if (this.grip && this.grip.has(i)) {
        // yarn held in the throat goes where the throat goes
        const l = this.grip.get(i);
        X[i] = madd(madd(madd(F1.tip, F1.a, l[0]), F1.d, l[1]), F1.s, l[2]);
        this.contact[i] = 1;
        continue;
      }
      const q = this.hookSDF(F1, X[i], !!(this.grip && this.grip.size));
      if (q.d >= need) continue;
      const pen = need - q.d;
      X[i] = madd(X[i], q.n, pen);
      this.contact[i] = 1;
      // Coulomb friction against the moving hook surface: the point of the hook under the yarn moved by hv
      const loc = toHookLocal(F0, prev[i]);
      const was = madd(madd(madd(F1.tip, F1.a, loc[0]), F1.d, loc[1]), F1.s, loc[2]);
      const hv = sub(was, prev[i]); // displacement the hook surface would give this point this substep
      const rel = sub(sub(X[i], prev[i]), hv);
      const tang = madd(rel, q.n, -dot(rel, q.n));
      const tl = len(tang);
      if (tl > 1e-9) {
        const lim = this.mu * pen;
        X[i] = madd(X[i], tang, -Math.min(1, lim / tl));
      }
    }
  }

  solveColliders() {
    const X = this.x, need = this.r;
    for (const c of this.colliders) {
      for (let i = 0; i < X.length; i++) {
        if (!this.w(i)) continue;
        const ab = sub(c.b, c.a), t = clamp(dot(sub(X[i], c.a), ab) / dot(ab, ab));
        const cp = madd(c.a, ab, t), d = sub(X[i], cp), l = len(d);
        const R = c.r + need;
        if (l < R && l > 1e-9) X[i] = madd(cp, d, R / l);
      }
    }
  }

  solveSelf(prev) {
    // segment-segment contact between non-neighbouring segments, spatial hash on segment midpoints
    const X = this.x, n = X.length - 1, D = this.r * 2 * 0.92, cell = 2.2;
    const grid = new Map();
    const key = (a, b, c) => `${a},${b},${c}`;
    for (let i = 0; i < n; i++) {
      const m = lerp(X[i], X[i + 1], 0.5);
      const k = key(Math.floor(m[0] / cell), Math.floor(m[1] / cell), Math.floor(m[2] / cell));
      if (!grid.has(k)) grid.set(k, []); grid.get(k).push(i);
    }
    for (let i = 0; i < n; i++) {
      const m = lerp(X[i], X[i + 1], 0.5);
      const cx = Math.floor(m[0] / cell), cy = Math.floor(m[1] / cell), cz = Math.floor(m[2] / cell);
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
        const L = grid.get(key(cx + a, cy + b, cz + c)); if (!L) continue;
        for (const j of L) {
          if (j <= i + 3) continue;
          const [sa, tb] = segSegCl(X[i], X[i + 1], X[j], X[j + 1]);
          const p = lerp(X[i], X[i + 1], sa), q = lerp(X[j], X[j + 1], tb);
          const d = sub(p, q), l = len(d);
          if (l >= D || l < 1e-9) continue;
          const nrm = mul(d, 1 / l), corr = (D - l) * 0.5;
          const wi0 = this.w(i) * (1 - sa), wi1 = this.w(i + 1) * sa, wj0 = this.w(j) * (1 - tb), wj1 = this.w(j + 1) * tb;
          const W = wi0 * wi0 + wi1 * wi1 + wj0 * wj0 + wj1 * wj1; if (W < 1e-9) continue;
          const lam = (D - l) / W;
          X[i] = madd(X[i], nrm, lam * wi0); X[i + 1] = madd(X[i + 1], nrm, lam * wi1);
          X[j] = madd(X[j], nrm, -lam * wj0); X[j + 1] = madd(X[j + 1], nrm, -lam * wj1);
          // friction between the two strands: damp their relative sliding a little
          const ri = sub(lerp(X[i], X[i + 1], sa), lerp(prev[i], prev[i + 1], sa));
          const rj = sub(lerp(X[j], X[j + 1], tb), lerp(prev[j], prev[j + 1], tb));
          const rel = sub(ri, rj), tg = madd(rel, nrm, -dot(rel, nrm)), tl = len(tg);
          if (tl > 1e-9) {
            const f = Math.min(1, (this.muSelf * corr) / tl) * 0.5;
            const dv = mul(tg, -f);
            X[i] = madd(X[i], dv, wi0); X[i + 1] = madd(X[i + 1], dv, wi1);
            X[j] = madd(X[j], dv, -wj0); X[j + 1] = madd(X[j + 1], dv, -wj1);
          }
        }
      }
    }
  }

  // the yarn hand: pays yarn out of the eyelet when the yarn is pulled, takes up slack gently otherwise
  feed() {
    if (!this.eye) return;
    const X = this.x, N = X.length, k = N - 2;
    const l = dist(X[k], X[N - 1]);
    const r = this.rest[k];
    // the hand keeps a light, constant tension: the eyelet segment is held at a small stretch s0 (= tension);
    // more stretch pays yarn out, less takes slack in, both at a limited rate (fingers, not a motor)
    const ds = (l - r) - this.feedT;
    this.rest[k] = Math.max(0.05, r + clamp(ds * 0.5, -this.takeUp, 0.03));
    // material enters/leaves through the eyelet: split or merge the last segment
    if (this.rest[k] > this.L0 * 1.5) {
      const p = lerp(X[k], X[N - 1], this.L0 / this.rest[k]);
      const sNew = this.s[k] + this.L0;
      X.splice(N - 1, 0, p); this.v.splice(N - 1, 0, this.v[k].slice()); this.s.splice(N - 1, 0, sNew);
      this.rest.splice(k, 1, this.L0, this.rest[k] - this.L0);
      this.s[X.length - 1] = this.s[X.length - 2] + this.rest[X.length - 2];
    } else if (this.rest[k] < this.L0 * 0.4 && N > 4) {
      X.splice(k, 1); this.v.splice(k, 1); this.s.splice(k, 1);
      this.rest.splice(k - 1, 2, this.rest[k - 1] + this.rest[k]);
      this.s[X.length - 1] = this.s[X.length - 2] + this.rest[X.length - 2];
    } else this.s[N - 1] = this.s[k] + this.rest[k];
  }
}

export function segSegCl(p1, q1, p2, q2) {
  const d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
  const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r), c = dot(d1, r), b = dot(d1, d2);
  const den = a * e - b * b;
  let s = den > 1e-9 ? clamp((b * f - c * e) / den) : 0;
  let t = (b * s + f) / e;
  if (t < 0) { t = 0; s = clamp(-c / a); } else if (t > 1) { t = 1; s = clamp((b - c) / a); }
  return [s, t];
}

/** Resample a polyline at a fixed spacing. */
export function resample(pts, L0) {
  const out = [pts[0]]; let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i], seg = dist(a, b); if (seg < 1e-9) continue;
    let s = L0 - carry;
    while (s <= seg) { out.push(lerp(a, b, s / seg)); s += L0; }
    carry = seg - (s - L0);
  }
  out.push(pts[pts.length - 1]);
  return out;
}
