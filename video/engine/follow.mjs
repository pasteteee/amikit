// Persistent yarn "follower": the stitch engine (sim.mjs) says where the yarn should be (topology, wraps,
// stitch shapes); this layer turns that target into a physical yarn that is rendered.
//  * every rendered particle is a fixed bit of material (material coordinate s = k*h) that lives across
//    frames, so the yarn can never be re-sampled, re-threaded or thinned from one frame to the next;
//  * each particle is pulled toward its target with a damped spring (inertia: no jerks, smooth catch-up);
//  * inextensible distance constraints keep the length, yarn-yarn and yarn-hook contacts keep it solid;
//  * radius is constant per region and changes only slowly (no "size pops").
import { hookFrame, pushOutOfHook } from './hook.mjs';
import { RY, RY_FAB } from './fabric.mjs';

export class YarnFollower {
  constructor(opts = {}) {
    this.h = opts.h ?? 0.4;
    this.sub = opts.substeps ?? 6;
    this.iters = opts.iters ?? 3;
    this.wWork = opts.wWork ?? 2 * Math.PI * 3.2; // natural frequency of free working yarn (rad/s)
    this.wFab = opts.wFab ?? 2 * Math.PI * 7; // fabric yarn follows its target closely
    this.zeta = opts.zeta ?? 0.95;
    this.snapDist = opts.snapDist ?? 9; // a target that teleports this far is a re-parameterisation: snap
    this.parts = new Map(); // k -> {x:[3], v:[3], r, ff}
    this.stats = { maxCorr: 0 };
  }

  // sample the target polyline (P, S monotone) at material coordinate s
  static sampler(P, S, F) {
    let j = 0;
    return (s) => {
      if (s <= S[0]) return [P[0], F ? F[0] : 0];
      if (s >= S[S.length - 1]) return [P[P.length - 1], F ? F[F.length - 1] : 0];
      while (j > 0 && S[j] > s) j--;
      while (j < S.length - 2 && S[j + 1] < s) j++;
      const u = (s - S[j]) / ((S[j + 1] - S[j]) || 1e-9);
      const a = P[j], b = P[j + 1];
      return [[a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u], F ? F[j] + (F[j + 1] - F[j]) * u : 0];
    };
  }

  step(st, dt) {
    const { h } = this;
    const S = st.s, P = st.pts, FF = st.ff || null;
    const k0 = Math.ceil(S[0] / h), k1 = Math.floor(S[S.length - 1] / h);
    const at = YarnFollower.sampler(P, S, FF);
    // birth / death of material bits at the ends (feed from the ball, tail)
    for (const k of this.parts.keys()) if (k < k0 || k > k1) this.parts.delete(k);
    const keys = [];
    for (let k = k0; k <= k1; k++) {
      const [T, ff] = at(k * h);
      let q = this.parts.get(k);
      if (!q) { q = { x: T.slice(), v: [0, 0, 0], r: ff > 0.5 ? RY_FAB : RY, ff, T }; this.parts.set(k, q); }
      q.T = T; q.ff = ff;
      if (Math.hypot(T[0] - q.x[0], T[1] - q.x[1], T[2] - q.x[2]) > this.snapDist && ff > 0.5) { q.x = T.slice(); q.v = [0, 0, 0]; }
      keys.push(k);
    }
    const n = keys.length;
    const X = keys.map((k) => this.parts.get(k));
    const speed = st.act?.type === 'stitch' ? Math.max(1, st.act.speed || 1) : 1;
    const F = hookFrame(st.hook);
    const hs = dt / this.sub;
    const xs = X.map((q) => q.x);
    for (let s = 0; s < this.sub; s++) {
      // spring toward target (semi-implicit), working yarn soft, fabric stiff; scale with playback speed
      const prev = xs.map((p) => p.slice());
      for (let i = 0; i < n; i++) {
        const q = X[i];
        const w = (q.ff > 0.5 ? this.wFab : this.wWork) * speed;
        const c = 2 * this.zeta * w;
        for (let m = 0; m < 3; m++) {
          const a = w * w * (q.T[m] - q.x[m]) - c * q.v[m];
          q.v[m] += a * hs;
          q.x[m] += q.v[m] * hs;
        }
      }
      for (let it = 0; it < this.iters; it++) {
        this.stretch(xs);
        this.collide(xs, X);
      }
      pushOutOfHook(F, xs, RY);
      for (let i = 0; i < n; i++) X[i].x = xs[i];
      // velocities follow the constraint corrections (PBD), lightly damped
      for (let i = 0; i < n; i++) for (let m = 0; m < 3; m++) X[i].v[m] = 0.98 * (xs[i][m] - prev[i][m]) / hs;
    }
    // radius: slow approach to the regional value (no pops)
    const kR = 1 - Math.exp(-dt / 0.35);
    for (const q of X) { const r0 = q.ff > 0.5 ? RY_FAB : RY; q.r += (r0 - q.r) * kR; }
    return { ...st, pts: xs.map((p) => p.slice()), s: keys.map((k) => k * h), r: X.map((q) => q.r) };
  }

  stretch(xs) {
    const h = this.h, n = xs.length;
    for (let pass = 0; pass < 2; pass++) {
      for (let q = 0; q < n - 1; q++) {
        const i = pass === 0 ? q : n - 2 - q, A = xs[i], B = xs[i + 1];
        const dx = B[0] - A[0], dy = B[1] - A[1], dz = B[2] - A[2];
        const d = Math.hypot(dx, dy, dz) || 1e-9;
        const c = (d - h) / d * 0.5;
        A[0] += dx * c; A[1] += dy * c; A[2] += dz * c;
        B[0] -= dx * c; B[1] -= dy * c; B[2] -= dz * c;
      }
    }
  }

  collide(xs, X) {
    const n = xs.length, cell = 2 * RY, inv = 1 / cell;
    const skip = Math.ceil((2 * RY) / this.h) + 2;
    const grid = new Map();
    const key = (a, b, c) => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
    for (let i = 0; i < n; i++) {
      const p = xs[i], kk = key(Math.floor(p[0] * inv), Math.floor(p[1] * inv), Math.floor(p[2] * inv));
      let L = grid.get(kk); if (!L) grid.set(kk, (L = [])); L.push(i);
    }
    for (let i = 0; i < n; i++) {
      const p = xs[i], cx = Math.floor(p[0] * inv), cy = Math.floor(p[1] * inv), cz = Math.floor(p[2] * inv);
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
        const L = grid.get(key(cx + a, cy + b, cz + c)); if (!L) continue;
        for (const j of L) {
          if (j <= i + skip) continue;
          const fi = X[i].ff > 0.5, fj = X[j].ff > 0.5;
          if (fi && fj) continue; // settled fabric already has its shape: leave it alone
          const q = xs[j];
          const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
          const R = X[i].r + X[j].r, d2 = dx * dx + dy * dy + dz * dz;
          if (d2 >= R * R || d2 < 1e-12) continue;
          const d = Math.sqrt(d2), corr = (R - d) / d;
          // the fabric is heavy: free yarn moves around it
          const wi = fi ? 0.15 : 1, wj = fj ? 0.15 : 1, ws = wi + wj;
          p[0] -= dx * corr * wi / ws; p[1] -= dy * corr * wi / ws; p[2] -= dz * corr * wi / ws;
          q[0] += dx * corr * wj / ws; q[1] += dy * corr * wj / ws; q[2] += dz * corr * wj / ws;
        }
      }
    }
  }
}
