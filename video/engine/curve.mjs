// Polyline helpers: Catmull-Rom sampling, arc-length resampling, smoothing, taut wraps.
import { add, sub, mul, madd, dot, cross, len, dist, norm, lerp, perp, anyPerp } from './vec.mjs';

/** Centripetal Catmull-Rom through control points; returns dense polyline (includes first & last). */
export function catmull(ctrl, perSeg = 6) {
  if (ctrl.length < 2) return ctrl.slice();
  const P = [madd(ctrl[0], sub(ctrl[0], ctrl[1]), 1), ...ctrl, madd(ctrl[ctrl.length - 1], sub(ctrl[ctrl.length - 1], ctrl[ctrl.length - 2]), 1)];
  const out = [ctrl[0]];
  for (let i = 1; i < P.length - 2; i++) {
    const p0 = P[i - 1], p1 = P[i], p2 = P[i + 1], p3 = P[i + 2];
    const t01 = Math.pow(dist(p0, p1) + 1e-6, 0.5), t12 = Math.pow(dist(p1, p2) + 1e-6, 0.5), t23 = Math.pow(dist(p2, p3) + 1e-6, 0.5);
    // standard centripetal tangents
    const tA = add(sub(mul(sub(p1, p0), 1 / t01), mul(sub(p2, p0), 1 / (t01 + t12))), mul(sub(p2, p1), 1 / t12));
    const tB = add(sub(mul(sub(p2, p1), 1 / t12), mul(sub(p3, p1), 1 / (t12 + t23))), mul(sub(p3, p2), 1 / t23));
    const ma = mul(tA, t12), mb = mul(tB, t12);
    for (let k = 1; k <= perSeg; k++) {
      const t = k / perSeg, t2 = t * t, t3 = t2 * t;
      const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
      out.push([
        h00 * p1[0] + h10 * ma[0] + h01 * p2[0] + h11 * mb[0],
        h00 * p1[1] + h10 * ma[1] + h01 * p2[1] + h11 * mb[1],
        h00 * p1[2] + h10 * ma[2] + h01 * p2[2] + h11 * mb[2],
      ]);
    }
  }
  return out;
}

export function polyLength(pts) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += dist(pts[i - 1], pts[i]);
  return L;
}

/** Resample a polyline into n points evenly spaced by arc length. */
export function resampleN(pts, n) {
  if (pts.length === 1) return Array.from({ length: n }, () => pts[0]);
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + dist(pts[i - 1], pts[i]));
  const L = cum[cum.length - 1] || 1e-9;
  const out = [];
  let j = 1;
  for (let k = 0; k < n; k++) {
    const s = (L * k) / (n - 1);
    while (j < cum.length - 1 && cum[j] < s) j++;
    const a = cum[j - 1], b = cum[j];
    out.push(lerp(pts[j - 1], pts[j], b > a ? (s - a) / (b - a) : 0));
  }
  return out;
}

/**
 * Taut-string wrap of yarn around a cylinder (hook shaft, throat). prev/next are the neighbouring
 * yarn points; the arc lies on the side away from the legs unless `via` is given.
 */
export function wrapArc(prev, next, center, axis, R, via = null, stepDeg = 18) {
  const e1 = via ? norm(perp(via, axis)) : (() => {
    const m = perp(sub(lerp(prev, next, 0.5), center), axis);
    return len(m) > 1e-6 ? mul(norm(m), -1) : anyPerp(axis);
  })();
  const e2 = cross(axis, e1);
  const P = sub(prev, center), Q = sub(next, center);
  const p2 = [dot(P, e1), dot(P, e2)], q2 = [dot(Q, e1), dot(Q, e2)];
  const dP = Math.max(Math.hypot(p2[0], p2[1]), R * 1.02), dQ = Math.max(Math.hypot(q2[0], q2[1]), R * 1.02);
  const aP = Math.atan2(p2[1], p2[0]), aQ = Math.atan2(q2[1], q2[0]);
  const bP = Math.acos(R / dP), bQ = Math.acos(R / dQ);
  // via direction is angle 0 in (e1,e2). Pick orientation whose arc contains angle 0.
  const TAU = Math.PI * 2;
  const mod = (x) => ((x % TAU) + TAU) % TAU;
  let best = null;
  for (const sg of [1, -1]) {
    const fin = sg > 0 ? aP + bP : aP - bP;
    const fout = sg > 0 ? aQ - bQ : aQ + bQ;
    const sweep = sg > 0 ? mod(fout - fin) : mod(fin - fout);
    const toVia = sg > 0 ? mod(0 - fin) : mod(fin - 0);
    const contains = toVia <= sweep;
    const score = (contains ? 0 : 10) + sweep;
    if (!best || score < best.score) best = { sg, fin, sweep, score };
  }
  const n = Math.max(2, Math.ceil((best.sweep * 180) / Math.PI / stepDeg));
  const axP = dot(P, axis), axQ = dot(Q, axis);
  const out = [];
  for (let k = 0; k <= n; k++) {
    const t = k / n;
    const f = best.fin + best.sg * best.sweep * t;
    // axial position drifts slightly toward the side of each leg so legs don't overlap
    const ax = (axP * (1 - t) + axQ * t) * 0.12;
    out.push(add(add(center, add(mul(e1, R * Math.cos(f)), mul(e2, R * Math.sin(f)))), mul(axis, ax)));
  }
  return out;
}

/** Resample polyline with per-point scalar attribute s at fixed spacing. */
export function resampleAttr(pts, attr, spacing) {
  const outP = [pts[0]], outA = [attr[0]];
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    const seg = dist(a, b);
    if (seg < 1e-9) continue;
    let s = spacing - carry;
    while (s <= seg) {
      const t = s / seg;
      outP.push(lerp(a, b, t));
      outA.push(attr[i - 1] + (attr[i] - attr[i - 1]) * t);
      s += spacing;
    }
    carry = seg - (s - spacing);
  }
  const last = pts[pts.length - 1];
  if (dist(outP[outP.length - 1], last) > spacing * 0.3) { outP.push(last); outA.push(attr[attr.length - 1]); }
  else { outP[outP.length - 1] = last; outA[outA.length - 1] = attr[attr.length - 1]; }
  return [outP, outA];
}

/** Light Laplacian smoothing (endpoints fixed). */
export function laplace(pts, iters = 2, alpha = 0.3, pinned = null) {
  let p = pts;
  for (let it = 0; it < iters; it++) {
    const q = p.slice();
    for (let i = 1; i < p.length - 1; i++) {
      if (pinned && pinned[i]) continue;
      const m = lerp(p[i - 1], p[i + 1], 0.5);
      q[i] = lerp(p[i], m, alpha);
    }
    p = q;
  }
  return p;
}

/** Closest point on segment ab to p, returning parameter t. */
export function closestOnSeg(p, a, b) {
  const ab = sub(b, a);
  const L2 = dot(ab, ab) || 1e-12;
  const t = Math.min(1, Math.max(0, dot(sub(p, a), ab) / L2));
  return [madd(a, ab, t), t];
}

/**
 * Bending limiter: iteratively relaxes points where the local radius of curvature is below rMin,
 * so a tube of radius ~rMin never folds through itself. Points with pinned[i] stay put.
 */
export function limitCurvature(pts, spacing, rMin, iters = 8, pinned = null) {
  const maxAng = spacing / rMin;
  const dmin = 2 * spacing * Math.cos(Math.min(maxAng, 1.2) / 2);
  const p = pts.map((q) => q.slice());
  for (let it = 0; it < iters; it++) {
    let moved = 0;
    for (let i = 1; i < p.length - 1; i++) {
      const a = p[i - 1], b = p[i + 1];
      const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
      const d = Math.hypot(dx, dy, dz);
      if (d >= dmin * 0.98) continue;
      moved++;
      const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
      if (!(pinned && pinned[i])) { p[i][0] += (m[0] - p[i][0]) * 0.5; p[i][1] += (m[1] - p[i][1]) * 0.5; p[i][2] += (m[2] - p[i][2]) * 0.5; }
      // push the neighbours apart along their chord
      if (d > 1e-6) {
        const k = ((dmin - d) / d) * 0.25;
        if (!(pinned && pinned[i - 1]) && i - 1 > 0) { a[0] -= dx * k; a[1] -= dy * k; a[2] -= dz * k; }
        if (!(pinned && pinned[i + 1]) && i + 1 < p.length - 1) { b[0] += dx * k; b[1] += dy * k; b[2] += dz * k; }
      }
    }
    if (!moved) break;
  }
  return p;
}

/**
 * Yarn–yarn contact: the strand is a solid tube, so no two non-neighbouring centreline points may be
 * closer than the sum of their radii. Spatial-hash projection (Jacobi-style per pass); `mob[i]` in [0,1]
 * says how easily a point moves (settled fabric yarn is stiff, working yarn is free).
 */
export function separateYarn(pts, rad, mob, spacing, iters = 4, skipArc = 3.0) {
  const n = pts.length;
  const skip = Math.max(2, Math.ceil(skipArc / spacing));
  let rmax = 0; for (let i = 0; i < n; i++) rmax = Math.max(rmax, rad[i]);
  const cell = 2 * rmax;
  const dx = new Float64Array(n * 3), cnt = new Float64Array(n);
  for (let it = 0; it < iters; it++) {
    const grid = new Map();
    const key = (x, y, z) => (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const k = key(Math.floor(p[0] / cell), Math.floor(p[1] / cell), Math.floor(p[2] / cell));
      let b = grid.get(k); if (!b) grid.set(k, (b = [])); b.push(i);
    }
    dx.fill(0); cnt.fill(0);
    let moved = 0;
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell), cz = Math.floor(p[2] / cell);
      for (let ax = -1; ax <= 1; ax++) for (let ay = -1; ay <= 1; ay++) for (let az = -1; az <= 1; az++) {
        const b = grid.get(key(cx + ax, cy + ay, cz + az)); if (!b) continue;
        for (const j of b) {
          if (j <= i + skip - 1 && j >= i - skip + 1) continue;
          if (j < i) continue;
          const q = pts[j];
          const ex = q[0] - p[0], ey = q[1] - p[1], ez = q[2] - p[2];
          const d = Math.hypot(ex, ey, ez), need = (rad[i] + rad[j]) * 0.96;
          if (d >= need || d < 1e-9) continue;
          const wi = mob[i], wj = mob[j], w = wi + wj;
          if (w < 0.5) continue; // settled fabric against settled fabric: its relaxed shape already accounts for contact
          const c = (need - d) / d;
          dx[i * 3] -= ex * c * wi / w; dx[i * 3 + 1] -= ey * c * wi / w; dx[i * 3 + 2] -= ez * c * wi / w;
          dx[j * 3] += ex * c * wj / w; dx[j * 3 + 1] += ey * c * wj / w; dx[j * 3 + 2] += ez * c * wj / w;
          cnt[i]++; cnt[j]++; moved++;
        }
      }
    }
    if (!moved) break;
    for (let i = 0; i < n; i++) if (cnt[i]) {
      const s = 0.8 / Math.sqrt(cnt[i]);
      pts[i] = [pts[i][0] + dx[i * 3] * s, pts[i][1] + dx[i * 3 + 1] * s, pts[i][2] + dx[i * 3 + 2] * s];
    }
  }
  return pts;
}
