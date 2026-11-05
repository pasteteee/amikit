// Crochet hook geometry (shared by simulation collisions and the renderer mesh).
// Hook-local frame: x along the axis toward the tip (tip at x = 0, shaft at x < 0),
// y = throat side (the open side of the throat), z = y × x side.
import { add, sub, mul, madd, dot, cross, norm, smoothstep } from './vec.mjs';

export const HOOK = {
  rShaft: 1.25, // 2.5 mm hook (matches the Newton simulation hook)
  rNeck: 0.6, // remaining spine thickness at the throat
  xThroat: -2.05, // where yarn sits in the throat
  throatHalf: 1.0,
  thumbStart: -46, thumbEnd: -70,
  handleStart: -74,
};

/** Base (round) radius of the hook body at axial position x. */
export function baseRadius(x) {
  const R = HOOK.rShaft;
  if (x > 0) return 0;
  // rounded head: tip is a slightly flattened dome
  const tipLen = 1.15;
  if (x > -tipLen) { const u = (x + tipLen) / tipLen; return R * 0.92 * Math.sqrt(Math.max(0, 1 - u * u)) + 0.02; }
  // a tiny waist between head and shaft so the head reads as a separate lip
  const waist = 0.07 * Math.exp(-(((x + 1.4) / 0.5) ** 2));
  let r = R * (x > -1.6 ? 0.92 : 1) - waist;
  if (x <= HOOK.handleStart) r = 4.3; // handle
  else if (x < HOOK.thumbStart + 6 && x > HOOK.handleStart - 0.0) r = R + (4.3 - R) * smoothstep(HOOK.thumbStart + 6, HOOK.handleStart, x);
  return r;
}

/** Depth of the throat groove at x (cut on the +y side): gentle ramp from the shaft, steep wall under the lip. */
export function grooveDepth(x) {
  const x0 = -4.4, xm = -2.0, x1 = -1.45;
  if (x <= x0 || x >= x1) return 0;
  const D = HOOK.rShaft - HOOK.rNeck;
  if (x < xm) { const t = (x - x0) / (xm - x0); return D * t * t * (3 - 2 * t); }
  const t = (x - xm) / (x1 - xm); return D * (1 - t * t * t);
}

/** Surface radius at axial x and angle phi measured from +y (throat side) toward +z. */
export function surfaceRadius(x, phi) {
  const b = baseRadius(x);
  const c = Math.max(0, Math.cos(phi));
  return Math.max(0.05, b - grooveDepth(x) * Math.min(1, Math.pow(c, 0.7) * 1.15));
}

/** Hook pose: tip position, axis a (toward tip), throat dir d (unit, perpendicular to a). */
export function hookFrame(pose) {
  const a = norm(pose.a);
  const d = norm(madd(pose.d, a, -dot(pose.d, a)));
  const s = cross(a, d);
  return { tip: pose.tip, a, d, s };
}
export function hookPoint(F, x, y = 0, z = 0) {
  return add(add(add(F.tip, mul(F.a, x)), mul(F.d, y)), mul(F.s, z));
}
export function toHookLocal(F, p) {
  const q = sub(p, F.tip);
  return [dot(q, F.a), dot(q, F.d), dot(q, F.s)];
}

/** Push yarn centreline points out of the hook body (yarn radius ry). Modifies pts in place. */
export function pushOutOfHook(F, pts, ry, from = 0) {
  const xMin = -40;
  for (let i = from; i < pts.length; i++) {
    const [x, y, z] = toHookLocal(F, pts[i]);
    if (x > 0.9 || x < xMin) continue;
    if (x > 0) {
      // beyond the tip: spherical cap
      const r = Math.hypot(x, y, z);
      const need = ry * 0.97;
      if (r < need && r > 1e-6) pts[i] = add(F.tip, mul(norm(sub(pts[i], F.tip)), need));
      continue;
    }
    const rr = Math.hypot(y, z);
    const phi = Math.atan2(z, y);
    const need = surfaceRadius(x, phi) + ry * 0.97;
    if (rr < need) {
      const dirLocal = rr > 1e-6 ? [y / rr, z / rr] : [-1, 0];
      pts[i] = add(add(add(F.tip, mul(F.a, x)), mul(F.d, dirLocal[0] * need)), mul(F.s, dirLocal[1] * need));
    }
  }
}
