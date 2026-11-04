// Minimal 3D vector / quaternion helpers on plain arrays (shared by Node and browser).
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const madd = (a, b, s) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len = (a) => Math.hypot(a[0], a[1], a[2]);
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
export const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const smooth = (t) => { t = clamp(t); return t * t * t * (t * (t * 6 - 15) + 10); };
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export const mix = (a, b, t) => a + (b - a) * t;
/** Component of v perpendicular to unit axis a. */
export const perp = (v, a) => madd(v, a, -dot(v, a));
/** Any unit vector perpendicular to unit a. */
export const anyPerp = (a) => norm(Math.abs(a[1]) < 0.9 ? cross(a, [0, 1, 0]) : cross(a, [1, 0, 0]));
/** Spherical interpolation of unit vectors. */
export function slerpV(a, b, t) {
  const d = clamp(dot(a, b), -1, 1);
  const th = Math.acos(d);
  if (th < 1e-4) return norm(lerp(a, b, t));
  const s = Math.sin(th);
  return norm(add(mul(a, Math.sin((1 - t) * th) / s), mul(b, Math.sin(t * th) / s)));
}
/** Rotate v around unit axis k by angle. */
export function rotAxis(v, k, ang) {
  const c = Math.cos(ang), s = Math.sin(ang);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
}
// ---- quaternions [x,y,z,w]
export function quatFromBasis(u, v, w) {
  // rotation matrix with columns u,v,w (maps local x,y,z to u,v,w)
  const m00 = u[0], m01 = v[0], m02 = w[0], m10 = u[1], m11 = v[1], m12 = w[1], m20 = u[2], m21 = v[2], m22 = w[2];
  const tr = m00 + m11 + m22;
  let x, y, z, q;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; q = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; q = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; q = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
  return [x, y, z, q];
}
export function quatSlerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (d < 0) { d = -d; bb = b.map((x) => -x); }
  if (d > 0.9995) { const r = a.map((x, i) => x + (bb[i] - x) * t); const l = Math.hypot(...r); return r.map((x) => x / l); }
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return a.map((x, i) => x * wa + bb[i] * wb);
}
export function quatToBasis(q) {
  const [x, y, z, w] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
    [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
    [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
  ];
}
