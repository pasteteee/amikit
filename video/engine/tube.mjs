// Tube mesh builder (rotation-minimising frames), shared by the browser renderer and tests.
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export function fillTube(pts, s, rad, RADIAL, RY, PLY_PITCH, P, N, U, I) {
  const n = pts.length;
  // parallel transport frames
  let T = [0, 0, 0], Nm = null, B = null;
  const tan = (i) => { const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)]; const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; const l = Math.hypot(...d) || 1; return d.map((x) => x / l); };
  let tPrev = null;
  for (let i = 0; i < n; i++) {
    let t = tan(i);
    if (!(Math.abs(t[0]) + Math.abs(t[1]) + Math.abs(t[2]) > 0.5)) t = tPrev || [1, 0, 0]; // duplicate points
    if (!Nm) {
      const up = Math.abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      Nm = norm3(cross3(cross3(t, up), t));
    } else {
      // rotate previous normal by the rotation taking tPrev to t (rotation-minimising frame)
      const ax = cross3(tPrev, t); const sn = Math.hypot(ax[0], ax[1], ax[2]); const cs = dot3(tPrev, t);
      if (sn > 1e-6) {
        const k = [ax[0] / sn, ax[1] / sn, ax[2] / sn];
        const kv = cross3(k, Nm), kd = dot3(k, Nm);
        Nm = [Nm[0] * cs + kv[0] * sn + k[0] * kd * (1 - cs), Nm[1] * cs + kv[1] * sn + k[1] * kd * (1 - cs), Nm[2] * cs + kv[2] * sn + k[2] * kd * (1 - cs)];
      }
      const d = dot3(Nm, t); const pr = [Nm[0] - t[0] * d, Nm[1] - t[1] * d, Nm[2] - t[2] * d];
      if (Math.hypot(pr[0], pr[1], pr[2]) < 1e-3) { const up = Math.abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]; Nm = norm3(cross3(cross3(t, up), t)); }
      else Nm = norm3(pr);
    }
    tPrev = t;
    B = cross3(t, Nm);
    T = t;
    for (let k = 0; k <= RADIAL; k++) {
      const a = (k / RADIAL) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      const nx = Nm[0] * ca + B[0] * sa, ny = Nm[1] * ca + B[1] * sa, nz = Nm[2] * ca + B[2] * sa;
      const o = (i * (RADIAL + 1) + k);
      const r = rad ? rad[i] : RY;
      P[o * 3] = pts[i][0] + nx * r; P[o * 3 + 1] = pts[i][1] + ny * r; P[o * 3 + 2] = pts[i][2] + nz * r;
      N[o * 3] = nx; N[o * 3 + 1] = ny; N[o * 3 + 2] = nz;
      U[o * 2] = s[i] / PLY_PITCH; U[o * 2 + 1] = k / RADIAL;
    }
  }
  void T;
  let q = 0;
  for (let i = 0; i < n - 1; i++) for (let k = 0; k < RADIAL; k++) {
    const a = i * (RADIAL + 1) + k, b = a + RADIAL + 1;
    I[q++] = a; I[q++] = a + 1; I[q++] = b; I[q++] = b; I[q++] = a + 1; I[q++] = b + 1;
  }
  return q;
}
