// Director: camera shots, captions and badges derived from the scenario timeline.
import { add, mul, lerp, mix, smooth, clamp } from './vec.mjs';

const SHOTS = {
  ring: { look: [-1, -6.5, -2], dist: 92, elev: 16, azim: 6, fov: 28 },
  close: { look: [-1.5, -5.5, 0], dist: 74, elev: 15, azim: 10, fov: 28 },
  mid: { look: [-1, -9, -2], dist: 108, elev: 19, azim: 8, fov: 28 },
  wide: { look: [0, -11, -3], dist: 122, elev: 20, azim: 4, fov: 28 },
  reveal: { look: 'fabric', dist: 118, elev: 24, azim: -8, fov: 28 },
};

function tracks(sim) {
  if (sim._tracks) return sim._tracks;
  const scn = sim.scn, A = sim.actions;
  const caps = [];
  const capByAi = {};
  for (const c of scn.captions) capByAi[c.ai] = c;
  A.forEach((act, ai) => {
    const c = capByAi[ai];
    if (c) caps.push({ t0: act.t0, head: c.head, text: c.text, ff: c.ff });
    if (act.phaseCaps && act.phases) {
      for (const ph of act.phases) {
        const pc = act.phaseCaps[ph.name];
        if (pc) caps.push({ t0: ph.t0, head: pc[0], text: pc[1] });
      }
    }
  });
  // merge identical consecutive captions; each lasts until the next one
  const merged = [];
  for (const c of caps) {
    const last = merged[merged.length - 1];
    if (last && last.text === c.text && last.head === c.head) continue;
    merged.push({ ...c });
  }
  merged.forEach((c, k) => { c.t1 = k + 1 < merged.length ? merged[k + 1].t0 : sim.duration; });
  const cams = scn.cams.map((c) => ({ t0: A[c.ai].t0, shot: c.shot }));
  cams.sort((a, b) => a.t0 - b.t0);
  sim._tracks = { caps: merged, cams };
  return sim._tracks;
}

export function captionAt(sim, t) {
  const { caps } = tracks(sim);
  for (let k = caps.length - 1; k >= 0; k--) {
    const c = caps[k];
    if (t >= c.t0 && t < c.t1) {
      const fadeIn = clamp((t - c.t0) / 0.25);
      return { head: c.head, text: c.text, alpha: fadeIn };
    }
  }
  return null;
}

export function badgeAt(sim, t) {
  const act = sim.actions[sim.ai];
  if (!act) return null;
  if (act.type === 'stitch') {
    const s = sim.fab.stitches[act.i];
    const sp = act.speed || 1;
    const lab = s.round >= 1 && s.kind !== 'ch' ? `Р${s.round} · ${s.j + 1}/${s.n}` : 'КА';
    return sp > 1.5 ? `${lab}  ×${Math.round(sp)}` : lab;
  }
  return null;
}

function shotParams(sim, name) {
  const s = SHOTS[name] || SHOTS.close;
  let look = s.look;
  if (look === 'fabric') {
    const X = sim.xf;
    const [rho, z] = sim.fab.profileAt(sim.fab.profile.K * 0.5);
    void rho;
    look = X.to([0, 0, z]);
    look = add(look, [0, -2, 0]);
  }
  return { ...s, look };
}

export function cameraAt(sim, t) {
  const { cams } = tracks(sim);
  let k = 0;
  while (k + 1 < cams.length && cams[k + 1].t0 <= t) k++;
  const cur = shotParams(sim, cams[k].shot);
  const prev = k > 0 ? shotParams(sim, cams[k - 1].shot) : cur;
  const e = smooth(clamp((t - cams[k].t0) / 1.6));
  const look = lerp(prev.look, cur.look, e);
  const dist = mix(prev.dist, cur.dist, e), elev = mix(prev.elev, cur.elev, e), azim = mix(prev.azim, cur.azim, e);
  // gentle handheld drift
  const dr = Math.PI / 180;
  const el = (elev + 0.6 * Math.sin(t * 0.37)) * dr, az = (azim + 0.8 * Math.sin(t * 0.23 + 1)) * dr;
  const pos = add(look, mul([Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)], dist));
  return { pos, look, fov: mix(prev.fov, cur.fov, e) };
}
