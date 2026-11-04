// Shorts v2: step-structured scenario (one toy part, ~55 s) and its director (cuts, overlays).
import { Fabric } from './fabric.mjs';
import { add, sub, mul, lerp, mix, smooth, clamp, norm } from './vec.mjs';

// Detailed stitches cut between angles phase by phase; fast stitches alternate wider angles.
const DETAIL_SHOTS = {
  insert: 'pov', yo1: 'back', pullup: 'back', raise: 'pov', yo2: 'macro', through1: 'macro', through2: 'pov', ret: 'pov',
  insertFL1: 'macro', insertFL2: 'macro',
};

/**
 * opts: {rounds, title, sub, steps:[...], colour, hook, toyImage, outroHead, outroText}
 * The step plan below covers rounds that use the magic ring, sc and inc (and dec, if present).
 */
export function buildShort(opts) {
  const fab = new Fabric(opts.rounds);
  const st = fab.stitches;
  const rounds = fab.rounds;
  const acts = [];
  const caps = []; // {ai, phase?, head, text}
  const steps = []; // {ai, n, label}
  const cams = []; // {ai, shot} or phase shots inside the action
  const push = (a) => { acts.push(a); return acts.length - 1; };
  const STEP = (ai, label) => steps.push({ ai, label });

  // 1. ring
  let ai = push({ type: 'hold', dur: 5.2, hook: { tip: [7, 9, 13], a: [-0.6, -0.4, -0.7], d: [0, -1, 0.3] } });
  STEP(ai, 'Кольцо амигуруми');
  caps.push({ ai, text: 'Сложите нить петлёй: хвостик внизу, рабочая нить уходит к пальцу.' });
  cams.push({ ai, shot: 'ringOrbit' });

  const nR1 = rounds[0].n;
  let incShown = 0, decShown = false;
  for (let i = 0; i < st.length; i++) {
    const s = st[i];
    const k = s.round;
    const a = { type: 'stitch', i, speed: 1 };
    if (s.kind === 'ch') {
      a.speed = 0.8; a.slow = { yo1: 1.3, pullup: 1.3 };
      a.phaseShots = DETAIL_SHOTS;
      a.phaseCaps = {
        insert: 'Крючок — в кольцо, спереди назад.',
        yo1: 'Накид: зев крючка сам подхватывает нить.',
        pullup: 'Вытяните петлю через кольцо.',
        yo2: 'Ещё накид…',
        through1: '…и через петлю на крючке.',
        ret: 'Это петля подъёма — в счёт не идёт.',
      };
      ai = push(a); STEP(ai, 'Петля подъёма');
    } else if (k === 1 && s.j === 0) {
      a.speed = 0.62; a.slow = { yo1: 1.35, pullup: 1.35, ret: 1.8 };
      a.phaseShots = DETAIL_SHOTS;
      a.phaseCaps = {
        insert: 'Сбн: крючок снова в кольцо.',
        yo1: 'Накид — подцепите рабочую нить.',
        pullup: 'Тяните петлю к себе через кольцо.',
        raise: 'На крючке 2 петли.',
        yo2: 'Снова накид…',
        through1: '…протяните через обе петли.',
        ret: 'Первый сбн готов!',
      };
      ai = push(a); STEP(ai, `Р1: ${nR1} сбн в кольцо`);
    } else if (k === 1 && s.j === 1) {
      a.speed = 1.05;
      a.phaseShots = { insert: 'pov', yo1: 'back', pullup: 'pov', raise: 'pov', yo2: 'macro', through1: 'pov', through2: 'pov', ret: 'mid' };
      ai = push(a);
      caps.push({ ai, text: 'Второй — так же: в кольцо, накид, вытянуть, накид, через 2.' });
    } else if (k === 1) {
      a.speed = 4.2;
      ai = push(a);
      if (s.j === 2) { caps.push({ ai, text: `Ещё ${nR1 - 2} сбн — всего ${nR1}.`, ff: true }); }
      cams.push({ ai, shot: s.j % 2 ? 'mid' : 'pov' });
    } else if (s.kind === 'inc' && incShown < 2) {
      a.speed = incShown === 0 ? 0.85 : 1.0;
      a.phaseShots = incShown === 0
        ? { insert: 'macro', yo1: 'back', pullup: 'pov', raise: 'pov', yo2: 'macro', through1: 'pov', through2: 'pov', ret: 'pov' }
        : { insert: 'macro', yo1: 'pov', pullup: 'pov', raise: 'pov', yo2: 'pov', through1: 'pov', through2: 'mid', ret: 'mid' };
      a.slow = incShown === 0 ? { insert: 1.4 } : { insert: 1.4, ret: 1.6 };
      ai = push(a);
      if (incShown === 0) { STEP(ai, `Р${k}: прибавки`); caps.push({ ai, text: 'Прибавка = 2 сбн в одну петлю. Первый — в петлю ряда ниже…' }); }
      else caps.push({ ai, text: '…второй — в ту же самую петлю. Прибавка готова!' });
      incShown++;
    } else if (s.kind === 'dec' && !decShown) {
      a.speed = 0.75; a.phaseShots = { ...DETAIL_SHOTS, yo1: 'macro', pullup: 'macro' };
      a.phaseCaps = {
        insertFL1: 'Убавка: под переднюю полупетлю 1-й петли…',
        insertFL2: '…и сразу под полупетлю 2-й.',
        yo1: 'Накид и протяните через обе полупетли.',
        yo2: 'Накид — через 2 петли на крючке.',
        ret: 'Две петли стали одной.',
      };
      ai = push(a); STEP(ai, `Р${k}: убавки`);
      decShown = true;
    } else {
      a.speed = s.kind === 'dec' ? 6 : k === 2 ? 5.5 : 8;
      ai = push(a);
      const first = !caps.some((c) => c.round === k && c.ff);
      if (first) caps.push({ ai, round: k, ff: true, text: roundExplain(rounds[k - 1], rounds[k - 2]?.n) });
      cams.push({ ai, shot: ['pov', 'mid', 'top'][Math.floor(s.j / 3) % 3] });
    }
    if (k === 1 && s.j === nR1 - 1) {
      const a2 = push({ type: 'closeRing', dur: 3.2, hook: { tip: [3, 8, 10], a: [-0.7, -0.3, -0.6], d: [0, -1, 0.3] } });
      STEP(a2, 'Затяните кольцо');
      caps.push({ ai: a2, text: 'Потяните за хвостик — кольцо стягивается, дырочки нет.' });
      cams.push({ ai: a2, shot: 'wide' });
    }
  }
  const aEnd = push({ type: 'outro', dur: 5.0, hook: { tip: [16, 24, 28], a: [-0.6, -0.5, -0.6], d: [0, -1, 0.2] } });
  STEP(aEnd, 'Готово!');
  caps.push({ ai: aEnd, text: opts.outroText || 'Закрепите нить и оставьте хвост для пришивания.' });
  cams.push({ ai: aEnd, shot: 'reveal' });
  return {
    style: 'short2', rounds: opts.rounds, actions: acts, captions: caps, cams, steps,
    fps: opts.fps || 30, tailLength: 36, theta0: opts.theta0, meta: opts,
  };
}

function roundExplain(rd, nPrev) {
  const kinds = rd.seq.map((s) => s.kind);
  if (kinds.every((x) => x === 'inc')) return `Прибавка в каждую петлю: было ${nPrev} → стало ${rd.n}.`;
  if (kinds.includes('inc')) return `Чередуйте сбн и прибавки: ${nPrev} → ${rd.n} петель.`;
  if (kinds.every((x) => x === 'dec')) return `Убавка в каждые 2 петли: ${nPrev} → ${rd.n}.`;
  if (kinds.includes('dec')) return `Чередуйте сбн и убавки: ${nPrev} → ${rd.n} петель.`;
  return `По 1 сбн в каждую петлю (${rd.n}).`;
}

// ---------------------------------------------------------------- director
const SHOTS = {
  pov: { look: [-2, -5, -1], dist: 74, elev: 36, azim: 26, fov: 30 }, // over the crocheter's shoulder
  back: { look: [-2.5, -3, -2.5], dist: 66, elev: 20, azim: -118, fov: 30 }, // behind the work: the catch
  macro: { look: 'throat', dist: 54, elev: 16, azim: 12, fov: 30 },
  mid: { look: [-1, -9, -2], dist: 96, elev: 24, azim: 10, fov: 30 },
  top: { look: 'fabric', dist: 104, elev: 6, azim: -6, fov: 30 },
  wide: { look: [0, -10, -2], dist: 110, elev: 22, azim: 8, fov: 30 },
  ringOrbit: { look: [-1, -6, -1], dist: 84, elev: 26, azim: [-30, 18], fov: 30 },
  reveal: { look: 'fabric', dist: 92, elev: 22, azim: [-38, 22], fov: 30 },
};

export function shortTracks(sim) {
  if (sim._tracks2) return sim._tracks2;
  const scn = sim.scn, A = sim.actions;
  const capByAi = {};
  for (const c of scn.captions) (capByAi[c.ai] ||= []).push(c);
  const caps = [], cams = [];
  const camByAi = {};
  for (const c of scn.cams) camByAi[c.ai] = c.shot;
  A.forEach((act, ai) => {
    for (const c of capByAi[ai] || []) caps.push({ t0: act.t0, text: c.text, ff: c.ff });
    if (act.phaseCaps) for (const ph of act.phases) if (act.phaseCaps[ph.name]) caps.push({ t0: ph.t0, text: act.phaseCaps[ph.name] });
    if (act.phaseShots && act.phases) for (const ph of act.phases) cams.push({ t0: ph.t0, shot: act.phaseShots[ph.name] || 'pov', t1a: act.t1 });
    else if (camByAi[ai]) cams.push({ t0: act.t0, shot: camByAi[ai], t1a: act.t1 });
  });
  caps.sort((a, b) => a.t0 - b.t0);
  const merged = [];
  for (const c of caps) { const l = merged[merged.length - 1]; if (l && l.text === c.text) continue; merged.push({ ...c }); }
  merged.forEach((c, k) => { c.t1 = k + 1 < merged.length ? merged[k + 1].t0 : sim.duration; });
  cams.sort((a, b) => a.t0 - b.t0);
  // drop camera moves that would last < 0.9 s (avoid jitter), keep the later one
  const cm = [];
  for (const c of cams) { const l = cm[cm.length - 1]; if (l && l.shot === c.shot) continue; if (l && c.t0 - l.t0 < 0.9 && cm.length > 1) cm.pop(); cm.push(c); }
  const steps = scn.steps.map((s, k) => ({ t0: A[s.ai].t0, label: s.label, n: k + 1, of: scn.steps.length }));
  sim._tracks2 = { caps: merged, cams: cm, steps };
  return sim._tracks2;
}

function shotParams(sim, name, t, st) {
  const s = SHOTS[name] || SHOTS.pov;
  let look = s.look;
  if (look === 'fabric') {
    const [, z] = sim.fab.profileAt(Math.max(0.5, sim.fab.profile.K * 0.5));
    look = add(sim.xf.to([0, 0, z]), [0, 0, 0]);
    // centre of the finished part (the disc centre sits at the fabric origin)
  } else if (look === 'throat') {
    const h = st?.hook;
    look = h ? add(h.tip, mul(h.a, -3.2)) : [-2, -4, 0];
    look = lerp(look, [-1.5, -4, 0], 0.35);
  }
  let azim = s.azim;
  if (Array.isArray(azim)) azim = mix(azim[0], azim[1], smooth(clamp((t - (sim._shotT0 ?? 0)) / 6)));
  return { look, dist: s.dist, elev: s.elev, azim, fov: s.fov };
}

/** Camera: eased 0.75 s moves between shots plus a slow push-in and a gentle handheld drift. */
export function shortCamera(sim, t, st) {
  const { cams } = shortTracks(sim);
  let k = 0;
  while (k + 1 < cams.length && cams[k + 1].t0 <= t) k++;
  sim._shotT0 = cams[k].t0;
  const cur = shotParams(sim, cams[k].shot, t, st);
  if (!sim._camLook) sim._camLook = cur.look;
  const prevName = k > 0 ? cams[k - 1].shot : cams[k].shot;
  const prev = shotParams(sim, prevName, t, st);
  const e = smooth(clamp((t - cams[k].t0) / 0.75));
  // look target is low-pass filtered so the dynamic 'throat' look never jitters
  const targetLook = lerp(prev.look, cur.look, e);
  sim._camLook = lerp(sim._camLook, targetLook, 0.35);
  const look = sim._camLook;
  const push = 1 - 0.06 * smooth(clamp((t - cams[k].t0) / 3.5));
  const dist = mix(prev.dist, cur.dist, e) * push, elev = mix(prev.elev, cur.elev, e), azim = mix(prev.azim, cur.azim, e);
  const dr = Math.PI / 180;
  const el = (elev + 0.7 * Math.sin(t * 0.41)) * dr, az = (azim + 0.9 * Math.sin(t * 0.27 + 1)) * dr;
  const pos = add(look, mul([Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)], dist));
  return { pos, look, fov: mix(prev.fov, cur.fov, e) };
}

export function shortOverlay(sim, t, st) {
  const { caps, steps } = shortTracks(sim);
  let cap = null;
  for (const c of caps) if (t >= c.t0 && t < c.t1) cap = { text: c.text, ff: c.ff, alpha: clamp((t - c.t0) / 0.2), age: t - c.t0 };
  let step = null;
  for (const s of steps) if (t >= s.t0) step = { ...s, age: t - s.t0 };
  const act = sim.actions[sim.ai];
  let counter = null, speed = 1;
  if (act?.type === 'stitch') {
    const s = sim.fab.stitches[act.i];
    if (s.kind !== 'ch') counter = { round: s.round, j: s.j + 1, n: s.n };
    speed = act.speed || 1;
  }
  return {
    cap, step, counter, speed, progress: clamp(t / sim.duration),
    phase: st?.phase, phaseU: st?.phaseU, onHook: st?.onHook ?? 0,
    detail: !!act?.phaseShots && speed < 1.5,
    outro: act?.type === 'outro' ? clamp((t - act.t0) / act.dur) : -1,
    intro: act?.type === 'hold' && sim.ai === 0 ? clamp((t - act.t0) / 2.4) : -1,
  };
}
