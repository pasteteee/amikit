// Scenario builder: turns a pattern part into a sequence of actions + captions + camera moves.
import { Fabric } from './fabric.mjs';

const RU = {
  stitchName: { sc: 'сбн', inc: 'прибавка', dec: 'убавка', ch: 'возд. петля' },
};

/**
 * Build a Shorts tutorial for one pattern part.
 * opts: {rounds, title, partName, colour, detail: how many stitches of each new element are shown slowly}
 */
export function buildPartTutorial(opts) {
  const fab = new Fabric(opts.rounds);
  const st = fab.stitches;
  const acts = [];
  const caps = []; // {act index, phase?, head, text}
  const cam = []; // {act index, shot}
  const push = (a, cap) => { acts.push(a); if (cap) caps.push({ ai: acts.length - 1, ...cap }); return acts.length - 1; };

  const rounds = fab.rounds;
  // intro: hook comes in, magic ring lies ready
  push({ type: 'hold', dur: 3.2, hook: { tip: [6, 9, 12], a: [-0.6, -0.4, -0.7], d: [0, -1, 0.3] } }, {
    head: 'Кольцо амигуруми (КА)',
    text: 'Сложите нить петлёй: хвостик снизу, рабочая нить сверху и уходит к указательному пальцу.',
  });
  cam.push({ ai: 0, shot: 'ring' });

  const seen = {};
  let roundStartAi = {};
  for (let i = 0; i < st.length; i++) {
    const s = st[i];
    const k = s.round;
    const first = !seen[s.kind + (k === 1 ? 'r1' : '')];
    let speed = 1;
    let cap = null;
    let phaseCaps = null;
    if (s.kind === 'ch') {
      speed = 0.62;
      phaseCaps = {
        insert: ['Начало', 'Введите крючок в кольцо спереди назад.'],
        yo1: ['Накид', 'Подхватите рабочую нить зевом крючка.'],
        pullup: ['Вытяните петлю', 'Протяните нить через кольцо — на крючке 1 петля.'],
        yo2: ['Воздушная петля', 'Накид…'],
        through1: ['Воздушная петля', '…и протяните через петлю на крючке. Это петля подъёма, она не считается.'],
      };
    } else if (k === 1 && s.j === 0) {
      speed = 0.62;
      phaseCaps = {
        insert: ['Р1: 6 сбн в КА', 'Сбн (столбик без накида): введите крючок в кольцо.'],
        yo1: ['Р1: 6 сбн в КА', 'Накид — захватите рабочую нить.'],
        pullup: ['Р1: 6 сбн в КА', 'Вытяните петлю: на крючке 2 петли.'],
        yo2: ['Р1: 6 сбн в КА', 'Снова накид…'],
        through1: ['Р1: 6 сбн в КА', '…и протяните через обе петли. Первый сбн готов!'],
      };
    } else if (k === 1 && s.j === 1) { speed = 1.0; cap = { head: 'Р1: 6 сбн в КА', text: 'Второй сбн — так же: в кольцо, накид, вытянуть, накид, через 2 петли.' }; }
    else if (k === 1) { speed = 4.5; cap = { head: 'Р1: 6 сбн в КА', text: `Сбн ${s.j + 1} из 6`, ff: true }; }
    else if (s.kind === 'inc' && first) {
      speed = s.sub === 0 ? 0.9 : 0.9;
      cap = s.sub === 0
        ? { head: `Р${k}: ${roundLabel(rounds[k - 1])}`, text: 'Прибавка: 2 сбн в одну петлю. Первый сбн — в петлю предыдущего ряда…' }
        : { head: `Р${k}: ${roundLabel(rounds[k - 1])}`, text: '…второй сбн — в ту же самую петлю. Было 6 петель — стало 12.' };
      if (s.sub === 1) seen[s.kind] = true;
    } else if (s.kind === 'dec' && first) {
      speed = 0.7;
      phaseCaps = {
        insertFL1: [`Р${k}: ${roundLabel(rounds[k - 1])}`, 'Убавка невидимкой: крючок под переднюю полупетлю 1-го столбика…'],
        insertFL2: [`Р${k}: ${roundLabel(rounds[k - 1])}`, '…и сразу под переднюю полупетлю 2-го.'],
        yo1: [`Р${k}: ${roundLabel(rounds[k - 1])}`, 'Накид и протяните через обе полупетли.'],
        yo2: [`Р${k}: ${roundLabel(rounds[k - 1])}`, 'Накид — через 2 петли на крючке. Две петли стали одной.'],
      };
      seen.dec = true;
    } else {
      // repeated stitches: fast-forward, faster for long rounds
      speed = s.kind === 'dec' ? 6 : k <= 2 ? 6 : k === 3 ? 9 : 11;
      cap = { head: `Р${k}: ${roundLabel(rounds[k - 1])}`, text: roundExplain(rounds[k - 1], k), ff: true };
    }
    if (s.kind !== 'inc') seen[s.kind + (k === 1 ? 'r1' : '')] = true;
    const ai = push({ type: 'stitch', i, speed, phaseCaps }, cap);
    if (roundStartAi[k] == null) { roundStartAi[k] = ai; cam.push({ ai, shot: k <= 1 ? 'close' : speed >= 3 ? 'mid' : 'close', round: k }); }
    if (cap && !cap.ff) cam.push({ ai, shot: 'close' });
    if (cap && cap.ff) cam.push({ ai, shot: 'mid' });
    if (k === 1 && s.j === rounds[0].n - 1) {
      // close the ring after round 1
      const a2 = push({ type: 'closeRing', dur: 3.0, hook: { tip: [2, 7, 9], a: [-0.7, -0.3, -0.6], d: [0, -1, 0.3] } }, {
        head: 'Затяните кольцо', text: 'Потяните за хвостик — кольцо стягивается, дырочки в центре нет.',
      });
      cam.push({ ai: a2, shot: 'wide' });
    }
  }
  const aEnd = push({ type: 'outro', dur: 4.5, hook: { tip: [14, 22, 26], a: [-0.6, -0.5, -0.6], d: [0, -1, 0.2] } }, {
    head: opts.partDone || 'Деталь готова!', text: opts.outroText || 'Закрепите нить и оставьте хвост для пришивания.',
  });
  cam.push({ ai: aEnd, shot: 'reveal' });
  return { rounds: opts.rounds, actions: acts, captions: caps, cams: cam, fps: opts.fps || 30, tailLength: 36, meta: opts };
}

function roundLabel(rd) {
  const kinds = rd.seq.map((s) => s.kind);
  const inc = kinds.filter((x) => x === 'inc').length, dec = kinds.filter((x) => x === 'dec').length;
  if (inc && inc === rd.seq.length) return `${inc} пр (${rd.n})`;
  if (dec && dec === rd.seq.length) return `${dec} уб (${rd.n})`;
  if (inc) { const g = rd.seq.length / inc; return `(${g - 1} сбн, пр) ×${inc} (${rd.n})`; }
  if (dec) { const g = rd.seq.length / dec; return `(${g - 1} сбн, уб) ×${dec} (${rd.n})`; }
  return `${rd.n} сбн (${rd.n})`;
}
function roundExplain(rd, k) {
  const kinds = rd.seq.map((s) => s.kind);
  if (kinds.every((x) => x === 'inc')) return 'По прибавке в каждую петлю.';
  if (kinds.includes('inc')) return 'Чередуйте: 1 сбн, затем прибавка — и так 6 раз.';
  if (kinds.every((x) => x === 'dec')) return 'Убавка в каждые 2 петли — отверстие стягивается.';
  if (kinds.includes('dec')) return 'Чередуйте: 1 сбн, затем убавка — и так 6 раз.';
  return k > 1 ? 'По 1 сбн в каждую петлю, без прибавок — деталь растёт в высоту.' : '';
}

export { RU };
