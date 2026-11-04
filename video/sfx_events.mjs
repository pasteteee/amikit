// Usage: node sfx_events.mjs <scenario.json> <out.json> -- timing of stitch clicks and step changes for the soundtrack
import { Sim } from './engine/sim.mjs';
import fs from 'fs';
const scn = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const sim = new Sim(scn);
const clicks = [], steps = [];
for (const a of sim.actions) if (a.type === 'stitch') { const p = a.phases.find((q) => q.name === 'through2'); clicks.push({ t: p.t0 + p.dur * 0.55, fast: (a.speed || 1) > 1.5 }); }
for (const s of scn.steps || []) steps.push(sim.actions[s.ai].t0);
fs.writeFileSync(process.argv[3], JSON.stringify({ duration: sim.duration, clicks, steps }));
