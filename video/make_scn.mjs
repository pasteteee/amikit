// Usage: node make_scn.mjs <out.json> -- builds the bear-ear tutorial scenario
import { buildPartTutorial } from './engine/scenario.mjs';
import fs from 'fs';
const scn = buildPartTutorial({ rounds: ['mr 6', 'inc 6 to 18', 'even 1', 'dec 6 to 6'], render: { colour: '#ddbe98' } });
scn.meta.render = { colour: '#ddbe98' };
fs.writeFileSync(process.argv[2], JSON.stringify(scn));
