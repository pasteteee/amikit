// Usage: node make_teddy_tail.mjs <out.json> -- Shorts v2: teddy bear tail (brown): MR, inc, sc, dec
import { buildShort } from './engine/short.mjs';
import fs from 'fs';
const scn = buildShort({
  rounds: ['mr 6', 'inc 6 to 12', 'even 1', 'dec 2 to 10'],
  title: 'Хвостик мишки',
  sub: 'крючком за минуту · кольцо, прибавка, сбн, убавка',
  outroHead: 'Хвостик готов!',
  outroSub: 'Набейте чуть-чуть и пришейте мишке',
  outroText: 'Оставьте хвост нити, чтобы пришить.',
});
scn.meta.render = {
  colour: '#9a5a2e', sheen: '#e0b080', hookColour: '#7fd6c8', handleColour: '#4b2a5e', exposure: 0.95,
  backdrop: [[0, '#fbf3ea'], [0.5, '#f4e4d4'], [1, '#e6cbb4']],
  toyImage: '/web/assets/teddy.jpg',
};
scn.meta.tailKeep = 0.93; // calm free tail: it hangs and follows the work without swinging
fs.writeFileSync(process.argv[2], JSON.stringify(scn));
